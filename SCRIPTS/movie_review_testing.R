# Load required packages
library(tidyverse)
library(rstatix)
library(ggplot2)
library(scales)

output_dir <- "../OUTPUT"

# Load dataset
review_data <- read.csv("DATA/reviews_with_severity.csv")

review_data_clean <- review_data %>%
  select(review, length_group, roberta_severity) %>%

# Clean review text and calculate length
  mutate(
    review_clean = str_remove_all(review, "<.*?>"),
    review_length = str_length(review_clean),
    sentiment_group = if_else(roberta_severity > 0, "positive", "negative"),  # sentiment group based on roberta_severity
    extremity = abs(roberta_severity), # extremity based on absolute value of sentiment
    length_group = as.factor(length_group),
    sentiment_group = as.factor(sentiment_group)
  )

# ----------------------------------------------------------
# Test 1: Do longer reviews tend to be positive or negative?

t_test_result <- review_data_clean %>%
  t_test(review_length ~ sentiment_group, 
         var.equal = FALSE,       # Welch's t-test
         alternative = "two.sided") %>%
  add_significance()


print("----- Test 1: Review length by sentiment -----")
print(t_test_result)

mean_neg <- review_data_clean %>% filter(sentiment_group == "negative") %>% summarise(mean(review_length)) %>% pull()
mean_pos <- review_data_clean %>% filter(sentiment_group == "positive") %>% summarise(mean(review_length)) %>% pull()

print(paste("Mean review length (negative):", round(mean_neg, 1)))
print(paste("Mean review length (positive):", round(mean_pos, 1)))

if(t_test_result$p < 0.05){
 print("The difference in review length between negative and positive reviews is statistically significant.")
  if(mean_neg > mean_pos){
    print("Negative reviews are longer than positive reviews.")
  } else {
    print("Positive reviews are longer than negative reviews.")
  }
} else {
  print("No statistically significant difference in review length between negative and positive reviews.")
}

write.csv(t_test_result, file.path("OUTPUT", "t_test_review_length_sentiment.csv"), row.names = FALSE)


# -----------------------------------------------------------------------------------
# Test 2: Do more extreme reviews have a greater mean length than moderate reviews?
anova_result <- review_data_clean %>%
  welch_anova_test(extremity ~ length_group)

print("----- Test 2: Extremity by length group (Welch ANOVA) -----")
print(anova_result)

write.csv(anova_result, file.path("OUTPUT", "anova_extremity_length_group.csv"), row.names = FALSE)



pairwise_with_means <- review_data_clean %>%
  pairwise_t_test(
    extremity ~ length_group,
    p.adjust.method = "bonferroni",
    pool.sd = FALSE
  ) %>%
  left_join(
    review_data_clean %>%
      group_by(length_group) %>%
      summarise(mean_extremity_group1 = mean(extremity)),
    by = c("group1" = "length_group")
  ) %>%
  left_join(
    review_data_clean %>%
      group_by(length_group) %>%
      summarise(mean_extremity_group2 = mean(extremity)),
    by = c("group2" = "length_group")
  ) %>%
  select(.y., group1, mean_extremity_group1, group2, mean_extremity_group2,
         n1, n2, statistic, df, p.adj, p.adj.signif)

print("----- Pairwise Comparisons of Extremity by Length Group -----")
print(pairwise_with_means)

write.csv(pairwise_with_means, file.path("OUTPUT", "pairwise_extremity_length_group.csv"), row.names = FALSE)

#---------------------------
# Extremity plot

extremity_summary <- review_data_clean %>%
  group_by(length_group) %>%
  summarise(mean_extremity = mean(extremity))


mean_extremity_plot <- ggplot(extremity_summary, aes(x = length_group, y = mean_extremity, fill = length_group)) +
  geom_col(width = 0.6) + 
  geom_text(aes(label = round(mean_extremity, 2)), vjust = -0.5, size = 4, fontface = "bold", color = "black") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title = "Mean Review Extremity by Length Group",
    subtitle = "Based on absolute RoBERTa sentiment scores",
    x = "Length Group",
    y = "Mean Extremity"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(face = "italic", size = 12),
    axis.text = element_text(face = "bold", color = "black"),
    axis.title = element_text(face = "bold")
  )

ggsave(
  filename = file.path("OUTPUT", "mean_extremity_by_length_group.pdf"),
  plot = mean_extremity_plot,
  width = 8,
  height = 6,
  dpi = 300
)