# Load required packages
import pandas as pd
import numpy as np
import re
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (same folder as script)
review_data = pd.read_csv("reviews_with_severity.csv", engine="python", on_bad_lines="skip")

review_data_clean = review_data[["review", "length_group", "roberta_severity"]].copy()

# Clean review text and calculate length
review_data_clean["review_clean"] = review_data_clean["review"].str.replace(r"<.*?>", "", regex=True)
review_data_clean["review_length"] = review_data_clean["review_clean"].str.len()

# Sentiment and extremity features
review_data_clean["sentiment_group"] = np.where(
    review_data_clean["roberta_severity"] > 0,
    "positive",
    "negative"
)

review_data_clean["extremity"] = abs(review_data_clean["roberta_severity"])

review_data_clean["length_group"] = review_data_clean["length_group"].astype("category")
review_data_clean["sentiment_group"] = review_data_clean["sentiment_group"].astype("category")

# ----------------------------------------------------------
# Test 1: Do longer reviews tend to be positive or negative?

neg = review_data_clean[review_data_clean["sentiment_group"] == "negative"]["review_length"]
pos = review_data_clean[review_data_clean["sentiment_group"] == "positive"]["review_length"]

t_stat, p_val = stats.ttest_ind(neg, pos, equal_var=False)

print("----- Test 1: Review length by binary sentiment -----")
print(f"t-statistic: {t_stat}, p-value: {p_val}")

mean_neg = neg.mean()
mean_pos = pos.mean()

print(f"Mean review length (negative): {round(mean_neg, 1)}")
print(f"Mean review length (positive): {round(mean_pos, 1)}")

if p_val < 0.05:
    print("The difference in review length between negative and positive reviews is statistically significant.")
    if mean_neg > mean_pos:
        print("Negative reviews are longer than positive reviews.")
    else:
        print("Positive reviews are longer than negative reviews.")
else:
    print("No statistically significant difference in review length between negative and positive reviews.")

# Saving results (witin same folder as script)
pd.DataFrame([{
    "t_stat": t_stat,
    "p_value": p_val,
    "mean_negative": mean_neg,
    "mean_positive": mean_pos
}]).to_csv("t_test_review_length_sentiment.csv", index=False)

# -----------------------------------------------------------------------------------
# Test 2: Do more extreme reviews vary by length group?

model = ols("extremity ~ C(length_group)", data=review_data_clean).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("----- Test 2: Extremity by length group (ANOVA) -----")
print(anova_table)

anova_table.to_csv("anova_extremity_length_group.csv")

# Pairwise comparisons
from itertools import combinations

groups = review_data_clean["length_group"].unique()

pairwise_results = []

for g1, g2 in combinations(groups, 2):
    group1 = review_data_clean[review_data_clean["length_group"] == g1]["extremity"]
    group2 = review_data_clean[review_data_clean["length_group"] == g2]["extremity"]

    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

    pairwise_results.append({
        "group1": g1,
        "group2": g2,
        "mean_group1": group1.mean(),
        "mean_group2": group2.mean(),
        "t_stat": t_stat,
        "p_value": p_val
    })

pairwise_df = pd.DataFrame(pairwise_results)

print("----- Pairwise Comparisons of Extremity by Length Group -----")
print(pairwise_df)

pairwise_df.to_csv("pairwise_extremity_length_group.csv", index=False)

# ---------------------------
# Plot: Extremity by length group
extremity_summary = review_data_clean.groupby("length_group")["extremity"].mean().reset_index()

plt.figure()
sns.barplot(data=extremity_summary, x="length_group", y="extremity")

for i, row in extremity_summary.iterrows():
    plt.text(i, row["extremity"], round(row["extremity"], 2),
             ha="center", va="bottom", fontweight="bold")

plt.title("Mean Review Extremity by Length Group")
plt.xlabel("Length Group")
plt.ylabel("Mean Extremity")

plt.tight_layout()

plt.savefig("mean_extremity_by_length_group.pdf")
plt.close()