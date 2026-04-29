import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import os


PROJECT_ROOT = Path.cwd()

data_path = PROJECT_ROOT
output_path = PROJECT_ROOT

os.makedirs(output_path, exist_ok=True)

#loading data
df = pd.read_csv("IMDB_Dataset.csv")

#cleaning

df['review'] = df['review'].str.replace('<br />', '', regex=True)

df['length'] = df['review'].str.len()

df['log_length'] = np.log1p(df['length'])

df['length_group'] = pd.qcut(
    df['log_length'],
    3,
    labels=['short', 'medium', 'long']
)

print('First threshold:',
      df[df['length_group'] == 'short']['log_length'].max())

print('Second threshold:',
      df[df['length_group'] == 'medium']['log_length'].max())

# -----------------------------
# EDA: RAW LENGTH
# -----------------------------

df['length'].hist(bins=20, color='#128bb5')

plt.title('IMDB Review Length (not log transformed)')
plt.xlabel('Character count')
plt.ylabel('Frequency')
plt.grid(False)

plt.savefig(output_path / "raw_review_length.png")
plt.show()

# -----------------------------
# EDA: LOG LENGTH
# -----------------------------

df['log_length'].hist(bins=20, color='#128bb5')

plt.axvline(
    x=df[df['length_group'] == 'short']['log_length'].max(),
    color='#DBA506',
    linestyle='--'
)

plt.axvline(
    x=df[df['length_group'] == 'medium']['log_length'].max(),
    color='#DBA506',
    linestyle='--'
)

plt.title('IMDB Review Length divided by length category')
plt.xlabel('Length (Log Scale)')
plt.ylabel('Frequency')
plt.grid(False)

plt.savefig(output_path / "log_transformed_review_length.png")
plt.show()

#qq-plot

sm.qqplot(df['log_length'], line='45')

plt.title("Q-Q Plot of Logarithmic Length")

plt.savefig(output_path / "q_q_plot.png")
plt.show()

#saving output

df.to_csv(output_path / "cleaned_reviews.csv", index=False)

print("Data cleaning + EDA complete.")