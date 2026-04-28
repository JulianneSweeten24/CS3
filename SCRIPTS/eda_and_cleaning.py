import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "DATA"
output_path = PROJECT_ROOT / "OUTPUT"

# Load data
df = pd.read_csv(data_path / "IMDB_Dataset.csv")

# Data Cleaning
# -------------

df['review'] = df['review'].str.replace('<br />', '')
df['length'] = df['review'].str.len() # Review length is based on character count
df['log_length'] = np.log1p(df['length']) # Taking log values creates normality of distribution; this will be the basis of our analysis
df['length_group'] = pd.qcut(df['log_length'], 3, labels=['short','medium','long']) # Create length groups based on log tertiles

print('First threshold:',
      df[df['length_group'] == 'short']['log_length'].max())

print('Second threshold:',
      df[df['length_group'] == 'medium']['log_length'].max())


# EDA
# -------------

# This produces a histogram of non log transformed review length
df['length'].hist(bins=20, color='#128bb5')

plt.title('IMDB Review Length (not log transformed)')
plt.xlabel('Character count')
plt.ylabel('Frequency')
plt.grid(False)

plt.savefig(output_path / "raw_review_length.png")
plt.show()

# This creates a histogram of review length (log transformed) with length groups marked off by dashed lines
df['log_length'].hist(bins=20, color='#128bb5')

plt.axvline(x=df[df['length_group'] == 'short']['log_length'].max(), color='#DBA506', linestyle='--', label='First threshold')
plt.axvline(x=df[df['length_group'] == 'medium']['log_length'].max(), color='#DBA506', linestyle='--', label='Second threshold')
plt.title('IMDB Review Length divided by length category')
plt.xlabel('Length (Logarithmic Scale)')
plt.ylabel('Frequency')
plt.grid(False)

plt.savefig(output_path / "log_transformed_review_length.png")
plt.show()

# This creates a Q-Q plot of logarithmic review lengthfig = sm.qqplot(df['log_length'])
fig = sm.qqplot(df['log_length'])

plt.title("Q-Q Plot of Logarithmic Length")

plt.savefig(output_path / "q_q_plot.png")
plt.show()

# Save cleaned dataset
df.to_csv(data_path / "cleaned_reviews.csv", index=False)

print("Data cleaning + EDA complete.")