import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, RobertaForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "DATA"

# Load cleaned data
df = pd.read_csv(data_path / "cleaned_reviews.csv")


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
torch.set_grad_enabled(False)



# Split long reviews into smaller character chunks.
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# Run a batch of chunks through the model
def roberta_sentiment_batch(chunks):
    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    scores = probs[:, 2] - probs[:, 0]

    return scores.cpu().numpy().tolist() # Return a list of scores for each chunk


def roberta_severity(text, sentiment_label):
    chunks = chunk_text(text)
    scores = roberta_sentiment_batch(chunks)
    avg_score = float(np.mean(scores))

    if sentiment_label.lower() == "negative":
        return -abs(avg_score)
    return abs(avg_score)

# Run model on each review
def compute_roberta_severity(df):
    severities = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        severity = roberta_severity(
            row["review"],
            row["sentiment"]
        )
        severities.append(severity)

    df["roberta_severity"] = severities
    return df


# Run model
# -------------

df = compute_roberta_severity(df)

df = df.drop(columns=['sentiment', 'length', 'log_length'])

df.to_csv(data_path / "reviews_with_severity.csv", index=False)

print("RoBERTa scoring complete.")
