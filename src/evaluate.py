"""
Loads the trained model and prints a full classification report + confusion
matrix on the held-out test split, using the same metrics (precision, recall,
F1) the original FYP report used, for direct comparison.

Usage:
    python src/evaluate.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import clean_for_transformer

DATA_PATH = "data/labeled_data.csv"
MODEL_DIR = "models/sentiment_model"
BATCH_SIZE = 32
LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality disorder",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["content", "label"])
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["content"].apply(clean_for_transformer)
    df["label_id"] = df["label"].map(LABEL2ID)

    _, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )
    texts = test_df["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Evaluating"):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            inputs.pop("token_type_ids", None)  # DistilBERT doesn't accept this field
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    y_true = test_df["label_id"].tolist()

    print("Classification report:")
    print(classification_report(y_true, preds, target_names=LABELS, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_true, preds),
        index=[f"true_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    ))


if __name__ == "__main__":
    main()