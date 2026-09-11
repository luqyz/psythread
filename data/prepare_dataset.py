"""
Loads and cleans the "Sentiment Analysis for Mental Health" Kaggle dataset
(Sarkar, 2024) — https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

This dataset is already labeled (no VADER weak-labeling step needed, unlike
a raw scrape), combining text from Reddit, Twitter, and Facebook into 7
mental-health status categories: Normal, Depression, Suicidal, Anxiety,
Stress, Bipolar, Personality disorder.

Setup:
    1. Download the dataset from the Kaggle link above (you'll need a free
       Kaggle account). The file is usually named "Combined Data.csv".
    2. Place it at data/raw_dataset.csv (rename it, or point RAW_PATH below
       at wherever you saved it).
    3. Run: python data/prepare_dataset.py

This produces data/labeled_data.csv with columns: content, label — ready
for src/train.py.
"""
import os
import pandas as pd

RAW_PATH = "data/raw_dataset.csv"
OUTPUT_PATH = "data/labeled_data.csv"

VALID_LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality disorder",
]


def prepare():
    if not os.path.exists(RAW_PATH):
        raise SystemExit(
            f"Couldn't find {RAW_PATH}. Download the dataset from Kaggle "
            f"and save it there first (see the docstring at the top of this file)."
        )

    df = pd.read_csv(RAW_PATH)

    # The Kaggle CSV usually has an unnamed index column plus 'statement' and 'status'.
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "statement" not in df.columns or "status" not in df.columns:
        raise SystemExit(
            f"Expected columns 'statement' and 'status', found: {list(df.columns)}. "
            f"Check you downloaded the right file."
        )

    df = df.dropna(subset=["statement", "status"])
    df["status"] = df["status"].str.strip()
    df = df[df["status"].isin(VALID_LABELS)]
    df = df.drop_duplicates(subset=["statement"])

    out = df.rename(columns={"statement": "content", "status": "label"})[["content", "label"]]
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out)} labeled examples to {OUTPUT_PATH}")
    print("\nLabel distribution:")
    print(out["label"].value_counts())


if __name__ == "__main__":
    prepare()
