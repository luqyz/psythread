"""
Merges consented feedback data (data/feedback_training_data.csv) into the
original labeled dataset (data/labeled_data.csv), producing a new file
ready to feed into src/train.py.

This does NOT overwrite labeled_data.csv or retrain anything automatically
— it just prepares the merged data. Review the output, then either rename
it to labeled_data.csv (backing up the original first) or point train.py
at it directly, before running training.

Usage:
    python data/merge_feedback.py
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
FEEDBACK_PATH = os.path.join(BASE_DIR, "data", "feedback_training_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "labeled_data_merged.csv")

VALID_LABELS = [
    "Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bipolar",
    "Personality disorder",
]


def main():
    if not os.path.exists(ORIGINAL_PATH):
        raise SystemExit(f"Couldn't find {ORIGINAL_PATH} — run data/prepare_dataset.py first.")
    if not os.path.exists(FEEDBACK_PATH):
        raise SystemExit(
            f"No feedback data yet at {FEEDBACK_PATH} — nothing to merge. "
            f"This file only appears once someone opts in via the 'Also save this "
            f"text' checkbox when submitting feedback."
        )

    original = pd.read_csv(ORIGINAL_PATH)[["content", "label"]]
    feedback = pd.read_csv(FEEDBACK_PATH)[["content", "label"]]

    feedback = feedback[feedback["label"].isin(VALID_LABELS)]
    feedback = feedback.dropna(subset=["content", "label"])
    feedback["content"] = feedback["content"].astype(str).str.strip()
    feedback = feedback[feedback["content"] != ""]

    print(f"Original dataset: {len(original)} rows")
    print(f"Feedback data:    {len(feedback)} rows")
    if len(feedback):
        print("\nFeedback label distribution:")
        print(feedback["label"].value_counts())

    # Feedback rows go FIRST, so if the same text appears in both (e.g. a
    # correction on a statement drawn from the original data), the
    # feedback's corrected label wins when duplicates are dropped.
    merged = pd.concat([feedback, original], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["content"], keep="first")
    after = len(merged)

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMerged: {before} rows -> {after} after removing {before - after} duplicate(s)")
    print(f"Saved to {OUTPUT_PATH}")
    print(
        "\nThis did NOT touch labeled_data.csv. To actually use this for training:\n"
        "  1. Review labeled_data_merged.csv\n"
        "  2. Back up labeled_data.csv (e.g. rename to labeled_data_backup.csv)\n"
        "  3. Rename labeled_data_merged.csv to labeled_data.csv\n"
        "  4. Run src/train.py as usual"
    )


if __name__ == "__main__":
    main()