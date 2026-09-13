"""
Merges the supplementary "Reddit Mental Health Data" Kaggle dataset
(neelghoshal/reddit-mental-health-data) into the main labeled dataset,
targeting specifically the four weakest classes: Stress, Bipolar,
Personality disorder, and Anxiety. Depression rows from this dataset are
skipped since that class already has plenty of examples (~15k) in the
original dataset.

Setup:
    1. Download from https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data
    2. Save the CSV as data/reddit_mental_health_extra.csv

Usage:
    python data/add_extra_dataset.py
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_PATH = os.path.join(BASE_DIR, "data", "reddit_mental_health_extra.csv")
MAIN_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "labeled_data_augmented.csv")

# This dataset's numeric target -> our label names. Depression (1) is
# deliberately excluded -- it's not one of the weak classes and adding more
# would just re-skew the class balance we're trying to fix.
TARGET_MAP = {
    0: "Stress",
    2: "Bipolar",
    3: "Personality disorder",
    4: "Anxiety",
}

# Rows shorter than this (in words) tend to be low-information ("lol", "same",
# spam, deleted-post placeholders like "[removed]") -- they add noise rather
# than useful signal for a classifier that needs to learn category-specific
# language patterns.
MIN_WORD_COUNT = 5

# This dataset skews toward long-form posts (median ~100-120 words, some
# outliers past 1000). The model truncates at 128 tokens (~100 English
# words) during training/inference, so extremely long posts mostly waste
# their length -- only the first ~100 words are ever actually seen. Capping
# here just drops the most extreme outliers (>500 words) rather than
# keeping posts where 90%+ of the content is invisible to the model anyway.
MAX_WORD_COUNT = 500

# Reddit's "[deleted]"/"[removed]" placeholders and near-empty posts that
# survive a naive NaN check.
JUNK_VALUES = {"[deleted]", "[removed]", "nan", "none", ""}

# Automod/sticky "check-in" and subreddit-rules posts that show up
# repeatedly in Reddit scrapes -- these describe the subreddit itself, not
# a person's actual experience, and would teach the model the wrong thing
# about what "Stress"/"Anxiety"/etc. language looks like.
BOILERPLATE_PATTERNS = "check-in post|sidebar|automod|this is a place|wiki/|private_contact"


def main():
    if not os.path.exists(EXTRA_PATH):
        raise SystemExit(
            f"Couldn't find {EXTRA_PATH}. Download the dataset from Kaggle "
            f"(neelghoshal/reddit-mental-health-data) and save it there first."
        )
    if not os.path.exists(MAIN_PATH):
        raise SystemExit(f"Couldn't find {MAIN_PATH} -- run data/prepare_dataset.py first.")

    extra = pd.read_csv(EXTRA_PATH)
    extra.columns = [c.strip().lower() for c in extra.columns]

    # Combine title + body when both exist, since the classifier expects a
    # single text field (same approach as the original FYP's PRAW scraper).
    if "title" in extra.columns and "text" in extra.columns:
        extra["content"] = (
            extra["title"].fillna("").astype(str) + " " + extra["text"].fillna("").astype(str)
        ).str.strip()
    elif "text" in extra.columns:
        extra["content"] = extra["text"].astype(str)
    else:
        raise SystemExit(f"Expected a 'text' column, found: {list(extra.columns)}")

    if "target" not in extra.columns:
        raise SystemExit(f"Expected a 'target' column, found: {list(extra.columns)}")

    extra["label"] = extra["target"].map(TARGET_MAP)
    extra = extra.dropna(subset=["content", "label"])
    extra["content"] = extra["content"].str.strip()

    # Quality filtering, on top of the structural cleanup above:
    before_quality = len(extra)
    extra = extra[~extra["content"].str.lower().isin(JUNK_VALUES)]
    word_counts = extra["content"].str.split().str.len()
    extra = extra[(word_counts >= MIN_WORD_COUNT) & (word_counts <= MAX_WORD_COUNT)]
    extra = extra[~extra["content"].str.contains(BOILERPLATE_PATTERNS, case=False, na=False, regex=True)]
    extra = extra.drop_duplicates(subset=["content"])  # dedup within this dataset itself
    print(f"Quality filtering: {before_quality} -> {len(extra)} rows "
          f"(removed junk/placeholder/too-short/too-long/boilerplate/duplicate posts)")

    extra = extra[["content", "label"]]

    print("New rows pulled from supplementary dataset (by label):")
    print(extra["label"].value_counts())

    main_df = pd.read_csv(MAIN_PATH)[["content", "label"]]
    print(f"\nOriginal dataset: {len(main_df)} rows")

    # New rows go FIRST so if the same text somehow exists in both, we keep
    # the new source's label after de-duplication (unlikely to matter here
    # since these are different platforms/scrapes, but keeps behavior
    # consistent with merge_feedback.py).
    merged = pd.concat([extra, main_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["content"], keep="first")
    after = len(merged)

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMerged: {before} rows -> {after} after removing {before - after} duplicate(s)")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nNew overall label distribution:")
    print(merged["label"].value_counts())
    print(
        "\nThis did NOT touch labeled_data.csv. To use this for training:\n"
        "  1. Review labeled_data_augmented.csv\n"
        "  2. Back up labeled_data.csv (e.g. rename to labeled_data_backup.csv)\n"
        "  3. Rename labeled_data_augmented.csv to labeled_data.csv\n"
        "  4. Run src/train.py as usual"
    )


if __name__ == "__main__":
    main()
