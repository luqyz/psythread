"""
Runs LDA topic modeling separately on each mental-health status category
(Depression, Anxiety, Stress, etc.), same approach as the original FYP —
topics like academic stress, social support, loneliness, or motivation
should emerge within each category without manual labeling.

Usage:
    python src/topic_model.py
"""
import json
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

from preprocess import clean_text

DATA_PATH = "data/labeled_data.csv"
OUTPUT_PATH = "data/topics.json"
NUM_TOPICS = 5
NUM_WORDS_PER_TOPIC = 8


def run_lda(texts, num_topics=NUM_TOPICS):
    tokenized = [clean_text(t).split() for t in texts]
    tokenized = [t for t in tokenized if len(t) > 2]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
    )

    topics = []
    for idx in range(num_topics):
        # show_topic returns real (word, probability) pairs directly from the
        # model — using these instead of parsing print_topics' formatted
        # string means the dashboard can chart actual topic weight, not just
        # word rank/position.
        word_weight_pairs = lda.show_topic(idx, topn=NUM_WORDS_PER_TOPIC)
        topics.append({
            "topic_id": idx,
            "keywords": [w for w, _ in word_weight_pairs],
            "weights": [round(float(weight), 5) for _, weight in word_weight_pairs],
        })
    return topics


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["content", "label"])

    results = {}
    for label in df["label"].unique():
        subset = df[df["label"] == label]["content"]
        print(f"Running LDA on {len(subset)} '{label}' posts...")
        if len(subset) < 10:
            print(f"  Skipping {label}: not enough data yet")
            continue
        results[label] = run_lda(subset)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    for label, topics in results.items():
        print(f"\n{label.upper()} topics:")
        for t in topics:
            print(f"  Topic {t['topic_id']}: {', '.join(t['keywords'])}")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()