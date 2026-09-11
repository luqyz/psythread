"""
Fine-tunes MentalBERT (mental/mental-bert-base-uncased) on the labeled
mental-health dataset for 7-class status classification (Normal, Depression,
Suicidal, Anxiety, Stress, Bipolar, Personality disorder).

MentalBERT is BERT-base further pretrained on mental-health-related Reddit
posts, so it starts with domain-relevant language understanding already
baked in — unlike generic distilbert-base-uncased, which has never seen
this kind of text before fine-tuning.

Note: this is a *gated* model on Hugging Face. Before running this script,
you must (1) have a Hugging Face account, (2) accept the license terms at
https://huggingface.co/mental/mental-bert-base-uncased, and (3) authenticate
locally, e.g. via `huggingface-cli login` or `login(token=...)` from
huggingface_hub — otherwise the download will fail with an auth error.

Uses class-weighted loss to counter the dataset's imbalance (Normal has
~16k examples, Personality disorder only ~900) — without this, the model
tends to under-predict rare classes since minimizing average loss is easiest
by just favoring the common ones.

Usage:
    python src/train.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from preprocess import clean_for_transformer

DATA_PATH = "data/labeled_data.csv"
MODEL_NAME = "mental/mental-bert-base-uncased"
OUTPUT_DIR = "models/sentiment_model"
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
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


class WeightedTrainer(Trainer):
    """A Trainer that applies class weights to the loss, so the model isn't
    free to just favor whichever classes have the most training examples."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["content", "label"])
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["content"].apply(clean_for_transformer)
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def tokenize_fn(tokenizer):
    def _tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    return _tokenize


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} labeled examples")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    train_ds = train_ds.map(tokenize_fn(tokenizer), batched=True)
    test_ds = test_ds.map(tokenize_fn(tokenizer), batched=True)

    # 'balanced' weighting: rarer classes get proportionally higher weight
    # in the loss, so misclassifying a rare Personality-disorder example
    # costs the model as much as misclassifying several common Normal ones.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABELS)),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("\nClass weights (higher = rarer class, weighted more in loss):")
    for label, weight in zip(LABELS, class_weights.tolist()):
        print(f"  {label}: {weight:.3f}")

    args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        # MentalBERT is full BERT-base (bigger than DistilBERT) — if you hit
        # a CUDA out-of-memory error on Colab, drop these to 8/16 instead.
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    print("\nFinal evaluation:")
    print(trainer.evaluate())

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()