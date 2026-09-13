"""
Exports feedback data from Firestore (where the production app writes it)
back into the same CSV format merge_feedback.py expects, so the existing
merge workflow keeps working unchanged.

Requires being authenticated to the same GCP project the app uses:
    gcloud auth application-default login
    gcloud config set project psythread   (or whatever your project id is)

Usage:
    python data/export_firestore_feedback.py
"""
import csv
import os

from google.cloud import firestore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "data", "feedback_log.csv")
FEEDBACK_TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "feedback_training_data.csv")


def export_collection(db, collection_name, output_path, fieldnames):
    docs = list(db.collection(collection_name).stream())
    print(f"{collection_name}: {len(docs)} documents found")
    if not docs:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for doc in docs:
            row = doc.to_dict()
            writer.writerow([row.get(field, "") for field in fieldnames])

    print(f"  -> saved to {output_path}")


def main():
    db = firestore.Client()

    export_collection(
        db, "feedback_log", FEEDBACK_LOG_PATH,
        ["timestamp_utc", "predicted_label", "was_accurate", "corrected_label"],
    )
    export_collection(
        db, "feedback_training_data", FEEDBACK_TRAINING_DATA_PATH,
        ["timestamp_utc", "content", "label", "was_correction"],
    )

    print("\nDone. You can now run data/merge_feedback.py as usual.")


if __name__ == "__main__":
    main()