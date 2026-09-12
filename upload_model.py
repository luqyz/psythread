"""
One-off script: uploads the trained model folder to Hugging Face Hub, so
it doesn't need to be pushed to GitHub (it's ~440MB, over GitHub's 100MB
per-file limit).

Usage:
    python upload_model.py
"""
from huggingface_hub import HfApi

REPO_ID = "luqyz/psythread-mentalbert"  
MODEL_FOLDER = "models/sentiment_model"

api = HfApi()
api.upload_folder(
    folder_path=MODEL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"Uploaded {MODEL_FOLDER} to https://huggingface.co/{REPO_ID}")