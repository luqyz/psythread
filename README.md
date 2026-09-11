# FYP v2 — Sentiment Analysis of Social Media Psychological Behavior

A rebuild of the original FYP (SVM + LDA + VADER on a static Reddit CSV, static
HTML dashboard) into a real, trainable pipeline:

- **Data**: the "Sentiment Analysis for Mental Health" Kaggle dataset (Sarkar, 2024) —
  ~52k statements combining Reddit, Twitter, and Facebook text, already labeled into
  7 mental-health status categories. Reddit's API now requires manual approval for any
  scraping used to train ML models (Responsible Builder Policy, updated Nov 2025), so
  this uses an existing, already-compliant public dataset instead of a fresh scrape.
- **Model**: fine-tuned DistilBERT (transformer) for 7-class classification — a real
  trained model, not just VADER lexicon scoring.
- **Topic modeling**: LDA, run separately per status category (Depression, Anxiety,
  Stress, etc.) — same unsupervised approach as the original.
- **Severity flagging**: the model's predicted status is mapped to the original FYP's
  four-tier system (Crisis/Red/Amber/Green), with an explicit crisis-keyword safety net
  layered on top.
- **App**: Flask backend + HTML/CSS/JS dashboard (Chart.js), same spirit as the
  original static dashboard, now backed by a real trained model.

## Project structure

```
fyp-sentiment-v2/
├── requirements.txt
├── data/
│   ├── raw_dataset.csv        # you download this from Kaggle, see step 1 below
│   ├── prepare_dataset.py     # cleans it into data/labeled_data.csv
│   └── labeled_data.csv       # output of prepare_dataset.py
├── src/
│   ├── preprocess.py          # cleaning, tokenizing (shared by training + inference)
│   ├── train.py                # fine-tunes DistilBERT on the labeled CSV
│   ├── topic_model.py          # LDA topic modeling per status category
│   └── evaluate.py             # accuracy / precision / recall / F1 + confusion matrix
├── models/                     # trained model gets saved here (large, gitignore it)
├── app/
│   ├── app.py                  # Flask API + dashboard routes
│   ├── templates/index.html
│   └── static/{style.css,script.js}
└── README.md
```

## Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder`).
2. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet omw-1.4
   ```

## Pipeline — run in this order

### 1. Download the dataset
Get a free Kaggle account, then download:
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

The file is usually named `Combined Data.csv`. Rename it (or copy it) to
`data/raw_dataset.csv`.

### 2. Prepare the data
```bash
python data/prepare_dataset.py
```
Cleans and validates the dataset, writes `data/labeled_data.csv` with just
`content` and `label` columns — ready for training.

### 3. Train the transformer
```bash
python src/train.py
```
Fine-tunes `distilbert-base-uncased` on the 7 status categories (Normal, Depression,
Suicidal, Anxiety, Stress, Bipolar, Personality disorder). Saves the model + tokenizer
to `models/sentiment_model/`. Takes longer than a 3-class model would (7 classes, ~52k
rows) — expect 30–90 min on CPU, much faster with a GPU (Google Colab if your machine
doesn't have one, same tool your original project already used).

### 4. Evaluate
```bash
python src/evaluate.py
```
Prints accuracy, precision, recall, F1, and a confusion matrix on a held-out test
split. Note this isn't directly comparable to your original SVM's 92.6% (that was
3-class positive/negative/neutral; this is a harder 7-class problem) — worth explaining
that distinction explicitly in your FYP as a "more granular but harder task" tradeoff.

### 5. Topic modeling
```bash
python src/topic_model.py
```
Runs LDA separately on each status category's subset, printing dominant keywords per
topic — e.g. what specifically Depression-labeled statements tend to discuss vs.
Anxiety-labeled ones.

### 6. Run the app
```bash
python app/app.py
```
Opens a Flask server at `http://127.0.0.1:5000` with:
- A text box to analyze any input live (calls the trained DistilBERT model)
- A dashboard showing status distribution, severity tiers, and topic keywords from the
  dataset (pie/bar charts)

## Notes for your FYP writeup

- Explaining *why* you switched from a fresh Reddit scrape to an existing Kaggle
  dataset is a legitimate, citable methodology decision: Reddit closed self-service API
  registration in November 2025 and now requires prior approval for any scraping used
  to train ML models, including non-commercial academic use (Reddit's Responsible
  Builder Policy). Citing this policy change is a stronger, more defensible explanation
  than just saying "used a Kaggle dataset instead."
- Swapping SVM+TF-IDF for a fine-tuned transformer is a legitimate, discussable
  improvement in itself: TF-IDF treats words as independent counts, DistilBERT captures
  context (e.g. "not happy" vs "happy"), which typically raises F1 on short, informal
  text.
- Keep your original CRISP-DM framing — it still applies cleanly: business
  understanding, data understanding (Kaggle dataset selection + justification),
  data preparation (prepare_dataset.py), modeling (train.py), evaluation
  (evaluate.py), deployment (app.py).
- The move from 3-class (positive/negative/neutral) to 7-class (specific mental health
  statuses) is itself a meaningful scope increase worth highlighting — it's a harder,
  more clinically useful classification task than the original.
