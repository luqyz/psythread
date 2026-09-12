"""
Flask app: serves the trained DistilBERT model for live text analysis, plus
a dashboard summarizing the scraped/labeled dataset (sentiment distribution,
severity tiers, topic keywords) -- same idea as the original static
HTML/CSS/JS dashboard, now backed by a real model instead of a static CSV read.

Usage:
    python app/app.py
Then open http://127.0.0.1:5000
"""
import csv
import os
import json
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from langdetect import detect, LangDetectException, DetectorFactory
from deep_translator import GoogleTranslator

# langdetect's algorithm is probabilistic and non-deterministic by default --
# the same short text can get different results on different runs. Seeding
# it makes detection consistent, which matters a lot for a safety-relevant
# language check.
DetectorFactory.seed = 0

# Limits PyTorch's intra-op thread pool -- on a memory/CPU-constrained deploy
# (e.g. Render's free 512MB tier), letting torch spawn multiple threads adds
# overhead without much benefit since the instance itself only has a
# fraction of a CPU core anyway.
torch.set_num_threads(1)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_for_transformer  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "sentiment_model")
# Fallback source when the local model folder isn't present -- e.g. on a
# fresh deploy where the ~440MB model file was never pushed to Git (it's
# hosted separately on Hugging Face Hub instead). Set via environment
# variable so this stays configurable without touching code.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
LABELED_DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
TOPICS_PATH = os.path.join(BASE_DIR, "data", "topics.json")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "data", "feedback_log.csv")
FEEDBACK_TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "feedback_training_data.csv")

LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality disorder",
]

# Maps the model's predicted status to the original FYP's four-tier severity
# system, so the UI still shows a Crisis/Red/Amber/Green badge like before.
STATUS_TO_SEVERITY = {
    "Suicidal": "Crisis",
    "Depression": "Red",
    "Bipolar": "Red",
    "Personality disorder": "Red",
    "Anxiety": "Amber",
    "Stress": "Amber",
    "Normal": "Green",
}

# Kept as a secondary signal alongside the model's prediction -- catches
# explicit crisis language even if the model's top prediction is uncertain.
CRISIS_KEYWORDS = ["suicide", "kill myself", "end it all", "want to die", "not exist"]

# Same idea, but checked against the RAW untranslated text -- a safety net
# independent of translation quality. Translation of short, informal, or
# code-mixed non-English text isn't always reliable (see detect_and_translate),
# and a missed translation should never mean a missed crisis signal. Covers
# common informal Malay/Indonesian phrasing.
NON_ENGLISH_CRISIS_KEYWORDS = [
    "nak mati", "bunuh diri", "mengakhiri hidup", "tak nak hidup",
    "tak larat nak hidup", "nak give up", "give up je", "tak larat dah",
]

# Below this confidence, the UI shows "Uncertain" instead of asserting the
# top label -- with 7 classes, random chance is ~14%, so 45% is a meaningful
# but not overly strict bar for "the model actually has a signal here".
CONFIDENCE_THRESHOLD = 0.45

# Malaysian mental health crisis resources, shown when severity is Crisis or
# Red. Verified current as of 2026 -- re-check periodically, hotline details
# can change.
CRISIS_RESOURCES = [
    {"name": "Talian Kasih", "contact": "15999", "note": "24/7 government helpline, also on WhatsApp 019-261 5999"},
    {"name": "Talian HEAL", "contact": "15555", "note": "National Mental Health Crisis Line"},
    {"name": "Befrienders KL", "contact": "03-7627 2929", "note": "24/7 confidential emotional support"},
    {"name": "Emergency", "contact": "999", "note": "Police, fire, and medical emergencies"},
]

# Curated phrases per theme, matched against the contraction-normalized text
# (so "cant sleep" and "can't sleep" both match "cannot sleep" etc). This is
# plain keyword matching, NOT a trained model -- it flags recognizable
# patterns, it doesn't diagnose. Phrases are written in their post-
# normalization form since detect_symptom_themes() runs on normalized text.
SYMPTOM_THEMES = {
    "Sleep disturbance": [
        "cannot sleep", "can not sleep", "have not slept", "havent slept",
        "insomnia", "sleeping too much", "sleep all day", "no sleep",
    ],
    "Loss of interest": [
        "no motivation", "do not want to do anything", "lost interest",
        "nothing feels fun", "do not enjoy", "dont enjoy", "nothing matters",
    ],
    "Appetite change": [
        "not eating", "lost my appetite", "eating too much", "no appetite",
        "stopped eating",
    ],
    "Fatigue / low energy": [
        "so tired", "no energy", "exhausted", "drained", "so exhausted",
    ],
    "Racing thoughts / restlessness": [
        "racing thoughts", "cannot stop thinking", "restless", "on edge",
        "mind will not stop", "mind wont stop",
    ],
    "Hopelessness": [
        "no point", "hopeless", "pointless", "whats the point",
        "what is the point",
    ],
    "Social withdrawal": [
        "do not want to see anyone", "dont want to see anyone",
        "avoiding people", "isolating myself", "stopped talking to",
        "do not want to talk to anyone",
    ],
    "Concentration issues": [
        "cannot focus", "cannot concentrate", "mind is foggy",
        "trouble focusing", "cant think straight",
    ],
    "Physical anxiety symptoms": [
        "heart racing", "chest tight", "cannot breathe", "panic attack",
        "heart is racing", "chest is tight",
    ],
}


def detect_symptom_themes(text: str):
    """
    Flags recognizable themes present in the text by matching against
    curated phrases (see SYMPTOM_THEMES). This is keyword matching, not a
    trained classifier -- it's meant as a lightweight, explainable "what
    patterns are present" signal, not a diagnostic tool. Matching runs on
    contraction-normalized text so apostrophe presence doesn't affect it,
    same reasoning as the model's own input cleaning.
    """
    normalized = clean_for_transformer(text).lower()
    matched = []
    for theme, phrases in SYMPTOM_THEMES.items():
        if any(phrase in normalized for phrase in phrases):
            matched.append(theme)
    return matched


MAX_BATCH_ROWS = 150  # keeps batch requests reasonably fast (no LIME per row)

# Display names for common languages the analyzer will see in a Malaysian
# context -- langdetect returns ISO 639-1 codes, this maps them to names for
# the UI note ("Translated from Malay...").
LANGUAGE_NAMES = {
    "ms": "Malay", "id": "Indonesian", "en": "English", "zh-cn": "Chinese",
    "zh-tw": "Chinese", "ta": "Tamil", "hi": "Hindi", "ar": "Arabic",
    "th": "Thai", "ja": "Japanese", "ko": "Korean", "fr": "French",
    "es": "Spanish",
}

# Common Malay/Indonesian function words -- short, high-frequency, and rarely
# used any other way. Used only as a tiebreaker when langdetect says "en"
# but several of these appear, since that combination usually means Malay
# text sprinkled with English loanwords, not genuine English.
MALAY_FUNCTION_WORDS = {
    "tak", "nak", "dah", "je", "sangat", "sempat", "semua", "sebelum",
    "kerja", "banyak", "siapkan", "saya", "rasa", "sedih", "larat",
    "macam", "ni", "dengan", "kawan", "esok", "risau", "boleh", "tak",
    "yang", "ada", "kena", "buat", "orang", "hari", "pun", "lah",
}


def detect_and_translate(text: str):
    """
    Detects the input language and translates non-English text to English
    before analysis, since the model was trained exclusively on English
    social-media text. Falls back to the original text unchanged if
    detection or translation fails (e.g. offline, or text too short/
    ambiguous for langdetect), rather than blocking the whole request.

    langdetect is known to be unreliable on short text, and especially on
    Malay/Indonesian text that includes English loanwords ("deadline",
    "give up") -- it can misdetect the whole sentence as English and skip
    translation entirely. As a safety net, if langdetect says "en" but the
    text contains several common Malay function words, we override to "ms"
    and attempt translation anyway, since silently skipping translation on
    Malay text risks the model reading language it was never trained on.

    Passes the detected language explicitly as the translation source
    (rather than "auto") -- Google's own auto-detect is unreliable on short,
    slang-heavy, code-mixed text (a real failure mode seen with informal
    Malay), sometimes silently returning the input unchanged. We also check
    that the translation actually differs from the input before trusting
    it, so a failed "translation" isn't mistaken for a real one downstream.

    Returns (text_for_analysis, was_translated, detected_lang_code).
    """
    try:
        lang = detect(text)
    except LangDetectException:
        return text, False, "unknown"

    if lang == "en":
        text_words = set(re.findall(r"[a-z]+", text.lower()))
        malay_hits = text_words & MALAY_FUNCTION_WORDS
        if len(malay_hits) >= 2:
            lang = "ms"
        else:
            return text, False, "en"

        try:
        translated = GoogleTranslator(source=lang, target="en").translate(text)
        if translated and translated.strip().lower() != text.strip().lower():
            return translated, True, lang
    except Exception as e:
        print(f"GoogleTranslator failed: {e}")
        try:
            translated = MyMemoryTranslator(source=lang, target="en").translate(text)
            if translated and translated.strip().lower() != text.strip().lower():
                return translated, True, lang
        except Exception as e2:
            print(f"MyMemoryTranslator failed: {e2}")

    return text, False, lang


app = Flask(__name__)

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        if os.path.exists(MODEL_DIR):
            source = MODEL_DIR
        elif HF_MODEL_REPO:
            print(f"Local model not found at {MODEL_DIR} -- downloading from Hugging Face Hub: {HF_MODEL_REPO}")
            source = HF_MODEL_REPO
        else:
            raise RuntimeError(
                f"No trained model found at {MODEL_DIR}, and HF_MODEL_REPO env var "
                f"is not set. Run src/train.py first, or set HF_MODEL_REPO to a "
                f"Hugging Face Hub repo id containing the trained model."
            )
        _tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
        model.eval()
        _model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return _model, _tokenizer


def predict_proba_batch(texts):
    model, tokenizer = get_model()
    BATCH_CHUNK_SIZE = 16
    all_probs = []
    for i in range(0, len(texts), BATCH_CHUNK_SIZE):
        chunk = texts[i:i + BATCH_CHUNK_SIZE]
        cleaned = [clean_for_transformer(t) for t in chunk]
        inputs = tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=128, padding=True
        )
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


_explainer = LimeTextExplainer(class_names=LABELS)

LIME_NUM_SAMPLES = int(os.environ.get("LIME_NUM_SAMPLES", "100"))


def explain_prediction(text: str, predicted_label: str, num_features: int = 8, num_samples: int = None):
    if num_samples is None:
        num_samples = LIME_NUM_SAMPLES
    label_idx = LABELS.index(predicted_label)
    explanation = _explainer.explain_instance(
        text,
        predict_proba_batch,
        num_features=num_features,
        num_samples=num_samples,
        labels=[label_idx],
    )
    word_weights = explanation.as_list(label=label_idx)
    return [{"word": w, "weight": round(weight, 4)} for w, weight in word_weights]


def split_into_chunks(text: str, tokenizer, max_tokens: int = 120):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return [text]

    chunks = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        token_count = len(tokenizer.encode(candidate, add_special_tokens=True))
        if token_count > max_tokens and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _predict_chunk(cleaned_text: str, model, tokenizer):
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return {LABELS[i]: p for i, p in enumerate(probs)}


def predict_sentiment(text: str):
    model, tokenizer = get_model()
    cleaned = clean_for_transformer(text)

    total_tokens = len(tokenizer.encode(cleaned, add_special_tokens=True))
    if total_tokens <= 128:
        probs = _predict_chunk(cleaned, model, tokenizer)
        num_segments = 1
    else:
        chunks = split_into_chunks(cleaned, tokenizer)
        chunk_probs = [_predict_chunk(c, model, tokenizer) for c in chunks]
        probs = {label: sum(cp[label] for cp in chunk_probs) / len(chunk_probs) for label in LABELS}
        num_segments = len(chunks)

    pred_label = max(probs, key=probs.get)
    confidence = round(probs[pred_label], 4)
    result = {
        "label": pred_label,
        "confidence": confidence,
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    }
    return result, num_segments


def detect_severity(analysis_text: str, predicted_status: str, original_text: str = None):
    matched_crisis_keywords = [kw for kw in CRISIS_KEYWORDS if kw in analysis_text.lower()]

    if original_text and original_text.strip().lower() != analysis_text.strip().lower():
        original_lower = original_text.lower()
        matched_crisis_keywords += [kw for kw in NON_ENGLISH_CRISIS_KEYWORDS if kw in original_lower]

    if matched_crisis_keywords:
        tier = "Crisis"
    else:
        tier = STATUS_TO_SEVERITY.get(predicted_status, "Amber")

    result = {"tier": tier, "matched_keywords": matched_crisis_keywords}
    if tier in ("Crisis", "Red"):
        result["resources"] = CRISIS_RESOURCES
    return result


@app.route("/")
def index():
    return render_template("index.html", crisis_resources=CRISIS_RESOURCES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    analysis_text, was_translated, detected_lang = detect_and_translate(text)

    try:
        sentiment, num_segments = predict_sentiment(analysis_text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    severity = detect_severity(analysis_text, sentiment["label"], original_text=text)

    if num_segments == 1:
        try:
            explanation = explain_prediction(analysis_text, sentiment["label"])
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            explanation = []
    else:
        explanation = []

    themes = detect_symptom_themes(analysis_text)

    return jsonify({
        "sentiment": sentiment,
        "severity": severity,
        "explanation": explanation,
        "themes": themes,
        "num_segments": num_segments,
        "was_translated": was_translated,
        "detected_language": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        "translated_text": analysis_text if was_translated else None,
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    predicted_label = data.get("predicted_label")
    was_accurate = data.get("was_accurate")
    corrected_label = data.get("corrected_label")
    consent_save_text = bool(data.get("consent_save_text", False))
    text = data.get("text")

    if predicted_label not in LABELS or not isinstance(was_accurate, bool):
        return jsonify({"error": "Invalid feedback payload"}), 400
    if corrected_label is not None and corrected_label not in LABELS:
        return jsonify({"error": "Invalid corrected_label"}), 400

    is_new_file = not os.path.exists(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp_utc", "predicted_label", "was_accurate", "corrected_label"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            predicted_label,
            was_accurate,
            corrected_label or "",
        ])

    if consent_save_text and text and text.strip():
        training_label = corrected_label or predicted_label
        is_new_training_file = not os.path.exists(FEEDBACK_TRAINING_DATA_PATH)
        with open(FEEDBACK_TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new_training_file:
                writer.writerow(["timestamp_utc", "content", "label", "was_correction"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                text.strip(),
                training_label,
                bool(corrected_label),
            ])

    return jsonify({"status": "ok"})


@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read that CSV: {e}"}), 400

    text_col = next((c for c in ["text", "content", "statement", "message"] if c in df.columns), None)
    if text_col is None:
        return jsonify({
            "error": "CSV must have a column named 'text', 'content', 'statement', or 'message'"
        }), 400

    texts = [t.strip() for t in df[text_col].dropna().astype(str).tolist() if t.strip()]
    truncated = len(texts) > MAX_BATCH_ROWS
    texts = texts[:MAX_BATCH_ROWS]
    if not texts:
        return jsonify({"error": "No usable text found in that column"}), 400

    results = []
    for t in texts:
        analysis_t, _, _ = detect_and_translate(t)
        try:
            sentiment, _ = predict_sentiment(analysis_t)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        severity = detect_severity(analysis_t, sentiment["label"], original_text=t)
        results.append({
            "text_preview": (t[:80] + "...") if len(t) > 80 else t,
            "label": sentiment["label"],
            "confidence": sentiment["confidence"],
            "severity": severity["tier"],
        })

    label_distribution = {}
    severity_distribution = {}
    for r in results:
        label_distribution[r["label"]] = label_distribution.get(r["label"], 0) + 1
        severity_distribution[r["severity"]] = severity_distribution.get(r["severity"], 0) + 1

    return jsonify({
        "total": len(results),
        "results": results,
        "label_distribution": label_distribution,
        "severity_distribution": severity_distribution,
        "truncated": truncated,
    })


@app.route("/api/dashboard-data")
def dashboard_data():
    if not os.path.exists(LABELED_DATA_PATH):
        return jsonify({"error": "No labeled data found. Run the pipeline first."}), 404

    df = pd.read_csv(LABELED_DATA_PATH)
    sentiment_counts = df["label"].value_counts().to_dict()

    topics = {}
    if os.path.exists(TOPICS_PATH):
        with open(TOPICS_PATH) as f:
            topics = json.load(f)

    return jsonify(
        {
            "sentiment_distribution": sentiment_counts,
            "total_posts": len(df),
            "topics": topics,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)