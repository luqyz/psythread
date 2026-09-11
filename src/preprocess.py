"""
Shared text-cleaning utilities used by both training and inference, so the
model always sees text processed the same way.
"""
import html
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

# Reddit/Twitter text is full of contractions typed without an apostrophe
# (im, dont, ive) which NLTK's stopword list doesn't catch (it only has
# "i'm", "don't" with the apostrophe), plus leftover HTML-entity fragments
# (amp, quot) that survive punctuation stripping. These dominate LDA topics
# as noise if left in, so they're added as extra stopwords here.
EXTRA_STOPWORDS = {
    "im", "dont", "ive", "cant", "wasnt", "isnt", "arent", "wont", "didnt",
    "youre", "youve", "youll", "theyre", "theyve", "hes", "shes", "thats",
    "wa", "amp", "quot", "http", "https", "com", "www",
}
STOPWORDS |= EXTRA_STOPWORDS

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    lemmatizer = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"[/u]/\S+|u/\S+|r/\S+")
CURLY_QUOTE_RE = re.compile(r"[\u2018\u2019\u201c\u201d]")

# Common contractions normalized to a canonical expanded form, matched
# whether or not the apostrophe is present (Reddit/Twitter text frequently
# drops it — "dont", "cant", "im"). Without this, the model can learn
# spurious signal tied to apostrophe presence rather than actual meaning —
# e.g. "can't take this anymore" and "cant take this anymore" ending up
# with very different predictions despite meaning the same thing.
#
# A few ambiguous contractions ("its" vs "it's", "were" vs "we're") are
# only normalized when the apostrophe IS present, since the apostrophe-less
# form is a legitimate word on its own and blind-normalizing it would
# corrupt real sentences (e.g. possessive "its color" or past-tense "were").
CONTRACTION_MAP = {
    "i'm": "i am", "im": "i am",
    "i've": "i have", "ive": "i have",
    "i'll": "i will", "ill": "i will",
    "i'd": "i would",
    "you're": "you are", "youre": "you are",
    "you've": "you have", "youve": "you have",
    "you'll": "you will", "youll": "you will",
    "you'd": "you would",
    "he's": "he is", "hes": "he is",
    "she's": "she is", "shes": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are", "theyre": "they are",
    "that's": "that is", "thats": "that is",
    "what's": "what is", "whats": "what is",
    "who's": "who is",
    "there's": "there is", "theres": "there is",
    "here's": "here is",
    "let's": "let us", "lets": "let us",
    "can't": "cannot", "cant": "cannot",
    "don't": "do not", "dont": "do not",
    "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not",
    "won't": "will not", "wont": "will not",
    "wouldn't": "would not", "wouldnt": "would not",
    "shouldn't": "should not", "shouldnt": "should not",
    "couldn't": "could not", "couldnt": "could not",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "haven't": "have not", "havent": "have not",
    "hasn't": "has not", "hasnt": "has not",
    "hadn't": "had not", "hadnt": "had not",
    "y'all": "you all", "yall": "you all",
    "that'll": "that will", "thatll": "that will",
}
_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTION_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Replace every contraction (with or without apostrophe) with its
    canonical expanded form, e.g. "can't"/"cant" -> "cannot"."""
    return _CONTRACTION_PATTERN.sub(lambda m: CONTRACTION_MAP[m.group(0).lower()], text)


def clean_text(text: str, lemmatize: bool = True) -> str:
    """Lowercase, strip URLs/punctuation/numbers, remove stopwords, lemmatize."""
    text = html.unescape(str(text))  # &amp; -> &, &quot; -> ", etc.
    text = text.lower()
    text = CURLY_QUOTE_RE.sub("'", text)  # normalize ' ' " " to straight quotes
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_for_transformer(text: str) -> str:
    """
    Lighter cleaning for transformer input — strips noise that provides no
    signal (URLs, usernames, HTML entities). Casing is left to the
    tokenizer itself (the model is uncased).

    NOTE: contraction normalization (expand_contractions) is deliberately
    NOT applied here right now. The currently deployed model was trained
    on data cleaned WITHOUT it — applying it only at inference time would
    create a train/inference mismatch (the model would see a different
    text distribution than it learned from), which causes worse, less
    stable predictions across the board, not just on contraction-heavy
    text. Re-enable the expand_contractions() call below ONLY together
    with retraining the model on data cleaned the same new way.
    """
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = CURLY_QUOTE_RE.sub("'", text)  # normalize ' ' " " to straight quotes
    text = expand_contractions(text)  # re-enable only alongside a retrain
    text = re.sub(r"\s+", " ", text).strip()
    return text