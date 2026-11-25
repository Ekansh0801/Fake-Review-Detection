"""
utils.py - helper functions for our Fake Review Detector

Used by both the Colab notebook (training) and app.py (the website),
so a review is processed the same way in both places.

Labels: 1 = fake (CG), 0 = genuine (OR)
"""

import re

import pandas as pd
from scipy.sparse import csr_matrix, hstack

# Words and phrases that fake reviews use a lot
GENERIC_WORDS = [
    "highly recommend", "must buy", "best product", "great product", "love it",
    "love this", "works great", "five stars", "perfect", "amazing",
    "excellent", "awesome", "very happy", "great quality", "would recommend",
]

FEATURE_NAMES = ["word_count", "avg_word_length", "unique_word_ratio",
                 "exclaim_count", "caps_ratio", "generic_count", "rating"]


def clean_text(text):
    """Lowercase the text and remove everything except letters, numbers and spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)        # remove links
    text = re.sub(r"[^a-z0-9 ]", " ", text)     # remove symbols
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


def extract_features(text, rating):
    """Return 7 simple numbers that describe one review."""
    text = str(text)
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    capitals = [c for c in letters if c.isupper()]
    lower_words = [w.lower() for w in words]

    return {
        "word_count": len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "unique_word_ratio": len(set(lower_words)) / len(words) if words else 0,
        "exclaim_count": text.count("!"),
        "caps_ratio": len(capitals) / len(letters) if letters else 0,
        "generic_count": sum(text.lower().count(w) for w in GENERIC_WORDS),
        "rating": float(rating),
    }


def features_table(texts, ratings):
    """Features for many reviews as a pandas DataFrame."""
    rows = [extract_features(t, r) for t, r in zip(texts, ratings)]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def make_input(model, texts, ratings):
    """Convert reviews into the numbers the model expects: TF-IDF words + 7 features."""
    texts = list(texts)
    X_words = model["tfidf"].transform([clean_text(t) for t in texts])
    X_feats = model["scaler"].transform(features_table(texts, ratings).values)
    return hstack([X_words, csr_matrix(X_feats)]).tocsr()


def predict_fake_probability(model, texts, ratings):
    """Probability (0 to 1) that each review is fake."""
    X = make_input(model, texts, ratings)
    return model["clf"].predict_proba(X)[:, 1]


def suspicious_words(model, text, top_n=5):
    """Words in this review that the model links most strongly with fake / genuine reviews."""
    vocab = model["tfidf"].vocabulary_
    weights = model["clf"].coef_[0]
    found = {w: weights[vocab[w]] for w in set(clean_text(text).split()) if w in vocab}
    ordered = sorted(found.items(), key=lambda x: x[1])
    fake_words = [w for w, v in reversed(ordered) if v > 0][:top_n]
    genuine_words = [w for w, v in ordered if v < 0][:top_n]
    return fake_words, genuine_words
