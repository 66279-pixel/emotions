"""
preprocessing.py — Text cleaning utilities for the emotion prediction pipeline.
"""

import re


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/digits, collapse whitespace. Keeps apostrophes."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)   # keep letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text
