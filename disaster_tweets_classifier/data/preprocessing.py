"""Text preprocessing utilities.

Light cleaning is intentionally minimal for transformer-based models:
DeBERTa's tokenizer handles most edge cases. Heavy cleaning (stopwords,
lemmatization) is reserved for the TF-IDF baseline.
"""

from __future__ import annotations

import re
import unicodedata

import emoji

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_WS_RE = re.compile(r"\s+")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")


def clean_text_for_transformer(text: str) -> str:
    """Apply minimal cleaning suitable for transformer tokenizers."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = _REPEATED_CHARS_RE.sub(r"\1\1", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def clean_text_for_baseline(text: str) -> str:
    """Aggressive cleaning for the TF-IDF baseline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text
