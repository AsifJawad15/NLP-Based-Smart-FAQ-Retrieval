"""Simple, reusable text preprocessing for FAQ questions."""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer


PROJECT_NLTK_DATA = Path(__file__).resolve().parents[1] / ".nltk_data"
if PROJECT_NLTK_DATA.is_dir():
    nltk.data.path.insert(0, str(PROJECT_NLTK_DATA))

TOKENIZER = TreebankWordTokenizer()
NEGATION_WORDS = {"no", "not", "nor"}
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
UNWANTED_PATTERN = re.compile(r"[^a-z0-9'\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _resource_error(resource_name: str) -> RuntimeError:
    """Create a beginner-friendly message for a missing NLTK resource."""

    return RuntimeError(
        f"NLTK resource '{resource_name}' is missing. Run: "
        f"python -m nltk.downloader {resource_name}"
    )


@lru_cache(maxsize=1)
def _english_stopwords() -> set[str]:
    """Load English stop words while retaining words that express negation."""

    try:
        words = set(stopwords.words("english"))
    except LookupError as error:
        raise _resource_error("stopwords") from error
    return words - NEGATION_WORDS


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    """Return one reusable WordNet lemmatizer."""

    for resource_path in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            nltk.data.find(resource_path)
            break
        except LookupError:
            continue
    else:
        raise _resource_error("wordnet")
    return WordNetLemmatizer()


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:
    """Normalize and tokenize text, returning a space-separated token string.

    Input: raw text and two optional preprocessing switches.
    Output: normalized text ready for TF-IDF vectorization.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned = html.unescape(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = HTML_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("’", "'")
    cleaned = UNWANTED_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    tokens = TOKENIZER.tokenize(cleaned)
    tokens = ["not" if token == "n't" else token for token in tokens]
    tokens = [token.strip("'") for token in tokens if token.strip("'")]

    if remove_stopwords:
        stop_words = _english_stopwords()
        tokens = [token for token in tokens if token not in stop_words]

    if lemmatize:
        lemmatizer = _lemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)
