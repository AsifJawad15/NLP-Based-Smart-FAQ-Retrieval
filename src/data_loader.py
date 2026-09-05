"""Corpus discovery, loading, and validation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FAQ_COLUMNS = ["id", "question", "answer", "category", "source", "source_type"]
QUERY_COLUMNS = ["query", "expected_faq_id", "is_answerable"]
DEFAULT_CONFIG = {
    "display_name": "FAQ Corpus",
    "remove_stopwords": False,
    "lemmatize": False,
    "similarity_threshold": 0.30,
}


def _normalized_question(text: str) -> str:
    """Normalize a question only for exact-like duplicate detection."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def discover_corpora(data_root: str | Path) -> dict[str, Path]:
    """Find corpus folders that contain an FAQ dataset.

    Input: directory containing one subdirectory per corpus.
    Output: sorted mapping from corpus key to corpus directory.
    """

    root = Path(data_root)
    if not root.exists():
        return {}

    corpora = {
        directory.name: directory
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "faq_dataset.csv").is_file()
    }
    return dict(sorted(corpora.items()))


def validate_faq_data(faq_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the common FAQ table schema."""

    missing = [column for column in FAQ_COLUMNS if column not in faq_data.columns]
    if missing:
        raise ValueError(f"FAQ dataset is missing columns: {', '.join(missing)}")

    data = faq_data[FAQ_COLUMNS].copy()
    if data.empty:
        raise ValueError("FAQ dataset must contain at least one row")

    try:
        numeric_ids = pd.to_numeric(data["id"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("FAQ ids must be integers") from error
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError("FAQ ids must be integers")
    data["id"] = numeric_ids.astype(int)

    if data["id"].duplicated().any():
        duplicate_ids = data.loc[data["id"].duplicated(), "id"].tolist()
        raise ValueError(f"FAQ dataset contains duplicate ids: {duplicate_ids[:5]}")

    for column in FAQ_COLUMNS[1:]:
        data[column] = data[column].fillna("").astype(str).str.strip()
        if data[column].eq("").any():
            raise ValueError(f"FAQ column '{column}' contains blank values")

    if data["question"].duplicated().any():
        raise ValueError("FAQ dataset contains exact duplicate questions")

    normalized = data["question"].map(_normalized_question)
    if normalized.duplicated().any():
        raise ValueError("FAQ dataset contains normalized duplicate questions")

    return data.reset_index(drop=True)


def load_faq_dataset(corpus_dir: str | Path) -> pd.DataFrame:
    """Load and validate `faq_dataset.csv` from a corpus directory."""

    path = Path(corpus_dir) / "faq_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(f"FAQ dataset not found: {path}")
    return validate_faq_data(pd.read_csv(path))


def _parse_answerable(value: object) -> bool:
    """Convert common CSV boolean spellings to a Python boolean."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid is_answerable value: {value!r}")


def load_query_dataset(
    path: str | Path,
    faq_ids: set[int] | None = None,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Load queries, allowing header-only manual templates only when requested."""

    query_path = Path(path)
    if not query_path.is_file():
        raise FileNotFoundError(f"Query dataset not found: {query_path}")

    raw = pd.read_csv(query_path)
    missing = [column for column in QUERY_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Query dataset is missing columns: {', '.join(missing)}")

    data = raw[QUERY_COLUMNS].copy()
    if data.empty:
        if allow_empty:
            return data
        raise ValueError("Query dataset must contain at least one row")
    data["query"] = data["query"].fillna("").astype(str).str.strip()
    if data["query"].eq("").any():
        raise ValueError("Query dataset contains blank queries")
    data["is_answerable"] = data["is_answerable"].map(_parse_answerable)
    data["expected_faq_id"] = pd.to_numeric(
        data["expected_faq_id"], errors="coerce"
    ).astype("Int64")

    answerable = data["is_answerable"]
    if data.loc[answerable, "expected_faq_id"].isna().any():
        raise ValueError("Answerable queries require an expected_faq_id")
    if data.loc[~answerable, "expected_faq_id"].notna().any():
        raise ValueError("Unanswerable queries must have a blank expected_faq_id")

    if faq_ids is not None:
        unknown_ids = set(data.loc[answerable, "expected_faq_id"].astype(int)) - faq_ids
        if unknown_ids:
            raise ValueError(f"Queries reference unknown FAQ ids: {sorted(unknown_ids)[:5]}")

    return data.reset_index(drop=True)


def load_corpus_config(corpus_dir: str | Path) -> dict[str, object]:
    """Load a corpus configuration, using baseline defaults when absent."""

    directory = Path(corpus_dir)
    config = DEFAULT_CONFIG.copy()
    config["display_name"] = directory.name.replace("_", " ").title()
    path = directory / "corpus_config.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            config.update(json.load(file))

    threshold = float(config["similarity_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    config["similarity_threshold"] = threshold
    config["remove_stopwords"] = _parse_flag(config["remove_stopwords"], "remove_stopwords")
    config["lemmatize"] = _parse_flag(config["lemmatize"], "lemmatize")
    return config


def _parse_flag(value: object, name: str) -> bool:
    """Read a preprocessing switch without treating "false" as True."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean for '{name}': {value!r}")
