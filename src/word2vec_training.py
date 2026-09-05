"""Question-only Word2Vec training and checked local model persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_faq_dataset
from src.preprocessing import preprocess_text


MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
BASIC_OPTIONS = {"remove_stopwords": False, "lemmatize": False}
TRAINING_SETTINGS = {
    "vector_size": 100, "window": 5, "min_count": 1,
    "sg": 1, "negative": 5, "hs": 0,
    "epochs": 30, "seed": 42, "workers": 1,
    "sample": 0, "shrink_windows": False, "sorted_vocab": 1,
    "alpha": 0.025, "min_alpha": 0.0001,
}


def _word2vec_class():
    """Import the extra dependency only for Word2Vec operations."""

    try:
        from gensim.models import Word2Vec
    except ImportError as error:
        raise RuntimeError(
            "Gensim is unavailable in this interpreter. Install the project requirements."
        ) from error
    return Word2Vec


def question_sentences(faq_data: pd.DataFrame) -> list[list[str]]:
    """Keep CSV question order; never read answers or evaluation queries."""

    sentences = [preprocess_text(q, **BASIC_OPTIONS).split() for q in faq_data["question"]]
    if not any(len(sentence) > 1 for sentence in sentences):
        raise ValueError("Word2Vec training requires questions with context word pairs")
    return sentences


def train_word2vec(faq_data: pd.DataFrame):
    """Train one model. Use the CLI's seeded subprocess for reproducibility."""

    return _word2vec_class()(sentences=question_sentences(faq_data), **TRAINING_SETTINGS)


def corpus_hash(corpus_dir: str | Path) -> str:
    return hashlib.sha256((Path(corpus_dir) / "faq_dataset.csv").read_bytes()).hexdigest()


def vector_hash(keyed_vectors) -> str:
    """Hash inference vocabulary/vectors, excluding serialization timestamps."""

    digest = hashlib.sha256(json.dumps(keyed_vectors.index_to_key).encode("utf-8"))
    vectors = np.asarray(keyed_vectors.vectors, dtype="<f4")
    digest.update(str(vectors.shape).encode("ascii"))
    digest.update(vectors.tobytes())
    return digest.hexdigest()


def artifact_id(metadata: dict[str, Any]) -> str:
    payload = {key: value for key, value in metadata.items() if key != "artifact_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def train_domain(corpus_dir: str | Path, models_root: str | Path = MODELS_ROOT) -> dict:
    """Explicitly train/save one domain, returning its reproducibility record."""

    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("Train through scripts/train_word2vec.py so Python starts with PYTHONHASHSEED=42")
    directory = Path(corpus_dir)
    faq_data = load_faq_dataset(directory)
    sentences = question_sentences(faq_data)
    model = train_word2vec(faq_data)
    import gensim
    import nltk
    import scipy

    metadata = {
        "corpus": directory.name,
        "corpus_sha256": corpus_hash(directory),
        "training_text": "faq_questions_only",
        "preprocessing_options": BASIC_OPTIONS.copy(),
        "training_settings": TRAINING_SETTINGS.copy(),
        "python_hash_seed": 42,
        "faq_count": len(faq_data),
        "token_count": sum(map(len, sentences)),
        "vocabulary_size": len(model.wv),
        "vector_sha256": vector_hash(model.wv),
        "package_versions": {
            "python": platform.python_version(), "gensim": gensim.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "nltk": nltk.__version__,
        },
    }
    metadata["artifact_id"] = artifact_id(metadata)
    target = Path(models_root) / directory.name
    target.mkdir(parents=True, exist_ok=True)
    model.save(str(target / "custom_word2vec.model"))
    (target / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_word2vec(corpus_dir: str | Path, models_root: str | Path = MODELS_ROOT):
    """Load a matching model; never train as an inference side effect."""

    directory = Path(corpus_dir)
    target = Path(models_root) / directory.name
    instruction = f"Run: python scripts/train_word2vec.py --corpus {directory.name}"
    model_path = target / "custom_word2vec.model"
    metadata_path = target / "training_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Word2Vec model or metadata missing for {directory.name}. {instruction}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = (
            metadata["corpus"] == directory.name
            and metadata["corpus_sha256"] == corpus_hash(directory)
            and metadata["training_text"] == "faq_questions_only"
            and metadata["preprocessing_options"] == BASIC_OPTIONS
            and metadata["training_settings"] == TRAINING_SETTINGS
            and metadata["artifact_id"] == artifact_id(metadata)
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid Word2Vec metadata. {instruction}") from error
    if not valid:
        raise ValueError(f"Stale or mismatched Word2Vec model for {directory.name}. {instruction}")
    model = _word2vec_class().load(str(model_path))
    if vector_hash(model.wv) != metadata["vector_sha256"]:
        raise ValueError(f"Word2Vec vectors do not match their metadata. {instruction}")
    return model, metadata
