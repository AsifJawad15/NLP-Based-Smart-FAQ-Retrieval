"""Small, explicit sentence-vector calculations for the lab demonstration."""

from __future__ import annotations

from collections import Counter

import numpy as np


def _usable_vector(vector: np.ndarray) -> np.ndarray | None:
    """No direction means no retrieval signal, including exact cancellation."""

    if not np.isfinite(vector).all():
        raise ValueError("Word embeddings must contain finite numbers")
    return vector if np.linalg.norm(vector) > 1e-12 else None


def sentence_to_mean_vector(tokens: list[str], keyed_vectors) -> np.ndarray | None:
    """Average known token occurrences, retaining the effect of repetitions."""

    vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
    if not vectors:
        return None
    return _usable_vector(np.mean(np.asarray(vectors, dtype=np.float64), axis=0))


def sentence_to_tfidf_weighted_vector(
    tokens: list[str], keyed_vectors, idf_lookup: dict[str, float],
) -> np.ndarray | None:
    """Sum unique terms once, weighted by count * corpus-fitted smoothed IDF."""

    counts = Counter(t for t in tokens if t in keyed_vectors and t in idf_lookup)
    if not counts:
        return None
    weights = np.array([counts[t] * idf_lookup[t] for t in counts], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("TF-IDF weights must be finite and nonnegative")
    if weights.sum() <= 0:
        return None
    vectors = np.array([keyed_vectors[t] for t in counts], dtype=np.float64)
    return _usable_vector((weights @ vectors) / weights.sum())
