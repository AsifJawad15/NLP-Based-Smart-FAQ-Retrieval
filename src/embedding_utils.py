"""Small, explicit sentence-vector calculations for the lab demonstration."""

from __future__ import annotations

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
