"""Dense FAQ vectors, cosine ranking, and the existing answer contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_utils import sentence_to_mean_vector
from src.preprocessing import preprocess_text
from src.word2vec_training import BASIC_OPTIONS


def build_word2vec_index(faq_data: pd.DataFrame, keyed_vectors) -> dict[str, Any]:
    """Build question vectors from an already trained model; do not train here."""

    vectors = [sentence_to_mean_vector(preprocess_text(q, **BASIC_OPTIONS).split(), keyed_vectors)
               for q in faq_data["question"]]
    valid = np.array([vector is not None for vector in vectors], dtype=bool)
    matrix = np.zeros((len(faq_data), keyed_vectors.vector_size), dtype=np.float64)
    for i, vector in enumerate(vectors):
        if vector is not None:
            matrix[i] = vector
    return {
        "keyed_vectors": keyed_vectors, "faq_matrix": matrix, "valid_faqs": valid,
        "faq_ids": faq_data["id"].to_numpy().copy(), "preprocessing_options": BASIC_OPTIONS.copy(),
    }


def rank_word2vec_queries(
    queries: pd.Series, faq_data: pd.DataFrame, index: dict[str, Any], top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Return the same masked ranking arrays used by TF-IDF evaluation."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not np.array_equal(index["faq_ids"], faq_data["id"].to_numpy()):
        raise ValueError("FAQ index ids/order do not match FAQ data")
    texts = [preprocess_text(str(q), **index["preprocessing_options"]) for q in queries]
    vectors = [sentence_to_mean_vector(text.split(), index["keyed_vectors"]) for text in texts]
    candidates = np.flatnonzero(index["valid_faqs"])
    # Retain one masked placeholder column if the corpus has no usable vectors.
    width = max(1, min(top_k, len(candidates)))
    ids = np.zeros((len(texts), width), dtype=np.int64)
    scores = np.zeros((len(texts), width), dtype=np.float64)
    has_features = np.array([v is not None and bool(len(candidates)) for v in vectors], dtype=bool)
    if has_features.any():
        query_matrix = np.stack([vectors[i] for i in np.flatnonzero(has_features)])
        similarities = cosine_similarity(query_matrix, index["faq_matrix"][candidates])
        # Clip roundoff only; preserve negative cosine similarities.
        similarities = np.clip(similarities, -1.0, 1.0)
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
        ids[has_features] = faq_data["id"].to_numpy()[candidates[order]]
        scores[has_features] = np.take_along_axis(similarities, order, axis=1)
    return {"ranked_ids": ids, "ranked_scores": scores,
            "has_tokens": np.array([bool(text) for text in texts]), "has_features": has_features}


def retrieve_word2vec(
    query: str, faq_data: pd.DataFrame, index: dict[str, Any], top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return descending FAQ dictionaries; OOV queries return an empty list."""

    ranking = rank_word2vec_queries(pd.Series([query]), faq_data, index, top_k)
    if not ranking["has_features"][0]:
        return []
    by_id = faq_data.set_index("id")
    matches = []
    for faq_id, score in zip(ranking["ranked_ids"][0], ranking["ranked_scores"][0]):
        row = by_id.loc[int(faq_id)]
        matches.append({"faq_id": int(faq_id), "question": str(row["question"]),
                        "answer": str(row["answer"]), "category": str(row["category"]),
                        "source": str(row["source"]), "similarity": float(score)})
    return matches


def answer_word2vec(
    query: str, faq_data: pd.DataFrame, index: dict[str, Any], threshold: float,
    top_k: int = 3,
) -> dict[str, Any]:
    """Apply the same inclusive nonnegative threshold policy as Phase 1."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    matches = retrieve_word2vec(query, faq_data, index, top_k)
    accepted = bool(matches and matches[0]["similarity"] >= threshold)
    return {"found": accepted,
            "message": "Relevant FAQ found." if accepted else "Sorry, I could not find a sufficiently relevant FAQ.",
            "best_match": matches[0] if accepted else None,
            "top_matches": matches, "threshold": threshold}
