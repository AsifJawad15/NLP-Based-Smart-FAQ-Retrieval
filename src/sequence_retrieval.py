"""Encode FAQ questions once with a trained sequence encoder; rank queries by cosine.

The ranking arrays match `rank_word2vec_queries`, so evaluation, threshold
tuning and paired predictions reuse the earlier phases' code unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.sequence_models import UNK_ID, pad_batch, token_ids


def encode_texts(
    encoder, vocabulary_index: dict[str, int], texts, max_len: int, batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sentence vectors, a has-tokens mask, and a usable-vector mask.

    A text with no word in the vocabulary is never encoded: an all-<UNK>
    sequence carries no information, exactly like an all-OOV Word2Vec query.
    """

    rows = [token_ids(str(text), vocabulary_index, max_len) for text in texts]
    has_tokens = np.array([bool(row) for row in rows], dtype=bool)
    known = np.array([any(index > UNK_ID for index in row) for row in rows], dtype=bool)
    vectors = np.zeros((len(rows), encoder.output_dim), dtype=np.float64)
    positions = np.flatnonzero(known)
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            ids, lengths = pad_batch([rows[position] for position in chunk])
            vectors[chunk] = encoder(ids, lengths).double().numpy()
    encoder.train(was_training)
    usable = known & (np.linalg.norm(vectors, axis=1) > 1e-12)
    return vectors, has_tokens, usable


def build_sequence_index(
    faq_data: pd.DataFrame, encoder, vocabulary_index: dict[str, int], max_len: int,
) -> dict[str, Any]:
    """Encode every FAQ question once; queries are encoded one batch at a time later."""

    vectors, _has_tokens, valid = encode_texts(encoder, vocabulary_index, faq_data["question"], max_len)
    return {
        "encoder": encoder,
        "vocabulary_index": vocabulary_index,
        "max_len": max_len,
        "faq_matrix": vectors,
        "valid_faqs": valid,
        "faq_ids": faq_data["id"].to_numpy().copy(),
    }


def rank_sequence_queries(
    queries, faq_data: pd.DataFrame, index: dict[str, Any], top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Return the same masked ranking arrays used by every other model."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not np.array_equal(index["faq_ids"], faq_data["id"].to_numpy()):
        raise ValueError("FAQ index ids/order do not match FAQ data")
    vectors, has_tokens, usable = encode_texts(
        index["encoder"], index["vocabulary_index"], list(queries), index["max_len"]
    )
    candidates = np.flatnonzero(index["valid_faqs"])
    # Retain one masked placeholder column if the corpus has no usable vectors.
    width = max(1, min(top_k, len(candidates)))
    ids = np.zeros((len(vectors), width), dtype=np.int64)
    scores = np.zeros((len(vectors), width), dtype=np.float64)
    has_features = usable & bool(len(candidates))
    if has_features.any():
        similarities = cosine_similarity(vectors[has_features], index["faq_matrix"][candidates])
        # Clip roundoff only; preserve negative cosine similarities.
        similarities = np.clip(similarities, -1.0, 1.0)
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
        ids[has_features] = faq_data["id"].to_numpy()[candidates[order]]
        scores[has_features] = np.take_along_axis(similarities, order, axis=1)
    return {"ranked_ids": ids, "ranked_scores": scores,
            "has_tokens": has_tokens, "has_features": has_features}


def retrieve_sequence(
    query: str, faq_data: pd.DataFrame, index: dict[str, Any], top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return descending FAQ dictionaries; a query with no known word returns []."""

    ranking = rank_sequence_queries([query], faq_data, index, top_k)
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


def answer_sequence(
    query: str, faq_data: pd.DataFrame, index: dict[str, Any], threshold: float, top_k: int = 3,
) -> dict[str, Any]:
    """Apply the same inclusive nonnegative threshold policy as every other model."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    matches = retrieve_sequence(query, faq_data, index, top_k)
    accepted = bool(matches and matches[0]["similarity"] >= threshold)
    return {"found": accepted,
            "message": "Relevant FAQ found." if accepted else "Sorry, I could not find a sufficiently relevant FAQ.",
            "best_match": matches[0] if accepted else None,
            "top_matches": matches, "threshold": threshold}
