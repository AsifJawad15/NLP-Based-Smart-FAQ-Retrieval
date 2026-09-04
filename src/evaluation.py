"""Validation threshold tuning and final test-set evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text
from src.tfidf_retrieval import build_tfidf_index


# Ordered from simplest to most complex, because the project plan breaks
# scoring ties by preferring the simpler preprocessing configuration.
PREPROCESSING_CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("basic", {"remove_stopwords": False, "lemmatize": False}),
    ("stopwords_removed", {"remove_stopwords": True, "lemmatize": False}),
    ("lemmatized", {"remove_stopwords": False, "lemmatize": True}),
]

# Integer steps avoid floating-point drift across the 0.00 to 1.00 sweep.
THRESHOLD_STEPS: list[float] = [step / 100 for step in range(101)]


def rank_queries(
    queries: pd.Series,
    faq_data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    faq_matrix: csr_matrix,
    preprocessing_options: dict[str, bool] | None = None,
    top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Rank every query once so threshold sweeps can reuse the similarities.

    Input: query texts, FAQ table, fitted index, preprocessing switches, top-k.
    Output: arrays of ranked FAQ ids, ranked scores, and a has-tokens mask.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if faq_matrix.shape[0] != len(faq_data):
        raise ValueError("FAQ matrix row count does not match FAQ data")

    options = preprocessing_options or {}
    processed = [preprocess_text(str(query), **options) for query in queries]

    # A query with no tokens left is always rejected, matching answer_query.
    has_tokens = np.array([bool(text) for text in processed], dtype=bool)

    query_matrix = vectorizer.transform(processed)
    similarities = cosine_similarity(query_matrix, faq_matrix)

    width = min(top_k, len(faq_data))
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
    faq_ids = faq_data["id"].to_numpy()

    return {
        "ranked_ids": faq_ids[order],
        "ranked_scores": np.take_along_axis(similarities, order, axis=1),
        "has_tokens": has_tokens,
    }


def _accepted(ranking: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    """Decide which queries clear the similarity threshold."""

    top_scores = ranking["ranked_scores"][:, 0]
    return ranking["has_tokens"] & (top_scores >= threshold)


def _split_masks(query_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a query table into answerable and unanswerable masks."""

    answerable = query_data["is_answerable"].to_numpy(dtype=bool)
    return answerable, ~answerable


def balanced_accept_reject_score(
    query_data: pd.DataFrame,
    ranking: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    """Score one threshold as the mean of accept and reject correctness.

    Input: query table, cached ranking, and one candidate threshold.
    Output: the acceptance rate, the rejection rate, and their average.
    """

    answerable, unanswerable = _split_masks(query_data)
    accepted = _accepted(ranking, threshold)

    accept_rate = float(accepted[answerable].mean()) if answerable.any() else 0.0
    reject_rate = float((~accepted[unanswerable]).mean()) if unanswerable.any() else 0.0
    return {
        "threshold": threshold,
        "answerable_acceptance_rate": accept_rate,
        "unanswerable_rejection_rate": reject_rate,
        "score": (accept_rate + reject_rate) / 2,
    }


def tune_threshold(
    validation_data: pd.DataFrame,
    faq_data: pd.DataFrame,
    configs: list[tuple[str, dict[str, bool]]] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Choose preprocessing and threshold using validation queries only.

    Input: validation queries, FAQ table, candidate configurations, top-k.
    Output: the selected configuration plus the full comparison sweep.
    """

    # An explicitly empty list is an error, so it must not fall back silently.
    candidates = PREPROCESSING_CONFIGS if configs is None else configs
    if not candidates:
        raise ValueError("At least one preprocessing configuration is required")

    sweep: list[dict[str, Any]] = []
    per_config: list[dict[str, Any]] = []

    for config_rank, (name, options) in enumerate(candidates):
        vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
        ranking = rank_queries(
            validation_data["query"], faq_data, vectorizer, faq_matrix, options, top_k
        )
        scored = [
            {
                "config": name,
                "config_rank": config_rank,
                **balanced_accept_reject_score(validation_data, ranking, threshold),
            }
            for threshold in THRESHOLD_STEPS
        ]
        sweep.extend(scored)

        # Ties inside one configuration are resolved towards the higher threshold.
        per_config.append(max(scored, key=lambda row: (row["score"], row["threshold"])))

    # Across configurations: highest score, then the simpler configuration,
    # then the higher threshold.
    best = max(
        sweep, key=lambda row: (row["score"], -row["config_rank"], row["threshold"])
    )

    return {
        "selected_config": best["config"],
        "selected_options": dict(candidates[best["config_rank"]][1]),
        "selected_threshold": best["threshold"],
        "selected_score": best["score"],
        "per_config_best": per_config,
        "sweep": sweep,
    }


def evaluate_tfidf(
    test_data: pd.DataFrame,
    faq_data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    faq_matrix: csr_matrix,
    threshold: float,
    preprocessing_options: dict[str, bool] | None = None,
    top_k: int = 3,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Report the frozen configuration's behaviour on a held-out query set.

    Input: query table, FAQ table, fitted index, frozen threshold and switches.
    Output: the metric dictionary required by the project plan.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    ranking = rank_queries(
        test_data["query"], faq_data, vectorizer, faq_matrix, preprocessing_options, top_k
    )
    answerable, unanswerable = _split_masks(test_data)
    accepted = _accepted(ranking, threshold)

    ranked_ids = ranking["ranked_ids"]
    top_scores = ranking["ranked_scores"][:, 0]
    # Unanswerable rows hold a blank expected id, so NaN keeps them unmatched.
    expected = pd.to_numeric(test_data["expected_faq_id"], errors="coerce").to_numpy(
        dtype="float64"
    )

    top1_correct = answerable & (ranked_ids[:, 0] == expected)
    top3_correct = answerable & (ranked_ids == expected[:, None]).any(axis=1)

    answerable_count = int(answerable.sum())
    unanswerable_count = int(unanswerable.sum())
    questions_by_id = faq_data.set_index("id")["question"]

    correct_scores = top_scores[top1_correct]
    incorrect_examples = [
        {
            "query": str(test_data.iloc[position]["query"]),
            "expected_faq_id": int(expected[position]),
            "retrieved_faq_id": int(ranked_ids[position, 0]),
            "retrieved_question": str(questions_by_id.loc[int(ranked_ids[position, 0])]),
            "similarity": float(top_scores[position]),
            "accepted": bool(accepted[position]),
        }
        for position in np.flatnonzero(answerable & ~top1_correct)[:max_examples]
    ]

    return {
        "threshold": threshold,
        "preprocessing_options": dict(preprocessing_options or {}),
        "answerable_queries": answerable_count,
        "unanswerable_queries": unanswerable_count,
        "top1_accuracy": (
            float(top1_correct.sum() / answerable_count) if answerable_count else 0.0
        ),
        "top3_accuracy": (
            float(top3_correct.sum() / answerable_count) if answerable_count else 0.0
        ),
        "mean_similarity_correct_top1": (
            float(correct_scores.mean()) if correct_scores.size else 0.0
        ),
        "answerable_acceptance_rate": (
            float(accepted[answerable].mean()) if answerable_count else 0.0
        ),
        "unanswerable_rejection_rate": (
            float((~accepted[unanswerable]).mean()) if unanswerable_count else 0.0
        ),
        "false_acceptance_count": int(accepted[unanswerable].sum()),
        "false_rejection_count": int((~accepted[answerable]).sum()),
        "incorrect_retrieval_examples": incorrect_examples,
    }
