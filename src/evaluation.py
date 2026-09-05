"""Validation-only selection and evaluation with frozen configurations."""

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
    Output: ranked ids/scores, a has-tokens mask, and a has-features mask.

    Ids in rows without features are sorting placeholders, not real matches.
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
    has_features = np.asarray(query_matrix.getnnz(axis=1)).ravel() > 0
    similarities = cosine_similarity(query_matrix, faq_matrix)

    width = min(top_k, len(faq_data))
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
    faq_ids = faq_data["id"].to_numpy()

    return {
        "ranked_ids": faq_ids[order],
        "ranked_scores": np.take_along_axis(similarities, order, axis=1),
        "has_tokens": has_tokens,
        "has_features": has_features,
    }


def _accepted(ranking: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    """Decide which queries clear the similarity threshold."""

    top_scores = ranking["ranked_scores"][:, 0]
    return ranking["has_features"] & (top_scores >= threshold)


def _split_masks(query_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a query table into answerable and unanswerable masks."""

    answerable = query_data["is_answerable"].to_numpy(dtype=bool)
    return answerable, ~answerable


def balanced_accept_reject_score(
    query_data: pd.DataFrame,
    ranking: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    """Score answerability decisions, independently of retrieval correctness.

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


def select_preprocessing(
    validation_data: pd.DataFrame,
    faq_data: pd.DataFrame,
    configs: list[tuple[str, dict[str, bool]]] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Step A: choose the representation using answerable validation queries.

    Prefer Top-1 accuracy, then Top-3 accuracy, then the candidate listed first.
    No similarity threshold or unanswerable query affects this comparison.
    At least three ranks are computed so Top-3 keeps its stated meaning.
    """

    # An explicitly empty list is an error, so it must not fall back silently.
    candidates = PREPROCESSING_CONFIGS if configs is None else configs
    if not candidates:
        raise ValueError("At least one preprocessing configuration is required")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    answerable, _ = _split_masks(validation_data)
    queries = validation_data.loc[answerable]
    if queries.empty:
        raise ValueError("Preprocessing selection requires answerable validation queries")
    expected = queries["expected_faq_id"].to_numpy(dtype="int64")
    comparison: list[dict[str, Any]] = []

    for config_rank, (name, options) in enumerate(candidates):
        vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
        ranking = rank_queries(
            queries["query"], faq_data, vectorizer, faq_matrix, options, max(3, top_k)
        )
        ids = ranking["ranked_ids"]
        valid = ranking["has_features"]
        comparison.append(
            {
                "config": name,
                "config_rank": config_rank,
                "top1_accuracy": float((valid & (ids[:, 0] == expected)).mean()),
                "top3_accuracy": float(
                    (valid & (ids[:, :3] == expected[:, None]).any(axis=1)).mean()
                ),
            }
        )

    best = max(
        comparison,
        key=lambda row: (row["top1_accuracy"], row["top3_accuracy"], -row["config_rank"]),
    )
    return {
        "selected_config": best["config"],
        "selected_options": dict(candidates[best["config_rank"]][1]),
        "preprocessing_comparison": comparison,
    }


def tune_ranked_threshold(validation_data: pd.DataFrame, ranking: dict) -> dict[str, Any]:
    """Sweep a cached ranking from any representation using Phase 1's policy."""

    answerable, unanswerable = _split_masks(validation_data)
    if not answerable.any() or not unanswerable.any():
        raise ValueError("Threshold tuning requires answerable and unanswerable validation queries")
    sweep = [balanced_accept_reject_score(validation_data, ranking, t) for t in THRESHOLD_STEPS]
    best = max(sweep, key=lambda row: (row["score"], row["threshold"]))
    return {"selected_threshold": best["threshold"], "selected_score": best["score"], "sweep": sweep}


def tune_threshold(
    validation_data: pd.DataFrame,
    faq_data: pd.DataFrame,
    configs: list[tuple[str, dict[str, bool]]] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Choose preprocessing (Step A), then its rejection threshold (Step B).

    Only validation queries are used. The threshold maximizes balanced
    answerable acceptance/unanswerable rejection, with ties going higher.
    """

    answerable, unanswerable = _split_masks(validation_data)
    if not answerable.any() or not unanswerable.any():
        raise ValueError(
            "Threshold tuning requires answerable and unanswerable validation queries"
        )

    selection = select_preprocessing(validation_data, faq_data, configs, top_k)
    options = selection["selected_options"]
    vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
    ranking = rank_queries(
        validation_data["query"], faq_data, vectorizer, faq_matrix, options, top_k
    )
    chosen = next(
        row for row in selection["preprocessing_comparison"]
        if row["config"] == selection["selected_config"]
    )
    threshold_result = tune_ranked_threshold(validation_data, ranking)
    sweep = [
        {
            "config": selection["selected_config"],
            "config_rank": chosen["config_rank"],
            **row,
        }
        for row in threshold_result["sweep"]
    ]
    best = max(sweep, key=lambda row: (row["score"], row["threshold"]))
    return {
        **selection,
        "selected_threshold": best["threshold"],
        "selected_score": best["score"],
        # Retained for callers: Step B now has only one configuration.
        "per_config_best": [best],
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
    """Report retrieval and answer-delivery metrics with a frozen configuration.

    Input: query table, FAQ table, fitted index, frozen threshold and switches.
    Output: the metric dictionary required by the project plan.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ranking = rank_queries(
        test_data["query"], faq_data, vectorizer, faq_matrix, preprocessing_options,
        max(3, top_k),
    )
    return evaluate_rankings(
        test_data, faq_data, ranking, threshold, preprocessing_options, max_examples
    )


def evaluate_rankings(
    test_data: pd.DataFrame,
    faq_data: pd.DataFrame,
    ranking: dict[str, np.ndarray],
    threshold: float,
    preprocessing_options: dict[str, bool] | None = None,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Apply identical retrieval and delivery metric definitions to any ranking."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    answerable, unanswerable = _split_masks(test_data)
    accepted = _accepted(ranking, threshold)

    ranked_ids = ranking["ranked_ids"]
    top_scores = ranking["ranked_scores"][:, 0]
    # Unanswerable rows hold a blank expected id, so NaN keeps them unmatched.
    expected = pd.to_numeric(test_data["expected_faq_id"], errors="coerce").to_numpy(
        dtype="float64"
    )

    has_features = ranking["has_features"]
    top1_correct = answerable & has_features & (ranked_ids[:, 0] == expected)
    top3_correct = (
        answerable & has_features & (ranked_ids[:, :3] == expected[:, None]).any(axis=1)
    )

    answerable_count = int(answerable.sum())
    unanswerable_count = int(unanswerable.sum())
    questions_by_id = faq_data.set_index("id")["question"]

    correct_scores = top_scores[top1_correct]
    incorrect_examples = [
        {
            "query": str(test_data.iloc[position]["query"]),
            "expected_faq_id": int(expected[position]),
            "retrieved_faq_id": (
                int(ranked_ids[position, 0]) if has_features[position] else None
            ),
            "retrieved_question": (
                str(questions_by_id.loc[int(ranked_ids[position, 0])])
                if has_features[position] else None
            ),
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
        "correct_answer_rate": (
            float((top1_correct & accepted).sum() / answerable_count)
            if answerable_count else 0.0
        ),
        "accepted_wrong_count": int((answerable & accepted & ~top1_correct).sum()),
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


def prediction_frame(
    test_data: pd.DataFrame,
    ranking: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """Expose the per-query decisions behind the aggregate metrics.

    Input: query table, cached ranking, and that model's frozen threshold.
    Output: one row per query, using the masks `evaluate_rankings` applies.

    A blank predicted id means the query produced no usable vector, so its
    similarity is a sorting placeholder rather than a real match.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    answerable, _ = _split_masks(test_data)
    has_features = ranking["has_features"]
    accepted = _accepted(ranking, threshold)
    ranked_ids = ranking["ranked_ids"]
    expected = pd.to_numeric(test_data["expected_faq_id"], errors="coerce").to_numpy(
        dtype="float64"
    )
    top1_correct = answerable & has_features & (ranked_ids[:, 0] == expected)

    predicted = pd.array(ranked_ids[:, 0], dtype="Int64")
    predicted[~has_features] = pd.NA
    return pd.DataFrame(
        {
            "predicted_faq_id": predicted,
            "similarity": ranking["ranked_scores"][:, 0],
            "accepted": accepted,
            "top1_correct": top1_correct,
            "correct_answer": top1_correct & accepted,
        },
        index=test_data.index,
    )
