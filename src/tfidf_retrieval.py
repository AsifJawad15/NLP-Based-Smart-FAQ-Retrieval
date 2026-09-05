"""TF-IDF indexing, cosine ranking, and threshold-based answering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


def build_tfidf_index(
    faq_data: pd.DataFrame,
    preprocessing_options: dict[str, bool] | None = None,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit one TF-IDF vectorizer on the stored FAQ questions.

    Input: validated FAQ rows and preprocessing switches.
    Output: fitted vectorizer and sparse FAQ matrix.
    """

    options = preprocessing_options or {}
    processed_questions = [
        preprocess_text(question, **options) for question in faq_data["question"]
    ]
    if not any(processed_questions):
        raise ValueError("FAQ questions are empty after preprocessing")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        norm="l2",
    )
    faq_matrix = vectorizer.fit_transform(processed_questions).tocsr()
    return vectorizer, faq_matrix


def retrieve_tfidf(
    query: str,
    faq_data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    faq_matrix: csr_matrix,
    top_k: int = 3,
    preprocessing_options: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Rank FAQs by cosine similarity to one user query.

    Input: query, FAQ table, fitted index, top-k, and preprocessing switches.
    Output: descending list of match dictionaries.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if faq_matrix.shape[0] != len(faq_data):
        raise ValueError("FAQ matrix row count does not match FAQ data")

    options = preprocessing_options or {}
    processed_query = preprocess_text(query, **options)
    if not processed_query:
        return []

    query_vector = vectorizer.transform([processed_query])
    if query_vector.nnz == 0:
        return []
    similarities = cosine_similarity(query_vector, faq_matrix).ravel()
    ranked_indices = np.argsort(-similarities, kind="stable")[: min(top_k, len(faq_data))]

    matches: list[dict[str, Any]] = []
    for index in ranked_indices:
        row = faq_data.iloc[int(index)]
        matches.append(
            {
                "faq_id": int(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "category": str(row["category"]),
                "source": str(row["source"]),
                "similarity": float(similarities[index]),
            }
        )
    return matches


def answer_query(
    query: str,
    faq_data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    faq_matrix: csr_matrix,
    threshold: float = 0.30,
    top_k: int = 3,
    preprocessing_options: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Apply a similarity threshold to ranked TF-IDF matches."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    matches = retrieve_tfidf(
        query,
        faq_data,
        vectorizer,
        faq_matrix,
        top_k=top_k,
        preprocessing_options=preprocessing_options,
    )
    accepted = bool(matches and matches[0]["similarity"] >= threshold)
    return {
        "found": accepted,
        "message": (
            "Relevant FAQ found."
            if accepted
            else "Sorry, I could not find a sufficiently relevant FAQ."
        ),
        "best_match": matches[0] if accepted else None,
        "top_matches": matches,
        "threshold": threshold,
    }
