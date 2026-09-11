"""Retrieval model keys, labels, families, and the comparison groups they form."""

from __future__ import annotations

# Ordered as the models appear in every comparison table.
MODEL_LABELS = {
    "tfidf": "TF-IDF",
    "w2v_mean": "Word2Vec mean",
    "w2v_tfidf": "Word2Vec TF-IDF weighted",
    "pw2v_mean": "Pretrained Word2Vec mean",
    "pw2v_tfidf": "Pretrained Word2Vec TF-IDF weighted",
}

# Longer menu wording for the terminal demonstration.
MODEL_DESCRIPTIONS = {
    "tfidf": "TF-IDF, the Phase 1 baseline",
    "w2v_mean": "Custom Word2Vec, mean vectors",
    "w2v_tfidf": "Custom Word2Vec, TF-IDF weighted vectors",
    "pw2v_mean": "Pretrained Google News Word2Vec, mean vectors",
    "pw2v_tfidf": "Pretrained Google News Word2Vec, TF-IDF weighted vectors",
}

# Families in the order the project introduced them.
FAMILY_ORDER = ["tfidf", "custom_w2v", "pretrained_w2v"]
MODEL_FAMILIES = {
    "tfidf": "tfidf",
    "w2v_mean": "custom_w2v",
    "w2v_tfidf": "custom_w2v",
    "pw2v_mean": "pretrained_w2v",
    "pw2v_tfidf": "pretrained_w2v",
}

# Dense models share the Phase 2 sentence-vector code; only the vectors differ.
AGGREGATIONS = {
    "w2v_mean": "w2v_mean",
    "w2v_tfidf": "w2v_tfidf",
    "pw2v_mean": "w2v_mean",
    "pw2v_tfidf": "w2v_tfidf",
}

# The custom model that combines word vectors the same way.
CUSTOM_COUNTERPARTS = {"pw2v_mean": "w2v_mean", "pw2v_tfidf": "w2v_tfidf"}

# `all` keeps its Phase 2 meaning, so the frozen Phase 2 reports reproduce.
MODEL_GROUPS = {
    "all": ["tfidf", "w2v_mean", "w2v_tfidf"],
    "phase2b": ["tfidf", "w2v_mean", "w2v_tfidf", "pw2v_mean", "pw2v_tfidf"],
}

# Reports containing a family's models go to that phase's directory.
REPORT_SUBDIRS = {"custom_w2v": "phase2", "pretrained_w2v": "phase2b"}


def newest_family(models: list[str]) -> str:
    """Return the most recently introduced family among the given models."""

    return max((MODEL_FAMILIES[item] for item in models), key=FAMILY_ORDER.index)


def report_subdir(models: list[str]) -> str | None:
    """Name the phase directory for these models; None means TF-IDF only."""

    return REPORT_SUBDIRS.get(newest_family(models))
