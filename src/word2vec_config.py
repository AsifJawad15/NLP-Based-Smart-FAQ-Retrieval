"""Word2Vec thresholds, stored apart from the frozen TF-IDF configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AGGREGATIONS = ("w2v_mean", "w2v_tfidf")
CONFIG_NAME = "word2vec_config.json"


def config_path(corpus_dir: str | Path) -> Path:
    return Path(corpus_dir) / CONFIG_NAME


def _tuning_instruction(corpus: str) -> str:
    return f"Run: python evaluate.py tune --corpus {corpus} --model all"


def save_word2vec_config(
    corpus_dir: str | Path, artifact_id: str, thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Freeze tuned thresholds beside the corpus, keyed to one model artifact.

    Thresholds tuned against a different artifact are dropped rather than
    merged, because a retrained model invalidates its previous sweep.
    """

    unknown = set(thresholds) - set(AGGREGATIONS)
    if unknown:
        raise ValueError(f"Unknown Word2Vec aggregations: {sorted(unknown)}")

    path = config_path(corpus_dir)
    config: dict[str, Any] = {"artifact_id": artifact_id, "preprocessing_config": "basic"}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("artifact_id") == artifact_id:
            config.update({k: v for k, v in existing.items() if k in AGGREGATIONS})
    for name, values in thresholds.items():
        threshold = float(values["similarity_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        config[name] = {
            "similarity_threshold": threshold,
            "validation_score": float(values["validation_score"]),
        }
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def load_word2vec_threshold(
    corpus_dir: str | Path, metadata: dict[str, Any], aggregation: str,
) -> float:
    """Read one frozen threshold, refusing a sweep from a different model."""

    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown Word2Vec aggregation: {aggregation}")
    directory = Path(corpus_dir)
    path = config_path(directory)
    instruction = _tuning_instruction(directory.name)
    if not path.is_file():
        raise FileNotFoundError(f"Word2Vec thresholds missing for {directory.name}. {instruction}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        artifact_id = config["artifact_id"]
        threshold = float(config[aggregation]["similarity_threshold"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid or incomplete Word2Vec thresholds for {directory.name} "
            f"({aggregation}). {instruction}"
        ) from error
    if artifact_id != metadata["artifact_id"]:
        raise ValueError(
            f"Word2Vec thresholds for {directory.name} were tuned for a different "
            f"model artifact. {instruction}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    return threshold
