"""Pure helpers shared by the Streamlit interface and its tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data_loader import discover_corpora, load_corpus_config
from src.retrieval_models import MODEL_LABELS


def corpora_for_purpose(data_root: str | Path, purpose: str) -> dict[str, Path]:
    """Discover only corpora intended for one GUI page."""

    return {
        name: directory
        for name, directory in discover_corpora(data_root).items()
        if load_corpus_config(directory)["purpose"] == purpose
    }


def file_change_signature(
    corpus_dir: str | Path, model: str, models_root: str | Path
) -> str:
    """Fingerprint the files capable of changing a cached answerer."""

    directory = Path(corpus_dir)
    model_root = Path(models_root)
    candidates = [directory / "faq_dataset.csv", directory / "corpus_config.json"]
    if model.startswith("w2v_"):
        candidates += [
            directory / "word2vec_config.json",
            model_root / directory.name / "custom_word2vec.model",
            model_root / directory.name / "training_metadata.json",
        ]
    elif model.startswith("pw2v_"):
        candidates += [
            directory / "pretrained_config.json",
            model_root / "pretrained" / "word2vec-google-news-300-lower.keys.tsv",
            model_root / "pretrained" / "word2vec-google-news-300-lower.vectors.npy",
            model_root / "pretrained" / "pretrained_metadata.json",
        ]
    elif model in {"rnn", "bilstm"}:
        candidates += [
            directory / f"sequence_{model}_config.json",
            model_root / directory.name / f"{model}.pt",
            model_root / directory.name / f"{model}_metadata.json",
            model_root / "pretrained" / "word2vec-google-news-300-lower.keys.tsv",
            model_root / "pretrained" / "word2vec-google-news-300-lower.vectors.npy",
            model_root / "pretrained" / "pretrained_metadata.json",
        ]

    digest = hashlib.sha256()
    for path in candidates:
        digest.update(str(path.resolve()).encode("utf-8"))
        if path.is_file():
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def saved_benchmark_rows(
    reports_root: str | Path, corpus: str
) -> list[dict[str, Any]]:
    """Read frozen research metrics without running evaluation or training."""

    root = Path(reports_root)
    search_dirs = [root / "phase3", root / "phase2b", root / "phase2", root]
    rows: list[dict[str, Any]] = []
    for model, label in MODEL_LABELS.items():
        report_path = next(
            (directory / f"{corpus}_{model}_evaluation.json"
             for directory in search_dirs
             if (directory / f"{corpus}_{model}_evaluation.json").is_file()),
            None,
        )
        if report_path is None and model == "tfidf":
            fallback = root / f"{corpus}_evaluation.json"
            report_path = fallback if fallback.is_file() else None
        if report_path is None:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Model": label + (" (experimental)" if model in {"rnn", "bilstm"} else ""),
                "Top-1 accuracy": report.get("top1_accuracy"),
                "Correct answer rate": report.get("correct_answer_rate"),
                "Unanswerable rejection rate": report.get("unanswerable_rejection_rate"),
            }
        )
    return rows
