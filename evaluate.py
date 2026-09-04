"""Tune thresholds on validation data, then evaluate the frozen configuration.

Examples:
    python evaluate.py tune
    python evaluate.py test
    python evaluate.py all --corpus university
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data_loader import (
    discover_corpora,
    load_corpus_config,
    load_faq_dataset,
    load_query_dataset,
)
from src.evaluation import PREPROCESSING_CONFIGS, evaluate_tfidf, tune_threshold
from src.tfidf_retrieval import build_tfidf_index


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


def selected_corpora(corpus: str) -> dict[str, Path]:
    """Return every discovered corpus, or only the requested one."""

    corpora = discover_corpora(DATA_ROOT)
    if not corpora:
        raise SystemExit(f"No corpora found under {DATA_ROOT}")
    if corpus == "all":
        return corpora
    if corpus not in corpora:
        raise SystemExit(f"Unknown corpus '{corpus}'; found {sorted(corpora)}")
    return {corpus: corpora[corpus]}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Save one report as readable, reproducible JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def run_tuning(corpus: str) -> None:
    """Select preprocessing and threshold from validation queries only."""

    for name, directory in selected_corpora(corpus).items():
        faq_data = load_faq_dataset(directory)
        validation = load_query_dataset(
            directory / "validation_queries.csv", set(faq_data["id"])
        )
        result = tune_threshold(validation, faq_data)

        print(f"\n=== Tuning {name} on {len(validation)} validation queries ===")
        for row in result["per_config_best"]:
            print(
                f"  {row['config']:<18} threshold={row['threshold']:.2f} "
                f"score={row['score']:.4f} "
                f"accept={row['answerable_acceptance_rate']:.3f} "
                f"reject={row['unanswerable_rejection_rate']:.3f}"
            )
        print(
            f"  selected: {result['selected_config']} at "
            f"{result['selected_threshold']:.2f} (score {result['selected_score']:.4f})"
        )

        # Freeze the validation-selected values before the test set is touched.
        config = load_corpus_config(directory)
        config.update(result["selected_options"])
        config["similarity_threshold"] = result["selected_threshold"]
        config["preprocessing_config"] = result["selected_config"]
        config["validation_score"] = result["selected_score"]
        write_json(directory / "corpus_config.json", config)

        write_json(
            REPORTS_DIR / f"{name}_tuning.json",
            {
                "corpus": name,
                "validation_queries": len(validation),
                "selected_config": result["selected_config"],
                "selected_threshold": result["selected_threshold"],
                "selected_score": result["selected_score"],
                "per_config_best": result["per_config_best"],
                "sweep": result["sweep"],
            },
        )


def run_testing(corpus: str) -> dict[str, dict[str, Any]]:
    """Evaluate each corpus once using its frozen configuration."""

    reports: dict[str, dict[str, Any]] = {}
    for name, directory in selected_corpora(corpus).items():
        config = load_corpus_config(directory)
        if "preprocessing_config" not in config:
            raise SystemExit(f"Run 'python evaluate.py tune' before testing {name}")

        faq_data = load_faq_dataset(directory)
        test = load_query_dataset(directory / "test_queries.csv", set(faq_data["id"]))
        options = {
            "remove_stopwords": bool(config["remove_stopwords"]),
            "lemmatize": bool(config["lemmatize"]),
        }
        vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
        report = evaluate_tfidf(
            test,
            faq_data,
            vectorizer,
            faq_matrix,
            threshold=float(config["similarity_threshold"]),
            preprocessing_options=options,
        )
        report["corpus"] = name
        report["display_name"] = str(config["display_name"])
        report["preprocessing_config"] = str(config["preprocessing_config"])
        report["faq_count"] = len(faq_data)
        reports[name] = report

        print(f"\n=== {config['display_name']} test results ===")
        print(f"  FAQs indexed              : {len(faq_data)}")
        print(f"  Preprocessing             : {config['preprocessing_config']}")
        print(f"  Frozen threshold          : {report['threshold']:.2f}")
        print(f"  Top-1 accuracy            : {report['top1_accuracy']:.3f}")
        print(f"  Top-3 accuracy            : {report['top3_accuracy']:.3f}")
        print(f"  Mean similarity (correct) : {report['mean_similarity_correct_top1']:.4f}")
        print(f"  Answerable acceptance     : {report['answerable_acceptance_rate']:.3f}")
        print(f"  Unanswerable rejection    : {report['unanswerable_rejection_rate']:.3f}")
        print(f"  False acceptances         : {report['false_acceptance_count']}")
        print(f"  False rejections          : {report['false_rejection_count']}")

        write_json(REPORTS_DIR / f"{name}_evaluation.json", report)

    write_markdown_report(reports)
    return reports


def write_markdown_report(reports: dict[str, dict[str, Any]]) -> None:
    """Write one shared Markdown summary of every evaluated corpus."""

    if not reports:
        return

    rows = [
        ("FAQs indexed", "faq_count", "{0}"),
        ("Preprocessing", "preprocessing_config", "{0}"),
        ("Threshold", "threshold", "{0:.2f}"),
        ("Top-1 accuracy", "top1_accuracy", "{0:.3f}"),
        ("Top-3 accuracy", "top3_accuracy", "{0:.3f}"),
        ("Mean similarity of correct Top-1", "mean_similarity_correct_top1", "{0:.4f}"),
        ("Answerable acceptance rate", "answerable_acceptance_rate", "{0:.3f}"),
        ("Unanswerable rejection rate", "unanswerable_rejection_rate", "{0:.3f}"),
        ("False acceptances", "false_acceptance_count", "{0}"),
        ("False rejections", "false_rejection_count", "{0}"),
    ]
    names = sorted(reports)
    lines = [
        "# Cross-Domain TF-IDF Evaluation",
        "",
        "Thresholds and preprocessing were selected on validation queries only,",
        "frozen into `corpus_config.json`, and then applied once to the test set.",
        "",
        "| Metric | " + " | ".join(reports[name]["display_name"] for name in names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for label, key, template in rows:
        values = " | ".join(template.format(reports[name][key]) for name in names)
        lines.append(f"| {label} | {values} |")

    for name in names:
        report = reports[name]
        lines += ["", f"## Incorrect retrievals: {report['display_name']}", ""]
        if not report["incorrect_retrieval_examples"]:
            lines.append("No incorrect Top-1 retrievals were recorded.")
            continue
        for example in report["incorrect_retrieval_examples"]:
            lines += [
                f"- Query: {example['query']}",
                f"  - Expected FAQ {example['expected_faq_id']}, "
                f"retrieved FAQ {example['retrieved_faq_id']} "
                f"(similarity {example['similarity']:.4f}, "
                f"accepted={example['accepted']})",
                f"  - Retrieved question: {example['retrieved_question']}",
            ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "evaluation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path.relative_to(BASE_DIR)}")


def parse_args() -> argparse.Namespace:
    """Parse the requested evaluation mode and corpus."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tune", "test", "all"])
    parser.add_argument("--corpus", default="all")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode in {"tune", "all"}:
        run_tuning(arguments.corpus)
    if arguments.mode in {"test", "all"}:
        run_testing(arguments.corpus)
