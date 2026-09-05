"""Tune thresholds on validation data, then evaluate the frozen configuration.

Examples:
    python evaluate.py tune
    python evaluate.py test
    python evaluate.py all --corpus university
    python evaluate.py all --model all
    python evaluate.py manual --corpus ecommerce --model all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import (
    discover_corpora,
    load_corpus_config,
    load_faq_dataset,
    load_query_dataset,
)
from src.evaluation import (
    evaluate_rankings,
    prediction_frame,
    rank_queries,
    tune_ranked_threshold,
    tune_threshold,
)
from src.tfidf_retrieval import build_tfidf_index
from src.word2vec_config import load_word2vec_threshold, save_word2vec_config
from src.word2vec_retrieval import build_word2vec_index, rank_word2vec_queries
from src.word2vec_training import BASIC_OPTIONS, load_word2vec


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_ROOT = BASE_DIR / "models"

MODELS = ["tfidf", "w2v_mean", "w2v_tfidf"]
MODEL_LABELS = {
    "tfidf": "TF-IDF",
    "w2v_mean": "Word2Vec mean",
    "w2v_tfidf": "Word2Vec TF-IDF weighted",
}


def phase2_dir() -> Path:
    """Resolve at call time so tests can redirect REPORTS_DIR."""

    return REPORTS_DIR / "phase2"


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


def selected_models(model: str) -> list[str]:
    """Return the retrieval models to run, in a fixed comparison order."""

    if model == "all":
        return list(MODELS)
    if model not in MODELS:
        raise SystemExit(f"Unknown model '{model}'; choose from {MODELS + ['all']}")
    return [model]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Save one report as readable, reproducible JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def tune_tfidf(name: str, directory: Path, faq_data, validation) -> None:
    """Select TF-IDF preprocessing and threshold from validation queries only."""

    result = tune_threshold(validation, faq_data)

    print(f"\n=== Tuning {name} TF-IDF on {len(validation)} validation queries ===")
    print("  Step A: answerable validation retrieval")
    for row in result["preprocessing_comparison"]:
        print(
            f"  {row['config']:<18} Top-1={row['top1_accuracy']:.3f} "
            f"Top-3={row['top3_accuracy']:.3f}"
        )
    print("  Step B: threshold for the selected preprocessing")
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
            "preprocessing_comparison": result["preprocessing_comparison"],
            "per_config_best": result["per_config_best"],
            "sweep": result["sweep"],
        },
    )


def tune_word2vec(
    name: str, directory: Path, faq_data, validation, aggregations: list[str]
) -> dict[str, Any]:
    """Sweep one threshold per aggregation; preprocessing is fixed to basic."""

    model, metadata = load_word2vec(directory, MODELS_ROOT)
    thresholds: dict[str, dict[str, float]] = {}
    print(f"\n=== Tuning {name} Word2Vec on {len(validation)} validation queries ===")
    print(
        f"  model artifact {metadata['artifact_id'][:12]}, "
        f"vocabulary {metadata['vocabulary_size']}, "
        f"{metadata['token_count']} training tokens"
    )
    for aggregation in aggregations:
        index = build_word2vec_index(faq_data, model.wv, aggregation)
        ranking = rank_word2vec_queries(validation["query"], faq_data, index, top_k=3)
        result = tune_ranked_threshold(validation, ranking)
        best = next(
            row for row in result["sweep"]
            if row["threshold"] == result["selected_threshold"]
        )
        thresholds[aggregation] = {
            "similarity_threshold": result["selected_threshold"],
            "validation_score": result["selected_score"],
        }
        print(
            f"  {MODEL_LABELS[aggregation]:<26} threshold={best['threshold']:.2f} "
            f"score={best['score']:.4f} "
            f"accept={best['answerable_acceptance_rate']:.3f} "
            f"reject={best['unanswerable_rejection_rate']:.3f}"
        )
        write_json(
            phase2_dir() / f"{name}_{aggregation}_tuning.json",
            {
                "corpus": name,
                "model": aggregation,
                "artifact_id": metadata["artifact_id"],
                "preprocessing_config": "basic",
                "validation_queries": len(validation),
                "usable_faq_vectors": int(index["valid_faqs"].sum()),
                "selected_threshold": result["selected_threshold"],
                "selected_score": result["selected_score"],
                "sweep": result["sweep"],
            },
        )
    saved = save_word2vec_config(directory, metadata["artifact_id"], thresholds)
    print(f"  frozen into {directory / 'word2vec_config.json'}")
    return saved


def run_tuning(corpus: str, model: str = "tfidf") -> None:
    """Tune only the requested models, using validation queries only."""

    models = selected_models(model)
    if model == "all":
        # The Phase 2 comparison reuses the frozen TF-IDF baseline unchanged.
        models = [item for item in models if item != "tfidf"]
        print("Reusing frozen TF-IDF settings; tuning Word2Vec thresholds only.")

    for name, directory in selected_corpora(corpus).items():
        faq_data = load_faq_dataset(directory)
        validation = load_query_dataset(
            directory / "validation_queries.csv", set(faq_data["id"])
        )
        if "tfidf" in models:
            tune_tfidf(name, directory, faq_data, validation)
        aggregations = [item for item in models if item != "tfidf"]
        if aggregations:
            tune_word2vec(name, directory, faq_data, validation, aggregations)


def _tfidf_ranking(faq_data, config, queries):
    """Rank queries with the frozen TF-IDF configuration."""

    options = {
        "remove_stopwords": bool(config["remove_stopwords"]),
        "lemmatize": bool(config["lemmatize"]),
    }
    vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
    ranking = rank_queries(queries, faq_data, vectorizer, faq_matrix, options, 3)
    return ranking, options, float(config["similarity_threshold"])


def print_report(title: str, report: dict[str, Any]) -> None:
    """Print one evaluated model exactly as Phase 1 printed TF-IDF."""

    print(f"\n=== {title} ===")
    print(f"  FAQs indexed              : {report['faq_count']}")
    print(f"  Preprocessing             : {report['preprocessing_config']}")
    print(f"  Frozen threshold          : {report['threshold']:.2f}")
    print(f"  Top-1 accuracy            : {report['top1_accuracy']:.3f}")
    print(f"  Top-3 accuracy            : {report['top3_accuracy']:.3f}")
    print(f"  Correct answer rate       : {report['correct_answer_rate']:.3f}")
    print(f"  Accepted but wrong        : {report['accepted_wrong_count']}")
    print(f"  Mean similarity (correct) : {report['mean_similarity_correct_top1']:.4f}")
    print(f"  Answerable acceptance     : {report['answerable_acceptance_rate']:.3f}")
    print(f"  Unanswerable rejection    : {report['unanswerable_rejection_rate']:.3f}")
    print(f"  False acceptances         : {report['false_acceptance_count']}")
    print(f"  False rejections          : {report['false_rejection_count']}")


def run_testing(
    corpus: str, model: str = "tfidf", *, manual: bool = False
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate every requested model on identical query rows and labels.

    Header-only manual templates are pending work, never zero-score results.
    Manual reports use separate filenames and never trigger tuning.
    """

    models = selected_models(model)
    baseline_only = models == ["tfidf"]
    destination = REPORTS_DIR if baseline_only else phase2_dir()
    prefix = "manual_" if manual else ""

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    pending: list[str] = []

    for name, directory in selected_corpora(corpus).items():
        config = load_corpus_config(directory)
        if "preprocessing_config" not in config:
            raise SystemExit(f"Run 'python evaluate.py tune' before testing {name}")

        faq_data = load_faq_dataset(directory)
        query_path = (
            DATA_ROOT / "manual_evaluation" / f"{name}_queries.csv"
            if manual else directory / "test_queries.csv"
        )
        test = load_query_dataset(query_path, set(faq_data["id"]), allow_empty=manual)
        if test.empty:
            pending.append(name)
            print(
                f"{name}: human evaluation pending - {query_path} has headers only; "
                "no current performance report written."
            )
            continue

        # Load the trained model once; both aggregations reuse it.
        word2vec = None
        if any(item != "tfidf" for item in models):
            word2vec = load_word2vec(directory, MODELS_ROOT)

        reports[name] = {}
        columns: dict[str, Any] = {
            "query": test["query"],
            "is_answerable": test["is_answerable"],
            "expected_faq_id": test["expected_faq_id"],
        }
        for item in models:
            if item == "tfidf":
                ranking, options, threshold = _tfidf_ranking(faq_data, config, test["query"])
                preprocessing = str(config["preprocessing_config"])
            else:
                trained, metadata = word2vec
                threshold = load_word2vec_threshold(directory, metadata, item)
                index = build_word2vec_index(faq_data, trained.wv, item)
                ranking = rank_word2vec_queries(test["query"], faq_data, index, top_k=3)
                options = dict(BASIC_OPTIONS)
                preprocessing = "basic"

            report = evaluate_rankings(test, faq_data, ranking, threshold, options)
            report["corpus"] = name
            report["model"] = item
            report["model_label"] = MODEL_LABELS[item]
            report["display_name"] = str(config["display_name"])
            report["preprocessing_config"] = preprocessing
            report["faq_count"] = len(faq_data)
            report["evaluation_set"] = "manual" if manual else "synthetic_benchmark"
            reports[name][item] = report

            label = "manual evaluation" if manual else "synthetic benchmark"
            print_report(
                f"{config['display_name']} {MODEL_LABELS[item]} {label} results", report
            )

            filename = (
                f"{prefix}{name}_evaluation.json" if baseline_only
                else f"{prefix}{name}_{item}_evaluation.json"
            )
            write_json(destination / filename, report)

            frame = prediction_frame(test, ranking, threshold)
            for column in frame.columns:
                columns[f"{item}_{column}"] = frame[column]

        if len(models) > 1:
            paired = pd.DataFrame(columns)
            predictions[name] = paired
            path = phase2_dir() / f"{prefix}{name}_model_comparison.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            paired.to_csv(path, index=False)
            print(f"  Saved {path}")

    if baseline_only:
        write_markdown_report(
            {key: value["tfidf"] for key, value in reports.items()},
            manual=manual,
            pending=pending,
        )
    else:
        write_comparison_report(reports, predictions, models, manual=manual, pending=pending)
    return reports


METRIC_ROWS = [
    ("FAQs indexed", "faq_count", "{0}"),
    ("Preprocessing", "preprocessing_config", "{0}"),
    ("Threshold", "threshold", "{0:.2f}"),
    ("Answerable queries", "answerable_queries", "{0}"),
    ("Unanswerable queries", "unanswerable_queries", "{0}"),
    ("Top-1 accuracy", "top1_accuracy", "{0:.3f}"),
    ("Top-3 accuracy", "top3_accuracy", "{0:.3f}"),
    ("Correct answer rate", "correct_answer_rate", "{0:.3f}"),
    ("Accepted but wrong", "accepted_wrong_count", "{0}"),
    ("Mean similarity of correct Top-1", "mean_similarity_correct_top1", "{0:.4f}"),
    ("Answerable acceptance rate", "answerable_acceptance_rate", "{0:.3f}"),
    ("Unanswerable rejection rate", "unanswerable_rejection_rate", "{0:.3f}"),
    ("False acceptances", "false_acceptance_count", "{0}"),
    ("False rejections", "false_rejection_count", "{0}"),
]


def write_markdown_report(
    reports: dict[str, dict[str, Any]],
    *,
    manual: bool = False,
    pending: list[str] | None = None,
) -> None:
    """Write one shared Markdown summary of every evaluated corpus."""

    if not reports:
        return

    names = sorted(reports)
    lines = [
        "# Manual TF-IDF Evaluation" if manual else "# Cross-Domain TF-IDF Synthetic Benchmark",
        "",
        "Thresholds and preprocessing were selected on validation queries only,",
        "frozen into `corpus_config.json`, and applied here without retuning.",
        "",
        (
            "These queries are supplied separately by the project team. "
            "The evaluation command cannot verify human authorship."
            if manual else
            "The synthetic query sets have already been inspected during development "
            "and review; these are reproducible benchmark results, not an untouched final assessment."
        ),
        "",
        "Correct answer rate = count(correct Top-1 AND accepted) / all answerable queries. "
        "Accepted but wrong counts answerable queries answered with a different FAQ.",
        "",
        "| Metric | " + " | ".join(reports[name]["display_name"] for name in names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for label, key, template in METRIC_ROWS:
        values = " | ".join(template.format(reports[name][key]) for name in names)
        lines.append(f"| {label} | {values} |")

    if pending:
        lines += ["", "Human evaluation pending (no scores): " + ", ".join(pending) + "."]

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
    path = REPORTS_DIR / ("manual_evaluation_report.md" if manual else "evaluation_report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


def paired_cases(paired: pd.DataFrame, item: str, *, improved: bool) -> pd.DataFrame:
    """Rows where exactly one of TF-IDF and this model delivers the answer."""

    baseline = paired["tfidf_correct_answer"].to_numpy(dtype=bool)
    other = paired[f"{item}_correct_answer"].to_numpy(dtype=bool)
    return paired.loc[(other & ~baseline) if improved else (baseline & ~other)]


def _case_lines(paired: pd.DataFrame, item: str, *, improved: bool, limit: int = 5) -> list[str]:
    """Describe real paired rows, or state plainly that there are none."""

    cases = paired_cases(paired, item, improved=improved)
    if cases.empty:
        return ["No queries fall in this category."]
    winner, loser = (item, "tfidf") if improved else ("tfidf", item)
    counted = f"{len(cases)} quer{'y' if len(cases) == 1 else 'ies'}"
    lines = [
        counted if len(cases) <= limit else f"{counted}; the first {limit} are shown.",
        "",
    ]
    for _, row in cases.head(limit).iterrows():
        lines += [
            f"- Query: {row['query']}",
            f"  - Expected FAQ {row['expected_faq_id']}; {MODEL_LABELS[winner]} "
            f"returned it at similarity {row[f'{winner}_similarity']:.4f}.",
            f"  - {MODEL_LABELS[loser]} predicted FAQ "
            f"{row[f'{loser}_predicted_faq_id']} at similarity "
            f"{row[f'{loser}_similarity']:.4f} "
            f"(accepted={bool(row[f'{loser}_accepted'])}).",
        ]
    return lines


def write_comparison_report(
    reports: dict[str, dict[str, dict[str, Any]]],
    predictions: dict[str, pd.DataFrame],
    models: list[str],
    *,
    manual: bool = False,
    pending: list[str] | None = None,
) -> None:
    """Write the Phase 2 comparison, leaving the Phase 1 reports intact."""

    if not reports:
        return

    lines = [
        f"# Phase 2 Model Comparison ({'Manual' if manual else 'Synthetic Benchmark'})",
        "",
        "TF-IDF, mean Word2Vec, and TF-IDF-weighted Word2Vec were evaluated on the",
        "identical query rows and labels. Each model applies its own threshold,",
        "tuned on validation queries only. TF-IDF reuses its frozen Phase 1",
        "configuration and was not retuned for this comparison.",
        "",
        "Both Word2Vec models are trained on that domain's FAQ questions alone, use",
        "basic preprocessing, and ignore word order. Their cosine similarities",
        "occupy a different range than sparse TF-IDF cosines, which is why each",
        "model needs its own threshold instead of a shared one.",
        "",
    ]

    for name in sorted(reports):
        corpus_reports = reports[name]
        present = [item for item in models if item in corpus_reports]
        lines += [
            f"## {corpus_reports[present[0]]['display_name']}",
            "",
            "| Metric | " + " | ".join(MODEL_LABELS[item] for item in present) + " |",
            "| --- | " + " | ".join("---" for _ in present) + " |",
        ]
        for row_label, key, template in METRIC_ROWS:
            values = " | ".join(template.format(corpus_reports[item][key]) for item in present)
            lines.append(f"| {row_label} | {values} |")
        lines.append("")

        paired = predictions.get(name)
        if paired is None or "tfidf" not in present:
            continue
        for item in present:
            if item == "tfidf":
                continue
            lines += [
                f"### {MODEL_LABELS[item]}: improvements over TF-IDF",
                "",
                "Queries where this model delivers the correct answer and TF-IDF does not.",
                "",
                *_case_lines(paired, item, improved=True),
                "",
                f"### {MODEL_LABELS[item]}: reverse cases",
                "",
                "Queries where TF-IDF delivers the correct answer and this model does not.",
                "",
                *_case_lines(paired, item, improved=False),
                "",
            ]

    if pending:
        lines += ["Human evaluation pending (no scores): " + ", ".join(pending) + ".", ""]

    directory = phase2_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("manual_comparison_report.md" if manual else "comparison_report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


def parse_args() -> argparse.Namespace:
    """Parse the requested evaluation mode, corpus, and retrieval model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tune", "test", "all", "manual"])
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--model", default="tfidf", choices=MODELS + ["all"])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        if arguments.mode in {"tune", "all"}:
            run_tuning(arguments.corpus, arguments.model)
        if arguments.mode in {"test", "all", "manual"}:
            run_testing(arguments.corpus, arguments.model, manual=arguments.mode == "manual")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
