"""Model selection, frozen Word2Vec thresholds, and paired comparison outputs."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate
from src.data_loader import load_query_dataset, validate_faq_data
from src.evaluation import evaluate_rankings, prediction_frame, rank_queries
from src.tfidf_retrieval import build_tfidf_index
from src.word2vec_config import config_path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = "query,expected_faq_id,is_answerable\n"
MODELS = ["tfidf", "w2v_mean", "w2v_tfidf"]


class PredictionFrameTests(unittest.TestCase):
    """The paired CSV must never disagree with the reported aggregates."""

    def setUp(self) -> None:
        self.faq = validate_faq_data(pd.read_csv(FIXTURES / "university_sample.csv"))
        self.queries = pd.concat(
            [
                load_query_dataset(FIXTURES / "university_validation_sample.csv"),
                # An in-domain wording that no fixture FAQ covers, plus an
                # all-out-of-vocabulary row that produces no query vector.
                pd.DataFrame({"query": ["zzqqxx wwvvuu"], "expected_faq_id": pd.array([1], dtype="Int64"),
                              "is_answerable": [True]}),
            ],
            ignore_index=True,
        )
        options = {"remove_stopwords": False, "lemmatize": False}
        vectorizer, matrix = build_tfidf_index(self.faq, options)
        self.ranking = rank_queries(self.queries["query"], self.faq, vectorizer, matrix, options, 3)

    def test_frame_totals_reproduce_the_aggregate_metrics(self) -> None:
        answerable = self.queries["is_answerable"].to_numpy(dtype=bool)
        for threshold in [0.0, 0.25, 0.6, 1.0]:
            with self.subTest(threshold=threshold):
                frame = prediction_frame(self.queries, self.ranking, threshold)
                report = evaluate_rankings(self.queries, self.faq, self.ranking, threshold)
                accepted = frame["accepted"].to_numpy(dtype=bool)
                self.assertEqual(len(frame), len(self.queries))
                self.assertAlmostEqual(
                    frame["top1_correct"].sum() / answerable.sum(), report["top1_accuracy"]
                )
                self.assertAlmostEqual(
                    frame["correct_answer"].sum() / answerable.sum(),
                    report["correct_answer_rate"],
                )
                self.assertEqual(int(accepted[~answerable].sum()), report["false_acceptance_count"])
                self.assertEqual(
                    int((~accepted[answerable]).sum()), report["false_rejection_count"]
                )

    def test_query_without_features_has_no_prediction_and_no_credit(self) -> None:
        frame = prediction_frame(self.queries, self.ranking, 0.0)
        position = len(self.queries) - 1
        self.assertFalse(self.ranking["has_features"][position])
        self.assertTrue(pd.isna(frame["predicted_faq_id"].iloc[position]))
        self.assertFalse(bool(frame["accepted"].iloc[position]))
        self.assertFalse(bool(frame["top1_correct"].iloc[position]))
        # Every row that does have features still names its top-ranked FAQ.
        known = np.flatnonzero(self.ranking["has_features"])
        self.assertEqual(
            list(frame["predicted_faq_id"].iloc[known]),
            list(self.ranking["ranked_ids"][known, 0]),
        )
        blank = np.flatnonzero(~self.ranking["has_features"])
        self.assertTrue(frame["predicted_faq_id"].iloc[blank].isna().all())

    def test_threshold_outside_range_is_rejected(self) -> None:
        for threshold in [-0.01, 1.01]:
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                prediction_frame(self.queries, self.ranking, threshold)


class Phase2CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        cls.models = cls.root / "models"
        cls.reports = cls.root / "reports"
        cls.corpus = cls.data / "university"
        cls.corpus.mkdir(parents=True)
        cls.reports.mkdir()

        shutil.copyfile(FIXTURES / "university_sample.csv", cls.corpus / "faq_dataset.csv")
        for name in ["validation_queries.csv", "test_queries.csv"]:
            shutil.copyfile(FIXTURES / "university_validation_sample.csv", cls.corpus / name)
        (cls.corpus / "corpus_config.json").write_text(
            json.dumps(
                {
                    "display_name": "University Fixture",
                    "remove_stopwords": False,
                    "lemmatize": False,
                    "similarity_threshold": 0.3,
                    "preprocessing_config": "basic",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        cls.manual = cls.data / "manual_evaluation"
        cls.manual.mkdir()

        # The launcher must relaunch Python itself to fix the hash seed.
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "3"
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train_word2vec.py"),
             "--corpus", "university", "--data-root", str(cls.data),
             "--models-root", str(cls.models)],
            capture_output=True, text=True, env=environment, timeout=180,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.metadata = json.loads(
            (cls.models / "university" / "training_metadata.json").read_text(encoding="utf-8")
        )

    def setUp(self) -> None:
        for attr, value in [("DATA_ROOT", self.data), ("REPORTS_DIR", self.reports),
                            ("MODELS_ROOT", self.models)]:
            patcher = patch.object(evaluate, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # Each test starts from a corpus with no tuned Word2Vec thresholds.
        config_path(self.corpus).unlink(missing_ok=True)
        shutil.rmtree(self.reports)
        self.reports.mkdir()
        (self.manual / "university_queries.csv").write_text(HEADER, encoding="utf-8")

        self.frozen = (self.corpus / "corpus_config.json").read_bytes()
        self.sentinels = {}
        for filename in ["university_evaluation.json", "evaluation_report.md",
                         "manual_university_evaluation.json", "manual_evaluation_report.md"]:
            path = self.reports / filename
            path.write_text("existing phase 1 report", encoding="utf-8")
            self.sentinels[path] = path.read_bytes()

    def assert_baseline_unchanged(self) -> None:
        self.assertEqual((self.corpus / "corpus_config.json").read_bytes(), self.frozen)
        for path, content in self.sentinels.items():
            self.assertEqual(path.read_bytes(), content)

    def tune(self, model: str = "all") -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.run_tuning("university", model)

    def test_model_all_tunes_word2vec_and_reuses_frozen_tfidf(self) -> None:
        with patch.object(evaluate, "tune_tfidf") as tfidf:
            self.tune()
        tfidf.assert_not_called()
        self.assert_baseline_unchanged()

        config = json.loads(config_path(self.corpus).read_text(encoding="utf-8"))
        self.assertEqual(config["artifact_id"], self.metadata["artifact_id"])
        self.assertEqual(config["preprocessing_config"], "basic")
        for model in ["w2v_mean", "w2v_tfidf"]:
            with self.subTest(model=model):
                self.assertTrue(0.0 <= config[model]["similarity_threshold"] <= 1.0)
                sweep = json.loads(
                    (self.reports / "phase2" / f"university_{model}_tuning.json").read_text()
                )
                self.assertEqual(len(sweep["sweep"]), 101)
                self.assertEqual(sweep["artifact_id"], self.metadata["artifact_id"])
                self.assertEqual(
                    sweep["selected_threshold"], config[model]["similarity_threshold"]
                )

    def test_default_tuning_touches_tfidf_only(self) -> None:
        with patch.object(evaluate, "tune_word2vec") as word2vec:
            self.tune("tfidf")
        word2vec.assert_not_called()
        self.assertFalse(config_path(self.corpus).is_file())
        self.assertTrue((self.reports / "university_tuning.json").is_file())

    def test_missing_and_stale_thresholds_fail_with_the_tuning_command(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "evaluate.py tune"):
            evaluate.run_testing("university", "w2v_mean")
        self.tune()
        config = json.loads(config_path(self.corpus).read_text(encoding="utf-8"))
        config["artifact_id"] = "0" * 64
        config_path(self.corpus).write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different model artifact"):
            evaluate.run_testing("university", "w2v_mean")

    def test_three_models_are_evaluated_into_phase2_only(self) -> None:
        self.tune()
        with contextlib.redirect_stdout(io.StringIO()):
            reports = evaluate.run_testing("university", "all")

        self.assertEqual(sorted(reports["university"]), sorted(MODELS))
        self.assert_baseline_unchanged()
        phase2 = self.reports / "phase2"
        for model in MODELS:
            with self.subTest(model=model):
                saved = json.loads(
                    (phase2 / f"university_{model}_evaluation.json").read_text()
                )
                self.assertEqual(saved, reports["university"][model])
                self.assertEqual(saved["evaluation_set"], "synthetic_benchmark")
        self.assertTrue((phase2 / "comparison_report.md").is_file())

        # Every model saw the identical query rows and labels.
        test = load_query_dataset(self.corpus / "test_queries.csv")
        paired = pd.read_csv(phase2 / "university_model_comparison.csv")
        self.assertEqual(list(paired["query"]), list(test["query"]))
        self.assertEqual(list(paired["is_answerable"]), list(test["is_answerable"]))

    def test_paired_csv_reproduces_every_model_report(self) -> None:
        self.tune()
        with contextlib.redirect_stdout(io.StringIO()):
            reports = evaluate.run_testing("university", "all")
        paired = pd.read_csv(self.reports / "phase2" / "university_model_comparison.csv")
        answerable = paired["is_answerable"].to_numpy(dtype=bool)

        for model in MODELS:
            with self.subTest(model=model):
                report = reports["university"][model]
                correct = paired[f"{model}_top1_correct"].to_numpy(dtype=bool)
                delivered = paired[f"{model}_correct_answer"].to_numpy(dtype=bool)
                accepted = paired[f"{model}_accepted"].to_numpy(dtype=bool)
                count = int(answerable.sum())
                self.assertAlmostEqual(correct.sum() / count, report["top1_accuracy"])
                self.assertAlmostEqual(delivered.sum() / count, report["correct_answer_rate"])
                self.assertEqual(
                    int((accepted & answerable & ~correct).sum()), report["accepted_wrong_count"]
                )
                self.assertEqual(
                    int(accepted[~answerable].sum()), report["false_acceptance_count"]
                )
                self.assertEqual(
                    int((~accepted[answerable]).sum()), report["false_rejection_count"]
                )
                # A blank prediction marks a query with no usable vector.
                blank = paired[f"{model}_predicted_faq_id"].isna().to_numpy()
                self.assertTrue(bool((~accepted[blank]).all()))

    def test_manual_templates_stay_pending_for_every_model(self) -> None:
        self.tune()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(evaluate, "tune_word2vec") as tune:
            reports = evaluate.run_testing("university", "all", manual=True)
        self.assertEqual(reports, {})
        self.assertEqual(output.getvalue().count("human evaluation pending"), 1)
        self.assertNotIn("Top-1 accuracy", output.getvalue())
        # Tuning wrote its sweeps, but the pending template produced no results.
        self.assertEqual(list((self.reports / "phase2").glob("manual_*")), [])
        self.assertEqual(list(self.reports.glob("*comparison*")), [])
        tune.assert_not_called()
        self.assert_baseline_unchanged()

    def test_populated_manual_queries_exercise_all_three_models(self) -> None:
        self.tune()
        (self.manual / "university_queries.csv").write_text(
            HEADER + '"How do I request a transcript?",6,True\nzzqqxx wwvvuu,,False\n',
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()), patch.object(evaluate, "tune_word2vec") as tune:
            reports = evaluate.run_testing("university", "all", manual=True)

        self.assertEqual(sorted(reports["university"]), sorted(MODELS))
        phase2 = self.reports / "phase2"
        for model in MODELS:
            with self.subTest(model=model):
                path = phase2 / f"manual_university_{model}_evaluation.json"
                self.assertEqual(json.loads(path.read_text()), reports["university"][model])
                self.assertEqual(reports["university"][model]["evaluation_set"], "manual")
                # An all-OOV unanswerable query is rejected by every model.
                self.assertEqual(reports["university"][model]["unanswerable_rejection_rate"], 1.0)
        self.assertTrue((phase2 / "manual_comparison_report.md").is_file())
        self.assertTrue((phase2 / "manual_university_model_comparison.csv").is_file())
        tune.assert_not_called()
        self.assert_baseline_unchanged()

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Unknown model"):
            evaluate.selected_models("bert")
        self.assertEqual(evaluate.selected_models("all"), MODELS)
        self.assertEqual(evaluate.selected_models("w2v_tfidf"), ["w2v_tfidf"])


class ComparisonNarrativeTests(unittest.TestCase):
    """Improvement and reverse cases must describe real rows or say there are none."""

    def paired(self, baseline: list[bool], other: list[bool]) -> pd.DataFrame:
        size = len(baseline)
        return pd.DataFrame({
            "query": [f"query {i}" for i in range(size)],
            "expected_faq_id": list(range(1, size + 1)),
            "tfidf_predicted_faq_id": list(range(1, size + 1)),
            "tfidf_similarity": [0.5] * size,
            "tfidf_accepted": baseline,
            "tfidf_correct_answer": baseline,
            "w2v_mean_predicted_faq_id": list(range(1, size + 1)),
            "w2v_mean_similarity": [0.9] * size,
            "w2v_mean_accepted": other,
            "w2v_mean_correct_answer": other,
        })

    def test_each_direction_selects_only_its_own_rows(self) -> None:
        paired = self.paired([True, False, True], [False, True, True])
        improved = evaluate.paired_cases(paired, "w2v_mean", improved=True)
        reversed_ = evaluate.paired_cases(paired, "w2v_mean", improved=False)
        self.assertEqual(list(improved["query"]), ["query 1"])
        self.assertEqual(list(reversed_["query"]), ["query 0"])

    def test_empty_category_is_stated_rather_than_invented(self) -> None:
        paired = self.paired([True, True], [False, False])
        lines = evaluate._case_lines(paired, "w2v_mean", improved=True)
        self.assertEqual(lines, ["No queries fall in this category."])
        reverse = evaluate._case_lines(paired, "w2v_mean", improved=False)
        self.assertIn("2 queries", reverse[0])
        self.assertTrue(any("query 0" in line for line in reverse))


class RepositoryTemplateTests(unittest.TestCase):
    def test_committed_manual_templates_remain_header_only(self) -> None:
        for name in ["university", "ecommerce"]:
            path = ROOT / "data" / "manual_evaluation" / f"{name}_queries.csv"
            with self.subTest(corpus=name):
                self.assertEqual(path.read_text(encoding="utf-8").strip(), HEADER.strip())


if __name__ == "__main__":
    unittest.main()
