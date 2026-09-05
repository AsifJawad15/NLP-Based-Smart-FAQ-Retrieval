"""Exercise manual evaluation with isolated CSVs and report destinations."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from src.data_loader import discover_corpora, load_query_dataset


FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = "query,expected_faq_id,is_answerable\n"


class ManualEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.manual = self.data / "manual_evaluation"
        self.manual.mkdir(parents=True)
        self.reports = self.root / "reports"
        self.reports.mkdir()
        self.config_paths = []
        for name in ["university", "ecommerce"]:
            directory = self.data / name
            directory.mkdir()
            shutil.copyfile(FIXTURES / "university_sample.csv", directory / "faq_dataset.csv")
            path = directory / "corpus_config.json"
            path.write_text(json.dumps({"preprocessing_config": "basic", "similarity_threshold": .3}), encoding="utf-8")
            self.config_paths.append(path)
            (self.manual / f"{name}_queries.csv").write_text(HEADER, encoding="utf-8")
        self.sentinels = {p: p.read_bytes() for p in self.config_paths}
        for filename in ["university_evaluation.json", "ecommerce_evaluation.json", "evaluation_report.md"]:
            path = self.reports / filename
            path.write_text("existing synthetic report", encoding="utf-8")
            self.sentinels[path] = path.read_bytes()
        for attr, value in [("DATA_ROOT", self.data), ("REPORTS_DIR", self.reports)]:
            patcher = patch.object(evaluate, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_benchmark_unchanged(self) -> None:
        for path, content in self.sentinels.items():
            self.assertEqual(path.read_bytes(), content)

    def test_empty_templates_are_pending_without_reports_or_tuning(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(evaluate, "tune_threshold") as tune:
            reports = evaluate.run_testing("all", manual=True)
        self.assertEqual(reports, {})
        self.assertEqual(output.getvalue().count("human evaluation pending"), 2)
        self.assertEqual(list(self.reports.glob("manual_*")), [])
        self.assertNotIn("Top-1 accuracy", output.getvalue())
        tune.assert_not_called()
        self.assert_benchmark_unchanged()
        self.assertEqual(set(discover_corpora(self.data)), {"university", "ecommerce"})

    def test_populated_queries_use_frozen_settings_and_separate_outputs(self) -> None:
        path = self.manual / "university_queries.csv"
        path.write_text(HEADER + '"How do I request a transcript?",6,True\nzzqqxx,,False\n', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), patch.object(evaluate, "tune_threshold") as tune:
            reports = evaluate.run_testing("university", manual=True)
        report = reports["university"]["tfidf"]
        self.assertEqual(report["threshold"], .3)
        self.assertEqual(report["correct_answer_rate"], 1.0)
        self.assertEqual(report["unanswerable_rejection_rate"], 1.0)
        self.assertEqual(report["evaluation_set"], "manual")
        self.assertEqual(json.loads((self.reports / "manual_university_evaluation.json").read_text()), report)
        self.assertTrue((self.reports / "manual_evaluation_report.md").is_file())
        self.assertFalse((self.reports / "manual_ecommerce_evaluation.json").exists())
        tune.assert_not_called()
        self.assert_benchmark_unchanged()

    def test_all_mode_reports_populated_corpus_and_names_pending_one(self) -> None:
        (self.manual / "university_queries.csv").write_text(
            HEADER + '"How do I request a transcript?",6,True\n', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            reports = evaluate.run_testing("all", manual=True)
        self.assertEqual(set(reports), {"university"})
        summary = (self.reports / "manual_evaluation_report.md").read_text()
        self.assertIn("Human evaluation pending (no scores): ecommerce", summary)
        self.assert_benchmark_unchanged()

    def test_bad_manual_labels_or_ids_are_rejected(self) -> None:
        path = self.manual / "university_queries.csv"
        for row, message in [
            ("question,99999,True\n", "unknown FAQ ids"),
            ("question,,True\n", "require an expected_faq_id"),
            ("question,6,False\n", "must have a blank"),
            ("question,6,perhaps\n", "Invalid is_answerable"),
        ]:
            with self.subTest(row=row):
                path.write_text(HEADER + row, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    evaluate.run_testing("university", manual=True)
        self.assertEqual(list(self.reports.glob("manual_*")), [])

    def test_empty_template_requires_opt_in_and_still_validates_header(self) -> None:
        path = self.manual / "university_queries.csv"
        with self.assertRaisesRegex(ValueError, "at least one row"):
            load_query_dataset(path)
        self.assertTrue(load_query_dataset(path, allow_empty=True).empty)
        path.write_text("query\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_query_dataset(path, allow_empty=True)


if __name__ == "__main__":
    unittest.main()
