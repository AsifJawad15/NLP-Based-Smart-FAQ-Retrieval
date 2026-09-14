"""KUET demo configuration, cache, report isolation, and GUI behavior."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

import evaluate
from main import build_answerer
from src.data_loader import (
    DEFAULT_SUPPORTED_MODELS,
    load_corpus_config,
    load_faq_dataset,
    load_query_dataset,
)
from src.gui_support import corpora_for_purpose, file_change_signature, saved_benchmark_rows


ROOT = Path(__file__).resolve().parents[1]
KUET = ROOT / "data" / "kuet"


class CorpusConfigurationTests(unittest.TestCase):
    def test_legacy_config_defaults_to_research_and_all_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "corpus_config.json").write_text(
                json.dumps({"display_name": "Legacy", "similarity_threshold": 0.4}),
                encoding="utf-8",
            )
            config = load_corpus_config(path)
        self.assertEqual(config["purpose"], "research")
        self.assertEqual(config["supported_models"], DEFAULT_SUPPORTED_MODELS)

    def test_default_evaluation_excludes_demo_but_explicit_selection_works(self) -> None:
        self.assertEqual(set(evaluate.selected_corpora("all")), {"ecommerce", "university"})
        self.assertEqual(set(evaluate.selected_corpora("kuet")), {"kuet"})
        self.assertEqual(set(corpora_for_purpose(ROOT / "data", "demo")), {"kuet"})

    def test_kuet_rejects_unsupported_models_before_artifact_loading(self) -> None:
        faq = load_faq_dataset(KUET)
        config = load_corpus_config(KUET)
        with self.assertRaisesRegex(ValueError, "not supported"):
            build_answerer(KUET, faq, config, "rnn")
        with self.assertRaisesRegex(ValueError, "does not support"):
            evaluate.ensure_supported_models("kuet", config, ["tfidf", "w2v_mean"])

    def test_demo_reports_have_an_isolated_destination(self) -> None:
        config = load_corpus_config(KUET)
        self.assertEqual(evaluate.corpus_report_dir(config), ROOT / "reports" / "demo")


class KuetDataTests(unittest.TestCase):
    def test_reviewed_corpus_and_development_splits(self) -> None:
        faq = load_faq_dataset(KUET)
        validation = load_query_dataset(KUET / "validation_queries.csv", set(faq["id"]))
        smoke = load_query_dataset(KUET / "smoke_queries.csv", set(faq["id"]))
        self.assertEqual(len(faq), 200)
        self.assertGreaterEqual(faq["category"].nunique(), 8)
        self.assertEqual(set(faq["source_type"]), {"official_web"})
        self.assertEqual((len(validation), int(validation["is_answerable"].sum())), (100, 70))
        self.assertEqual((len(smoke), int(smoke["is_answerable"].sum())), (60, 45))
        self.assertEqual(int((faq["category"] == "department_overviews").sum()), 20)
        self.assertEqual(int((faq["category"] == "student_clubs").sum()), 10)
        self.assertTrue((KUET / "SOURCE_REVIEW.md").is_file())

    def test_file_signature_changes_when_a_resource_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "demo"
            corpus.mkdir()
            faq = corpus / "faq_dataset.csv"
            config = corpus / "corpus_config.json"
            faq.write_text("one", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            first = file_change_signature(corpus, "tfidf", root / "models")
            faq.write_text("content with a different size", encoding="utf-8")
            second = file_change_signature(corpus, "tfidf", root / "models")
        self.assertNotEqual(first, second)

    def test_all_seven_saved_research_metrics_are_available(self) -> None:
        rows = saved_benchmark_rows(ROOT / "reports", "university")
        self.assertEqual(len(rows), 7)
        self.assertTrue(any("experimental" in row["Model"] for row in rows))


class StreamlitAppTests(unittest.TestCase):
    def app(self) -> AppTest:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_page_has_separate_assistant_and_comparison_tabs(self) -> None:
        app = self.app()
        self.assertEqual([tab.label for tab in app.tabs], ["FAQ Assistant", "Model Comparison"])
        self.assertEqual(app.selectbox(key="assistant_corpus").value, "kuet")

    def test_accepted_question_shows_stored_answer_and_source(self) -> None:
        app = self.app()
        app.text_input(key="assistant_query").set_value(
            "How many books may a KUET undergraduate borrow?"
        )
        app.button(key="assistant_search").click()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Relevant FAQ found", [item.value for item in app.success])
        markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("An undergraduate student may borrow three books", markdown)
        self.assertIn("https://library.kuet.ac.bd/about.php", markdown)

    def test_rejected_question_never_shows_a_candidate_answer(self) -> None:
        app = self.app()
        app.text_input(key="assistant_query").set_value("volcano cryptocurrency astronomy")
        app.button(key="assistant_search").click()
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn(
            "No sufficiently relevant FAQ found", [item.value for item in app.warning]
        )
        self.assertNotIn("**Answer:**", "\n".join(item.value for item in app.markdown))

    def test_club_overview_can_be_evaluated_without_knowing_faq_ids(self) -> None:
        app = self.app()
        app.text_input(key="assistant_query").set_value("Tell me about KUET Career Club")
        app.button(key="assistant_search").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("BizBattle", "\n".join(item.value for item in app.markdown))
        next(item for item in app.selectbox if item.label == "Your assessment").select("Correct")
        next(item for item in app.button if item.label == "Save evaluation").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        feedback = app.session_state["human_evaluations"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["query"], "Tell me about KUET Career Club")
        self.assertEqual(feedback[0]["assessment"], "Correct")
        self.assertEqual(feedback[0]["faq_id"], 114)


if __name__ == "__main__":
    unittest.main()
