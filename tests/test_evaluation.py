"""Unit tests for threshold tuning, evaluation metrics, and corpus errors."""

from pathlib import Path
import unittest

import pandas as pd

from src.data_loader import (
    discover_corpora,
    load_faq_dataset,
    load_query_dataset,
    validate_faq_data,
)
from src.evaluation import (
    PREPROCESSING_CONFIGS,
    THRESHOLD_STEPS,
    evaluate_tfidf,
    rank_queries,
    tune_threshold,
)
from src.tfidf_retrieval import build_tfidf_index


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture_faqs() -> pd.DataFrame:
    """Load the small curated fixture corpus used by every test here."""

    return validate_faq_data(pd.read_csv(FIXTURES / "university_sample.csv"))


def build_queries(rows: list[tuple[str, object, bool]]) -> pd.DataFrame:
    """Create a small query table in the shared project schema."""

    return pd.DataFrame(rows, columns=["query", "expected_faq_id", "is_answerable"])


class ThresholdSweepTests(unittest.TestCase):
    def test_sweep_covers_zero_to_one_in_hundredth_steps(self) -> None:
        self.assertEqual(len(THRESHOLD_STEPS), 101)
        self.assertEqual(THRESHOLD_STEPS[0], 0.0)
        self.assertEqual(THRESHOLD_STEPS[-1], 1.0)
        # Integer division keeps every step exact, so 0.07 is not 0.0700000001.
        self.assertEqual(THRESHOLD_STEPS[7], 0.07)

    def test_three_preprocessing_configurations_are_ordered_simplest_first(self) -> None:
        names = [name for name, _ in PREPROCESSING_CONFIGS]
        self.assertEqual(names, ["basic", "stopwords_removed", "lemmatized"])


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.faq = load_fixture_faqs()
        cls.validation = build_queries(
            [
                ("What steps should I follow to apply for undergraduate admission?", 1, True),
                ("Could you explain how to register for courses?", 3, True),
                ("Which page shows the examination schedule?", 5, True),
                ("How do I request a transcript?", 6, True),
                ("volcano cryptocurrency astronomy", pd.NA, False),
                ("Who won the football tournament last night?", pd.NA, False),
            ]
        )

    def test_tuning_is_deterministic(self) -> None:
        first = tune_threshold(self.validation, self.faq)
        second = tune_threshold(self.validation, self.faq)
        self.assertEqual(first["selected_config"], second["selected_config"])
        self.assertEqual(first["selected_threshold"], second["selected_threshold"])
        self.assertEqual(first["selected_score"], second["selected_score"])

    def test_every_configuration_is_swept_across_all_thresholds(self) -> None:
        result = tune_threshold(self.validation, self.faq)
        self.assertEqual(len(result["sweep"]), len(PREPROCESSING_CONFIGS) * 101)

    def test_selected_threshold_is_within_range_and_scored(self) -> None:
        result = tune_threshold(self.validation, self.faq)
        self.assertTrue(0.0 <= result["selected_threshold"] <= 1.0)
        self.assertTrue(0.0 <= result["selected_score"] <= 1.0)

    def test_tie_prefers_the_simpler_configuration(self) -> None:
        # Two identical configurations can only be separated by the tie rule,
        # so the one listed first must win.
        duplicated = [
            ("simple_first", {"remove_stopwords": False, "lemmatize": False}),
            ("complex_second", {"remove_stopwords": False, "lemmatize": False}),
        ]
        result = tune_threshold(self.validation, self.faq, configs=duplicated)
        self.assertEqual(result["selected_config"], "simple_first")

    def test_tie_prefers_the_higher_threshold(self) -> None:
        result = tune_threshold(self.validation, self.faq)
        best_score = result["selected_score"]
        tied = [
            row["threshold"]
            for row in result["sweep"]
            if row["score"] == best_score
            and row["config"] == result["selected_config"]
        ]
        self.assertEqual(result["selected_threshold"], max(tied))

    def test_empty_configuration_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one preprocessing"):
            tune_threshold(self.validation, self.faq, configs=[])


class EvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.faq = load_fixture_faqs()
        cls.vectorizer, cls.matrix = build_tfidf_index(cls.faq)
        cls.test = build_queries(
            [
                ("How do I request a transcript?", 6, True),
                ("I forgot my student portal password. What should I do?", 8, True),
                ("volcano cryptocurrency astronomy", pd.NA, False),
            ]
        )

    def evaluate(self, threshold: float) -> dict:
        return evaluate_tfidf(
            self.test, self.faq, self.vectorizer, self.matrix, threshold=threshold
        )

    def test_exact_questions_reach_top1_and_top3(self) -> None:
        report = self.evaluate(0.30)
        self.assertEqual(report["top1_accuracy"], 1.0)
        self.assertEqual(report["top3_accuracy"], 1.0)
        self.assertEqual(report["answerable_queries"], 2)
        self.assertEqual(report["unanswerable_queries"], 1)

    def test_mean_similarity_uses_only_correct_top1_matches(self) -> None:
        report = self.evaluate(0.30)
        self.assertAlmostEqual(report["mean_similarity_correct_top1"], 1.0)

    def test_false_acceptance_is_counted_at_a_permissive_threshold(self) -> None:
        report = self.evaluate(0.0)
        self.assertEqual(report["false_acceptance_count"], 1)
        self.assertEqual(report["unanswerable_rejection_rate"], 0.0)

    def test_false_rejection_is_counted_at_a_strict_threshold(self) -> None:
        strict = build_queries(
            [("Which office publishes the examination timetable?", 5, True)]
        )
        report = evaluate_tfidf(
            strict, self.faq, self.vectorizer, self.matrix, threshold=1.0
        )
        self.assertEqual(report["false_rejection_count"], 1)
        self.assertEqual(report["answerable_acceptance_rate"], 0.0)

    def test_out_of_vocabulary_unanswerable_query_is_rejected(self) -> None:
        report = self.evaluate(0.30)
        self.assertEqual(report["unanswerable_rejection_rate"], 1.0)
        self.assertEqual(report["false_acceptance_count"], 0)

    def test_incorrect_retrievals_are_reported_with_context(self) -> None:
        wrong = build_queries([("How do I request a transcript?", 1, True)])
        report = evaluate_tfidf(
            wrong, self.faq, self.vectorizer, self.matrix, threshold=0.30
        )
        self.assertEqual(report["top1_accuracy"], 0.0)
        example = report["incorrect_retrieval_examples"][0]
        self.assertEqual(example["expected_faq_id"], 1)
        self.assertEqual(example["retrieved_faq_id"], 6)
        self.assertTrue(example["accepted"])

    def test_threshold_outside_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be between 0 and 1"):
            self.evaluate(1.5)

    def test_empty_query_never_counts_as_accepted(self) -> None:
        blank = build_queries([("!!!", pd.NA, False)])
        report = evaluate_tfidf(
            blank, self.faq, self.vectorizer, self.matrix, threshold=0.0
        )
        self.assertEqual(report["false_acceptance_count"], 0)
        self.assertEqual(report["unanswerable_rejection_rate"], 1.0)

    def test_ranking_marks_token_free_queries(self) -> None:
        ranking = rank_queries(
            pd.Series(["!!!", "transcript"]), self.faq, self.vectorizer, self.matrix
        )
        self.assertFalse(bool(ranking["has_tokens"][0]))
        self.assertTrue(bool(ranking["has_tokens"][1]))


class InvalidCorpusTests(unittest.TestCase):
    def test_missing_data_root_discovers_nothing(self) -> None:
        self.assertEqual(discover_corpora(ROOT / "no_such_directory"), {})

    def test_missing_faq_dataset_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_faq_dataset(ROOT / "no_such_directory")

    def test_missing_query_dataset_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_query_dataset(ROOT / "no_such_directory" / "test_queries.csv")

    def test_query_referencing_unknown_faq_id_is_rejected(self) -> None:
        faq = load_fixture_faqs()
        path = FIXTURES / "university_validation_sample.csv"
        with self.assertRaisesRegex(ValueError, "unknown FAQ ids"):
            load_query_dataset(path, {int(faq["id"].iloc[0])})


class IndexingContractTests(unittest.TestCase):
    def test_answers_are_never_part_of_the_similarity_vectors(self) -> None:
        faq = load_fixture_faqs()
        vectorizer, _ = build_tfidf_index(faq)
        vocabulary = set(vectorizer.vocabulary_)
        # "receipt" appears only in an answer, never in a question.
        self.assertIn("transcript", vocabulary)
        self.assertNotIn("receipt", vocabulary)


if __name__ == "__main__":
    unittest.main()
