"""Unit tests for threshold tuning, evaluation metrics, and corpus errors."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
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
    select_preprocessing,
    tune_threshold,
)
from src.tfidf_retrieval import answer_query, build_tfidf_index


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

    def test_only_retrieval_selected_configuration_is_threshold_swept(self) -> None:
        result = tune_threshold(self.validation, self.faq)
        self.assertEqual(len(result["preprocessing_comparison"]), 3)
        self.assertEqual(len(result["sweep"]), 101)
        self.assertEqual({row["config"] for row in result["sweep"]}, {result["selected_config"]})

    def test_selection_prefers_correct_retrieval_over_a_high_wrong_score(self) -> None:
        faq = self.faq.iloc[:2].copy()
        faq["question"] = ["the", "beta gamma delta"]
        validation = build_queries([("the beta", 2, True), ("the", pd.NA, False)])
        result = tune_threshold(validation, faq, configs=PREPROCESSING_CONFIGS[:2])
        self.assertEqual(result["selected_config"], "stopwords_removed")
        comparison = result["preprocessing_comparison"]
        self.assertEqual(comparison[0]["top1_accuracy"], 0.0)
        self.assertEqual(comparison[1]["top1_accuracy"], 1.0)

    def test_top3_breaks_top1_ties_before_simplicity(self) -> None:
        validation = build_queries([("first", 1, True), ("second", 2, True),
                                    ("ignore this", pd.NA, False)])
        configs = [(name, {}) for name in ["simple", "better_top3", "worse_top1"]]
        rankings = [
            {"ranked_ids": np.array(ids), "has_features": np.array([True, True])}
            for ids in [[[1, 2, 3], [4, 5, 6]],
                        [[1, 2, 3], [3, 2, 4]],
                        [[2, 1, 3], [3, 2, 4]]]
        ]
        with patch("src.evaluation.rank_queries", side_effect=rankings) as rank:
            result = select_preprocessing(validation, self.faq, configs=configs)
        self.assertEqual(result["selected_config"], "better_top3")
        for call in rank.call_args_list:
            self.assertEqual(call.args[0].tolist(), ["first", "second"])

    def test_oov_queries_cannot_win_preprocessing_selection_by_id_ties(self) -> None:
        validation = build_queries([("zzqqxx wwvvuu", 1, True)])
        result = select_preprocessing(validation, self.faq, configs=[("basic", {})])
        row = result["preprocessing_comparison"][0]
        self.assertEqual((row["top1_accuracy"], row["top3_accuracy"]), (0.0, 0.0))

    def test_threshold_tuning_requires_both_answerability_classes(self) -> None:
        for label in [True, False]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "answerable and unanswerable"):
                tune_threshold(self.validation[self.validation.is_answerable == label], self.faq)

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
        # An in-vocabulary but wrongly labelled question tests the metric;
        # zero-feature questions must be rejected at every threshold.
        unanswerable = build_queries([("How do I request a transcript?", pd.NA, False)])
        report = evaluate_tfidf(unanswerable, self.faq, self.vectorizer, self.matrix, 0.0)
        self.assertEqual(report["false_acceptance_count"], 1)
        self.assertEqual(report["unanswerable_rejection_rate"], 0.0)

    def test_zero_features_never_receive_retrieval_credit_or_an_answer(self) -> None:
        query = build_queries([("zzqqxx wwvvuu", 1, True)])
        for threshold in [0.0, 0.3, 1.0]:
            with self.subTest(threshold=threshold):
                report = evaluate_tfidf(query, self.faq, self.vectorizer, self.matrix, threshold)
                for metric in ["top1_accuracy", "top3_accuracy", "correct_answer_rate",
                               "mean_similarity_correct_top1", "answerable_acceptance_rate"]:
                    self.assertEqual(report[metric], 0.0)
                self.assertEqual(report["false_rejection_count"], 1)
                self.assertIsNone(report["incorrect_retrieval_examples"][0]["retrieved_faq_id"])
        self.assertEqual(self.evaluate(0.0)["false_acceptance_count"], 0)

    def test_runtime_and_batch_agree_on_features_scores_and_acceptance(self) -> None:
        for name, options in PREPROCESSING_CONFIGS:
            vectorizer, matrix = build_tfidf_index(self.faq, options)
            queries = pd.Series(["", "!!!", "the and", "zzqqxx wwvvuu", "transcript zzqqxx",
                                 "How do I request a transcript?"])
            ranking = rank_queries(queries, self.faq, vectorizer, matrix, options)
            for i, query in enumerate(queries):
                for threshold in [0.0, 0.3, 1.0]:
                    with self.subTest(config=name, query=query, threshold=threshold):
                        result = answer_query(query, self.faq, vectorizer, matrix,
                                              threshold, preprocessing_options=options)
                        expected = bool(ranking["has_features"][i] and
                                        ranking["ranked_scores"][i, 0] >= threshold)
                        self.assertEqual(result["found"], expected)
                        if ranking["has_features"][i]:
                            self.assertEqual(result["top_matches"][0]["faq_id"], ranking["ranked_ids"][i, 0])
                            self.assertAlmostEqual(result["top_matches"][0]["similarity"], ranking["ranked_scores"][i, 0])
                        else:
                            self.assertEqual(result["top_matches"], [])

    def test_answer_delivery_metrics_separate_wrong_answers_and_rejections(self) -> None:
        # Four answerable queries: correct+accepted, wrong+accepted,
        # correct+rejected, OOV+rejected. Plus one unanswerable rejection.
        data = build_queries([("a", 1, True), ("b", 2, True), ("c", 3, True),
                              ("d", 1, True), ("e", pd.NA, False)])
        ranking = {
            "ranked_ids": np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1], [1, 2, 3], [1, 2, 3]]),
            "ranked_scores": np.array([[.9, .2, .1], [.8, .3, .1], [.2, .1, 0], [0, 0, 0], [0, 0, 0]]),
            "has_features": np.array([True, True, True, False, False]),
        }
        with patch("src.evaluation.rank_queries", return_value=ranking):
            report = evaluate_tfidf(data, self.faq, self.vectorizer, self.matrix, .5)
        self.assertEqual(report["top1_accuracy"], .5)
        self.assertEqual(report["top3_accuracy"], .75)
        self.assertEqual(report["correct_answer_rate"], .25)
        self.assertEqual(report["accepted_wrong_count"], 1)
        self.assertEqual(report["false_rejection_count"], 2)
        self.assertEqual(report["false_acceptance_count"], 0)
        self.assertEqual(report["unanswerable_rejection_rate"], 1.0)

    def test_no_answerable_queries_have_zero_answer_delivery_metrics(self) -> None:
        data = build_queries([("zzqqxx", pd.NA, False)])
        report = evaluate_tfidf(data, self.faq, self.vectorizer, self.matrix, 0.0)
        self.assertEqual(report["correct_answer_rate"], 0.0)
        self.assertEqual(report["accepted_wrong_count"], 0)

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
        ranking = rank_queries(pd.Series(["zzqqxx", "transcript"]), self.faq,
                               self.vectorizer, self.matrix)
        self.assertTrue(ranking["has_tokens"][0])
        self.assertFalse(ranking["has_features"][0])
        self.assertTrue(ranking["has_features"][1])


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
