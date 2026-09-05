"""Numerical vectors, retrieval parity, and reproducible isolated training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from gensim.models import KeyedVectors, Word2Vec
import numpy as np
import pandas as pd

from src.data_loader import load_faq_dataset
from src.embedding_utils import sentence_to_mean_vector, sentence_to_tfidf_weighted_vector
from src.evaluation import evaluate_rankings, tune_ranked_threshold
from src.word2vec_retrieval import (
    answer_word2vec, build_word2vec_index, rank_word2vec_queries, retrieve_word2vec,
)
from src.word2vec_training import BASIC_OPTIONS, load_word2vec, question_sentences


ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = ["w2v_mean", "w2v_tfidf"]


def toy_vectors():
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["apple", "banana", "opposite", "cancel"],
                        np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32))
    return vectors


def toy_faq():
    return pd.DataFrame({
        "id": [1, 2, 3], "question": ["apple", "banana", "zzqqxx"],
        "answer": ["red fruit", "yellow fruit", "absent"], "category": ["food"] * 3,
        "source": ["https://example.test/faq"] * 3, "source_type": ["fixture"] * 3,
    })


class MeanVectorTests(unittest.TestCase):
    def test_mean_counts_repeated_words_and_ignores_oov(self):
        vector = sentence_to_mean_vector(["apple", "apple", "banana", "unknown"], toy_vectors())
        np.testing.assert_allclose(vector, [2 / 3, 1 / 3])
        self.assertEqual(vector.shape, (2,))

    def test_empty_oov_and_cancellation_have_no_vector(self):
        for tokens in [[], ["unknown"], ["apple", "opposite"]]:
            self.assertIsNone(sentence_to_mean_vector(tokens, toy_vectors()))

    def test_nonfinite_vectors_are_rejected(self):
        vectors = toy_vectors()
        vectors["apple"] = np.array([np.nan, 0.])
        with self.assertRaisesRegex(ValueError, "finite"):
            sentence_to_mean_vector(["apple"], vectors)


class WeightedVectorTests(unittest.TestCase):
    IDF = {"apple": 1.0, "banana": 2.0}

    def test_weighted_counts_each_repeated_term_once(self):
        vector = sentence_to_tfidf_weighted_vector(
            ["apple", "apple", "banana"], toy_vectors(), self.IDF
        )
        # Unique terms: apple weighs 2 * 1.0 and banana weighs 1 * 2.0, so the
        # two contributions are equal.
        np.testing.assert_allclose(vector, [.5, .5])
        # Lab 3 multiplies by the count and then loops over the repeats again,
        # which would weigh apple 4.0 against banana 2.0.
        self.assertFalse(np.allclose(vector, np.array([4., 2.]) / 6))

    def test_weighted_skips_unknown_embeddings_and_terms_without_idf(self):
        # 'cancel' has an embedding but no corpus IDF; 'unknown' has neither.
        vector = sentence_to_tfidf_weighted_vector(
            ["apple", "cancel", "unknown"], toy_vectors(), self.IDF
        )
        np.testing.assert_allclose(vector, [1., 0.])

    def test_weighted_empty_oov_zero_weight_and_cancellation_have_no_vector(self):
        for tokens, idf in [([], self.IDF), (["unknown"], self.IDF), (["cancel"], self.IDF),
                            (["apple"], {"apple": 0.}),
                            (["apple", "opposite"], {"apple": 1., "opposite": 1.})]:
            with self.subTest(tokens=tokens):
                self.assertIsNone(sentence_to_tfidf_weighted_vector(tokens, toy_vectors(), idf))

    def test_weighted_rejects_negative_or_nonfinite_idf(self):
        for idf in [{"apple": -1.}, {"apple": float("nan")}, {"apple": float("inf")}]:
            with self.subTest(idf=idf), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                sentence_to_tfidf_weighted_vector(["apple"], toy_vectors(), idf)


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.faq = toy_faq()
        self.indexes = {
            name: build_word2vec_index(self.faq, toy_vectors(), name) for name in AGGREGATIONS
        }
        self.index = self.indexes["w2v_mean"]

    # The toy FAQ gives every indexed term the same IDF, so both aggregations
    # must agree on ranking, ties, acceptance, and argument validation.

    def test_unknown_aggregation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregation"):
            build_word2vec_index(self.faq, toy_vectors(), "bag_of_words")

    def test_exact_noncolliding_question_and_match_contract(self):
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                result = answer_word2vec("apple", self.faq, index, 1.0)
                self.assertTrue(result["found"])
                self.assertEqual(result["best_match"]["faq_id"], 1)
                self.assertEqual(set(result["best_match"]),
                                 {"faq_id", "question", "answer", "category", "source", "similarity"})

    def test_sorting_topk_and_invalid_faq_exclusion(self):
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                matches = retrieve_word2vec("banana", self.faq, index, 10)
                self.assertEqual([m["faq_id"] for m in matches], [2, 1])
                self.assertEqual(len(retrieve_word2vec("apple", self.faq, index, 1)), 1)
                self.assertEqual([m["similarity"] for m in matches], [1., 0.])

    def test_stable_tie_uses_original_faq_order(self):
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                matches = retrieve_word2vec("apple banana", self.faq, index)
                self.assertEqual([m["faq_id"] for m in matches], [1, 2])

    def test_mean_signed_cosine_cancellation_and_zero_threshold(self):
        faq = self.faq.iloc[:1]
        index = build_word2vec_index(faq, toy_vectors())
        result = answer_word2vec("opposite", faq, index, 0.0)
        self.assertEqual(result["top_matches"][0]["similarity"], -1.)
        self.assertFalse(result["found"])
        self.assertEqual(answer_word2vec("apple opposite", self.faq, self.index, 0.)["top_matches"], [])

    def test_weighted_signed_cosine_and_cancellation_use_corpus_idf(self):
        # Both questions are one term of one document, so their IDFs are equal.
        faq = self.faq.iloc[:2].copy()
        faq["question"] = ["apple", "opposite"]
        index = build_word2vec_index(faq, toy_vectors(), "w2v_tfidf")
        matches = answer_word2vec("opposite", faq, index, 0.0)["top_matches"]
        self.assertEqual([(m["faq_id"], m["similarity"]) for m in matches], [(2, 1.), (1, -1.)])
        self.assertEqual(answer_word2vec("apple opposite", faq, index, 0.)["top_matches"], [])

    def test_weighted_ignores_query_terms_without_corpus_idf(self):
        # 'opposite' has an embedding but never appears in the FAQ questions, so
        # the frozen corpus IDF gives it no weight instead of being refitted.
        index = self.indexes["w2v_tfidf"]
        self.assertEqual(retrieve_word2vec("opposite", self.faq, index), [])
        match = retrieve_word2vec("apple opposite", self.faq, index, 1)[0]
        self.assertEqual((match["faq_id"], match["similarity"]), (1, 1.))

    def test_empty_and_oov_queries_reject_at_zero(self):
        for name, index in self.indexes.items():
            for query in ["", "!!!", "unknown"]:
                with self.subTest(aggregation=name, query=query):
                    result = answer_word2vec(query, self.faq, index, 0.0)
                    self.assertFalse(result["found"])
                    self.assertEqual(result["top_matches"], [])

    def test_runtime_batch_and_metrics_agree(self):
        queries = pd.DataFrame({"query": ["apple unknown", "banana", "unknown"],
                                "expected_faq_id": [1, 2, pd.NA],
                                "is_answerable": [True, True, False]})
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                ranking = rank_word2vec_queries(queries["query"], self.faq, index)
                for i, query in enumerate(queries["query"]):
                    result = answer_word2vec(query, self.faq, index, .5)
                    self.assertEqual(result["found"], bool(ranking["has_features"][i] and
                                                           ranking["ranked_scores"][i, 0] >= .5))
                    if result["found"]:
                        self.assertEqual(result["best_match"]["faq_id"], ranking["ranked_ids"][i, 0])
                report = evaluate_rankings(queries, self.faq, ranking, .5, BASIC_OPTIONS)
                self.assertEqual(report["correct_answer_rate"], 1.)
                self.assertEqual(report["unanswerable_rejection_rate"], 1.)
                tuning = tune_ranked_threshold(queries, ranking)
                self.assertEqual(tuning["selected_threshold"], 1.)

    def test_no_usable_faqs_or_oov_queries_receive_no_metric_credit(self):
        queries = pd.DataFrame({"query": ["unknown"], "expected_faq_id": [1], "is_answerable": [True]})
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                faq = self.faq.iloc[2:]
                self.assertEqual(
                    retrieve_word2vec("apple", faq, build_word2vec_index(faq, toy_vectors(), name)), []
                )
                ranking = rank_word2vec_queries(queries["query"], self.faq, index)
                report = evaluate_rankings(queries, self.faq, ranking, 0.)
                self.assertEqual(report["top1_accuracy"], 0.)
                self.assertEqual(report["top3_accuracy"], 0.)
                self.assertIsNone(report["incorrect_retrieval_examples"][0]["retrieved_faq_id"])

    def test_bad_order_topk_and_threshold_fail(self):
        for name, index in self.indexes.items():
            with self.subTest(aggregation=name):
                with self.assertRaisesRegex(ValueError, "ids/order"):
                    retrieve_word2vec("apple", self.faq.iloc[::-1], index)
                with self.assertRaisesRegex(ValueError, "top_k"):
                    retrieve_word2vec("apple", self.faq, index, 0)
                with self.assertRaisesRegex(ValueError, "threshold"):
                    answer_word2vec("apple", self.faq, index, -0.1)

class TrainingPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        cls.models = cls.root / "models"
        for name, questions in [("university", ["campus library portal", "student portal campus"]),
                                ("ecommerce", ["parcel refund purchase", "purchase delivery parcel"])]:
            directory = cls.data / name
            directory.mkdir(parents=True)
            faq = toy_faq().iloc[:2].copy()
            faq["question"] = questions
            faq["answer"] = "answeronlytoken"
            faq.to_csv(directory / "faq_dataset.csv", index=False)
            (directory / "test_queries.csv").write_text("query\nevaluationonlytoken\n")
        cls.run_training(cls.models)

    @classmethod
    def run_training(cls, target):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "7"  # CLI must relaunch before training.
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train_word2vec.py"), "--corpus", "all",
             "--data-root", str(cls.data), "--models-root", str(target)],
            capture_output=True, text=True, env=environment, timeout=120,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_fresh_process_reproducibility(self):
        second = self.root / "repeated_models"
        self.run_training(second)
        for name in ["university", "ecommerce"]:
            first, meta = load_word2vec(self.data / name, self.models)
            repeated, other = load_word2vec(self.data / name, second)
            self.assertEqual(first.wv.index_to_key, repeated.wv.index_to_key)
            np.testing.assert_array_equal(first.wv.vectors, repeated.wv.vectors)
            self.assertEqual(meta["artifact_id"], other["artifact_id"])

    def test_question_only_separate_domain_vocabulary(self):
        uni, _ = load_word2vec(self.data / "university", self.models)
        shop, _ = load_word2vec(self.data / "ecommerce", self.models)
        self.assertIn("campus", uni.wv)
        self.assertNotIn("campus", shop.wv)
        self.assertIn("parcel", shop.wv)
        self.assertNotIn("parcel", uni.wv)
        for model in [uni, shop]:
            self.assertNotIn("answeronlytoken", model.wv)
            self.assertNotIn("evaluationonlytoken", model.wv)

    def test_save_reload_preserves_prediction(self):
        directory = self.data / "university"
        model, _ = load_word2vec(directory, self.models)
        faq = load_faq_dataset(directory)
        first = retrieve_word2vec(faq.question.iloc[0], faq, build_word2vec_index(faq, model.wv))
        path = self.root / "reload.model"
        model.save(str(path))
        reloaded = Word2Vec.load(str(path))
        second = retrieve_word2vec(faq.question.iloc[0], faq, build_word2vec_index(faq, reloaded.wv))
        self.assertEqual(first, second)

    def test_missing_and_stale_models_fail_with_training_instruction(self):
        with self.assertRaisesRegex(FileNotFoundError, "train_word2vec.py"):
            load_word2vec(self.data / "university", self.root / "missing")
        copied = self.root / "altered" / "university"
        copied.mkdir(parents=True)
        shutil.copyfile(self.data / "university/faq_dataset.csv", copied / "faq_dataset.csv")
        with (copied / "faq_dataset.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Stale.*train_word2vec.py"):
            load_word2vec(copied, self.models)


if __name__ == "__main__":
    unittest.main()
