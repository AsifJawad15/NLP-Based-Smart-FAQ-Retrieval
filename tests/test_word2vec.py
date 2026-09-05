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
from src.embedding_utils import sentence_to_mean_vector
from src.evaluation import evaluate_rankings, tune_ranked_threshold
from src.word2vec_retrieval import (
    answer_word2vec, build_word2vec_index, rank_word2vec_queries, retrieve_word2vec,
)
from src.word2vec_training import BASIC_OPTIONS, load_word2vec, question_sentences


ROOT = Path(__file__).resolve().parents[1]


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


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.faq = toy_faq()
        self.index = build_word2vec_index(self.faq, toy_vectors())

    def test_exact_noncolliding_question_and_match_contract(self):
        result = answer_word2vec("apple", self.faq, self.index, 1.0)
        self.assertTrue(result["found"])
        self.assertEqual(result["best_match"]["faq_id"], 1)
        self.assertEqual(set(result["best_match"]),
                         {"faq_id", "question", "answer", "category", "source", "similarity"})

    def test_sorting_topk_and_invalid_faq_exclusion(self):
        matches = retrieve_word2vec("banana", self.faq, self.index, 10)
        self.assertEqual([m["faq_id"] for m in matches], [2, 1])
        self.assertEqual(len(retrieve_word2vec("apple", self.faq, self.index, 1)), 1)
        self.assertEqual([m["similarity"] for m in matches], [1., 0.])

    def test_stable_tie_uses_original_faq_order(self):
        matches = retrieve_word2vec("apple banana", self.faq, self.index)
        self.assertEqual([m["faq_id"] for m in matches], [1, 2])

    def test_signed_cosine_and_zero_threshold(self):
        faq = self.faq.iloc[:1]
        index = build_word2vec_index(faq, toy_vectors())
        result = answer_word2vec("opposite", faq, index, 0.0)
        self.assertEqual(result["top_matches"][0]["similarity"], -1.)
        self.assertFalse(result["found"])

    def test_empty_oov_and_cancellation_reject_at_zero(self):
        for query in ["", "!!!", "unknown", "apple opposite"]:
            result = answer_word2vec(query, self.faq, self.index, 0.0)
            self.assertFalse(result["found"])
            self.assertEqual(result["top_matches"], [])

    def test_runtime_batch_and_metrics_agree(self):
        queries = pd.DataFrame({"query": ["apple unknown", "banana", "unknown"],
                                "expected_faq_id": [1, 2, pd.NA],
                                "is_answerable": [True, True, False]})
        ranking = rank_word2vec_queries(queries["query"], self.faq, self.index)
        for i, query in enumerate(queries["query"]):
            result = answer_word2vec(query, self.faq, self.index, .5)
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
        faq = self.faq.iloc[2:]
        index = build_word2vec_index(faq, toy_vectors())
        self.assertEqual(retrieve_word2vec("apple", faq, index), [])
        queries = pd.DataFrame({"query": ["unknown"], "expected_faq_id": [1], "is_answerable": [True]})
        ranking = rank_word2vec_queries(queries["query"], self.faq, self.index)
        report = evaluate_rankings(queries, self.faq, ranking, 0.)
        self.assertEqual(report["top1_accuracy"], 0.)
        self.assertEqual(report["top3_accuracy"], 0.)
        self.assertIsNone(report["incorrect_retrieval_examples"][0]["retrieved_faq_id"])

    def test_bad_order_topk_and_threshold_fail(self):
        with self.assertRaisesRegex(ValueError, "ids/order"):
            retrieve_word2vec("apple", self.faq.iloc[::-1], self.index)
        with self.assertRaisesRegex(ValueError, "top_k"):
            retrieve_word2vec("apple", self.faq, self.index, 0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            answer_word2vec("apple", self.faq, self.index, -0.1)


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
