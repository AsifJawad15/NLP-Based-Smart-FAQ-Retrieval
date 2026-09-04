"""Unit tests for loading, preprocessing, ranking, and rejection."""

from pathlib import Path
import unittest

import pandas as pd

from src.data_loader import discover_corpora, load_faq_dataset, validate_faq_data
from src.preprocessing import preprocess_text
from src.tfidf_retrieval import answer_query, build_tfidf_index, retrieve_tfidf


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class PreprocessingTests(unittest.TestCase):
    def test_basic_cleaning_removes_url_html_and_punctuation(self) -> None:
        processed = preprocess_text("Visit <b>Portal</b> at https://test.edu NOW!!!")
        self.assertEqual(processed, "visit portal at now")

    def test_stopword_removal_preserves_negation(self) -> None:
        processed = preprocess_text("I do not want the course", remove_stopwords=True)
        self.assertIn("not", processed.split())
        self.assertNotIn("the", processed.split())

    def test_optional_lemmatization(self) -> None:
        self.assertEqual(preprocess_text("books", lemmatize=True), "book")


class DataLoaderTests(unittest.TestCase):
    def test_fixture_loads_with_integer_ids(self) -> None:
        data = validate_faq_data(pd.read_csv(FIXTURES / "university_sample.csv"))
        self.assertTrue(pd.api.types.is_integer_dtype(data["id"]))

    def test_duplicate_ids_are_rejected(self) -> None:
        data = pd.read_csv(FIXTURES / "university_sample.csv")
        data.loc[1, "id"] = data.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            validate_faq_data(data)

    def test_corpora_are_discovered_from_data_directory(self) -> None:
        corpora = discover_corpora(ROOT / "data")
        self.assertEqual(set(corpora), {"ecommerce", "university"})


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_faq_dataset(ROOT / "data" / "university")
        cls.vectorizer, cls.matrix = build_tfidf_index(cls.data)

    def test_exact_question_is_ranked_first(self) -> None:
        matches = retrieve_tfidf(
            "How do I request a transcript?",
            self.data,
            self.vectorizer,
            self.matrix,
        )
        self.assertEqual(matches[0]["faq_id"], 6)
        self.assertAlmostEqual(matches[0]["similarity"], 1.0)

    def test_paraphrase_with_shared_terms_is_retrieved(self) -> None:
        matches = retrieve_tfidf(
            "I need to reset my forgotten portal password",
            self.data,
            self.vectorizer,
            self.matrix,
        )
        self.assertEqual(matches[0]["faq_id"], 8)

    def test_top_k_is_ordered_and_limited(self) -> None:
        matches = retrieve_tfidf(
            "student portal registration",
            self.data,
            self.vectorizer,
            self.matrix,
            top_k=3,
        )
        self.assertEqual(len(matches), 3)
        scores = [match["similarity"] for match in matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_oov_query_is_rejected(self) -> None:
        result = answer_query(
            "volcano cryptocurrency astronomy",
            self.data,
            self.vectorizer,
            self.matrix,
            threshold=0.30,
        )
        self.assertFalse(result["found"])
        self.assertIsNone(result["best_match"])

    def test_empty_query_returns_no_matches(self) -> None:
        matches = retrieve_tfidf("!!!", self.data, self.vectorizer, self.matrix)
        self.assertEqual(matches, [])

    def test_threshold_boundary_is_inclusive(self) -> None:
        result = answer_query(
            "How do I request a transcript?",
            self.data,
            self.vectorizer,
            self.matrix,
            threshold=1.0,
        )
        self.assertTrue(result["found"])


if __name__ == "__main__":
    unittest.main()
