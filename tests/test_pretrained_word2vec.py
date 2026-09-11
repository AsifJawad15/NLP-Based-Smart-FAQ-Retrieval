"""Pretrained subset construction, checked loading, and Phase 2B isolation."""

import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

import evaluate
import main
from src.data_loader import load_corpus_config, load_faq_dataset, load_query_dataset
from src.preprocessing import preprocess_text
from src.pretrained_embeddings import (
    CONFIG_NAME,
    KEYS_FILE,
    METADATA_FILE,
    VECTORS_FILE,
    build_pretrained_subset,
    folded_source_keys,
    iter_word2vec_binary,
    load_pretrained_threshold,
    load_pretrained_vectors,
    token_coverage,
)
from src.word2vec_config import config_path
from src.word2vec_retrieval import answer_word2vec, build_word2vec_index


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = "query,expected_faq_id,is_answerable\n"
PHASE2B = ["tfidf", "w2v_mean", "w2v_tfidf", "pw2v_mean", "pw2v_tfidf"]

# Eight bytes of 0x20: a vector made of space bytes must not end a key early.
SPACE_VECTOR = np.frombuffer(b" " * 8, dtype="<f4")
SOURCE_ROWS = [
    ("Flipkart", [1.0, 0.0]),
    ("the", [0.0, 1.0]),
    ("The", [5.0, 5.0]),
    ("New_York", [2.0, 2.0]),
    ("refund", SPACE_VECTOR),
    ("REFUND", [9.0, 9.0]),
    ("a-b", [3.0, 3.0]),
    ("don't", [0.5, -0.5]),
    ("2024", [-1.0, 0.0]),
    ("café", [4.0, 4.0]),
]


def write_word2vec_binary(path: Path, rows, *, newline_after_vector: bool = False) -> None:
    """Write the Google News layout: key, one space, raw little-endian float32."""

    size = len(rows[0][1])
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as stream:
        stream.write(f"{len(rows)} {size}\n".encode("ascii"))
        for key, vector in rows:
            stream.write(key.encode("utf-8") + b" " + np.asarray(vector, dtype="<f4").tobytes())
            if newline_after_vector:
                stream.write(b"\n")


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class SubsetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "vectors.bin.gz"
        write_word2vec_binary(self.source, SOURCE_ROWS)
        self.models = self.root / "models"
        self.metadata = build_pretrained_subset(self.source, self.models)

    def test_lowercase_single_words_keep_their_most_frequent_casing(self) -> None:
        vectors, metadata = load_pretrained_vectors(self.models, verify_vectors=True)
        self.assertEqual(vectors.index_to_key, ["flipkart", "the", "refund", "don't", "2024"])
        np.testing.assert_array_equal(vectors["flipkart"], [1.0, 0.0])
        # 'The' and 'REFUND' come later in frequency order, so they are dropped.
        np.testing.assert_array_equal(vectors["the"], [0.0, 1.0])
        np.testing.assert_array_equal(vectors["refund"], SPACE_VECTOR)
        self.assertNotIn("new_york", vectors)
        self.assertEqual(folded_source_keys(self.models), {"flipkart": "Flipkart"})
        counts = ["source_key_count", "key_count", "rejected_by_pattern", "dropped_case_duplicates"]
        self.assertEqual([metadata[key] for key in counts], [10, 5, 3, 2])
        self.assertEqual(metadata, self.metadata)

    def test_streaming_parser_agrees_with_gensim(self) -> None:
        reference = KeyedVectors.load_word2vec_format(str(self.source), binary=True)
        plain = self.root / "plain.bin"
        write_word2vec_binary(plain, SOURCE_ROWS, newline_after_vector=True)
        for path in [self.source, plain]:
            with self.subTest(path=path.name):
                # A tiny chunk size forces every key and vector across refills.
                rows = list(iter_word2vec_binary(path, chunk_size=3))
                self.assertEqual([key for key, _ in rows], reference.index_to_key)
                for key, data in rows:
                    np.testing.assert_array_equal(np.frombuffer(data, dtype="<f4"), reference[key])

    def test_rebuilding_is_deterministic(self) -> None:
        again = build_pretrained_subset(self.source, self.models)
        self.assertEqual(again["artifact_id"], self.metadata["artifact_id"])

    def test_missing_subset_names_the_download_command(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "download_pretrained_embeddings.py"):
            load_pretrained_vectors(self.root / "missing")

    def test_altered_keys_or_metadata_are_refused(self) -> None:
        keys = self.models / "pretrained" / KEYS_FILE
        keys.write_text(
            keys.read_text(encoding="utf-8").replace("refund\t", "refind\t"),
            encoding="utf-8", newline="\n",
        )
        with self.assertRaisesRegex(ValueError, "do not match.*download_pretrained_embeddings.py"):
            load_pretrained_vectors(self.models)

        build_pretrained_subset(self.source, self.models)
        path = self.models / "pretrained" / METADATA_FILE
        path.write_text(json.dumps(dict(self.metadata, key_count=4)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale or mismatched"):
            load_pretrained_vectors(self.models)

    def test_altered_vectors_are_caught_by_full_verification(self) -> None:
        path = self.models / "pretrained" / VECTORS_FILE
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        # The cheap default check covers metadata, keys and shape only.
        load_pretrained_vectors(self.models)
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_pretrained_vectors(self.models, verify_vectors=True)

    def test_truncated_source_is_rejected(self) -> None:
        plain = self.root / "plain.bin"
        write_word2vec_binary(plain, SOURCE_ROWS)
        broken = self.root / "broken.bin"
        broken.write_bytes(plain.read_bytes()[:-3])
        with self.assertRaisesRegex(ValueError, "ended before"):
            build_pretrained_subset(broken, self.root / "other")

    def test_subset_serves_phase2_retrieval_and_skips_unknown_words(self) -> None:
        vectors, _ = load_pretrained_vectors(self.models)
        faq = pd.DataFrame({
            "id": [1, 2], "question": ["Flipkart refund", "the 2024"],
            "answer": ["one", "two"], "category": ["shop"] * 2,
            "source": ["https://example.test/faq"] * 2, "source_type": ["fixture"] * 2,
        })
        for aggregation in ["w2v_mean", "w2v_tfidf"]:
            with self.subTest(aggregation=aggregation):
                index = build_word2vec_index(faq, vectors, aggregation)
                result = answer_word2vec("FLIPKART refund cafe", faq, index, 0.99)
                self.assertTrue(result["found"])
                self.assertEqual(result["best_match"]["faq_id"], 1)
                # No token has a vector, so nothing is ranked at any threshold.
                self.assertEqual(answer_word2vec("New York cafe", faq, index, 0.0)["top_matches"], [])

    def test_coverage_counts_known_unknown_and_case_folded_tokens(self) -> None:
        vectors, _ = load_pretrained_vectors(self.models)
        report = token_coverage(["Flipkart refund policy", "zzqq"], vectors, folded_source_keys(self.models))
        self.assertEqual((report["texts"], report["token_occurrences"], report["distinct_tokens"]), (2, 4, 4))
        self.assertAlmostEqual(report["known_occurrence_rate"], 0.5)
        self.assertEqual(report["texts_without_known_token"], 1)
        self.assertEqual(report["occurrences_using_case_folded_vector"], 1)
        self.assertEqual(report["most_frequent_unknown"], [["policy", 1], ["zzqq", 1]])


class Phase2BCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        cls.models = cls.root / "models"
        cls.reports = cls.root / "reports"
        cls.corpus = cls.data / "university"
        cls.corpus.mkdir(parents=True)
        cls.manual = cls.data / "manual_evaluation"
        cls.manual.mkdir()
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

        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "5"
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train_word2vec.py"),
             "--corpus", "university", "--data-root", str(cls.data),
             "--models-root", str(cls.models)],
            capture_output=True, text=True, env=environment, timeout=180,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

        # Every fixture token gets a pretrained vector except 'a' and the
        # nonsense unanswerable words; some keys arrive capitalised.
        texts = [
            *load_faq_dataset(cls.corpus)["question"],
            *load_query_dataset(cls.corpus / "validation_queries.csv")["query"],
        ]
        tokens = sorted(
            {token for text in texts for token in preprocess_text(text).split()}
            - {"a", "volcano", "cryptocurrency", "astronomy"}
        )
        rng = np.random.default_rng(11)
        rows = [
            (token.capitalize() if position % 4 == 0 else token, rng.normal(size=8))
            for position, token in enumerate(tokens)
        ]
        rows.append(("New_York", rng.normal(size=8)))
        source = cls.root / "source.bin.gz"
        write_word2vec_binary(source, rows)
        cls.pretrained_metadata = build_pretrained_subset(source, cls.models)

    def setUp(self) -> None:
        for attr, value in [("DATA_ROOT", self.data), ("REPORTS_DIR", self.reports),
                            ("MODELS_ROOT", self.models)]:
            patcher = patch.object(evaluate, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        for name in ["word2vec_config.json", CONFIG_NAME]:
            (self.corpus / name).unlink(missing_ok=True)
        shutil.rmtree(self.reports, ignore_errors=True)
        self.reports.mkdir()
        (self.manual / "university_queries.csv").write_text(HEADER, encoding="utf-8")

        # Frozen Phase 1 and Phase 2 outputs that no Phase 2B command may touch.
        (self.reports / "university_evaluation.json").write_text("existing phase 1 report", encoding="utf-8")
        quietly(evaluate.run_tuning, "university", "all")
        quietly(evaluate.run_testing, "university", "all")
        self.frozen = {path: path.read_bytes() for path in self.frozen_files()}

    def frozen_files(self) -> set[Path]:
        earlier = {
            path for path in self.reports.rglob("*")
            if path.is_file() and "phase2b" not in path.parts
        }
        return earlier | {self.corpus / "corpus_config.json", self.corpus / "word2vec_config.json"}

    def assert_frozen_unchanged(self) -> None:
        self.assertEqual(self.frozen_files(), set(self.frozen))
        for path, content in self.frozen.items():
            self.assertEqual(path.read_bytes(), content, path)

    def test_phase2b_tuning_touches_only_pretrained_thresholds(self) -> None:
        with patch.object(evaluate, "tune_tfidf") as tfidf, \
                patch.object(evaluate, "tune_word2vec") as custom:
            quietly(evaluate.run_tuning, "university", "phase2b")
        tfidf.assert_not_called()
        custom.assert_not_called()
        self.assert_frozen_unchanged()

        config = json.loads(config_path(self.corpus, CONFIG_NAME).read_text(encoding="utf-8"))
        self.assertEqual(config["artifact_id"], self.pretrained_metadata["artifact_id"])
        for model in ["pw2v_mean", "pw2v_tfidf"]:
            with self.subTest(model=model):
                sweep = json.loads(
                    (self.reports / "phase2b" / f"university_{model}_tuning.json").read_text()
                )
                self.assertEqual(len(sweep["sweep"]), 101)
                self.assertEqual(sweep["selected_threshold"], config[model]["similarity_threshold"])
                self.assertEqual(
                    load_pretrained_threshold(self.corpus, self.pretrained_metadata, model),
                    config[model]["similarity_threshold"],
                )

    def test_missing_or_stale_thresholds_name_the_phase2b_command(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "--model phase2b"):
            evaluate.run_testing("university", "pw2v_mean")
        quietly(evaluate.run_tuning, "university", "phase2b")
        path = config_path(self.corpus, CONFIG_NAME)
        config = json.loads(path.read_text(encoding="utf-8"))
        config["artifact_id"] = "0" * 64
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different model artifact.*--model phase2b"):
            evaluate.run_testing("university", "pw2v_mean")
        self.assert_frozen_unchanged()

    def test_five_models_are_compared_into_phase2b_only(self) -> None:
        quietly(evaluate.run_tuning, "university", "phase2b")
        reports = quietly(evaluate.run_testing, "university", "phase2b")

        self.assertEqual(list(reports["university"]), PHASE2B)
        self.assert_frozen_unchanged()
        phase2b = self.reports / "phase2b"
        for model in PHASE2B:
            with self.subTest(model=model):
                saved = json.loads((phase2b / f"university_{model}_evaluation.json").read_text())
                self.assertEqual(saved, reports["university"][model])

        test = load_query_dataset(self.corpus / "test_queries.csv")
        paired = pd.read_csv(phase2b / "university_model_comparison.csv")
        self.assertEqual(list(paired["query"]), list(test["query"]))
        for model in PHASE2B:
            self.assertIn(f"{model}_correct_answer", paired.columns)

        text = (phase2b / "comparison_report.md").read_text(encoding="utf-8")
        self.assertIn("# Phase 2B Model Comparison (Synthetic Benchmark)", text)
        self.assertIn("### Pretrained Word2Vec mean: case A", text)
        self.assertIn("### Pretrained Word2Vec TF-IDF weighted: case C", text)
        # Phase 2 already narrates the custom models, so they are not repeated here.
        self.assertNotIn("### Word2Vec mean: improvements over TF-IDF", text)

        cases = pd.read_csv(phase2b / "university_error_cases.csv")
        self.assertEqual(list(cases.columns[:5]), ["case", "model", "custom_model", "query", "expected_faq_id"])
        delivered = {model: paired[f"{model}_correct_answer"].to_numpy(dtype=bool) for model in PHASE2B}
        for model, custom in [("pw2v_mean", "w2v_mean"), ("pw2v_tfidf", "w2v_tfidf")]:
            with self.subTest(model=model):
                expected = {
                    "A": int((delivered[model] & ~delivered["tfidf"] & ~delivered[custom]).sum()),
                    "B": int((delivered["tfidf"] & ~delivered[model]).sum()),
                    "C": int((delivered[custom] & ~delivered[model]).sum()),
                }
                counts = cases.loc[cases["model"] == model, "case"].value_counts().to_dict()
                self.assertEqual({case: counts.get(case, 0) for case in expected}, expected)

        coverage = json.loads((phase2b / "university_pretrained_coverage.json").read_text())
        faq_coverage = coverage["sets"]["faq_questions"]
        self.assertLess(faq_coverage["pretrained"]["known_occurrence_rate"], 1.0)
        self.assertIn("a", [token for token, _ in faq_coverage["pretrained"]["most_frequent_unknown"]])
        self.assertEqual(faq_coverage["custom_word2vec"]["known_distinct_rate"], 1.0)
        self.assertGreaterEqual(
            coverage["sets"]["test_queries"]["pretrained"]["texts_without_known_token"], 1
        )

    def test_single_model_writes_to_phase2b_and_manual_stays_pending(self) -> None:
        quietly(evaluate.run_tuning, "university", "phase2b")
        quietly(evaluate.run_testing, "university", "pw2v_tfidf")
        self.assertTrue((self.reports / "phase2b" / "university_pw2v_tfidf_evaluation.json").is_file())

        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(evaluate, "tune_pretrained") as tune:
            reports = evaluate.run_testing("university", "phase2b", manual=True)
        self.assertEqual(reports, {})
        self.assertEqual(output.getvalue().count("human evaluation pending"), 1)
        self.assertEqual(list((self.reports / "phase2b").glob("manual_*")), [])
        tune.assert_not_called()
        self.assert_frozen_unchanged()

    def test_group_names_and_unknown_models(self) -> None:
        self.assertEqual(evaluate.selected_models("phase2b"), PHASE2B)
        self.assertEqual(evaluate.selected_models("all"), PHASE2B[:3])
        with self.assertRaisesRegex(SystemExit, "Unknown model"):
            evaluate.selected_models("glove")

    def test_terminal_answerer_uses_the_frozen_pretrained_threshold(self) -> None:
        quietly(evaluate.run_tuning, "university", "phase2b")
        faq = load_faq_dataset(self.corpus)
        config = load_corpus_config(self.corpus)
        with patch.object(main, "MODELS_ROOT", self.models):
            answer, threshold, preprocessing = main.build_answerer(self.corpus, faq, config, "pw2v_mean")
        self.assertEqual(threshold, load_pretrained_threshold(self.corpus, self.pretrained_metadata, "pw2v_mean"))
        self.assertEqual(preprocessing, "basic")
        # An exact stored question ranks itself first.
        self.assertEqual(answer(faq["question"].iloc[2])["top_matches"][0]["faq_id"], faq["id"].iloc[2])
        self.assertEqual(answer("volcano cryptocurrency astronomy")["top_matches"], [])


if __name__ == "__main__":
    unittest.main()
