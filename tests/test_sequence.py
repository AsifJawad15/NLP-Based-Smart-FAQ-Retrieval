"""Phase 3: sequence inputs, Siamese encoders, independent training data and isolation."""

import ast
import contextlib
import gzip
import importlib.util
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
import torch

import evaluate
import main
from src.data_loader import load_corpus_config, load_faq_dataset, load_query_dataset, validate_faq_data
from src.evaluation import rank_queries
from src.pretrained_embeddings import build_pretrained_subset, load_pretrained_vectors
from src.sequence_data import (
    DUPLICATE_SIMILARITY,
    audit_training_data,
    build_pairs,
    choose_max_len,
    fit_to_question,
    generate_paraphrases,
    normalized,
    protected_tokens,
    synonym_substitutes,
    synonyms,
    template_openings,
    tokens,
)
from src.sequence_models import (
    PAD_ID,
    UNK_ID,
    SequenceEncoder,
    SiameseMatcher,
    build_vocabulary,
    embedding_matrix,
    pad_batch,
    token_ids,
)
from src.sequence_retrieval import answer_sequence, build_sequence_index, rank_sequence_queries
from src.sequence_training import (
    PAIRS_FILE,
    config_name,
    load_sequence_model,
    load_sequence_threshold,
    save_sequence_config,
)
from src.tfidf_retrieval import build_tfidf_index


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = "query,expected_faq_id,is_answerable\n"
BASIC = {"remove_stopwords": False, "lemmatize": False}
PHASE3 = ["tfidf", "w2v_mean", "w2v_tfidf", "pw2v_mean", "pw2v_tfidf", "rnn", "bilstm"]
NONSENSE = {"volcano", "cryptocurrency", "astronomy"}


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def toy_vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=4)
    vectors.add_vectors(
        ["reset", "password", "portal", "course", "student"],
        np.eye(5, 4, dtype=np.float32) + 0.1,
    )
    return vectors


def fixture_rows(texts, seed: int = 11):
    """A vector for every fixture word, plus close vectors for up to two WordNet synonyms of each."""

    rng = np.random.default_rng(seed)
    words = sorted({token for text in texts for token in tokens(text)} - NONSENSE)
    base = {word: rng.normal(size=8) for word in words}
    rows = [(word, base[word]) for word in words]
    keys = set(words)
    for word in words:
        options = sorted(candidate for candidate in synonyms(word) if candidate.isalpha() and candidate not in keys)
        for synonym in options[:2]:
            rows.append((synonym, base[word] + rng.normal(scale=0.05, size=8)))
            keys.add(synonym)
    return rows


def fixture_vectors(texts) -> KeyedVectors:
    rows = fixture_rows(texts)
    vectors = KeyedVectors(vector_size=8)
    vectors.add_vectors([key for key, _ in rows], np.array([vector for _, vector in rows], dtype=np.float32))
    return vectors


def write_word2vec_binary(path: Path, rows) -> None:
    with gzip.open(path, "wb") as stream:
        stream.write(f"{len(rows)} {len(rows[0][1])}\n".encode("ascii"))
        for key, vector in rows:
            stream.write(key.encode("utf-8") + b" " + np.asarray(vector, dtype="<f4").tobytes())


def run_script(script: str, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "5"  # Training scripts must relaunch with 42.
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / script), *arguments],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class VocabularyAndInputTests(unittest.TestCase):
    def test_vocabulary_keeps_frequent_words_and_known_training_words(self) -> None:
        vectors = toy_vectors()
        vocabulary = build_vocabulary(vectors, ["Reset my password", "zzqq course"], limit=2)
        self.assertEqual(vocabulary, ["<PAD>", "<UNK>", "reset", "password", "course"])
        matrix = embedding_matrix(vocabulary, vectors)
        self.assertEqual(tuple(matrix.shape), (5, 4))
        self.assertTrue(torch.equal(matrix[PAD_ID], torch.zeros(4)))
        self.assertTrue(torch.equal(matrix[UNK_ID], torch.zeros(4)))
        np.testing.assert_allclose(matrix[4].numpy(), vectors["course"])

    def test_unknown_words_map_to_unk_and_long_questions_truncate(self) -> None:
        index = {word: position for position, word in enumerate(["<PAD>", "<UNK>", "reset", "password"])}
        self.assertEqual(token_ids("Reset my PASSWORD now", index, max_len=3), [2, UNK_ID, 3])
        self.assertEqual(token_ids("", index, max_len=3), [])

    def test_padding_uses_pad_id_and_true_lengths(self) -> None:
        ids, lengths = pad_batch([[2, 3], [4]])
        self.assertEqual(ids.tolist(), [[2, 3], [4, PAD_ID]])
        self.assertEqual(lengths.tolist(), [2, 1])
        with self.assertRaisesRegex(ValueError, "at least one token"):
            pad_batch([[2], []])

    def test_max_len_covers_the_99th_percentile_of_lengths(self) -> None:
        texts = ["reset password"] * 99 + ["a b c d e f g h i j"]
        expected = int(np.ceil(np.percentile([2] * 99 + [10], 99)))
        self.assertEqual(choose_max_len(texts), expected)


class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.weights = torch.randn(6, 5)
        self.weights[PAD_ID] = 0

    def test_output_dimensions(self) -> None:
        for architecture, size in [("rnn", 128), ("bilstm", 256)]:
            with self.subTest(architecture=architecture):
                encoder = SequenceEncoder(self.weights, architecture)
                self.assertEqual(tuple(encoder(*pad_batch([[2, 3, 4], [5]])).shape), (2, size))
                self.assertEqual(encoder.output_dim, size)
        with self.assertRaisesRegex(ValueError, "architecture"):
            SequenceEncoder(self.weights, "gru")

    def test_packing_makes_padding_irrelevant(self) -> None:
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                encoder = SequenceEncoder(self.weights, architecture).eval()
                with torch.no_grad():
                    alone = encoder(*pad_batch([[2, 3]]))
                    padded = encoder(*pad_batch([[2, 3], [4, 5, 2, 3, 4]]))
                torch.testing.assert_close(alone[0], padded[0])

    def test_word_order_changes_sequence_vectors(self) -> None:
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                encoder = SequenceEncoder(self.weights, architecture).eval()
                with torch.no_grad():
                    forward, backward = encoder(*pad_batch([[2, 3, 4], [4, 3, 2]]))
                self.assertFalse(torch.allclose(forward, backward))

    def test_embeddings_stay_frozen_while_recurrent_weights_learn(self) -> None:
        model = SiameseMatcher(SequenceEncoder(self.weights.clone(), "bilstm"))
        embedding_before = model.encoder.embedding.weight.detach().clone()
        recurrent_before = model.encoder.recurrent.weight_ih_l0.detach().clone()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
        logits = model(*pad_batch([[2, 3], [4]]), *pad_batch([[3, 2], [5, 4]]))
        torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0])).backward()
        optimizer.step()
        self.assertFalse(model.encoder.embedding.weight.requires_grad)
        self.assertTrue(torch.equal(model.encoder.embedding.weight, embedding_before))
        self.assertFalse(torch.equal(model.encoder.recurrent.weight_ih_l0, recurrent_before))

    def test_one_shared_encoder_scores_identical_texts_at_scale_plus_bias(self) -> None:
        model = SiameseMatcher(SequenceEncoder(self.weights, "rnn")).eval()
        ids, lengths = pad_batch([[2, 3, 4]])
        with torch.no_grad():
            logit = model(ids, lengths, ids, lengths)
        torch.testing.assert_close(logit, torch.tensor([5.0]))
        self.assertEqual(sum(isinstance(module, SequenceEncoder) for module in model.modules()), 1)


class ParaphraseAndPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.faq = validate_faq_data(pd.read_csv(FIXTURES / "university_sample.csv"))
        self.vectors = fixture_vectors(self.faq["question"])
        words = {token for question in self.faq["question"] for token in tokens(question)}
        self.neighbours = synonym_substitutes(words, self.vectors)

    def test_substitutes_are_close_wordnet_synonyms_never_protected_or_banned(self) -> None:
        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(
            ["purchase", "buy", "sell", "leverage", "purchased", "more", "additional"],
            np.array([[1, 0], [0.9, 0.2], [1, 0.05], [0, 1], [1, 0.01], [0.3, 1], [0.3, 1]],
                     dtype=np.float32),
        )
        # 'sell' is close but not a synonym, 'leverage' is a synonym but far,
        # 'purchased' is an inflection, and 'more' is a protected polarity word.
        self.assertEqual(synonym_substitutes(["purchase", "more"], vectors), {"purchase": ["buy"]})
        self.assertEqual(synonym_substitutes(["purchase"], vectors, [("purchase", "buy")]), {})

    def test_context_rejects_a_synonym_from_the_wrong_sense(self) -> None:
        vectors = KeyedVectors(vector_size=3)
        vectors.add_vectors(
            ["quickly", "battery", "charge", "power", "accusation"],
            np.array([[0.5, 0.5, 0], [1, 0, 0], [1, 0.2, 0], [1, 0.1, 0], [0, 0, 1]], dtype=np.float32),
        )
        local = fit_to_question(
            ["how", "quickly", "does", "battery", "charge"], {"charge": ["accusation", "power"]}, vectors
        )
        self.assertEqual(local, {"charge": ["power"]})

    def test_quantities_and_polarity_words_are_never_substituted(self) -> None:
        faq = pd.DataFrame({
            "id": [1], "question": ["can i return more than one item within seven days"],
            "answer": ["x"], "category": ["c"], "source": ["s"], "source_type": ["t"],
        })
        substitutes = {"return": ["repay"], "more": ["additional"], "one": ["single"],
                       "item": ["product"], "seven": ["eight"], "days": ["years"], "within": ["inside"]}
        for query in generate_paraphrases(faq, substitutes)["query"]:
            words = set(tokens(query))
            self.assertTrue({"more", "one", "seven", "days"} <= words)
            self.assertFalse({"additional", "single", "eight", "years"} & words)

    def test_three_distinct_deterministic_paraphrases_per_faq(self) -> None:
        first = generate_paraphrases(self.faq, self.neighbours)
        self.assertTrue(first.equals(generate_paraphrases(self.faq, self.neighbours)))
        self.assertEqual(len(first), 3 * len(self.faq))
        questions = self.faq.set_index("id")["question"]
        for faq_id, group in first.groupby("expected_faq_id"):
            with self.subTest(faq_id=faq_id):
                self.assertEqual(sorted(group["split"]), ["dev", "train", "train"])
                texts = [normalized(query) for query in group["query"]]
                self.assertNotIn(normalized(questions[faq_id]), texts)
                self.assertEqual(len(set(texts)), 3)
        for row in first.itertuples():
            for pair in row.substitutions.split():
                original, replacement = pair.split(">")
                self.assertIn(replacement, tokens(row.query))
                self.assertIn(replacement, self.neighbours[original])

    def test_capitalised_names_are_never_substituted(self) -> None:
        faq = pd.DataFrame({
            "id": [1], "question": ["Can I use Flipkart Pay Later for EMI?"], "answer": ["x"],
            "category": ["c"], "source": ["s"], "source_type": ["t"],
        })
        self.assertEqual(protected_tokens(faq["question"][0]), {"i", "flipkart", "pay", "later", "emi"})
        neighbours = {"flipkart": ["amazon"], "pay": ["spend"], "later": ["afterwards"],
                      "emi": ["loan"], "use": ["utilise"]}
        for query in generate_paraphrases(faq, neighbours)["query"]:
            words = set(tokens(query))
            self.assertTrue({"flipkart", "pay", "later", "emi"} <= words)
            self.assertFalse({"amazon", "spend", "afterwards", "loan"} & words)

    def test_the_opening_word_is_never_reworded_or_dropped(self) -> None:
        faq = pd.DataFrame({
            "id": [1], "question": ["describe the course structure"], "answer": ["x"],
            "category": ["c"], "source": ["s"], "source_type": ["t"],
        })
        neighbours = {"describe": ["explain"], "course": ["module"], "structure": ["layout"]}
        for query in generate_paraphrases(faq, neighbours)["query"]:
            words = tokens(query)
            self.assertIn("describe", words)
            self.assertNotIn("explain", words)

    def test_avoided_openings_are_never_introduced(self) -> None:
        faq = pd.DataFrame({
            "id": [1], "question": ["Am I able to defer my offer of admission?"], "answer": ["x"],
            "category": ["c"], "source": ["s"], "source_type": ["t"],
        })
        neighbours = {"able": ["allowed"], "defer": ["postpone"], "offer": ["proposal"], "admission": ["entry"]}
        paraphrases = generate_paraphrases(faq, neighbours, avoid_openings=["am i allowed to"])
        for query in paraphrases["query"]:
            self.assertNotEqual(tokens(query)[:4], ["am", "i", "allowed", "to"])

    def test_pairs_use_gold_positives_and_never_near_duplicate_negatives(self) -> None:
        paraphrases = generate_paraphrases(self.faq, self.neighbours)
        pairs = build_pairs(self.faq, paraphrases)
        self.assertEqual(len(pairs), 3 * len(paraphrases))
        self.assertEqual(int((pairs["pair_type"] == "positive").sum()), len(paraphrases))
        vectorizer, matrix = build_tfidf_index(self.faq, BASIC)
        similarity = (matrix @ matrix.T).toarray()
        position = {int(faq_id): index for index, faq_id in enumerate(self.faq["id"])}
        questions = self.faq.set_index("id")["question"]
        for row in pairs.itertuples():
            if row.label == 1:
                self.assertEqual(row.text_b, questions[row.faq_id])
                self.assertEqual(row.candidate_faq_id, row.faq_id)
            else:
                self.assertLess(similarity[position[row.faq_id], position[row.candidate_faq_id]],
                                DUPLICATE_SIMILARITY)

        # The hard negative is the best-ranked allowed TF-IDF candidate.
        ranking = rank_queries(paraphrases["query"], self.faq, vectorizer, matrix, BASIC, 10)
        third = pairs[~pairs["pair_type"].isin(["positive", "random"])].reset_index(drop=True)
        for row_number, faq_id in enumerate(paraphrases["expected_faq_id"]):
            if third["pair_type"][row_number] != "hard":
                continue
            gold = position[int(faq_id)]
            expected = next(
                int(candidate) for candidate in ranking["ranked_ids"][row_number]
                if similarity[gold, position[int(candidate)]] < DUPLICATE_SIMILARITY
            )
            self.assertEqual(third["candidate_faq_id"][row_number], expected)


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.faq = pd.DataFrame({
            "id": [1, 2], "question": ["How do I reset my portal password?", "When does registration begin?"],
            "answer": ["a", "b"], "category": ["c"] * 2, "source": ["s"] * 2, "source_type": ["t"] * 2,
        })
        self.paraphrases = pd.DataFrame({
            "query": ["Hi, how do i change my gateway password?",
                      "Quick question: when does enrolment start?",
                      "Help needed: when does registration open?"],
            "expected_faq_id": [1, 2, 2],
            "split": ["train", "train", "dev"],
            "substitutions": ["reset>change portal>gateway", "registration>enrolment begin>start", "begin>open"],
        })
        self.evaluation = {"test": pd.DataFrame({
            "query": ["What is the capital of France?"],
            "expected_faq_id": pd.array([pd.NA], dtype="Int64"), "is_answerable": [False],
        })}

    def audit(self, paraphrases=None, evaluation=None, openings=(), banned=()):
        return audit_training_data(
            self.faq, self.paraphrases if paraphrases is None else paraphrases,
            self.evaluation if evaluation is None else evaluation, openings, banned,
        )

    def test_clean_rows_pass(self) -> None:
        kept, report = self.audit()
        self.assertTrue(report["passed"])
        self.assertEqual(len(kept), 3)
        self.assertEqual(report["kept"], {"train": 2, "dev": 1})

    def test_injected_evaluation_queries_are_removed_and_counted(self) -> None:
        evaluation = {"test": pd.DataFrame({
            "query": ["Quick question: when does enrolment start?",
                      "hi how do i change my gateway password today"],
            "expected_faq_id": pd.array([2, 1], dtype="Int64"), "is_answerable": [True, True],
        })}
        kept, report = self.audit(evaluation=evaluation)
        self.assertEqual(report["removed_exact_evaluation_match"], 1)
        self.assertEqual(report["removed_high_overlap"], 1)
        self.assertEqual(list(kept["query"]), ["Help needed: when does registration open?"])

    def test_template_openings_banned_substitutions_or_copying_fail(self) -> None:
        _kept, report = self.audit(openings=template_openings(["Quick question: {0}.", "How is {0} organised?"]))
        self.assertEqual(report["introduced_template_openings"], 1)
        self.assertFalse(report["passed"])

        _kept, report = self.audit(banned={("reset", "change")})
        self.assertEqual(report["banned_substitutions_used"], 1)
        self.assertFalse(report["passed"])

        copied = self.paraphrases.assign(query=[
            "How do I reset my portal password now?",
            "When does registration begin today?",
            "When does registration begin at all?",
        ])
        _kept, report = self.audit(paraphrases=copied)
        self.assertGreater(report["mean_source_token_coverage"], 0.8)
        self.assertFalse(report["passed"])

    def test_training_data_code_never_uses_the_evaluation_generator(self) -> None:
        tree = ast.parse((ROOT / "src" / "sequence_data.py").read_text(encoding="utf-8"))
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        modules += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(any("prepare_datasets" in module for module in modules))
        self.assertFalse(names & {"CONTENT_SYNONYMS", "PARAPHRASE_PATTERNS",
                                  "STRUCTURAL_FALLBACKS", "break_verbatim_reuse"})

    def test_audit_rules_match_the_evaluation_generator(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "build_sequence_pairs", ROOT / "scripts" / "build_sequence_pairs.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        source = (ROOT / "scripts" / "prepare_datasets.py").read_text(encoding="utf-8")
        for opening in builder.REBUILT_OPENINGS:
            self.assertIn(opening.replace("{0}", "{body}"), source)
        templates, banned = builder.evaluation_generator_rules(ROOT / "scripts" / "prepare_datasets.py")
        self.assertIn("Explain how this works: {0}.", templates)
        self.assertTrue({("warranty", "guarantee"), ("guarantee", "warranty"), ("return", "send")} <= banned)


class SequenceTrainingTests(unittest.TestCase):
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
        (cls.data / "manual_evaluation").mkdir()
        (cls.data / "manual_evaluation" / "university_queries.csv").write_text(HEADER, encoding="utf-8")
        shutil.copyfile(FIXTURES / "university_sample.csv", cls.corpus / "faq_dataset.csv")
        for name in ["validation_queries.csv", "test_queries.csv"]:
            shutil.copyfile(FIXTURES / "university_validation_sample.csv", cls.corpus / name)
        (cls.corpus / "corpus_config.json").write_text(json.dumps({
            "display_name": "University Fixture", "remove_stopwords": False, "lemmatize": False,
            "similarity_threshold": 0.3, "preprocessing_config": "basic",
        }), encoding="utf-8")

        texts = [*load_faq_dataset(cls.corpus)["question"],
                 *load_query_dataset(cls.corpus / "validation_queries.csv")["query"]]
        source = cls.root / "source.bin.gz"
        write_word2vec_binary(source, fixture_rows(texts))
        build_pretrained_subset(source, cls.models)

        roots = ["--data-root", str(cls.data), "--models-root", str(cls.models)]
        run_script("train_word2vec.py", "--corpus", "university", *roots)
        run_script("build_sequence_pairs.py", "--corpus", "university", *roots,
                   "--reports-root", str(cls.reports))
        run_script("train_sequence_models.py", "--corpus", "university", *roots,
                   "--reports-root", str(cls.reports))
        cls.pretrained = load_pretrained_vectors(cls.models)

    def metadata(self, architecture: str, models_root: Path | None = None) -> dict:
        path = (models_root or self.models) / "university" / f"{architecture}_metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_training_selects_the_best_dev_epoch_and_saves_history(self) -> None:
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                metadata = self.metadata(architecture)
                self.assertTrue((self.models / "university" / f"{architecture}.pt").is_file())
                self.assertTrue(metadata["training_settings"]["freeze_embeddings"])
                history = pd.read_csv(
                    self.reports / "phase3" / f"university_{architecture}_training_history.csv"
                )
                self.assertEqual(len(history), metadata["epochs_run"])
                keys = list(zip(history["dev_top1"], history["dev_top3"], -history["dev_loss"]))
                self.assertEqual(max(keys), keys[metadata["best_epoch"] - 1])
        audit = json.loads((self.reports / "phase3" / "university_training_data_audit.json").read_text())
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["introduced_template_openings"], 0)
        self.assertEqual(audit["banned_substitutions_used"], 0)

    def test_fresh_process_training_is_reproducible(self) -> None:
        second = self.root / "repeated_models"
        shutil.copytree(self.models / "pretrained", second / "pretrained")
        run_script("train_sequence_models.py", "--corpus", "university", "--data-root", str(self.data),
                   "--models-root", str(second), "--reports-root", str(self.root / "repeated_reports"))
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                first, repeated = self.metadata(architecture), self.metadata(architecture, second)
                self.assertEqual(first["state_sha256"], repeated["state_sha256"])
                self.assertEqual(first["artifact_id"], repeated["artifact_id"])

    def test_loaded_encoders_rank_exact_questions_and_reject_unknown_words(self) -> None:
        faq = load_faq_dataset(self.corpus)
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                encoder, vocabulary_index, metadata = load_sequence_model(
                    self.corpus, architecture, self.pretrained, self.models
                )
                index = build_sequence_index(faq, encoder, vocabulary_index, metadata["max_len"])
                ranking = rank_sequence_queries(faq["question"], faq, index, top_k=3)
                np.testing.assert_allclose(ranking["ranked_scores"][:, 0], 1.0, atol=1e-6)
                for row, faq_id in enumerate(faq["id"]):
                    self.assertIn(faq_id, ranking["ranked_ids"][row])
                for query in ["", "!!!", "volcano cryptocurrency astronomy"]:
                    result = answer_sequence(query, faq, index, 0.0)
                    self.assertFalse(result["found"])
                    self.assertEqual(result["top_matches"], [])

    def test_runtime_answers_agree_with_batch_ranking(self) -> None:
        faq = load_faq_dataset(self.corpus)
        queries = load_query_dataset(self.corpus / "validation_queries.csv")
        encoder, vocabulary_index, metadata = load_sequence_model(self.corpus, "bilstm", self.pretrained, self.models)
        index = build_sequence_index(faq, encoder, vocabulary_index, metadata["max_len"])
        ranking = rank_sequence_queries(queries["query"], faq, index)
        for row, query in enumerate(queries["query"]):
            with self.subTest(query=query):
                result = answer_sequence(query, faq, index, 0.5)
                expected = bool(ranking["has_features"][row] and ranking["ranked_scores"][row, 0] >= 0.5)
                self.assertEqual(result["found"], expected)
                if result["top_matches"]:
                    self.assertEqual(result["top_matches"][0]["faq_id"], ranking["ranked_ids"][row, 0])
        nonsense = list(queries["query"]).index("volcano cryptocurrency astronomy")
        self.assertFalse(ranking["has_features"][nonsense])

    def test_missing_or_stale_models_and_thresholds_are_refused(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "train_sequence_models.py"):
            load_sequence_model(self.corpus, "rnn", self.pretrained, self.root / "missing")
        altered = self.root / "altered" / "university"
        shutil.copytree(self.corpus, altered)
        with (altered / PAIRS_FILE).open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Stale or mismatched.*train_sequence_models.py"):
            load_sequence_model(altered, "rnn", self.pretrained, self.models)

        _encoder, _index, metadata = load_sequence_model(self.corpus, "rnn", self.pretrained, self.models)
        directory = self.root / "thresholds" / "university"
        directory.mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "--model phase3"):
            load_sequence_threshold(directory, metadata, "rnn")
        save_sequence_config(directory, "rnn", "0" * 64,
                             {"rnn": {"similarity_threshold": 0.7, "validation_score": 0.8}})
        self.assertTrue((directory / config_name("rnn")).is_file())
        with self.assertRaisesRegex(ValueError, "different model artifact.*--model phase3"):
            load_sequence_threshold(directory, metadata, "rnn")

    def test_phase3_group_tunes_only_sequence_models_and_reports_into_phase3(self) -> None:
        reports = self.root / "command_reports"
        with patch.object(evaluate, "DATA_ROOT", self.data), \
                patch.object(evaluate, "REPORTS_DIR", reports), \
                patch.object(evaluate, "MODELS_ROOT", self.models):
            quietly(evaluate.run_tuning, "university", "all")
            quietly(evaluate.run_tuning, "university", "phase2b")
            quietly(evaluate.run_testing, "university", "phase2b")
            earlier = [path for path in reports.rglob("*") if path.is_file()]
            earlier += [self.corpus / name for name in
                        ["corpus_config.json", "word2vec_config.json", "pretrained_config.json"]]
            frozen = {path: path.read_bytes() for path in earlier}
            with patch.object(evaluate, "tune_tfidf") as tfidf, \
                    patch.object(evaluate, "tune_word2vec") as custom, \
                    patch.object(evaluate, "tune_pretrained") as pretrained:
                quietly(evaluate.run_tuning, "university", "phase3")
            tfidf.assert_not_called()
            custom.assert_not_called()
            pretrained.assert_not_called()
            results = quietly(evaluate.run_testing, "university", "phase3")

        self.assertEqual(evaluate.selected_models("phase3"), PHASE3)
        self.assertEqual(list(results["university"]), PHASE3)
        for path, content in frozen.items():
            self.assertEqual(path.read_bytes(), content, path)
        phase3 = reports / "phase3"
        text = (phase3 / "comparison_report.md").read_text(encoding="utf-8")
        self.assertIn("# Phase 3 Model Comparison (Synthetic Benchmark)", text)
        self.assertIn("### Siamese BiLSTM: case A", text)
        self.assertIn("Human evaluation is still pending", text)
        cases = pd.read_csv(phase3 / "university_error_cases.csv")
        self.assertEqual(list(cases.columns[:3]), ["case", "model", "pretrained_mean_model"])
        self.assertTrue(set(cases["model"]) <= {"rnn", "bilstm"})
        for architecture in ["rnn", "bilstm"]:
            with self.subTest(architecture=architecture):
                self.assertTrue((self.corpus / config_name(architecture)).is_file())
                sweep = json.loads((phase3 / f"university_{architecture}_tuning.json").read_text())
                self.assertEqual(len(sweep["sweep"]), 101)
        self.assertFalse((phase3 / "university_pretrained_coverage.json").exists())

    def test_terminal_answerer_supports_sequence_models(self) -> None:
        with patch.object(evaluate, "DATA_ROOT", self.data), \
                patch.object(evaluate, "REPORTS_DIR", self.root / "terminal_reports"), \
                patch.object(evaluate, "MODELS_ROOT", self.models):
            quietly(evaluate.run_tuning, "university", "phase3")
        faq = load_faq_dataset(self.corpus)
        config = load_corpus_config(self.corpus)
        with patch.object(main, "MODELS_ROOT", self.models):
            answer, threshold, preprocessing = main.build_answerer(self.corpus, faq, config, "bilstm")
        _encoder, _index, metadata = load_sequence_model(self.corpus, "bilstm", self.pretrained, self.models)
        self.assertEqual(threshold, load_sequence_threshold(self.corpus, metadata, "bilstm"))
        self.assertEqual(preprocessing, "basic")
        result = answer(faq["question"].iloc[0])
        self.assertIn(int(faq["id"].iloc[0]), [match["faq_id"] for match in result["top_matches"]])
        self.assertIn("bilstm", dict(main.MODELS))


if __name__ == "__main__":
    unittest.main()
