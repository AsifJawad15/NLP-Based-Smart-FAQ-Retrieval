"""Terminal demonstration for the corpus-configurable Smart FAQ system.

Examples:
    python main.py
    python main.py --corpus university --query "How do I request a transcript?"
    python main.py --corpus university --model w2v_mean --query "How do I receive university alerts?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from src.data_loader import discover_corpora, load_corpus_config, load_faq_dataset
from src.tfidf_retrieval import answer_query, build_tfidf_index
from src.word2vec_config import load_word2vec_threshold
from src.word2vec_retrieval import answer_word2vec, build_word2vec_index
from src.word2vec_training import load_word2vec


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

# Ordered as they appear in the Phase 2 comparison.
MODELS = [
    ("tfidf", "TF-IDF, the Phase 1 baseline"),
    ("w2v_mean", "Custom Word2Vec, mean vectors"),
    ("w2v_tfidf", "Custom Word2Vec, TF-IDF weighted vectors"),
]

# A Windows console defaults to cp1252, which cannot print every character an
# FAQ answer may contain. Replacing unprintable characters keeps the
# demonstration running instead of raising UnicodeEncodeError mid-answer.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def choose_from_menu(labels: list[str], title: str, prompt: str) -> int:
    """Show a numbered menu and return the selected position."""

    print(title)
    for number, label in enumerate(labels, start=1):
        print(f"  {number}. {label}")

    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            # No interactive terminal, so fall back to the first option.
            print("\nNo input available; using the first option.")
            return 0
        if choice.isdigit() and 1 <= int(choice) <= len(labels):
            return int(choice) - 1
        print("Please enter one of the displayed numbers.")


def choose_corpus(corpora: dict[str, Path]) -> tuple[str, Path]:
    """Show a numbered corpus menu and return the selected item."""

    items = list(corpora.items())
    labels = [
        f"{load_corpus_config(path)['display_name']} ({key})" for key, path in items
    ]
    return items[choose_from_menu(labels, "Available corpora:", "Select a corpus number: ")]


def choose_model() -> str:
    """Show a numbered retrieval-model menu and return the selected key."""

    labels = [description for _key, description in MODELS]
    position = choose_from_menu(
        labels, "\nAvailable retrieval models:", "Select a model number: "
    )
    return MODELS[position][0]


def build_answerer(
    corpus_dir: Path, faq_data, config: dict[str, Any], model: str
) -> tuple[Callable[[str], dict[str, Any]], float, str]:
    """Build one model's index once and return how to answer with it.

    A missing or stale Word2Vec artifact raises with its training instruction
    rather than training silently during a demonstration.
    """

    if model == "tfidf":
        options = {
            "remove_stopwords": bool(config["remove_stopwords"]),
            "lemmatize": bool(config["lemmatize"]),
        }
        vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
        threshold = float(config["similarity_threshold"])

        def answer(query: str) -> dict[str, Any]:
            return answer_query(
                query, faq_data, vectorizer, faq_matrix,
                threshold=threshold, top_k=3, preprocessing_options=options,
            )

        return answer, threshold, str(config.get("preprocessing_config", "basic"))

    trained, metadata = load_word2vec(corpus_dir)
    threshold = load_word2vec_threshold(corpus_dir, metadata, model)
    index = build_word2vec_index(faq_data, trained.wv, model)

    def answer(query: str) -> dict[str, Any]:
        return answer_word2vec(query, faq_data, index, threshold, top_k=3)

    return answer, threshold, "basic"


def print_result(result: dict[str, object]) -> None:
    """Print one answer, or the rejection message and the best score."""

    if not result["found"]:
        matches = result["top_matches"]
        best_score = matches[0]["similarity"] if matches else 0.0
        print(f"\n{result['message']}")
        print(f"Best similarity score: {best_score:.4f}\n")
        return

    best = result["best_match"]
    print(f"\nMatched FAQ: {best['question']}")
    print(f"Similarity Score: {best['similarity']:.4f}")
    print(f"Answer: {best['answer']}")
    print("\nTop matches:")
    for rank, match in enumerate(result["top_matches"], start=1):
        print(f"  {rank}. {match['question']} - {match['similarity']:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    """Parse the optional non-interactive demonstration arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="Corpus key, for example 'university'")
    parser.add_argument(
        "--model",
        choices=[key for key, _description in MODELS],
        help="Retrieval model; interactive mode asks when this is omitted",
    )
    parser.add_argument("--query", help="Answer one question and exit")
    return parser.parse_args()


def main() -> None:
    """Load one corpus, build its index once, and answer terminal queries."""

    arguments = parse_args()

    print("=" * 44)
    print("          SMART FAQ RETRIEVAL SYSTEM")
    print("=" * 44)

    corpora = discover_corpora(DATA_ROOT)
    if not corpora:
        raise SystemExit(f"No corpora found under {DATA_ROOT}")

    if arguments.corpus:
        if arguments.corpus not in corpora:
            raise SystemExit(
                f"Unknown corpus '{arguments.corpus}'; found {sorted(corpora)}"
            )
        corpus_dir = corpora[arguments.corpus]
    else:
        _corpus_key, corpus_dir = choose_corpus(corpora)

    # A single-shot query keeps the Phase 1 default; the menu is interactive only.
    model = arguments.model or ("tfidf" if arguments.query else choose_model())

    faq_data = load_faq_dataset(corpus_dir)
    config = load_corpus_config(corpus_dir)
    answer, threshold, preprocessing = build_answerer(corpus_dir, faq_data, config, model)
    description = dict(MODELS)[model]

    print(f"\nLoaded {len(faq_data)} FAQs from {config['display_name']}.")
    print(f"Model: {description}")
    print(f"Preprocessing: {preprocessing}, threshold: {threshold:.2f}")

    if arguments.query:
        print(f"\nAsk your question: {arguments.query}")
        print_result(answer(arguments.query))
        return

    print("Type 'exit' to close the program.\n")
    while True:
        try:
            query = input("Ask your question: ").strip()
        except EOFError:
            print("\nGoodbye!")
            return
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if not query:
            print("Please enter a question.\n")
            continue

        print_result(answer(query))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
