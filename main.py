"""Terminal demonstration for the corpus-configurable Smart FAQ system.

Examples:
    python main.py
    python main.py --corpus university --query "How do I request a transcript?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import discover_corpora, load_corpus_config, load_faq_dataset
from src.tfidf_retrieval import answer_query, build_tfidf_index


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"


def choose_corpus(corpora: dict[str, Path]) -> tuple[str, Path]:
    """Show a numbered corpus menu and return the selected item."""

    items = list(corpora.items())
    print("Available corpora:")
    for number, (key, path) in enumerate(items, start=1):
        config = load_corpus_config(path)
        print(f"  {number}. {config['display_name']} ({key})")

    while True:
        try:
            choice = input("Select a corpus number: ").strip()
        except EOFError:
            # No interactive terminal, so fall back to the first corpus.
            print("\nNo input available; using the first corpus.")
            return items[0]
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1]
        print("Please enter one of the displayed numbers.")


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

    faq_data = load_faq_dataset(corpus_dir)
    config = load_corpus_config(corpus_dir)
    options = {
        "remove_stopwords": bool(config["remove_stopwords"]),
        "lemmatize": bool(config["lemmatize"]),
    }
    vectorizer, faq_matrix = build_tfidf_index(faq_data, options)
    threshold = float(config["similarity_threshold"])

    print(f"\nLoaded {len(faq_data)} FAQs from {config['display_name']}.")
    print(f"Preprocessing: {config.get('preprocessing_config', 'basic')}, "
          f"threshold: {threshold:.2f}")

    if arguments.query:
        print(f"\nAsk your question: {arguments.query}")
        print_result(
            answer_query(
                arguments.query,
                faq_data,
                vectorizer,
                faq_matrix,
                threshold=threshold,
                top_k=3,
                preprocessing_options=options,
            )
        )
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

        print_result(
            answer_query(
                query,
                faq_data,
                vectorizer,
                faq_matrix,
                threshold=threshold,
                top_k=3,
                preprocessing_options=options,
            )
        )


if __name__ == "__main__":
    main()
