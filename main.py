"""Terminal demonstration for the corpus-configurable Smart FAQ system."""

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
        choice = input("Select a corpus number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1]
        print("Please enter one of the displayed numbers.")


def main() -> None:
    """Load one corpus, build its index once, and answer terminal queries."""

    print("=" * 44)
    print("          SMART FAQ RETRIEVAL SYSTEM")
    print("=" * 44)

    corpora = discover_corpora(DATA_ROOT)
    if not corpora:
        raise SystemExit(f"No corpora found under {DATA_ROOT}")

    _corpus_key, corpus_dir = choose_corpus(corpora)
    faq_data = load_faq_dataset(corpus_dir)
    config = load_corpus_config(corpus_dir)
    options = {
        "remove_stopwords": bool(config["remove_stopwords"]),
        "lemmatize": bool(config["lemmatize"]),
    }
    vectorizer, faq_matrix = build_tfidf_index(faq_data, options)

    print(f"\nLoaded {len(faq_data)} FAQs from {config['display_name']}.")
    print("Type 'exit' to close the program.\n")

    while True:
        query = input("Ask your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if not query:
            print("Please enter a question.\n")
            continue

        result = answer_query(
            query,
            faq_data,
            vectorizer,
            faq_matrix,
            threshold=float(config["similarity_threshold"]),
            top_k=3,
            preprocessing_options=options,
        )

        if not result["found"]:
            best_score = (
                result["top_matches"][0]["similarity"]
                if result["top_matches"]
                else 0.0
            )
            print(f"\n{result['message']}")
            print(f"Best similarity score: {best_score:.4f}\n")
            continue

        best = result["best_match"]
        print(f"\nMatched FAQ: {best['question']}")
        print(f"Similarity Score: {best['similarity']:.4f}")
        print(f"Answer: {best['answer']}")
        print("\nTop matches:")
        for rank, match in enumerate(result["top_matches"], start=1):
            print(f"  {rank}. {match['question']} - {match['similarity']:.4f}")
        print()


if __name__ == "__main__":
    main()
