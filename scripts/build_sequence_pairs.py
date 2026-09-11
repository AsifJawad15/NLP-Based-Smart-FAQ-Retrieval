"""Build Phase 3 training paraphrases and Siamese pairs, with a blocking leakage audit.

Example: python scripts/build_sequence_pairs.py --corpus all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

# break_verbatim_reuse in prepare_datasets.py builds these openings inline, so
# they cannot be imported as a constant; a test keeps this list in sync.
REBUILT_OPENINGS = ("Please explain {0}.", "I need a clear description of {0}.", "Describe for me {0}.")


def evaluation_generator_rules(prepare_path: Path) -> tuple[list[str], set[tuple[str, str]]]:
    """Read what training must avoid from the generator that built the evaluation queries.

    Only its template strings and substitution table are read; its code is
    never used to generate training data.
    """

    spec = importlib.util.spec_from_file_location("prepare_datasets", prepare_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = [template for _pattern, options in module.PARAPHRASE_PATTERNS for template in options]
    templates += [*module.STRUCTURAL_FALLBACKS, *REBUILT_OPENINGS]
    banned: set[tuple[str, str]] = set()
    for original, replacement in module.CONTENT_SYNONYMS.items():
        # Ban both directions, and each word of a multi-word replacement.
        for word in replacement.lower().split():
            banned.add((original.lower(), word))
            banned.add((word, original.lower()))
    return templates, banned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--prepare-script", type=Path, default=ROOT / "scripts" / "prepare_datasets.py")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from src.data_loader import discover_corpora, load_faq_dataset, load_query_dataset
    from src.pretrained_embeddings import load_pretrained_vectors
    from src import sequence_data
    from src.sequence_data import (
        audit_training_data, build_pairs, choose_max_len, generate_paraphrases,
        is_content_word, normalized, synonym_substitutes, template_openings, tokens,
    )
    from src.sequence_models import build_vocabulary
    from src.sequence_training import PAIRS_FILE, PARAPHRASES_FILE, TRAINING_SETTINGS

    generator_settings = {
        name.lower(): getattr(sequence_data, name) for name in [
            "SEED", "PARAPHRASES_PER_FAQ", "SUBSTITUTE_CANDIDATES", "SUBSTITUTE_FLOOR",
            "SUBSTITUTES_KEPT", "SUBSTITUTION_RATE", "DROPOUT_RATE", "HARD_NEGATIVE_POOL",
            "DUPLICATE_SIMILARITY", "FRAMES",
        ]
    }
    generator_settings["substitute_source"] = "WordNet synonyms ranked by pretrained cosine similarity"
    generator_settings["protected_word_count"] = len(sequence_data.PROTECTED_WORDS)

    corpora = discover_corpora(args.data_root)
    if args.corpus != "all":
        if args.corpus not in corpora:
            raise SystemExit(f"Unknown corpus '{args.corpus}'; found {sorted(corpora)}")
        corpora = {args.corpus: corpora[args.corpus]}
    if not corpora:
        raise SystemExit(f"No FAQ corpora found under {args.data_root}")

    templates, banned = evaluation_generator_rules(args.prepare_script)
    openings = template_openings(templates)
    vectors, metadata = load_pretrained_vectors(args.models_root)
    reports = args.reports_root / "phase3"
    reports.mkdir(parents=True, exist_ok=True)

    for name, directory in corpora.items():
        faq = load_faq_dataset(directory)
        faq_ids = set(faq["id"])
        words = {token for question in faq["question"] for token in tokens(question) if is_content_word(token)}
        substitutes = synonym_substitutes(words, vectors, banned)
        paraphrases = generate_paraphrases(
            faq, substitutes, keyed_vectors=vectors, avoid_openings=openings
        )
        manual_path = args.data_root / "manual_evaluation" / f"{name}_queries.csv"
        evaluation = {
            "validation": load_query_dataset(directory / "validation_queries.csv", faq_ids),
            "test": load_query_dataset(directory / "test_queries.csv", faq_ids),
        }
        if manual_path.is_file():
            evaluation["manual"] = load_query_dataset(manual_path, faq_ids, allow_empty=True)
        kept, audit = audit_training_data(faq, paraphrases, evaluation, openings, banned)

        # How many evaluation tokens the pretrained subset knows but the Phase 3
        # vocabulary does not; descriptive only.
        vocabulary = set(build_vocabulary(
            vectors, [*faq["question"], *kept["query"]], TRAINING_SETTINGS["frequent_vocabulary_limit"]
        ))
        evaluation_tokens = [token for frame in evaluation.values() for query in frame["query"] for token in tokens(query)]
        known = [token for token in evaluation_tokens if token in vectors.key_to_index]
        substitutions = [pair for text in kept["substitutions"] for pair in str(text).split()]
        audit.update({
            "corpus": name,
            "generator_settings": generator_settings,
            "pretrained_artifact_id": metadata["artifact_id"],
            "words_with_substitutes": len(substitutes),
            "content_words": len(words),
            "substitutions_made": len(substitutions),
            "distinct_substitution_pairs": len(set(substitutions)),
            "template_openings_checked": len(openings),
            "banned_substitution_pairs": len(banned),
            "max_len": choose_max_len([*faq["question"], *kept.loc[kept["split"] == "train", "query"]]),
            "evaluation_tokens_known_to_pretrained": len(known),
            "evaluation_tokens_known_to_pretrained_but_outside_vocabulary": sum(token not in vocabulary for token in known),
            "evaluation_queries": {key: len(frame) for key, frame in evaluation.items()},
            "normalized_duplicates_within_training": int(kept["query"].map(normalized).duplicated().sum()),
        })
        audit_path = reports / f"{name}_training_data_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not audit["passed"]:
            raise SystemExit(f"Training data audit failed for {name}; see {audit_path}")

        pairs = build_pairs(faq, kept)
        kept.to_csv(directory / PARAPHRASES_FILE, index=False, lineterminator="\n")
        pairs.to_csv(directory / PAIRS_FILE, index=False, lineterminator="\n")
        counts = pairs.groupby(["split", "pair_type"]).size().to_dict()
        print(
            f"BUILT {name}: {audit['kept']['train']} train and {audit['kept']['dev']} dev paraphrases "
            f"({audit['removed_exact_evaluation_match'] + audit['removed_high_overlap']} removed for "
            f"evaluation overlap), coverage {audit['mean_source_token_coverage']:.3f}, "
            f"{len(pairs)} pairs {counts}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
