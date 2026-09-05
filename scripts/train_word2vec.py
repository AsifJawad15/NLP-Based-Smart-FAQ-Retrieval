"""Train separate question-only models in a reproducible subprocess.

Example: python scripts/train_word2vec.py --corpus all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    # Setting this inside the current interpreter would be too late.
    if os.environ.get("PYTHONHASHSEED") != "42":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "42"
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment, check=False,
        )
        raise SystemExit(completed.returncode)

    sys.path.insert(0, str(ROOT))
    from src.data_loader import discover_corpora
    from src.word2vec_training import train_domain

    corpora = discover_corpora(args.data_root)
    if not corpora:
        raise SystemExit(f"No FAQ corpora found under {args.data_root}")
    if args.corpus != "all":
        if args.corpus not in corpora:
            raise SystemExit(f"Unknown corpus '{args.corpus}'; found {sorted(corpora)}")
        corpora = {args.corpus: corpora[args.corpus]}
    for name, directory in corpora.items():
        metadata = train_domain(directory, args.models_root)
        print(f"TRAINED {name}: {metadata['faq_count']} questions, "
              f"{metadata['token_count']} tokens, {metadata['vocabulary_size']} words, "
              f"dimension {metadata['training_settings']['vector_size']}", flush=True)
        print(f"  artifact: {metadata['artifact_id']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
