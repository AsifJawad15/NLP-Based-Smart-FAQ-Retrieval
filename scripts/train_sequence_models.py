"""Train Siamese RNN and BiLSTM encoders per corpus in a reproducible subprocess.

Example: python scripts/train_sequence_models.py --corpus all --arch all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--arch", default="all", choices=["all", "rnn", "bilstm"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
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
    from src.data_loader import discover_corpora, load_corpus_config
    from src.pretrained_embeddings import load_pretrained_vectors
    from src.sequence_models import ARCHITECTURES
    from src.sequence_training import train_sequence_model

    corpora = discover_corpora(args.data_root)
    if not corpora:
        raise SystemExit(f"No FAQ corpora found under {args.data_root}")
    if args.corpus != "all":
        if args.corpus not in corpora:
            raise SystemExit(f"Unknown corpus '{args.corpus}'; found {sorted(corpora)}")
        corpora = {args.corpus: corpora[args.corpus]}
    else:
        corpora = {
            name: directory for name, directory in corpora.items()
            if load_corpus_config(directory)["purpose"] == "research"
        }
    architectures = list(ARCHITECTURES) if args.arch == "all" else [args.arch]

    pretrained = load_pretrained_vectors(args.models_root, verify_vectors=True)
    for name, directory in corpora.items():
        supported = load_corpus_config(directory)["supported_models"]
        unsupported = [item for item in architectures if item not in supported]
        if unsupported:
            raise ValueError(f"{name} does not support: {', '.join(unsupported)}")
        for architecture in architectures:
            started = time.perf_counter()
            metadata = train_sequence_model(
                directory, architecture, pretrained, args.models_root, args.reports_root
            )
            print(
                f"TRAINED {name} {architecture}: best epoch {metadata['best_epoch']} of "
                f"{metadata['epochs_run']}, dev Top-1 {metadata['best_dev_top1']:.3f}, "
                f"dev Top-3 {metadata['best_dev_top3']:.3f}, {metadata['train_pairs']} train pairs, "
                f"max_len {metadata['max_len']}, {time.perf_counter() - started:.0f}s",
                flush=True,
            )
            print(f"  artifact: {metadata['artifact_id']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
