"""Download Google News Word2Vec once and build the project's lowercase subset.

Examples:
    python scripts/download_pretrained_embeddings.py
    python scripts/download_pretrained_embeddings.py --verify
    python scripts/download_pretrained_embeddings.py --source GoogleNews-vectors-negative300.bin.gz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--source", type=Path,
        help="Build from an already downloaded word2vec binary file instead of downloading",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only check the existing subset against its metadata",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from src.pretrained_embeddings import (
        PRETRAINED_MODEL, build_pretrained_subset, load_pretrained_vectors,
    )

    if not args.verify:
        source = args.source
        if source is None:
            # gensim reads this variable when its downloader module is imported,
            # so the download lands in the project instead of the home folder.
            data_dir = args.models_root / "pretrained" / "gensim-data"
            os.environ["GENSIM_DATA_DIR"] = str(data_dir)
            import gensim.downloader as api

            print(f"Downloading {PRETRAINED_MODEL} (about 1.7 GB) into {data_dir}", flush=True)
            source = Path(api.load(PRETRAINED_MODEL, return_path=True))
        print(f"Building the lowercase subset from {source}", flush=True)
        started = time.perf_counter()
        metadata = build_pretrained_subset(source, args.models_root)
        print(
            f"  kept {metadata['key_count']:,} of {metadata['source_key_count']:,} keys "
            f"({metadata['rejected_by_pattern']:,} phrases or symbols, "
            f"{metadata['dropped_case_duplicates']:,} less frequent casings) "
            f"in {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    vectors, metadata = load_pretrained_vectors(args.models_root, verify_vectors=True)
    print(f"VERIFIED {PRETRAINED_MODEL}: {len(vectors):,} keys x {vectors.vector_size} dimensions")
    print(f"  artifact: {metadata['artifact_id']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
