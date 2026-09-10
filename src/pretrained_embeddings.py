"""Pretrained Google News Word2Vec: one case-folded subset and its checked loader.

The downloaded model holds 3,000,000 case-sensitive keys, many of them phrases
such as ``New_York``. `preprocess_text` lowercases every token and keeps only
letters, digits and apostrophes, so the project keeps exactly the keys it can
ever look up: one vector per lowercase word, copied from that word's most
frequent casing. The subset is built once and then memory-mapped, so no
command loads the full 3.6 GB model into memory.

Nothing here downloads or builds implicitly; a missing or altered subset is
refused with the command that rebuilds it.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
from typing import Any, BinaryIO, Iterable, Iterator

import numpy as np

from src.preprocessing import preprocess_text
from src.word2vec_config import load_word2vec_threshold, save_word2vec_config
from src.word2vec_training import BASIC_OPTIONS


MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
PRETRAINED_MODEL = "word2vec-google-news-300"
MODEL_KEYS = ("pw2v_mean", "pw2v_tfidf")
CONFIG_NAME = "pretrained_config.json"
TUNING_GROUP = "phase2b"
KEYS_FILE = f"{PRETRAINED_MODEL}-lower.keys.tsv"
VECTORS_FILE = f"{PRETRAINED_MODEL}-lower.vectors.npy"
METADATA_FILE = "pretrained_metadata.json"
DOWNLOAD_COMMAND = "python scripts/download_pretrained_embeddings.py"

# Basic preprocessing can only emit lowercase letters, digits and apostrophes.
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
CASE_FOLDING = (
    "lowercase each key; keep keys that fully match [a-z0-9']+; "
    "when casings collide, the first (most frequent) one wins"
)
# The artifact id names the vectors, not the environment that copied them.
IDENTITY_FIELDS = (
    "pretrained_model", "source_md5", "vector_size", "token_pattern",
    "case_folding", "key_count", "keys_sha256", "vectors_sha256",
)


def _file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Hash a file in blocks, so gigabyte files never sit in memory."""

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_binary(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _parse_header(line: bytes, path: Path) -> tuple[int, int]:
    try:
        count, size = (int(value) for value in line.split())
    except ValueError as error:
        raise ValueError(f"Not a word2vec binary file: {path}") from error
    if count < 1 or size < 1:
        raise ValueError(f"Not a word2vec binary file: {path}")
    return count, size


def read_word2vec_header(path: str | Path) -> tuple[int, int]:
    """Return the declared key count and vector size of a word2vec binary file."""

    with _open_binary(Path(path)) as stream:
        return _parse_header(stream.readline(), Path(path))


def iter_word2vec_binary(
    path: str | Path, *, with_vectors: bool = True, chunk_size: int = 1 << 20,
) -> Iterator[tuple[str, bytes | None]]:
    """Stream (key, little-endian float32 bytes) pairs from a word2vec binary file.

    A key ends at the first space after the previous vector, and exactly
    4 * vector_size bytes follow it. Vector bytes may contain the space byte
    themselves, so they are counted, never searched.
    """

    source = Path(path)
    with _open_binary(source) as stream:
        count, size = _parse_header(stream.readline(), source)
        width = 4 * size
        buffer, start = b"", 0
        for _ in range(count):
            while True:
                space = buffer.find(b" ", start)
                if space >= 0 and len(buffer) - space - 1 >= width:
                    break
                more = stream.read(chunk_size)
                if not more:
                    raise ValueError(f"{source.name} ended before its {count} declared vectors")
                buffer, start = buffer[start:] + more, 0
            # Some writers end every vector with a newline.
            key = buffer[start:space].lstrip(b"\n").decode("utf-8", errors="replace")
            vector = buffer[space + 1 : space + 1 + width] if with_vectors else None
            start = space + 1 + width
            yield key, vector


def _select_keys(source: Path) -> tuple[list[int], list[str], list[str], int, int]:
    """First pass: choose file positions without reading any vector."""

    seen: set[str] = set()
    positions: list[int] = []
    keys: list[str] = []
    source_keys: list[str] = []
    rejected = duplicates = 0
    for position, (key, _vector) in enumerate(iter_word2vec_binary(source, with_vectors=False)):
        folded = key.lower()
        if not TOKEN_PATTERN.fullmatch(folded):
            rejected += 1
        elif folded in seen:
            duplicates += 1
        else:
            seen.add(folded)
            positions.append(position)
            keys.append(folded)
            source_keys.append(key)
    return positions, keys, source_keys, rejected, duplicates


def pretrained_artifact_id(metadata: dict[str, Any]) -> str:
    payload = {field: metadata[field] for field in IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_pretrained_subset(
    source_path: str | Path, models_root: str | Path = MODELS_ROOT,
) -> dict[str, Any]:
    """Copy one vector per lowercase word into a memory-mappable subset.

    Two streaming passes keep memory flat: the first chooses keys, the second
    copies only their vectors. Metadata is written last, so an interrupted
    rebuild is refused by the loader instead of passing as the old subset.
    """

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Pretrained source file not found: {source}")
    count, size = read_word2vec_header(source)
    positions, keys, source_keys, rejected, duplicates = _select_keys(source)
    if not keys:
        raise ValueError(f"{source.name} contains no lowercase word keys")

    directory = Path(models_root) / "pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / METADATA_FILE
    metadata_path.unlink(missing_ok=True)
    keys_path = directory / KEYS_FILE
    vectors_path = directory / VECTORS_FILE

    matrix = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype="<f4", shape=(len(keys), size)
    )
    try:
        wanted = iter(positions)
        target = next(wanted)
        row = 0
        for position, (key, vector) in enumerate(iter_word2vec_binary(source)):
            if position != target:
                continue
            values = np.frombuffer(vector, dtype="<f4")
            if not np.isfinite(values).all():
                raise ValueError(f"Pretrained vector for {key!r} is not finite")
            matrix[row] = values
            row += 1
            target = next(wanted, None)
            if target is None:
                break
        if row != len(keys):
            raise ValueError(f"{source.name} changed while the subset was being built")
        matrix.flush()
    finally:
        # Release the mapping so Windows allows the file to be reopened.
        del matrix
    keys_path.write_text(
        "".join(f"{key}\t{original}\n" for key, original in zip(keys, source_keys)),
        encoding="utf-8", newline="\n",
    )

    metadata: dict[str, Any] = {
        "pretrained_model": PRETRAINED_MODEL,
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        # gensim-data publishes this MD5 checksum for its download.
        "source_md5": _file_hash(source, "md5"),
        "source_key_count": count,
        "vector_size": size,
        "token_pattern": TOKEN_PATTERN.pattern,
        "case_folding": CASE_FOLDING,
        "rejected_by_pattern": rejected,
        "dropped_case_duplicates": duplicates,
        "key_count": len(keys),
        "keys_from_non_lowercase_source": sum(key != original for key, original in zip(keys, source_keys)),
        "keys_file": KEYS_FILE,
        "keys_sha256": _file_hash(keys_path),
        "vectors_file": VECTORS_FILE,
        "vectors_sha256": _file_hash(vectors_path),
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "gensim": version("gensim"),
        },
    }
    metadata["artifact_id"] = pretrained_artifact_id(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _wrap_vectors(keys: list[str], vectors: np.ndarray):
    """Expose the mapped matrix through gensim's KeyedVectors interface."""

    from gensim.models import KeyedVectors

    wrapped = KeyedVectors(vector_size=int(vectors.shape[1]))
    wrapped.index_to_key = keys
    wrapped.key_to_index = {key: index for index, key in enumerate(keys)}
    wrapped.next_index = len(keys)
    wrapped.vectors = vectors
    return wrapped


def load_pretrained_vectors(
    models_root: str | Path = MODELS_ROOT, *, verify_vectors: bool = False,
):
    """Memory-map the subset after checking it against its metadata.

    The keys file is always hashed. Hashing the vectors reads every byte of a
    file over a gigabyte, so evaluation requests it and the demonstration
    does not.
    """

    directory = Path(models_root) / "pretrained"
    keys_path = directory / KEYS_FILE
    vectors_path = directory / VECTORS_FILE
    metadata_path = directory / METADATA_FILE
    instruction = f"Run: {DOWNLOAD_COMMAND}"
    if not (keys_path.is_file() and vectors_path.is_file() and metadata_path.is_file()):
        raise FileNotFoundError(
            f"Pretrained {PRETRAINED_MODEL} subset or metadata missing under {directory}. {instruction}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = (
            metadata["pretrained_model"] == PRETRAINED_MODEL
            and metadata["token_pattern"] == TOKEN_PATTERN.pattern
            and metadata["case_folding"] == CASE_FOLDING
            and metadata["artifact_id"] == pretrained_artifact_id(metadata)
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid pretrained metadata. {instruction}") from error
    if not valid:
        raise ValueError(f"Stale or mismatched pretrained metadata. {instruction}")

    mismatch = ValueError(f"Pretrained files do not match their metadata. {instruction}")
    if _file_hash(keys_path) != metadata["keys_sha256"]:
        raise mismatch
    if verify_vectors and _file_hash(vectors_path) != metadata["vectors_sha256"]:
        raise mismatch
    keys = [line.split("\t", 1)[0] for line in keys_path.read_text(encoding="utf-8").split("\n") if line]
    vectors = np.load(vectors_path, mmap_mode="r")
    expected = (metadata["key_count"], metadata["vector_size"])
    if len(keys) != metadata["key_count"] or vectors.shape != expected or vectors.dtype != np.dtype("<f4"):
        del vectors
        raise mismatch
    return _wrap_vectors(keys, vectors), metadata


def folded_source_keys(models_root: str | Path = MODELS_ROOT) -> dict[str, str]:
    """Return the keys whose vector was copied from a differently cased key."""

    path = Path(models_root) / "pretrained" / KEYS_FILE
    pairs = (line.split("\t", 1) for line in path.read_text(encoding="utf-8").split("\n") if line)
    return {key: original for key, original in pairs if key != original}


def token_coverage(
    texts: Iterable[str], keyed_vectors, folded: dict[str, str] | None = None, top: int = 20,
) -> dict[str, Any]:
    """Describe how many basic-preprocessed tokens have a vector.

    This is a diagnostic for reports only; no selection step reads it.
    """

    sentences = [preprocess_text(str(text), **BASIC_OPTIONS).split() for text in texts]
    counts = Counter(token for sentence in sentences for token in sentence)
    known = {token for token in counts if token in keyed_vectors}
    total = sum(counts.values())
    report: dict[str, Any] = {
        "texts": len(sentences),
        "token_occurrences": total,
        "distinct_tokens": len(counts),
        "known_occurrence_rate": sum(counts[token] for token in known) / total if total else 0.0,
        "known_distinct_rate": len(known) / len(counts) if counts else 0.0,
        "texts_without_known_token": sum(
            1 for sentence in sentences if not any(token in known for token in sentence)
        ),
        "most_frequent_unknown": [
            [token, count] for token, count in counts.most_common() if token not in known
        ][:top],
    }
    if folded is not None:
        report["occurrences_using_case_folded_vector"] = sum(
            count for token, count in counts.items() if token in known and token in folded
        )
    return report


def save_pretrained_config(
    corpus_dir: str | Path, artifact_id: str, thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Freeze pretrained thresholds in their own file, keyed to the subset."""

    return save_word2vec_config(
        corpus_dir, artifact_id, thresholds, config_name=CONFIG_NAME, model_keys=MODEL_KEYS
    )


def load_pretrained_threshold(corpus_dir: str | Path, metadata: dict[str, Any], model: str) -> float:
    """Read one frozen pretrained threshold, refusing one tuned on other vectors."""

    return load_word2vec_threshold(
        corpus_dir, metadata, model,
        config_name=CONFIG_NAME, model_keys=MODEL_KEYS, group=TUNING_GROUP,
    )
