"""Seeded Siamese training, dev-split checkpoint selection and checked persistence.

Validation queries are only logged during training. They are used to tune
thresholds afterwards, exactly as for every earlier model, and never select a
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data_loader import load_faq_dataset, load_query_dataset
from src.evaluation import evaluate_rankings
from src.sequence_data import choose_max_len
from src.sequence_models import (
    ARCHITECTURES,
    BIAS_INIT,
    SCALE_INIT,
    UNK_ID,
    SequenceEncoder,
    SiameseMatcher,
    build_vocabulary,
    embedding_matrix,
    pad_batch,
    token_ids,
)
from src.sequence_retrieval import build_sequence_index, rank_sequence_queries
from src.word2vec_config import load_word2vec_threshold, save_word2vec_config
from src.word2vec_training import artifact_id, corpus_hash


MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
REPORTS_ROOT = Path(__file__).resolve().parents[1] / "reports"
PARAPHRASES_FILE = "train_paraphrases.csv"
PAIRS_FILE = "sequence_train_pairs.csv"
BUILD_COMMAND = "python scripts/build_sequence_pairs.py"
TRAIN_COMMAND = "python scripts/train_sequence_models.py"
TUNING_GROUP = "phase3"
TRAINING_SETTINGS = {
    "seed": 42,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 20,
    "early_stopping_patience": 3,
    "loss": "BCEWithLogitsLoss on scale * cosine + bias",
    "scale_init": SCALE_INIT,
    "bias_init": BIAS_INIT,
    "freeze_embeddings": True,
    "frequent_vocabulary_limit": 100_000,
    "torch_threads": 1,
    "selection": "highest dev Top-1, then dev Top-3, then lowest dev loss",
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vocabulary_hash(vocabulary: list[str]) -> str:
    return hashlib.sha256(json.dumps(vocabulary).encode("utf-8")).hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    return digest.hexdigest()


def _trainable_state(model: SiameseMatcher) -> dict[str, torch.Tensor]:
    """Everything except the frozen embedding, which is rebuilt from the subset."""

    return {
        key: value.detach().clone() for key, value in model.state_dict().items()
        if key != "encoder.embedding.weight"
    }


def config_name(architecture: str) -> str:
    return f"sequence_{architecture}_config.json"


def save_sequence_config(
    corpus_dir: str | Path, architecture: str, artifact: str, thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Freeze one architecture's threshold, keyed to its own checkpoint."""

    return save_word2vec_config(
        corpus_dir, artifact, thresholds,
        config_name=config_name(architecture), model_keys=(architecture,),
    )


def load_sequence_threshold(corpus_dir: str | Path, metadata: dict[str, Any], architecture: str) -> float:
    """Read one frozen sequence threshold, refusing one tuned on another checkpoint."""

    return load_word2vec_threshold(
        corpus_dir, metadata, architecture,
        config_name=config_name(architecture), model_keys=(architecture,), group=TUNING_GROUP,
    )


def _load_training_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paraphrase_path = directory / PARAPHRASES_FILE
    pairs_path = directory / PAIRS_FILE
    if not paraphrase_path.is_file() or not pairs_path.is_file():
        raise FileNotFoundError(
            f"Sequence training data missing for {directory.name}. "
            f"Run: {BUILD_COMMAND} --corpus {directory.name}"
        )
    return (pd.read_csv(paraphrase_path, keep_default_na=False),
            pd.read_csv(pairs_path, keep_default_na=False))


def _pair_rows(pairs: pd.DataFrame, vocabulary_index: dict[str, int], max_len: int):
    """Token ids for both sides; a pair where either side has no known word is dropped."""

    left = [token_ids(text, vocabulary_index, max_len) for text in pairs["text_a"]]
    right = [token_ids(text, vocabulary_index, max_len) for text in pairs["text_b"]]
    usable = [
        index for index in range(len(left))
        if any(value > UNK_ID for value in left[index]) and any(value > UNK_ID for value in right[index])
    ]
    labels = torch.tensor(pairs["label"].to_numpy(dtype=np.float32)[usable])
    return [left[i] for i in usable], [right[i] for i in usable], labels, len(left) - len(usable)


def _logits(model: SiameseMatcher, left: list[list[int]], right: list[list[int]]) -> torch.Tensor:
    left_ids, left_lengths = pad_batch(left)
    right_ids, right_lengths = pad_batch(right)
    return model(left_ids, left_lengths, right_ids, right_lengths)


def _mean_loss(model, criterion, left, right, labels, batch_size: int = 256) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(left), batch_size):
            stop = start + batch_size
            loss = criterion(_logits(model, left[start:stop], right[start:stop]), labels[start:stop])
            total += float(loss) * len(left[start:stop])
    return total / len(left)


def _retrieval_accuracy(encoder, vocabulary_index, max_len, faq_data, queries) -> tuple[float, float]:
    index = build_sequence_index(faq_data, encoder, vocabulary_index, max_len)
    ranking = rank_sequence_queries(queries["query"], faq_data, index, top_k=3)
    report = evaluate_rankings(queries, faq_data, ranking, 0.0)
    return report["top1_accuracy"], report["top3_accuracy"]


def train_sequence_model(
    corpus_dir: str | Path, architecture: str, pretrained,
    models_root: str | Path = MODELS_ROOT, reports_root: str | Path = REPORTS_ROOT,
) -> dict[str, Any]:
    """Train one architecture for one corpus and save its best dev checkpoint."""

    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError(
            f"Train through {TRAIN_COMMAND} so Python starts with PYTHONHASHSEED=42"
        )
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown sequence architecture: {architecture}")
    settings = TRAINING_SETTINGS
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)

    directory = Path(corpus_dir)
    faq_data = load_faq_dataset(directory)
    paraphrases, pairs = _load_training_data(directory)
    vectors, pretrained_metadata = pretrained

    vocabulary = build_vocabulary(
        vectors, [*faq_data["question"], *paraphrases["query"]], settings["frequent_vocabulary_limit"]
    )
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}
    max_len = choose_max_len(
        [*faq_data["question"], *paraphrases.loc[paraphrases["split"] == "train", "query"]]
    )
    train_left, train_right, train_labels, dropped_train = _pair_rows(
        pairs[pairs["split"] == "train"], vocabulary_index, max_len
    )
    dev_left, dev_right, dev_labels, dropped_dev = _pair_rows(
        pairs[pairs["split"] == "dev"], vocabulary_index, max_len
    )
    dev_queries = paraphrases.loc[paraphrases["split"] == "dev", ["query", "expected_faq_id"]].copy()
    dev_queries["expected_faq_id"] = dev_queries["expected_faq_id"].astype("Int64")
    dev_queries["is_answerable"] = True
    dev_queries = dev_queries.reset_index(drop=True)
    validation = load_query_dataset(directory / "validation_queries.csv", set(faq_data["id"]))

    model = SiameseMatcher(SequenceEncoder(embedding_matrix(vocabulary, vectors), architecture))
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"],
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(settings["seed"])

    history: list[dict[str, Any]] = []
    best_key = best_state = None
    best_epoch = waited = 0
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_left), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start : start + settings["batch_size"]]
            optimizer.zero_grad()
            logits = _logits(model, [train_left[i] for i in batch], [train_right[i] for i in batch])
            loss = criterion(logits, train_labels[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)

        model.eval()
        dev_loss = _mean_loss(model, criterion, dev_left, dev_right, dev_labels)
        dev_top1, dev_top3 = _retrieval_accuracy(model.encoder, vocabulary_index, max_len, faq_data, dev_queries)
        validation_top1, validation_top3 = _retrieval_accuracy(
            model.encoder, vocabulary_index, max_len, faq_data, validation
        )
        history.append({
            "epoch": epoch, "train_loss": total / len(order), "dev_loss": dev_loss,
            "dev_top1": dev_top1, "dev_top3": dev_top3,
            "validation_top1_logged_only": validation_top1,
            "validation_top3_logged_only": validation_top3,
            "scale": float(model.scale), "bias": float(model.bias),
        })

        key = (dev_top1, dev_top3, -dev_loss)
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, waited = key, _trainable_state(model), epoch, 0
        else:
            waited += 1
            if waited >= settings["early_stopping_patience"]:
                break

    name = directory.name
    target = Path(models_root) / name
    target.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"architecture": architecture, "state_dict": best_state, "vocabulary": vocabulary},
        target / f"{architecture}.pt",
    )
    history_dir = Path(reports_root) / "phase3"
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        history_dir / f"{name}_{architecture}_training_history.csv", index=False, lineterminator="\n"
    )

    best = history[best_epoch - 1]
    metadata: dict[str, Any] = {
        "corpus": name,
        "architecture": architecture,
        "architecture_settings": ARCHITECTURES[architecture],
        "corpus_sha256": corpus_hash(directory),
        "paraphrases_sha256": _file_hash(directory / PARAPHRASES_FILE),
        "pairs_sha256": _file_hash(directory / PAIRS_FILE),
        "pretrained_model": pretrained_metadata["pretrained_model"],
        "pretrained_artifact_id": pretrained_metadata["artifact_id"],
        "embedding_dim": int(vectors.vector_size),
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": _vocabulary_hash(vocabulary),
        "max_len": max_len,
        "training_settings": settings,
        "python_hash_seed": 42,
        "train_pairs": len(train_left),
        "dev_pairs": len(dev_left),
        "pairs_dropped_without_known_words": dropped_train + dropped_dev,
        "dev_queries": len(dev_queries),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_dev_top1": best["dev_top1"],
        "best_dev_top3": best["dev_top3"],
        "best_dev_loss": best["dev_loss"],
        "state_sha256": _state_hash(best_state),
        "package_versions": {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        },
    }
    metadata["artifact_id"] = artifact_id(metadata)
    (target / f"{architecture}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_sequence_model(
    corpus_dir: str | Path, architecture: str, pretrained, models_root: str | Path = MODELS_ROOT,
):
    """Rebuild a trained encoder; never train as an inference side effect.

    Returns the encoder, its vocabulary index and the checkpoint metadata.
    """

    directory = Path(corpus_dir)
    name = directory.name
    target = Path(models_root) / name
    instruction = f"Run: {TRAIN_COMMAND} --corpus {name} --arch {architecture}"
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown sequence architecture: {architecture}")
    checkpoint_path = target / f"{architecture}.pt"
    metadata_path = target / f"{architecture}_metadata.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Sequence model {architecture} or its metadata missing for {name}. {instruction}")

    vectors, pretrained_metadata = pretrained
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = (
            metadata["corpus"] == name
            and metadata["architecture"] == architecture
            and metadata["architecture_settings"] == ARCHITECTURES[architecture]
            and metadata["corpus_sha256"] == corpus_hash(directory)
            and metadata["paraphrases_sha256"] == _file_hash(directory / PARAPHRASES_FILE)
            and metadata["pairs_sha256"] == _file_hash(directory / PAIRS_FILE)
            and metadata["pretrained_artifact_id"] == pretrained_metadata["artifact_id"]
            and metadata["training_settings"] == TRAINING_SETTINGS
            and metadata["artifact_id"] == artifact_id(metadata)
        )
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        raise ValueError(f"Invalid sequence model metadata for {name} ({architecture}). {instruction}") from error
    if not valid:
        raise ValueError(f"Stale or mismatched sequence model for {name} ({architecture}). {instruction}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    state = checkpoint["state_dict"]
    if _vocabulary_hash(vocabulary) != metadata["vocabulary_sha256"] or _state_hash(state) != metadata["state_sha256"]:
        raise ValueError(f"Sequence model files do not match their metadata. {instruction}")
    model = SiameseMatcher(SequenceEncoder(embedding_matrix(vocabulary, vectors), architecture))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or list(missing) != ["encoder.embedding.weight"]:
        raise ValueError(f"Sequence checkpoint has unexpected weights. {instruction}")
    model.eval()
    return model.encoder, {word: index for index, word in enumerate(vocabulary)}, metadata
