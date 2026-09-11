"""Siamese sentence encoders: a vanilla RNN and a stacked BiLSTM over frozen vectors.

Both encoders read the same frozen pretrained Word2Vec vectors that Phase 2B
averages, so the comparison with Phase 2B changes only how word vectors are
combined into one question vector: in order, instead of as an average.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.preprocessing import preprocess_text
from src.word2vec_training import BASIC_OPTIONS


PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_ID, UNK_ID = 0, 1

ARCHITECTURES = {
    "rnn": {"cell": "RNN", "hidden_dim": 128, "num_layers": 1, "bidirectional": False},
    "bilstm": {"cell": "LSTM", "hidden_dim": 128, "num_layers": 2, "bidirectional": True},
}
SCALE_INIT = 10.0
BIAS_INIT = -5.0


def build_vocabulary(keyed_vectors, texts: Iterable[str], limit: int) -> list[str]:
    """<PAD>, <UNK>, the most frequent pretrained words, then known training words.

    Frequent pretrained words are included so that a query word never seen in
    training still has its pretrained vector instead of becoming <UNK>.
    """

    vocabulary = [PAD_TOKEN, UNK_TOKEN, *keyed_vectors.index_to_key[:limit]]
    included = set(vocabulary)
    extra = sorted({
        token
        for text in texts
        for token in preprocess_text(str(text), **BASIC_OPTIONS).split()
        if token not in included and token in keyed_vectors.key_to_index
    })
    return vocabulary + extra


def embedding_matrix(vocabulary: list[str], keyed_vectors) -> torch.Tensor:
    """Copy each vocabulary word's pretrained vector; <PAD> and <UNK> stay zero."""

    matrix = np.zeros((len(vocabulary), keyed_vectors.vector_size), dtype=np.float32)
    positions = [keyed_vectors.key_to_index[word] for word in vocabulary[2:]]
    matrix[2:] = np.asarray(keyed_vectors.vectors[positions], dtype=np.float32)
    return torch.from_numpy(matrix)


def token_ids(text: str, vocabulary_index: dict[str, int], max_len: int) -> list[int]:
    """Basic-preprocess, map unknown words to <UNK>, and truncate."""

    tokens = preprocess_text(str(text), **BASIC_OPTIONS).split()[:max_len]
    return [vocabulary_index.get(token, UNK_ID) for token in tokens]


def pad_batch(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad id lists with <PAD> and return them with their true lengths."""

    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    if (lengths < 1).any():
        raise ValueError("Every sequence needs at least one token")
    ids = torch.full((len(rows), int(lengths.max())), PAD_ID, dtype=torch.long)
    for position, row in enumerate(rows):
        ids[position, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids, lengths


class SequenceEncoder(nn.Module):
    """Token ids -> frozen embeddings -> RNN or BiLSTM -> one sentence vector.

    Sequences are packed, so padding never enters the recurrence: the final
    forward state belongs to each question's last real token, and the final
    backward state to its first.
    """

    def __init__(self, embedding_weights: torch.Tensor, architecture: str) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown sequence architecture: {architecture}")
        settings = ARCHITECTURES[architecture]
        self.architecture = architecture
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True, padding_idx=PAD_ID
        )
        recurrent = nn.RNN if settings["cell"] == "RNN" else nn.LSTM
        self.recurrent = recurrent(
            embedding_weights.shape[1],
            settings["hidden_dim"],
            num_layers=settings["num_layers"],
            batch_first=True,
            bidirectional=settings["bidirectional"],
        )
        self.bidirectional = settings["bidirectional"]
        self.output_dim = settings["hidden_dim"] * (2 if self.bidirectional else 1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _outputs, hidden = self.recurrent(packed)
        # An LSTM returns (h_n, c_n); only the hidden state represents the sentence.
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden
        if self.bidirectional:
            # The top layer's last forward and last backward states.
            return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]


class SiameseMatcher(nn.Module):
    """One shared encoder for both texts; a learnable scale and bias turn cosine into a logit."""

    def __init__(self, encoder: SequenceEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.scale = nn.Parameter(torch.tensor(SCALE_INIT))
        self.bias = nn.Parameter(torch.tensor(BIAS_INIT))

    def forward(
        self, query_ids: torch.Tensor, query_lengths: torch.Tensor,
        faq_ids: torch.Tensor, faq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        query = self.encoder(query_ids, query_lengths)
        faq = self.encoder(faq_ids, faq_lengths)
        return self.scale * nn.functional.cosine_similarity(query, faq, dim=1) + self.bias
