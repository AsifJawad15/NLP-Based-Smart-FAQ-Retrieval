# Local Word2Vec artifacts

Train from the finalized local FAQ questions:

```powershell
python scripts/train_word2vec.py --corpus all
```

The launcher starts Python with `PYTHONHASHSEED=42`. Training uses one worker,
one fixed parameter set, and a separate model for each corpus. Each FAQ is one
token sequence; answers and evaluation queries are not training text.

Each domain directory contains `custom_word2vec.model` (generated, Git-ignored)
and `training_metadata.json` (committed). Metadata records corpus and vector
hashes, preprocessing, settings, versions, and vocabulary size. Inference
checks metadata and refuses missing, stale, or mismatched artifacts. It never
trains automatically.

The thresholds tuned for these models live in
`data/<corpus>/word2vec_config.json`, not in `models/`, because they come from
validation queries rather than from training. Each file records the artifact id
it was tuned against, so inference refuses thresholds that belong to a different
model.

After cloning, run the training command before choosing a Word2Vec model. This
uses no network after Python dependencies are installed. If retraining changes
the artifact id, retune its thresholds before evaluation. Reproducibility means
matching numerical vectors in the pinned environment, not identical pickle
bytes: Gensim serialization contains lifecycle timestamps.

## Pretrained vectors (Phase 2B)

`models/pretrained/` holds the case-folded Google News subset used by both
pretrained Word2Vec models. Build it once with
`python scripts/download_pretrained_embeddings.py`; only its README and
`pretrained_metadata.json` are committed. See `models/pretrained/README.md`.

## Sequence encoders (Phase 3)

This is an experimental checkpoint: only e-commerce RNN/BiLSTM metadata and
training histories are currently committed. University artifacts and frozen
sequence thresholds are pending. The generated training data contains known
meaning-changing substitutions; see the root [README](../README.md#phase-3-siamese-rnn-and-bilstm--work-in-progress)
before rebuilding or interpreting development scores as model quality.

Build the training data, then train both encoders for every corpus:

```powershell
python scripts/build_sequence_pairs.py --corpus all
python scripts/train_sequence_models.py --corpus all --arch all
```

Each domain directory then holds `rnn.pt` and `bilstm.pt` (generated,
Git-ignored) with `rnn_metadata.json` and `bilstm_metadata.json` (committed).
A checkpoint stores only the trainable recurrent weights, the cosine scale and
bias, and its vocabulary; the frozen embedding matrix is rebuilt from the
pretrained subset when the model loads. Metadata records the architecture,
hyperparameters, corpus, paraphrase and pair file hashes, the pretrained
subset's artifact id, the best dev epoch, a hash of the saved weights and
package versions. Loading refuses a missing, stale or mismatched checkpoint and
names the training command; nothing trains automatically.

Thresholds tuned for each encoder live in
`data/<corpus>/sequence_rnn_config.json` and `sequence_bilstm_config.json`,
keyed to that checkpoint's artifact id.
