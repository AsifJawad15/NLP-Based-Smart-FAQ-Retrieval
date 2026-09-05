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
