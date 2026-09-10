# Pretrained Google News Word2Vec (Phase 2B)

Build once, after installing the requirements:

```powershell
python scripts/download_pretrained_embeddings.py
```

The script downloads `word2vec-google-news-300` (about 1.7 GB) into
`models/pretrained/gensim-data/` through gensim's downloader, which checks the
published MD5 checksum. It then streams the file twice and writes:

| File | Committed | Contents |
| --- | --- | --- |
| `word2vec-google-news-300-lower.keys.tsv` | no | One lowercase key per line, with the source key its vector came from |
| `word2vec-google-news-300-lower.vectors.npy` | no | The matching 300-dimensional float32 vectors |
| `pretrained_metadata.json` | yes | Counts, case-folding rule, source checksum, file hashes and artifact id |

## Why a subset

Google News keys are case-sensitive and include phrases such as `New_York`.
`preprocess_text` lowercases every token and emits only letters, digits and
apostrophes, so a query token can only ever match a lowercase single-word key.
The subset keeps exactly those keys. When two casings fold together (`The`,
`the`), the first one in the file, which is the more frequent, wins. Nothing
else is filtered.

The vectors are memory-mapped, so evaluation and the terminal demonstration
never hold the full 3.6 GB model in memory. The loader always checks the keys
file hash; `evaluate.py` also hashes the vector file. A missing, altered or
stale subset is refused with the command above. Nothing downloads or builds
automatically.

Thresholds tuned against this subset live in
`data/<corpus>/pretrained_config.json`, keyed to the artifact id recorded here.

Check an existing subset without rebuilding it:

```powershell
python scripts/download_pretrained_embeddings.py --verify
```
