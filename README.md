# NLP-Based Smart FAQ Retrieval and Question Answering System

A corpus-configurable FAQ retrieval engine for the CSE 4122 Natural Language
Processing Laboratory. Phase 1 uses TF-IDF vectors and cosine similarity to
**retrieve an existing answer**. It never generates an answer.

The same engine runs over two independently indexed corpora — a University FAQ
set and an E-commerce FAQ set — each with its own preprocessing configuration
and its own similarity threshold. Once the data is prepared, everything runs
offline.

## Results

Preprocessing and threshold were selected on validation queries only, frozen
into `corpus_config.json`, and then applied once to the held-out test set.

| Metric | University FAQ | E-commerce FAQ |
| --- | --- | --- |
| FAQs indexed | 500 | 500 |
| Preprocessing | lemmatized | basic |
| Threshold | 0.46 | 0.58 |
| Top-1 accuracy | 0.847 | 0.953 |
| Top-3 accuracy | 0.987 | 0.980 |
| Mean similarity of correct Top-1 | 0.756 | 0.740 |
| Answerable acceptance rate | 0.947 | 0.827 |
| Unanswerable rejection rate | 0.840 | 0.960 |
| False acceptances | 8 | 2 |
| False rejections | 8 | 26 |

Test set: 200 queries per corpus (150 answerable paraphrases, 50 unanswerable).
Full reports, including incorrect-retrieval examples, are in [reports/](reports/).

The two corpora select different configurations, which is the point of the
comparison: lemmatization helps the university corpus, whose questions vary in
number and tense, but does not help the e-commerce corpus, whose questions are
dominated by brand and product names that lemmatization leaves untouched.

## How it works

```text
FAQ questions -> preprocess -> fit_transform()   (index built once)
User query    -> same preprocessing -> transform()
Query vector  -> cosine similarity -> top-k ranking -> threshold
```

Answers are never included in the similarity vectors — only questions are
indexed. A query whose best similarity falls below the corpus threshold is
rejected rather than answered.

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_nltk.py
```

`scripts/setup_nltk.py` downloads the `stopwords`, `wordnet`, and `omw-1.4`
corpora into a project-local `.nltk_data/` folder. This is the only step that
needs the internet. The university corpus uses the lemmatized configuration, so
this step is required, not optional.

## Running

```powershell
# Interactive demonstration with a corpus menu
python main.py

# Non-interactive, for a scripted or repeatable demonstration
python main.py --corpus university --query "How do I receive university alerts?"
python main.py --corpus ecommerce  --query "How can I save my cards on Flipkart?"

# A question the system correctly refuses to answer
python main.py --corpus ecommerce --query "How do I train a puppy to sit?"

# The same question on the university corpus is wrongly accepted at 0.472,
# just above its 0.46 threshold. See Limitations.
python main.py --corpus university --query "How do I train a puppy to sit?"

# Validate the finalized data without touching the network
python scripts/prepare_datasets.py validate

# Select preprocessing and threshold on validation data, then freeze them
python evaluate.py tune

# Apply the frozen configuration once to the test set and write reports
python evaluate.py test

# Unit tests
python -m unittest discover -s tests
```

`evaluate.py all` runs tuning and testing in order. Tuning always writes
`corpus_config.json` before the test set is read, so the test set cannot
influence the configuration.

## Rebuilding the corpora

Only needed if the source data changes. This is the one workflow that uses the
network.

```powershell
python scripts/prepare_datasets.py download
python scripts/prepare_datasets.py convert
python scripts/prepare_datasets.py validate
```

Raw downloads land in `data/staging/`, which is git-ignored. The finalized CSVs
in `data/university/` and `data/ecommerce/` are committed.

## Project layout

```text
Smart_FAQ/
├── main.py                     Terminal demonstration
├── evaluate.py                 Threshold tuning and test evaluation
├── src/
│   ├── data_loader.py          Corpus discovery, loading, schema validation
│   ├── preprocessing.py        Normalization, optional stop words / lemmas
│   ├── tfidf_retrieval.py      Index building, cosine ranking, thresholding
│   └── evaluation.py           tune_threshold, evaluate_tfidf
├── scripts/
│   ├── prepare_datasets.py     Download, convert, validate the corpora
│   └── setup_nltk.py           One-time NLTK resource download
├── data/<corpus>/              faq_dataset.csv, validation_queries.csv,
│                               test_queries.csv, corpus_config.json
├── reports/                    Tuning sweeps and evaluation results
├── notebooks/                  Teacher demonstration notebook
├── docs/DATA_SOURCES.md        Corpus sources and manual review record
└── tests/                      34 unit tests
```

FAQ schema: `id,question,answer,category,source,source_type`
Query schema: `query,expected_faq_id,is_answerable`

## Data

Both corpora hold 500 reviewed FAQs. The university corpus combines a screened
subset of the [CPath dataset](https://huggingface.co/datasets/houcine-bdk/cpath-mcgill-ubc)
with FAQs scraped from official `utoronto.ca` and `ubc.ca` pages; the e-commerce
corpus comes from [NebulaByte/E-Commerce_FAQs](https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs).

[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) records every source, what was
rejected during review and why, the adjudication of all seven flagged semantic
duplicate pairs, and the measured quality of the generated paraphrases.

## Limitations

- **TF-IDF matches words, not meaning.** A paraphrase that shares no vocabulary
  with its FAQ will be missed. E-commerce shows this most clearly: 26 of 150
  answerable queries score below the threshold and are rejected.
- **A single threshold cannot separate every case.** On the university corpus,
  answerable scores range from 0.302 to 0.984 while unanswerable scores range
  from 0.222 to 0.556. Those ranges overlap, so 8 false acceptances and 8 false
  rejections remain at the best available threshold.

  A concrete example worth knowing before a viva: *"How do I train a puppy to
  sit?"* scores **0.472** against the university corpus, just above its 0.46
  threshold, and is wrongly accepted — matched to a question about amending an
  application, purely on the shared words *how*, *do*, *I*, *to*. The same
  question is correctly rejected by the e-commerce corpus, whose threshold is
  0.58. This is the cost of choosing a threshold that keeps the answerable
  acceptance rate at 0.947.
- **The evaluation queries are generated, not human-written.** They are
  measurably distinct from their sources (no query reproduces its source
  question, mean source-token overlap 0.59 and 0.77), but some are not fluent.
  Reported accuracy should be read as an estimate on synthetic paraphrases.
- **The university corpus mixes two source types**, which differ in tone and
  answer length.
- **The e-commerce corpus is region-specific** (rupees, PhonePe, SuperCoins).

## Scope

Phase 1 is TF-IDF only. Word2Vec and Transformer/BERT work will reuse these
corpora and evaluation sets but requires separate approval. This phase
deliberately contains no RNN/LSTM, no GUI, no Flask or Streamlit, and no
generative answering.
