# NLP-Based Smart FAQ Retrieval and Question Answering System

A corpus-configurable FAQ retrieval engine for the CSE 4122 Natural Language
Processing Laboratory. Phase 1 uses TF-IDF vectors and cosine similarity to
**retrieve an existing answer**. It never generates an answer.

The same engine runs over two independently indexed corpora — a University FAQ
set and an E-commerce FAQ set — each with its own preprocessing configuration
and its own similarity threshold. Once the data is prepared, everything runs
offline.

## Synthetic benchmark results

Preprocessing and threshold were selected on validation queries only, frozen
into `corpus_config.json`, and applied to the synthetic test queries without
retuning. These queries have already been inspected during development and
review; they are a reproducible benchmark, not an untouched final assessment.

| Metric | University FAQ | E-commerce FAQ |
| --- | --- | --- |
| FAQs indexed | 500 | 500 |
| Preprocessing | basic | basic |
| Threshold | 0.46 | 0.58 |
| Top-1 accuracy | 0.860 | 0.953 |
| Top-3 accuracy | 0.993 | 0.980 |
| Correct answer rate | 0.833 | 0.820 |
| Accepted but wrong | 17 | 1 |
| Mean similarity of correct Top-1 | 0.760 | 0.740 |
| Answerable acceptance rate | 0.947 | 0.827 |
| Unanswerable rejection rate | 0.840 | 0.960 |
| False acceptances | 8 | 2 |
| False rejections | 8 | 26 |

Test set: 200 queries per corpus (150 answerable paraphrases, 50 unanswerable).
Full reports, including incorrect-retrieval examples, are in [reports/](reports/).

**Correct answer rate** is the number of answerable queries whose correct FAQ
is ranked first **and accepted**, divided by all answerable queries. Top-1 alone
does not account for threshold rejection. **Accepted but wrong** counts
answerable queries accepted with a different FAQ; false acceptance counts
unanswerable queries that receive an answer.

Selection has two stages, both using validation data only:

1. Compare answerable Top-1 accuracy, then Top-3, then prefer the simpler
   preprocessing configuration. University basic and lemmatized both reach
   0.900 Top-1 and 1.000 Top-3; basic wins the simplicity tie. E-commerce ties
   across all three configurations at 0.900 Top-1 and Top-3, also selecting basic.
2. For that configuration, sweep thresholds from 0.00 to 1.00 in 0.01 steps.
   Maximize the mean of answerable acceptance and unanswerable rejection;
   prefer the higher threshold on ties. This score measures answerability
   decisions, not whether the returned FAQ is correct.

The comparison supports basic preprocessing for these validation sets; it does
not establish that lemmatization improves retrieval or never helps other data.

## How it works

```text
FAQ questions -> preprocess -> fit_transform()   (index built once)
User query    -> same preprocessing -> transform()
Query vector  -> cosine similarity -> top-k ranking -> threshold
```

Answers are never included in the similarity vectors — only questions are
indexed. A query whose best similarity falls below the corpus threshold is
rejected rather than answered. Empty and all-out-of-vocabulary queries return
no matches, even at threshold 0.00. An OOV query also receives no Top-1/Top-3
credit from arbitrary zero-score ties.

## Relationship to the lab topics

| Lab | Topic used or deferred |
| --- | --- |
| 1 | Regex cleaning, tokenization, optional stopword removal and lemmatization |
| 2 | TF-IDF representation using scikit-learn |
| 3 | Cosine similarity for retrieval; Word2Vec remains an optional extension |
| 4-5 | RNN/LSTM and Transformers are outside this TF-IDF project phase |

The implementation follows these topics without copying the lab code. We use
basic WordNet lemmatization without POS tagging: WordNet defaults to nouns, so
`books` becomes `book` but `running` remains `running`. The notebook explains
raw term counts, smoothed IDF, L2 normalization, cosine similarity, and
`fit_transform` versus `transform` with a checked numerical example.

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_nltk.py
```

`scripts/setup_nltk.py` downloads the `stopwords`, `wordnet`, and `omw-1.4`
corpora into a project-local `.nltk_data/` folder. Both frozen configurations
now use basic preprocessing: downloaded NLTK corpora are needed for reproducing
the optional comparisons, unit tests, and full notebook, but not the basic
terminal demonstration. The NLTK **package** remains required for tokenization.
Initial package/resource installation and source-data downloads need internet;
after setup, demonstrations and evaluation use local files only.

## Running

```powershell
# Interactive demonstration with a corpus menu
python main.py

# Non-interactive, for a scripted or repeatable demonstration
python main.py --corpus university --query "How do I receive university alerts?"
python main.py --corpus ecommerce  --query "What cards can I save on Flipkart?"

# A question the system correctly refuses to answer
python main.py --corpus ecommerce --query "How do I train a puppy to sit?"

# The same question on the university corpus is wrongly accepted at 0.470,
# just above its 0.46 threshold. See Limitations.
python main.py --corpus university --query "How do I train a puppy to sit?"

# Validate the finalized data without touching the network
python scripts/prepare_datasets.py validate

# Select preprocessing and threshold on validation data, then freeze them
python evaluate.py tune

# Apply the frozen configuration to the synthetic benchmark and write reports
python evaluate.py test

# Evaluate team-written queries, or report that the templates are still empty
python evaluate.py manual
python evaluate.py manual --corpus university

# Unit tests
python -m unittest discover -s tests
```

`evaluate.py all` runs tuning and testing in order. Tuning always writes
`corpus_config.json` before the test set is read in that run. The tuning functions
receive only validation data. This separation does not erase the fact that
benchmark results have been examined while revising the methodology.

Human evaluation is currently pending. The header-only templates and collection
instructions are in [data/manual_evaluation/](data/manual_evaluation/). Target
20 answerable and 10 unanswerable team-written queries per corpus; keep format
examples out of the measured files. Manual mode uses frozen configurations and
writes `manual_<corpus>_evaluation.json` plus `manual_evaluation_report.md`,
separately from the synthetic reports. Empty templates produce no current scores.

## Rebuilding the corpora

Only needed when intentionally rebuilding from source data. Download mode uses
the network; convert mode overwrites finalized FAQ and query CSVs. Neither is
needed to run the committed project or reproduce its benchmark.

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
│   └── evaluation.py           select_preprocessing, tune_threshold, evaluate_tfidf
├── scripts/
│   ├── prepare_datasets.py     Download, convert, validate the corpora
│   └── setup_nltk.py           One-time NLTK resource download
├── data/<corpus>/              faq_dataset.csv, validation_queries.csv,
│                               test_queries.csv, corpus_config.json
├── data/manual_evaluation/     Empty team-query templates and collection guide
├── reports/                    Tuning sweeps and evaluation results
├── notebooks/                  Teacher demonstration notebook
├── docs/DATA_SOURCES.md        Corpus sources and manual review record
└── tests/                      Retrieval, selection, metrics, and manual-mode tests
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
Schema and count validation cannot establish that every source answer or
generated paraphrase is semantically correct.

## Limitations

- **TF-IDF matches words, not meaning.** A paraphrase that shares no vocabulary
  with its FAQ will be missed. E-commerce shows this most clearly: 26 of 150
  answerable queries score below the threshold and are rejected.
- **A single threshold cannot separate every case.** On the university corpus,
  answerable scores range from 0.299 to 0.984 while unanswerable scores range
  from 0.222 to 0.525. Those ranges overlap, so 8 false acceptances and 8 false
  rejections remain at the validation-selected threshold.

  A concrete example worth knowing before a viva: *"How do I train a puppy to
  sit?"* scores **0.470** against the university corpus, just above its 0.46
  threshold, and is wrongly accepted as FAQ 241, *"Do I have to be a teacher
  to apply to OISE?"*, through shared common words such as *do*, *I*, *a*, *to*. The same
  question is correctly rejected by the e-commerce corpus, whose threshold is
  0.58. This is the cost of choosing a threshold that keeps the answerable
  acceptance rate at 0.947.
- **Natural paraphrases can fail.** The notebook's *"How do I send a product
  back and get my money returned?"* is rejected at 0.335 and ranks a warranty
  FAQ first. It is a failure demonstration, not evidence of successful matching.
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
