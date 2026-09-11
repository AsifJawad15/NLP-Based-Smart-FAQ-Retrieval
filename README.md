# NLP-Based Smart FAQ Retrieval and Question Answering System

A corpus-configurable FAQ retrieval engine for the CSE 4122 Natural Language
Processing Laboratory. It uses TF-IDF vectors and cosine similarity to
**retrieve an existing answer**. It never generates an answer. Phase 2 adds a
custom Word2Vec model per domain and compares three retrieval models. Phase 2B
adds pretrained Google News Word2Vec vectors and compares five. Phase 3 adds
Siamese RNN and BiLSTM encoders; their implementation and preliminary training
are in progress, with no final seven-model benchmark yet.

The same engine runs over two independently indexed corpora — a University FAQ
set and an E-commerce FAQ set — each with its own preprocessing configuration
and its own similarity threshold. Once the data is prepared, everything runs
offline.

## Completion and verification

**Phase 1 and Phase 2A are complete. Phase 2B retrieval and evaluation are
verified; its notebook section remains unfinished. Phase 3 is a work in
progress. Human evaluation and the presentation interface are pending.**

| Phase | Models | Current status |
| --- | --- | --- |
| 1 | TF-IDF | Frozen baseline with test results |
| 2A | Custom Word2Vec mean and weighted | Trained per domain; frozen thresholds and test results |
| 2B | Pretrained Word2Vec mean and weighted | Five-model comparison verified; notebook extension pending |
| 3 | Siamese RNN and BiLSTM | Both e-commerce checkpoints trained; university artifacts, thresholds and final comparison pending |

The 11 September 2026 publication review passed **125 tests**, `pip check`,
both corpus validators and **all 29 notebook code cells**. It reproduced all **10 model/domain
evaluation JSON reports** in `reports/phase2b/` exactly, using a temporary
output directory. The pretrained subset passed full checksum verification.
Both e-commerce sequence checkpoints loaded with matching metadata and their
saved training histories agreed with their selected epochs. The notebook
contains **54 cells, including 29 executable code cells**; it currently covers
TF-IDF and custom Word2Vec only. See the Phase 3 section for the limitations
of the preliminary sequence training.

| Dataset count | University | E-commerce |
| --- | ---: | ---: |
| FAQs | 500 | 500 |
| Categories | 14 | 10 |
| Validation queries | 50 (30 answerable, 20 unanswerable) | 50 (30 answerable, 20 unanswerable) |
| Test queries | 200 (150 answerable, 50 unanswerable) | 200 (150 answerable, 50 unanswerable) |
| Collected human-evaluation queries | 0 | 0 |

The Phase 2 review on 5 September 2026 verified 83 passing tests, both corpus
validators, and all 29 notebook code cells executing successfully. Fresh
subprocess training reproduced both full-corpus Word2Vec models exactly at the
numerical-vector level. Validation reproduced all four Word2Vec thresholds and
sweeps; testing reproduced all six model reports, both paired prediction CSVs,
and the comparison summary. The two baseline evaluation JSONs have gained only
`model` and `model_label` fields since the Phase 1 checkpoint.

The tests cover repeated-token weighting, OOV and zero-vector rejection,
stable ranking and threshold boundaries, runtime/batch agreement, separate
question-only training, persistence, stale artifacts, the streaming pretrained
subset builder and its checked loader, Phase 2B report isolation, and
empty/populated temporary manual evaluations. Passing these checks verifies the
experiment's implementation; it does not establish real-user accuracy or every
label's semantic correctness.

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

## Phase 2A: TF-IDF against a custom Word2Vec

Phase 2 trains one Skip-Gram Word2Vec model per domain on that domain's **FAQ
questions only**, then compares three ways of ranking the same 200 test queries.
For each domain, both Word2Vec aggregation methods reuse the same trained file
and differ only in how word vectors are combined into one question vector.

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted |
| --- | --- | --- | --- |
| University Top-1 accuracy | **0.860** | 0.587 | 0.693 |
| University correct answer rate | **0.833** | 0.493 | 0.493 |
| University threshold | 0.46 | 0.93 | 0.95 |
| E-commerce Top-1 accuracy | **0.953** | 0.753 | 0.740 |
| E-commerce correct answer rate | **0.820** | 0.633 | 0.687 |
| E-commerce threshold | 0.58 | 0.92 | 0.91 |

**TF-IDF has higher Top-1 accuracy and correct answer rate on both corpora.**
The following observations help interpret this result; the experiment does not
isolate a single cause:

1. University supplies 6,257 training tokens and e-commerce 6,665. This small
   training set limits the evidence available to learn word relationships;
   the notebook shows both plausible and noisy nearest neighbours.
2. These trained sentence vectors produce closely grouped scores. The mean
   best-match cosine differs between answerable and unanswerable queries by
   0.34-0.37 for TF-IDF and only 0.03-0.09 for Word2Vec. Validation selects
   Word2Vec thresholds of 0.91-0.95, which reject some correct Top-1 matches.
   On university, the weighted model ranks 104 of 150 answerable queries first
   but delivers only 74.

IDF weighting improves university Top-1 over plain averaging (0.693 against
0.587), but reduces e-commerce Top-1 (0.740 against 0.753). It does not improve
university correct answer rate, while it raises e-commerce correct answer rate
from 0.633 to 0.687. Higher cosine values are neither calibrated probabilities
nor evidence of better retrieval; compare the measured decisions and errors.

Full tables, per-model sweeps and a paired per-query CSV are in
[reports/phase2/](reports/phase2/); sections 11 to 13 of the notebook explain
Skip-Gram, both aggregations, and the paired errors.

```powershell
# Train one model per corpus. Required once before any Word2Vec command.
python scripts/train_word2vec.py --corpus all

# Tune both Word2Vec thresholds, then evaluate all three models
python evaluate.py all --model all

# Reproduce the comparison using frozen thresholds, without tuning
python evaluate.py test --model all

# Ask one question with a chosen model
python main.py --corpus university --model w2v_mean --query "How do I receive university alerts?"
```

Word2Vec thresholds live in `data/<corpus>/word2vec_config.json`, keyed to the
trained model's artifact id, so this Phase 2 workflow preserves the frozen
Phase 1 `corpus_config.json`. A changed artifact id invalidates old Word2Vec
thresholds; an identical reproducible retraining keeps them valid. Model binaries
are Git-ignored; `models/<corpus>/training_metadata.json` is committed and
records the corpus hash, settings, package versions and vector hash.
Reproducibility means matching numerical vectors in the pinned environment, not
identical pickle bytes.

## Phase 2B: pretrained Google News Word2Vec

Phase 2B keeps both custom Word2Vec models and adds the same two aggregations
over `word2vec-google-news-300`, pretrained news vectors also used by Lab 4.
Preprocessing, the
aggregation functions, cosine ranking, the threshold protocol, the metrics and
the 200 test queries per domain are all unchanged; only the source of the word
vectors differs. Each pretrained model gets its own threshold, tuned on
validation queries only.

| Metric | TF-IDF | W2V mean | W2V weighted | Pretrained mean | Pretrained weighted |
| --- | --- | --- | --- | --- | --- |
| University Top-1 accuracy | 0.860 | 0.587 | 0.693 | **0.873** | 0.867 |
| University Top-3 accuracy | **0.993** | 0.673 | 0.773 | 0.953 | 0.987 |
| University correct answer rate | **0.833** | 0.493 | 0.493 | 0.627 | 0.813 |
| University unanswerable rejection | 0.840 | 0.760 | **0.980** | 0.960 | 0.880 |
| University threshold | 0.46 | 0.93 | 0.95 | 0.86 | 0.80 |
| E-commerce Top-1 accuracy | **0.953** | 0.753 | 0.740 | 0.827 | 0.867 |
| E-commerce Top-3 accuracy | **0.980** | 0.860 | 0.880 | 0.913 | 0.927 |
| E-commerce correct answer rate | 0.820 | 0.633 | 0.687 | 0.787 | **0.840** |
| E-commerce unanswerable rejection | **0.960** | 0.940 | 0.920 | 0.760 | 0.520 |
| E-commerce threshold | 0.58 | 0.92 | 0.91 | 0.81 | 0.76 |

**Pretrained vectors beat the custom vectors for every aggregation and corpus**,
in both Top-1 accuracy and correct answer rate. **Against TF-IDF the result is
mixed**, and no model is best on every metric:

- University: the pretrained models rank 131 and 130 of 150 answerable queries
  first, against 129 for TF-IDF — a difference of one or two queries. TF-IDF
  still delivers the most correct answers (0.833). The mean model's 0.86
  threshold rejects 37 queries it had ranked correctly.
- E-commerce: the weighted model delivers the most correct answers (0.840
  against 0.820), but it also answers 24 of 50 unanswerable queries, against 2
  for TF-IDF. Its threshold was chosen on only 20 unanswerable validation queries.

Two measurements help interpret the gain over custom vectors; the experiment
does not isolate a single cause:

1. **Coverage.** A custom model knows only the words of its own FAQ questions:
   81.8% (university) and 79.3% (e-commerce) of test-query tokens. The
   pretrained subset knows 99.5% and 98.8%. Its unknown FAQ words are mostly
   brand and product names (`supercoins`, `bgauss`, `finserv`, `phonepe`) and
   numbers.
2. **Score separation.** The gap between the mean best-match score of
   answerable and unanswerable queries is 0.13-0.17 for the pretrained models,
   0.03-0.09 for the custom models and 0.34-0.37 for TF-IDF.

Error cases compare each pretrained model with TF-IDF and with the custom model
that combines vectors the same way:

| Model | Case A: TF-IDF and custom fail, pretrained answers | Case B: TF-IDF answers, pretrained does not | Case C: custom answers, pretrained does not |
| --- | --- | --- | --- |
| University, pretrained mean | 2 | 34 | 6 |
| University, pretrained weighted | 6 | 10 | 1 |
| E-commerce, pretrained mean | 9 | 18 | 8 |
| E-commerce, pretrained weighted | 9 | 10 | 6 |

Full tables, sweeps, paired predictions, every case row and the coverage
reports are in [reports/phase2b/](reports/phase2b/).

### How the pretrained vectors are stored

Google News keys are case-sensitive, and 71% of them are phrases or symbols
such as `New_York`. Because `preprocess_text` lowercases every token, the build
script keeps only lowercase single-word keys (710,048 of 3,000,000); when two
casings fold together, the more frequent one supplies the vector. This matters:
Google News has no lowercase `a`, `to`, `of`, `and` or `flipkart`, only `A`,
`To`, `Of`, `And` and `Flipkart`. The 852 MB subset is memory-mapped, so no
command loads the full 3.6 GB vector matrix into memory. Startup time depends
on the machine and filesystem cache.

```powershell
# Once: download the model (about 1.7 GB) and build the subset
python scripts/download_pretrained_embeddings.py

# Tune both pretrained thresholds, then evaluate all five models
python evaluate.py all --model phase2b

# Ask one question with a pretrained model
python main.py --corpus ecommerce --model pw2v_tfidf --query "What cards can I save on Flipkart?"
```

Pretrained thresholds live in `data/<corpus>/pretrained_config.json`, keyed to
the subset's artifact id in `models/pretrained/pretrained_metadata.json`. The
download and the generated subset are Git-ignored; see
[models/pretrained/README.md](models/pretrained/README.md).

## Phase 3: Siamese RNN and BiLSTM — work in progress

Both models use the same frozen 300-dimensional pretrained word vectors as
Phase 2B. A shared encoder reads the query and FAQ question. The RNN produces
a 128-dimensional vector; the two-layer BiLSTM concatenates forward and backward
states into 256 dimensions. Packed sequences prevent padding from changing
the final state. Training uses `BCEWithLogitsLoss` on a learned scale and bias
applied to cosine similarity, with Adam, learning rate 0.001, batch size 32,
at most 20 epochs, seed 42 and early-stopping patience 3.

Checkpoint selection uses held-out **development paraphrases**: highest Top-1,
then Top-3, then lowest dev loss. Existing validation queries are logged only
during training and are reserved for subsequent threshold selection. Test
queries are not training pairs or checkpoint-selection inputs. Each loaded
encoder builds its FAQ index once in memory.

### Saved artifacts at this checkpoint

| Artifact | E-commerce | University |
| --- | --- | --- |
| Training paraphrases | 1,000 | Not saved |
| Development paraphrases | 500 | Not saved |
| Training pairs | 3,000: 1,000 positive + 2,000 negative | Not saved |
| Development pairs | 1,500: 500 positive + 1,000 negative | Not saved |
| RNN checkpoint and metadata | Available locally; metadata committed | Not available |
| BiLSTM checkpoint and metadata | Available locally; metadata committed | Not available |
| Frozen sequence thresholds | Not available | Not available |
| Final sequence test results | Not available | Not available |

Every paraphrase produces one positive, one random negative and one TF-IDF
hard negative. Near-duplicate FAQ questions are excluded as negatives.
The saved e-commerce artifacts use a maximum sequence length of 29 tokens
and a vocabulary of 100,013 entries, including `<PAD>` and `<UNK>`.

| E-commerce model | Epochs run | Selected epoch | Dev Top-1 | Dev Top-3 |
| --- | ---: | ---: | ---: | ---: |
| Siamese RNN | 13 | 10 | 0.738 | 0.816 |
| Siamese BiLSTM | 16 | 13 | 0.982 | 0.998 |

These are **development-set ranking scores, not final test accuracy or
correct-answer rates**. They cannot be compared directly with the earlier
models' 200-query test results. Checkpoint metadata is in
[models/ecommerce/](models/ecommerce/) and histories are in
[reports/phase3/](reports/phase3/). Generated `.pt` binaries are Git-ignored.

### Remaining work and confirmed data-quality issue

- The saved e-commerce audit passes its overlap rules, with mean source-token
  coverage 0.7601. The latest saved university audit records `passed: false`,
  coverage 0.7527 and one introduced evaluation-template opening. The current
  generator includes an opening-avoidance fix, but a successful regenerated
  university dataset/audit is not included in this checkpoint.
- **Passing the overlap audit does not establish correct training labels.**
  Saved e-commerce rows replace `different` with `similar`, `buy` with `sell`,
  `more` with `less`, and `one` with `three` while retaining the original FAQ
  label. Word-vector neighbours are not necessarily interchangeable words.
  Review and correct meaning-changing replacements, protect quantities and
  polarity, then rebuild and retrain before freezing Phase 3 results.
- Complete both university models, tune each model/domain threshold on the
  existing validation set, then produce the seven-model comparison on the
  unchanged test sets. No Phase 3 benchmark scores are claimed yet.
- Extend the notebook with Phase 2B and Phase 3 explanations, comparisons and
  demonstrations. Human-written evaluation remains pending for both corpora.

The following is the development workflow **after training-data quality is
resolved**; rebuilding replaces generated paraphrase and pair CSVs:

```powershell
python scripts/build_sequence_pairs.py --corpus all
python scripts/train_sequence_models.py --corpus all --arch all
python evaluate.py all --model phase3
python evaluate.py manual --model phase3
python main.py --corpus ecommerce --model bilstm --query "How can I cancel my order?"
```

Each encoder requires its own `sequence_rnn_config.json` or
`sequence_bilstm_config.json` in the selected corpus directory. Loading checks
the checkpoint, corpus, training-data and pretrained-vector identities; missing
artifacts or thresholds produce an error rather than silently training.

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
| 3 | Cosine similarity for retrieval, and custom Word2Vec sentence vectors (Phase 2) |
| 4 | Pretrained Word2Vec (Phase 2B); Siamese RNN and BiLSTM implementations and preliminary training (Phase 3) |
| 5 | Transformers remain outside the completed phases |

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
after setup, demonstrations and evaluation use local files only. The verified
environment has NLTK 3.10.3, its local corpora, Gensim 4.4.0 and CPU PyTorch
2.14.0+cpu available. The requirements include the PyTorch CPU package index.
Train the local Word2Vec artifacts before running those models or the full
notebook; their binaries are deliberately not included in a Git clone.
Phase 2B also needs `python scripts/download_pretrained_embeddings.py` once,
which downloads about 1.7 GB and writes a 0.9 GB subset.

## Running

```powershell
# Interactive demonstration with corpus and model menus
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

# Train the Phase 2 Word2Vec models, tune them, and compare all three models
python scripts/train_word2vec.py --corpus all
python evaluate.py all --model all
python evaluate.py manual --model all

# Build the Phase 2B subset, tune both pretrained models, and compare all five
python scripts/download_pretrained_embeddings.py
python evaluate.py all --model phase2b
python evaluate.py manual --model phase2b

# Unit tests
python -m unittest discover -s tests
```

Evaluation and non-interactive query commands default to TF-IDF. Both CLIs
accept `--model tfidf`, `w2v_mean`, `w2v_tfidf`, `pw2v_mean`, `pw2v_tfidf`,
`rnn` or `bilstm`. The last two require completed training and frozen thresholds.
Only `evaluate.py` also accepts the comparison groups `all` (TF-IDF and both
custom Word2Vec models, written to `reports/phase2/`) and `phase2b` (all five
models, written to `reports/phase2b/`), plus `phase3` (all seven models,
written to `reports/phase3/`). Interactive `main.py` asks for a model
when omitted. A group reuses every earlier frozen setting and tunes only its
newest models: `evaluate.py all --model all` tunes the two custom Word2Vec
thresholds, and `evaluate.py all --model phase2b` tunes only the two pretrained
ones. The `phase3` group tunes only RNN/BiLSTM thresholds. These groups do not
retune earlier model families or overwrite their phase reports. Manual mode
never tunes.

Use `python evaluate.py test --model all` or `--model phase2b` to regenerate a
complete comparison summary. A current reporting limitation is that a run
selecting a single dense model rewrites that phase's `comparison_report.md`
with only that model, while its introductory text still names every model in
the group. It leaves the paired CSVs from the earlier full comparison in place.
Rerun the full comparison before presenting reports.

The default TF-IDF command `evaluate.py all` runs tuning and testing in order.
Its tuning step writes
`corpus_config.json` before the test set is read in that run. The tuning functions
receive only validation data. This separation does not erase the fact that
benchmark results have been examined while revising the methodology.

Human evaluation is currently pending. The header-only templates and collection
instructions are in [data/manual_evaluation/](data/manual_evaluation/). Target
20 answerable and 10 unanswerable team-written queries per corpus; keep format
examples out of the measured files. Manual mode uses frozen configurations and
writes `manual_<corpus>_evaluation.json` plus `manual_evaluation_report.md`,
separately from the synthetic reports. With a model group, manual JSON reports,
paired CSVs and `manual_comparison_report.md` instead go under that group's
report directory. Empty templates produce no current scores. Team members still
need to collect and independently label the proposed 30 questions per domain.

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
│   ├── evaluation.py           Shared metrics, threshold scoring, paired predictions
│   ├── embedding_utils.py      Mean and TF-IDF-weighted sentence vectors
│   ├── word2vec_training.py    Question-only training and artifact validation
│   ├── word2vec_retrieval.py   Dense cosine ranking and answer delivery
│   ├── word2vec_config.py      Artifact-specific frozen dense-model thresholds
│   ├── pretrained_embeddings.py  Streaming Google News subset, checked loader, coverage
│   ├── retrieval_models.py     Model keys, labels, families and comparison groups
│   ├── sequence_data.py        Training paraphrases, pairs and overlap audit
│   ├── sequence_models.py      Shared RNN and BiLSTM encoders
│   ├── sequence_training.py    Seeded training, checkpoint selection and metadata
│   └── sequence_retrieval.py   Cached sequence vectors and cosine retrieval
├── scripts/
│   ├── prepare_datasets.py     Download, convert, validate the corpora
│   ├── setup_nltk.py           One-time NLTK resource download
│   ├── train_word2vec.py       Seeded subprocess training for each domain
│   ├── download_pretrained_embeddings.py  Google News download and subset build
│   ├── build_sequence_pairs.py  Generate and audit sequence training data
│   └── train_sequence_models.py Train RNN/BiLSTM models per domain
├── data/<corpus>/              faq_dataset.csv, validation_queries.csv,
│                               test_queries.csv, corpus_config.json,
│                               word2vec_config.json, pretrained_config.json
├── data/manual_evaluation/     Empty team-query templates and collection guide
├── models/<corpus>/            Word2Vec training metadata (binaries Git-ignored)
├── models/pretrained/          Subset metadata (download and vectors Git-ignored)
├── reports/                    Tuning sweeps and evaluation results
├── reports/phase2/             Three-model comparison, sweeps and paired CSVs
├── reports/phase2b/            Five-model comparison, error cases and coverage
├── reports/phase3/             Preliminary training histories and data audits
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
  acceptance rate at 0.947. The pretrained mean model rejects the same question
  on the university corpus (0.791 against its 0.86 threshold).
- **Natural paraphrases can fail.** The notebook's *"How do I send a product
  back and get my money returned?"* is rejected by TF-IDF at 0.335, with a
  warranty FAQ ranked first. The pretrained weighted model does worse on it: it
  accepts *"Do I have to return the freebie when I return a product?"* at 0.780,
  just above its 0.76 threshold. It is a failure demonstration, not evidence of
  successful matching.
- **Pretrained vectors answer more unanswerable questions on e-commerce.** The
  pretrained weighted model rejects only 26 of 50 unanswerable test queries.
  Its threshold was tuned on 20 unanswerable validation queries, which is too
  few to estimate that rate precisely.
- **Every dense model still ignores word order.** Mean and TF-IDF-weighted
  vectors are unchanged if the words of a question are shuffled.
- **The evaluation queries are generated, not human-written.** They are
  measurably distinct from their sources (no query reproduces its source
  question, mean source-token overlap 0.59 and 0.77), but some are not fluent.
  Reported accuracy should be read as an estimate on synthetic paraphrases.
- **The university corpus mixes two source types**, which differ in tone and
  answer length.
- **The e-commerce corpus is region-specific** (rupees, PhonePe, SuperCoins).

## Scope

Phase 1 is TF-IDF; Phase 2A adds custom Word2Vec vectors trained on the same FAQ
questions; Phase 2B adds pretrained Google News Word2Vec vectors. All three
retrieve stored answers only. Phase 3 Siamese RNN and BiLSTM code is included
as work in progress, with the outstanding work documented above. Training on
answers, Transformer/BERT, and generative answering are outside the current
implementation. A presentation interface is planned after model validation;
no graphical interface is included yet. It can reuse `main.py:build_answerer`
to expose domain/model selection, stored answers, sources, top matches and
similarity thresholds without retraining during a question submission.
