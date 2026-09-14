# NLP-Based Smart FAQ Retrieval and Question Answering System

A corpus-configurable FAQ retrieval engine for the CSE 4122 Natural Language
Processing Laboratory. Every model **retrieves an existing answer**; nothing is
generated. The same engine runs over two research corpora and a separate
official-source KUET demonstration corpus, each with its own frozen threshold.
Once the data and models are prepared, retrieval and the local GUI run offline.

## Latest implementation at a glance

Seven retrieval models are implemented, trained where needed, tuned on
validation queries only, and evaluated on the same 200 test queries per corpus.

| Phase | Key | Model | How a question becomes a vector |
| --- | --- | --- | --- |
| 1 | `tfidf` | TF-IDF | Sparse word weights fitted on the FAQ questions |
| 2A | `w2v_mean` | Custom Word2Vec mean | Average of Skip-Gram vectors trained on each domain's FAQ questions |
| 2A | `w2v_tfidf` | Custom Word2Vec weighted | Same vectors, TF-IDF-weighted average |
| 2B | `pw2v_mean` | Pretrained Word2Vec mean | Average of Google News `word2vec-google-news-300` vectors |
| 2B | `pw2v_tfidf` | Pretrained Word2Vec weighted | Same vectors, TF-IDF-weighted average |
| 3 | `rnn` | Siamese RNN | Frozen Google News vectors read in order by a trained vanilla RNN |
| 3 | `bilstm` | Siamese BiLSTM | Frozen Google News vectors read in order by a trained two-layer BiLSTM |

**Headline result on the synthetic test set:** TF-IDF still delivers the most
correct answers on university (0.833), and pretrained Word2Vec weighted narrowly
leads on e-commerce (0.840 against 0.820 for TF-IDF, at the cost of many more
false acceptances). The Phase 3 sequence encoders are the weakest models in this
experiment (correct answer rate 0.347–0.533), despite near-perfect scores on
their own development paraphrases. See [Phase 3](#phase-3-siamese-rnn-and-bilstm).

| Status | Item |
| --- | --- |
| Done | All seven models, thresholds, seven-model comparison reports, terminal demo for every model |
| Done | Reviewed 200-FAQ KUET corpus, frozen TF-IDF configuration and separate development checks |
| Done | Streamlit FAQ Assistant and separate seven-model Model Comparison page |
| Done | 144 tests passing, including Streamlit AppTest and model/corpus isolation checks |
| Pending | Human-written evaluation queries (templates are still empty) |

| Dataset count | University research | E-commerce research | KUET demo |
| --- | ---: | ---: | ---: |
| FAQs | 500 | 500 | 200 |
| Categories | 14 | 10 | 20 |
| Validation queries | 50 (30/20) | 50 (30/20) | 100 (70/30) |
| Final test / development smoke | 200 (150/50) | 200 (150/50) | 60 (45/15) |
| Phase 3 training / dev paraphrases | 1,000 / 500 | 1,000 / 500 | Not applicable |

Counts in parentheses are answerable/unanswerable. KUET's validation and smoke
queries are developer-authored and are not an independent human evaluation.

## Seven-model results

Each model uses its own threshold, selected on validation queries only and
frozen before the test queries were read. Test set: 150 answerable paraphrases
and 50 unanswerable queries per corpus. The best value in each row is bold.

### University FAQ

| Metric | TF-IDF | W2V mean | W2V weighted | Pretrained mean | Pretrained weighted | Siamese RNN | Siamese BiLSTM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold | 0.46 | 0.93 | 0.95 | 0.86 | 0.80 | 0.80 | 0.75 |
| Top-1 accuracy | 0.860 | 0.587 | 0.693 | **0.873** | 0.867 | 0.553 | 0.633 |
| Top-3 accuracy | **0.993** | 0.673 | 0.773 | 0.953 | 0.987 | 0.593 | 0.720 |
| Correct answer rate | **0.833** | 0.493 | 0.493 | 0.627 | 0.813 | 0.420 | 0.347 |
| Accepted but wrong | 17 | 17 | 9 | 3 | 15 | 2 | **1** |
| Answerable acceptance | **0.947** | 0.607 | 0.553 | 0.647 | 0.913 | 0.433 | 0.353 |
| Unanswerable rejection | 0.840 | 0.760 | 0.980 | 0.960 | 0.880 | **1.000** | 0.960 |
| False acceptances | 8 | 12 | 1 | 2 | 6 | **0** | 2 |
| False rejections | **8** | 59 | 67 | 53 | 13 | 85 | 97 |

### E-commerce FAQ

| Metric | TF-IDF | W2V mean | W2V weighted | Pretrained mean | Pretrained weighted | Siamese RNN | Siamese BiLSTM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold | 0.58 | 0.92 | 0.91 | 0.81 | 0.76 | 0.69 | 0.70 |
| Top-1 accuracy | **0.953** | 0.753 | 0.740 | 0.827 | 0.867 | 0.607 | 0.507 |
| Top-3 accuracy | **0.980** | 0.860 | 0.880 | 0.913 | 0.927 | 0.707 | 0.640 |
| Correct answer rate | 0.820 | 0.633 | 0.687 | 0.787 | **0.840** | 0.533 | 0.347 |
| Accepted but wrong | **1** | 16 | 21 | 18 | 17 | 23 | 7 |
| Answerable acceptance | 0.827 | 0.740 | 0.827 | 0.907 | **0.953** | 0.687 | 0.393 |
| Unanswerable rejection | **0.960** | 0.940 | 0.920 | 0.760 | 0.520 | 0.740 | **0.960** |
| False acceptances | **2** | 3 | 4 | 12 | 24 | 13 | **2** |
| False rejections | 26 | 39 | 26 | 14 | **7** | 47 | 91 |

**Correct answer rate** is the number of answerable queries whose correct FAQ
is ranked first **and accepted**, divided by all answerable queries; Top-1
alone ignores threshold rejection. **Accepted but wrong** counts answerable
queries answered with a different FAQ; **false acceptances** count unanswerable
queries that receive any answer.

No model is best on every metric. Which to prefer depends on whether a wrong
answer or a refusal costs more. These are **synthetic benchmark** results: the
test queries were generated from templates and inspected during development,
so they are a reproducible estimate, not a final human-rated assessment.
Full reports are in [reports/](reports/), [reports/phase2/](reports/phase2/),
[reports/phase2b/](reports/phase2b/) and [reports/phase3/](reports/phase3/).

## Phase 1: TF-IDF

```text
FAQ questions -> preprocess -> fit_transform()   (index built once)
User query    -> same preprocessing -> transform()
Query vector  -> cosine similarity -> top-k ranking -> threshold
```

Answers are never included in the similarity vectors — only questions are
indexed. A query whose best similarity falls below the corpus threshold is
rejected. Empty and all-out-of-vocabulary queries return no matches, even at
threshold 0.00, and receive no Top-1/Top-3 credit from zero-score ties.

Selection has two stages, both using validation data only:

1. Compare answerable Top-1 accuracy, then Top-3, then prefer the simpler
   preprocessing configuration. University basic and lemmatized both reach
   0.900 Top-1 and 1.000 Top-3; basic wins the simplicity tie. E-commerce ties
   across all three configurations at 0.900, also selecting basic.
2. For that configuration, sweep thresholds from 0.00 to 1.00 in 0.01 steps.
   Maximize the mean of answerable acceptance and unanswerable rejection;
   prefer the higher threshold on ties.

Every later model reuses step 2 for its own threshold. The comparison supports
basic preprocessing for these validation sets; it does not establish that
lemmatization never helps.

## Phase 2A: custom Word2Vec

One Skip-Gram Word2Vec model per domain is trained on that domain's **FAQ
questions only** (6,257 university and 6,665 e-commerce tokens). Both
aggregations reuse the same trained vectors.

**TF-IDF beats both custom models on both corpora.** The training text is small,
and the averaged vectors crowd scores together: the mean best-match cosine
separates answerable from unanswerable queries by 0.34–0.37 for TF-IDF but only
0.03–0.09 for custom Word2Vec, so validation picks thresholds of 0.91–0.95 that
reject correct matches. On university, the weighted model ranks 104 of 150
answerable queries first but delivers only 74.

Thresholds live in `data/<corpus>/word2vec_config.json`, keyed to the trained
model's artifact id; `models/<corpus>/training_metadata.json` records corpus
hash, settings, package versions and vector hash. Model binaries are
Git-ignored.

## Phase 2B: pretrained Google News Word2Vec

Phase 2B keeps the Phase 2A aggregation code and swaps in
`word2vec-google-news-300`, the pretrained vectors also used by Lab 4. Only the
source of the word vectors changes.

**Pretrained vectors beat the custom vectors for every aggregation and corpus.**
Against TF-IDF the result is mixed: on university the pretrained models rank
131 and 130 of 150 answerable queries first against 129, but TF-IDF still
delivers more correct answers; on e-commerce the weighted model delivers the
most correct answers but answers 24 of 50 unanswerable queries.

Two measurements help explain the gain over custom vectors:

1. **Coverage.** Custom models know 81.8% (university) and 79.3% (e-commerce)
   of test-query tokens; the pretrained subset knows 99.5% and 98.8%. Its
   unknown words are mostly brand names (`supercoins`, `bgauss`, `phonepe`)
   and numbers.
2. **Score separation.** 0.13–0.17 for the pretrained models, against
   0.03–0.09 for the custom ones.

| Model | Case A: TF-IDF and custom fail, pretrained answers | Case B: TF-IDF answers, pretrained does not | Case C: custom answers, pretrained does not |
| --- | ---: | ---: | ---: |
| University, pretrained mean | 2 | 34 | 6 |
| University, pretrained weighted | 6 | 10 | 1 |
| E-commerce, pretrained mean | 9 | 18 | 8 |
| E-commerce, pretrained weighted | 9 | 10 | 6 |

**How the vectors are stored.** Google News keys are case-sensitive and 71% are
phrases such as `New_York`. Because every token is lowercased, the build script
keeps only lowercase single-word keys (710,048 of 3,000,000); when casings fold
together, the more frequent one supplies the vector. Google News has no
lowercase `a`, `to`, `of`, `and` or `flipkart`, only `A`, `To`, `Of`, `And` and
`Flipkart`. The 852 MB subset is memory-mapped. Thresholds live in
`data/<corpus>/pretrained_config.json`; see
[models/pretrained/README.md](models/pretrained/README.md).

## Phase 3: Siamese RNN and BiLSTM

### Architecture and training

Both encoders read the **same frozen** 300-dimensional Google News vectors as
Phase 2B, so Phase 2B and Phase 3 differ only in how word vectors become one
question vector: averaged, or read in word order.

- **Shared (Siamese) encoder:** the query and the FAQ question pass through the
  same weights. The RNN outputs its final 128-dimensional hidden state; the
  two-layer BiLSTM concatenates its top forward and backward states (256
  dimensions). Packed sequences keep padding out of the final state.
- **Objective:** `BCEWithLogitsLoss` on `scale * cosine + bias`, with a learned
  scale and bias; Adam, learning rate 0.001, batch 32, at most 20 epochs,
  early-stopping patience 3, seed 42, one CPU thread.
- **Vocabulary:** `<PAD>`, `<UNK>`, the 100,000 most frequent subset words and
  any other training word the subset knows (100,028 university, 100,013
  e-commerce entries); maximum length 28 tokens.
- **Checkpoint selection** uses held-out **development paraphrases** only
  (highest Top-1, then Top-3, then lowest dev loss). Validation queries are
  logged during training but only choose thresholds afterwards. Test queries
  never enter training or selection.

### Training data

Each FAQ gets three reworded questions: two for training, one for development.
Every paraphrase yields one positive pair, one random negative and one TF-IDF
hard negative (near-duplicate FAQ questions are never negatives): 3,000
training and 1,500 development pairs per corpus.

The generator shares no code with the one that built the test queries, and it
must not change a question's meaning:

- A word is replaced only by a **WordNet synonym** that the pretrained vectors
  place close to it (cosine ≥ 0.35) and that fits the rest of the question
  nearly as well as the original word (within 0.10).
- **Never replaced:** WordNet antonyms, numbers, time and money units, polarity
  and order words (more/less, same/different, before/after), capitalised
  programme or brand names, and a question's first word.
- Conversational frames ("Quick question: …?") and 25% function-word dropout
  add variety.

An earlier version used raw nearest neighbours, which swapped related but
different words (buy/sell, more/less, one/three, seller/buyer) while keeping
the FAQ label; those checkpoints were discarded and every model retrained.

A **blocking audit** checks each corpus before pairs are written:

| Audit | University | E-commerce |
| --- | ---: | ---: |
| Rows removed for matching or ≥ 0.8 Jaccard overlap with any evaluation query | 0 | 0 |
| Highest Jaccard overlap with any evaluation query | 0.786 | 0.733 |
| Introduced test-template openings | 0 | 0 |
| Substitutions reused from the test generator | 0 | 0 |
| Mean source-token coverage (limit 0.80) | 0.768 | 0.748 |
| Substitutions made (distinct pairs) | 2,517 (472) | 3,153 (479) |

Passing the audit shows no leakage; it does not guarantee correct labels. Some
WordNet senses still misfire in short questions, for example battery
*charge → accusation* and *card → scorecard*.

### Results

| Model | Epochs run / selected | Dev Top-1 | Dev Top-3 | Validation Top-1 (logged) | Test Top-1 | Test correct answer rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| University RNN | 16 / 13 | 0.790 | 0.862 | 0.700 | 0.553 | 0.420 |
| University BiLSTM | 12 / 9 | 0.988 | 0.998 | 0.700 | 0.633 | 0.347 |
| E-commerce RNN | 16 / 13 | 0.746 | 0.850 | 0.533 | 0.607 | 0.533 |
| E-commerce BiLSTM | 9 / 6 | 0.976 | 0.992 | 0.567 | 0.507 | 0.347 |

**The sequence encoders do not beat averaging the same pretrained vectors.**
On both corpora they have the lowest Top-1 accuracy and correct answer rate of
all seven models. The BiLSTM is almost perfect on development paraphrases
(0.976–0.988 Top-1) but falls to 0.507–0.633 on the test queries. Development
paraphrases come from the same generator as the training data, so this gap
suggests the encoders learned that generator's style rather than general
paraphrasing. About 1,000 positive pairs per domain is also little data for
recurrent weights, and some training labels are noisy. The experiment does not
isolate a single cause.

The sequence models are conservative: university RNN rejects all 50
unanswerable queries, and both BiLSTMs make only 1–7 accepted-but-wrong
answers, but they reject far more answerable queries (85–97 false rejections
on university). Measured against Pretrained Word2Vec mean, the model that
averages the same vectors:

| Model | Case A: TF-IDF and pretrained mean fail, sequence answers | Case B: TF-IDF answers, sequence does not | Case C: pretrained mean answers, sequence does not |
| --- | ---: | ---: | ---: |
| University, Siamese RNN | 0 | 62 | 35 |
| University, Siamese BiLSTM | 0 | 73 | 43 |
| E-commerce, Siamese RNN | 1 | 48 | 43 |
| E-commerce, Siamese BiLSTM | 0 | 73 | 68 |

Checkpoint metadata is in [models/university/](models/university/) and
[models/ecommerce/](models/ecommerce/); training histories, audits, sweeps,
paired predictions and error cases are in [reports/phase3/](reports/phase3/).
Thresholds live in `data/<corpus>/sequence_rnn_config.json` and
`sequence_bilstm_config.json`, keyed to each checkpoint's artifact id. `.pt`
binaries are Git-ignored; loading refuses a missing, stale or mismatched
checkpoint instead of retraining.

## Relationship to the lab topics

| Lab | Topic used |
| --- | --- |
| 1 | Regex cleaning, tokenization, optional stopword removal and lemmatization |
| 2 | TF-IDF representation using scikit-learn |
| 3 | Cosine similarity and custom Word2Vec sentence vectors (Phase 2A) |
| 4 | Pretrained Google News Word2Vec (Phase 2B); `nn.Embedding.from_pretrained`, vanilla RNN, stacked BiLSTM, packed sequences and `BCEWithLogitsLoss` (Phase 3) |
| 5 | Transformers are not implemented |

The implementation follows these topics without copying the lab code. We use
basic WordNet lemmatization without POS tagging: `books` becomes `book` but
`running` remains `running`.

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_nltk.py
```

`requirements.txt` includes the PyTorch CPU package index (verified with
PyTorch 2.14.0+cpu, Gensim 4.4.0 and NLTK 3.10.3). `scripts/setup_nltk.py`
downloads `stopwords`, `wordnet` and `omw-1.4` into `.nltk_data/`; WordNet is
required to rebuild the Phase 3 training data. Model binaries are not in a Git
clone, so train or build them before using those models:

```powershell
python scripts/train_word2vec.py --corpus all                     # Phase 2A
python scripts/download_pretrained_embeddings.py                  # Phase 2B, ~1.7 GB download, 0.9 GB subset
python scripts/build_sequence_pairs.py --corpus all               # Phase 3 training data and audit
python scripts/train_sequence_models.py --corpus all --arch all   # Phase 3, about 10 minutes on CPU
```

After setup, demonstrations and evaluation use local files only.

Start the localhost-only GUI from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

The **FAQ Assistant** opens on KUET and uses TF-IDF only. It shows a stored
answer and official source only when the frozen threshold accepts the match.
The **Model Comparison** tab runs the seven research models on one query. RNN
and BiLSTM are labeled experimental. Heavy models load only after Compare is
pressed, and cached indexes are invalidated when their source files change.

## Running

```powershell
# Interactive demonstration with corpus and model menus
python main.py

# One question with a chosen model (default model: tfidf)
python main.py --corpus kuet --query "How many books may an undergraduate borrow?"
python main.py --corpus university --query "How do I receive university alerts?"
python main.py --corpus ecommerce --model pw2v_tfidf --query "What cards can I save on Flipkart?"

# Failure example: the BiLSTM answers with "How can I pay for my order?" (0.757)
# and ranks the correct "How long does it take to cancel an order?" second (0.756)
python main.py --corpus ecommerce --model bilstm --query "How can I cancel my order?"

# Tune thresholds on validation queries, then evaluate on the test queries
python evaluate.py all                    # Phase 1 TF-IDF
python evaluate.py all --model all        # Phase 2A: TF-IDF + custom Word2Vec -> reports/phase2/
python evaluate.py all --model phase2b    # + pretrained Word2Vec -> reports/phase2b/
python evaluate.py all --model phase3     # all seven models -> reports/phase3/

# Re-run a comparison with frozen thresholds, without tuning
python evaluate.py test --model phase3

# KUET is explicit and writes only to reports/demo/
python evaluate.py tune --corpus kuet --model tfidf
python evaluate.py test --corpus kuet --model tfidf

# Type questions and receive answers or None (no FAQ IDs needed)
python evaluate.py manual --corpus university
python evaluate.py manual --corpus university --query "How is the Curriculum and Pedagogy program structured?"

# Score labeled team-written CSV queries with all seven models
python evaluate.py human-benchmark --model phase3

# Validate the finalized data, and run the unit tests
python scripts/prepare_datasets.py validate
python scripts/prepare_datasets.py validate --domain kuet
python -m unittest discover -s tests
```

Interactive `manual` testing defaults to TF-IDF and asks you to select a corpus
when `--corpus` is omitted. Type any question; the program shows the stored answer
or `Result: None`, similarity, acceptance threshold, and up to three ranked FAQ
questions. Rejected candidates are marked unaccepted. Questions with no usable
features have no candidates. Type `exit` or `quit` to finish. This mode reads no
evaluation CSV, writes no performance report, and does not calculate accuracy.
Similarity is a matching score, not a probability that an answer is correct.

Both CLIs accept `--model tfidf`, `w2v_mean`, `w2v_tfidf`, `pw2v_mean`,
`pw2v_tfidf`, `rnn` or `bilstm`. `evaluate.py` also accepts the groups `all`,
`phase2b` and `phase3` for benchmark modes; `manual` requires one model and one
corpus. Batch `--corpus all` includes research corpora only. KUET must be
selected explicitly, supports TF-IDF only, and rejects incompatible models
before loading artifacts. A group reuses every earlier frozen setting and tunes
only its newest models, so no group retunes an earlier phase or overwrites its
reports. Neither `manual` nor `human-benchmark` tunes. A run selecting a single dense or sequence
model rewrites that phase's `comparison_report.md` with only that model; rerun
the full group before presenting reports.

The former `evaluate.py manual` CSV workflow is now `evaluate.py human-benchmark`.
This optional scientific benchmark is for annotators who independently establish
correct FAQ IDs, allowing accuracy to be measured. Ordinary testers can use
`manual` without knowing the corpus contents. Collection instructions are in
[data/manual_evaluation/](data/manual_evaluation/): target 20 answerable and
10 unanswerable team-written queries per corpus. Existing CSVs and `manual_*`
report filenames are retained; header-only files produce no scores.

## Project layout

```text
Smart_FAQ/
├── app.py                      Streamlit FAQ Assistant and Model Comparison
├── main.py                     Terminal demonstration for every model
├── evaluate.py                 Threshold tuning, evaluation and comparison reports
├── src/
│   ├── data_loader.py          Corpus discovery, loading, schema validation
│   ├── gui_support.py          GUI corpus, cache-signature and report helpers
│   ├── preprocessing.py        Normalization, optional stop words / lemmas
│   ├── tfidf_retrieval.py      TF-IDF index, cosine ranking, thresholding
│   ├── evaluation.py           Shared metrics, threshold sweep, paired predictions
│   ├── embedding_utils.py      Mean and TF-IDF-weighted sentence vectors
│   ├── word2vec_training.py    Question-only Word2Vec training and artifact checks
│   ├── word2vec_retrieval.py   Dense cosine ranking and answer delivery
│   ├── word2vec_config.py      Artifact-keyed frozen thresholds
│   ├── pretrained_embeddings.py  Google News subset builder, checked loader, coverage
│   ├── retrieval_models.py     Model keys, labels, families and comparison groups
│   ├── sequence_data.py        Synonym paraphrases, pairs and leakage audit
│   ├── sequence_models.py      Vocabulary, frozen embeddings, RNN/BiLSTM encoders
│   ├── sequence_training.py    Seeded training, dev checkpoint selection, metadata
│   └── sequence_retrieval.py   FAQ vector index and cosine retrieval for encoders
├── scripts/
│   ├── prepare_datasets.py     Download, convert, validate the corpora
│   ├── setup_nltk.py           One-time NLTK resource download
│   ├── train_word2vec.py       Seeded Word2Vec training per domain
│   ├── download_pretrained_embeddings.py  Google News download and subset build
│   ├── build_sequence_pairs.py  Phase 3 training data and blocking audit
│   └── train_sequence_models.py Seeded RNN/BiLSTM training per domain
├── data/<corpus>/              FAQs, validation/test queries, every model's frozen
│                               thresholds, Phase 3 paraphrases and pairs
├── data/kuet/                  Official FAQs, source review, validation and smoke queries
├── data/manual_evaluation/     Empty team-query templates and collection guide
├── models/<corpus>/            Word2Vec and RNN/BiLSTM metadata (binaries Git-ignored)
├── models/pretrained/          Subset metadata (download and vectors Git-ignored)
├── reports/                    Phase 1 tuning and evaluation
├── reports/phase2/             Three-model comparison
├── reports/phase2b/            Five-model comparison, error cases and coverage
├── reports/phase3/             Seven-model comparison, audits, histories, error cases
├── reports/demo/               KUET tuning and development smoke results
├── notebooks/                  Teacher notebook and KUET demonstration
├── docs/DATA_SOURCES.md        Corpus sources and manual review record
└── tests/                      Unit and Streamlit AppTest coverage
```

FAQ schema: `id,question,answer,category,source,source_type`
Query schema: `query,expected_faq_id,is_answerable`

## Data

Both research corpora hold 500 reviewed FAQs. The university corpus combines a screened
subset of the [CPath dataset](https://huggingface.co/datasets/houcine-bdk/cpath-mcgill-ubc)
with FAQs scraped from official `utoronto.ca` and `ubc.ca` pages; the e-commerce
corpus comes from [NebulaByte/E-Commerce_FAQs](https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs).
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) records every source, what was
rejected during review and why, the seven flagged duplicate pairs, and the
measured quality of the generated paraphrases.

The KUET demo contains 200 distinct English FAQs supported by official KUET,
Central Library, Admission Portal, Academic System, BHTPA and club-owned pages. The expansion adds 20 department overviews, CSE details, department milestones, IT park information and 10 club/activity FAQs. Conflicting leadership claims are excluded; four corroborated departmental heads are dated. Some club descriptions explicitly use historical official records. Its companion
[source review](data/kuet/SOURCE_REVIEW.md) records the verification date,
official URL and reviewed section for every FAQ ID. Runtime retrieval reads the
local CSV and does not browse the web.

To rebuild the corpora from source (network needed; overwrites the finalized
CSVs):

```powershell
python scripts/prepare_datasets.py download
python scripts/prepare_datasets.py convert
python scripts/prepare_datasets.py validate
```

## Limitations

- **The evaluation queries are generated, not human-written.** They differ from
  their sources (mean source-token overlap 0.59 and 0.77, no verbatim copies)
  but some are not fluent. Every number above is a synthetic-benchmark estimate.
- **TF-IDF matches words, not meaning.** E-commerce rejects 26 of 150 answerable
  queries whose wording differs from the FAQ.
- **One threshold cannot separate every case.** *"How do I train a puppy to
  sit?"* scores 0.470 on the university corpus, just above the 0.46 TF-IDF
  threshold, and is wrongly answered with FAQ 241 through shared words such as
  *do*, *I*, *a*, *to*. The pretrained mean model rejects it (0.791 against 0.86).
- **Natural paraphrases can fail.** *"How do I send a product back and get my
  money returned?"* is rejected by TF-IDF at 0.335; the pretrained weighted
  model wrongly accepts a freebie-return FAQ at 0.780.
- **Pretrained weighted answers many unanswerable e-commerce queries** (24 of
  50); its threshold came from only 20 unanswerable validation queries.
- **Averaged models ignore word order**, and the sequence models that do not
  ignore it generalize poorly from about 1,000 generated training pairs per
  domain, some of them with wrong-sense substitutions.
- **The university corpus mixes two source types**, and the e-commerce corpus
  is region-specific (rupees, PhonePe, SuperCoins).
- **KUET information can change.** Admission-session facts are dated to the
  source review. Users should follow the linked current notice for deadlines,
  fees and eligibility. The demo covers its 200 documented intents.

## Scope

Implemented: TF-IDF, custom and pretrained Word2Vec, Siamese RNN/BiLSTM,
stored-answer retrieval, the official-source KUET demo, reproducible evaluation,
and a localhost Streamlit interface. Deferred: general e-commerce demo data and
formal human evaluation. The project does not train on answers, generate answers,
or include Transformer/BERT models, authentication, deployment or a feedback database.

## KUET human evaluation (14 September 2026)

Run `.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501 --server.headless true` and open http://localhost:8501. Type questions naturally, assess each response, and use **Save evaluation**. Download the evaluation CSV before closing or reloading: ratings are held only in that browser session. No FAQ IDs are required from the evaluator.

The 100 validation queries and 60 smoke queries are developer-authored. The expanded smoke check retrieved the intended FAQ first for 42/45 answerable questions, gave accepted correct answers for 41/45, and falsely accepted 7/15 unsupported questions. These are development results, not human validation. Similarity-based retrieval can confuse departments, leadership roles, unknown clubs and schedules; verify against the displayed source and record incorrect answers. See [development report](reports/demo/evaluation_report.md).

Source snapshots can be refreshed with `scripts/snapshot_kuet_sources.py` using explicit official paths. This script only collects evidence; it never automatically edits FAQs. Review source changes before editing the CSV. The two dated expansion scripts reproduce this development snapshot and should not be rerun over later manual curation.
