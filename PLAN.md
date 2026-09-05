# Smart FAQ: Phase-1 Design and Refinement Plan

## Summary

The existing TF-IDF architecture is retained as the undergraduate NLP lab project.
The local corpora have been collected, filtered, reshaped, and reviewed as
documented in `docs/DATA_SOURCES.md`. This refinement does not rebuild them.

The running FAQ application will **not load data directly from Hugging Face**. It will always read the finalized local CSV files, making the teacher demonstration offline, fast, and reproducible.

- University source: [CPath university Q&A dataset](https://huggingface.co/datasets/houcine-bdk/cpath-mcgill-ubc).
- E-commerce source: [NebulaByte E-Commerce FAQs](https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs).
- Target: 500 reviewed FAQs per domain.
- If filtering leaves fewer than 500 suitable entries, fill the shortfall only with source-backed FAQs from official public FAQ pages—not repetitive generated filler.
- Keep the official project title while describing the engine as corpus-configurable.

## Corpus Preparation and Data Design

The existing `scripts/prepare_datasets.py` has three preparation modes:

1. Download raw datasets into an ignored staging/cache directory.
2. Convert source columns into the common project schema.
3. Validate finalized CSVs without downloading again.

Conversion rules:

- CPath: `instruction → question`, `output → answer`, preserve `source_url`, remove “As CPath” boilerplate, reject malformed, excessively long, outdated, irrelevant, or mismatched entries.
- NebulaByte: retain `question`, `answer`, `category`, and `faq_url`; exclude blank rows, travel/insurance/loan material, and unrelated support areas; consolidate source categories into understandable e-commerce categories.
- Automatically reject duplicate IDs, exact duplicate questions, and normalized duplicate questions.
- Flag high-TF-IDF-similarity pairs as possible semantic duplicates for manual review; do not automatically claim that they are true duplicates.
- Preserve source wording and meaning. The engine is generic, but individual records may mention their originating university or store.

Final FAQ schema:

```text
id,question,answer,category,source,source_type
```

Query schema:

```text
query,expected_faq_id,is_answerable
```

Each domain contains:

- 500 finalized FAQs.
- 50 validation queries: 30 answerable paraphrases and 20 unanswerable queries.
- 200 synthetic benchmark test queries: 150 answerable paraphrases and 50 unanswerable queries.
- `corpus_config.json` containing display name, selected preprocessing configuration, and validation-selected threshold.

```text
data/
├── university/
│   ├── faq_dataset.csv
│   ├── validation_queries.csv
│   ├── test_queries.csv
│   └── corpus_config.json
└── ecommerce/
    ├── faq_dataset.csv
    ├── validation_queries.csv
    ├── test_queries.csv
    └── corpus_config.json
```

Paraphrases must be written only after the FAQ corpus is frozen. They must change sentence structure and wording rather than merely replacing “how can I” with “how do I.”

## TF-IDF Engine and Evaluation

Retain these class-free, viva-friendly interfaces:

- `discover_corpora(data_root)`
- `load_faq_dataset(corpus_dir)`
- `load_query_dataset(path, faq_ids=None, *, allow_empty=False)`
- `preprocess_text(text, remove_stopwords=False, lemmatize=False)`
- `build_tfidf_index(faq_data, preprocessing_options)`
- `retrieve_tfidf(query, faq_data, vectorizer, faq_matrix, top_k=3)`
- `answer_query(..., threshold, top_k=3)`
- `select_preprocessing(validation_data, faq_data, configs=None, top_k=3)`
- `tune_threshold(validation_data, ...)`
- `evaluate_tfidf(test_data, ...)`

Processing flow:

```text
FAQ questions → preprocess → fit_transform()
User query    → same preprocessing → transform()
Query vector  → cosine similarity → top-k ranking → threshold
```

Answers are never included in the similarity vectors. Empty or all-OOV query
vectors return no matches at every threshold. Batch evaluation uses a
`has_features` mask for both acceptance and Top-1/Top-3 correctness; zero-score
sorting placeholders must not be counted as retrieved answers.

Use basic normalization as the initial default:

```python
remove_stopwords=False
lemmatize=False
```

Compare three TF-IDF preprocessing configurations:

1. Basic normalization.
2. Basic normalization plus stop-word removal.
3. Basic normalization plus lemmatization.

For each corpus, select preprocessing and threshold using validation data only:

1. **Step A:** `select_preprocessing` compares answerable validation Top-1
   accuracy, then Top-3 accuracy, then configuration simplicity in the order
   above. Unanswerable queries and thresholds do not influence this stage.
2. **Step B:** `tune_threshold` orchestrates Step A, then sweeps thresholds from
   `0.00` to `1.00` in `0.01` steps only for the selected representation. Maximize
   the mean of answerable acceptance and unanswerable rejection, choosing the
   higher threshold on ties. Both answerability classes are required for tuning.

Freeze the selected values in `corpus_config.json` before benchmark evaluation.
Keep existing selected-result fields; add `preprocessing_comparison` to tuning
results and reports. The `sweep` now holds 101 rows for one configuration and
`per_config_best` holds its single best threshold row.

This supersedes the original combined criterion, which could reward an accepted
wrong FAQ during preprocessing selection. It does not redefine acceptance as
retrieval correctness; the two are reported separately. Expected results from
the current validation sets are basic at 0.46 for university and basic at 0.58
for e-commerce. These are computed results, not constants to force in code.

Report per corpus:

- Top-1 accuracy on answerable queries.
- Top-3 accuracy on answerable queries.
- Correct answer rate: `(correct Top-1 AND accepted) / all answerable queries`.
- Accepted-wrong count: answerable queries accepted with the wrong FAQ.
- Mean similarity for correct Top-1 matches.
- Answerable acceptance rate.
- Unanswerable rejection rate.
- False acceptance count.
- False rejection count.
- Incorrect retrieval examples.

## Human Evaluation and Benchmark Interpretation

The synthetic test queries have been inspected during development and review.
Describe their results as reproducible synthetic benchmarks, not an untouched,
one-time final assessment. Never use the test queries inside the selection code.

`data/manual_evaluation/` contains header-only `university_queries.csv` and
`ecommerce_queries.csv`, using the existing query schema. The README carries
format examples outside the measured files. Team members should write queries
from intent descriptions without seeing FAQ wording and independently verify
labels. Target 20 answerable + 10 unanswerable queries per corpus; actual human
authorship and completion remain team responsibilities.

`python evaluate.py manual [--corpus university|ecommerce]` uses frozen settings
without tuning. Header-only templates report human evaluation pending and
produce no current metrics. Populated files validate labels and domain-specific
FAQ ids, then write separate `manual_<corpus>_evaluation.json` files and
`manual_evaluation_report.md`. Synthetic reports and configurations are preserved.
The default query loader still rejects empty validation/test files.

## Teaching Material and Implementation Status

The initial scaffold, corpus preparation, engine, evaluation and notebook exist
in the project Git root `Smart_FAQ`. This refinement updates them locally;
no Git commit, push, deployment, or corpus rebuild is included.

- Lab 1 supplies preprocessing topics; Lab 2 supplies TF-IDF; Lab 3 supplies
  cosine similarity. Follow those topics without requiring identical lab code.
- Keep noun-default WordNet lemmatization as an optional comparison, accurately
  described as basic lemmatization without POS tagging. Both frozen corpora now
  use basic preprocessing. NLTK remains a runtime package dependency; downloaded
  resources support optional preprocessing, comparison tests, and the notebook.
- Explain raw term counts, smoothed IDF, L2 normalization, cosine similarity and
  fit versus transform with a numerical notebook example checked against the
  production vectorizer. Use the same preprocessing for the vocabulary check.
- Replace unsupported claims about lemmatization with the measured validation
  ties. Refresh reports, score ranges and demonstration FAQ ids/scores; explicitly
  label the rejected returns paraphrase and the university puppy false acceptance.

## Verification and Boundaries

- Install stable Python 3.12-compatible packages first; after successful installation and testing, freeze the verified environment into `requirements.txt`.
- Validate counts, schemas, sources, IDs, blank values, expected-ID references, and duplicate questions.
- Test exact matches, paraphrases, top-k ordering, empty input, OOV input, invalid corpora, threshold boundaries, false acceptance, and false rejection.
- Verify runtime/batch agreement, OOV metric masking, two-stage selection and
  ties, correct-answer arithmetic, and empty/populated manual evaluation.
- Run unit tests, `compileall`, `pip check`, both evaluation commands, terminal demonstrations for both corpora, and headless notebook execution.
- After preparation, the system must run without internet access.
- Keep finalized FAQ and synthetic query CSVs unchanged. Data validation checks
  structure and consistency, not the semantic truth of every answer or paraphrase.
- Phase 2 Word2Vec and later Transformer/BERT work will reuse the same corpora and evaluation sets but require separate approval and Git pushes.
- Do not add RNN/LSTM, GUI, Flask, Streamlit, or generative-answer behavior during Phase 1.
