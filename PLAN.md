# Revised Domain-Adaptive Smart FAQ Plan

## Summary

The corpora will be **downloaded once from Hugging Face, trimmed, reshaped, and manually reviewed**.

The running FAQ application will **not load data directly from Hugging Face**. It will always read the finalized local CSV files, making the teacher demonstration offline, fast, and reproducible.

- University source: [CPath university Q&A dataset](https://huggingface.co/datasets/houcine-bdk/cpath-mcgill-ubc).
- E-commerce source: [NebulaByte E-Commerce FAQs](https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs).
- Target: 500 reviewed FAQs per domain.
- If filtering leaves fewer than 500 suitable entries, fill the shortfall only with source-backed FAQs from official public FAQ pages—not repetitive generated filler.
- Keep the official project title while describing the engine as corpus-configurable.

## Corpus Preparation and Data Design

Create `scripts/prepare_datasets.py` with three modes:

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
- 200 final test queries: 150 answerable paraphrases and 50 unanswerable queries.
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

Implement these class-free, viva-friendly interfaces:

- `discover_corpora(data_root)`
- `load_faq_dataset(corpus_dir)`
- `load_query_dataset(path)`
- `preprocess_text(text, remove_stopwords=False, lemmatize=False)`
- `build_tfidf_index(faq_data, preprocessing_options)`
- `retrieve_tfidf(query, faq_data, vectorizer, faq_matrix, top_k=3)`
- `answer_query(..., threshold, top_k=3)`
- `tune_threshold(validation_data, ...)`
- `evaluate_tfidf(test_data, ...)`

Processing flow:

```text
FAQ questions → preprocess → fit_transform()
User query    → same preprocessing → transform()
Query vector  → cosine similarity → top-k ranking → threshold
```

Answers are never included in the similarity vectors.

Use basic normalization as the initial default:

```python
remove_stopwords=False
lemmatize=False
```

Compare three TF-IDF preprocessing configurations:

1. Basic normalization.
2. Basic normalization plus stop-word removal.
3. Basic normalization plus lemmatization.

For each corpus, select preprocessing and threshold using validation data only. Sweep thresholds from `0.00` to `1.00` in `0.01` steps and maximize the average of:

- Correctly accepted answerable-query rate.
- Correctly rejected unanswerable-query rate.

Resolve ties by preferring the simpler preprocessing configuration and then the higher threshold. Freeze the selected values in `corpus_config.json` before evaluating the final test set.

Report per corpus:

- Top-1 accuracy on answerable queries.
- Top-3 accuracy on answerable queries.
- Mean similarity for correct Top-1 matches.
- Answerable acceptance rate.
- Unanswerable rejection rate.
- False acceptance count.
- False rejection count.
- Incorrect retrieval examples.

## Implementation and Git Phases

Create the Git root at `D:\4.1\NLP\lab\Smart_FAQ`. The target [GitHub repository](https://github.com/AsifJawad15/NLP-Based-Smart-FAQ-Retrieval) is currently empty.

1. **Scaffold**
   - Create structure, `.venv`, `.gitignore`, initial README, dependencies, and small test fixtures.
   - Commit: `chore: scaffold reusable FAQ retrieval project`
   - Push and verify `origin/main`.

2. **Small working baseline**
   - Implement loading, preprocessing, TF-IDF, cosine ranking, threshold handling, tests, and terminal corpus selection using 20–30 fixture FAQs.
   - Commit: `feat: implement TF-IDF FAQ retrieval baseline`
   - Push and verify.

3. **Full corpus preparation**
   - Add the preparation script, download both sources, filter and reshape candidates, manually review the final 500 records per corpus, and create validation/test data.
   - Commit: `data: add reviewed university and ecommerce corpora`
   - Push and verify.

4. **Evaluation**
   - Compare preprocessing configurations, tune thresholds on validation only, freeze configurations, run final tests once, and save reproducible reports.
   - Commit: `test: add cross-domain retrieval evaluation`
   - Push and verify.

5. **Teacher demonstration**
   - Add the notebook, final README, data-flow explanation, corpus-source documentation, actual results, limitations, and run commands.
   - Commit: `docs: add project notebook and evaluation report`
   - Push and verify.

After every push, compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/main` and require a clean working tree.

## Verification and Boundaries

- Install stable Python 3.12-compatible packages first; after successful installation and testing, freeze the verified environment into `requirements.txt`.
- Validate counts, schemas, sources, IDs, blank values, expected-ID references, and duplicate questions.
- Test exact matches, paraphrases, top-k ordering, empty input, OOV input, invalid corpora, threshold boundaries, false acceptance, and false rejection.
- Run unit tests, `compileall`, `pip check`, both evaluation commands, terminal demonstrations for both corpora, and headless notebook execution.
- After preparation, the system must run without internet access.
- Phase 2 Word2Vec and later Transformer/BERT work will reuse the same corpora and evaluation sets but require separate approval and Git pushes.
- Do not add RNN/LSTM, GUI, Flask, Streamlit, or generative-answer behavior during Phase 1.
