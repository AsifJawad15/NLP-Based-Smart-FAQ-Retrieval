# Smart FAQ: KUET Demo and Local GUI

## Delivered scope

This release adds an English KUET corpus and a localhost Streamlit interface.
The FAQ Assistant uses TF-IDF on KUET. A separate Model Comparison page runs
the seven existing models on the university and e-commerce research corpora.
General e-commerce demo data and formal human evaluation remain deferred until
after the first live KUET evaluation.

## KUET data

`data/kuet/faq_dataset.csv` contains 110 distinct FAQs across university
information, programs, admissions, library and academic services, halls,
medical services, scholarships, transport, and campus facilities. Every answer
uses `official_web` and an exact official KUET URL. `SOURCE_REVIEW.md` records
the 11 September 2026 verification date and supporting material for every ID.
Dated admission information names its reviewed session and sends users to the
current official notice.

The developer-authored sets are separate:

- 50 validation queries: 30 answerable and 20 unanswerable;
- 20 smoke queries: 15 answerable and 5 unanswerable.

The two-stage selection froze stopword removal and a 0.47 TF-IDF threshold.
Results under `reports/demo/` are development checks, not human evaluation.

## Retrieval and reports

Corpus configurations now have backward-compatible `purpose` and
`supported_models` fields. Older corpora default to research and all seven
models. Batch research evaluation and training commands exclude demo corpora;
KUET must be explicit and rejects non-TF-IDF model requests before loading an
artifact. Demo tuning and evaluation never overwrite research reports.

## GUI

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

The FAQ Assistant shows an accepted FAQ, stored answer, and source. Rejection
shows “No sufficiently relevant FAQ found” and no answer. NLP Details contains
the processed query, model, ID, score, threshold, and top-three candidates with
unaccepted labels where applicable.

Model Comparison loads all seven models only after submission. Each working
model reports its predicted FAQ, similarity, own threshold, and decision. RNN
and BiLSTM are labeled experimental; one artifact error does not stop the other
models. Saved benchmark metrics appear separately, with a warning that raw
similarities from different vector spaces are not directly comparable.

Cached answerers use corpus, model, and file-change signatures. The GUI runs
inference only and clears displayed results when corpus selection changes.

## Verification and handoff

The implementation validates corpus schema, review coverage, split counts,
unsupported models, report isolation, cache invalidation, acceptance/rejection,
CLI/GUI agreement, research-report reproduction, and the live browser layout.
`docs/DEMO_REHEARSAL.md` contains the easy match, natural paraphrase, rejection,
and seven-model comparison cases for presentation.

Later evaluator questions are feedback-driven development iterations. They are
not retroactively treated as an untouched human benchmark.
