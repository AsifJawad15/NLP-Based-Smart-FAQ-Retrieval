# KUET TF-IDF Development Smoke Check

Thresholds and preprocessing were selected on validation queries only,
frozen into `corpus_config.json`, and applied here without retuning.

These developer-authored smoke queries are separate from threshold tuning. Their results are development checks, not an untouched human evaluation.

Correct answer rate = count(correct Top-1 AND accepted) / all answerable queries. Accepted but wrong counts answerable queries answered with a different FAQ.

| Metric | KUET University |
| --- | --- |
| FAQs indexed | 110 |
| Preprocessing | stopwords_removed |
| Threshold | 0.47 |
| Answerable queries | 15 |
| Unanswerable queries | 5 |
| Top-1 accuracy | 1.000 |
| Top-3 accuracy | 1.000 |
| Correct answer rate | 1.000 |
| Accepted but wrong | 0 |
| Mean similarity of correct Top-1 | 0.7812 |
| Answerable acceptance rate | 1.000 |
| Unanswerable rejection rate | 1.000 |
| False acceptances | 0 |
| False rejections | 0 |

## Incorrect retrievals: KUET University

No incorrect Top-1 retrievals were recorded.
