# KUET TF-IDF Development Smoke Check

Thresholds and preprocessing were selected on validation queries only,
frozen into `corpus_config.json`, and applied here without retuning.

These developer-authored smoke queries are separate from threshold tuning. Their results are development checks, not an untouched human evaluation.

Correct answer rate = count(correct Top-1 AND accepted) / all answerable queries. Accepted but wrong counts answerable queries answered with a different FAQ.

| Metric | KUET University |
| --- | --- |
| FAQs indexed | 200 |
| Preprocessing | stopwords_removed |
| Threshold | 0.47 |
| Answerable queries | 45 |
| Unanswerable queries | 15 |
| Top-1 accuracy | 0.933 |
| Top-3 accuracy | 0.978 |
| Correct answer rate | 0.911 |
| Accepted but wrong | 2 |
| Mean similarity of correct Top-1 | 0.7826 |
| Answerable acceptance rate | 0.956 |
| Unanswerable rejection rate | 0.533 |
| False acceptances | 7 |
| False rejections | 2 |

## Incorrect retrievals: KUET University

- Query: Where is the official CSE faculty directory?
  - Expected FAQ 156, retrieved FAQ 182 (similarity 0.6653, accepted=True)
  - Retrieved question: Which faculty includes KUET CSE?
- Query: Can CSE undergraduate students do research?
  - Expected FAQ 161, retrieved FAQ 100 (similarity 0.3423, accepted=False)
  - Retrieved question: Which KUET hall is for female students?
- Query: Which faculty does CSE belong to?
  - Expected FAQ 182, retrieved FAQ 183 (similarity 0.6278, accepted=True)
  - Retrieved question: Which departments belong to KUET EEE faculty?
