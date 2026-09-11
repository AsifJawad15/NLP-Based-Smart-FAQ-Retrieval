# Manual TF-IDF Evaluation

Thresholds and preprocessing were selected on validation queries only,
frozen into `corpus_config.json`, and applied here without retuning.

These queries are supplied separately by the project team. The evaluation command cannot verify human authorship.

Correct answer rate = count(correct Top-1 AND accepted) / all answerable queries. Accepted but wrong counts answerable queries answered with a different FAQ.

| Metric | University FAQ |
| --- | --- |
| FAQs indexed | 500 |
| Preprocessing | basic |
| Threshold | 0.46 |
| Answerable queries | 1 |
| Unanswerable queries | 1 |
| Top-1 accuracy | 0.000 |
| Top-3 accuracy | 0.000 |
| Correct answer rate | 0.000 |
| Accepted but wrong | 1 |
| Mean similarity of correct Top-1 | 0.0000 |
| Answerable acceptance rate | 1.000 |
| Unanswerable rejection rate | 1.000 |
| False acceptances | 0 |
| False rejections | 0 |

## Incorrect retrievals: University FAQ

- Query: explain the syllabus structure
  - Expected FAQ 8, retrieved FAQ 65 (similarity 0.5786, accepted=True)
  - Retrieved question: can you explain the structure of English at UOFT?
