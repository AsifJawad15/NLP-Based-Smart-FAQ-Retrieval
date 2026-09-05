# Cross-Domain TF-IDF Synthetic Benchmark

Thresholds and preprocessing were selected on validation queries only,
frozen into `corpus_config.json`, and applied here without retuning.

The synthetic query sets have already been inspected during development and review; these are reproducible benchmark results, not an untouched final assessment.

Correct answer rate = count(correct Top-1 AND accepted) / all answerable queries. Accepted but wrong counts answerable queries answered with a different FAQ.

| Metric | E-commerce FAQ | University FAQ |
| --- | --- | --- |
| FAQs indexed | 500 | 500 |
| Preprocessing | basic | basic |
| Threshold | 0.58 | 0.46 |
| Answerable queries | 150 | 150 |
| Unanswerable queries | 50 | 50 |
| Top-1 accuracy | 0.953 | 0.860 |
| Top-3 accuracy | 0.980 | 0.993 |
| Correct answer rate | 0.820 | 0.833 |
| Accepted but wrong | 1 | 17 |
| Mean similarity of correct Top-1 | 0.7400 | 0.7602 |
| Answerable acceptance rate | 0.827 | 0.947 |
| Unanswerable rejection rate | 0.960 | 0.840 |
| False acceptances | 2 | 8 |
| False rejections | 26 | 8 |

## Incorrect retrievals: E-commerce FAQ

- Query: What should be understood about the guarantee being provided by Ather?
  - Expected FAQ 330, retrieved FAQ 341 (similarity 0.4680, accepted=False)
  - Retrieved question: What after sales services are being provided by Ather for 2-wheelers?
- Query: What should be understood about Money on Shipment?
  - Expected FAQ 294, retrieved FAQ 324 (similarity 0.3618, accepted=False)
  - Retrieved question: I have a query about the EMI charge. What should I do?
- Query: What steps should someone follow to settle the amount for my purchase?
  - Expected FAQ 302, retrieved FAQ 205 (similarity 0.3863, accepted=False)
  - Retrieved question: Is there a minimum purchase amount for Flipkart Quick orders?
- Query: I need to establish what will happen if the make approved retailer who will reach me is very far from my location.
  - Expected FAQ 452, retrieved FAQ 449 (similarity 0.6811, accepted=True)
  - Retrieved question: What will happen if the Bounce authorised dealer who will contact me is very far from my location?
- Query: What steps should someone follow to order with a Present card?
  - Expected FAQ 350, retrieved FAQ 327 (similarity 0.3721, accepted=False)
  - Retrieved question: What if the Gift Card is transferred to someone when I am expecting a refund for an order placed using the same Gift Card?

## Incorrect retrievals: University FAQ

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47, retrieved FAQ 97 (similarity 0.7771, accepted=True)
  - Retrieved question: what subjects will I study in Germanic Literature, Culture and Theory at UOFT?
- Query: What investigations are carried out in Master of Applied Science (MASc)?
  - Expected FAQ 221, retrieved FAQ 101 (similarity 0.7089, accepted=True)
  - Retrieved question: what subjects will I study in Master of Applied Science (MASc) at UOFT?
- Query: Which study areas does Finance pursue?
  - Expected FAQ 238, retrieved FAQ 71 (similarity 0.3602, accepted=False)
  - Retrieved question: can you explain the structure of Finance at UOFT?
- Query: Which areas of learning belong to Ecology and Evolutionary Biology?
  - Expected FAQ 64, retrieved FAQ 48 (similarity 0.6661, accepted=True)
  - Retrieved question: can you explain the structure of Ecology and Evolutionary Biology at UOFT?
- Query: What shape does Counselling & Clinical Psychology take?
  - Expected FAQ 83, retrieved FAQ 39 (similarity 0.5722, accepted=True)
  - Retrieved question: what makes the Counselling and Clinical Psychology - Field in Clinical and Counselling Psychology (OISE) program at UOFT unique?
