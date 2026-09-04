# Cross-Domain TF-IDF Evaluation

Thresholds and preprocessing were selected on validation queries only,
frozen into `corpus_config.json`, and then applied once to the test set.

| Metric | E-commerce FAQ | University FAQ |
| --- | --- | --- |
| FAQs indexed | 500 | 500 |
| Preprocessing | basic | lemmatized |
| Threshold | 0.58 | 0.46 |
| Top-1 accuracy | 0.953 | 0.847 |
| Top-3 accuracy | 0.980 | 0.987 |
| Mean similarity of correct Top-1 | 0.7400 | 0.7563 |
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
  - Expected FAQ 47, retrieved FAQ 97 (similarity 0.7701, accepted=True)
  - Retrieved question: what subjects will I study in Germanic Literature, Culture and Theory at UOFT?
- Query: What investigations are carried out in Master of Applied Science (MASc)?
  - Expected FAQ 221, retrieved FAQ 101 (similarity 0.7098, accepted=True)
  - Retrieved question: what subjects will I study in Master of Applied Science (MASc) at UOFT?
- Query: Which study areas does Finance pursue?
  - Expected FAQ 238, retrieved FAQ 71 (similarity 0.3623, accepted=False)
  - Retrieved question: can you explain the structure of Finance at UOFT?
- Query: Which areas of learning belong to Ecology and Evolutionary Biology?
  - Expected FAQ 64, retrieved FAQ 48 (similarity 0.6661, accepted=True)
  - Retrieved question: can you explain the structure of Ecology and Evolutionary Biology at UOFT?
- Query: What shape does Counselling & Clinical Psychology take?
  - Expected FAQ 83, retrieved FAQ 39 (similarity 0.5762, accepted=True)
  - Retrieved question: what makes the Counselling and Clinical Psychology - Field in Clinical and Counselling Psychology (OISE) program at UOFT unique?
