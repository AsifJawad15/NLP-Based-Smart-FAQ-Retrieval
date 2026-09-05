# Phase 2 Model Comparison (Synthetic Benchmark)

TF-IDF, mean Word2Vec, and TF-IDF-weighted Word2Vec were evaluated on the
identical query rows and labels. Each model applies its own threshold,
tuned on validation queries only. TF-IDF reuses its frozen Phase 1
configuration and was not retuned for this comparison.

Both Word2Vec models are trained on that domain's FAQ questions alone, use
basic preprocessing, and ignore word order. Their cosine similarities
occupy a different range than sparse TF-IDF cosines, which is why each
model needs its own threshold instead of a shared one.

## E-commerce FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted |
| --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic |
| Threshold | 0.58 | 0.92 | 0.91 |
| Answerable queries | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 |
| Top-1 accuracy | 0.953 | 0.753 | 0.740 |
| Top-3 accuracy | 0.980 | 0.860 | 0.880 |
| Correct answer rate | 0.820 | 0.633 | 0.687 |
| Accepted but wrong | 1 | 16 | 21 |
| Mean similarity of correct Top-1 | 0.7400 | 0.9491 | 0.9558 |
| Answerable acceptance rate | 0.827 | 0.740 | 0.827 |
| Unanswerable rejection rate | 0.960 | 0.940 | 0.920 |
| False acceptances | 2 | 3 | 4 |
| False rejections | 26 | 39 | 26 |

### Word2Vec mean: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

6 queries; the first 5 are shown.

- Query: Which page should someone visit to locate the code for the benefits claimed as part of the Money and Coins for Benefits?
  - Expected FAQ 420; Word2Vec mean returned it at similarity 0.9278.
  - TF-IDF predicted FAQ 420 at similarity 0.4661 (accepted=False).
- Query: Give me the details of the money back timelines if I call off or send back a item.
  - Expected FAQ 281; Word2Vec mean returned it at similarity 0.9209.
  - TF-IDF predicted FAQ 281 at similarity 0.3703 (accepted=False).
- Query: Explain how this works: how will I get my money back for returning an product I paid for with Money on Shipment.
  - Expected FAQ 181; Word2Vec mean returned it at similarity 0.9558.
  - TF-IDF predicted FAQ 181 at similarity 0.5348 (accepted=False).
- Query: I need guidance on this matter: i lost my guarantee card. How can I get guarantee.
  - Expected FAQ 98; Word2Vec mean returned it at similarity 0.9316.
  - TF-IDF predicted FAQ 98 at similarity 0.5736 (accepted=False).
- Query: Is it permitted to call off my EMI after I've placed the purchase with the Bajaj Finserv transaction choice?
  - Expected FAQ 104; Word2Vec mean returned it at similarity 0.9528.
  - TF-IDF predicted FAQ 104 at similarity 0.5624 (accepted=False).

### Word2Vec mean: reverse cases

Queries where TF-IDF delivers the correct answer and this model does not.

34 queries; the first 5 are shown.

- Query: Tell me if faster shipment choices like Same Day & In-a-Day on offer on send back requests.
  - Expected FAQ 128; TF-IDF returned it at similarity 0.8051.
  - Word2Vec mean predicted FAQ 128 at similarity 0.9078 (accepted=False).
- Query: I need a clear description of does 'Out of Stock' mean.
  - Expected FAQ 177; TF-IDF returned it at similarity 0.8694.
  - Word2Vec mean predicted FAQ 177 at similarity 0.8838 (accepted=False).
- Query: Explain the cause: do I see a 'shipment charge'.
  - Expected FAQ 182; TF-IDF returned it at similarity 0.6767.
  - Word2Vec mean predicted FAQ 188 at similarity 0.9457 (accepted=True).
- Query: What is the correct procedure to foreclose my EMI with Bajaj Finserv?
  - Expected FAQ 439; TF-IDF returned it at similarity 0.7691.
  - Word2Vec mean predicted FAQ 499 at similarity 0.9237 (accepted=True).
- Query: I am trying to make use of a new email location to log in to my Flipkart profile - what is the process?
  - Expected FAQ 75; TF-IDF returned it at similarity 0.6768.
  - Word2Vec mean predicted FAQ 256 at similarity 0.9221 (accepted=True).

### Word2Vec TF-IDF weighted: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

6 queries; the first 5 are shown.

- Query: Which page should someone visit to locate the code for the benefits claimed as part of the Money and Coins for Benefits?
  - Expected FAQ 420; Word2Vec TF-IDF weighted returned it at similarity 0.9128.
  - TF-IDF predicted FAQ 420 at similarity 0.4661 (accepted=False).
- Query: Explain how this works: how will I get my money back for returning an product I paid for with Money on Shipment.
  - Expected FAQ 181; Word2Vec TF-IDF weighted returned it at similarity 0.9319.
  - TF-IDF predicted FAQ 181 at similarity 0.5348 (accepted=False).
- Query: Would it be possible to choose PhonePe digital purse as a money back choice?
  - Expected FAQ 275; Word2Vec TF-IDF weighted returned it at similarity 0.9129.
  - TF-IDF predicted FAQ 275 at similarity 0.3949 (accepted=False).
- Query: Am I allowed to make use of an overseas identifier to sign up?
  - Expected FAQ 197; Word2Vec TF-IDF weighted returned it at similarity 0.9253.
  - TF-IDF predicted FAQ 197 at similarity 0.5298 (accepted=False).
- Query: Is it permitted to call off my EMI after I've placed the purchase with the Bajaj Finserv transaction choice?
  - Expected FAQ 104; Word2Vec TF-IDF weighted returned it at similarity 0.9583.
  - TF-IDF predicted FAQ 104 at similarity 0.5624 (accepted=False).

### Word2Vec TF-IDF weighted: reverse cases

Queries where TF-IDF delivers the correct answer and this model does not.

26 queries; the first 5 are shown.

- Query: Explain the cause: do I see a 'shipment charge'.
  - Expected FAQ 182; TF-IDF returned it at similarity 0.6767.
  - Word2Vec TF-IDF weighted predicted FAQ 188 at similarity 0.9289 (accepted=True).
- Query: I am trying to make use of a new email location to log in to my Flipkart profile - what is the process?
  - Expected FAQ 75; TF-IDF returned it at similarity 0.6768.
  - Word2Vec TF-IDF weighted predicted FAQ 256 at similarity 0.9321 (accepted=True).
- Query: Explain how this works: does a Present Card expire.
  - Expected FAQ 453; TF-IDF returned it at similarity 0.7158.
  - Word2Vec TF-IDF weighted predicted FAQ 412 at similarity 0.9025 (accepted=False).
- Query: I am trying to locate the offers in SuperCoin Zone - what is the process?
  - Expected FAQ 432; TF-IDF returned it at similarity 0.6352.
  - Word2Vec TF-IDF weighted predicted FAQ 201 at similarity 0.9006 (accepted=False).
- Query: Describe for me is my EMI debited.
  - Expected FAQ 409; TF-IDF returned it at similarity 0.7998.
  - Word2Vec TF-IDF weighted predicted FAQ 409 at similarity 0.8890 (accepted=False).

## University FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted |
| --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic |
| Threshold | 0.46 | 0.93 | 0.95 |
| Answerable queries | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 |
| Top-1 accuracy | 0.860 | 0.587 | 0.693 |
| Top-3 accuracy | 0.993 | 0.673 | 0.773 |
| Correct answer rate | 0.833 | 0.493 | 0.493 |
| Accepted but wrong | 17 | 17 | 9 |
| Mean similarity of correct Top-1 | 0.7602 | 0.9588 | 0.9643 |
| Answerable acceptance rate | 0.947 | 0.607 | 0.553 |
| Unanswerable rejection rate | 0.840 | 0.760 | 0.980 |
| False acceptances | 8 | 12 | 1 |
| False rejections | 8 | 59 | 67 |

### Word2Vec mean: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

1 query

- Query: Which employers take on people who finish Master of Education in Counselling Psychology (Global Mental Health & Counselling Psychology Field)?
  - Expected FAQ 10; Word2Vec mean returned it at similarity 0.9673.
  - TF-IDF predicted FAQ 15 at similarity 0.7100 (accepted=True).

### Word2Vec mean: reverse cases

Queries where TF-IDF delivers the correct answer and this model does not.

52 queries; the first 5 are shown.

- Query: Explain how this works: how will I get alerts.
  - Expected FAQ 288; TF-IDF returned it at similarity 0.6572.
  - Word2Vec mean predicted FAQ 288 at similarity 0.9215 (accepted=False).
- Query: Which areas of learning belong to Language and Literacies Education?
  - Expected FAQ 118; TF-IDF returned it at similarity 0.6094.
  - Word2Vec mean predicted FAQ 137 at similarity 0.8776 (accepted=False).
- Query: Describe the way Museum Studies is arranged.
  - Expected FAQ 91; TF-IDF returned it at similarity 0.5966.
  - Word2Vec mean predicted FAQ 289 at similarity 0.8535 (accepted=False).
- Query: I want to establish whether there any financial support choices for me.
  - Expected FAQ 323; TF-IDF returned it at similarity 0.7303.
  - Word2Vec mean predicted FAQ 316 at similarity 0.8878 (accepted=False).
- Query: What kind of workplace suits a Master of Education in Developmental Psychology & Education leaver?
  - Expected FAQ 77; TF-IDF returned it at similarity 0.7049.
  - Word2Vec mean predicted FAQ 15 at similarity 0.9536 (accepted=True).

### Word2Vec TF-IDF weighted: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

1 query

- Query: Which employers take on people who finish Master of Education in Counselling Psychology (Global Mental Health & Counselling Psychology Field)?
  - Expected FAQ 10; Word2Vec TF-IDF weighted returned it at similarity 0.9652.
  - TF-IDF predicted FAQ 15 at similarity 0.7100 (accepted=True).

### Word2Vec TF-IDF weighted: reverse cases

Queries where TF-IDF delivers the correct answer and this model does not.

52 queries; the first 5 are shown.

- Query: Explain how this works: how will I get alerts.
  - Expected FAQ 288; TF-IDF returned it at similarity 0.6572.
  - Word2Vec TF-IDF weighted predicted FAQ 288 at similarity 0.9105 (accepted=False).
- Query: Which areas of learning belong to Language and Literacies Education?
  - Expected FAQ 118; TF-IDF returned it at similarity 0.6094.
  - Word2Vec TF-IDF weighted predicted FAQ 118 at similarity 0.9039 (accepted=False).
- Query: Describe the way Museum Studies is arranged.
  - Expected FAQ 91; TF-IDF returned it at similarity 0.5966.
  - Word2Vec TF-IDF weighted predicted FAQ 91 at similarity 0.8936 (accepted=False).
- Query: I want to establish whether there any financial support choices for me.
  - Expected FAQ 323; TF-IDF returned it at similarity 0.7303.
  - Word2Vec TF-IDF weighted predicted FAQ 323 at similarity 0.9222 (accepted=False).
- Query: What kind of workplace suits a Master of Education in Developmental Psychology & Education leaver?
  - Expected FAQ 77; TF-IDF returned it at similarity 0.7049.
  - Word2Vec TF-IDF weighted predicted FAQ 15 at similarity 0.9653 (accepted=True).

