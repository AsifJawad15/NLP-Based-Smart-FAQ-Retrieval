# Phase 3 Model Comparison (Synthetic Benchmark)

All seven models were evaluated on the identical query rows and labels.
Each applies its own threshold, tuned on validation queries only. TF-IDF,
both custom Word2Vec models and both pretrained Word2Vec models reuse their
frozen settings from earlier phases.

The Siamese RNN and BiLSTM read the same frozen Google News vectors as the
pretrained models, but combine them in word order instead of averaging
them. Both are trained on paraphrases generated independently of the
evaluation queries, with the checkpoint chosen on a held-out dev split of
those paraphrases; validation queries only set thresholds.

Case A: TF-IDF and Pretrained Word2Vec mean both fail, and the sequence
model answers. Case B: TF-IDF answers and the sequence model does not.
Case C: Pretrained Word2Vec mean answers and the sequence model does not.

The synthetic test queries were generated from templates and inspected
during development. Human evaluation is still pending, so these results
describe synthetic paraphrases only.

## E-commerce FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted | Pretrained Word2Vec mean | Pretrained Word2Vec TF-IDF weighted | Siamese RNN | Siamese BiLSTM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic | basic | basic | basic | basic |
| Threshold | 0.58 | 0.92 | 0.91 | 0.81 | 0.76 | 0.69 | 0.70 |
| Answerable queries | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Top-1 accuracy | 0.953 | 0.753 | 0.740 | 0.827 | 0.867 | 0.607 | 0.507 |
| Top-3 accuracy | 0.980 | 0.860 | 0.880 | 0.913 | 0.927 | 0.707 | 0.640 |
| Correct answer rate | 0.820 | 0.633 | 0.687 | 0.787 | 0.840 | 0.533 | 0.347 |
| Accepted but wrong | 1 | 16 | 21 | 18 | 17 | 23 | 7 |
| Mean similarity of correct Top-1 | 0.7400 | 0.9491 | 0.9558 | 0.9024 | 0.9028 | 0.8428 | 0.7675 |
| Answerable acceptance rate | 0.827 | 0.740 | 0.827 | 0.907 | 0.953 | 0.687 | 0.393 |
| Unanswerable rejection rate | 0.960 | 0.940 | 0.920 | 0.760 | 0.520 | 0.740 | 0.960 |
| False acceptances | 2 | 3 | 4 | 12 | 24 | 13 | 2 |
| False rejections | 26 | 39 | 26 | 14 | 7 | 47 | 91 |

### Siamese RNN: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

5 queries

- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Siamese RNN returned it at similarity 0.8377.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
- Query: Explain how this works: how will I get my money back for returning an product I paid for with Money on Shipment.
  - Expected FAQ 181; Siamese RNN returned it at similarity 0.8060.
  - TF-IDF predicted FAQ 181 at similarity 0.5348 (accepted=False).
- Query: What should be understood about the guarantee being provided by Ather?
  - Expected FAQ 330; Siamese RNN returned it at similarity 0.7235.
  - TF-IDF predicted FAQ 341 at similarity 0.4680 (accepted=False).
- Query: Am I allowed to make use of an overseas identifier to sign up?
  - Expected FAQ 197; Siamese RNN returned it at similarity 0.7302.
  - TF-IDF predicted FAQ 197 at similarity 0.5298 (accepted=False).
- Query: Am I expected to have to send back the freebie when I send back a item?
  - Expected FAQ 230; Siamese RNN returned it at similarity 0.7679.
  - TF-IDF predicted FAQ 230 at similarity 0.5669 (accepted=False).

### Siamese RNN: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

48 queries; the first 5 are shown.

- Query: Is it permitted to choose a preferred time slot for the shipment of Flipkart Quick purchases?
  - Expected FAQ 189; TF-IDF returned it at similarity 0.7790.
  - Siamese RNN predicted FAQ 189 at similarity 0.6491 (accepted=False).
- Query: I need a clear description of does 'Out of Stock' mean.
  - Expected FAQ 177; TF-IDF returned it at similarity 0.8694.
  - Siamese RNN predicted FAQ 177 at similarity 0.5872 (accepted=False).
- Query: Explain how this works: i missed the shipment of my purchase today. What should I do.
  - Expected FAQ 57; TF-IDF returned it at similarity 0.7203.
  - Siamese RNN predicted FAQ 4 at similarity 0.8906 (accepted=True).
- Query: I am trying to make use of a new email location to log in to my Flipkart profile - what is the process?
  - Expected FAQ 75; TF-IDF returned it at similarity 0.6768.
  - Siamese RNN predicted FAQ 305 at similarity 0.7247 (accepted=True).
- Query: Explain how this works: does a Present Card expire.
  - Expected FAQ 453; TF-IDF returned it at similarity 0.7158.
  - Siamese RNN predicted FAQ 453 at similarity 0.6666 (accepted=False).

### Siamese RNN: case A

Queries where TF-IDF and Pretrained Word2Vec mean both fail and this model delivers the correct answer.

1 query

- Query: What should be understood about the guarantee being provided by Ather?
  - Expected FAQ 330; Siamese RNN returned it at similarity 0.7235.
  - TF-IDF predicted FAQ 341 at similarity 0.4680 (accepted=False).
  - Pretrained Word2Vec mean predicted FAQ 23 at similarity 0.7915 (accepted=False).

### Siamese RNN: case C

Queries where Pretrained Word2Vec mean delivers the correct answer and this model does not.

43 queries; the first 5 are shown.

- Query: Is it permitted to choose a preferred time slot for the shipment of Flipkart Quick purchases?
  - Expected FAQ 189; Pretrained Word2Vec mean returned it at similarity 0.9077.
  - Siamese RNN predicted FAQ 189 at similarity 0.6491 (accepted=False).
- Query: Which page should someone visit to locate the code for the benefits claimed as part of the Money and Coins for Benefits?
  - Expected FAQ 420; Pretrained Word2Vec mean returned it at similarity 0.8857.
  - Siamese RNN predicted FAQ 420 at similarity 0.6794 (accepted=False).
- Query: I need a clear description of does 'Out of Stock' mean.
  - Expected FAQ 177; Pretrained Word2Vec mean returned it at similarity 0.8807.
  - Siamese RNN predicted FAQ 177 at similarity 0.5872 (accepted=False).
- Query: Explain how this works: i missed the shipment of my purchase today. What should I do.
  - Expected FAQ 57; Pretrained Word2Vec mean returned it at similarity 0.9225.
  - Siamese RNN predicted FAQ 4 at similarity 0.8906 (accepted=True).
- Query: I am trying to make use of a new email location to log in to my Flipkart profile - what is the process?
  - Expected FAQ 75; Pretrained Word2Vec mean returned it at similarity 0.9153.
  - Siamese RNN predicted FAQ 305 at similarity 0.7247 (accepted=True).

### Siamese BiLSTM: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

2 queries

- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Siamese BiLSTM returned it at similarity 0.7135.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
- Query: Am I allowed to make use of an overseas identifier to sign up?
  - Expected FAQ 197; Siamese BiLSTM returned it at similarity 0.7813.
  - TF-IDF predicted FAQ 197 at similarity 0.5298 (accepted=False).

### Siamese BiLSTM: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

73 queries; the first 5 are shown.

- Query: Explain how this works: if I request for a exchange, when will I get it.
  - Expected FAQ 32; TF-IDF returned it at similarity 0.6888.
  - Siamese BiLSTM predicted FAQ 274 at similarity 0.5166 (accepted=False).
- Query: I need a clear description of should I do if I don't get the OTP or verification code.
  - Expected FAQ 107; TF-IDF returned it at similarity 0.9115.
  - Siamese BiLSTM predicted FAQ 372 at similarity 0.6422 (accepted=False).
- Query: Tell me if faster shipment choices like Same Day & In-a-Day on offer on send back requests.
  - Expected FAQ 128; TF-IDF returned it at similarity 0.8051.
  - Siamese BiLSTM predicted FAQ 128 at similarity 0.5919 (accepted=False).
- Query: I need guidance on this matter: will guarantee be applicable for my item since I bought it online.
  - Expected FAQ 50; TF-IDF returned it at similarity 0.7725.
  - Siamese BiLSTM predicted FAQ 207 at similarity 0.5885 (accepted=False).
- Query: Is it permitted to choose a preferred time slot for the shipment of Flipkart Quick purchases?
  - Expected FAQ 189; TF-IDF returned it at similarity 0.7790.
  - Siamese BiLSTM predicted FAQ 91 at similarity 0.5610 (accepted=False).

### Siamese BiLSTM: case A

Queries where TF-IDF and Pretrained Word2Vec mean both fail and this model delivers the correct answer.

No queries fall in this category.

### Siamese BiLSTM: case C

Queries where Pretrained Word2Vec mean delivers the correct answer and this model does not.

68 queries; the first 5 are shown.

- Query: Explain how this works: if I request for a exchange, when will I get it.
  - Expected FAQ 32; Pretrained Word2Vec mean returned it at similarity 0.9195.
  - Siamese BiLSTM predicted FAQ 274 at similarity 0.5166 (accepted=False).
- Query: I need a clear description of should I do if I don't get the OTP or verification code.
  - Expected FAQ 107; Pretrained Word2Vec mean returned it at similarity 0.9643.
  - Siamese BiLSTM predicted FAQ 372 at similarity 0.6422 (accepted=False).
- Query: Tell me if faster shipment choices like Same Day & In-a-Day on offer on send back requests.
  - Expected FAQ 128; Pretrained Word2Vec mean returned it at similarity 0.8591.
  - Siamese BiLSTM predicted FAQ 128 at similarity 0.5919 (accepted=False).
- Query: I need guidance on this matter: will guarantee be applicable for my item since I bought it online.
  - Expected FAQ 50; Pretrained Word2Vec mean returned it at similarity 0.8723.
  - Siamese BiLSTM predicted FAQ 207 at similarity 0.5885 (accepted=False).
- Query: Is it permitted to choose a preferred time slot for the shipment of Flipkart Quick purchases?
  - Expected FAQ 189; Pretrained Word2Vec mean returned it at similarity 0.9077.
  - Siamese BiLSTM predicted FAQ 91 at similarity 0.5610 (accepted=False).

## University FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted | Pretrained Word2Vec mean | Pretrained Word2Vec TF-IDF weighted | Siamese RNN | Siamese BiLSTM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic | basic | basic | basic | basic |
| Threshold | 0.46 | 0.93 | 0.95 | 0.86 | 0.80 | 0.80 | 0.75 |
| Answerable queries | 150 | 150 | 150 | 150 | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Top-1 accuracy | 0.860 | 0.587 | 0.693 | 0.873 | 0.867 | 0.553 | 0.633 |
| Top-3 accuracy | 0.993 | 0.673 | 0.773 | 0.953 | 0.987 | 0.593 | 0.720 |
| Correct answer rate | 0.833 | 0.493 | 0.493 | 0.627 | 0.813 | 0.420 | 0.347 |
| Accepted but wrong | 17 | 17 | 9 | 3 | 15 | 2 | 1 |
| Mean similarity of correct Top-1 | 0.7602 | 0.9588 | 0.9643 | 0.8929 | 0.9140 | 0.8647 | 0.7707 |
| Answerable acceptance rate | 0.947 | 0.607 | 0.553 | 0.647 | 0.913 | 0.433 | 0.353 |
| Unanswerable rejection rate | 0.840 | 0.760 | 0.980 | 0.960 | 0.880 | 1.000 | 0.960 |
| False acceptances | 8 | 12 | 1 | 2 | 6 | 0 | 2 |
| False rejections | 8 | 59 | 67 | 53 | 13 | 85 | 97 |

### Siamese RNN: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

No queries fall in this category.

### Siamese RNN: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

62 queries; the first 5 are shown.

- Query: Explain how this works: how will I get alerts.
  - Expected FAQ 288; TF-IDF returned it at similarity 0.6572.
  - Siamese RNN predicted FAQ 288 at similarity 0.6551 (accepted=False).
- Query: I need guidance on this matter: will credits earned in secondary school or at a distinct post-secondary institution be qualified for entry conditions of the Computer Science degree.
  - Expected FAQ 401; TF-IDF returned it at similarity 0.8044.
  - Siamese RNN predicted FAQ 401 at similarity 0.6828 (accepted=False).
- Query: Tell me the moment at which is the move-in date? Do I require to put in an application for an early move-in.
  - Expected FAQ 326; TF-IDF returned it at similarity 0.7477.
  - Siamese RNN predicted FAQ 33 at similarity 0.7217 (accepted=False).
- Query: I need guidance on this matter: can transfer students participate in the Jump Start degree.
  - Expected FAQ 322; TF-IDF returned it at similarity 0.7833.
  - Siamese RNN predicted FAQ 406 at similarity 0.6219 (accepted=False).
- Query: Which areas of learning belong to Language and Literacies Education?
  - Expected FAQ 118; TF-IDF returned it at similarity 0.6094.
  - Siamese RNN predicted FAQ 170 at similarity 0.7470 (accepted=False).

### Siamese RNN: case A

Queries where TF-IDF and Pretrained Word2Vec mean both fail and this model delivers the correct answer.

No queries fall in this category.

### Siamese RNN: case C

Queries where Pretrained Word2Vec mean delivers the correct answer and this model does not.

35 queries; the first 5 are shown.

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec mean returned it at similarity 0.8720.
  - Siamese RNN predicted FAQ 121 at similarity 0.7554 (accepted=False).
- Query: I need guidance on this matter: will credits earned in secondary school or at a distinct post-secondary institution be qualified for entry conditions of the Computer Science degree.
  - Expected FAQ 401; Pretrained Word2Vec mean returned it at similarity 0.9227.
  - Siamese RNN predicted FAQ 401 at similarity 0.6828 (accepted=False).
- Query: Tell me the moment at which is the move-in date? Do I require to put in an application for an early move-in.
  - Expected FAQ 326; Pretrained Word2Vec mean returned it at similarity 0.9542.
  - Siamese RNN predicted FAQ 33 at similarity 0.7217 (accepted=False).
- Query: I need guidance on this matter: i would like to put in an application to a former student degree at the University of Toronto. Where should I get the submission form.
  - Expected FAQ 410; Pretrained Word2Vec mean returned it at similarity 0.9413.
  - Siamese RNN predicted FAQ 445 at similarity 0.6424 (accepted=False).
- Query: I am trying to submit my final, official academic record (or other required documents) - what is the process?
  - Expected FAQ 385; Pretrained Word2Vec mean returned it at similarity 0.8855.
  - Siamese RNN predicted FAQ 25 at similarity 0.7154 (accepted=False).

### Siamese BiLSTM: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

No queries fall in this category.

### Siamese BiLSTM: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

73 queries; the first 5 are shown.

- Query: Explain how this works: how will I get alerts.
  - Expected FAQ 288; TF-IDF returned it at similarity 0.6572.
  - Siamese BiLSTM predicted FAQ 231 at similarity 0.5864 (accepted=False).
- Query: I need guidance on this matter: will credits earned in secondary school or at a distinct post-secondary institution be qualified for entry conditions of the Computer Science degree.
  - Expected FAQ 401; TF-IDF returned it at similarity 0.8044.
  - Siamese BiLSTM predicted FAQ 401 at similarity 0.5552 (accepted=False).
- Query: Tell me the moment at which is the move-in date? Do I require to put in an application for an early move-in.
  - Expected FAQ 326; TF-IDF returned it at similarity 0.7477.
  - Siamese BiLSTM predicted FAQ 326 at similarity 0.7399 (accepted=False).
- Query: I need guidance on this matter: can transfer students participate in the Jump Start degree.
  - Expected FAQ 322; TF-IDF returned it at similarity 0.7833.
  - Siamese BiLSTM predicted FAQ 92 at similarity 0.4589 (accepted=False).
- Query: Which areas of learning belong to Language and Literacies Education?
  - Expected FAQ 118; TF-IDF returned it at similarity 0.6094.
  - Siamese BiLSTM predicted FAQ 280 at similarity 0.6759 (accepted=False).

### Siamese BiLSTM: case A

Queries where TF-IDF and Pretrained Word2Vec mean both fail and this model delivers the correct answer.

No queries fall in this category.

### Siamese BiLSTM: case C

Queries where Pretrained Word2Vec mean delivers the correct answer and this model does not.

43 queries; the first 5 are shown.

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec mean returned it at similarity 0.8720.
  - Siamese BiLSTM predicted FAQ 161 at similarity 0.5966 (accepted=False).
- Query: I need guidance on this matter: will credits earned in secondary school or at a distinct post-secondary institution be qualified for entry conditions of the Computer Science degree.
  - Expected FAQ 401; Pretrained Word2Vec mean returned it at similarity 0.9227.
  - Siamese BiLSTM predicted FAQ 401 at similarity 0.5552 (accepted=False).
- Query: Tell me the moment at which is the move-in date? Do I require to put in an application for an early move-in.
  - Expected FAQ 326; Pretrained Word2Vec mean returned it at similarity 0.9542.
  - Siamese BiLSTM predicted FAQ 326 at similarity 0.7399 (accepted=False).
- Query: Explain how this works: i am qualified for both the Summer and the Fall-Winter degree streams. Which one should I choose.
  - Expected FAQ 363; Pretrained Word2Vec mean returned it at similarity 0.9396.
  - Siamese BiLSTM predicted FAQ 363 at similarity 0.7275 (accepted=False).
- Query: I need guidance on this matter: i would like to put in an application to a former student degree at the University of Toronto. Where should I get the submission form.
  - Expected FAQ 410; Pretrained Word2Vec mean returned it at similarity 0.9413.
  - Siamese BiLSTM predicted FAQ 255 at similarity 0.6631 (accepted=False).

