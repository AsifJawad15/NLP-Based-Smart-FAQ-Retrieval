# Phase 2B Model Comparison (Synthetic Benchmark)

TF-IDF, both custom Word2Vec models, and both pretrained Word2Vec models
were evaluated on the identical query rows and labels. Each model applies
its own threshold, tuned on validation queries only. TF-IDF and the custom
models reuse their frozen Phase 1 and Phase 2 settings.

The pretrained models look up the same case-folded Google News vectors
(word2vec-google-news-300) and combine them with exactly the code the
custom models use, so only the source of the word vectors changes. Unknown
words are skipped, a query with no known word is rejected, and word order
is still ignored.

Case A: TF-IDF and the custom model with the same aggregation both fail, and
the pretrained model answers. Case B: TF-IDF answers and the pretrained model
does not. Case C: the custom model answers and the pretrained model does not.

The synthetic queries were generated from templates and inspected during
development; human evaluation is still pending.

## E-commerce FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted | Pretrained Word2Vec mean | Pretrained Word2Vec TF-IDF weighted |
| --- | --- | --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic | basic | basic |
| Threshold | 0.58 | 0.92 | 0.91 | 0.81 | 0.76 |
| Answerable queries | 150 | 150 | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 | 50 | 50 |
| Top-1 accuracy | 0.953 | 0.753 | 0.740 | 0.827 | 0.867 |
| Top-3 accuracy | 0.980 | 0.860 | 0.880 | 0.913 | 0.927 |
| Correct answer rate | 0.820 | 0.633 | 0.687 | 0.787 | 0.840 |
| Accepted but wrong | 1 | 16 | 21 | 18 | 17 |
| Mean similarity of correct Top-1 | 0.7400 | 0.9491 | 0.9558 | 0.9024 | 0.9028 |
| Answerable acceptance rate | 0.827 | 0.740 | 0.827 | 0.907 | 0.953 |
| Unanswerable rejection rate | 0.960 | 0.940 | 0.920 | 0.760 | 0.520 |
| False acceptances | 2 | 3 | 4 | 12 | 24 |
| False rejections | 26 | 39 | 26 | 14 | 7 |

### Pretrained Word2Vec mean: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

13 queries; the first 5 are shown.

- Query: Which page should someone visit to locate the code for the benefits claimed as part of the Money and Coins for Benefits?
  - Expected FAQ 420; Pretrained Word2Vec mean returned it at similarity 0.8857.
  - TF-IDF predicted FAQ 420 at similarity 0.4661 (accepted=False).
- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Pretrained Word2Vec mean returned it at similarity 0.9128.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
- Query: Am I allowed to redeem all of my on offer SuperCoins while with the SuperCoin Settle the amount function?
  - Expected FAQ 111; Pretrained Word2Vec mean returned it at similarity 0.8652.
  - TF-IDF predicted FAQ 111 at similarity 0.5713 (accepted=False).
- Query: I need guidance on this matter: how long does it take to call off an purchase.
  - Expected FAQ 112; Pretrained Word2Vec mean returned it at similarity 0.8660.
  - TF-IDF predicted FAQ 112 at similarity 0.5197 (accepted=False).
- Query: Explain how this works: how will I get my money back for returning an product I paid for with Money on Shipment.
  - Expected FAQ 181; Pretrained Word2Vec mean returned it at similarity 0.9150.
  - TF-IDF predicted FAQ 181 at similarity 0.5348 (accepted=False).

### Pretrained Word2Vec mean: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

18 queries; the first 5 are shown.

- Query: Explain how this works: does a Present Card expire.
  - Expected FAQ 453; TF-IDF returned it at similarity 0.7158.
  - Pretrained Word2Vec mean predicted FAQ 453 at similarity 0.7834 (accepted=False).
- Query: Is it permitted to order another Present Card with the Present Card I have?
  - Expected FAQ 378; TF-IDF returned it at similarity 0.6486.
  - Pretrained Word2Vec mean predicted FAQ 451 at similarity 0.8506 (accepted=True).
- Query: Explain Flipkart Plus degree.
  - Expected FAQ 183; TF-IDF returned it at similarity 0.6387.
  - Pretrained Word2Vec mean predicted FAQ 183 at similarity 0.6498 (accepted=False).
- Query: Please clarify the following point: i want to verify my guarantee particulars for the item but I don't have the guarantee card. What should I do.
  - Expected FAQ 146; TF-IDF returned it at similarity 0.6548.
  - Pretrained Word2Vec mean predicted FAQ 2 at similarity 0.9089 (accepted=True).
- Query: I need guidance on this matter: how is the Send back Fee determined.
  - Expected FAQ 317; TF-IDF returned it at similarity 0.6988.
  - Pretrained Word2Vec mean predicted FAQ 188 at similarity 0.8231 (accepted=True).

### Pretrained Word2Vec mean: case A

Queries where TF-IDF and Word2Vec mean both fail and this model delivers the correct answer.

9 queries; the first 5 are shown.

- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Pretrained Word2Vec mean returned it at similarity 0.9128.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
  - Word2Vec mean predicted FAQ 123 at similarity 0.8995 (accepted=False).
- Query: Am I allowed to redeem all of my on offer SuperCoins while with the SuperCoin Settle the amount function?
  - Expected FAQ 111; Pretrained Word2Vec mean returned it at similarity 0.8652.
  - TF-IDF predicted FAQ 111 at similarity 0.5713 (accepted=False).
  - Word2Vec mean predicted FAQ 166 at similarity 0.9256 (accepted=True).
- Query: I need guidance on this matter: how long does it take to call off an purchase.
  - Expected FAQ 112; Pretrained Word2Vec mean returned it at similarity 0.8660.
  - TF-IDF predicted FAQ 112 at similarity 0.5197 (accepted=False).
  - Word2Vec mean predicted FAQ 112 at similarity 0.9189 (accepted=False).
- Query: Am I allowed to redeem SuperCoins to avail price reduction for Flipkart Present Card?
  - Expected FAQ 384; Pretrained Word2Vec mean returned it at similarity 0.8862.
  - TF-IDF predicted FAQ 384 at similarity 0.5664 (accepted=False).
  - Word2Vec mean predicted FAQ 187 at similarity 0.9331 (accepted=True).
- Query: Give me the details of the amount I can shop for with Money on Shipment transaction choice.
  - Expected FAQ 210; Pretrained Word2Vec mean returned it at similarity 0.8725.
  - TF-IDF predicted FAQ 210 at similarity 0.5051 (accepted=False).
  - Word2Vec mean predicted FAQ 210 at similarity 0.9021 (accepted=False).

### Pretrained Word2Vec mean: case C

Queries where Word2Vec mean delivers the correct answer and this model does not.

8 queries; the first 5 are shown.

- Query: Give me the details of the money back timelines if I call off or send back a item.
  - Expected FAQ 281; Word2Vec mean returned it at similarity 0.9209.
  - Pretrained Word2Vec mean predicted FAQ 7 at similarity 0.8138 (accepted=True).
- Query: Please clarify the following point: i want to verify my guarantee particulars for the item but I don't have the guarantee card. What should I do.
  - Expected FAQ 146; Word2Vec mean returned it at similarity 0.9591.
  - Pretrained Word2Vec mean predicted FAQ 2 at similarity 0.9089 (accepted=True).
- Query: I need guidance on this matter: i see that the guarantee terms for my item have changed on Flipkart from when I bought the item. Will this affect my guarantee.
  - Expected FAQ 154; Word2Vec mean returned it at similarity 0.9508.
  - Pretrained Word2Vec mean predicted FAQ 7 at similarity 0.8906 (accepted=True).
- Query: I need guidance on this matter: i lost my guarantee card. How can I get guarantee.
  - Expected FAQ 98; Word2Vec mean returned it at similarity 0.9316.
  - Pretrained Word2Vec mean predicted FAQ 2 at similarity 0.8804 (accepted=True).
- Query: Is it necessary to have to settle the amount anything extra to order or make use of a Present Card?
  - Expected FAQ 298; Word2Vec mean returned it at similarity 0.9287.
  - Pretrained Word2Vec mean predicted FAQ 495 at similarity 0.8813 (accepted=True).

### Pretrained Word2Vec TF-IDF weighted: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

13 queries; the first 5 are shown.

- Query: Which page should someone visit to locate the code for the benefits claimed as part of the Money and Coins for Benefits?
  - Expected FAQ 420; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8205.
  - TF-IDF predicted FAQ 420 at similarity 0.4661 (accepted=False).
- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8726.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
- Query: What steps should someone follow to claim guarantee services for refurbished items?
  - Expected FAQ 66; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.7642.
  - TF-IDF predicted FAQ 66 at similarity 0.5436 (accepted=False).
- Query: Am I allowed to redeem all of my on offer SuperCoins while with the SuperCoin Settle the amount function?
  - Expected FAQ 111; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8034.
  - TF-IDF predicted FAQ 111 at similarity 0.5713 (accepted=False).
- Query: I need guidance on this matter: how long does it take to call off an purchase.
  - Expected FAQ 112; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8504.
  - TF-IDF predicted FAQ 112 at similarity 0.5197 (accepted=False).

### Pretrained Word2Vec TF-IDF weighted: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

10 queries; the first 5 are shown.

- Query: I need guidance on this matter: should I register the item with the make to avail the guarantee benefits on it.
  - Expected FAQ 58; TF-IDF returned it at similarity 0.6992.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 192 at similarity 0.8516 (accepted=True).
- Query: Explain Flipkart Plus degree.
  - Expected FAQ 183; TF-IDF returned it at similarity 0.6387.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 183 at similarity 0.6889 (accepted=False).
- Query: Please clarify the following point: i want to verify my guarantee particulars for the item but I don't have the guarantee card. What should I do.
  - Expected FAQ 146; TF-IDF returned it at similarity 0.6548.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 2 at similarity 0.8534 (accepted=True).
- Query: I need guidance on this matter: how secure is the transaction through SuperCoin Settle the amount.
  - Expected FAQ 296; TF-IDF returned it at similarity 0.6057.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 495 at similarity 0.7893 (accepted=True).
- Query: Clarify what will happen if Ather make approved retailer who will reach me is very far from my location.
  - Expected FAQ 440; TF-IDF returned it at similarity 0.7139.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 449 at similarity 0.8384 (accepted=True).

### Pretrained Word2Vec TF-IDF weighted: case A

Queries where TF-IDF and Word2Vec TF-IDF weighted both fail and this model delivers the correct answer.

9 queries; the first 5 are shown.

- Query: In which place is it possible to verify the expiry related particulars of my Flipkart Plus membership?
  - Expected FAQ 11; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8726.
  - TF-IDF predicted FAQ 11 at similarity 0.5209 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 232 at similarity 0.8926 (accepted=False).
- Query: What steps should someone follow to claim guarantee services for refurbished items?
  - Expected FAQ 66; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.7642.
  - TF-IDF predicted FAQ 66 at similarity 0.5436 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 10 at similarity 0.9115 (accepted=True).
- Query: Am I allowed to redeem all of my on offer SuperCoins while with the SuperCoin Settle the amount function?
  - Expected FAQ 111; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8034.
  - TF-IDF predicted FAQ 111 at similarity 0.5713 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 166 at similarity 0.9289 (accepted=True).
- Query: I need guidance on this matter: how long does it take to call off an purchase.
  - Expected FAQ 112; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8504.
  - TF-IDF predicted FAQ 112 at similarity 0.5197 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 112 at similarity 0.9097 (accepted=False).
- Query: Am I allowed to redeem SuperCoins to avail price reduction for Flipkart Present Card?
  - Expected FAQ 384; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8349.
  - TF-IDF predicted FAQ 384 at similarity 0.5664 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 187 at similarity 0.9351 (accepted=True).

### Pretrained Word2Vec TF-IDF weighted: case C

Queries where Word2Vec TF-IDF weighted delivers the correct answer and this model does not.

6 queries; the first 5 are shown.

- Query: I need guidance on this matter: should I register the item with the make to avail the guarantee benefits on it.
  - Expected FAQ 58; Word2Vec TF-IDF weighted returned it at similarity 0.9143.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 192 at similarity 0.8516 (accepted=True).
- Query: Would it be possible to choose PhonePe digital purse as a money back choice?
  - Expected FAQ 275; Word2Vec TF-IDF weighted returned it at similarity 0.9129.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 495 at similarity 0.7993 (accepted=True).
- Query: Am I allowed to make use of an overseas identifier to sign up?
  - Expected FAQ 197; Word2Vec TF-IDF weighted returned it at similarity 0.9253.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 345 at similarity 0.8016 (accepted=True).
- Query: Explain how this works: how will I claim guarantee for refurbished items.
  - Expected FAQ 74; Word2Vec TF-IDF weighted returned it at similarity 0.9277.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 74 at similarity 0.7346 (accepted=False).
- Query: Explain a 3D Secure passcode.
  - Expected FAQ 471; Word2Vec TF-IDF weighted returned it at similarity 0.9565.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 471 at similarity 0.7293 (accepted=False).

## University FAQ

| Metric | TF-IDF | Word2Vec mean | Word2Vec TF-IDF weighted | Pretrained Word2Vec mean | Pretrained Word2Vec TF-IDF weighted |
| --- | --- | --- | --- | --- | --- |
| FAQs indexed | 500 | 500 | 500 | 500 | 500 |
| Preprocessing | basic | basic | basic | basic | basic |
| Threshold | 0.46 | 0.93 | 0.95 | 0.86 | 0.80 |
| Answerable queries | 150 | 150 | 150 | 150 | 150 |
| Unanswerable queries | 50 | 50 | 50 | 50 | 50 |
| Top-1 accuracy | 0.860 | 0.587 | 0.693 | 0.873 | 0.867 |
| Top-3 accuracy | 0.993 | 0.673 | 0.773 | 0.953 | 0.987 |
| Correct answer rate | 0.833 | 0.493 | 0.493 | 0.627 | 0.813 |
| Accepted but wrong | 17 | 17 | 9 | 3 | 15 |
| Mean similarity of correct Top-1 | 0.7602 | 0.9588 | 0.9643 | 0.8929 | 0.9140 |
| Answerable acceptance rate | 0.947 | 0.607 | 0.553 | 0.647 | 0.913 |
| Unanswerable rejection rate | 0.840 | 0.760 | 0.980 | 0.960 | 0.880 |
| False acceptances | 8 | 12 | 1 | 2 | 6 |
| False rejections | 8 | 59 | 67 | 53 | 13 |

### Pretrained Word2Vec mean: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

3 queries

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec mean returned it at similarity 0.8720.
  - TF-IDF predicted FAQ 97 at similarity 0.7771 (accepted=True).
- Query: Which employers take on people who finish Master of Education in Counselling Psychology (Global Mental Health & Counselling Psychology Field)?
  - Expected FAQ 10; Pretrained Word2Vec mean returned it at similarity 0.9483.
  - TF-IDF predicted FAQ 15 at similarity 0.7100 (accepted=True).
- Query: Give an outline of PhD in Architecture, Landscape, and Design syllabus.
  - Expected FAQ 3; Pretrained Word2Vec mean returned it at similarity 0.8791.
  - TF-IDF predicted FAQ 196 at similarity 0.6530 (accepted=True).

### Pretrained Word2Vec mean: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

34 queries; the first 5 are shown.

- Query: Explain how this works: how will I get alerts.
  - Expected FAQ 288; TF-IDF returned it at similarity 0.6572.
  - Pretrained Word2Vec mean predicted FAQ 288 at similarity 0.8522 (accepted=False).
- Query: I need guidance on this matter: can transfer students participate in the Jump Start degree.
  - Expected FAQ 322; TF-IDF returned it at similarity 0.7833.
  - Pretrained Word2Vec mean predicted FAQ 322 at similarity 0.8466 (accepted=False).
- Query: Which areas of learning belong to Language and Literacies Education?
  - Expected FAQ 118; TF-IDF returned it at similarity 0.6094.
  - Pretrained Word2Vec mean predicted FAQ 118 at similarity 0.8181 (accepted=False).
- Query: Describe the way Museum Studies is arranged.
  - Expected FAQ 91; TF-IDF returned it at similarity 0.5966.
  - Pretrained Word2Vec mean predicted FAQ 91 at similarity 0.7545 (accepted=False).
- Query: I want to establish whether there any financial support choices for me.
  - Expected FAQ 323; TF-IDF returned it at similarity 0.7303.
  - Pretrained Word2Vec mean predicted FAQ 323 at similarity 0.8387 (accepted=False).

### Pretrained Word2Vec mean: case A

Queries where TF-IDF and Word2Vec mean both fail and this model delivers the correct answer.

2 queries

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec mean returned it at similarity 0.8720.
  - TF-IDF predicted FAQ 97 at similarity 0.7771 (accepted=True).
  - Word2Vec mean predicted FAQ 197 at similarity 0.9564 (accepted=True).
- Query: Give an outline of PhD in Architecture, Landscape, and Design syllabus.
  - Expected FAQ 3; Pretrained Word2Vec mean returned it at similarity 0.8791.
  - TF-IDF predicted FAQ 196 at similarity 0.6530 (accepted=True).
  - Word2Vec mean predicted FAQ 53 at similarity 0.9308 (accepted=True).

### Pretrained Word2Vec mean: case C

Queries where Word2Vec mean delivers the correct answer and this model does not.

6 queries; the first 5 are shown.

- Query: I need guidance on this matter: can transfer students participate in the Jump Start degree.
  - Expected FAQ 322; Word2Vec mean returned it at similarity 0.9446.
  - Pretrained Word2Vec mean predicted FAQ 322 at similarity 0.8466 (accepted=False).
- Query: I need guidance on this matter: how should I settle the amount the submission fee.
  - Expected FAQ 418; Word2Vec mean returned it at similarity 0.9420.
  - Pretrained Word2Vec mean predicted FAQ 418 at similarity 0.8553 (accepted=False).
- Query: Please clarify the following point: am I qualified to work in Canada while learning.
  - Expected FAQ 452; Word2Vec mean returned it at similarity 0.9440.
  - Pretrained Word2Vec mean predicted FAQ 452 at similarity 0.8418 (accepted=False).
- Query: Am I expected to require a learn permit to learn in Canada as an overseas student?
  - Expected FAQ 449; Word2Vec mean returned it at similarity 0.9440.
  - Pretrained Word2Vec mean predicted FAQ 450 at similarity 0.8303 (accepted=False).
- Query: Clarify what identifier will alerts come from.
  - Expected FAQ 292; Word2Vec mean returned it at similarity 0.9970.
  - Pretrained Word2Vec mean predicted FAQ 292 at similarity 0.8356 (accepted=False).

### Pretrained Word2Vec TF-IDF weighted: improvements over TF-IDF

Queries where this model delivers the correct answer and TF-IDF does not.

7 queries; the first 5 are shown.

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9411.
  - TF-IDF predicted FAQ 97 at similarity 0.7771 (accepted=True).
- Query: What investigations are carried out in Master of Applied Science (MASc)?
  - Expected FAQ 221; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9020.
  - TF-IDF predicted FAQ 101 at similarity 0.7089 (accepted=True).
- Query: Which scholarly work happens within Ecology and Evolutionary Biology?
  - Expected FAQ 127; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9142.
  - TF-IDF predicted FAQ 48 at similarity 0.5686 (accepted=True).
- Query: What investigations are carried out in Electrical and Computer Engineering?
  - Expected FAQ 67; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8907.
  - TF-IDF predicted FAQ 43 at similarity 0.6784 (accepted=True).
- Query: Which employers take on people who finish Master of Education in Counselling Psychology (Global Mental Health & Counselling Psychology Field)?
  - Expected FAQ 10; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9492.
  - TF-IDF predicted FAQ 15 at similarity 0.7100 (accepted=True).

### Pretrained Word2Vec TF-IDF weighted: reverse cases (case B)

Queries where TF-IDF delivers the correct answer and this model does not.

10 queries; the first 5 are shown.

- Query: Which classes can be taken within Master of Visual Studies in Studio Art?
  - Expected FAQ 1; TF-IDF returned it at similarity 0.6468.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 57 at similarity 0.8936 (accepted=True).
- Query: Which professions follow on from Global Affairs?
  - Expected FAQ 210; TF-IDF returned it at similarity 0.5616.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 96 at similarity 0.7902 (accepted=False).
- Query: What material does Adult Education and Community Development teach?
  - Expected FAQ 20; TF-IDF returned it at similarity 0.8181.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 4 at similarity 0.9242 (accepted=True).
- Query: I need guidance on this matter: how should I settle the amount the submission fee.
  - Expected FAQ 418; TF-IDF returned it at similarity 0.5577.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 252 at similarity 0.8637 (accepted=True).
- Query: Give me the details of an Extra.
  - Expected FAQ 392; TF-IDF returned it at similarity 0.5902.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 392 at similarity 0.7994 (accepted=False).

### Pretrained Word2Vec TF-IDF weighted: case A

Queries where TF-IDF and Word2Vec TF-IDF weighted both fail and this model delivers the correct answer.

6 queries; the first 5 are shown.

- Query: What investigations are carried out in Germanic Literature, Culture and Theory?
  - Expected FAQ 47; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9411.
  - TF-IDF predicted FAQ 97 at similarity 0.7771 (accepted=True).
  - Word2Vec TF-IDF weighted predicted FAQ 161 at similarity 0.9574 (accepted=True).
- Query: What investigations are carried out in Master of Applied Science (MASc)?
  - Expected FAQ 221; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9020.
  - TF-IDF predicted FAQ 101 at similarity 0.7089 (accepted=True).
  - Word2Vec TF-IDF weighted predicted FAQ 25 at similarity 0.9654 (accepted=True).
- Query: Which scholarly work happens within Ecology and Evolutionary Biology?
  - Expected FAQ 127; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.9142.
  - TF-IDF predicted FAQ 48 at similarity 0.5686 (accepted=True).
  - Word2Vec TF-IDF weighted predicted FAQ 64 at similarity 0.9403 (accepted=False).
- Query: What investigations are carried out in Electrical and Computer Engineering?
  - Expected FAQ 67; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8907.
  - TF-IDF predicted FAQ 43 at similarity 0.6784 (accepted=True).
  - Word2Vec TF-IDF weighted predicted FAQ 43 at similarity 0.9215 (accepted=False).
- Query: Is it permitted to modify my class timetable?
  - Expected FAQ 465; Pretrained Word2Vec TF-IDF weighted returned it at similarity 0.8134.
  - TF-IDF predicted FAQ 465 at similarity 0.4235 (accepted=False).
  - Word2Vec TF-IDF weighted predicted FAQ 496 at similarity 0.8949 (accepted=False).

### Pretrained Word2Vec TF-IDF weighted: case C

Queries where Word2Vec TF-IDF weighted delivers the correct answer and this model does not.

1 query

- Query: Am I expected to require a learn permit to learn in Canada as an overseas student?
  - Expected FAQ 449; Word2Vec TF-IDF weighted returned it at similarity 0.9535.
  - Pretrained Word2Vec TF-IDF weighted predicted FAQ 449 at similarity 0.7615 (accepted=False).

