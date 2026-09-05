# Review Findings and Implemented Refinements

Updated 2026-09-05. The historical measurements below describe baseline commit
`d5c0f1a`; the local refinement supersedes the old behavior and exclusions.

**Current status:** retain the TF-IDF architecture. Implemented zero-vector
rejection and metric masking, two-stage validation selection, correct-answer
metrics, a manual-evaluation scaffold, and notebook/documentation corrections.
Both corpora now select basic preprocessing; thresholds remain 0.46 / 0.58.
See the implementation record below for current results. No corpus rebuild or
Git push is part of this refinement.

An external review of this project was supplied as a 20-page PDF. This file
records what was checked, what turned out to be true, what turned out to be
false, and the implemented changes. Historical Git checks below are the
original review snapshots, not a fresh remote verification.

## What the reviewed document actually is

The file is `lab project/NLP - NLP Project Proposal Guide.pdf`. Despite the
name it is **not a course document, not a rubric, and not an instructor
brief** — it is a printed transcript of a prior AI code review of
`Smart_FAQ.zip`, stored in the student's own planning folder alongside
`project idea.txt`, `plan2.txt` and `plan3.txt`.

The genuine submitted proposal is `lab4/project proposal.pdf`
(Asif Jawad 2107007, Salek Bin Hossain 2107026).

The review is treated here as a competent peer opinion. Every substantive
claim was re-tested against the running code before any of it was acted on.

## Historical verification against d5c0f1a

| # | Claim | Verdict |
| --- | --- | --- |
| 1 | All-OOV query is accepted at threshold 0.00 | **Confirmed** |
| 2 | Tuning criterion ignores retrieval correctness | **Confirmed, and it costs accuracy** |
| 3 | An end-to-end "Correct Answer Rate" metric is missing | **Confirmed** (0.820 / 0.820) |
| 4 | Notebook lacks the TF-IDF and cosine equations | **Confirmed** |
| 5 | Lemmatization is noun-only, not POS-aware | **Confirmed** |
| 6 | Evaluation queries are machine-generated | Already documented; recommendation accepted |
| 7 | Corpus is not the original 10 x 50 balanced design | Already documented; no change needed |
| 8 | Learn five functions deeply for the viva | Advice, no code implication |
| 9 | Add dataset licence attribution | Valid, optional |
| 10 | **"This repository is empty" on GitHub** | **False** |
| 11 | Submit a clean archive, not the working directory | Valid, optional |
| 12 | Working tree is dirty with CRLF noise | **Does not apply** |

### 1. All-OOV queries are accepted at threshold 0.00 — confirmed

`retrieve_tfidf` returns `[]` only when preprocessing leaves no tokens. A query
made of real words that are absent from the fitted vocabulary survives that
check, transforms to an all-zero vector, and every cosine becomes exactly 0.0.

Measured on both corpora:

```
processed = 'zzqqxx wwvvuu ttrrpp nnmmkk'   query vector nnz = 0
threshold 0.00  -> found=True   best_id=1   similarity=0.0
frozen threshold -> found=False
```

This contradicts the project plan's rule that blank or out-of-vocabulary input
must produce no answer. It is harmless at the frozen thresholds (0.46 / 0.58)
and does not affect any published number, but it is trivially reproducible in a
live demonstration.

### 2. The tuning criterion ignores retrieval correctness — confirmed, and material

`balanced_accept_reject_score` asks only whether the top similarity cleared the
threshold. It never asks whether the FAQ ranked first was the expected one, so
an answerable query that returns the **wrong** answer still counts as a success
when selecting the preprocessing configuration.

Validation Top-1 accuracy, measured per configuration (30 answerable queries):

| Configuration | University Top-1 | University Top-3 | E-commerce Top-1 |
| --- | --- | --- | --- |
| basic | **0.9000** | 1.0000 | 0.9000 |
| stopwords_removed | 0.8667 | 1.0000 | 0.9000 |
| lemmatized | **0.9000** | 1.0000 | 0.9000 |

`basic` and `lemmatized` are **tied** on university retrieval. Lemmatization was
selected purely on the rejection score — and it retrieves *worse* on the
held-out test set.

Three candidate designs, measured end to end on the test set:

| Design | University | Top-1 | Correct Answer Rate | FA | FR |
| --- | --- | --- | --- | --- | --- |
| Current (single criterion) | lemmatized @ 0.46 | 0.847 | 0.820 | 8 | 8 |
| Correctness-aware single criterion | basic @ 0.60 | 0.860 | 0.733 | 0 | 29 |
| **Two-step selection** | **basic @ 0.46** | **0.860** | **0.833** | 8 | 8 |

E-commerce selects `basic @ 0.58` under all three designs and is unchanged
(Top-1 0.953, Top-3 0.980).

The correctness-aware single criterion drives false acceptances to zero but
over-tightens the threshold, rejecting 29 answerable queries and dropping the
end-to-end rate to 0.733. The two-step design is the best of the three.

**Consequence of this finding: the README sentence "lemmatization helps the
university corpus" is not supported by the experiment and must be corrected.**

### 3. Correct Answer Rate — confirmed

Top-1 accuracy is computed before the threshold decision, so it overstates what
a user actually receives. The end-to-end figure (correct Top-1 **and** accepted,
over answerable queries):

| Corpus | Top-1 | Acceptance | Correct Answer Rate | Accepted but wrong |
| --- | --- | --- | --- | --- |
| University | 0.847 | 0.947 | **0.820** | 19 |
| E-commerce | 0.953 | 0.827 | **0.820** | 1 |

Both corpora deliver a correct answer 82% of the time despite very different
Top-1 scores, because e-commerce rejects far more and university accepts more
wrong answers. This matches the reviewer's independently computed figures
exactly.

### 4. Notebook has no equations — confirmed

The notebook names TF-IDF four times and explains it in prose, but contains no
LaTeX: no IDF formula, no cosine formula.

### 5. Lemmatization is noun-only — confirmed

`preprocessing.py` calls `lemmatizer.lemmatize(token)` with no POS argument, so
WordNet assumes a noun:

```
books   -> book       runs    -> run        studies -> study
running -> running    ran     -> ran        better  -> better    applied -> applied
```

This must not be described as full morphological or POS-aware lemmatization.

### 10. "This repository is empty" — false

```
local  HEAD              d5c0f1afcbce7a45c365f0c469ce570faae62dc0
origin/main              d5c0f1afcbce7a45c365f0c469ce570faae62dc0
files on origin/main     33
working tree changes     0
```

`origin/main` carries the full project. The reviewer most likely opened a cached
page, or checked before the final push. No action is required beyond opening the
link in a private window to confirm.

### 12. Dirty working tree — does not apply

`git status --short` reports zero changes. The CRLF noise the reviewer saw was
an artifact of unzipping the archive, not a property of the repository.

## Course alignment (Labs 1-5)

Surveyed from the lab materials in `d:\4.1\NLP\lab`.

| Lab | Topic | Relationship to this project |
| --- | --- | --- |
| 1 | Regex cleaning, tokenization, stopwords, Porter stemming, WordNet lemmatization, edit distance | Used. Lab 1 demonstrates a fixed verb POS argument; the project uses the noun default |
| 2 | CountVectorizer, **TF-IDF**, n-gram language models, smoothing, Shannon's game | The core of Phase 1. The `full_preprocess(text, remove_stopwords, use_lemmatization)` signature this project follows comes from here, and Lab 2 also uses the plain noun-default lemmatizer |
| 3 | Word2Vec skip-gram from scratch, **cosine similarity**, `find_closest_word`, Naive Bayes, logistic regression, mean and TF-IDF-weighted document vectors | Cosine similarity is used. Word2Vec is correctly deferred to Phase 2 |
| 4 | PyTorch RNN / BiLSTM / POS tagging / seq2seq, gensim embeddings | Correctly excluded from Phase 1 |
| 5 | Encoder-only Transformer (BERT-like) | Future phase |

Three observations worth carrying into the viva:

1. **Cosine similarity is taught in Lab 3, not Lab 2.** Lab 2 stops at TF-IDF as
   a representation. This project therefore spans two labs, which is currently
   not stated anywhere in the documentation.
2. **No lab demonstrates validation splits, threshold tuning, top-k retrieval,
   or rejection metrics.** Only Lab 4 has any train/test split, and it is a
   two-way split with no validation set. This project's evaluation methodology
   is more rigorous than anything the syllabus shows.
3. **The noun-default lemmatizer matches Lab 2 exactly**, so the implementation
   is syllabus-aligned even though Lab 1 demonstrates a fixed verb POS argument. Neither that argument alone nor this project performs POS tagging. The
   safe framing is "basic WordNet lemmatization without POS tagging".

Assessment mode across the course materials is **viva** (oral defence); several
lab notebooks ship viva cheat-sheets. No marking rubric, marks breakdown, or
deadline exists anywhere in the lab tree.

The only recorded instructor instruction, from `plan3.txt`, is that **corpus
collection should be documented** — which `docs/DATA_SOURCES.md` already
satisfies directly.

## Current assessment and implementation

The architecture remains suitable for an undergraduate NLP lab project. The
submitted proposal makes embedding-based methods optional. The application uses
Lab 1 preprocessing, Lab 2 TF-IDF and Lab 3 cosine similarity without requiring
identical lab code. RNN/LSTM, POS tagging, Word2Vec and Transformers are not
needed to finish this phase.

### 1. Zero-vector correctness: implemented

`retrieve_tfidf` returns no matches when `query_vector.nnz == 0`, including at
threshold zero. `rank_queries` retains `has_tokens` and adds `has_features`.
Acceptance and Top-1/Top-3 correctness use the feature mask.

The implementation review found an additional case missed by the original
review: an answerable OOV query labelled FAQ 1 received Top-1 and Top-3 credit
from stable zero-score ties even at a frozen positive threshold. It now receives
zero retrieval credit. Error examples record no retrieved FAQ for such rows.

Runtime and batch behavior are checked together for empty, punctuation-only,
stopword-only, all-OOV, mixed known/OOV and exact-match inputs.

### 2. Two-stage selection: implemented

- Step A, `select_preprocessing`: use answerable validation queries only and
  select by Top-1, then Top-3, then simpler configuration.
- Step B, `tune_threshold`: sweep all 101 thresholds for the selected
  representation only; maximize balanced answerable acceptance/unanswerable
  rejection and break ties toward the higher threshold.

The result retains existing selected-result fields and adds
`preprocessing_comparison`. `per_config_best` now contains one row for the
selected configuration, and `sweep` contains 101 rows. The CLI prints both
stages and saves the comparison table. Both classes are required for tuning.

The protocol is chosen for its separation of retrieval quality and rejection,
not as a claim of statistically proven superiority on a small benchmark. The
synthetic test results have already been examined while revising the project;
we no longer call them an untouched, one-time final assessment.

### 3. Correct-answer metrics: implemented

`correct_answer_rate` is correct Top-1 AND accepted, divided by all answerable
queries. `accepted_wrong_count` counts accepted answerable queries with an
incorrect Top-1. Neither is a replacement for false acceptance, which counts
unanswerable queries that receive an answer. Both metrics appear in JSON,
terminal and Markdown reports and the notebook.

Regenerated synthetic benchmark results:

| Metric | University | E-commerce |
| --- | --- | --- |
| Configuration | basic | basic |
| Threshold | 0.46 | 0.58 |
| Top-1 | 0.860 | 0.953 |
| Top-3 | 0.993 | 0.980 |
| Correct answer rate | 0.833 | 0.820 |
| Accepted but wrong | 17 | 1 |
| False acceptance | 8 | 2 |
| False rejection | 8 | 26 |

### 4. Human evaluation: scaffold implemented, collection pending

`data/manual_evaluation/` contains two header-only query CSVs and a collection
guide. Format examples live in the guide, not in measured files. The target is
20 answerable + 10 unanswerable questions per domain, written from intent
descriptions without seeing FAQ wording and independently labelled by the team.

`python evaluate.py manual [--corpus university|ecommerce]` uses frozen settings
without tuning. Header-only files print human evaluation pending and produce no
current metrics. Populated files validate labels and FAQ references and save
separate manual JSON/Markdown reports. The default loader continues to reject
empty validation/test files. No human-written performance has been measured.

### 5. Teaching and documentation: corrected

- The notebook explains raw TF counts, natural-log smoothed IDF, L2 normalization
  and cosine, with a numerical example checked against the production vectorizer.
- Its vocabulary check uses the index's actual preprocessing options. Previously
  it compared a lemmatized vocabulary against basic question tokens, showing 38
  misleading extra terms.
- Positive demonstrations check expected FAQ ids. The returns/refund paraphrase
  is explicitly a failure: rejected at 0.335 with warranty FAQ 42 ranked first.
- The puppy false acceptance is remeasured under basic preprocessing: 0.469757,
  now matching FAQ 241, "Do I have to be a teacher to apply to OISE?". The former
  amending-application example and 0.472 score belong only to the old baseline.
- The unsupported claim that lemmatization improves university retrieval is
  removed. Basic and lemmatized tie on validation Top-1/Top-3; simplicity selects
  basic. Lemmatization remains a noun-default WordNet comparison, without POS
  tagging. NLTK remains a runtime package; downloaded corpora support optional
  preprocessing, comparison tests and the full notebook.

## Decisions and boundaries

The project owner explicitly included the OOV fix and correct-answer metrics in
this refinement, superseding the original review's exclusions. Human query
collection remains a team task; only its scaffold is implemented.

Keep the existing 500-record corpora, synthetic query sets, separate domain
indexes, function-based modules, terminal interface and notebook. No new model,
POS tagger, spelling correction, GUI, deployment, Git push or corpus rebuild.
Repository licensing/packaging changes remain outside this refinement.

Structural validation does not certify every source answer or paraphrase's
semantic correctness. Source truth and human authorship require separate review.

## Verification commands

Verified locally on 2026-09-05: 47 unit tests passed; all 22 notebook code cells
executed with zero errors and outputs saved; corpus validation, compileall and
pip check passed. Both corpora's synthetic reports reproduce the current numbers above.
Terminal demonstrations also passed with socket connections blocked and NLTK
resource paths empty, confirming basic-mode operation without downloaded corpora.
The manual CLI reports pending work for each header-only template. The six
finalized FAQ/validation/test CSVs remain unchanged.

Use the project `.venv` interpreter from `Smart_FAQ`:

```powershell
python -m unittest discover -s tests
python scripts/prepare_datasets.py validate
python evaluate.py all
python evaluate.py manual
python evaluate.py manual --corpus university
python -m compileall -q src tests scripts main.py evaluate.py
python -m pip check
python main.py --corpus university --query "How do I receive university alerts?"
python main.py --corpus ecommerce --query "What cards can I save on Flipkart?"
python main.py --corpus university --query "How do I train a puppy to sit?"
```

Re-execute the notebook headlessly; require all assertions to pass and its
computed values to match saved reports. Verify finalized FAQ and synthetic
query CSVs are unchanged. No remote verification or push is required.
