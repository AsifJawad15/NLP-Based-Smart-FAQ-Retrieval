# NLP-Based Smart FAQ Retrieval — Sentence-BERT Version

A copy of the Minimal version with one extra model: a **pretrained Sentence-BERT** (`all-MiniLM-L6-v2`). TF-IDF, Word2Vec and weighted Word2Vec are unchanged. No Transformer is trained here; the pretrained model is only used to turn sentences into vectors.

A small, explainable FAQ retrieval project using the KUET FAQ dataset.

## Pipeline

```text
User Question
     ↓
Spelling Correction (edit distance, optional)
     ↓
Text Preprocessing (+ stop words & stemming, optional)
     ↓
Retrieval Mode
   ↙        ↓            ↓              ↘
TF-IDF  Word2Vec  Word2Vec (TF-IDF   Sentence-BERT
                   weighted)         (pretrained, raw sentence)
   ↘        ↓            ↓              ↙
Cosine Similarity
     ↓
Top-3 FAQ
     ↓
Threshold Check
  ↙           ↘
Match       No Match
  ↓             ↓
Answer      Try another
```

## Files

- `data/faq.csv` — 215 KUET FAQs, each with its official `source` page and a `last_verified` date (2026-09-20); the app shows both before every answer
- `data/dev_queries.csv` — development set (53 in-scope + 8 out-of-scope), used to tune thresholds
- `data/final_test_queries.csv` — held-out final test (15 in-scope + 5 out-of-scope), never used for tuning
- `.streamlit/config.toml` — disables the file watcher so the terminal stays free of `torchvision` warnings
- `preprocessing.py` — text cleaning, stop words, stemming, edit-distance spelling correction
- `word2vec.py` — skip-gram Word2Vec built from scratch with numpy (Lab 3): one-hot input, `W1`/`W2` matrices, softmax, cross-entropy, gradient descent
- `retrieval.py` — TF-IDF, Word2Vec, TF-IDF weighted Word2Vec, Sentence-BERT (`load_sbert`, `build_sbert`, `search_sbert`), cosine similarity, ranking, threshold
- `evaluate.py` — compares all models on the development set and the held-out final test
- `app.py` — Streamlit interface (4 models + "Compare all models" table)

## Run on Windows PowerShell

```powershell
cd "PATH\TO\Smart-FAQ-SBERT"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
python evaluate.py        # model comparison table
```

The first run downloads NLTK's English word list (spelling correction) and the Sentence-BERT model (~90 MB, from Hugging Face). Both need internet once, then are cached. If Sentence-BERT cannot load, the other three models still work.

To save disk space, install CPU-only PyTorch first: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Models

### TF-IDF
Converts FAQ questions and the user query into TF-IDF vectors, then compares them with cosine similarity.

### Word2Vec
**Built from scratch** in `word2vec.py`, following Lab 3's skip-gram model (no gensim). For every (centre word, context word) pair within a 3-word window: `h = W1ᵀx`, `u = W2ᵀh`, `ŷ = softmax(u)`, cross-entropy loss, `e = ŷ − y`, `dW2 = h eᵀ`, `dW1 = x (W2 e)ᵀ`, then gradient descent. It is trained on the FAQ questions and answers (100-dimensional vectors, 40 epochs). The only change from Lab 3 is that pairs are processed in mini-batches of 256 instead of one at a time, because the FAQ corpus has ~15,000 pairs instead of Lab 3's 10-word sentence. Each FAQ question/query is represented by the average of its known word vectors (Lab 3's mean word embedding). Retrieval still compares the user query only with FAQ questions.

TF-IDF uses scikit-learn's `TfidfVectorizer`, exactly as Lab 2 does.

### Word2Vec (TF-IDF weighted)
Same Word2Vec model, but each word vector is weighted by its IDF value before averaging (Lab 3). Rare, informative words such as "scholarship" count more than common words such as "what" or "does".

### Sentence-BERT
A pretrained Transformer sentence encoder. Each FAQ question is encoded once at startup; the user query is encoded at search time; the top-3 FAQs by cosine similarity are returned (same `make_results()` as the other models). It receives the raw (spell-corrected) sentence, not the stop-word-removed/stemmed version, because it uses word order and context.

**How it differs:** TF-IDF looks at important word overlap. Word2Vec uses a fixed vector per word, so a word has the same vector in every context. Sentence-BERT uses self-attention, so each word's representation depends on its neighbours, and the whole sentence is pooled into one vector.

## Lab 1 Extensions

- **Stop words + stemming:** removes function words and reduces words to their stem (Porter), so "books" and "book" match. Question words (what, when, where…) are kept.
- **Spelling correction:** unknown words are replaced by the closest FAQ word by Levenshtein distance (e.g. `admisson` → `admission`). Only words that are not real English words are corrected, so "cook" is not turned into "book".

## Evaluation

`python evaluate.py` reports top-1 / top-3 accuracy, typo accuracy, answered-correctly rate (top-1 correct and above threshold), and out-of-scope rejection for every model, on two separate query sets:

- **Development set** (`dev_queries.csv`, 53 in-scope + 8 out-of-scope, 37% average word overlap with the expected FAQ). The thresholds were tuned on it, so its scores are optimistic.
- **Held-out final test** (`final_test_queries.csv`, 15 in-scope + 5 out-of-scope, 36% overlap). Written after tuning; thresholds stay frozen. This is the honest final number.

Both sets are small, so treat the numbers as trends, not precise results. With stop words + stemming + spelling correction on (the app's default):

| Model | Dev Top-1 | Dev Top-3 | Dev OOS rejected | **Final Top-1** | **Final Top-3** | **Final OOS rejected** |
|---|---|---|---|---|---|---|
| TF-IDF | 42% | 53% | 88% | 53% | 67% | 100% |
| Word2Vec (from scratch) | 45% | 58% | 62% | 40% | 60% | 60% |
| Word2Vec (TF-IDF weighted) | 43% | 58% | 88% | 40% | 67% | 60% |
| **Sentence-BERT** | **79%** | **91%** | **100%** | **93%** | **93%** | 60% |

Sentence-BERT has the best retrieval accuracy on both sets. A likely reason: the queries are paraphrases that share few words with the FAQ, and a model pretrained on large amounts of text already knows that e.g. "doctorate" and "PhD" mean the same thing, while our Word2Vec only saw 215 FAQs.

**Known limitation (found by the held-out test):** Sentence-BERT's out-of-scope rejection fell from 100% on the development set to 60% on the final test. It answered *"What is the admission fee at BUET?"* (0.71) and *"How do I apply for a UK student visa?"* (0.43) with KUET admission FAQs. Similarity alone cannot separate "admission at another university" from "admission at KUET", so raising the threshold would reject correct answers too.

## Thresholds

- TF-IDF: `0.56`
- Word2Vec: `0.80`
- Word2Vec (TF-IDF weighted): `0.79`
- Sentence-BERT: `0.40`

If the best similarity score is below the selected model's threshold, the system returns no answer instead of forcing an unrelated FAQ. The first three were tuned on the development set `data/dev_queries.csv` by maximising the average of "answered right" and "out-of-scope rejected" (`evaluate.py`'s `Balanced-best thr` column). The one exception is Word2Vec: its balanced-best threshold (0.95) would answer only 9% of development questions correctly, so `0.80` was chosen instead. It answers more questions correctly than 0.85–0.90 and rejects the same share of out-of-scope questions.

For Sentence-BERT, the sweep showed every threshold from 0.30 to 0.45 gives the same best balance. On the development set, the highest out-of-scope score is 0.29 ("tuition fee at Dhaka University") and the lowest correct answer scores 0.45, so `0.40` was chosen near the middle of that gap rather than at its edge. With only 8 out-of-scope queries, this is a starting point, not a guarantee.
