from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import load_english_words, spell_correct, tokenize
from word2vec import SkipGramWord2Vec


MODELS = ["TF-IDF", "Word2Vec", "Word2Vec (TF-IDF weighted)", "Sentence-BERT"]
THRESHOLDS = {
    "TF-IDF": 0.56,
    "Word2Vec": 0.80,
    "Word2Vec (TF-IDF weighted)": 0.79,
    "Sentence-BERT": 0.40,
}
TOP_K = 3
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_data(csv_path):
    """Load the FAQ dataset."""
    return pd.read_csv(csv_path)


def build_tfidf(faq_df, clean=False):
    """Convert all FAQ questions into TF-IDF vectors."""
    questions = [" ".join(tokenize(q, clean)) for q in faq_df["question"]]
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(questions)
    return vectorizer, faq_vectors


def build_word2vec(faq_df, clean=False):
    """Train our from-scratch skip-gram Word2Vec and build one vector per FAQ question."""
    training_texts = list(faq_df["question"]) + list(faq_df["answer"])
    sentences = [tokenize(text, clean) for text in training_texts]

    model = SkipGramWord2Vec(
        sentences,
        vector_size=100,
        window=3,
        epochs=40,
        learning_rate=1.0,
        batch_size=256,
        seed=42,
    )

    faq_vectors = np.vstack([
        sentence_vector(question, model, clean)
        for question in faq_df["question"]
    ])
    return model, faq_vectors


def idf_weights(vectorizer):
    """Map each word to its IDF value from the fitted TF-IDF vectorizer."""
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))


def sentence_vector(text, model, clean=False, weights=None):
    """Represent a sentence by averaging its known Word2Vec word vectors.

    With `weights` (word -> IDF), rare informative words count more than
    common ones such as "what" or "does" (Lab 3, TF-IDF weighted embeddings).
    Words missing from `weights` get the highest IDF, since they are rare.
    """
    words = [word for word in tokenize(text, clean) if word in model]

    if not words:
        return np.zeros(model.vector_size)

    vectors = [model[word] for word in words]

    if weights is None:
        return np.mean(vectors, axis=0)

    max_idf = max(weights.values())
    word_weights = [weights.get(word, max_idf) for word in words]
    return np.average(vectors, axis=0, weights=word_weights)


@lru_cache(maxsize=1)
def load_sbert():
    """Load the pretrained Sentence-BERT model (downloaded once, then cached).

    It is only used to create embeddings; it is never trained on our FAQs.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL_NAME)


def build_sbert(faq_df):
    """Encode every FAQ question into one Sentence-BERT vector.

    Sentence-BERT reads the whole sentence, so it gets the raw question:
    no stop-word removal or stemming (it needs the grammar and word order).
    Vectors are L2-normalised so a dot product equals cosine similarity.
    """
    model = load_sbert()
    faq_vectors = model.encode(
        list(faq_df["question"]), normalize_embeddings=True
    )
    return model, faq_vectors


def build_system(faq_df, clean=False):
    """Build every retrieval model once, sharing the same preprocessing."""
    vectorizer, tfidf_vectors = build_tfidf(faq_df, clean)
    model, w2v_vectors = build_word2vec(faq_df, clean)
    weights = idf_weights(vectorizer)
    weighted_vectors = np.vstack([
        sentence_vector(question, model, clean, weights)
        for question in faq_df["question"]
    ])

    try:
        sbert, sbert_vectors = build_sbert(faq_df)
    except Exception as error:  # e.g. offline on first run
        print(f"Sentence-BERT unavailable: {error}")
        sbert, sbert_vectors = None, None

    vocabulary = set()
    for text in list(faq_df["question"]) + list(faq_df["answer"]):
        vocabulary.update(tokenize(text))

    return {
        "faq_df": faq_df,
        "clean": clean,
        "vectorizer": vectorizer,
        "tfidf_vectors": tfidf_vectors,
        "word2vec": model,
        "word2vec_vectors": w2v_vectors,
        "idf": weights,
        "weighted_vectors": weighted_vectors,
        "sbert": sbert,
        "sbert_vectors": sbert_vectors,
        "vocabulary": vocabulary,
        "english_words": load_english_words(),
    }


def make_results(faq_df, scores):
    """Return the top FAQ matches with their similarity scores."""
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    results = []

    for index in top_indices:
        row = faq_df.iloc[index]
        results.append({
            "id": int(row["id"]),
            "question": row["question"],
            "answer": row["answer"],
            "category": row["category"],
            "source": row["source"],
            "last_verified": row["last_verified"],
            "similarity": float(scores[index]),
        })

    return results


def search_tfidf(query, system):
    """Search FAQs using TF-IDF + cosine similarity."""
    query_text = " ".join(tokenize(query, system["clean"]))
    query_vector = system["vectorizer"].transform([query_text])

    if query_vector.nnz == 0:
        return []

    scores = cosine_similarity(query_vector, system["tfidf_vectors"])[0]
    return make_results(system["faq_df"], scores)


def search_word2vec(query, system, weighted=False):
    """Search FAQs using (optionally TF-IDF weighted) Word2Vec + cosine similarity."""
    weights = system["idf"] if weighted else None
    faq_vectors = system["weighted_vectors"] if weighted else system["word2vec_vectors"]
    query_vector = sentence_vector(query, system["word2vec"], system["clean"], weights)

    if not np.any(query_vector):
        return []

    scores = cosine_similarity(query_vector.reshape(1, -1), faq_vectors)[0]
    return make_results(system["faq_df"], scores)


def search_sbert(query, system):
    """Search FAQs using Sentence-BERT embeddings + cosine similarity."""
    if system["sbert"] is None:
        return []

    query_vector = system["sbert"].encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_vector, system["sbert_vectors"])[0]
    return make_results(system["faq_df"], scores)


def search(model_name, query, system, correct_spelling=False):
    """Run one retrieval model. Returns (results, threshold, spelling changes)."""
    changes = []
    if correct_spelling:
        query, changes = spell_correct(
            query, system["vocabulary"], system["english_words"]
        )

    if model_name == "TF-IDF":
        results = search_tfidf(query, system)
    elif model_name == "Word2Vec":
        results = search_word2vec(query, system)
    elif model_name == "Sentence-BERT":
        results = search_sbert(query, system)
    else:
        results = search_word2vec(query, system, weighted=True)

    return results, THRESHOLDS[model_name], changes
