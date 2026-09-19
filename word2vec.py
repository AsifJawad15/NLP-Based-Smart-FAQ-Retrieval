"""Word2Vec (skip-gram) built from scratch with numpy, following Lab 3.

Same model and maths as Lab 3's Topic 1:
- W1 (V x d) is the embedding matrix, W2 (d x V) the output matrix.
- Forward pass:  h = W1^T x   (x = one-hot centre word, so h is just row W1[centre])
                 u = W2^T h,  y_hat = softmax(u)
- Loss: cross-entropy  -log y_hat[context]
- Backward pass: e = y_hat - y,  dW2 = h e^T,  dW1 = x (W2 e)^T
- Update: plain gradient descent, W -= learning_rate * gradient

The only change from Lab 3 is speed: Lab 3 loops over one (centre, context)
pair at a time on a 10-word toy corpus. Our FAQ corpus has ~15,000 pairs, so
the same forward/backward pass is run on a mini-batch of pairs at once, and
multiplying by a one-hot vector is written as the row lookup it equals.
"""
import numpy as np


def skipgram_pairs(sentences, word2id, window):
    """(centre id, context id) for every word within `window` of a centre word."""
    pairs = []
    for tokens in sentences:
        for i, centre in enumerate(tokens):
            start, end = max(0, i - window), min(len(tokens), i + window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((word2id[centre], word2id[tokens[j]]))
    return np.array(pairs)


def softmax(u):
    e_u = np.exp(u - u.max(axis=1, keepdims=True))  # subtract max for stability
    return e_u / e_u.sum(axis=1, keepdims=True)


class SkipGramWord2Vec:
    def __init__(self, sentences, vector_size=100, window=2, epochs=100,
                 learning_rate=0.5, batch_size=64, seed=42):
        vocab = sorted({word for tokens in sentences for word in tokens})
        self.word2id = {word: i for i, word in enumerate(vocab)}
        self.id2word = vocab
        self.vector_size = vector_size

        rng = np.random.default_rng(seed)
        V = len(vocab)
        self.W1 = rng.normal(0, 0.1, (V, vector_size)).astype(np.float32)
        self.W2 = rng.normal(0, 0.1, (vector_size, V)).astype(np.float32)

        pairs = skipgram_pairs(sentences, self.word2id, window)
        self.loss_history = []
        for _ in range(epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            for start in range(0, len(pairs), batch_size):
                centre, context = pairs[start:start + batch_size].T
                epoch_loss += self._train_batch(centre, context, learning_rate)
            self.loss_history.append(epoch_loss / len(pairs))

    def _train_batch(self, centre, context, learning_rate):
        n = len(centre)
        rows = np.arange(n)

        # Forward pass
        h = self.W1[centre]              # (n x d)  = W1^T x for each one-hot x
        u = h @ self.W2                  # (n x V)  output scores
        y_hat = softmax(u)               # (n x V)  probabilities
        loss = -np.log(y_hat[rows, context] + 1e-12).sum()

        # Backward pass: e = y_hat - y
        e = y_hat
        e[rows, context] -= 1.0
        dW2 = h.T @ e / n                # (d x V)
        dh = e @ self.W2.T / n           # (n x d)  = W2 e, one row per pair

        # Gradient descent; np.add.at adds up repeated centre words correctly
        self.W2 -= learning_rate * dW2
        np.add.at(self.W1, centre, -learning_rate * dh)
        return loss

    def __contains__(self, word):
        return word in self.word2id

    def __getitem__(self, word):
        return self.W1[self.word2id[word]]

    def __len__(self):
        return len(self.id2word)

    def most_similar(self, word, topn=5):
        """Nearest words by cosine similarity, like Lab 3's find_closest_word."""
        norms = np.linalg.norm(self.W1, axis=1)
        target = self[word]
        scores = self.W1 @ target / (norms * np.linalg.norm(target) + 1e-12)
        ranked = [i for i in np.argsort(scores)[::-1] if self.id2word[i] != word]
        return [(self.id2word[i], float(scores[i])) for i in ranked[:topn]]
