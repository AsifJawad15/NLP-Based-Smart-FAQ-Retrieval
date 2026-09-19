import re

import nltk
from nltk.stem import PorterStemmer


# Common function words. Question words (what, when, where, who, which, how)
# are kept on purpose: they separate "When was X founded?" from "Where is X?".
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "do", "does", "did", "i", "me", "my", "we", "our", "you", "your",
    "it", "its", "this", "that", "these", "those", "of", "in", "on", "at",
    "to", "for", "by", "with", "from", "and", "or", "about", "can", "could",
    "should", "would", "may", "will", "there", "any", "some", "tell",
    "please", "kuet",
}

stemmer = PorterStemmer()


def preprocess(text):
    """Lowercase text, remove punctuation, and normalize spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text, clean=False):
    """Split preprocessed text into words; optionally drop stop words and stem (Lab 1)."""
    words = preprocess(text).split()

    if clean:
        words = [stemmer.stem(word) for word in words if word not in STOP_WORDS]

    return words


def edit_distance(source, target):
    """Levenshtein distance with insert, delete, and substitute cost 1 (Lab 1)."""
    previous = list(range(len(target) + 1))

    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            cost = 0 if source_char == target_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current

    return previous[-1]


def load_english_words():
    """Load NLTK's English word list, downloading it once if needed.

    Returns an empty set when it is unavailable (for example, offline).
    """
    try:
        from nltk.corpus import words
        try:
            return {word.lower() for word in words.words()}
        except LookupError:
            nltk.download("words", quiet=True)
            return {word.lower() for word in words.words()}
    except Exception:
        return set()


SUFFIXES = [("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
            ("ing", ""), ("ing", "e"), ("est", ""), ("er", "")]


def is_known(word, known_words):
    """True if the word, or the word without a common suffix, is known."""
    if word in known_words:
        return True

    for suffix, replacement in SUFFIXES:
        if word.endswith(suffix) and word[: -len(suffix)] + replacement in known_words:
            return True

    return False


def spell_correct(text, faq_vocabulary, english_words):
    """Replace misspelled words with the closest FAQ vocabulary word (Lab 1).

    A word is only treated as misspelled if it is neither an English word
    nor an FAQ word; otherwise real words such as "cook" would be "fixed"
    to FAQ words such as "book". Words of 3 letters or fewer and numbers
    are left alone. Returns the corrected text and (wrong, fixed) pairs.
    """
    if not english_words:
        return preprocess(text), []

    known_words = english_words | faq_vocabulary
    corrected_words = []
    changes = []

    for word in preprocess(text).split():
        if (len(word) <= 3 or any(c.isdigit() for c in word)
                or is_known(word, known_words)):
            corrected_words.append(word)
            continue

        # Long words may have two typos; shorter ones only one.
        limit = 2 if len(word) >= 7 else 1
        candidates = [
            (edit_distance(word, faq_word), faq_word)
            for faq_word in faq_vocabulary
            if abs(len(faq_word) - len(word)) <= limit
        ]
        best = min(candidates, default=None)

        if best and best[0] <= limit:
            corrected_words.append(best[1])
            changes.append((word, best[1]))
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), changes
