"""Compare every retrieval model on two separate query sets.

Run:  python evaluate.py

Query sets
- data/dev_queries.csv (development set, 53 in-scope + 8 out-of-scope):
  used to compare settings and to TUNE the thresholds in retrieval.py.
  Its scores are optimistic, because the thresholds were picked on it.
- data/final_test_queries.csv (held-out final test, 15 in-scope + 5
  out-of-scope): written after tuning and never used to change anything.
  Thresholds stay frozen, so this is the honest final number.

Metrics
- Top-1 / Top-3: is the expected FAQ ranked first / in the top 3? (in-scope queries)
- Answered right: top-1 is correct AND its score passes the model's threshold.
- OOS rejected: share of out-of-scope queries whose best score is below the threshold.
- Balanced-best thr: the threshold that maximises the AVERAGE of "answered
  right" and "OOS rejected" on the development set. It is not a raw vote
  count, because there are only 8 out-of-scope queries against 53 in-scope
  ones, so an unweighted count would always favour the in-scope side.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import tokenize
from retrieval import MODELS, THRESHOLDS, build_system, load_data, search


BASE = Path(__file__).parent
faq_df = load_data(BASE / "data" / "faq.csv")
dev = pd.read_csv(BASE / "data" / "dev_queries.csv")
final_test = pd.read_csv(BASE / "data" / "final_test_queries.csv")
questions = dict(zip(faq_df["id"], faq_df["question"]))
FLOAT = lambda x: f"{x:.2f}"  # noqa: E731
pd.set_option("display.width", 200)


def word_overlap(query, expected_id):
    """Share of the query's words that also appear in the expected FAQ question."""
    query_words = set(tokenize(query, clean=True))
    faq_words = set(tokenize(questions[int(expected_id)], clean=True))
    return len(query_words & faq_words) / max(len(query_words), 1)


def describe(name, queries):
    in_scope = queries[queries["type"] != "out_of_scope"]
    overlap = np.mean([word_overlap(t.query, t.expected_id) for t in in_scope.itertuples()])
    print(f"{name}: {len(in_scope)} in-scope, {len(queries) - len(in_scope)} "
          f"out-of-scope queries; average word overlap with the expected FAQ "
          f"question {overlap:.0%} (lower means a harder test)")


def best_threshold(rows):
    """Pick the threshold that maximises balanced accept/reject accuracy."""
    candidates = np.round(np.arange(0.05, 1.0, 0.01), 2)
    scoped = [row for row in rows if row["in_scope"]]
    oos = [row for row in rows if not row["in_scope"]]

    def balanced_score(t):
        answered_right = np.mean([row["score"] >= t and row["top1"] for row in scoped])
        oos_rejected = np.mean([row["score"] < t for row in oos]) if oos else 0.0
        return (answered_right + oos_rejected) / 2

    return max(candidates, key=balanced_score)


def score_queries(system, model_name, queries, correct_spelling):
    rows = []
    for test in queries.itertuples():
        results, _, _ = search(model_name, test.query, system, correct_spelling)
        ids = [r["id"] for r in results]
        in_scope_query = test.type != "out_of_scope"
        expected = int(test.expected_id) if in_scope_query else None
        rows.append({
            "type": test.type,
            "in_scope": in_scope_query,
            "top1": bool(ids) and ids[0] == expected,
            "top3": expected in ids,
            "score": results[0]["similarity"] if results else 0.0,
        })
    return rows


def metrics(rows, model_name):
    threshold = THRESHOLDS[model_name]
    scoped = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    typos = [r for r in scoped if r["type"] == "typo"]
    return {
        "Top-1": np.mean([r["top1"] for r in scoped]),
        "Top-3": np.mean([r["top3"] for r in scoped]),
        "Typo top-1": np.mean([r["top1"] for r in typos]),
        "Answered right": np.mean([r["top1"] and r["score"] >= threshold for r in scoped]),
        "OOS rejected": np.mean([r["score"] < threshold for r in oos]),
    }


describe("Development set", dev)
describe("Final test set ", final_test)

systems = {clean: build_system(faq_df, clean) for clean in [False, True]}

print("\n=== Development set (thresholds were tuned on this set) ===")
table = []
for clean, system in systems.items():
    for correct_spelling in [False, True]:
        for model_name in MODELS:
            rows = score_queries(system, model_name, dev, correct_spelling)
            table.append({
                "Model": model_name,
                "Stopwords+stem": "yes" if clean else "no",
                "Spell fix": "yes" if correct_spelling else "no",
                **metrics(rows, model_name),
                "Balanced-best thr": best_threshold(rows),
            })
print(pd.DataFrame(table).to_string(index=False, float_format=FLOAT))

print("\n=== Held-out final test (thresholds frozen; stop words + stemming "
      "and spelling correction on, as in the app) ===")
final_table = [
    {"Model": model_name,
     **metrics(score_queries(systems[True], model_name, final_test, True), model_name)}
    for model_name in MODELS
]
print(pd.DataFrame(final_table).to_string(index=False, float_format=FLOAT))


# Sentence-BERT threshold sweep on the development set only.
print("\nSentence-BERT threshold sweep on the development set (spelling correction on)")
sweep_rows = score_queries(systems[True], "Sentence-BERT", dev, correct_spelling=True)
sweep = []
for threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
    answered = np.mean([r["top1"] and r["score"] >= threshold
                        for r in sweep_rows if r["in_scope"]])
    rejected = np.mean([r["score"] < threshold
                        for r in sweep_rows if not r["in_scope"]])
    sweep.append({"Threshold": threshold, "Answered right": answered,
                  "OOS rejected": rejected, "Balanced": (answered + rejected) / 2})
print(pd.DataFrame(sweep).to_string(index=False, float_format=FLOAT))
