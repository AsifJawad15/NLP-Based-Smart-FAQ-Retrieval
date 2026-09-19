from datetime import datetime
from pathlib import Path

import streamlit as st

from retrieval import MODELS, build_system, load_data, search


DATA_PATH = Path(__file__).parent / "data" / "faq.csv"


def as_of(item):
    return datetime.strptime(item["last_verified"], "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")


def provenance(item):
    return f"📅 Information as of {as_of(item)}  |  Source: {item['source']}"


@st.cache_resource
def load_system(clean):
    return build_system(load_data(DATA_PATH), clean)


st.set_page_config(page_title="Smart FAQ Retrieval", page_icon="🔎")
st.title("🔎 NLP-Based Smart FAQ Retrieval")
st.caption("Ask a question about KUET and retrieve the most relevant FAQ.")

model_name = st.radio("Retrieval model", MODELS, horizontal=True)

col1, col2 = st.columns(2)
clean = col1.toggle(
    "Remove stop words + stemming",
    value=True,
    help="Lab 1 preprocessing applied to FAQs and query.",
)
correct_spelling = col2.toggle(
    "Spelling correction",
    value=True,
    help="Lab 1 edit distance: fixes typos such as 'admisson'.",
)

compare_all = st.checkbox(
    "Compare all models on this query",
    help="Shows each model's best FAQ match side by side.",
)

if model_name == "Sentence-BERT":
    st.caption(
        "Sentence-BERT (pretrained all-MiniLM-L6-v2) reads the full sentence, "
        "so stop-word removal and stemming are not applied to it."
    )

with st.spinner("Loading models..."):
    system = load_system(clean)

if system["sbert"] is None:
    st.warning(
        "Sentence-BERT could not be loaded (install sentence-transformers and "
        "connect to the internet once to download the model)."
    )

query = st.text_input(
    "Your question",
    placeholder="Example: When was KUET founded?",
)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        results, threshold, changes = search(
            model_name, query, system, correct_spelling
        )

        if changes:
            fixes = ", ".join(f"{wrong} → {right}" for wrong, right in changes)
            st.info(f"Did you mean: {fixes}")

        if not results or results[0]["similarity"] < threshold:
            st.error("No sufficiently relevant FAQ was found. Try another question.")
        else:
            best = results[0]
            st.success("Relevant FAQ found")
            st.subheader(best["question"])
            st.caption(provenance(best))
            st.write(best["answer"])
            st.caption(
                f"Category: {best['category']}  |  "
                f"Similarity: {best['similarity']:.2%}  |  "
                f"Threshold: {threshold:.2f}"
            )

        if results:
            st.divider()
            st.subheader("Top 3 FAQ Matches")

            for rank, item in enumerate(results, start=1):
                with st.expander(
                    f"{rank}. {item['question']} — {item['similarity']:.2%}"
                ):
                    st.caption(provenance(item))
                    st.write(item["answer"])
                    st.caption(f"Category: {item['category']} | FAQ ID: {item['id']}")

        if compare_all:
            st.divider()
            st.subheader("Model Comparison")
            rows = []
            for name in MODELS:
                other_results, other_threshold, _ = search(
                    name, query, system, correct_spelling
                )
                top = other_results[0] if other_results else None
                accepted = bool(top) and top["similarity"] >= other_threshold
                rows.append({
                    "Model": name,
                    "Best FAQ match": top["question"] if top else "—",
                    "Similarity": f"{top['similarity']:.2f}" if top else "—",
                    "Threshold": f"{other_threshold:.2f}",
                    "Decision": "Answer" if accepted else "Reject",
                })
            st.dataframe(rows, hide_index=True, width="stretch")
