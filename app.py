"""Local Streamlit interface for the KUET demo and research comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from main import build_answerer
from src.data_loader import load_corpus_config, load_faq_dataset
from src.gui_support import corpora_for_purpose, file_change_signature, saved_benchmark_rows
from src.preprocessing import preprocess_text
from src.retrieval_models import MODEL_LABELS


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
MODELS_ROOT = BASE_DIR / "models"
REPORTS_ROOT = BASE_DIR / "reports"


@st.cache_resource(show_spinner=False)
def cached_answerer(
    corpus_key: str, model: str, signature: str
) -> tuple[Any, float, str, Any, dict[str, Any]]:
    """Load a frozen inference resource; signature invalidates stale entries."""

    del signature
    directory = DATA_ROOT / corpus_key
    faq_data = load_faq_dataset(directory)
    config = load_corpus_config(directory)
    answer, threshold, preprocessing = build_answerer(directory, faq_data, config, model)
    return answer, threshold, preprocessing, faq_data, config


def get_answerer(corpus_key: str, model: str):
    directory = DATA_ROOT / corpus_key
    signature = file_change_signature(directory, model, MODELS_ROOT)
    return cached_answerer(corpus_key, model, signature)


def clear_if_selection_changed(state_key: str, selection: str, result_key: str) -> None:
    previous = st.session_state.get(state_key)
    if previous is not None and previous != selection:
        st.session_state.pop(result_key, None)
    st.session_state[state_key] = selection


def render_details(
    result: dict[str, Any], query: str, corpus_name: str,
    model: str, preprocessing_name: str, config: dict[str, Any],
) -> None:
    with st.expander("Show NLP Details"):
        options = {
            "remove_stopwords": bool(config["remove_stopwords"]) if model == "tfidf" else False,
            "lemmatize": bool(config["lemmatize"]) if model == "tfidf" else False,
        }
        st.write(f"**Corpus:** {corpus_name}")
        st.write(f"**Model:** {MODEL_LABELS[model]}")
        st.write(f"**Preprocessing:** {preprocessing_name}")
        st.code(preprocess_text(query, **options) or "(no usable tokens)")
        matches = result["top_matches"]
        best = matches[0] if matches else None
        st.write(f"**Nearest FAQ ID:** {best['faq_id'] if best else '—'}")
        st.write(f"**Nearest FAQ:** {best['question'] if best else 'No candidate'}")
        st.write(f"**Similarity:** {best['similarity'] if best else 0.0:.4f}")
        st.write(f"**Threshold:** {result['threshold']:.4f}")
        rows = [
            {
                "Rank": rank,
                "FAQ ID": match["faq_id"],
                "Candidate question": match["question"],
                "Similarity": round(match["similarity"], 4),
                "Status": "accepted" if result["found"] and rank == 1 else "unaccepted",
            }
            for rank, match in enumerate(matches, start=1)
        ]
        st.dataframe(
            pd.DataFrame(
                rows,
                columns=["Rank", "FAQ ID", "Candidate question", "Similarity", "Status"],
            ),
            hide_index=True,
            width="stretch",
        )


def render_assistant() -> None:
    st.subheader("FAQ Assistant")
    st.caption("Ask a natural English question. The assistant returns only a stored, sourced answer.")
    corpora = corpora_for_purpose(DATA_ROOT, "demo")
    if not corpora:
        st.error("No demo corpus is available.")
        return
    ordered = sorted(corpora, key=lambda key: (key != "kuet", key))
    corpus_key = st.selectbox(
        "FAQ corpus",
        ordered,
        format_func=lambda key: str(load_corpus_config(corpora[key])["display_name"]),
        key="assistant_corpus",
    )
    clear_if_selection_changed("assistant_previous_corpus", corpus_key, "assistant_result")
    query = st.text_input(
        "Ask a question",
        placeholder="For example: How many books can an undergraduate borrow?",
        key="assistant_query",
    )
    if st.button("Search FAQ", type="primary", key="assistant_search"):
        if not query.strip():
            st.session_state.pop("assistant_result", None)
            st.warning("Enter a question before searching.")
        else:
            try:
                answer, threshold, preprocessing_name, _faq, config = get_answerer(
                    corpus_key, "tfidf"
                )
                st.session_state["assistant_result"] = {
                    "query": query,
                    "result": answer(query),
                    "threshold": threshold,
                    "preprocessing": preprocessing_name,
                    "config": config,
                    "corpus_key": corpus_key,
                }
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                st.error(str(error))

    saved = st.session_state.get("assistant_result")
    if not saved:
        return
    result = saved["result"]
    status_slot = st.empty()
    question_slot = st.empty()
    answer_slot = st.empty()
    source_slot = st.empty()
    if result["found"]:
        best = result["best_match"]
        status_slot.success("Relevant FAQ found")
        question_slot.markdown(f"**Matched FAQ:** {best['question']}")
        answer_slot.markdown(f"**Answer:** {best['answer']}")
        source_slot.markdown(f"[Open official source]({best['source']})")
    else:
        status_slot.warning("No sufficiently relevant FAQ found")
        question_slot.caption(
            "Nearest candidates are available below for diagnosis, but no candidate answer was accepted."
        )
    render_details(
        result,
        saved["query"],
        str(saved["config"]["display_name"]),
        "tfidf",
        saved["preprocessing"],
        saved["config"],
    )


def render_comparison() -> None:
    st.subheader("Model Comparison")
    st.caption("Run the seven existing research models on one query and one frozen research corpus.")
    corpora = corpora_for_purpose(DATA_ROOT, "research")
    if not corpora:
        st.error("No research corpus is available.")
        return
    corpus_key = st.selectbox(
        "Research corpus",
        sorted(corpora),
        format_func=lambda key: str(load_corpus_config(corpora[key])["display_name"]),
        key="comparison_corpus",
    )
    clear_if_selection_changed("comparison_previous_corpus", corpus_key, "comparison_result")
    query = st.text_input(
        "Comparison question",
        placeholder="Ask the same question of all seven models",
        key="comparison_query",
    )
    if st.button("Compare models", type="primary", key="comparison_search"):
        if not query.strip():
            st.session_state.pop("comparison_result", None)
            st.warning("Enter a question before comparing models.")
        else:
            rows = []
            errors = []
            with st.spinner("Loading requested research models..."):
                for model, label in MODEL_LABELS.items():
                    display = label + (" (experimental)" if model in {"rnn", "bilstm"} else "")
                    try:
                        answer, threshold, _preprocessing, _faq, _config = get_answerer(
                            corpus_key, model
                        )
                        result = answer(query)
                        matches = result["top_matches"]
                        rows.append(
                            {
                                "Model": display,
                                "Predicted FAQ": matches[0]["question"] if matches else "—",
                                "Similarity": round(matches[0]["similarity"], 4) if matches else 0.0,
                                "Threshold": round(threshold, 4),
                                "Decision": "Answer" if result["found"] else "Reject",
                            }
                        )
                    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
                        errors.append(f"{display}: {error}")
            st.session_state["comparison_result"] = {"rows": rows, "errors": errors}

    saved = st.session_state.get("comparison_result")
    if saved:
        if saved["rows"]:
            st.dataframe(pd.DataFrame(saved["rows"]), hide_index=True, width="stretch")
        for error in saved["errors"]:
            st.error(error)
    st.info(
        "Similarity values come from different vector spaces and are not directly comparable. "
        "Each decision uses that model's own validation-tuned threshold."
    )
    with st.expander("Saved benchmark metrics"):
        rows = saved_benchmark_rows(REPORTS_ROOT, corpus_key)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No saved benchmark reports were found for this corpus.")


st.set_page_config(page_title="Smart FAQ", page_icon="🎓", layout="wide")
st.title("Smart FAQ Retrieval System")
st.write("A local, stored-answer FAQ assistant and a separate seven-model research comparison.")
assistant_tab, comparison_tab = st.tabs(["FAQ Assistant", "Model Comparison"])
with assistant_tab:
    render_assistant()
with comparison_tab:
    render_comparison()
