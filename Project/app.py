"""
Streamlit chat interface. Run with: streamlit run app.py

Kept deliberately thin — all real logic lives in src/rag/pipeline.py so
it's testable and reusable from a CLI or API without dragging Streamlit
along as a dependency of your core logic.
"""
from __future__ import annotations

import streamlit as st

from src.db import init_db
from src.monitoring.feedback import save_feedback
from src.rag.pipeline import answer_question

st.set_page_config(page_title="RAG Assistant", page_icon="🔎")
st.title("🔎 RAG Assistant")

# init_db() is idempotent — safe to call on every Streamlit rerun, and
# guarantees the app never queries tables that don't exist yet on a
# fresh container.
init_db()

with st.sidebar:
    strategy = st.selectbox("Retrieval strategy", ["hybrid", "vector", "text"], index=0)
    prompt_variant = st.selectbox("Prompt variant", ["baseline", "strict_citation", "concise"], index=0)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, conversation_pk, sources

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for src in turn["sources"]:
                st.caption(f"doc #{src['id']} · score {src['score']:.3f}")
                st.text(src["content"][:300] + ("…" if len(src["content"]) > 300 else ""))
        col1, col2 = st.columns(2)
        if col1.button("👍", key=f"up_{turn['conversation_pk']}"):
            save_feedback(turn["conversation_pk"], rating=1)
            st.toast("Thanks for the feedback!")
        if col2.button("👎", key=f"down_{turn['conversation_pk']}"):
            save_feedback(turn["conversation_pk"], rating=-1)
            st.toast("Thanks — noted.")

question = st.chat_input("Ask a question")
if question:
    with st.spinner("Retrieving and generating…"):
        result = answer_question(question, strategy=strategy, prompt_variant=prompt_variant)
    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "conversation_pk": result["conversation_pk"],
            "sources": result["sources"],
        }
    )
    st.rerun()
