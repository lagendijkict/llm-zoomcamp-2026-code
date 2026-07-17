"""
Monitoring dashboard. Run with: streamlit run dashboard.py

Reads directly from conversations + feedback — the same tables the app
writes to — so "what the dashboard shows" and "what actually happened"
can never drift apart the way they would if metrics were computed from a
separate logging pipeline.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.db import get_conn

st.set_page_config(page_title="RAG Monitoring", page_icon="📊", layout="wide")
st.title("📊 RAG Monitoring Dashboard")


@st.cache_data(ttl=60)  # refresh at most once a minute — dashboard doesn't need to be real-time
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_conn() as conn:
        conversations = pd.read_sql(
            "SELECT * FROM conversations ORDER BY created_at DESC LIMIT 5000", conn
        )
        feedback = pd.read_sql(
            """
            SELECT f.*, c.retrieval_strategy, c.model, c.created_at AS conversation_created_at
            FROM feedback f
            JOIN conversations c ON c.id = f.conversation_pk
            ORDER BY f.created_at DESC LIMIT 5000
            """,
            conn,
        )
    return conversations, feedback


conversations, feedback = load_data()

if conversations.empty:
    st.info("No conversations logged yet — ask a question in the main app first.")
    st.stop()

# --- Top-line metrics -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", len(conversations))
col2.metric("Avg response time (s)", f"{conversations['response_time_s'].mean():.2f}")
thumbs_up_rate = (feedback["rating"] == 1).mean() if not feedback.empty else None
col3.metric("Thumbs-up rate", f"{thumbs_up_rate:.0%}" if thumbs_up_rate is not None else "—")
col4.metric("Feedback volume", len(feedback))

st.divider()

# 1. Volume over time
conversations["date"] = pd.to_datetime(conversations["created_at"]).dt.date
volume_by_day = conversations.groupby("date").size().reset_index(name="count")
st.plotly_chart(px.line(volume_by_day, x="date", y="count", title="Conversations per day"), use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    # 2. Response time distribution — surfaces latency regressions, e.g.
    # from a retrieval strategy or context size that got slower.
    st.plotly_chart(
        px.histogram(conversations, x="response_time_s", nbins=30, title="Response time distribution"),
        use_container_width=True,
    )

    # 3. Usage by retrieval strategy — which strategy people/your test
    # traffic actually exercised, useful cross-referenced against
    # retrieval_eval.py's offline hit-rate/MRR numbers.
    strategy_counts = conversations["retrieval_strategy"].value_counts().reset_index()
    strategy_counts.columns = ["strategy", "count"]
    st.plotly_chart(px.pie(strategy_counts, names="strategy", values="count", title="Requests by retrieval strategy"), use_container_width=True)

with col_b:
    # 4. Feedback breakdown by strategy — the online signal that should
    # correlate with your offline retrieval eval; if it doesn't, that's
    # worth investigating (eval set may not represent real usage).
    if not feedback.empty:
        fb_by_strategy = feedback.groupby(["retrieval_strategy", "rating"]).size().reset_index(name="count")
        fb_by_strategy["rating_label"] = fb_by_strategy["rating"].map({1: "👍", -1: "👎"})
        st.plotly_chart(
            px.bar(fb_by_strategy, x="retrieval_strategy", y="count", color="rating_label", barmode="group", title="Feedback by retrieval strategy"),
            use_container_width=True,
        )
    else:
        st.info("No feedback collected yet.")

    # 5. Token usage over time — cost tracking; a silent prompt-size
    # regression (e.g. someone bumps top_k) shows up here before it shows
    # up in a bill.
    if conversations["prompt_tokens"].notna().any():
        tokens_by_day = conversations.groupby("date")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
        st.plotly_chart(
            px.bar(tokens_by_day, x="date", y=["prompt_tokens", "completion_tokens"], title="Token usage per day", barmode="stack"),
            use_container_width=True,
        )

# 6. Slowest recent queries — actionable outlier list, not just an aggregate.
st.subheader("Slowest recent queries")
st.dataframe(
    conversations.nlargest(10, "response_time_s")[["question", "retrieval_strategy", "response_time_s", "created_at"]],
    use_container_width=True,
)
