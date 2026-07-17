"""
End-to-end RAG turn: retrieve -> build prompt -> call LLM -> persist.

Persisting every turn to `conversations` here (not as an afterthought in
the UI layer) means the monitoring dashboard and the eval scripts share
one source of truth for "what actually happened," instead of the UI
logging one thing and eval scripts assuming another.
"""
from __future__ import annotations

import logging
import time
import uuid

from psycopg.rows import dict_row

from src.config import CONFIG
from src.db import get_conn
from src.ingestion.embed import get_client
from src.rag.prompts import build_prompt
from src.retrieval.hybrid import STRATEGIES

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    conversation_id: str | None = None,
    strategy: str = "hybrid",
    prompt_variant: str = "baseline",
) -> dict:
    """
    Run one full RAG turn and log it. Returns a dict rather than a bare
    string so the caller (UI, eval harness) gets retrieval provenance
    alongside the answer — useful for showing "sources" in the UI and
    required for retrieval eval.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    start = time.monotonic()

    search_fn = STRATEGIES.get(strategy)
    if search_fn is None:
        raise ValueError(f"Unknown retrieval strategy: {strategy!r}. Options: {list(STRATEGIES)}")

    results = search_fn(question, top_k=CONFIG.top_k)
    if not results:
        logger.warning("No documents retrieved for question: %r", question)

    prompt = build_prompt(question, [r.content for r in results], variant=prompt_variant)

    client = get_client()
    response = client.chat.completions.create(
        model=CONFIG.llm.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temp: we want grounded, reproducible-ish answers, not creative ones
    )
    answer = response.choices[0].message.content
    elapsed = time.monotonic() - start

    row_id = _log_conversation(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        retrieved_ids=[r.doc_id for r in results],
        strategy=strategy,
        model=CONFIG.llm.chat_model,
        response_time_s=elapsed,
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
    )

    return {
        "conversation_pk": row_id,  # use this for save_feedback(), not conversation_id
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": [{"id": r.doc_id, "content": r.content, "score": r.score} for r in results],
        "response_time_s": elapsed,
    }


def _log_conversation(**fields) -> int:
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversations
                    (conversation_id, question, answer, retrieved_ids,
                     retrieval_strategy, model, response_time_s,
                     prompt_tokens, completion_tokens)
                VALUES (%(conversation_id)s, %(question)s, %(answer)s, %(retrieved_ids)s,
                        %(strategy)s, %(model)s, %(response_time_s)s,
                        %(prompt_tokens)s, %(completion_tokens)s)
                RETURNING id
                """,
                fields,
            )
            row_id = cur.fetchone()["id"]
        conn.commit()
    return row_id
