"""Pure vector (embedding cosine similarity) search."""
from __future__ import annotations

from dataclasses import dataclass

from psycopg.rows import dict_row

from src.config import CONFIG
from src.db import get_conn
from src.ingestion.embed import embed_texts


@dataclass
class SearchResult:
    doc_id: int
    content: str
    metadata: dict
    score: float  # higher is better, normalized per-strategy for comparability


def vector_search(query: str, top_k: int = CONFIG.top_k) -> list[SearchResult]:
    """
    Embed the query and find nearest neighbors by cosine distance.
    `<=>` is pgvector's cosine distance operator; we return 1 - distance
    so score is "higher is better" and consistent with text_search below.
    """
    query_vector = embed_texts([query])[0]

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, content, metadata, 1 - (embedding <=> %s) AS score
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()

    return [
        SearchResult(doc_id=r["id"], content=r["content"], metadata=r["metadata"], score=r["score"])
        for r in rows
    ]
