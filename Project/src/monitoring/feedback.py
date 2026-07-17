"""Save thumbs up/down feedback tied to a specific conversation turn."""
from __future__ import annotations

from psycopg.rows import dict_row

from src.db import get_conn


def save_feedback(conversation_pk: int, rating: int, comment: str | None = None) -> None:
    if rating not in (-1, 1):
        raise ValueError(f"rating must be -1 or 1, got {rating}")

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_pk, rating, comment) VALUES (%s, %s, %s)",
                (conversation_pk, rating, comment),
            )
        conn.commit()
