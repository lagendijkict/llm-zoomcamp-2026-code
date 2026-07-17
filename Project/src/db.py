"""
Database layer: connection pooling + idempotent schema management.

Key DE principle applied here: init_db() must be safe to call on every
container start, not just the first one. In a Docker Compose / Codespaces
setup, your app container can restart independently of the Postgres
container's data volume — if init_db() assumes a fresh DB, you'll get
"relation already exists" crashes on the second run. Using
`CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` makes schema
setup a no-op on repeat calls, which is what idempotent actually means here
(not "runs fast" — "produces the same end state no matter how many times
you run it").
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from src.config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Module-level pool, created lazily. A pool (not one connection per call)
# matters once you have concurrent Streamlit sessions or an eval script
# hammering the DB — psycopg3's ConnectionPool handles reuse and
# reconnection for you instead of you hand-rolling retry logic.
_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        logger.info("Opening connection pool to %s:%s/%s", CONFIG.db.host, CONFIG.db.port, CONFIG.db.dbname)
        _pool = ConnectionPool(
            conninfo=CONFIG.db.dsn,
            min_size=1,
            max_size=10,
            # Fail fast on a dead pool rather than hanging a Streamlit request
            timeout=10,
            # Registers the pgvector type adapter on every connection the
            # pool hands out — without this, inserting a Python list into
            # a `vector` column either errors or silently mis-serializes.
            # This is the single most common "works in psql, breaks in the
            # app" bug with pgvector + psycopg3.
            configure=register_vector,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool; always returns it, even on error."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- Source documents after chunking. content_hash makes ingestion idempotent:
-- re-running the pipeline on unchanged source data upserts nothing.
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    source_id       TEXT NOT NULL,           -- stable id from the source system
    content_hash    TEXT NOT NULL UNIQUE,    -- sha256 of chunk text, drives upsert
    content         TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding       vector({CONFIG.llm.embedding_dim}),
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat for approximate nearest-neighbor search. `lists` should scale
-- with row count (~sqrt(n) is the common rule of thumb) — 100 is a
-- reasonable default up to ~100k rows; revisit as your corpus grows or
-- recall will degrade silently (queries still "work", just get worse).
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_tsv
    ON documents USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_documents_source_id
    ON documents (source_id);

-- One row per RAG turn, for monitoring + eval replay.
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    retrieved_ids   BIGINT[] NOT NULL DEFAULT '{{}}',
    retrieval_strategy TEXT NOT NULL DEFAULT 'hybrid',
    model           TEXT NOT NULL,
    response_time_s DOUBLE PRECISION,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);

-- Thumbs up/down + optional free-text, foreign-keyed to the turn it judges.
CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    conversation_pk BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_conversation_pk ON feedback (conversation_pk);
"""


def init_db() -> None:
    """
    Create schema if absent. Safe to call on every app/container startup.

    Raises on genuine connection failure (bad host, DB not up yet) rather
    than swallowing the error — in Codespaces' nested-container networking,
    a silent failure here is exactly what produces the "dashboard loads
    but shows no data" symptom that's hard to debug later.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()
        logger.info("Schema initialized (or already present).")
    except psycopg.OperationalError:
        logger.exception(
            "Could not connect to Postgres at %s:%s — is the container up "
            "and is the hostname resolvable from this container's network?",
            CONFIG.db.host, CONFIG.db.port,
        )
        raise
    except Exception:
        logger.exception("Schema initialization failed")
        raise


if __name__ == "__main__":
    init_db()
