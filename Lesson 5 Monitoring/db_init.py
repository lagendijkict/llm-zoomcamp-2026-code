import logging
import os
import psycopg
from contextlib import closing
from datetime import datetime
from psycopg import OperationalError

DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")

logger = logging.getLogger(__name__)

CREATE_CONVERSATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        course TEXT NOT NULL,
        model TEXT NOT NULL,
        instructions TEXT NOT NULL,
        prompt TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        response_time FLOAT NOT NULL,
        cost FLOAT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
"""

# Separate DDL statements, run individually — see note below on why
# these aren't jammed into the CREATE TABLE call.
CREATE_CONVERSATIONS_TIMESTAMP_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
    ON conversations (timestamp DESC)
"""

CREATE_CONVERSATIONS_COURSE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_conversations_course
    ON conversations (course)
"""

def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )

def init_db(drop: bool = False) -> None:
    """Create the conversations table (and its indexes) if they don't exist.

    Idempotent and safe to call on every app startup or Streamlit rerun.

    Args:
        drop: Destructive escape hatch for local schema iteration.
              Never call with drop=True against data you care about —
              there is no confirmation step here on purpose, since this
              is meant to be called programmatically, not interactively.

    Raises:
        OperationalError: if the database is unreachable. Left uncaught
            deliberately — a dashboard that starts against a dead DB
            should fail loudly at init, not proceed and fail confusingly
            later on the first query.
    """
    try:
        # `closing` guarantees conn.close() runs even if an exception
        # escapes the `with` block below — plain try/finally would too,
        # but this reads more clearly when nested with the cursor's own
        # context manager, and avoids a bare `finally: conn.close()`
        # shadowing the real exception if close() itself raises.
        with closing(get_db_connection()) as conn:
            with conn.cursor() as cur:
                if drop:
                    logger.warning("Dropping conversations table (drop=True)")
                    cur.execute("DROP TABLE IF EXISTS conversations")

                cur.execute(CREATE_CONVERSATIONS_TABLE)
                cur.execute(CREATE_CONVERSATIONS_TIMESTAMP_INDEX)
                cur.execute(CREATE_CONVERSATIONS_COURSE_INDEX)

            conn.commit()
            logger.info("conversations table ready")

    except OperationalError:
        logger.exception("Could not reach Postgres during init_db()")
        raise

def init_feedback():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")

            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    #init_db()
    init_feedback()
    print("Database initialized")