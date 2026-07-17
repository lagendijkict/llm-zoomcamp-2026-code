"""
Central configuration, loaded once from environment variables.

Why this pattern instead of `os.environ.get(...)` scattered across modules:
- Single place to see every config knob the app depends on
- Fails fast at import time if something required is missing, instead of
  failing deep inside a pipeline run at 2am
- Type-safe: downstream code gets an int/str it can trust, not a raw string
  that might be "5432" (str) when a driver expects 5432 (int)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Check your .env file against .env.example."
        )
    return val


@dataclass(frozen=True)
class DBConfig:
    host: str = os.environ.get("POSTGRES_HOST", "localhost")
    port: int = int(os.environ.get("POSTGRES_PORT", "5432"))
    dbname: str = os.environ.get("POSTGRES_DB", "rag_db")
    user: str = os.environ.get("POSTGRES_USER", "rag_user")
    password: str = os.environ.get("POSTGRES_PASSWORD", "rag_password")

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


@dataclass(frozen=True)
class LLMConfig:
    # Swap providers by changing these two — pipeline code doesn't care.
    provider: str = os.environ.get("LLM_PROVIDER", "openai")  # openai | ollama
    chat_model: str = os.environ.get("LLM_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.environ.get("LLM_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = int(os.environ.get("LLM_EMBEDDING_DIM", "1536"))
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    base_url: str = os.environ.get("LLM_BASE_URL", "")  # set for Ollama, e.g. http://localhost:11434/v1


@dataclass(frozen=True)
class AppConfig:
    db: DBConfig = DBConfig()
    llm: LLMConfig = LLMConfig()
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
    rrf_k: int = int(os.environ.get("RRF_K", "60"))  # standard RRF constant, see hybrid.py


CONFIG = AppConfig()
