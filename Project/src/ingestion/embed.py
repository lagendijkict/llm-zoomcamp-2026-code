"""
Embedding client. Batches requests — one HTTP round-trip per N chunks
instead of per chunk, which matters once your corpus is more than a
few hundred chunks (rate limits and latency both scale with call count,
not token count, for most providers).
"""
from __future__ import annotations

import logging

from openai import OpenAI

from src.config import CONFIG

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        kwargs = {"api_key": CONFIG.llm.api_key or "not-needed-for-local"}
        if CONFIG.llm.base_url:  # e.g. Ollama's OpenAI-compatible endpoint
            kwargs["base_url"] = CONFIG.llm.base_url
        _client = OpenAI(**kwargs)
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of strings, batching to respect provider request limits.
    Order of returned vectors matches order of input texts.
    """
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=CONFIG.llm.embedding_model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        logger.info("Embedded batch %d-%d of %d", i, i + len(batch), len(texts))
    return vectors
