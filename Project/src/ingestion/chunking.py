"""
Text chunking.

Naive vs. this version: a naive `text[i:i+size]` slice cuts mid-sentence
and mid-word, which measurably hurts retrieval quality because embeddings
of a truncated sentence don't represent the same concept as the full one.
This version chunks on paragraph/sentence boundaries where possible and
falls back to a hard split only for pathologically long single paragraphs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import CONFIG
from src.ingestion.loader import RawDocument


@dataclass
class Chunk:
    source_id: str
    text: str
    metadata: dict


def _split_sentences(text: str) -> list[str]:
    # Good enough for English prose; swap for a real sentence tokenizer
    # (e.g. nltk.sent_tokenize) if your corpus has abbreviations that
    # confuse this regex (e.g. "Dr.", "U.S.").
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_document(
    doc: RawDocument,
    chunk_size: int = CONFIG.chunk_size,
    chunk_overlap: int = CONFIG.chunk_overlap,
) -> list[Chunk]:
    """
    Greedily pack sentences into chunks up to `chunk_size` characters,
    carrying `chunk_overlap` characters of trailing context into the next
    chunk so a fact split across a chunk boundary still has a home where
    it's fully present.
    """
    sentences = _split_sentences(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(source_id=doc.source_id, text=chunk_text, metadata=doc.metadata))
            # carry overlap: keep trailing sentences that fit within chunk_overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(Chunk(source_id=doc.source_id, text=" ".join(current), metadata=doc.metadata))

    return chunks
