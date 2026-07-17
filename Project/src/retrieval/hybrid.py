"""
Hybrid search via Reciprocal Rank Fusion (RRF).

Why RRF instead of, say, a weighted average of the two raw scores:
cosine similarity (roughly 0-1) and ts_rank (unbounded, corpus-dependent)
live on completely different scales, so averaging them directly means
whichever score happens to have the bigger numbers dominates — not
whichever result is actually more relevant. RRF sidesteps this entirely
by discarding the raw scores and fusing on *rank position*:

    RRF_score(doc) = sum over each ranked list containing doc of  1 / (k + rank)

A doc that shows up near the top of both lists wins; a doc that's #1 in
one list and absent from the other gets a modest score, not an inflated
one. `k` (default 60, per the original RRF paper) dampens the influence
of rank differences far down the list — the difference between rank 1
and rank 2 matters more than between rank 50 and 51.
"""
from __future__ import annotations

from src.config import CONFIG
from src.retrieval.text_search import text_search
from src.retrieval.vector_search import SearchResult, vector_search


def hybrid_search(query: str, top_k: int = CONFIG.top_k, rrf_k: int = CONFIG.rrf_k) -> list[SearchResult]:
    # Over-fetch from each strategy so fusion has enough candidates to
    # rerank from — if you only fetch top_k from each side, RRF can't
    # promote a doc that was e.g. rank 8 in vector search but rank 1 in
    # text search.
    fetch_k = max(top_k * 4, 20)
    vector_results = vector_search(query, top_k=fetch_k)
    text_results = text_search(query, top_k=fetch_k)

    rrf_scores: dict[int, float] = {}
    doc_lookup: dict[int, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        doc_lookup[result.doc_id] = result

    for rank, result in enumerate(text_results, start=1):
        rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        doc_lookup.setdefault(result.doc_id, result)

    ranked_ids = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)[:top_k]

    return [
        SearchResult(
            doc_id=doc_id,
            content=doc_lookup[doc_id].content,
            metadata=doc_lookup[doc_id].metadata,
            score=rrf_scores[doc_id],
        )
        for doc_id in ranked_ids
    ]


STRATEGIES = {
    "vector": vector_search,
    "text": text_search,
    "hybrid": hybrid_search,
}
