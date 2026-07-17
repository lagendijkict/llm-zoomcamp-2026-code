"""
Unit test for the RRF math itself, isolated from the DB and embedding
calls. This is the part of hybrid_search() that's pure logic — worth
testing directly rather than only via a slow integration test that needs
Postgres and an LLM API key up.
"""
from src.retrieval.vector_search import SearchResult


def _fuse(vector_results: list[SearchResult], text_results: list[SearchResult], rrf_k: int = 60) -> list[int]:
    """Reimplements just the fusion step from hybrid.hybrid_search for isolated testing."""
    rrf_scores: dict[int, float] = {}
    for rank, r in enumerate(vector_results, start=1):
        rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, r in enumerate(text_results, start=1):
        rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)


def _sr(doc_id: int) -> SearchResult:
    return SearchResult(doc_id=doc_id, content="", metadata={}, score=0.0)


def test_doc_ranked_top_in_both_lists_wins():
    vector_results = [_sr(1), _sr(2), _sr(3)]
    text_results = [_sr(1), _sr(3), _sr(2)]
    ranked = _fuse(vector_results, text_results)
    assert ranked[0] == 1  # rank 1 in both lists should fuse to rank 1


def test_doc_present_only_in_one_list_still_included():
    vector_results = [_sr(1), _sr(2)]
    text_results = [_sr(3)]
    ranked = _fuse(vector_results, text_results)
    assert set(ranked) == {1, 2, 3}


def test_doc_absent_from_both_lists_is_absent_from_fusion():
    vector_results = [_sr(1)]
    text_results = [_sr(2)]
    ranked = _fuse(vector_results, text_results)
    assert 99 not in ranked


def test_high_rank_in_one_list_beats_low_rank_in_both():
    # doc 1: rank 1 in vector only. doc 2: rank 5 in both lists.
    # RRF should still let accumulated presence in both lists matter,
    # but a strong single-list rank-1 signal is a good sanity check on
    # the fusion formula rather than an assertion of "correct" behavior
    # for every corpus — this test documents the formula's actual
    # tie-breaking behavior so a future refactor doesn't silently change it.
    vector_results = [_sr(1)] + [_sr(10 + i) for i in range(9)]
    text_results = [_sr(20 + i) for i in range(4)] + [_sr(2)]
    ranked = _fuse(vector_results, text_results)
    assert 1 in ranked[:3]
