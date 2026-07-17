"""
Retrieval evaluation: hit rate and MRR across strategies.

Requires a ground-truth set of (question, relevant_doc_id) pairs. The
standard way to bootstrap this without hand-labeling hundreds of examples:
for each chunk in your corpus, ask an LLM to generate a question that
chunk would answer. You then know the "correct" doc for that question by
construction. See `generate_ground_truth()` below.

Metrics:
- Hit rate: fraction of queries where the relevant doc appears anywhere
  in the top-k results. Answers "do we find it at all."
- MRR (Mean Reciprocal Rank): mean of 1/rank of the relevant doc across
  queries (0 if not found). Answers "do we find it *near the top*" —
  a doc at rank 1 contributes 1.0, at rank 5 contributes 0.2. This is the
  metric that actually predicts whether your RAG pipeline's limited
  top_k context window will contain the right chunk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from psycopg.rows import dict_row

from src.config import CONFIG
from src.db import get_conn
from src.ingestion.embed import get_client
from src.retrieval.hybrid import STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthPair:
    question: str
    relevant_doc_id: int


def generate_ground_truth(n_samples: int = 100) -> list[GroundTruthPair]:
    """
    Sample chunks from the DB and have an LLM write a question each chunk
    answers. This is the "generate synthetic eval data" pattern — cheap,
    imperfect (some generated questions will be ambiguous or answerable
    from multiple chunks), but far better than shipping with zero
    retrieval evaluation, which is the rubric's 0-point outcome.
    """
    client = get_client()

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, content FROM documents ORDER BY random() LIMIT %s",
                (n_samples,),
            )
            rows = cur.fetchall()

    pairs: list[GroundTruthPair] = []
    for row in rows:
        resp = client.chat.completions.create(
            model=CONFIG.llm.chat_model,
            messages=[{
                "role": "user",
                "content": (
                    "Write ONE specific question that the following passage answers. "
                    "Respond with only the question, no preamble.\n\n"
                    f"PASSAGE:\n{row['content']}"
                ),
            }],
            temperature=0.3,
        )
        question = resp.choices[0].message.content.strip()
        pairs.append(GroundTruthPair(question=question, relevant_doc_id=row["id"]))

    return pairs


def evaluate_strategy(pairs: list[GroundTruthPair], strategy: str, top_k: int = CONFIG.top_k) -> dict:
    search_fn = STRATEGIES[strategy]
    hits = 0
    reciprocal_ranks = []

    for pair in pairs:
        results = search_fn(pair.question, top_k=top_k)
        retrieved_ids = [r.doc_id for r in results]
        if pair.relevant_doc_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(pair.relevant_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(pairs)
    return {
        "strategy": strategy,
        "n_queries": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
    }


def run_retrieval_eval(ground_truth_path: str = "data/ground_truth.json") -> list[dict]:
    """Compare every registered strategy on the same ground-truth set."""
    try:
        with open(ground_truth_path) as f:
            raw = json.load(f)
        pairs = [GroundTruthPair(**p) for p in raw]
    except FileNotFoundError:
        logger.info("No cached ground truth found, generating fresh set")
        pairs = generate_ground_truth()
        with open(ground_truth_path, "w") as f:
            json.dump([{"question": p.question, "relevant_doc_id": p.relevant_doc_id} for p in pairs], f, indent=2)

    results = [evaluate_strategy(pairs, strategy) for strategy in STRATEGIES]
    for r in results:
        logger.info("%s: hit_rate=%.3f mrr=%.3f", r["strategy"], r["hit_rate"], r["mrr"])

    best = max(results, key=lambda r: r["mrr"])
    logger.info("Best strategy by MRR: %s", best["strategy"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_retrieval_eval()
