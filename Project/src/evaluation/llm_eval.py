"""
LLM-as-judge evaluation of final answers, comparing prompt variants.

Distinct failure mode from retrieval_eval.py: retrieval can be perfect
(the right chunk is in context) while the generation step still produces
a bad answer (ignores context, hallucinates beyond it, or is unhelpfully
terse). This eval isolates that generation step by holding retrieval
fixed and varying only the prompt template.

Judge pattern: ask a (typically stronger/cheaper-to-run-many-times) model
to score each answer on a fixed rubric, returned as strict JSON so scores
are parseable rather than free text you'd have to regex out — same reason
you'd ask an upstream API for structured JSON instead of scraping HTML.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src.config import CONFIG
from src.ingestion.embed import get_client
from src.rag.prompts import PROMPT_VARIANTS, build_prompt
from src.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating an AI-generated answer for quality.

QUESTION: {question}
CONTEXT PROVIDED TO THE MODEL: {context}
GENERATED ANSWER: {answer}

Score the answer from 1-5 on each dimension:
- relevance: does it address the question asked?
- faithfulness: is it fully supported by the context (no hallucination)?
- completeness: does it use the relevant information available in the context?

Respond with ONLY valid JSON: {{"relevance": <int>, "faithfulness": <int>, "completeness": <int>}}"""


@dataclass
class JudgeScore:
    relevance: int
    faithfulness: int
    completeness: int

    @property
    def mean(self) -> float:
        return (self.relevance + self.faithfulness + self.completeness) / 3


def _judge(question: str, context: str, answer: str) -> JudgeScore:
    client = get_client()
    resp = client.chat.completions.create(
        model=CONFIG.llm.chat_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, context=context, answer=answer)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return JudgeScore(**data)
    except (json.JSONDecodeError, TypeError) as e:
        # A judge that returns malformed JSON should fail loud, not
        # silently contribute a zero score that quietly skews the average.
        raise RuntimeError(f"Judge returned unparseable output: {resp.choices[0].message.content}") from e


def evaluate_prompt_variants(eval_questions: list[str]) -> dict[str, dict]:
    """
    For each question: retrieve once (shared across variants, so we're
    isolating the prompt's effect, not retrieval noise), generate an
    answer per prompt variant, judge each.
    """
    client = get_client()
    results: dict[str, list[JudgeScore]] = {v: [] for v in PROMPT_VARIANTS}

    for question in eval_questions:
        retrieved = hybrid_search(question, top_k=CONFIG.top_k)
        context_chunks = [r.content for r in retrieved]
        context_joined = "\n\n".join(context_chunks)

        for variant in PROMPT_VARIANTS:
            prompt = build_prompt(question, context_chunks, variant=variant)
            resp = client.chat.completions.create(
                model=CONFIG.llm.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = resp.choices[0].message.content
            score = _judge(question, context_joined, answer)
            results[variant].append(score)
            logger.info("variant=%s question=%r mean_score=%.2f", variant, question[:50], score.mean)

    summary = {
        variant: {
            "mean_relevance": sum(s.relevance for s in scores) / len(scores),
            "mean_faithfulness": sum(s.faithfulness for s in scores) / len(scores),
            "mean_completeness": sum(s.completeness for s in scores) / len(scores),
            "mean_overall": sum(s.mean for s in scores) / len(scores),
        }
        for variant, scores in results.items()
        if scores
    }
    best = max(summary, key=lambda v: summary[v]["mean_overall"])
    logger.info("Best prompt variant: %s", best)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Replace with real eval questions representative of your domain.
    sample_questions = ["Example question one?", "Example question two?"]
    print(json.dumps(evaluate_prompt_variants(sample_questions), indent=2))
