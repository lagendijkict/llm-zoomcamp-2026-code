"""
Prompt templates. Kept as a dict of named variants, not one hardcoded
f-string, specifically so llm_eval.py can compare them — the rubric wants
multiple prompt approaches evaluated, not just multiple retrieval
strategies.
"""
from __future__ import annotations

PROMPT_VARIANTS: dict[str, str] = {
    "baseline": """You are a helpful assistant. Answer the QUESTION using only the CONTEXT below.
If the CONTEXT doesn't contain the answer, say you don't know — do not make things up.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
    "strict_citation": """Answer the QUESTION using ONLY the CONTEXT below. Every claim in your
answer must be traceable to the CONTEXT. If the CONTEXT is insufficient, respond exactly with:
"I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
    "concise": """Using the CONTEXT, answer the QUESTION in 2-3 sentences maximum. Be direct.
If the CONTEXT doesn't cover it, say so in one sentence.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
}


def build_prompt(question: str, context_chunks: list[str], variant: str = "baseline") -> str:
    context = "\n\n---\n\n".join(context_chunks)
    template = PROMPT_VARIANTS[variant]
    return template.format(context=context, question=question)
