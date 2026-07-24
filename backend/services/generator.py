"""
backend/services/generator.py
-------------------------------
Builds a strict, context-only prompt from retrieved chunks and calls Groq
to generate the final answer.

Hallucination-prevention control points:
  1. Similarity-threshold filtering (retriever.py) happens before this file
     ever runs.
  2. Zero-chunk short-circuit: if no chunks survive retrieval, the LLM is
     never called at all — no context in, no made-up answer out.
  3. Strict system prompt restricts the model to the supplied context, with
     a fixed, exact fallback sentence as its "I don't know" escape hatch.
  4. temperature = 0 (deterministic, non-creative generation).

The answer text and its supporting sources are returned as SEPARATE fields
(GeneratedAnswer.answer vs GeneratedAnswer.sources) rather than being
concatenated into one blob of text — this is what the API returns as JSON
and what the frontend renders in clearly separated sections, so the answer
and its citations never visually run together.
"""

from dataclasses import dataclass

from groq import Groq

from backend.config import settings
from backend.services.retriever import RetrievedChunk

NO_ANSWER_MESSAGE = "The answer is not available in the provided document."

_SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the user's question using ONLY the CONTEXT provided below.

Rules:
- Do not use any outside knowledge, even if you are confident it is correct.
- Do not guess, infer beyond what the text states, or fill gaps with assumptions.
- If the context does not contain enough information to answer, respond with
  exactly this sentence and nothing else: "The answer is not available in the provided document."
- When you do answer, be concise and factual, and stay strictly within what the context supports.
- Do not mention "the context" or "the provided text" in your answer; just answer naturally,
  as if you had read the document yourself.
"""


@dataclass
class GeneratedAnswer:
    answer: str
    used_context: bool
    confidence: float
    sources: list[RetrievedChunk]


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i} | {c.document_name}, page {c.page_number}]\n{c.text}")
    return "\n\n".join(parts)


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    avg_score = sum(c.score for c in chunks) / len(chunks)
    coverage_penalty = min(len(chunks) / 3, 1.0)
    return round(avg_score * coverage_penalty, 3)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    """Generate an answer strictly from `chunks`. Returns the refusal message if chunks is empty."""
    if not chunks:
        return GeneratedAnswer(answer=NO_ANSWER_MESSAGE, used_context=False, confidence=0.0, sources=[])

    context_block = _build_context_block(chunks)
    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    client = Groq(api_key=settings.groq_api_key)
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}") from e

    used_context = NO_ANSWER_MESSAGE not in answer_text
    confidence = _compute_confidence(chunks) if used_context else 0.0

    return GeneratedAnswer(
        answer=answer_text,
        used_context=used_context,
        confidence=confidence,
        sources=chunks if used_context else [],
    )
