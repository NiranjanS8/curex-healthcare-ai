"""Faithfulness scoring for generated healthcare responses."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, Field


FAITHFULNESS_SYSTEM_PROMPT = """You judge whether a healthcare assistant response is supported by retrieved context.

Score from 0.0 to 1.0:
- 1.0 means every factual claim is explicitly supported by the context.
- 0.7 means mostly supported with minor missing details.
- 0.0 means unsupported, contradictory, or no relevant context.

Return a score and list unsupported claims. Do not judge medical correctness beyond the provided context."""


class FaithfulnessResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class FaithfulnessJudge(Protocol):
    def invoke(self, messages):
        """Return structured faithfulness scoring output."""


def _format_context(context_docs: list[Document]) -> str:
    if not context_docs:
        return "No context provided."
    blocks: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or f"source {index}"
        chunk_id = doc.metadata.get("chunk_id", f"chunk-{index}")
        blocks.append(f"[{index}] {title} | chunk {chunk_id}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _coerce_result(result: Any) -> FaithfulnessResult:
    if isinstance(result, FaithfulnessResult):
        return result
    if isinstance(result, dict):
        return FaithfulnessResult.model_validate(result)
    raise TypeError(f"Unsupported faithfulness judge result: {type(result)!r}")


def get_faithfulness_judge():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(
        FaithfulnessResult
    )


def _token_set(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "source",
        "the",
        "this",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _fallback_score(response: str, context_docs: list[Document]) -> FaithfulnessResult:
    if not context_docs:
        if "not enough" in response.lower() or "do not have enough" in response.lower():
            return FaithfulnessResult(score=1.0, unsupported_claims=[])
        return FaithfulnessResult(score=0.0, unsupported_claims=["No retrieved context was provided."])

    context_text = "\n".join(doc.page_content for doc in context_docs)
    response_tokens = _token_set(response)
    context_tokens = _token_set(context_text)
    if not response_tokens:
        return FaithfulnessResult(score=0.0, unsupported_claims=["Response is empty."])

    overlap = len(response_tokens & context_tokens) / len(response_tokens)
    has_citation = "[source:" in response.lower()
    score = min(1.0, overlap + (0.3 if has_citation else 0.0))
    unsupported = [] if score >= 0.7 else ["Response contains claims not strongly supported by context."]
    return FaithfulnessResult(score=round(score, 2), unsupported_claims=unsupported)


def score_faithfulness_result(
    response: str,
    context_docs: list[Document],
    *,
    judge: FaithfulnessJudge | None = None,
) -> FaithfulnessResult:
    """Score whether a response is supported by retrieved context."""

    if judge is not None:
        return _coerce_result(
            judge.invoke(
                [
                    ("system", FAITHFULNESS_SYSTEM_PROMPT),
                    ("human", f"Response:\n{response}\n\nContext:\n{_format_context(context_docs)}"),
                ]
            )
        )

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            return _coerce_result(
                get_faithfulness_judge().invoke(
                    [
                        ("system", FAITHFULNESS_SYSTEM_PROMPT),
                        ("human", f"Response:\n{response}\n\nContext:\n{_format_context(context_docs)}"),
                    ]
                )
            )
        except Exception:
            pass

    return _fallback_score(response, context_docs)


def score_faithfulness(
    response: str,
    context_docs: list[Document],
    *,
    judge: FaithfulnessJudge | None = None,
) -> float:
    """Return a 0.0-1.0 support score for a response against context."""

    return score_faithfulness_result(response, context_docs, judge=judge).score
