from __future__ import annotations

from langchain_core.documents import Document

from backend.generation.faithfulness import (
    FAITHFULNESS_SYSTEM_PROMPT,
    FaithfulnessResult,
    score_faithfulness,
    score_faithfulness_result,
)


class FakeJudge:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_score_faithfulness_uses_structured_judge() -> None:
    judge = FakeJudge({"score": 0.82, "unsupported_claims": ["minor detail"]})
    docs = [Document(page_content="Aspirin may increase bleeding risk.", metadata={"chunk_id": "c1"})]

    result = score_faithfulness_result("Aspirin may increase bleeding risk.", docs, judge=judge)

    assert result == FaithfulnessResult(score=0.82, unsupported_claims=["minor detail"])
    assert judge.messages[0] == ("system", FAITHFULNESS_SYSTEM_PROMPT)
    assert "Response:" in judge.messages[1][1]
    assert "Context:" in judge.messages[1][1]


def test_score_faithfulness_returns_float() -> None:
    judge = FakeJudge({"score": 0.91, "unsupported_claims": []})

    assert score_faithfulness("answer", [], judge=judge) == 0.91


def test_fallback_score_rewards_context_overlap_and_citations(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    docs = [
        Document(
            page_content="Warfarin and aspirin may increase bleeding risk.",
            metadata={"title": "Drug Safety", "chunk_id": "drug-1"},
        )
    ]

    score = score_faithfulness(
        "Warfarin and aspirin may increase bleeding risk. [Source: Drug Safety, chunk drug-1]",
        docs,
    )

    assert score >= 0.7


def test_fallback_scores_unsupported_response_low(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    docs = [Document(page_content="Metformin lowers blood glucose.", metadata={})]

    result = score_faithfulness_result("Aspirin prevents all strokes.", docs)

    assert result.score < 0.7
    assert result.unsupported_claims


def test_no_context_is_faithful_when_response_admits_insufficient_context(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert score_faithfulness("I do not have enough retrieved context to answer.", []) == 1.0
