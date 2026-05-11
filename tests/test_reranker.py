from __future__ import annotations

from langchain_core.documents import Document

from backend.retrieval.reranker import CrossEncoderReranker, rerank


class FakeCrossEncoder:
    def predict(self, pairs):
        return [len(document_text) for _, document_text in pairs]


def test_reranker_sorts_by_cross_encoder_score() -> None:
    docs = [
        Document(page_content="short", metadata={"chunk_id": "short"}),
        Document(page_content="a much longer and more relevant chunk", metadata={"chunk_id": "long"}),
        Document(page_content="medium length", metadata={"chunk_id": "medium"}),
    ]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(), log_results=False)

    reranked = reranker.rerank("query", docs, top_k=2)

    assert [doc.metadata["chunk_id"] for doc in reranked] == ["long", "medium"]
    assert reranked[0].metadata["rerank_rank"] == 1
    assert reranked[0].metadata["rerank_score"] > reranked[1].metadata["rerank_score"]


def test_reranker_handles_empty_docs() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(), log_results=False)

    assert reranker.rerank("query", [], top_k=5) == []


def test_module_rerank_uses_default_reranker(monkeypatch) -> None:
    docs = [Document(page_content="content", metadata={"chunk_id": "chunk"})]
    fake_default = CrossEncoderReranker(model=FakeCrossEncoder(), log_results=False)
    monkeypatch.setattr("backend.retrieval.reranker.get_default_reranker", lambda: fake_default)

    result = rerank("query", docs, top_k=1)

    assert result[0].metadata["chunk_id"] == "chunk"
    assert "rerank_score" in result[0].metadata
