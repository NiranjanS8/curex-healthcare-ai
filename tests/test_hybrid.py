from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from backend.retrieval.hybrid import HybridRetriever, document_key, tokenize
from backend.retrieval.reranker import CrossEncoderReranker


class FakeVectorStore(VectorStore):
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        docs = [
            Document(page_content=text, metadata=(metadatas or [{} for _ in texts])[index])
            for index, text in enumerate(texts)
        ]
        return cls(docs)

    def add_texts(self, texts, metadatas=None, **kwargs):
        docs = [
            Document(page_content=text, metadata=(metadatas or [{} for _ in texts])[index])
            for index, text in enumerate(texts)
        ]
        self.docs.extend(docs)
        return [doc.metadata.get("chunk_id", str(index)) for index, doc in enumerate(docs)]

    def similarity_search_with_score(self, query: str, k: int = 20):
        return [(doc, 1.0 / (index + 1)) for index, doc in enumerate(self.docs[:k])]

    def similarity_search(self, query: str, k: int = 20):
        return self.docs[:k]


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for _, document_text in pairs:
            if "aspirin bleeding precautions" in document_text:
                scores.append(5.0)
            elif "warfarin and aspirin" in document_text:
                scores.append(4.0)
            else:
                scores.append(1.0)
        return scores


def test_tokenize_lowercases_words() -> None:
    assert tokenize("Warfarin + aspirin: bleeding-risk!") == [
        "warfarin",
        "aspirin",
        "bleeding",
        "risk",
    ]


def test_document_key_prefers_chunk_id() -> None:
    doc = Document(page_content="content", metadata={"chunk_id": "abc"})

    assert document_key(doc) == "abc"


def test_hybrid_retriever_fuses_dense_and_bm25_results() -> None:
    dense_doc = Document(
        page_content="general anticoagulant overview",
        metadata={"chunk_id": "dense-only"},
    )
    shared_doc = Document(
        page_content="warfarin and aspirin increase bleeding risk",
        metadata={"chunk_id": "shared"},
    )
    bm25_doc = Document(
        page_content="aspirin bleeding precautions",
        metadata={"chunk_id": "bm25-only"},
    )
    corpus = [shared_doc, bm25_doc]
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([dense_doc, shared_doc]),
        corpus=corpus,
        dense_k=2,
        bm25_k=2,
        fusion_k=3,
        top_k=3,
        log_results=False,
    )

    docs = retriever.invoke("warfarin aspirin bleeding")

    assert [doc.metadata["chunk_id"] for doc in docs] == ["shared", "dense-only", "bm25-only"]
    shared = docs[0]
    assert shared.metadata["dense_rank"] == 2
    assert shared.metadata["bm25_rank"] == 1
    assert shared.metadata["retrieval_score"] > docs[1].metadata["retrieval_score"]


def test_hybrid_retriever_handles_empty_corpus() -> None:
    dense_doc = Document(page_content="dense result", metadata={"chunk_id": "dense"})
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([dense_doc]),
        corpus=[],
        dense_k=1,
        fusion_k=1,
        top_k=1,
        log_results=False,
    )

    docs = retriever.invoke("query")

    assert len(docs) == 1
    assert docs[0].metadata["chunk_id"] == "dense"
    assert docs[0].metadata["bm25_rank"] is None


def test_hybrid_retriever_integrates_reranker() -> None:
    dense_doc = Document(page_content="general overview", metadata={"chunk_id": "dense-only"})
    shared_doc = Document(
        page_content="warfarin and aspirin increase bleeding risk",
        metadata={"chunk_id": "shared"},
    )
    bm25_doc = Document(page_content="aspirin bleeding precautions", metadata={"chunk_id": "bm25-only"})
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([dense_doc, shared_doc]),
        corpus=[shared_doc, bm25_doc],
        dense_k=2,
        bm25_k=2,
        fusion_k=3,
        top_k=2,
        reranker=CrossEncoderReranker(model=FakeCrossEncoder(), log_results=False),
        log_results=False,
    )

    docs = retriever.invoke("warfarin aspirin bleeding")

    assert [doc.metadata["chunk_id"] for doc in docs] == ["bm25-only", "shared"]
    assert all("rerank_score" in doc.metadata for doc in docs)
