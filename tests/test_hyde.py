from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.retrieval.hyde import HyDERetriever, generate_hypothetical_answer, run_hyde_ab_test
from backend.retrieval.hybrid import HybridRetriever
from tests.test_hybrid import FakeVectorStore


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.response


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text))]


class FakeHydeVectorStore(FakeVectorStore):
    def __init__(self, docs: list[Document]) -> None:
        super().__init__(docs)
        self.vector_queries: list[list[float]] = []
        self.text_queries: list[str] = []

    def similarity_search_with_score(self, query: str, k: int = 20):
        self.text_queries.append(query)
        return super().similarity_search_with_score(query, k=k)

    def similarity_search_with_score_by_vector(self, embedding: list[float], k: int = 20):
        self.vector_queries.append(embedding)
        return [(doc, 0.9 / (index + 1)) for index, doc in enumerate(self.docs[:k])]


def test_generate_hypothetical_answer_uses_medical_prompt() -> None:
    llm = FakeLlm("Aspirin may increase bleeding risk when combined with warfarin.")

    answer = generate_hypothetical_answer("warfarin aspirin interaction", llm=llm)

    assert "bleeding risk" in answer
    assert "medical textbook" in llm.prompts[0]
    assert "warfarin aspirin interaction" in llm.prompts[0]


def test_hyde_retriever_uses_hypothetical_answer_embedding() -> None:
    docs = [
        Document(page_content="anticoagulant bleeding risk", metadata={"chunk_id": "a"}),
        Document(page_content="aspirin precautions", metadata={"chunk_id": "b"}),
    ]
    store = FakeHydeVectorStore(docs)
    embeddings = FakeEmbeddings()
    retriever = HyDERetriever(
        vector_store=store,
        corpus=docs,
        dense_k=2,
        bm25_k=2,
        fusion_k=2,
        top_k=2,
        embeddings=embeddings,
        hyde_llm=FakeLlm("Textbook answer about anticoagulant bleeding risk."),
        log_results=False,
    )

    results = retriever.invoke("warfarin aspirin bleeding")

    assert store.vector_queries
    assert not store.text_queries
    assert embeddings.queries == ["Textbook answer about anticoagulant bleeding risk."]
    assert all(doc.metadata["hyde"] is True for doc in results)
    assert all("hyde_answer" in doc.metadata for doc in results)


def test_hyde_retriever_can_fallback_to_standard_dense_search() -> None:
    docs = [Document(page_content="standard dense result", metadata={"chunk_id": "standard"})]
    store = FakeHydeVectorStore(docs)
    retriever = HyDERetriever(
        vector_store=store,
        corpus=[],
        dense_k=1,
        fusion_k=1,
        top_k=1,
        hyde=False,
        log_results=False,
    )

    results = retriever.invoke("plain query")

    assert store.text_queries == ["plain query"]
    assert not store.vector_queries
    assert results[0].metadata["chunk_id"] == "standard"


def test_run_hyde_ab_test_returns_both_result_sets(capsys) -> None:
    standard_doc = Document(page_content="standard", metadata={"chunk_id": "standard"})
    hyde_doc = Document(page_content="hyde", metadata={"chunk_id": "hyde"})
    standard = HybridRetriever(
        vector_store=FakeVectorStore([standard_doc]),
        corpus=[],
        dense_k=1,
        fusion_k=1,
        top_k=1,
        log_results=False,
    )
    hyde = HyDERetriever(
        vector_store=FakeHydeVectorStore([hyde_doc]),
        corpus=[],
        dense_k=1,
        fusion_k=1,
        top_k=1,
        embeddings=FakeEmbeddings(),
        hyde_llm=FakeLlm("hypothetical answer"),
        log_results=False,
    )

    results = run_hyde_ab_test("query", standard_retriever=standard, hyde_retriever=hyde)

    assert results["standard"][0].metadata["chunk_id"] == "standard"
    assert results["hyde"][0].metadata["chunk_id"] == "hyde"
    assert "Standard retrieval" in capsys.readouterr().out
