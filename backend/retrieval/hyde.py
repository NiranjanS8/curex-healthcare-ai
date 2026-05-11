"""HyDE query expansion for hybrid retrieval."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from backend.ingestion.indexer import get_embeddings
from backend.retrieval.hybrid import HybridRetriever


HYDE_PROMPT_TEMPLATE = (
    "Write a short paragraph that would appear in a medical textbook answering: {query}\n\n"
    "Keep it factual, concise, and focused on clinically relevant terminology."
)


class ChatModelLike(Protocol):
    def invoke(self, prompt: str) -> Any:
        """Return a chat model response for the given prompt."""


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


def generate_hypothetical_answer(query: str, *, llm: ChatModelLike | None = None) -> str:
    """Generate a short textbook-like answer for HyDE retrieval."""

    model = llm
    if model is None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    answer = _response_text(model.invoke(prompt))
    if not answer:
        raise ValueError("HyDE generation returned an empty answer.")
    return answer


class HyDERetriever(HybridRetriever):
    """Hybrid retriever that uses a generated hypothetical answer for dense search."""

    hyde: bool = True
    embeddings: Embeddings | None = None
    hyde_llm: Any | None = None

    def _dense_search(self, query: str) -> list[tuple[Document, float | None]]:
        if not self.hyde:
            return super()._dense_search(query)

        hypothetical_answer = generate_hypothetical_answer(query, llm=self.hyde_llm)
        embedding_model = self.embeddings or get_embeddings()
        vector = embedding_model.embed_query(hypothetical_answer)

        if hasattr(self.vector_store, "similarity_search_with_score_by_vector"):
            results = self.vector_store.similarity_search_with_score_by_vector(vector, k=self.dense_k)
        else:
            docs = self.vector_store.similarity_search_by_vector(vector, k=self.dense_k)
            results = [(doc, None) for doc in docs]

        hyde_results: list[tuple[Document, float | None]] = []
        for doc, score in results:
            metadata = dict(doc.metadata)
            metadata["hyde"] = True
            metadata["hyde_answer"] = hypothetical_answer
            hyde_results.append((Document(page_content=doc.page_content, metadata=metadata), score))

        return hyde_results


def _result_panel(title: str, docs: list[Document]) -> Panel:
    lines: list[str] = []
    for index, doc in enumerate(docs, start=1):
        chunk = doc.metadata.get("chunk_id") or doc.metadata.get("source") or "unknown"
        score = doc.metadata.get("retrieval_score")
        score_text = "-" if score is None else f"{float(score):.4f}"
        lines.append(f"{index}. {chunk} | {score_text}")
    return Panel("\n".join(lines) or "No results", title=title)


def run_hyde_ab_test(
    query: str,
    *,
    standard_retriever: HybridRetriever,
    hyde_retriever: HyDERetriever,
) -> dict[str, list[Document]]:
    """Run the same query with standard retrieval and HyDE, then print side-by-side results."""

    standard_docs = standard_retriever.invoke(query)
    hyde_docs = hyde_retriever.invoke(query)
    Console().print(
        Columns(
            [
                _result_panel("Standard retrieval", standard_docs),
                _result_panel("HyDE retrieval", hyde_docs),
            ],
            equal=True,
        )
    )
    return {"standard": standard_docs, "hyde": hyde_docs}
