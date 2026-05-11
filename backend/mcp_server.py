"""MCP server exposing healthcare assistant tools."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from mcp.server.fastmcp import FastMCP

from backend.agent.graph import get_retriever
from backend.agent.tools import calculate_bmi, check_drug_interactions, lookup_icd10
from backend.ingestion.indexer import get_vector_store


DEFAULT_RETRIEVAL_TOP_K = 5


def drug_interaction_lookup(drug_names: list[str]) -> dict[str, Any]:
    """Check RxNav interactions for two or more drug names."""

    return check_drug_interactions.invoke({"drug_names": drug_names})


def icd10_lookup(condition: str) -> dict[str, Any]:
    """Look up ICD-10-CM codes for a condition."""

    return lookup_icd10.invoke({"condition": condition})


def bmi_calculator(weight_kg: float, height_cm: float) -> dict[str, Any]:
    """Calculate adult BMI and the healthy weight range for a height."""

    return calculate_bmi.invoke({"weight_kg": weight_kg, "height_cm": height_cm})


def _serialize_document(doc: Document) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    return {
        "content": doc.page_content,
        "metadata": metadata,
        "chunk_id": metadata.get("chunk_id"),
        "title": metadata.get("title") or metadata.get("source"),
        "source": metadata.get("source") or metadata.get("source_url"),
        "score": metadata.get("rerank_score") or metadata.get("retrieval_score") or metadata.get("dense_score"),
    }


def _search_with_retriever(retriever: BaseRetriever, query: str, top_k: int) -> list[Document]:
    docs = retriever.invoke(query)
    return list(docs)[:top_k]


def _search_with_vector_store(vector_store: VectorStore, query: str, top_k: int) -> list[Document]:
    if hasattr(vector_store, "similarity_search_with_score"):
        try:
            return [
                Document(page_content=doc.page_content, metadata={**dict(doc.metadata), "dense_score": score})
                for doc, score in vector_store.similarity_search_with_score(query, k=top_k)
            ]
        except NotImplementedError:
            pass
    return list(vector_store.similarity_search(query, k=top_k))


def retrieval_search(
    query: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    *,
    retriever: BaseRetriever | None = None,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Search indexed healthcare chunks through the configured retriever or pgvector."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    active_retriever = retriever or get_retriever()
    if active_retriever is not None:
        docs = _search_with_retriever(active_retriever, query, top_k)
        backend = "configured_retriever"
    else:
        store = vector_store or get_vector_store()
        docs = _search_with_vector_store(store, query, top_k)
        backend = "pgvector"

    return {
        "query": query,
        "top_k": top_k,
        "backend": backend,
        "results": [_serialize_document(doc) for doc in docs],
    }


def create_mcp_server() -> FastMCP:
    """Create the MCP server with all healthcare tools registered."""

    mcp = FastMCP(
        "healthcare-rag-tools",
        instructions=(
            "Healthcare RAG Assistant tool server. Tools provide educational medical lookup, "
            "retrieval search, and calculators. They do not provide diagnosis or emergency triage."
        ),
    )

    @mcp.tool(name="drug_interaction_lookup")
    def _drug_interaction_lookup(drug_names: list[str]) -> dict[str, Any]:
        """Check RxNav interactions for two or more drug names."""

        return drug_interaction_lookup(drug_names)

    @mcp.tool(name="icd10_lookup")
    def _icd10_lookup(condition: str) -> dict[str, Any]:
        """Look up ICD-10-CM diagnosis codes for a condition."""

        return icd10_lookup(condition)

    @mcp.tool(name="bmi_calculator")
    def _bmi_calculator(weight_kg: float, height_cm: float) -> dict[str, Any]:
        """Calculate adult BMI and the healthy weight range for a height."""

        return bmi_calculator(weight_kg, height_cm)

    @mcp.tool(name="retrieval_search")
    def _retrieval_search(query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> dict[str, Any]:
        """Search indexed healthcare chunks through the configured retriever or pgvector."""

        return retrieval_search(query, top_k)

    return mcp


def main() -> None:
    """Run the MCP server over stdio."""

    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
