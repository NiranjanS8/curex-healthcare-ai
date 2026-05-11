"""Hybrid dense + BM25 retrieval with reciprocal rank fusion."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, PrivateAttr
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from backend.retrieval.reranker import CrossEncoderReranker


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 using a stable lowercase word tokenizer."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def document_key(doc: Document) -> str:
    """Return a stable key for merging dense and BM25 results."""

    metadata = doc.metadata or {}
    for key in ("chunk_id", "id"):
        if metadata.get(key):
            return str(metadata[key])
    source_parts = [
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        str(metadata.get("section", "")),
        doc.page_content,
    ]
    return hashlib.sha256("::".join(source_parts).encode("utf-8")).hexdigest()


class HybridRetriever(BaseRetriever):
    """Retrieve chunks with dense vector search, BM25, and RRF rank fusion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: VectorStore
    corpus: list[Document] = Field(default_factory=list)
    dense_k: int = 20
    bm25_k: int = 20
    fusion_k: int = 20
    top_k: int = 5
    rrf_k: int = 60
    log_results: bool = True
    reranker: CrossEncoderReranker | None = None

    _bm25: BM25Okapi | None = PrivateAttr(default=None)
    _tokenized_corpus: list[list[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._build_bm25()

    def _build_bm25(self) -> None:
        self._tokenized_corpus = [tokenize(doc.page_content) for doc in self.corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def _dense_search(self, query: str) -> list[tuple[Document, float | None]]:
        if hasattr(self.vector_store, "similarity_search_with_score"):
            try:
                return list(self.vector_store.similarity_search_with_score(query, k=self.dense_k))
            except NotImplementedError:
                pass

        return [(doc, None) for doc in self.vector_store.similarity_search(query, k=self.dense_k)]

    def _bm25_search(self, query: str) -> list[tuple[Document, float]]:
        if not self._bm25 or not self.corpus:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[tuple[Document, float]] = []
        for index in ranked_indices[: self.bm25_k]:
            results.append((self.corpus[index], float(scores[index])))
        return results

    def _fuse(
        self,
        dense_results: list[tuple[Document, float | None]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[Document]:
        merged: dict[str, dict[str, Any]] = {}

        for rank, (doc, score) in enumerate(dense_results, start=1):
            key = document_key(doc)
            record = merged.setdefault(
                key,
                {"doc": doc, "rrf_score": 0.0, "dense_rank": None, "bm25_rank": None},
            )
            record["rrf_score"] += 1 / (self.rrf_k + rank)
            record["dense_rank"] = rank
            if score is not None:
                record["dense_score"] = float(score)

        for rank, (doc, score) in enumerate(bm25_results, start=1):
            key = document_key(doc)
            record = merged.setdefault(
                key,
                {"doc": doc, "rrf_score": 0.0, "dense_rank": None, "bm25_rank": None},
            )
            record["rrf_score"] += 1 / (self.rrf_k + rank)
            record["bm25_rank"] = rank
            record["bm25_score"] = float(score)

        fused_records = sorted(
            merged.values(),
            key=lambda record: (record["rrf_score"], -(record.get("dense_rank") or 10_000)),
            reverse=True,
        )

        fused_docs: list[Document] = []
        for record in fused_records[: self.fusion_k]:
            doc = record["doc"]
            metadata = dict(doc.metadata)
            metadata["retrieval_score"] = record["rrf_score"]
            metadata["dense_rank"] = record.get("dense_rank")
            metadata["bm25_rank"] = record.get("bm25_rank")
            if "dense_score" in record:
                metadata["dense_score"] = record["dense_score"]
            if "bm25_score" in record:
                metadata["bm25_score"] = record["bm25_score"]
            fused_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        return fused_docs

    def _log_score_breakdown(self, query: str, docs: list[Document]) -> None:
        if not self.log_results:
            return

        table = Table(title=f"Hybrid Retrieval: {query}")
        table.add_column("Rank", justify="right")
        table.add_column("Chunk")
        table.add_column("Dense", justify="right")
        table.add_column("BM25", justify="right")
        table.add_column("RRF", justify="right")

        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            table.add_row(
                str(rank),
                str(metadata.get("chunk_id") or metadata.get("source") or "unknown")[:40],
                "-" if metadata.get("dense_rank") is None else str(metadata["dense_rank"]),
                "-" if metadata.get("bm25_rank") is None else str(metadata["bm25_rank"]),
                f"{metadata['retrieval_score']:.4f}",
            )

        Console().print(table)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)
        docs = self._fuse(dense_results, bm25_results)
        self._log_score_breakdown(query, docs)
        if self.reranker is not None:
            return self.reranker.rerank(query, docs, top_k=self.top_k)
        return docs[: self.top_k]
