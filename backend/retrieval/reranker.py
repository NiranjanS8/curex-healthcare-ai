"""Cross-encoder reranking for retrieved documents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderLike(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score per query/document pair."""


class CrossEncoderReranker:
    """Rerank candidate chunks with a sentence-transformers cross encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: CrossEncoderLike | None = None,
        log_results: bool = True,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self.log_results = log_results

    @property
    def model(self) -> CrossEncoderLike:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """Score query/document pairs and return the top_k documents."""

        if not docs:
            return []
        if top_k <= 0:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = [float(score) for score in self.model.predict(pairs)]
        scored_docs = sorted(zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True)

        reranked: list[Document] = []
        for rank, (doc, score) in enumerate(scored_docs[:top_k], start=1):
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = score
            metadata["rerank_rank"] = rank
            reranked.append(Document(page_content=doc.page_content, metadata=metadata))

        self._log_comparison(docs, reranked)
        return reranked

    def _log_comparison(self, before: list[Document], after: list[Document]) -> None:
        if not self.log_results:
            return

        table = Table(title="Reranker Before/After")
        table.add_column("New Rank", justify="right")
        table.add_column("Chunk")
        table.add_column("Original Rank", justify="right")
        table.add_column("Score", justify="right")

        original_ranks = {
            str(doc.metadata.get("chunk_id") or index): index for index, doc in enumerate(before, start=1)
        }

        for rank, doc in enumerate(after, start=1):
            chunk = str(doc.metadata.get("chunk_id") or "unknown")
            table.add_row(
                str(rank),
                chunk[:40],
                str(original_ranks.get(chunk, "-")),
                f"{doc.metadata['rerank_score']:.4f}",
            )

        Console().print(table)


_DEFAULT_RERANKER: CrossEncoderReranker | None = None


def get_default_reranker() -> CrossEncoderReranker:
    global _DEFAULT_RERANKER
    if _DEFAULT_RERANKER is None:
        _DEFAULT_RERANKER = CrossEncoderReranker()
    return _DEFAULT_RERANKER


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Convenience function using the default cross-encoder reranker."""

    return get_default_reranker().rerank(query, docs, top_k=top_k)
