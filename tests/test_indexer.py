from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.ingestion import indexer


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[list[Document]] = []

    def add_documents(self, documents: list[Document], **kwargs):
        self.calls.append(documents)
        return [doc.metadata.get("chunk_id", str(index)) for index, doc in enumerate(documents)]


def test_get_vector_store_rejects_non_pgvector() -> None:
    with pytest.raises(ValueError, match="Only the local pgvector"):
        indexer.get_vector_store("unsupported")


def test_get_vector_store_requires_postgres_url(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_URL", raising=False)

    with pytest.raises(ValueError, match="POSTGRES_URL"):
        indexer.get_vector_store("pgvector")


def test_batch_upsert_batches_documents() -> None:
    chunks = [
        Document(page_content=f"chunk {number}", metadata={"chunk_id": f"chunk-{number}"})
        for number in range(5)
    ]
    store = FakeVectorStore()

    result = indexer.batch_upsert(chunks, vector_store=store, batch_size=2)

    assert result["chunks_indexed"] == 5
    assert result["batches"] == 3
    assert result["ids"] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert [len(call) for call in store.calls] == [2, 2, 1]


def test_draw_pipeline_graph_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "pipeline_graph.png"

    result = indexer.draw_pipeline_graph(output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_run_ingestion_pipeline_orchestrates(monkeypatch, tmp_path: Path) -> None:
    docs = [Document(page_content="Metformin evidence.", metadata={"source": "fixture"})]
    chunks = [Document(page_content="Metformin evidence.", metadata={"chunk_id": "chunk-1"})]
    store = FakeVectorStore()

    monkeypatch.setattr(indexer, "draw_pipeline_graph", lambda output_path="pipeline_graph.png": tmp_path / "pipeline_graph.png")
    monkeypatch.setattr(indexer, "load_all", lambda config: docs)
    monkeypatch.setattr(indexer, "chunk_all", lambda loaded_docs: chunks)

    result = indexer.run_ingestion_pipeline({"pdf_paths": []}, vector_store=store)

    assert result["docs_loaded"] == 1
    assert result["chunks_indexed"] == 1
    assert result["batches"] == 1
    assert result["estimated_cost_usd"] > 0
    assert store.calls == [chunks]
