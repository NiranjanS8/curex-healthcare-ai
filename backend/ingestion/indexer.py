"""Embed document chunks and index them into local pgvector."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.ingestion.chunker import chunk_all
from backend.ingestion.loaders import load_all, load_uploaded_file


DEFAULT_COLLECTION_NAME = "healthcare_rag_chunks"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
EMBEDDING_COST_PER_1K_TOKENS_USD = 0.00013


def get_embeddings() -> Embeddings:
    """Return the Gemini embedding model used by the ingestion pipeline."""

    return GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        output_dimensionality=DEFAULT_EMBEDDING_DIMENSIONS,
    )


def get_vector_store(
    backend: str | None = None,
    *,
    embeddings: Embeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    connection_string: str | None = None,
) -> VectorStore:
    """Create a pgvector-backed LangChain vector store."""

    selected_backend = (backend or os.getenv("VECTOR_BACKEND", "pgvector")).lower()
    if selected_backend != "pgvector":
        raise ValueError("Only the local pgvector backend is supported for this project.")

    postgres_url = connection_string or os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL must be set when using pgvector.")

    return PGVector(
        connection_string=postgres_url,
        embedding_function=embeddings or get_embeddings(),
        collection_name=collection_name,
        embedding_length=DEFAULT_EMBEDDING_DIMENSIONS,
        use_jsonb=True,
    )


def _batches(items: Sequence[Document], batch_size: int) -> list[Sequence[Document]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _add_batch(vector_store: VectorStore, batch: Sequence[Document]) -> list[str]:
    return vector_store.add_documents(list(batch))


def batch_upsert(
    chunks: list[Document],
    *,
    vector_store: VectorStore | None = None,
    backend: str | None = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Upsert chunk documents in retrying batches."""

    store = vector_store or get_vector_store(backend)
    inserted_ids: list[str] = []
    batches = _batches(chunks, batch_size)
    console = Console()

    with Progress(console=console) as progress:
        task = progress.add_task("Indexing chunks into pgvector", total=len(batches))
        for batch in batches:
            inserted_ids.extend(_add_batch(store, batch))
            progress.advance(task)

    return {
        "chunks_indexed": len(chunks),
        "batches": len(batches),
        "ids": inserted_ids,
    }


def draw_pipeline_graph(output_path: str | Path = "pipeline_graph.png") -> Path:
    """Draw the ingestion DAG, falling back to a simple PNG if Graphviz is unavailable."""

    output = Path(output_path)
    graph = Digraph("healthcare_rag_ingestion", format="png")
    graph.attr(rankdir="LR", label="Healthcare RAG Ingestion Pipeline", labelloc="t")
    graph.node("sources", "PDFs + PubMed", shape="folder")
    graph.node("load", "Load Documents", shape="box")
    graph.node("chunk", "NER + Semantic Chunking", shape="box")
    graph.node("embed", "Gemini Embeddings", shape="box")
    graph.node("pgvector", "PostgreSQL + pgvector", shape="cylinder")
    graph.edges([("sources", "load"), ("load", "chunk"), ("chunk", "embed"), ("embed", "pgvector")])

    try:
        rendered = Path(graph.render(filename=output.with_suffix("").as_posix(), cleanup=True))
        if rendered != output:
            rendered.replace(output)
    except ExecutableNotFound:
        _write_fallback_png(output)

    return output


def _write_fallback_png(output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    nodes = [
        ("PDFs + PubMed", 40),
        ("Load Documents", 275),
        ("NER + Chunking", 510),
        ("Embeddings", 745),
        ("pgvector", 980),
    ]
    y = 150
    for label, x in nodes:
        draw.rounded_rectangle((x, y, x + 170, y + 70), radius=14, outline="#0f766e", width=3)
        draw.text((x + 24, y + 27), label, fill="#111827", font=font)
    for (_, x1), (_, x2) in zip(nodes, nodes[1:]):
        draw.line((x1 + 170, y + 35, x2, y + 35), fill="#0f766e", width=3)
        draw.polygon([(x2 - 10, y + 28), (x2, y + 35), (x2 - 10, y + 42)], fill="#0f766e")
    draw.text((40, 40), "Healthcare RAG Ingestion Pipeline", fill="#111827", font=font)
    image.save(output)


def _estimate_embedding_cost(chunks: list[Document]) -> float:
    token_estimate = sum(max(1, len(chunk.page_content.split())) for chunk in chunks)
    return (token_estimate / 1000) * EMBEDDING_COST_PER_1K_TOKENS_USD


def _print_summary(
    *,
    docs_loaded: int,
    chunks_indexed: int,
    elapsed_seconds: float,
    estimated_cost_usd: float,
) -> None:
    table = Table(title="Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Documents loaded", str(docs_loaded))
    table.add_row("Chunks indexed", str(chunks_indexed))
    table.add_row("Time taken", f"{elapsed_seconds:.2f}s")
    table.add_row("Embedding cost estimate", f"${estimated_cost_usd:.4f}")
    Console().print(table)


def run_ingestion_pipeline(
    config: dict[str, Any] | None = None,
    *,
    vector_store: VectorStore | None = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Load sources, chunk documents, index chunks, and print a final summary."""

    started_at = time.perf_counter()
    pipeline_config = config or {}

    draw_pipeline_graph()
    docs = load_all(pipeline_config)
    chunks = chunk_all(docs)
    upsert_result = batch_upsert(chunks, vector_store=vector_store, batch_size=batch_size)

    elapsed = time.perf_counter() - started_at
    estimated_cost = _estimate_embedding_cost(chunks)
    _print_summary(
        docs_loaded=len(docs),
        chunks_indexed=upsert_result["chunks_indexed"],
        elapsed_seconds=elapsed,
        estimated_cost_usd=estimated_cost,
    )

    return {
        "docs_loaded": len(docs),
        "chunks_indexed": upsert_result["chunks_indexed"],
        "batches": upsert_result["batches"],
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": estimated_cost,
    }


def index_uploaded_document(
    path: str | Path,
    *,
    owner_user_id: str,
    vector_store: VectorStore | None = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Chunk and index one uploaded document into pgvector with owner metadata."""

    started_at = time.perf_counter()
    upload_path = Path(path)
    docs = load_uploaded_file(str(upload_path))
    chunks = chunk_all(docs)
    for chunk in chunks:
        chunk.metadata["owner_user_id"] = owner_user_id
        chunk.metadata["uploaded_filename"] = upload_path.name

    upsert_result = batch_upsert(chunks, vector_store=vector_store, batch_size=batch_size)
    elapsed = time.perf_counter() - started_at
    estimated_cost = _estimate_embedding_cost(chunks)

    return {
        "filename": upload_path.name,
        "docs_loaded": len(docs),
        "chunks_indexed": upsert_result["chunks_indexed"],
        "batches": upsert_result["batches"],
        "ids": upsert_result["ids"],
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": estimated_cost,
    }
