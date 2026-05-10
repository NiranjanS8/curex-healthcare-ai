# Healthcare RAG Assistant

Agentic healthcare RAG assistant for medical literature Q&A, hybrid retrieval, tool use, safety guardrails, faithfulness checks, and RAGAS evaluation.

## Architecture

The system is built as an end-to-end healthcare RAG application spanning ingestion, retrieval, agent orchestration, safety checks, evaluation, and a React interface.

## Setup

```powershell
python -m uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
python -m uv sync
docker compose up -d
```

Frontend development will live in `frontend/` using Vite, React, and Tailwind.

## Storage

This project uses local PostgreSQL with pgvector for embedding storage. Pinecone is intentionally not used.
