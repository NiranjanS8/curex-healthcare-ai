# Healthcare RAG Assistant

Agentic healthcare RAG assistant for medical literature Q&A, hybrid retrieval, tool use, safety guardrails, faithfulness checks, and RAGAS evaluation.

## Architecture

The target architecture is captured in [rag_healthcare_architecture.svg](rag_healthcare_architecture.svg). The system will be built phase by phase from ingestion through deployment.

## Phase 1.1 Setup

```powershell
python -m uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
python -m uv sync
docker compose up -d
```

Frontend development will live in `frontend/` using Vite, React, and Tailwind.

## Storage

This project uses local PostgreSQL with pgvector for embedding storage. Pinecone is intentionally not used.
