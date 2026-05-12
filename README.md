# CureX Healthcare RAG Assistant

CureX is an agentic healthcare RAG application for asking evidence-grounded questions over medical documents. It combines document ingestion, local pgvector retrieval, cited responses, safety checks, JWT authentication, user-scoped memory, human review, and an evaluation dashboard.

The assistant is for educational and research workflows only. It is not a substitute for professional medical advice, diagnosis, or treatment.

## Core Features

- Healthcare-focused chat assistant with streamed responses
- PDF, TXT, and Markdown document upload and indexing
- Local PostgreSQL + pgvector vector storage
- Gemini-based generation and embeddings
- User login with JWT authentication
- User-scoped chat sessions and memory isolation
- Source citations for retrieved document chunks
- Medical safety guardrails and faithfulness scoring
- Optional visible agent workflow toggle
- Human-in-the-loop answer review: helpful, unsupported, unsafe, needs review
- Evaluation dashboard for RAG metrics, citation coverage, and feedback signals
- MCP tool server for drug interaction lookup, ICD-10 lookup, BMI calculation, and retrieval search

## Architecture

```text
Frontend React app
        |
        v
FastAPI backend
        |
        +-- Auth and user-scoped sessions
        +-- Document upload and ingestion
        +-- Chunking and metadata enrichment
        +-- pgvector retrieval
        +-- Agent routing, safety checks, and response generation
        +-- Feedback and evaluation APIs
```

## Tech Stack

- Backend: FastAPI, LangGraph, LangChain, Pydantic
- Frontend: React, Vite, lucide-react
- Vector DB: PostgreSQL with pgvector
- Models: Gemini generation and Gemini embeddings
- Storage: SQLite for local auth, memory, feedback, safety logs, and query logs
- Evaluation: RAGAS-style metrics and faithfulness checks
- Tooling: uv, pytest, Docker Compose

## Setup

Create the Python environment with uv:

```powershell
python -m uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
python -m uv sync
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Create a local `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Set at least:

```text
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_google_api_key
POSTGRES_URL=postgresql://healthcare:healthcare@localhost:5432/healthcare_rag
VECTOR_BACKEND=pgvector
```

Start local infrastructure:

```powershell
docker compose up -d
```

## Run

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload
```

Start the frontend:

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

## Usage Flow

1. Create an account or sign in.
2. Start a new healthcare conversation.
3. Upload PDF, TXT, or Markdown medical documents.
4. Ask questions about the uploaded content or clinical research topics.
5. Inspect citations and optional agent workflow details.
6. Review answers as helpful, unsupported, unsafe, or needs review.
7. Open the evaluation dashboard to monitor quality signals.

## API Highlights

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /session/new`
- `GET /session/{session_id}/history`
- `POST /documents/upload`
- `POST /chat`
- `POST /feedback`
- `GET /feedback/summary`
- `GET /eval/metrics`

## Testing

Run the backend test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## MCP Tool Server

Run the local healthcare tool server:

```powershell
.\.venv\Scripts\healthcare-mcp-server.exe
```

Available tools include drug interaction lookup, ICD-10 lookup, BMI calculation, and retrieval search over indexed healthcare chunks.

