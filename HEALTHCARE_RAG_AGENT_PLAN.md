# Healthcare RAG Assistant — Agent Implementation Plan

> **Agent instructions:** Work through phases sequentially. Complete every task in a sub-phase, run `git add . && git commit -m "phase-X.Y: <task>"` after each sub-phase, then move to the next. Never skip a commit. Use Graphify (or `rich` progress output) to visualise pipeline steps during development. If a step fails, fix it before proceeding.

---

## Project overview

A production-grade, agentic RAG system for healthcare Q&A. Ingests clinical PDFs, PubMed abstracts, drug databases, and medical ontologies. Retrieves with hybrid search + reranking. Reasons via a LangGraph ReAct agent with tool use. Evaluates with RAGAS. Serves via FastAPI + React frontend.

**Core stack:** Python 3.11, LangChain, LangGraph, OpenAI (GPT-4o + text-embedding-3-large), pgvector, scispaCy / BioBERT, RAGAS, LangSmith, FastAPI, React + Vite + Tailwind, Docker, Render.com

---

## Repository structure

```
healthcare-rag/
├── backend/
│   ├── ingestion/
│   │   ├── loaders.py          # PDF + PubMed loaders
│   │   ├── chunker.py          # NER + semantic chunking
│   │   └── indexer.py          # Embed + upsert to vector store
│   ├── retrieval/
│   │   ├── hybrid.py           # Dense + BM25 + RRF
│   │   ├── reranker.py         # Cross-encoder reranker
│   │   └── hyde.py             # HyDE query expansion
│   ├── agent/
│   │   ├── graph.py            # LangGraph StateGraph definition
│   │   ├── router.py           # Intent classifier + query router
│   │   ├── tools.py            # Drug checker, calculator tools
│   │   └── memory.py           # Session + long-term memory
│   ├── generation/
│   │   ├── prompts.py          # System prompts, CoT templates
│   │   ├── safety.py           # Guardrails, scope detection
│   │   └── faithfulness.py     # Hallucination scoring
│   ├── evaluation/
│   │   ├── golden_dataset.json # 50 Q&A evaluation triples
│   │   └── ragas_runner.py     # RAGAS evaluation script
│   ├── api/
│   │   └── main.py             # FastAPI app
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── CitationCard.jsx
│   │   │   ├── MetricsPanel.jsx
│   │   │   └── DisclaimerBanner.jsx
│   │   ├── hooks/
│   │   │   └── useChat.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── render.yaml
├── .github/workflows/deploy.yml
├── .env.example
└── README.md
```

---

## Phase 1 — Project scaffold + data ingestion

**Goal:** Chunks stored in vector DB, queryable via a test script.
**Commit after each sub-phase.**

---

### Phase 1.1 — Repo setup & dependencies

**Files to create:** `pyproject.toml`, `.env.example`, `docker-compose.yml`, `README.md`, `backend/__init__.py`, `frontend/` Vite scaffold

**Tasks:**
- Init git repo, create branch `phase-1`
- Create `pyproject.toml` with dependencies:
  ```toml
  [project]
  name = "healthcare-rag"
  version = "0.1.0"
  requires-python = ">=3.11"
  dependencies = [
    "langchain>=0.2", "langgraph>=0.1", "langchain-openai",
    "pgvector", "psycopg2-binary",
    "scispacy", "rank_bm25", "sentence-transformers",
    "pdfplumber", "biopython", "ragas", "langsmith",
    "fastapi", "uvicorn[standard]", "redis", "pydantic>=2",
    "rich", "graphviz"
  ]
  ```
- Create `.env.example`:
  ```
  OPENAI_API_KEY=
  VECTOR_BACKEND=pgvector
  POSTGRES_URL=postgresql://...
  LANGCHAIN_API_KEY=
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=healthcare-rag
  REDIS_URL=redis://localhost:6379
  ```
- Create `docker-compose.yml` with services: `pgvector` (ankane/pgvector), `redis`
- Scaffold React frontend: `npm create vite@latest frontend -- --template react`, install `tailwindcss`, `@tailwindcss/typography`, `react-markdown`, `lucide-react`
- Print project tree using `rich.tree` (Graphify output for agent visibility)

**Commit:** `git commit -m "phase-1.1: project scaffold, deps, docker-compose, vite frontend"`

---

### Phase 1.2 — PDF + PubMed loader

**File:** `backend/ingestion/loaders.py`

**Tasks:**
- `load_pdf(path: str) -> list[Document]`
  - Use `pdfplumber` to extract text page by page
  - Detect section headings with regex (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
  - Attach metadata: `{source, doc_type: "clinical_pdf", specialty, date, title, page}`
- `load_pubmed(pmids: list[str]) -> list[Document]`
  - Use `Bio.Entrez` to fetch abstracts in XML
  - Parse title, abstract, authors, journal, pubdate
  - Attach metadata: `{source: "pubmed", pmid, title, journal, date, doc_type: "abstract"}`
- `load_all(config: dict) -> list[Document]`
  - Orchestrate both loaders
  - Print progress with `rich.Progress` (Graphify-style pipeline visualisation)
- Write unit test: `tests/test_loaders.py` — mock Entrez, test PDF with a sample fixture

**Commit:** `git commit -m "phase-1.2: pdf and pubmed loaders with metadata"`

---

### Phase 1.3 — Medical NER + semantic chunking

**File:** `backend/ingestion/chunker.py`

**Tasks:**
- Load `en_core_sci_sm` model from scispaCy
- `extract_entities(text: str) -> list[dict]`
  - Run NER, return list of `{text, label}` for DISEASE, CHEMICAL, DOSAGE entities
- `chunk_document(doc: Document) -> list[Document]`
  - Split by detected section headings first
  - Apply sliding window (max 512 tokens, 50-token overlap) within each section
  - Attach entities to each chunk's metadata: `chunk.metadata["entities"] = [...]`
  - Attach `chunk_id` (uuid4), `section`, `char_offset`
- `chunk_all(docs: list[Document]) -> list[Document]`
  - Process all documents, log statistics with `rich.Table` (total chunks, avg length, entity hit rate)
- Visualise chunk distribution with Graphify / `rich` bar chart in terminal

**Commit:** `git commit -m "phase-1.3: scispacy NER and semantic chunking"`

---

### Phase 1.4 — Embed + index to vector store

**File:** `backend/ingestion/indexer.py`

**Tasks:**
- `get_embeddings() -> Embeddings` — return `OpenAIEmbeddings(model="text-embedding-3-large")`
- `get_vector_store(backend: str) -> VectorStore`
  - `"pgvector"` → connect via `POSTGRES_URL`, return `PGVector` store
- `batch_upsert(chunks: list[Document], batch_size=100)`
  - Upsert in batches with retry (tenacity, max 3 retries, exponential backoff)
  - Progress bar via `rich.Progress`
- `run_ingestion_pipeline()`
  - Orchestrate: load → chunk → upsert
  - Draw pipeline graph using `graphviz.Digraph`, save as `pipeline_graph.png`
  - Print final summary table: docs loaded, chunks indexed, time taken, cost estimate

**Commit:** `git commit -m "phase-1.4: embedding, vector store upsert, graphviz pipeline diagram"`

---

## Phase 2 — Hybrid retrieval pipeline

**Goal:** A retriever returning top-5 ranked, cited chunks for any medical query.

---

### Phase 2.1 — Hybrid search (dense + BM25 + RRF)

**File:** `backend/retrieval/hybrid.py`

**Tasks:**
- `class HybridRetriever(BaseRetriever):`
  - `_get_relevant_documents(query: str) -> list[Document]`
  - Step 1: Dense vector search → top-20 results
  - Step 2: BM25 over the same corpus (`rank_bm25.BM25Okapi`) → top-20 results
  - Step 3: Merge with Reciprocal Rank Fusion: `score = Σ 1/(k + rank_i)` where k=60
  - Return top-10 by RRF score
- Attach retrieval scores to `doc.metadata["retrieval_score"]`
- Log retrieval steps with `rich` (Graphify-style) showing score breakdown per result

**Commit:** `git commit -m "phase-2.1: hybrid retriever dense+BM25+RRF"`

---

### Phase 2.2 — Cross-encoder reranker

**File:** `backend/retrieval/reranker.py`

**Tasks:**
- Load `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`
- `rerank(query: str, docs: list[Document], top_k=5) -> list[Document]`
  - Score each `(query, doc.page_content)` pair
  - Sort descending, return top_k
  - Attach `doc.metadata["rerank_score"]`
- Integrate into `HybridRetriever` as a post-step: dense+BM25 → RRF top-20 → rerank → top-5
- Print before/after ranking comparison table with `rich.Table`

**Commit:** `git commit -m "phase-2.2: cross-encoder reranker integrated"`

---

### Phase 2.3 — HyDE query expansion

**File:** `backend/retrieval/hyde.py`

**Tasks:**
- `generate_hypothetical_answer(query: str) -> str`
  - Call GPT-4o-mini: "Write a short paragraph that would appear in a medical textbook answering: {query}"
  - Return the generated paragraph
- `class HyDERetriever(HybridRetriever):`
  - Override: embed the hypothetical answer, use that vector for dense search
  - Accept `hyde: bool = True` flag; fall back to standard retrieval if False
- A/B test helper: run same query both ways, print side-by-side results with `rich.Columns`

**Commit:** `git commit -m "phase-2.3: HyDE query expansion"`

---

## Phase 3 — Agentic layer

**Goal:** A LangGraph ReAct agent that routes queries, calls tools, and reasons multi-step.

---

### Phase 3.1 — Query router + intent classifier

**File:** `backend/agent/router.py`

**Tasks:**
- Define Pydantic model:
  ```python
  class QueryIntent(BaseModel):
      category: Literal["drug_info", "symptom_diagnosis",
                         "clinical_guideline", "drug_interaction",
                         "general_health", "out_of_scope"]
      confidence: float
      entities: list[str]
  ```
- `classify_intent(query: str) -> QueryIntent`
  - Use `GPT-4o-mini` with `with_structured_output(QueryIntent)`
  - System prompt: explain each category with 2 examples
- `route(intent: QueryIntent) -> str` — return node name for LangGraph routing
- Print intent classification result with `rich.Panel`

**Commit:** `git commit -m "phase-3.1: query router and intent classifier"`

---

### Phase 3.2 — LangGraph ReAct agent

**File:** `backend/agent/graph.py`

**Tasks:**
- Define `AgentState(TypedDict)`:
  ```python
  class AgentState(TypedDict):
      messages: list[BaseMessage]
      query: str
      intent: QueryIntent
      retrieved_docs: list[Document]
      tool_results: list[dict]
      response: str
      faithfulness_score: float
      session_id: str
  ```
- Define nodes: `query_router`, `retriever`, `tool_executor`, `response_generator`, `safety_check`, `faithfulness_check`
- Define edges with conditional routing:
  - After `query_router` → route to `retriever` or `tool_executor` based on intent
  - After `faithfulness_check` → if score < 0.7, loop back to `response_generator` (max 2 retries)
- Compile graph: `graph = workflow.compile()`
- Save graph visualisation: `graph.get_graph().draw_mermaid_png()` → `agent_graph.png`
- Print graph structure with `rich` on startup

**Commit:** `git commit -m "phase-3.2: LangGraph ReAct agent with state graph"`

---

### Phase 3.3 — Tools: drug checker + dose calculator

**File:** `backend/agent/tools.py`

**Tasks:**
- `@tool check_drug_interactions(drug_names: list[str]) -> dict`
  - Resolve each drug to RxCUI via `https://rxnav.nlm.nih.gov/REST/rxcui.json?name=`
  - Check interactions via `https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis=`
  - Return: `{pairs: [{drug_a, drug_b, severity, description}]}`
- `@tool calculate_bmi(weight_kg: float, height_cm: float) -> dict`
  - Return: `{bmi, category, healthy_range}`
- `@tool lookup_icd10(condition: str) -> dict`
  - Query `https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search`
  - Return top 3 matching codes + descriptions
- Register all tools in a `TOOLS` list for the agent

**Commit:** `git commit -m "phase-3.3: drug interaction checker and medical tools"`

---

### Phase 3.4 — Memory (session + long-term)

**File:** `backend/agent/memory.py`

**Tasks:**
- Session memory: `ConversationBufferWindowMemory(k=6)` — inject last 6 turns into every prompt
- Long-term memory:
  - After each session, run extraction chain: extract `{age, conditions, medications, allergies}` from conversation
  - Persist to SQLite table `patient_context(session_id, key, value, updated_at)`
  - On session start: load prior context, prepend as system message: "Known patient context: ..."
- `get_memory(session_id: str) -> dict` — returns combined session + long-term context
- Use Redis for session store if `REDIS_URL` is set, else in-memory dict

**Commit:** `git commit -m "phase-3.4: session and long-term memory"`

---

## Phase 4 — Safety, generation & citations

**Goal:** Responses with inline citations, safety filtering, faithfulness ≥ 0.75.

---

### Phase 4.1 — Chain-of-thought prompt + citations

**File:** `backend/generation/prompts.py`

**Tasks:**
- `SYSTEM_PROMPT`:
  ```
  You are a medical information assistant. You provide information based strictly
  on the retrieved medical literature. Never diagnose. Always cite sources.

  Rules:
  1. Reason step by step before giving your final answer.
  2. Every factual claim must be cited as [Source: {title}, chunk {chunk_id}].
  3. If the retrieved context does not contain enough information, say so explicitly.
  4. End every response with: "⚕ This information is for educational purposes only.
     Always consult a qualified healthcare professional."
  5. Never suggest specific dosages without citing a clinical source.
  ```
- `build_prompt(query, context_docs, memory) -> ChatPromptTemplate`
  - Format retrieved docs with chunk IDs for citation
  - Inject memory context
- `format_citations(docs: list[Document]) -> list[dict]`
  - Return `[{chunk_id, title, source_url, excerpt}]` for frontend rendering

**Commit:** `git commit -m "phase-4.1: CoT system prompt and citation formatting"`

---

### Phase 4.2 — Safety guardrails

**File:** `backend/generation/safety.py`

**Tasks:**
- `class SafetyResult(BaseModel): safe: bool; reason: str; modified_query: str`
- `pre_check(query: str) -> SafetyResult`
  - Use GPT-4o-mini to classify: in_scope / off_topic / harmful
  - Off-topic → return safe=False with redirect message
  - Harmful (e.g. drug dosages for self-harm) → return safe=False with crisis resources
- `post_check(response: str) -> str`
  - Detect definitive diagnosis language ("you have", "you are diagnosed")
  - Soften to "this may indicate", "consult a doctor for diagnosis"
  - Ensure disclaimer is present
- Log all flagged queries to `safety_log` table (SQLite)
- Print safety check result with `rich.Panel` (green/red)

**Commit:** `git commit -m "phase-4.2: safety guardrails pre and post check"`

---

### Phase 4.3 — Faithfulness hallucination check

**File:** `backend/generation/faithfulness.py`

**Tasks:**
- `score_faithfulness(response: str, context_docs: list[Document]) -> float`
  - LLM-as-judge: send response + context to GPT-4o-mini
  - Prompt: "Score 0.0–1.0 how well this response is supported by the context. List any unsupported claims."
  - Parse structured output: `{score: float, unsupported_claims: list[str]}`
  - Return score
- In `graph.py` faithfulness node:
  - If score < 0.70 → regenerate with stricter prompt ("only use facts explicitly stated in the context")
  - Max 2 retries; if still < 0.70, prepend a low-confidence warning to response
- Attach score to response metadata for frontend display

**Commit:** `git commit -m "phase-4.3: faithfulness hallucination check with retry"`

---

## Phase 5 — Evaluation harness + observability

**Goal:** RAGAS scores on golden dataset + LangSmith traces — quantifiable proof of quality.

---

### Phase 5.1 — Golden Q&A dataset

**File:** `backend/evaluation/golden_dataset.json`

**Tasks:**
- Create 50 question-answer-context triples:
  ```json
  [
    {
      "id": "drug-001",
      "category": "drug_interaction",
      "difficulty": "medium",
      "question": "What are the risks of combining warfarin and aspirin?",
      "ground_truth_answer": "...",
      "relevant_context": "...",
      "expected_citations": ["...", "..."]
    }
  ]
  ```
- 10 questions per category: `drug_interaction`, `clinical_guideline`, `symptom_info`, `dosage_query`, `contraindication`
- Mix difficulty: 30% easy, 50% medium, 20% hard
- Generate using GPT-4o with a careful prompt, then manually review each one

**Commit:** `git commit -m "phase-5.1: 50-question golden evaluation dataset"`

---

### Phase 5.2 — RAGAS evaluation runner

**File:** `backend/evaluation/ragas_runner.py`

**Tasks:**
- `run_evaluation(dataset_path: str) -> pd.DataFrame`
  - Load golden dataset
  - For each question: run full RAG pipeline → collect `{answer, contexts}`
  - Evaluate with RAGAS metrics:
    - `faithfulness` — is the answer grounded in context?
    - `answer_relevancy` — does the answer address the question?
    - `context_precision` — are retrieved chunks relevant?
    - `context_recall` — were all relevant chunks retrieved?
  - Return DataFrame with per-question scores
- Print results table with `rich.Table`, colour-code by score threshold (green ≥ 0.75, amber ≥ 0.5, red < 0.5)
- Save to `eval_results.csv`
- Print aggregate summary: mean scores per category and overall

**Commit:** `git commit -m "phase-5.2: RAGAS evaluation runner with score reporting"`

---

### Phase 5.3 — LangSmith tracing + cost tracking

**File:** `backend/agent/graph.py` (additions)

**Tasks:**
- Enable tracing: set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=healthcare-rag` in env
- Add custom metadata to every run:
  ```python
  config = {"metadata": {"query_category": intent.category,
                          "session_id": state["session_id"],
                          "faithfulness_score": state["faithfulness_score"]}}
  ```
- `CostTracker` class:
  - Callback that captures `prompt_tokens`, `completion_tokens` per run
  - Calculates USD cost using current OpenAI pricing (gpt-4o: $5/$15 per 1M tokens)
  - Persists to `query_log` SQLite table: `(session_id, query, cost_usd, latency_ms, faithfulness, timestamp)`
- `get_run_summary(n: int = 20) -> dict` — return avg latency, avg cost, avg faithfulness over last N runs

**Commit:** `git commit -m "phase-5.3: LangSmith tracing and cost tracking"`

---

## Phase 6 — React frontend + API + deployment

**Goal:** Live demo URL you can link from your resume.

---

### Phase 6.1 — FastAPI backend

**File:** `backend/api/main.py`

**Tasks:**
- `POST /chat` — accepts `{session_id, message}`, returns `StreamingResponse` (SSE)
  - Stream tokens as `data: {token}\n\n`
  - On completion, send `data: [DONE]\n\n` with citations JSON
- `GET /session/{session_id}/history` — return conversation history
- `POST /session/new` — return new `session_id` (uuid4)
- `GET /eval/metrics` — return latest RAGAS scores from `eval_results.csv`
- `GET /health` — return `{status: "ok", version}`
- CORS: allow `http://localhost:5173` (Vite dev) and production domain
- Add request logging middleware with `rich`

**Commit:** `git commit -m "phase-6.1: FastAPI backend with SSE streaming"`

---

### Phase 6.2 — React frontend

**Directory:** `frontend/src/`

**Tasks:**

**`App.jsx`**
- Layout: left sidebar (session list + metrics panel) + main chat area
- Manage `sessionId` state, create new session on mount via `POST /session/new`
- Fetch eval metrics from `GET /eval/metrics` on load

**`components/ChatWindow.jsx`**
- Render list of messages
- Auto-scroll to bottom on new message
- Input bar with send button (Enter to send)
- Show typing indicator while streaming

**`components/MessageBubble.jsx`**
- User messages: right-aligned, teal background
- Assistant messages: left-aligned, white card
- Render markdown with `react-markdown`
- Show `CitationCard` components inline below assistant messages

**`components/CitationCard.jsx`**
- Collapsed by default: shows source title + doc type badge
- Expand to show chunk excerpt + metadata (date, specialty)
- Faithfulness score badge (colour-coded)

**`components/MetricsPanel.jsx`**
- Show 4 metric cards: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Pull from `GET /eval/metrics`
- Colour-code: green ≥ 0.75, amber ≥ 0.50, red < 0.50

**`components/DisclaimerBanner.jsx`**
- Sticky top banner: "⚕ For informational purposes only. Not a substitute for professional medical advice."
- Dismissible, re-shown on new session

**`hooks/useChat.js`**
- `useChat(sessionId)` hook
- Manages `messages`, `isStreaming` state
- `sendMessage(text)`: POST to `/chat`, consume SSE stream, append tokens to last message, parse citations on `[DONE]`

**Styling:** Tailwind. Clean white/gray UI, medical-grade aesthetic. No dark mode needed for MVP.

**Commit:** `git commit -m "phase-6.2: React frontend with streaming chat and citation cards"`

---

### Phase 6.3 — Docker + CI/CD + deploy

**Files:** `Dockerfile.backend`, `Dockerfile.frontend`, `render.yaml`, `.github/workflows/deploy.yml`

**Tasks:**

**`Dockerfile.backend`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install --no-cache-dir -e .
RUN python -m spacy download en_core_sci_sm
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`Dockerfile.frontend`:**
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/ .
RUN npm ci && npm run build
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

**`render.yaml`** — define:
- Web service: backend (Docker), env vars from Render secrets
- Static site: frontend (from `dist/` folder)
- Redis instance (free tier)

**`.github/workflows/deploy.yml`:**
```yaml
on: [push to main]
jobs:
  eval:
    - Run RAGAS evaluation against staging
    - Fail deploy if faithfulness < 0.75
  deploy:
    - needs: eval
    - Deploy to Render via API trigger
```

**Commit:** `git commit -m "phase-6.3: Dockerfile, render.yaml, CI/CD with eval gate"`

---

## Commit log convention

```
phase-1.1: project scaffold, deps, docker-compose, vite frontend
phase-1.2: pdf and pubmed loaders with metadata
phase-1.3: scispacy NER and semantic chunking
phase-1.4: embedding, vector store upsert, graphviz pipeline diagram
phase-2.1: hybrid retriever dense+BM25+RRF
phase-2.2: cross-encoder reranker integrated
phase-2.3: HyDE query expansion
phase-3.1: query router and intent classifier
phase-3.2: LangGraph ReAct agent with state graph
phase-3.3: drug interaction checker and medical tools
phase-3.4: session and long-term memory
phase-4.1: CoT system prompt and citation formatting
phase-4.2: safety guardrails pre and post check
phase-4.3: faithfulness hallucination check with retry
phase-5.1: 50-question golden evaluation dataset
phase-5.2: RAGAS evaluation runner with score reporting
phase-5.3: LangSmith tracing and cost tracking
phase-6.1: FastAPI backend with SSE streaming
phase-6.2: React frontend with streaming chat and citation cards
phase-6.3: Dockerfile, render.yaml, CI/CD with eval gate
```

---

## Graphify / visualisation checkpoints

Use these at the agent checkpoints listed below. Use `graphviz.Digraph` for pipeline graphs and `rich` for terminal output.

| Phase | What to visualise |
|-------|-------------------|
| 1.4   | Full ingestion pipeline DAG → `pipeline_graph.png` |
| 2.1   | Retrieval score breakdown table per query |
| 3.2   | LangGraph agent state graph → `agent_graph.png` (mermaid PNG) |
| 5.2   | RAGAS scores table, colour-coded by threshold |
| 6.1   | Startup: print API route map with `rich.Table` |

---

## Definition of done

- [ ] All 20 sub-phase commits present in git log
- [ ] `pipeline_graph.png` and `agent_graph.png` generated
- [ ] RAGAS faithfulness ≥ 0.75 on golden dataset
- [ ] CI/CD pipeline blocks deploy if faithfulness < 0.75
- [ ] React frontend streams responses with inline citations
- [ ] Live URL deployed to Render.com
- [ ] `README.md` has architecture diagram, setup instructions, and RAGAS scores table
