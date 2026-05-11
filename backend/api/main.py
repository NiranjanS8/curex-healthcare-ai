"""FastAPI backend for the healthcare RAG assistant."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from backend import __version__
from backend.agent.graph import AgentState, CostTracker, invoke_agent
from backend.agent.memory import (
    append_session_messages,
    get_memory,
    load_session_messages,
    persist_conversation_context,
)
from backend.generation.prompts import format_citations


console = Console()
DEFAULT_EVAL_RESULTS_PATH = Path("eval_results.csv")
AgentRunner = Callable[[AgentState], AgentState]
ContextPersister = Callable[[str, list[BaseMessage]], Any]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class NewSessionResponse(BaseModel):
    session_id: str


def create_app(
    *,
    agent_runner: AgentRunner | None = None,
    context_persister: ContextPersister | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print_route_map(app)
        yield

    app = FastAPI(title="Healthcare RAG Assistant API", version=__version__, lifespan=lifespan)
    app.state.agent_runner = agent_runner or _run_agent_with_tracking
    app.state.context_persister = context_persister or persist_conversation_context

    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *(origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "").split(",") if origin.strip()),
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        console.log(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({elapsed_ms:.1f} ms)"
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/session/new", response_model=NewSessionResponse)
    async def new_session() -> NewSessionResponse:
        return NewSessionResponse(session_id=str(uuid4()))

    @app.get("/session/{session_id}/history")
    async def session_history(session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": [_serialize_message(message) for message in load_session_messages(session_id)],
        }

    @app.get("/eval/metrics")
    async def eval_metrics() -> dict[str, Any]:
        return load_latest_eval_metrics()

    @app.post("/chat")
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _chat_event_stream(
                payload,
                request.app.state.agent_runner,
                request.app.state.context_persister,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _run_agent_with_tracking(state: AgentState) -> AgentState:
    tracker = CostTracker(session_id=state["session_id"], query=state["query"])
    result = invoke_agent(state, callbacks=[tracker])
    tracker.finish(faithfulness=result.get("faithfulness_score"))
    return result


async def _chat_event_stream(
    payload: ChatRequest,
    agent_runner: AgentRunner,
    context_persister: ContextPersister,
) -> AsyncIterator[str]:
    user_message = HumanMessage(content=payload.message)
    memory = get_memory(payload.session_id)
    state: AgentState = {
        "query": payload.message,
        "session_id": payload.session_id,
        "messages": [*memory.get("session_messages", []), user_message],
    }
    if memory.get("system_message") is not None:
        state["messages"].insert(0, memory["system_message"])

    result = await asyncio.to_thread(agent_runner, state)
    response_text = str(result.get("response") or "")
    retrieved_docs = result.get("retrieved_docs", [])
    citations = format_citations(retrieved_docs)
    faithfulness_score = result.get("faithfulness_score")

    for token in _stream_response_tokens(response_text):
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0)

    done_payload = {
        "citations": citations,
        "faithfulness_score": faithfulness_score,
        "session_id": payload.session_id,
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"

    append_session_messages(payload.session_id, [user_message, AIMessage(content=response_text)])
    await asyncio.to_thread(context_persister, payload.session_id, load_session_messages(payload.session_id))


def _stream_response_tokens(response_text: str, *, chunk_size: int = 24) -> list[str]:
    if not response_text:
        return [""]
    return [response_text[index : index + chunk_size] for index in range(0, len(response_text), chunk_size)]


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    return {
        "type": message.type,
        "content": message.content,
        "metadata": dict(getattr(message, "response_metadata", {}) or {}),
    }


def load_latest_eval_metrics(path: str | Path | None = None) -> dict[str, Any]:
    csv_path = Path(path or os.getenv("EVAL_RESULTS_PATH") or DEFAULT_EVAL_RESULTS_PATH)
    if not csv_path.exists():
        return {
            "available": False,
            "path": str(csv_path),
            "metrics": {},
            "message": "No evaluation results found.",
        }

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metrics: dict[str, float] = {}
    for metric in metric_names:
        values = [_safe_float(row.get(metric)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            metrics[metric] = round(sum(values) / len(values), 4)

    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({row.get("category", "") for row in rows if row.get("category")})
    for category in categories:
        category_rows = [row for row in rows if row.get("category") == category]
        category_metrics: dict[str, float] = {}
        for metric in metric_names:
            values = [_safe_float(row.get(metric)) for row in category_rows]
            values = [value for value in values if value is not None]
            if values:
                category_metrics[metric] = round(sum(values) / len(values), 4)
        by_category[category] = category_metrics

    return {
        "available": True,
        "path": str(csv_path),
        "runs": len(rows),
        "metrics": metrics,
        "by_category": by_category,
    }


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def print_route_map(app: FastAPI) -> None:
    table = Table(title="Healthcare RAG API Routes")
    table.add_column("Method", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Name")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", "")
        if not methods or path.startswith("/openapi") or path.startswith("/docs") or path.startswith("/redoc"):
            continue
        table.add_row(",".join(sorted(methods)), path, getattr(route, "name", ""))
    console.print(table)


app = create_app()
