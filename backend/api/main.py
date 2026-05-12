"""FastAPI backend for the healthcare RAG assistant."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
from backend.auth import (
    AuthRequest,
    AuthUser,
    TokenResponse,
    authenticate_user,
    create_user,
    get_current_user,
    init_auth_db,
    token_response,
)
from backend.generation.prompts import format_citations
from backend.ingestion.indexer import index_uploaded_document
from backend.review import FeedbackPayload, FeedbackRecord, feedback_summary, init_feedback_db, save_feedback


console = Console()
DEFAULT_EVAL_RESULTS_PATH = Path("eval_results.csv")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
DEFAULT_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024
logger = logging.getLogger("healthcare_rag.api")
AgentRunner = Callable[[AgentState], AgentState]
ContextPersister = Callable[[str, list[BaseMessage]], Any]
DocumentIndexer = Callable[[Path, AuthUser], dict[str, Any]]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class NewSessionResponse(BaseModel):
    session_id: str


class DocumentIngestResponse(BaseModel):
    filename: str
    docs_loaded: int
    chunks_indexed: int
    batches: int
    elapsed_seconds: float
    estimated_cost_usd: float


def create_app(
    *,
    agent_runner: AgentRunner | None = None,
    context_persister: ContextPersister | None = None,
    document_indexer: DocumentIndexer | None = None,
    request_timeout_seconds: float | None = None,
    agent_timeout_seconds: float | None = None,
    rate_limit_per_minute: int | None = None,
    auth_db_path: str | Path | None = None,
    feedback_db_path: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print_route_map(app)
        yield

    app = FastAPI(title="Healthcare RAG Assistant API", version=__version__, lifespan=lifespan)
    app.state.auth_db_path = auth_db_path
    app.state.feedback_db_path = feedback_db_path
    init_auth_db(auth_db_path)
    init_feedback_db(feedback_db_path)
    app.state.agent_runner = agent_runner or _run_agent_with_tracking
    app.state.context_persister = context_persister or persist_conversation_context
    app.state.document_indexer = document_indexer or _index_uploaded_document_for_user
    app.state.request_timeout_seconds = float(
        request_timeout_seconds
        if request_timeout_seconds is not None
        else os.getenv("API_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    )
    app.state.agent_timeout_seconds = float(
        agent_timeout_seconds
        if agent_timeout_seconds is not None
        else os.getenv("AGENT_RUN_TIMEOUT_SECONDS", app.state.request_timeout_seconds)
    )
    app.state.rate_limit_per_minute = int(
        rate_limit_per_minute
        if rate_limit_per_minute is not None
        else os.getenv("API_RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE)
    )
    app.state.rate_limit_windows = {}

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
    async def reliability_middleware(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        limited_response = _check_rate_limit(request, request_id)
        if limited_response is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            _log_request(request, limited_response, request_id, elapsed_ms)
            return limited_response

        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=request.app.state.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            response = JSONResponse(
                status_code=504,
                content={
                    "error": "request_timeout",
                    "message": "The request exceeded the configured timeout.",
                    "request_id": request_id,
                },
            )
        except Exception:
            logger.exception("Unhandled API request error", extra={"request_id": request_id})
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "Unexpected server error.",
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = (time.perf_counter() - started) * 1000
        _log_request(request, response, request_id, elapsed_ms)
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/auth/register", response_model=TokenResponse)
    async def register(payload: AuthRequest, request: Request) -> TokenResponse:
        try:
            user = create_user(payload.username, payload.password, db_path=request.app.state.auth_db_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return token_response(user)

    @app.post("/auth/login", response_model=TokenResponse)
    async def login(payload: AuthRequest, request: Request) -> TokenResponse:
        user = authenticate_user(payload.username, payload.password, db_path=request.app.state.auth_db_path)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token_response(user)

    @app.get("/auth/me")
    async def me(current_user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
        return {"id": current_user.user_id, "username": current_user.username}

    @app.post("/session/new", response_model=NewSessionResponse)
    async def new_session(current_user: AuthUser = Depends(get_current_user)) -> NewSessionResponse:
        _ = current_user
        return NewSessionResponse(session_id=str(uuid4()))

    @app.get("/session/{session_id}/history")
    async def session_history(
        session_id: str,
        current_user: AuthUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        scoped_session_id = scoped_memory_session_id(current_user, session_id)
        return {
            "session_id": session_id,
            "messages": [_serialize_message(message) for message in load_session_messages(scoped_session_id)],
        }

    @app.get("/eval/metrics")
    async def eval_metrics(current_user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
        _ = current_user
        return load_latest_eval_metrics()

    @app.post("/feedback", response_model=FeedbackRecord)
    async def submit_feedback(
        payload: FeedbackPayload,
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
    ) -> FeedbackRecord:
        return save_feedback(payload, user_id=current_user.user_id, db_path=request.app.state.feedback_db_path)

    @app.get("/feedback/summary")
    async def get_feedback_summary(
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        return feedback_summary(user_id=current_user.user_id, db_path=request.app.state.feedback_db_path)

    @app.post("/documents/upload", response_model=DocumentIngestResponse)
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        current_user: AuthUser = Depends(get_current_user),
    ) -> DocumentIngestResponse:
        saved_path = await _save_upload(file)
        try:
            result = await asyncio.to_thread(request.app.state.document_indexer, saved_path, current_user)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        finally:
            saved_path.unlink(missing_ok=True)
            try:
                saved_path.parent.rmdir()
            except OSError:
                logger.warning("Upload temp directory cleanup skipped", extra={"path": str(saved_path.parent)})
        return DocumentIngestResponse(**result)

    @app.post("/chat")
    async def chat(
        payload: ChatRequest,
        request: Request,
        current_user: AuthUser = Depends(get_current_user),
    ) -> StreamingResponse:
        return StreamingResponse(
            _chat_event_stream(
                payload,
                current_user,
                request.app.state.agent_runner,
                request.app.state.context_persister,
                timeout_seconds=request.app.state.agent_timeout_seconds,
                request_id=request.state.request_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def scoped_memory_session_id(user: AuthUser, session_id: str) -> str:
    return f"user:{user.user_id}:session:{session_id}"


def _index_uploaded_document_for_user(path: Path, user: AuthUser) -> dict[str, Any]:
    return index_uploaded_document(path, owner_user_id=user.user_id)


async def _save_upload(upload: UploadFile) -> Path:
    raw_name = Path(upload.filename or "upload.txt").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, TXT, and MD uploads are supported.",
        )

    data = await upload.read()
    await upload.close()
    if len(data) > int(os.getenv("DOCUMENT_UPLOAD_LIMIT_BYTES", DEFAULT_UPLOAD_LIMIT_BYTES)):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded document is too large.",
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="curex_upload_"))
    saved_path = temp_dir / raw_name
    saved_path.write_bytes(data)
    return saved_path


def _log_request(request: Request, response: Response, request_id: str, elapsed_ms: float) -> None:
    log_payload = {
        "event": "api_request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(elapsed_ms, 2),
        "client": request.client.host if request.client else "unknown",
    }
    logger.info(json.dumps(log_payload, sort_keys=True))
    console.print_json(data=log_payload)


def _check_rate_limit(request: Request, request_id: str) -> Response | None:
    limit = int(request.app.state.rate_limit_per_minute)
    if limit <= 0 or request.url.path == "/health":
        return None

    now = time.time()
    window_start = now - 60
    key = _client_key(request)
    windows: dict[str, list[float]] = request.app.state.rate_limit_windows
    hits = [timestamp for timestamp in windows.get(key, []) if timestamp >= window_start]
    if len(hits) >= limit:
        retry_after = max(1, int(60 - (now - min(hits))))
        response = JSONResponse(
            status_code=429,
            content={
                "error": "rate_limited",
                "message": "Too many requests. Please try again later.",
                "request_id": request_id,
            },
        )
        response.headers["Retry-After"] = str(retry_after)
        response.headers["X-Request-ID"] = request_id
        return response
    hits.append(now)
    windows[key] = hits
    return None


def _run_agent_with_tracking(state: AgentState) -> AgentState:
    tracker = CostTracker(session_id=state["session_id"], query=state["query"])
    result = invoke_agent(state, callbacks=[tracker])
    tracker.finish(faithfulness=result.get("faithfulness_score"))
    return result


async def _chat_event_stream(
    payload: ChatRequest,
    current_user: AuthUser,
    agent_runner: AgentRunner,
    context_persister: ContextPersister,
    *,
    timeout_seconds: float,
    request_id: str,
) -> AsyncIterator[str]:
    user_message = HumanMessage(content=payload.message)
    scoped_session_id = scoped_memory_session_id(current_user, payload.session_id)
    memory = get_memory(scoped_session_id)
    state: AgentState = {
        "query": payload.message,
        "session_id": scoped_session_id,
        "user_id": current_user.user_id,
        "messages": [*memory.get("session_messages", []), user_message],
    }
    if memory.get("system_message") is not None:
        state["messages"].insert(0, memory["system_message"])

    try:
        result = await asyncio.wait_for(asyncio.to_thread(agent_runner, state), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        error_payload = {
            "error": "agent_timeout",
            "message": "The assistant run exceeded the configured timeout.",
            "request_id": request_id,
        }
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception:
        logger.exception("Chat stream failed", extra={"request_id": request_id})
        error_payload = {
            "error": "agent_error",
            "message": "The assistant run failed.",
            "request_id": request_id,
        }
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return
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
        "request_id": request_id,
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"

    append_session_messages(scoped_session_id, [user_message, AIMessage(content=response_text)])
    await asyncio.to_thread(context_persister, scoped_session_id, load_session_messages(scoped_session_id))


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
