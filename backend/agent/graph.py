"""LangGraph state graph for the healthcare RAG assistant."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.table import Table

from backend.agent.router import QueryIntent, classify_intent, route
from backend.agent.tools import check_drug_interactions
from backend.generation.faithfulness import score_faithfulness
from backend.generation.safety import post_check, pre_check


os.environ.setdefault("LANGCHAIN_PROJECT", "healthcare-rag")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

DEFAULT_QUERY_LOG_DB_PATH = Path("healthcare_query_log.sqlite")
MODEL_PRICING_PER_1M_TOKENS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class AgentState(TypedDict):
    messages: list[BaseMessage]
    query: str
    intent: NotRequired[QueryIntent]
    retrieved_docs: NotRequired[list[Document]]
    tool_results: NotRequired[list[dict]]
    safety_result: NotRequired[dict]
    response: NotRequired[str]
    faithfulness_score: NotRequired[float]
    unsupported_claims: NotRequired[list[str]]
    session_id: str
    faithfulness_retries: NotRequired[int]


_RETRIEVER: BaseRetriever | None = None


def configure_retriever(retriever: BaseRetriever | None) -> None:
    """Configure the retriever used by the graph retriever node."""

    global _RETRIEVER
    _RETRIEVER = retriever


def get_retriever() -> BaseRetriever | None:
    return _RETRIEVER


def run_drug_interaction_tool(drug_names: list[str]) -> dict:
    return check_drug_interactions.invoke({"drug_names": drug_names})


def query_router(state: AgentState) -> dict:
    safety_result = pre_check(state["query"])
    if not safety_result.safe:
        intent = QueryIntent(category="out_of_scope", confidence=1.0, entities=[])
        return {
            "intent": intent,
            "safety_result": safety_result.model_dump(),
            "response": safety_result.reason,
        }
    intent = classify_intent(safety_result.modified_query)
    return {
        "intent": intent,
        "query": safety_result.modified_query,
        "safety_result": safety_result.model_dump(),
    }


def retriever(state: AgentState) -> dict:
    configured_retriever = get_retriever()
    docs = configured_retriever.invoke(state["query"]) if configured_retriever is not None else []
    return {"retrieved_docs": docs}


def tool_executor(state: AgentState) -> dict:
    intent = state["intent"]
    if intent.category == "drug_interaction":
        result = run_drug_interaction_tool(intent.entities)
        return {"tool_results": [{"tool": "check_drug_interactions", "result": result}]}

    return {
        "tool_results": [
            {
                "tool": "pending_medical_tool",
                "intent": intent.category,
                "entities": intent.entities,
                "status": "not_configured",
            }
        ]
    }


def response_generator(state: AgentState) -> dict:
    if state.get("response") and state.get("faithfulness_score", 1.0) < 0.7:
        response = (
            "I can only use facts explicitly stated in the retrieved context. "
            "The available context is not strong enough for a detailed answer."
        )
    elif state.get("retrieved_docs"):
        citations = []
        for doc in state["retrieved_docs"][:5]:
            title = doc.metadata.get("title") or doc.metadata.get("source") or "retrieved source"
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            citations.append(f"[Source: {title}, chunk {chunk_id}]")
        response = "I found relevant medical context. " + " ".join(citations)
    elif state.get("tool_results"):
        response = "I routed this query to a medical tool. Tool execution will be expanded in an upcoming update."
    else:
        response = "I do not have enough retrieved medical context to answer this safely."

    messages = list(state.get("messages", []))
    checked_response = post_check(response)
    messages.append(AIMessage(content=checked_response))
    return {"response": checked_response, "messages": messages}


def safety_check(state: AgentState) -> dict:
    intent = state.get("intent")
    if intent and intent.category == "out_of_scope":
        response = post_check(state.get("response") or "I can only help with educational healthcare questions.")
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=response))
        return {"response": response, "messages": messages, "faithfulness_score": 1.0}
    if state.get("response"):
        response = post_check(state["response"])
        return {"response": response}
    return {}


def faithfulness_check(state: AgentState) -> dict:
    if state.get("retrieved_docs") and state.get("response"):
        score = score_faithfulness(state["response"], state["retrieved_docs"])
    elif state.get("tool_results") or state.get("safety_result", {}).get("safe") is False:
        score = 1.0
    else:
        score = 0.0

    updates: dict = {"faithfulness_score": score}
    if score < 0.7:
        next_retry = state.get("faithfulness_retries", 0) + 1
        updates["faithfulness_retries"] = next_retry
        if next_retry >= 2 and state.get("response"):
            warning = "Low confidence: this answer may not be fully supported by the retrieved context.\n\n"
            updates["response"] = warning + state["response"]
    return updates


def route_after_query_router(state: AgentState) -> str:
    return route(state["intent"])


def route_after_faithfulness(state: AgentState) -> str:
    if state.get("faithfulness_score", 0.0) < 0.7 and state.get("faithfulness_retries", 0) < 2:
        return "response_generator"
    return END


def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("query_router", query_router)
    workflow.add_node("retriever", retriever)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("response_generator", response_generator)
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("faithfulness_check", faithfulness_check)

    workflow.add_edge(START, "query_router")
    workflow.add_conditional_edges(
        "query_router",
        route_after_query_router,
        {
            "retriever": "retriever",
            "tool_executor": "tool_executor",
            "safety_check": "safety_check",
        },
    )
    workflow.add_edge("retriever", "response_generator")
    workflow.add_edge("tool_executor", "response_generator")
    workflow.add_edge("response_generator", "safety_check")
    workflow.add_edge("safety_check", "faithfulness_check")
    workflow.add_conditional_edges(
        "faithfulness_check",
        route_after_faithfulness,
        {
            "response_generator": "response_generator",
            END: END,
        },
    )
    return workflow.compile()


graph = build_agent_graph()


def _intent_category(state: AgentState) -> str:
    intent = state.get("intent")
    return intent.category if intent else "unknown"


def build_run_config(state: AgentState, *, callbacks: list[Any] | None = None) -> dict[str, Any]:
    """Build LangChain run config with tracing metadata."""

    config: dict[str, Any] = {
        "metadata": {
            "query_category": _intent_category(state),
            "session_id": state["session_id"],
            "faithfulness_score": state.get("faithfulness_score"),
        }
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config


def invoke_agent(state: AgentState, *, callbacks: list[Any] | None = None) -> AgentState:
    """Invoke the compiled graph with standard metadata attached."""

    return graph.invoke(state, config=build_run_config(state, callbacks=callbacks))


def init_query_log(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or os.getenv("QUERY_LOG_DB_PATH") or DEFAULT_QUERY_LOG_DB_PATH)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms REAL NOT NULL,
                faithfulness REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
    return path


def log_query_run(
    *,
    session_id: str,
    query: str,
    cost_usd: float,
    latency_ms: float,
    faithfulness: float | None,
    db_path: str | Path | None = None,
) -> None:
    with sqlite3.connect(init_query_log(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO query_log (session_id, query, cost_usd, latency_ms, faithfulness, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                query,
                float(cost_usd),
                float(latency_ms),
                faithfulness,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _token_usage_from_llm_result(response: LLMResult) -> dict[str, int]:
    usage: dict[str, int] = {}
    llm_output = response.llm_output or {}
    if isinstance(llm_output, dict):
        usage.update(llm_output.get("token_usage") or {})
        usage.update(llm_output.get("usage") or {})

    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", {}) if message is not None else {}
            usage.update(metadata.get("token_usage") or {})
            usage.update(metadata.get("usage") or {})

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
    }


class CostTracker(BaseCallbackHandler):
    """Track token usage, latency, and estimated model cost for a query run."""

    def __init__(
        self,
        *,
        session_id: str,
        query: str,
        model_name: str = "gpt-4o",
        faithfulness: float | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.query = query
        self.model_name = model_name
        self.faithfulness = faithfulness
        self.db_path = db_path
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.started_at = time.perf_counter()
        self.cost_usd = 0.0
        self.latency_ms = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _token_usage_from_llm_result(response)
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]

    def calculate_cost(self) -> float:
        pricing = MODEL_PRICING_PER_1M_TOKENS.get(
            self.model_name,
            MODEL_PRICING_PER_1M_TOKENS["gpt-4o"],
        )
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    def finish(self, *, faithfulness: float | None = None) -> dict[str, float | int]:
        self.latency_ms = (time.perf_counter() - self.started_at) * 1000
        if faithfulness is not None:
            self.faithfulness = faithfulness
        self.cost_usd = self.calculate_cost()
        log_query_run(
            session_id=self.session_id,
            query=self.query,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
            faithfulness=self.faithfulness,
            db_path=self.db_path,
        )
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


def get_run_summary(n: int = 20, *, db_path: str | Path | None = None) -> dict[str, float | int]:
    with sqlite3.connect(init_query_log(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT cost_usd, latency_ms, faithfulness
            FROM query_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

    if not rows:
        return {
            "runs": 0,
            "avg_latency_ms": 0.0,
            "avg_cost_usd": 0.0,
            "avg_faithfulness": 0.0,
        }

    faithfulness_values = [row[2] for row in rows if row[2] is not None]
    return {
        "runs": len(rows),
        "avg_latency_ms": sum(row[1] for row in rows) / len(rows),
        "avg_cost_usd": sum(row[0] for row in rows) / len(rows),
        "avg_faithfulness": (
            sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0.0
        ),
    }


def print_graph_structure() -> None:
    table = Table(title="Healthcare RAG Agent Graph")
    table.add_column("Node")
    table.add_column("Role")
    rows = [
        ("query_router", "Classify intent and select route"),
        ("retriever", "Fetch medical chunks"),
        ("tool_executor", "Run medical tools"),
        ("response_generator", "Draft response from context/tool output"),
        ("safety_check", "Apply scope and safety checks"),
        ("faithfulness_check", "Score support and retry if needed"),
    ]
    for node, role_text in rows:
        table.add_row(node, role_text)
    Console().print(table)


def save_agent_graph_png(output_path: str | Path = "agent_graph.png") -> Path:
    output = Path(output_path)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        output.write_bytes(png_bytes)
    except Exception:
        _write_fallback_graph_png(output)
    return output


def _write_fallback_graph_png(output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1100, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    nodes = {
        "query_router": (430, 40),
        "retriever": (190, 180),
        "tool_executor": (430, 180),
        "response_generator": (330, 320),
        "safety_check": (330, 460),
        "faithfulness_check": (330, 590),
    }
    for label, (x, y) in nodes.items():
        draw.rounded_rectangle((x, y, x + 210, y + 70), radius=12, outline="#0f766e", width=3)
        draw.text((x + 24, y + 28), label, fill="#111827", font=font)

    edges = [
        ("query_router", "retriever"),
        ("query_router", "tool_executor"),
        ("retriever", "response_generator"),
        ("tool_executor", "response_generator"),
        ("response_generator", "safety_check"),
        ("safety_check", "faithfulness_check"),
        ("faithfulness_check", "response_generator"),
    ]
    for start, end in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        draw.line((x1 + 105, y1 + 70, x2 + 105, y2), fill="#0f766e", width=2)
        draw.polygon([(x2 + 98, y2 - 8), (x2 + 105, y2), (x2 + 112, y2 - 8)], fill="#0f766e")

    draw.text((40, 20), "Healthcare RAG Agent State Graph", fill="#111827", font=font)
    image.save(output)


if __name__ == "__main__":
    print_graph_structure()
    save_agent_graph_png()
