"""LangGraph state graph for the healthcare RAG assistant."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.table import Table

from backend.agent.router import QueryIntent, classify_intent, route


class AgentState(TypedDict):
    messages: list[BaseMessage]
    query: str
    intent: NotRequired[QueryIntent]
    retrieved_docs: NotRequired[list[Document]]
    tool_results: NotRequired[list[dict]]
    response: NotRequired[str]
    faithfulness_score: NotRequired[float]
    session_id: str
    faithfulness_retries: NotRequired[int]


_RETRIEVER: BaseRetriever | None = None


def configure_retriever(retriever: BaseRetriever | None) -> None:
    """Configure the retriever used by the graph retriever node."""

    global _RETRIEVER
    _RETRIEVER = retriever


def get_retriever() -> BaseRetriever | None:
    return _RETRIEVER


def query_router(state: AgentState) -> dict:
    intent = classify_intent(state["query"])
    return {"intent": intent}


def retriever(state: AgentState) -> dict:
    configured_retriever = get_retriever()
    docs = configured_retriever.invoke(state["query"]) if configured_retriever is not None else []
    return {"retrieved_docs": docs}


def tool_executor(state: AgentState) -> dict:
    intent = state["intent"]
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
            "I can only answer from explicitly retrieved evidence. "
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
    messages.append(AIMessage(content=response))
    return {"response": response, "messages": messages}


def safety_check(state: AgentState) -> dict:
    intent = state.get("intent")
    if intent and intent.category == "out_of_scope":
        response = "I can only help with educational healthcare questions."
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=response))
        return {"response": response, "messages": messages, "faithfulness_score": 1.0}
    return {}


def faithfulness_check(state: AgentState) -> dict:
    if "faithfulness_score" in state:
        score = float(state["faithfulness_score"])
    elif state.get("retrieved_docs") or state.get("tool_results") or state.get("response"):
        score = 1.0
    else:
        score = 0.0

    updates: dict = {"faithfulness_score": score}
    if score < 0.7:
        updates["faithfulness_retries"] = state.get("faithfulness_retries", 0) + 1
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
