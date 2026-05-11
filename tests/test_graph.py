from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from backend.agent import graph as agent_graph
from backend.agent.router import QueryIntent


class FakeRetriever:
    def invoke(self, query: str):
        return [
            Document(
                page_content=f"evidence for {query}",
                metadata={"title": "Clinical Source", "chunk_id": "chunk-1"},
            )
        ]


def test_route_after_query_router_uses_intent_route() -> None:
    state = {
        "query": "Can I take warfarin with aspirin?",
        "session_id": "s1",
        "messages": [],
        "intent": QueryIntent(category="drug_interaction", confidence=0.9, entities=["warfarin"]),
    }

    assert agent_graph.route_after_query_router(state) == "tool_executor"


def test_faithfulness_route_retries_at_most_twice() -> None:
    state = {
        "query": "q",
        "session_id": "s1",
        "messages": [],
        "faithfulness_score": 0.4,
        "faithfulness_retries": 1,
    }
    assert agent_graph.route_after_faithfulness(state) == "response_generator"

    state["faithfulness_retries"] = 2
    assert agent_graph.route_after_faithfulness(state) == agent_graph.END


def test_graph_retrieval_path_invokes_retriever(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "classify_intent",
        lambda query: QueryIntent(category="general_health", confidence=0.8, entities=[]),
    )
    agent_graph.configure_retriever(FakeRetriever())

    try:
        result = agent_graph.graph.invoke(
            {
                "query": "What is diabetes?",
                "session_id": "session-1",
                "messages": [HumanMessage(content="What is diabetes?")],
            }
        )
    finally:
        agent_graph.configure_retriever(None)

    assert result["intent"].category == "general_health"
    assert result["retrieved_docs"][0].metadata["chunk_id"] == "chunk-1"
    assert "Clinical Source" in result["response"]
    assert result["faithfulness_score"] == 1.0


def test_graph_tool_path_skips_retriever(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "classify_intent",
        lambda query: QueryIntent(category="drug_interaction", confidence=0.9, entities=["warfarin", "aspirin"]),
    )
    monkeypatch.setattr(
        agent_graph,
        "run_drug_interaction_tool",
        lambda drug_names: {"pairs": [], "resolved": {drug: drug for drug in drug_names}},
    )

    result = agent_graph.graph.invoke(
        {
            "query": "Can I take warfarin with aspirin?",
            "session_id": "session-1",
            "messages": [],
        }
    )

    assert result["tool_results"][0]["tool"] == "check_drug_interactions"
    assert "medical tool" in result["response"]


def test_save_agent_graph_png_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "agent_graph.png"

    result = agent_graph.save_agent_graph_png(output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
