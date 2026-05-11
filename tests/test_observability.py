from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from backend.agent.graph import (
    CostTracker,
    build_run_config,
    get_run_summary,
    init_query_log,
    invoke_agent,
    log_query_run,
)


def test_build_run_config_includes_metadata() -> None:
    state = {"query": "q", "session_id": "s1", "messages": [], "faithfulness_score": 0.8}

    config = build_run_config(state)

    assert config["metadata"] == {
        "query_category": "unknown",
        "session_id": "s1",
        "faithfulness_score": 0.8,
    }


def test_cost_tracker_extracts_tokens_and_logs_run(tmp_path) -> None:
    db_path = tmp_path / "query_log.sqlite"
    tracker = CostTracker(session_id="s1", query="hello", model_name="gpt-4o", db_path=db_path)
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={"token_usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
    )

    tracker.on_llm_end(response)
    result = tracker.finish(faithfulness=0.75)

    assert tracker.prompt_tokens == 1000
    assert tracker.completion_tokens == 500
    assert result["cost_usd"] == 0.0075
    summary = get_run_summary(db_path=db_path)
    assert summary["runs"] == 1
    assert summary["avg_cost_usd"] == 0.0075
    assert summary["avg_faithfulness"] == 0.75


def test_cost_tracker_reads_response_metadata_tokens(tmp_path) -> None:
    tracker = CostTracker(session_id="s1", query="hello", model_name="gpt-4o-mini", db_path=tmp_path / "q.sqlite")
    message = AIMessage(
        content="hi",
        response_metadata={"usage": {"input_tokens": 2000, "output_tokens": 1000}},
    )
    response = LLMResult(generations=[[ChatGeneration(message=message)]])

    tracker.on_llm_end(response)
    result = tracker.finish()

    assert result["cost_usd"] == 0.0009


def test_get_run_summary_averages_recent_runs(tmp_path) -> None:
    db_path = tmp_path / "query_log.sqlite"
    init_query_log(db_path)
    log_query_run(session_id="s1", query="a", cost_usd=0.1, latency_ms=100, faithfulness=0.8, db_path=db_path)
    log_query_run(session_id="s2", query="b", cost_usd=0.3, latency_ms=300, faithfulness=0.6, db_path=db_path)

    summary = get_run_summary(n=2, db_path=db_path)

    assert summary == {
        "runs": 2,
        "avg_latency_ms": 200.0,
        "avg_cost_usd": 0.2,
        "avg_faithfulness": 0.7,
    }


def test_get_run_summary_handles_empty_log(tmp_path) -> None:
    assert get_run_summary(db_path=tmp_path / "empty.sqlite") == {
        "runs": 0,
        "avg_latency_ms": 0.0,
        "avg_cost_usd": 0.0,
        "avg_faithfulness": 0.0,
    }


def test_invoke_agent_passes_standard_config(monkeypatch) -> None:
    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            captured["config"] = config
            return {**state, "response": "ok"}

    monkeypatch.setattr("backend.agent.graph.graph", FakeGraph())
    state = {"query": "q", "session_id": "s1", "messages": []}

    result = invoke_agent(state)

    assert result["response"] == "ok"
    assert captured["config"]["metadata"]["session_id"] == "s1"
