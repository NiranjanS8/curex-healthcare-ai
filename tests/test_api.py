from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.api.main import create_app, load_latest_eval_metrics


def fake_agent_runner(state):
    return {
        **state,
        "response": "Aspirin and warfarin can increase bleeding risk. Consult a clinician.",
        "retrieved_docs": [
            Document(
                page_content="Warfarin and aspirin together may increase bleeding risk.",
                metadata={
                    "title": "Anticoagulant Safety",
                    "chunk_id": "drug-1",
                    "source": "pubmed",
                    "doc_type": "abstract",
                },
            )
        ],
        "faithfulness_score": 0.91,
    }


def test_health_and_new_session() -> None:
    client = TestClient(create_app(agent_runner=fake_agent_runner, context_persister=lambda *_: None))

    health = client.get("/health")
    session = client.post("/session/new")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert session.status_code == 200
    assert session.json()["session_id"]


def test_chat_streams_tokens_done_payload_and_persists_history() -> None:
    client = TestClient(create_app(agent_runner=fake_agent_runner, context_persister=lambda *_: None))
    session_id = "api-test-session"

    with client.stream("POST", "/chat", json={"session_id": session_id, "message": "Can I mix them?"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data: [DONE]" in body
    assert "event: done" in body
    assert "bleeding risk" in body
    done_data = body.split("event: done\ndata: ", 1)[1].split("\n\n", 1)[0]
    done_payload = json.loads(done_data)
    assert done_payload["faithfulness_score"] == 0.91
    assert done_payload["citations"][0]["chunk_id"] == "drug-1"

    history = client.get(f"/session/{session_id}/history").json()
    assert [message["type"] for message in history["messages"]] == ["human", "ai"]


def test_eval_metrics_returns_aggregate_scores(tmp_path: Path) -> None:
    path = tmp_path / "eval_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "category": "drug_interaction",
                "faithfulness": "0.8",
                "answer_relevancy": "0.9",
                "context_precision": "0.7",
                "context_recall": "0.6",
            }
        )
        writer.writerow(
            {
                "category": "drug_interaction",
                "faithfulness": "1.0",
                "answer_relevancy": "0.8",
                "context_precision": "0.9",
                "context_recall": "0.7",
            }
        )

    metrics = load_latest_eval_metrics(path)

    assert metrics["available"] is True
    assert metrics["runs"] == 2
    assert metrics["metrics"]["faithfulness"] == 0.9
    assert metrics["by_category"]["drug_interaction"]["context_recall"] == 0.65
