from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.api.main import create_app, load_latest_eval_metrics


def auth_headers(client: TestClient, username: str = "doctor@example.com") -> dict[str, str]:
    response = client.post(
        "/auth/register",
        json={"username": username, "password": "correct horse battery staple"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def make_client(tmp_path: Path, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            agent_runner=kwargs.pop("agent_runner", fake_agent_runner),
            context_persister=kwargs.pop("context_persister", lambda *_: None),
            auth_db_path=tmp_path / "auth.sqlite",
            **kwargs,
        )
    )


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


def slow_agent_runner(state):
    time.sleep(0.05)
    return {**state, "response": "late response", "retrieved_docs": [], "faithfulness_score": 1.0}


def test_health_and_new_session(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    headers = auth_headers(client)

    health = client.get("/health")
    session = client.post("/session/new", headers=headers)

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert session.status_code == 200
    assert session.json()["session_id"]
    assert health.headers["x-request-id"]


def test_auth_register_login_and_me(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    register = client.post(
        "/auth/register",
        json={"username": "Clinician@Example.com", "password": "correct horse battery staple"},
    )
    login = client.post(
        "/auth/login",
        json={"username": "clinician@example.com", "password": "correct horse battery staple"},
    )
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"})

    assert register.status_code == 200
    assert login.status_code == 200
    assert me.status_code == 200
    assert me.json()["username"] == "clinician@example.com"


def test_protected_endpoints_require_bearer_token(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/session/new")

    assert response.status_code == 401


def test_request_id_header_accepts_caller_value(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/health", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-123"


def test_rate_limit_returns_429_with_retry_after(tmp_path: Path) -> None:
    client = make_client(tmp_path, rate_limit_per_minute=3)
    headers = auth_headers(client)

    first = client.post("/session/new", headers=headers)
    second = client.post("/session/new", headers=headers)
    third = client.post("/session/new", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"] == "rate_limited"
    assert third.headers["retry-after"]


def test_chat_streams_tokens_done_payload_and_persists_history(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    headers = auth_headers(client)
    session_id = "api-test-session"

    with client.stream(
        "POST",
        "/chat",
        headers=headers,
        json={"session_id": session_id, "message": "Can I mix them?"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data: [DONE]" in body
    assert "event: done" in body
    assert "bleeding risk" in body
    done_data = body.split("event: done\ndata: ", 1)[1].split("\n\n", 1)[0]
    done_payload = json.loads(done_data)
    assert done_payload["faithfulness_score"] == 0.91
    assert done_payload["citations"][0]["chunk_id"] == "drug-1"

    history = client.get(f"/session/{session_id}/history", headers=headers).json()
    assert [message["type"] for message in history["messages"]] == ["human", "ai"]


def test_session_memory_is_user_scoped(tmp_path: Path) -> None:
    persisted_session_ids: list[str] = []
    client = make_client(
        tmp_path,
        context_persister=lambda session_id, messages: persisted_session_ids.append(session_id),
    )
    first_user = auth_headers(client, "first@example.com")
    second_user = auth_headers(client, "second@example.com")
    session_id = "shared-session-id"

    with client.stream(
        "POST",
        "/chat",
        headers=first_user,
        json={"session_id": session_id, "message": "Can I mix them?"},
    ) as response:
        _ = "".join(response.iter_text())

    first_history = client.get(f"/session/{session_id}/history", headers=first_user).json()
    second_history = client.get(f"/session/{session_id}/history", headers=second_user).json()

    assert len(first_history["messages"]) == 2
    assert second_history["messages"] == []
    assert persisted_session_ids[0] != session_id
    assert persisted_session_ids[0].endswith(f":session:{session_id}")


def test_chat_stream_reports_agent_timeout(tmp_path: Path) -> None:
    client = make_client(
        tmp_path,
        agent_runner=slow_agent_runner,
        request_timeout_seconds=1,
        agent_timeout_seconds=0.001,
    )
    headers = auth_headers(client)

    with client.stream(
        "POST",
        "/chat",
        headers=headers,
        json={"session_id": "slow", "message": "hello"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: error" in body
    assert "agent_timeout" in body
    assert "data: [DONE]" in body


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


def test_upload_document_requires_authentication(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/documents/upload",
        files={"file": ("note.txt", b"Metformin is used for type 2 diabetes.", "text/plain")},
    )

    assert response.status_code == 401


def test_upload_document_indexes_user_document(tmp_path: Path) -> None:
    indexed: list[tuple[Path, str]] = []

    def fake_document_indexer(path: Path, user) -> dict:
        indexed.append((path, user.user_id))
        assert path.read_text(encoding="utf-8") == "Warfarin and aspirin may increase bleeding risk."
        return {
            "filename": path.name,
            "docs_loaded": 1,
            "chunks_indexed": 1,
            "batches": 1,
            "elapsed_seconds": 0.01,
            "estimated_cost_usd": 0.0,
        }

    client = make_client(tmp_path, document_indexer=fake_document_indexer)
    headers = auth_headers(client)

    response = client.post(
        "/documents/upload",
        headers=headers,
        files={
            "file": (
                "interaction-note.txt",
                b"Warfarin and aspirin may increase bleeding risk.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 1
    assert indexed
    assert not indexed[0][0].exists()


def test_upload_document_rejects_unsupported_file_type(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    headers = auth_headers(client)

    response = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("spreadsheet.csv", b"drug,evidence", "text/csv")},
    )

    assert response.status_code == 400
