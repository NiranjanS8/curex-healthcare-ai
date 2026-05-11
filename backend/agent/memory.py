"""Session and long-term memory for the healthcare agent."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, SystemMessage, messages_from_dict, messages_to_dict
from pydantic import BaseModel, Field


DEFAULT_MEMORY_DB_PATH = Path("healthcare_memory.sqlite")
SESSION_WINDOW_TURNS = 6
SESSION_MESSAGE_LIMIT = SESSION_WINDOW_TURNS * 2
_IN_MEMORY_SESSIONS: dict[str, list[dict[str, Any]]] = {}


class PatientContext(BaseModel):
    age: str | None = None
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


@dataclass
class MemoryContext:
    session_id: str
    session_messages: list[BaseMessage]
    long_term: dict[str, Any]
    system_message: SystemMessage | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_messages": self.session_messages,
            "long_term": self.long_term,
            "system_message": self.system_message,
        }


class StructuredContextExtractor(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        """Return structured patient context for a conversation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(path: str | Path | None = None) -> Path:
    return Path(path or os.getenv("MEMORY_DB_PATH") or DEFAULT_MEMORY_DB_PATH)


def init_db(path: str | Path | None = None) -> Path:
    db_path = _db_path(path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_context (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, key)
            )
            """
        )
    return db_path


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = init_db(path)
    return sqlite3.connect(db_path)


def _get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _session_key(session_id: str) -> str:
    return f"healthcare-rag:session:{session_id}:messages"


def save_session_messages(session_id: str, messages: list[BaseMessage]) -> None:
    payload = messages_to_dict(messages)
    client = _get_redis_client()
    if client is not None:
        client.set(_session_key(session_id), json.dumps(payload))
    else:
        _IN_MEMORY_SESSIONS[session_id] = payload


def load_session_messages(session_id: str) -> list[BaseMessage]:
    client = _get_redis_client()
    if client is not None:
        raw = client.get(_session_key(session_id))
        payload = json.loads(raw) if raw else []
    else:
        payload = _IN_MEMORY_SESSIONS.get(session_id, [])
    return messages_from_dict(payload)[-SESSION_MESSAGE_LIMIT:]


def append_session_messages(session_id: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    existing = load_session_messages(session_id)
    updated = (existing + messages)[-SESSION_MESSAGE_LIMIT:]
    save_session_messages(session_id, updated)
    return updated


def get_session_memory(session_id: str):
    from langchain_classic.memory import ConversationBufferWindowMemory

    memory = ConversationBufferWindowMemory(k=SESSION_WINDOW_TURNS, return_messages=True)
    for message in load_session_messages(session_id):
        memory.chat_memory.add_message(message)
    return memory


def _coerce_patient_context(result: Any) -> PatientContext:
    if isinstance(result, PatientContext):
        return result
    if isinstance(result, dict):
        return PatientContext.model_validate(result)
    return PatientContext()


def _extract_with_regex(transcript: str) -> PatientContext:
    age_match = re.search(r"\b(?:age|aged|i am|i'm)\s*(\d{1,3})\b", transcript, re.IGNORECASE)
    medications = sorted(
        {
            match.group(1).strip()
            for match in re.finditer(
                r"\b(?:take|taking|medication|medications|on)\s+([A-Za-z][A-Za-z0-9 -]{1,40})",
                transcript,
                re.IGNORECASE,
            )
        }
    )
    allergies = sorted(
        {
            match.group(1).strip()
            for match in re.finditer(
                r"\ballerg(?:y|ic|ies)\s+(?:to\s+)?([A-Za-z][A-Za-z0-9 -]{1,40})",
                transcript,
                re.IGNORECASE,
            )
        }
    )
    conditions = sorted(
        {
            condition
            for condition in ["asthma", "diabetes", "hypertension", "heart disease", "kidney disease"]
            if re.search(rf"\b{re.escape(condition)}\b", transcript, re.IGNORECASE)
        }
    )
    return PatientContext(
        age=age_match.group(1) if age_match else None,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
    )


def extract_patient_context(
    conversation: list[BaseMessage],
    *,
    extractor: StructuredContextExtractor | None = None,
) -> PatientContext:
    transcript = "\n".join(f"{message.type}: {message.content}" for message in conversation)
    if extractor is not None:
        return _coerce_patient_context(
            extractor.invoke(
                [
                    (
                        "system",
                        "Extract patient context as age, conditions, medications, and allergies. "
                        "Only include facts explicitly stated in the conversation.",
                    ),
                    ("human", transcript),
                ]
            )
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            structured = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(PatientContext)
            return _coerce_patient_context(
                structured.invoke(
                    [
                        (
                            "system",
                            "Extract patient context as age, conditions, medications, and allergies. "
                            "Only include facts explicitly stated in the conversation.",
                        ),
                        ("human", transcript),
                    ]
                )
            )
        except Exception:
            pass

    return _extract_with_regex(transcript)


def save_patient_context(
    session_id: str,
    context: PatientContext | dict[str, Any],
    *,
    db_path: str | Path | None = None,
) -> None:
    patient_context = context if isinstance(context, PatientContext) else PatientContext.model_validate(context)
    values = patient_context.model_dump()
    timestamp = _utc_now()
    with _connect(db_path) as conn:
        for key, value in values.items():
            if value in (None, [], ""):
                continue
            conn.execute(
                """
                INSERT INTO patient_context (session_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (session_id, key, json.dumps(value), timestamp),
            )


def load_patient_context(session_id: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key, value FROM patient_context WHERE session_id = ? ORDER BY key",
            (session_id,),
        ).fetchall()
    return {key: json.loads(value) for key, value in rows}


def build_patient_context_message(context: dict[str, Any]) -> SystemMessage | None:
    if not context:
        return None
    parts: list[str] = []
    for key in ["age", "conditions", "medications", "allergies"]:
        value = context.get(key)
        if not value:
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    if not parts:
        return None
    return SystemMessage(content="Known patient context: " + "; ".join(parts))


def persist_conversation_context(
    session_id: str,
    conversation: list[BaseMessage],
    *,
    extractor: StructuredContextExtractor | None = None,
    db_path: str | Path | None = None,
) -> PatientContext:
    context = extract_patient_context(conversation, extractor=extractor)
    save_patient_context(session_id, context, db_path=db_path)
    return context


def get_memory(session_id: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    long_term = load_patient_context(session_id, db_path=db_path)
    context = MemoryContext(
        session_id=session_id,
        session_messages=load_session_messages(session_id),
        long_term=long_term,
        system_message=build_patient_context_message(long_term),
    )
    return asdict(context)
