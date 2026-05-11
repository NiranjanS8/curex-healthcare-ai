"""Safety guardrails for healthcare assistant queries and responses."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from backend.generation.prompts import DISCLAIMER


DEFAULT_SAFETY_DB_PATH = Path("healthcare_safety.sqlite")
CRISIS_MESSAGE = (
    "I cannot help with instructions for self-harm or medication misuse. "
    "If you may hurt yourself or someone else, call your local emergency number now. "
    "In the U.S. or Canada, call or text 988 for immediate crisis support."
)
OFF_TOPIC_MESSAGE = "I can only help with educational healthcare questions."
DIAGNOSIS_REPLACEMENTS = {
    r"\byou have\b": "this may indicate",
    r"\byou are diagnosed with\b": "a clinician may evaluate for",
    r"\byou suffer from\b": "this may be consistent with",
}


class SafetyResult(BaseModel):
    safe: bool
    reason: str
    modified_query: str


class SafetyClassification(BaseModel):
    label: Literal["in_scope", "off_topic", "harmful"]
    reason: str
    modified_query: str = ""


class StructuredSafetyClassifier(Protocol):
    def invoke(self, messages):
        """Return structured safety classification."""


SAFETY_SYSTEM_PROMPT = """Classify a user query for a healthcare information assistant.

Labels:
- in_scope: educational healthcare, medication, symptom, guideline, or wellness question.
- off_topic: unrelated to healthcare or asks for non-health tasks.
- harmful: requests self-harm, medication misuse, poisoning, evading care, or dangerous medical instructions.

Return label, reason, and a modified_query that is safe and medically educational when possible."""


def _db_path(path: str | Path | None = None) -> Path:
    return Path(path or os.getenv("SAFETY_DB_PATH") or DEFAULT_SAFETY_DB_PATH)


def init_safety_db(path: str | Path | None = None) -> Path:
    db_path = _db_path(path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                safe INTEGER NOT NULL,
                reason TEXT NOT NULL,
                modified_query TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    return db_path


def log_safety_result(
    query: str,
    result: SafetyResult,
    *,
    db_path: str | Path | None = None,
) -> None:
    with sqlite3.connect(init_safety_db(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO safety_log (query, safe, reason, modified_query, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                query,
                1 if result.safe else 0,
                result.reason,
                result.modified_query,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_safety_log(*, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with sqlite3.connect(init_safety_db(db_path)) as conn:
        rows = conn.execute(
            "SELECT query, safe, reason, modified_query, created_at FROM safety_log ORDER BY id"
        ).fetchall()
    return [
        {
            "query": query,
            "safe": bool(safe),
            "reason": reason,
            "modified_query": modified_query,
            "created_at": created_at,
        }
        for query, safe, reason, modified_query, created_at in rows
    ]


def get_safety_classifier():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(
        SafetyClassification
    )


def _coerce_classification(result: Any) -> SafetyClassification:
    if isinstance(result, SafetyClassification):
        return result
    if isinstance(result, dict):
        return SafetyClassification.model_validate(result)
    raise TypeError(f"Unsupported safety classifier result: {type(result)!r}")


def _heuristic_classify(query: str) -> SafetyClassification:
    lowered = query.lower()
    harmful_patterns = [
        "kill myself",
        "suicide",
        "overdose",
        "poison",
        "lethal dose",
        "how many pills",
    ]
    health_terms = [
        "health",
        "doctor",
        "symptom",
        "medicine",
        "medication",
        "drug",
        "dose",
        "pain",
        "diabetes",
        "blood",
        "fever",
        "guideline",
        "bmi",
    ]
    if any(pattern in lowered for pattern in harmful_patterns):
        return SafetyClassification(label="harmful", reason="Potential self-harm or medication misuse.")
    if not any(term in lowered for term in health_terms):
        return SafetyClassification(label="off_topic", reason="The query is outside healthcare scope.")
    return SafetyClassification(label="in_scope", reason="The query is healthcare-related.", modified_query=query)


def pre_check(
    query: str,
    *,
    classifier: StructuredSafetyClassifier | None = None,
    db_path: str | Path | None = None,
    log: bool = True,
) -> SafetyResult:
    """Classify and optionally rewrite a query before retrieval or tool use."""

    if classifier is not None:
        classification = _coerce_classification(
            classifier.invoke([("system", SAFETY_SYSTEM_PROMPT), ("human", query)])
        )
    elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            classification = _coerce_classification(
                get_safety_classifier().invoke([("system", SAFETY_SYSTEM_PROMPT), ("human", query)])
            )
        except Exception:
            classification = _heuristic_classify(query)
    else:
        classification = _heuristic_classify(query)

    if classification.label == "harmful":
        result = SafetyResult(safe=False, reason=CRISIS_MESSAGE, modified_query="")
    elif classification.label == "off_topic":
        result = SafetyResult(safe=False, reason=OFF_TOPIC_MESSAGE, modified_query="")
    else:
        result = SafetyResult(
            safe=True,
            reason=classification.reason,
            modified_query=classification.modified_query or query,
        )

    if log or not result.safe:
        log_safety_result(query, result, db_path=db_path)
    print_safety_result(result)
    return result


def post_check(response: str) -> str:
    """Soften diagnosis language and ensure the medical disclaimer is present."""

    checked = response
    for pattern, replacement in DIAGNOSIS_REPLACEMENTS.items():
        checked = re.sub(pattern, replacement, checked, flags=re.IGNORECASE)
    if DISCLAIMER not in checked:
        checked = checked.rstrip() + "\n\n" + DISCLAIMER
    return checked


def print_safety_result(result: SafetyResult) -> None:
    color = "green" if result.safe else "red"
    body = (
        f"[bold]Safe:[/bold] {result.safe}\n"
        f"[bold]Reason:[/bold] {result.reason}\n"
        f"[bold]Modified query:[/bold] {result.modified_query or 'n/a'}"
    )
    Console().print(Panel(body, title="Safety Check", border_style=color))
