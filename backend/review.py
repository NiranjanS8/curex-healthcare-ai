"""Human-in-the-loop feedback storage for assistant responses."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


DEFAULT_FEEDBACK_DB_PATH = Path("healthcare_feedback.sqlite")
FeedbackRating = Literal["helpful", "unsupported", "unsafe", "needs_review"]


class FeedbackPayload(BaseModel):
    session_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    rating: FeedbackRating
    request_id: str | None = None
    answer: str | None = None
    comment: str | None = Field(default=None, max_length=1000)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRecord(FeedbackPayload):
    feedback_id: str
    user_id: str
    created_at: str


def init_feedback_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or DEFAULT_FEEDBACK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                request_id TEXT,
                rating TEXT NOT NULL,
                answer TEXT,
                comment TEXT,
                citations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user_created ON feedback(user_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user_session ON feedback(user_id, session_id)"
        )
    return path


def save_feedback(
    payload: FeedbackPayload,
    *,
    user_id: str,
    db_path: str | Path | None = None,
) -> FeedbackRecord:
    record = FeedbackRecord(
        **payload.model_dump(),
        feedback_id=str(uuid4()),
        user_id=user_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with sqlite3.connect(init_feedback_db(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                feedback_id,
                user_id,
                session_id,
                message_id,
                request_id,
                rating,
                answer,
                comment,
                citations_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.feedback_id,
                record.user_id,
                record.session_id,
                record.message_id,
                record.request_id,
                record.rating,
                record.answer,
                record.comment,
                json.dumps(record.citations, sort_keys=True),
                record.created_at,
            ),
        )
    return record


def feedback_summary(*, user_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    with sqlite3.connect(init_feedback_db(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT rating, COUNT(*)
            FROM feedback
            WHERE user_id = ?
            GROUP BY rating
            """,
            (user_id,),
        ).fetchall()

    counts = {rating: count for rating, count in rows}
    return {
        "total": sum(counts.values()),
        "counts": {
            "helpful": counts.get("helpful", 0),
            "unsupported": counts.get("unsupported", 0),
            "unsafe": counts.get("unsafe", 0),
            "needs_review": counts.get("needs_review", 0),
        },
    }
