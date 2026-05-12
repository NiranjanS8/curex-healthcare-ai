from __future__ import annotations

from backend.review import FeedbackPayload, feedback_summary, save_feedback


def test_save_feedback_and_summary_are_user_scoped(tmp_path) -> None:
    db_path = tmp_path / "feedback.sqlite"
    payload = FeedbackPayload(
        session_id="session-1",
        message_id="assistant-1",
        request_id="req-1",
        rating="needs_review",
        answer="Answer text.",
        citations=[{"chunkId": "chunk-1"}],
    )

    record = save_feedback(payload, user_id="user-1", db_path=db_path)

    assert record.feedback_id
    assert record.user_id == "user-1"
    assert feedback_summary(user_id="user-1", db_path=db_path)["counts"]["needs_review"] == 1
    assert feedback_summary(user_id="user-2", db_path=db_path)["total"] == 0
