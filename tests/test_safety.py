from __future__ import annotations

from backend.generation import safety
from backend.generation.prompts import DISCLAIMER


class FakeClassifier:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_pre_check_allows_in_scope_query(tmp_path, capsys) -> None:
    classifier = FakeClassifier(
        {"label": "in_scope", "reason": "medical question", "modified_query": "What is diabetes?"}
    )

    result = safety.pre_check("What is diabetes?", classifier=classifier, db_path=tmp_path / "safety.sqlite")

    assert result.safe is True
    assert result.modified_query == "What is diabetes?"
    assert classifier.messages[0][0] == "system"
    assert "Safety Check" in capsys.readouterr().out


def test_pre_check_blocks_off_topic_and_logs(tmp_path) -> None:
    db_path = tmp_path / "safety.sqlite"
    classifier = FakeClassifier({"label": "off_topic", "reason": "not health", "modified_query": ""})

    result = safety.pre_check("Write a car poem", classifier=classifier, db_path=db_path)

    assert result.safe is False
    assert result.reason == safety.OFF_TOPIC_MESSAGE
    rows = safety.get_safety_log(db_path=db_path)
    assert rows[0]["query"] == "Write a car poem"
    assert rows[0]["safe"] is False


def test_pre_check_blocks_harmful_heuristic(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = safety.pre_check("What is a lethal dose for overdose?", db_path=tmp_path / "safety.sqlite")

    assert result.safe is False
    assert "988" in result.reason


def test_post_check_softens_diagnosis_and_adds_disclaimer() -> None:
    response = "You have pneumonia based on these symptoms."

    checked = safety.post_check(response)

    assert "You have" not in checked
    assert "this may indicate pneumonia" in checked
    assert DISCLAIMER in checked


def test_post_check_does_not_duplicate_disclaimer() -> None:
    response = "Educational answer.\n\n" + DISCLAIMER

    checked = safety.post_check(response)

    assert checked.count(DISCLAIMER) == 1
