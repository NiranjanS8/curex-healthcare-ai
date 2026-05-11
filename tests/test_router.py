from __future__ import annotations

import pytest

from backend.agent.router import QueryIntent, classify_intent, route


class FakeClassifier:
    def __init__(self, result) -> None:
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_query_intent_normalises_entities() -> None:
    intent = QueryIntent(
        category="drug_interaction",
        confidence=0.91,
        entities=[" warfarin ", "Warfarin", "aspirin", ""],
    )

    assert intent.entities == ["warfarin", "aspirin"]


def test_classify_intent_uses_structured_classifier(capsys) -> None:
    classifier = FakeClassifier(
        {
            "category": "drug_interaction",
            "confidence": 0.86,
            "entities": ["warfarin", "aspirin"],
        }
    )

    intent = classify_intent("Can I take warfarin with aspirin?", classifier=classifier)

    assert intent.category == "drug_interaction"
    assert intent.confidence == 0.86
    assert intent.entities == ["warfarin", "aspirin"]
    assert classifier.messages[0][0] == "system"
    assert "Categories:" in classifier.messages[0][1]
    assert classifier.messages[1] == ("human", "Can I take warfarin with aspirin?")
    assert "Intent Classification" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("category", "expected_route"),
    [
        ("drug_info", "retriever"),
        ("symptom_diagnosis", "retriever"),
        ("clinical_guideline", "retriever"),
        ("drug_interaction", "tool_executor"),
        ("general_health", "retriever"),
        ("out_of_scope", "safety_check"),
    ],
)
def test_route_maps_categories(category, expected_route) -> None:
    intent = QueryIntent(category=category, confidence=0.75, entities=[])

    assert route(intent) == expected_route


def test_query_intent_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        QueryIntent(category="general_health", confidence=1.2, entities=[])
