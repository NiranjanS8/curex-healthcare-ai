from __future__ import annotations

import pytest

from backend.agent import tools


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


def test_calculate_bmi_returns_category_and_healthy_range() -> None:
    result = tools.calculate_bmi.invoke({"weight_kg": 70, "height_cm": 175})

    assert result["bmi"] == 22.9
    assert result["category"] == "healthy_weight"
    assert result["healthy_range"] == {"min_weight_kg": 56.7, "max_weight_kg": 76.3}


def test_calculate_bmi_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        tools.calculate_bmi.invoke({"weight_kg": 0, "height_cm": 175})


def test_check_drug_interactions_resolves_and_fetches_pairs(monkeypatch) -> None:
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if url.endswith("/rxcui.json") and params["name"] == "warfarin":
            return FakeResponse({"idGroup": {"rxnormId": ["11289"]}})
        if url.endswith("/rxcui.json") and params["name"] == "aspirin":
            return FakeResponse({"idGroup": {"rxnormId": ["1191"]}})
        if url.endswith("/interaction/list.json"):
            assert params == {"rxcuis": "11289+1191"}
            return FakeResponse(
                {
                    "fullInteractionTypeGroup": [
                        {
                            "fullInteractionType": [
                                {
                                    "interactionPair": [
                                        {
                                            "severity": "high",
                                            "description": "Increased bleeding risk.",
                                            "interactionConcept": [
                                                {"minConceptItem": {"rxcui": "11289", "name": "warfarin"}},
                                                {"minConceptItem": {"rxcui": "1191", "name": "aspirin"}},
                                            ],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(tools.requests, "get", fake_get)

    result = tools.check_drug_interactions.invoke({"drug_names": ["warfarin", "aspirin"]})

    assert result["resolved"] == {"warfarin": "11289", "aspirin": "1191"}
    assert result["unresolved"] == []
    assert result["pairs"] == [
        {
            "drug_a": "warfarin",
            "drug_b": "aspirin",
            "severity": "high",
            "description": "Increased bleeding risk.",
        }
    ]
    assert all(call[2] == tools.REQUEST_TIMEOUT_SECONDS for call in calls)


def test_check_drug_interactions_reports_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(tools, "resolve_rxcui", lambda drug_name: None)

    result = tools.check_drug_interactions.invoke({"drug_names": ["unknown"]})

    assert result["pairs"] == []
    assert result["unresolved"] == ["unknown"]
    assert "At least two drugs" in result["message"]


def test_lookup_icd10_returns_top_matches(monkeypatch) -> None:
    def fake_get(url, params=None, timeout=None):
        assert "terms=type%202%20diabetes" in url
        return FakeResponse(
            [
                2,
                ["E11.9", "E11.65"],
                None,
                [["E11.9", "Type 2 diabetes mellitus without complications"], ["E11.65", "Type 2 diabetes mellitus with hyperglycemia"]],
            ]
        )

    monkeypatch.setattr(tools.requests, "get", fake_get)

    result = tools.lookup_icd10.invoke({"condition": "type 2 diabetes"})

    assert result["condition"] == "type 2 diabetes"
    assert result["matches"][0] == {
        "code": "E11.9",
        "description": "Type 2 diabetes mellitus without complications",
    }


def test_tools_registry_and_lookup() -> None:
    names = {registered_tool.name for registered_tool in tools.TOOLS}

    assert names == {"check_drug_interactions", "calculate_bmi", "lookup_icd10"}
    assert tools.run_tool_by_name("calculate_bmi", {"weight_kg": 70, "height_cm": 175})["bmi"] == 22.9
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.run_tool_by_name("missing", {})
