from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path("backend/evaluation/golden_dataset.json")
EXPECTED_CATEGORIES = {
    "drug_interaction",
    "clinical_guideline",
    "symptom_info",
    "dosage_query",
    "contraindication",
}


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_golden_dataset_has_required_size_and_categories() -> None:
    records = load_dataset()

    assert len(records) == 50
    assert Counter(record["category"] for record in records) == {
        category: 10 for category in EXPECTED_CATEGORIES
    }


def test_golden_dataset_has_required_difficulty_mix() -> None:
    records = load_dataset()

    assert Counter(record["difficulty"] for record in records) == {
        "easy": 15,
        "medium": 25,
        "hard": 10,
    }


def test_golden_dataset_records_have_required_fields() -> None:
    records = load_dataset()
    ids = set()

    for record in records:
        assert set(record) == {
            "id",
            "category",
            "difficulty",
            "question",
            "ground_truth_answer",
            "relevant_context",
            "expected_citations",
        }
        assert record["id"] not in ids
        ids.add(record["id"])
        assert record["category"] in EXPECTED_CATEGORIES
        assert record["difficulty"] in {"easy", "medium", "hard"}
        assert record["question"].strip().endswith("?")
        assert len(record["ground_truth_answer"]) > 40
        assert len(record["relevant_context"]) > 40
        assert isinstance(record["expected_citations"], list)
        assert len(record["expected_citations"]) >= 2
