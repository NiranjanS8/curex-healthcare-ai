from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.evaluation import ragas_runner


def _write_dataset(path: Path) -> None:
    records = [
        {
            "id": "drug-001",
            "category": "drug_interaction",
            "difficulty": "easy",
            "question": "What is the concern with warfarin and aspirin?",
            "ground_truth_answer": "They may increase bleeding risk.",
            "relevant_context": "Warfarin and aspirin may increase bleeding risk.",
            "expected_citations": ["source-1", "source-2"],
        },
        {
            "id": "guide-001",
            "category": "clinical_guideline",
            "difficulty": "medium",
            "question": "What lifestyle steps help hypertension?",
            "ground_truth_answer": "Diet and exercise are commonly recommended.",
            "relevant_context": "Hypertension guidance includes diet and physical activity.",
            "expected_citations": ["source-3", "source-4"],
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_load_golden_dataset_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path)

    records = ragas_runner.load_golden_dataset(path)

    assert len(records) == 2
    assert records[0]["id"] == "drug-001"


def test_run_evaluation_collects_pipeline_outputs_and_saves_csv(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "eval_results.csv"
    _write_dataset(dataset_path)
    questions: list[str] = []

    def fake_pipeline(question: str):
        questions.append(question)
        return {
            "answer": f"Answer for {question}",
            "contexts": [f"Context for {question}"],
        }

    def fake_evaluator(dataset, metrics):
        rows = []
        for index, row in enumerate(dataset):
            rows.append(
                {
                    "faithfulness": 0.8 - (index * 0.1),
                    "answer_relevancy": 0.9,
                    "context_precision": 0.7,
                    "context_recall": 0.6,
                }
            )
        assert len(metrics) == 4
        return pd.DataFrame(rows)

    results = ragas_runner.run_evaluation(
        dataset_path,
        pipeline=fake_pipeline,
        evaluator=fake_evaluator,
        output_path=output_path,
    )

    assert questions == [
        "What is the concern with warfarin and aspirin?",
        "What lifestyle steps help hypertension?",
    ]
    assert list(results["id"]) == ["drug-001", "guide-001"]
    assert list(results["faithfulness"]) == [0.8, 0.7000000000000001]
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert list(saved["category"]) == ["drug_interaction", "clinical_guideline"]
    output = capsys.readouterr().out
    assert "RAGAS Evaluation Results" in output
    assert "RAGAS Aggregate Summary" in output


def test_records_for_ragas_falls_back_to_relevant_context() -> None:
    golden = [
        {
            "id": "x",
            "category": "symptom_info",
            "difficulty": "hard",
            "question": "Question?",
            "ground_truth_answer": "Reference",
            "relevant_context": "Fallback context",
            "expected_citations": ["a", "b"],
        }
    ]

    records = ragas_runner._records_for_ragas(golden, [{"answer": "Answer", "contexts": []}])

    assert records[0]["user_input"] == "Question?"
    assert records[0]["response"] == "Answer"
    assert records[0]["retrieved_contexts"] == ["Fallback context"]
    assert records[0]["reference"] == "Reference"


def test_result_to_dataframe_preserves_record_metadata() -> None:
    records = [
        {
            "id": "x",
            "category": "dosage_query",
            "difficulty": "medium",
            "user_input": "Question?",
            "response": "Answer",
            "retrieved_contexts": ["Context"],
            "reference": "Reference",
            "expected_citations": ["a"],
        }
    ]
    metric_df = pd.DataFrame(
        [
            {
                "user_input": "Question?",
                "response": "Answer",
                "faithfulness": 0.75,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.6,
            }
        ]
    )

    result = ragas_runner._result_to_dataframe(metric_df, records)

    assert result.loc[0, "id"] == "x"
    assert result.loc[0, "faithfulness"] == 0.75
    assert "expected_citations" in result.columns
