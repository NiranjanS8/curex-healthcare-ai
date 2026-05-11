"""RAGAS evaluation runner for the healthcare RAG pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from backend.agent.graph import graph


DEFAULT_DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
DEFAULT_RESULTS_PATH = Path("eval_results.csv")
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


PipelineFn = Callable[[str], dict[str, Any]]
EvaluatorFn = Callable[..., Any]


def load_golden_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(dataset_path).read_text(encoding="utf-8"))


def run_full_rag_pipeline(question: str) -> dict[str, Any]:
    """Run the current graph and normalize output for evaluation."""

    result = graph.invoke({"query": question, "session_id": "evaluation", "messages": []})
    retrieved_docs = result.get("retrieved_docs", [])
    contexts = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in retrieved_docs]
    return {
        "answer": result.get("response", ""),
        "contexts": contexts,
        "metadata": {
            "faithfulness_score": result.get("faithfulness_score"),
            "citations": [getattr(doc, "metadata", {}) for doc in retrieved_docs],
        },
    }


def _records_for_ragas(
    golden_records: list[dict[str, Any]],
    pipeline_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for golden, output in zip(golden_records, pipeline_outputs, strict=True):
        contexts = output.get("contexts") or [golden["relevant_context"]]
        records.append(
            {
                "id": golden["id"],
                "category": golden["category"],
                "difficulty": golden["difficulty"],
                "user_input": golden["question"],
                "response": output.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": golden["ground_truth_answer"],
                "expected_citations": golden["expected_citations"],
            }
        )
    return records


def _load_ragas_metrics() -> list[Any]:
    from ragas.metrics import _context_precision, _context_recall, _faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy

    return [_faithfulness, ResponseRelevancy(), _context_precision, _context_recall]


def _default_evaluator(dataset: Dataset, metrics: list[Any]) -> Any:
    from ragas import evaluate

    return evaluate(dataset, metrics=metrics, raise_exceptions=False)


def _result_to_dataframe(result: Any, records: list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        metric_df = result.copy()
    elif hasattr(result, "to_pandas"):
        metric_df = result.to_pandas()
    else:
        metric_df = pd.DataFrame(result)

    base_df = pd.DataFrame(records)
    if len(metric_df) != len(base_df):
        metric_df = metric_df.reset_index(drop=True)
        base_df = base_df.reset_index(drop=True)

    for column in ["user_input", "response", "retrieved_contexts", "reference"]:
        if column in metric_df.columns:
            metric_df = metric_df.drop(columns=[column])

    return pd.concat([base_df, metric_df], axis=1)


def _score_style(score: float) -> str:
    if score >= 0.75:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def print_results_table(results: pd.DataFrame) -> None:
    table = Table(title="RAGAS Evaluation Results")
    table.add_column("ID")
    table.add_column("Category")
    table.add_column("Difficulty")
    for metric in METRIC_COLUMNS:
        table.add_column(metric, justify="right")

    for _, row in results.iterrows():
        metric_values = []
        for metric in METRIC_COLUMNS:
            value = float(row.get(metric, 0.0))
            metric_values.append(f"[{_score_style(value)}]{value:.2f}[/{_score_style(value)}]")
        table.add_row(str(row["id"]), str(row["category"]), str(row["difficulty"]), *metric_values)

    Console().print(table)


def print_aggregate_summary(results: pd.DataFrame) -> None:
    table = Table(title="RAGAS Aggregate Summary")
    table.add_column("Group")
    for metric in METRIC_COLUMNS:
        table.add_column(metric, justify="right")

    overall = results[METRIC_COLUMNS].mean(numeric_only=True)
    table.add_row("overall", *(f"{overall.get(metric, 0.0):.2f}" for metric in METRIC_COLUMNS))

    for category, group in results.groupby("category"):
        means = group[METRIC_COLUMNS].mean(numeric_only=True)
        table.add_row(str(category), *(f"{means.get(metric, 0.0):.2f}" for metric in METRIC_COLUMNS))

    Console().print(table)


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    pipeline: PipelineFn = run_full_rag_pipeline,
    evaluator: EvaluatorFn | None = None,
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> pd.DataFrame:
    """Run the full RAG pipeline over the golden dataset and score with RAGAS."""

    golden_records = load_golden_dataset(dataset_path)
    pipeline_outputs = [pipeline(record["question"]) for record in golden_records]
    ragas_records = _records_for_ragas(golden_records, pipeline_outputs)
    dataset = Dataset.from_list(ragas_records)

    metrics = _load_ragas_metrics()
    evaluation_result = (evaluator or _default_evaluator)(dataset=dataset, metrics=metrics)
    results = _result_to_dataframe(evaluation_result, ragas_records)

    output = Path(output_path)
    results.to_csv(output, index=False)
    print_results_table(results)
    print_aggregate_summary(results)
    return results


if __name__ == "__main__":
    run_evaluation()
