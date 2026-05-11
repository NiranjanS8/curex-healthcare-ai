from __future__ import annotations

from backend.agent import graph
from backend.ingestion import indexer


def test_gemini_model_defaults_are_configured() -> None:
    assert indexer.DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2-preview"
    assert indexer.DEFAULT_EMBEDDING_DIMENSIONS == 3072
    assert graph.MODEL_PRICING_PER_1M_TOKENS["gemini-2.5-flash"] == {
        "input": 0.30,
        "output": 2.50,
    }
    assert graph.MODEL_PRICING_PER_1M_TOKENS["gemini-2.5-flash-lite"] == {
        "input": 0.10,
        "output": 0.40,
    }
