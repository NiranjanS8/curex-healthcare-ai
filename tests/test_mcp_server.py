from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from backend import mcp_server


class FakeRetriever:
    def invoke(self, query: str):
        return [
            Document(
                page_content=f"retrieved evidence for {query}",
                metadata={
                    "chunk_id": "chunk-1",
                    "title": "Clinical Guideline",
                    "source": "guideline.pdf",
                    "retrieval_score": 0.42,
                },
            )
        ]


def test_bmi_calculator_tool_wrapper() -> None:
    result = mcp_server.bmi_calculator(weight_kg=70, height_cm=175)

    assert result["bmi"] == 22.9
    assert result["category"] == "healthy_weight"
    assert result["healthy_range"]["min_weight_kg"] == 56.7


def test_retrieval_search_uses_injected_retriever() -> None:
    result = mcp_server.retrieval_search("hypertension guideline", retriever=FakeRetriever())

    assert result["backend"] == "configured_retriever"
    assert result["results"][0]["chunk_id"] == "chunk-1"
    assert result["results"][0]["score"] == 0.42
    assert "hypertension guideline" in result["results"][0]["content"]


def test_mcp_server_registers_expected_tools() -> None:
    server = mcp_server.create_mcp_server()

    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert {
        "drug_interaction_lookup",
        "icd10_lookup",
        "bmi_calculator",
        "retrieval_search",
    }.issubset(names)


def test_mcp_server_can_call_bmi_tool() -> None:
    server = mcp_server.create_mcp_server()

    content_blocks, structured_result = asyncio.run(
        server.call_tool("bmi_calculator", {"weight_kg": 70, "height_cm": 175})
    )

    assert structured_result["bmi"] == 22.9
    assert "healthy_weight" in content_blocks[0].text
