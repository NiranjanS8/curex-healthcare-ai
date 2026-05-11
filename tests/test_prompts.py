from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from backend.generation.prompts import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    build_prompt,
    format_citations,
    format_context_docs,
)


def test_system_prompt_contains_required_safety_and_citation_rules() -> None:
    assert "Never diagnose" in SYSTEM_PROMPT
    assert "[Source: {title}, chunk {chunk_id}]" in SYSTEM_PROMPT
    assert DISCLAIMER in SYSTEM_PROMPT
    assert "Never suggest specific dosages" in SYSTEM_PROMPT


def test_format_context_docs_includes_chunk_metadata() -> None:
    doc = Document(
        page_content="Metformin is used for type 2 diabetes.",
        metadata={
            "title": "Diabetes Guideline",
            "chunk_id": "abc-123",
            "source": "guideline.pdf",
            "section": "abstract",
        },
    )

    rendered = format_context_docs([doc])

    assert "Title: Diabetes Guideline" in rendered
    assert "Chunk ID: abc-123" in rendered
    assert "Source: guideline.pdf" in rendered
    assert "Section: abstract" in rendered
    assert "Metformin is used" in rendered


def test_format_context_docs_handles_empty_context() -> None:
    assert format_context_docs([]) == "No retrieved context was provided."


def test_build_prompt_injects_query_context_and_memory() -> None:
    docs = [
        Document(
            page_content="Warfarin and aspirin may increase bleeding risk.",
            metadata={"title": "Drug Safety Review", "chunk_id": "drug-1"},
        )
    ]
    memory = {
        "system_message": SystemMessage(content="Known patient context: medications: warfarin"),
        "session_messages": [HumanMessage(content="I take warfarin.")],
    }

    prompt = build_prompt("Can I take aspirin?", docs, memory)
    messages = prompt.format_messages()

    assert messages[0].content == SYSTEM_PROMPT
    assert "Known patient context: medications: warfarin" in messages[1].content
    assert any(message.content == "I take warfarin." for message in messages)
    assert "Question:\nCan I take aspirin?" in messages[-1].content
    assert "Drug Safety Review" in messages[-1].content
    assert "drug-1" in messages[-1].content


def test_format_citations_returns_frontend_records() -> None:
    docs = [
        Document(
            page_content="  Aspirin can increase bleeding risk when combined with warfarin.  ",
            metadata={
                "chunk_id": "chunk-1",
                "title": "Interaction Review",
                "source_url": "https://example.test/review",
                "doc_type": "abstract",
                "date": "2025",
                "specialty": "cardiology",
                "source": "pubmed",
            },
        )
    ]

    citations = format_citations(docs)

    assert citations == [
        {
            "chunk_id": "chunk-1",
            "title": "Interaction Review",
            "source_url": "https://example.test/review",
            "excerpt": "Aspirin can increase bleeding risk when combined with warfarin.",
            "doc_type": "abstract",
            "date": "2025",
            "specialty": "cardiology",
            "metadata": docs[0].metadata,
        }
    ]


def test_format_citations_truncates_long_excerpt() -> None:
    doc = Document(page_content="word " * 200, metadata={"source": "source.pdf"})

    citation = format_citations([doc])[0]

    assert citation["chunk_id"] == "chunk-1"
    assert citation["title"] == "source.pdf"
    assert len(citation["excerpt"]) == 500
