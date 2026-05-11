"""Prompt templates and citation formatting for healthcare responses."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


DISCLAIMER = (
    "This information is for educational purposes only. "
    "Always consult a qualified healthcare professional."
)

SYSTEM_PROMPT = f"""You are a medical information assistant. You provide information based strictly
on the retrieved medical literature. Never diagnose. Always cite sources.

Rules:
1. Think through the retrieved evidence internally, then provide only the final answer.
2. Every factual claim must be cited as [Source: {{title}}, chunk {{chunk_id}}].
3. If the retrieved context does not contain enough information, say so explicitly.
4. End every response with: "{DISCLAIMER}"
5. Never suggest specific dosages without citing a clinical source."""


def _metadata_value(doc: Document, key: str, default: str = "unknown") -> str:
    value = doc.metadata.get(key, default)
    return str(value) if value not in (None, "") else default


def format_context_docs(context_docs: list[Document]) -> str:
    """Render retrieved documents with chunk IDs for citation grounding."""

    if not context_docs:
        return "No retrieved context was provided."

    rendered_docs: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        title = _metadata_value(doc, "title", _metadata_value(doc, "source", f"source {index}"))
        chunk_id = _metadata_value(doc, "chunk_id", f"chunk-{index}")
        source = _metadata_value(doc, "source", "unknown")
        section = _metadata_value(doc, "section", "unknown")
        rendered_docs.append(
            "\n".join(
                [
                    f"[{index}] Title: {title}",
                    f"Chunk ID: {chunk_id}",
                    f"Source: {source}",
                    f"Section: {section}",
                    f"Text: {doc.page_content}",
                ]
            )
        )
    return "\n\n".join(rendered_docs)


def _render_memory(memory: dict[str, Any] | None) -> tuple[str, list[BaseMessage]]:
    if not memory:
        return "No known patient context.", []

    system_message = memory.get("system_message")
    if isinstance(system_message, BaseMessage):
        memory_context = str(system_message.content)
    elif system_message:
        memory_context = str(system_message)
    elif memory.get("long_term"):
        entries = [f"{key}: {value}" for key, value in sorted(memory["long_term"].items())]
        memory_context = "Known patient context: " + "; ".join(entries)
    else:
        memory_context = "No known patient context."

    session_messages = memory.get("session_messages") or []
    return memory_context, list(session_messages)


def build_prompt(
    query: str,
    context_docs: list[Document],
    memory: dict[str, Any] | None = None,
) -> ChatPromptTemplate:
    """Build a chat prompt with source context, memory, and citation instructions."""

    memory_context, session_messages = _render_memory(memory)
    context = format_context_docs(context_docs)
    escaped_system_prompt = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages(
        [
            ("system", escaped_system_prompt),
            ("system", "Memory context:\n{memory_context}"),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question:\n{query}\n\nRetrieved context:\n{context}\n\n"
                "Answer using only the retrieved context. Include citations inline.",
            ),
        ]
    ).partial(
        query=query,
        context=context,
        memory_context=memory_context,
        history=session_messages,
    )


def format_citations(docs: list[Document]) -> list[dict[str, Any]]:
    """Return frontend-ready citation records for retrieved documents."""

    citations: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        chunk_id = _metadata_value(doc, "chunk_id", f"chunk-{index}")
        title = _metadata_value(doc, "title", _metadata_value(doc, "source", f"Source {index}"))
        excerpt = " ".join(doc.page_content.split())
        citations.append(
            {
                "chunk_id": chunk_id,
                "title": title,
                "source_url": metadata.get("source_url") or metadata.get("url") or metadata.get("source"),
                "excerpt": excerpt[:500],
                "doc_type": metadata.get("doc_type"),
                "date": metadata.get("date"),
                "specialty": metadata.get("specialty"),
                "metadata": metadata,
            }
        )
    return citations
