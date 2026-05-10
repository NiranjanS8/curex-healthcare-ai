"""Medical entity extraction and semantic chunking."""

from __future__ import annotations

import re
import statistics
import uuid
from functools import lru_cache
from typing import Iterable

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table


SECTION_HEADING_RE = re.compile(
    r"^\s*(abstract|introduction|methods?|results?|discussion|conclusions?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TOKEN_RE = re.compile(r"\S+")
DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|kg|ml|l|iu|units?|tablets?|capsules?)"
    r"(?:/(?:kg|day|dose|ml))?\b",
    re.IGNORECASE,
)

CHEMICAL_TERMS = {
    "acetaminophen",
    "aspirin",
    "atorvastatin",
    "ibuprofen",
    "insulin",
    "metformin",
    "warfarin",
}
DISEASE_TERMS = {
    "asthma",
    "cancer",
    "diabetes",
    "hypertension",
    "infection",
    "influenza",
    "pneumonia",
}


def _normalise_section(section: str | None) -> str:
    if not section:
        return "body"
    value = section.strip().lower()
    if value == "conclusions":
        return "conclusion"
    if value == "method":
        return "methods"
    if value == "result":
        return "results"
    return value


@lru_cache(maxsize=1)
def get_nlp():
    """Load the scispaCy model, falling back to a blank pipeline for local tests."""

    import spacy

    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        return spacy.blank("en")


def _classify_entity(text: str, source_label: str) -> str | None:
    label = source_label.upper()
    lowered = text.lower()

    if label in {"DISEASE", "CHEMICAL", "DOSAGE"}:
        return label
    if DOSAGE_RE.search(text):
        return "DOSAGE"
    if lowered in CHEMICAL_TERMS or lowered.endswith(("cillin", "mab", "pril", "sartan", "statin")):
        return "CHEMICAL"
    if lowered in DISEASE_TERMS or lowered.endswith(("itis", "osis", "emia", "pathy")):
        return "DISEASE"
    return None


def _dedupe_entities(entities: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for entity in entities:
        key = (entity["text"].lower(), entity["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped


def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract medical entities as DISEASE, CHEMICAL, and DOSAGE labels."""

    nlp = get_nlp()
    doc = nlp(text)
    entities: list[dict[str, str]] = []

    for ent in getattr(doc, "ents", []):
        label = _classify_entity(ent.text, ent.label_)
        if label:
            entities.append({"text": ent.text, "label": label})

    for match in DOSAGE_RE.finditer(text):
        entities.append({"text": match.group(0), "label": "DOSAGE"})

    return _dedupe_entities(entities)


def _split_sections(text: str, default_section: str = "body") -> list[tuple[str, str, int]]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        leading_offset = len(text) - len(text.lstrip())
        return [(_normalise_section(default_section), stripped, leading_offset)] if stripped else []

    sections: list[tuple[str, str, int]] = []
    leading_text = text[: matches[0].start()].strip()
    if leading_text:
        leading_offset = len(text[: matches[0].start()]) - len(text[: matches[0].start()].lstrip())
        sections.append((_normalise_section(default_section), leading_text, leading_offset))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_section = text[start:end]
        section_text = raw_section.strip()
        if section_text:
            char_offset = start + len(raw_section) - len(raw_section.lstrip())
            sections.append((_normalise_section(match.group(1)), section_text, char_offset))

    return sections


def _window_text(section_text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str, int]]:
    tokens = list(TOKEN_RE.finditer(section_text))
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [(section_text, 0)]

    windows: list[tuple[str, int]] = []
    step = max(1, max_tokens - overlap_tokens)
    start_token = 0

    while start_token < len(tokens):
        end_token = min(start_token + max_tokens, len(tokens))
        start_char = tokens[start_token].start()
        end_char = tokens[end_token - 1].end()
        windows.append((section_text[start_char:end_char], start_char))
        if end_token == len(tokens):
            break
        start_token += step

    return windows


def chunk_document(
    doc: Document,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Document]:
    """Split a source document into section-aware sliding-window chunks."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    default_section = str(doc.metadata.get("section", "body"))
    chunks: list[Document] = []

    for section, section_text, section_offset in _split_sections(doc.page_content, default_section):
        for window_text, window_offset in _window_text(section_text, max_tokens, overlap_tokens):
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "section": section,
                    "char_offset": section_offset + window_offset,
                    "entities": extract_entities(window_text),
                }
            )
            chunks.append(Document(page_content=window_text, metadata=metadata))

    return chunks


def _render_distribution(console: Console, chunks: list[Document]) -> None:
    section_counts: dict[str, int] = {}
    for chunk in chunks:
        section = str(chunk.metadata.get("section", "body"))
        section_counts[section] = section_counts.get(section, 0) + 1

    if not section_counts:
        return

    max_count = max(section_counts.values())
    console.print("[bold]Chunk distribution by section[/bold]")
    for section, count in sorted(section_counts.items()):
        width = max(1, round((count / max_count) * 32))
        console.print(f"{section:>12} | {'█' * width} {count}")


def chunk_all(docs: list[Document]) -> list[Document]:
    """Chunk all documents and print summary statistics."""

    console = Console()
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))

    token_lengths = [len(TOKEN_RE.findall(chunk.page_content)) for chunk in chunks]
    entity_hits = sum(1 for chunk in chunks if chunk.metadata.get("entities"))
    avg_length = statistics.mean(token_lengths) if token_lengths else 0
    entity_hit_rate = (entity_hits / len(chunks)) if chunks else 0

    table = Table(title="Chunking Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Source documents", str(len(docs)))
    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Average length", f"{avg_length:.1f} tokens")
    table.add_row("Entity hit rate", f"{entity_hit_rate:.0%}")
    console.print(table)
    _render_distribution(console, chunks)

    return chunks
