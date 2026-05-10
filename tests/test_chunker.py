from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from backend.ingestion import chunker


@dataclass
class FakeEntity:
    text: str
    label_: str


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNlp:
    def __call__(self, text: str):
        ents = []
        if "diabetes" in text.lower():
            ents.append(FakeEntity("diabetes", "DISEASE"))
        if "metformin" in text.lower():
            ents.append(FakeEntity("metformin", "CHEMICAL"))
        return FakeDoc(ents)


def test_extract_entities_returns_medical_labels(monkeypatch) -> None:
    monkeypatch.setattr(chunker, "get_nlp", lambda: FakeNlp())

    entities = chunker.extract_entities("Metformin 500 mg is used for diabetes.")

    assert {"text": "metformin", "label": "CHEMICAL"} in entities
    assert {"text": "diabetes", "label": "DISEASE"} in entities
    assert {"text": "500 mg", "label": "DOSAGE"} in entities


def test_chunk_document_splits_sections_and_attaches_metadata(monkeypatch) -> None:
    monkeypatch.setattr(chunker, "get_nlp", lambda: FakeNlp())
    doc = Document(
        page_content=(
            "Clinical Note\n"
            "Abstract\n"
            "Metformin 500 mg supports glycemic control.\n"
            "Methods\n"
            "Patients with diabetes were reviewed."
        ),
        metadata={"source": "fixture", "title": "Clinical Note"},
    )

    chunks = chunker.chunk_document(doc, max_tokens=8, overlap_tokens=2)

    assert {chunk.metadata["section"] for chunk in chunks} >= {"body", "abstract", "methods"}
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert all(isinstance(chunk.metadata["char_offset"], int) for chunk in chunks)
    abstract_chunk = next(chunk for chunk in chunks if chunk.metadata["section"] == "abstract")
    assert {"text": "500 mg", "label": "DOSAGE"} in abstract_chunk.metadata["entities"]


def test_chunk_document_applies_sliding_window(monkeypatch) -> None:
    monkeypatch.setattr(chunker, "get_nlp", lambda: FakeNlp())
    doc = Document(
        page_content=" ".join(f"token{i}" for i in range(14)),
        metadata={"source": "fixture", "section": "discussion"},
    )

    chunks = chunker.chunk_document(doc, max_tokens=6, overlap_tokens=2)

    assert len(chunks) == 3
    assert chunks[0].page_content.split()[-2:] == chunks[1].page_content.split()[:2]
    assert chunks[0].metadata["section"] == "discussion"
    assert chunks[1].metadata["char_offset"] > chunks[0].metadata["char_offset"]


def test_chunk_all_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(chunker, "get_nlp", lambda: FakeNlp())
    docs = [
        Document(
            page_content="Abstract\nMetformin 500 mg is used for diabetes.",
            metadata={"source": "fixture"},
        )
    ]

    chunks = chunker.chunk_all(docs)

    captured = capsys.readouterr()
    assert len(chunks) == 1
    assert "Chunking Summary" in captured.out
    assert "Chunk distribution by section" in captured.out
