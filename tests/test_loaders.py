from __future__ import annotations

from pathlib import Path

from backend.ingestion import loaders


def _pdf_object(object_id: int, body: str) -> str:
    return f"{object_id} 0 obj\n{body}\nendobj\n"


def _write_sample_pdf(path: Path, lines: list[str]) -> None:
    text_ops = ["BT", "/F1 12 Tf", "72 740 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_ops.append(f"({escaped}) Tj")
        text_ops.append("T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops)

    objects = [
        _pdf_object(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        _pdf_object(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        _pdf_object(
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        ),
        _pdf_object(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        _pdf_object(5, f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"),
    ]

    output = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(output.encode("latin-1")))
        output += obj
    xref_offset = len(output.encode("latin-1"))
    output += f"xref\n0 {len(objects) + 1}\n"
    output += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        output += f"{offset:010d} 00000 n \n"
    output += (
        "trailer\n"
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        "startxref\n"
        f"{xref_offset}\n"
        "%%EOF\n"
    )
    path.write_bytes(output.encode("latin-1"))


def test_load_pdf_extracts_sections_and_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "diabetes_guideline.pdf"
    _write_sample_pdf(
        pdf_path,
        [
            "Diabetes Care Guideline",
            "Abstract",
            "Metformin is commonly used in type 2 diabetes care.",
            "Methods",
            "Clinical guideline evidence was reviewed.",
            "Conclusion",
            "Patients should consult qualified clinicians.",
        ],
    )

    docs = loaders.load_pdf(str(pdf_path))

    assert {doc.metadata["section"] for doc in docs} >= {"body", "abstract", "methods", "conclusion"}
    abstract_doc = next(doc for doc in docs if doc.metadata["section"] == "abstract")
    assert "Metformin" in abstract_doc.page_content
    assert abstract_doc.metadata["source"] == str(pdf_path)
    assert abstract_doc.metadata["doc_type"] == "clinical_pdf"
    assert abstract_doc.metadata["title"] == "Diabetes Care Guideline"
    assert abstract_doc.metadata["page"] == 1
    assert abstract_doc.metadata["specialty"] == "unknown"
    assert abstract_doc.metadata["date"]


def test_load_pubmed_fetches_abstracts_with_metadata(monkeypatch) -> None:
    class FakeHandle:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake_handle = FakeHandle()

    def fake_efetch(**kwargs):
        assert kwargs["db"] == "pubmed"
        assert kwargs["id"] == "12345"
        assert kwargs["retmode"] == "xml"
        return fake_handle

    def fake_read(handle):
        assert handle is fake_handle
        return {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "12345",
                        "Article": {
                            "ArticleTitle": "Aspirin and warfarin bleeding risk",
                            "Abstract": {
                                "AbstractText": [
                                    "Concomitant aspirin and warfarin may increase bleeding risk."
                                ]
                            },
                            "AuthorList": [{"ForeName": "Ada", "LastName": "Lovelace"}],
                            "Journal": {
                                "Title": "Journal of Clinical Safety",
                                "JournalIssue": {"PubDate": {"Year": "2024", "Month": "05"}},
                            },
                        },
                    }
                }
            ]
        }

    monkeypatch.setattr(loaders.Entrez, "efetch", fake_efetch)
    monkeypatch.setattr(loaders.Entrez, "read", fake_read)

    docs = loaders.load_pubmed(["12345"])

    assert len(docs) == 1
    assert "bleeding risk" in docs[0].page_content
    assert docs[0].metadata == {
        "source": "pubmed",
        "pmid": "12345",
        "title": "Aspirin and warfarin bleeding risk",
        "journal": "Journal of Clinical Safety",
        "date": "2024-05",
        "doc_type": "abstract",
        "authors": ["Ada Lovelace"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
    }


def test_load_all_orchestrates_sources(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf_doc = loaders.Document(page_content="pdf", metadata={"source": str(pdf_path)})
    pubmed_doc = loaders.Document(page_content="pubmed", metadata={"source": "pubmed"})

    monkeypatch.setattr(loaders, "load_pdf", lambda path: [pdf_doc])
    monkeypatch.setattr(loaders, "load_pubmed", lambda pmids: [pubmed_doc])

    docs = loaders.load_all(
        {
            "pdf_paths": [str(pdf_path)],
            "pmids": ["12345"],
            "entrez_email": "engineer@example.com",
        }
    )

    assert docs == [pdf_doc, pubmed_doc]
    assert loaders.Entrez.email == "engineer@example.com"
