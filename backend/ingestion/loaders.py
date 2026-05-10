"""Load healthcare source documents into LangChain Documents."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber
from Bio import Entrez
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


SECTION_HEADING_RE = re.compile(
    r"^\s*(abstract|introduction|methods?|results?|discussion|conclusions?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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


def _split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [("body", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    leading_text = text[: matches[0].start()].strip()
    if leading_text:
        sections.append(("body", leading_text))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((_normalise_section(match.group(1)), section_text))

    return sections


def _extract_title(path: Path, first_page_text: str) -> str:
    for line in first_page_text.splitlines():
        candidate = line.strip()
        if candidate and not SECTION_HEADING_RE.match(candidate):
            return candidate[:200]
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _file_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()


def load_pdf(path: str) -> list[Document]:
    """Extract a clinical PDF into page/section-level Documents."""

    pdf_path = Path(path)
    documents: list[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = ""
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if page_index == 1:
                first_page_text = text

            for section, section_text in _split_sections(text):
                documents.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            "source": str(pdf_path),
                            "doc_type": "clinical_pdf",
                            "specialty": "unknown",
                            "date": _file_date(pdf_path),
                            "title": _extract_title(pdf_path, first_page_text or text),
                            "page": page_index,
                            "section": section,
                        },
                    )
                )

    return documents


def _article_id(article: dict[str, Any]) -> str:
    citation = article.get("MedlineCitation", {})
    pmid = citation.get("PMID", "")
    return str(pmid)


def _article_title(article: dict[str, Any]) -> str:
    return str(
        article.get("MedlineCitation", {})
        .get("Article", {})
        .get("ArticleTitle", "")
    )


def _article_abstract(article: dict[str, Any]) -> str:
    abstract = (
        article.get("MedlineCitation", {})
        .get("Article", {})
        .get("Abstract", {})
        .get("AbstractText", [])
    )
    return "\n".join(str(part) for part in abstract if str(part).strip())


def _article_authors(article: dict[str, Any]) -> list[str]:
    author_list = (
        article.get("MedlineCitation", {})
        .get("Article", {})
        .get("AuthorList", [])
    )
    authors: list[str] = []
    for author in author_list:
        last_name = str(author.get("LastName", "")).strip()
        fore_name = str(author.get("ForeName", "")).strip()
        collective = str(author.get("CollectiveName", "")).strip()
        if collective:
            authors.append(collective)
        elif last_name or fore_name:
            authors.append(" ".join(part for part in [fore_name, last_name] if part))
    return authors


def _article_journal(article: dict[str, Any]) -> str:
    return str(
        article.get("MedlineCitation", {})
        .get("Article", {})
        .get("Journal", {})
        .get("Title", "")
    )


def _article_date(article: dict[str, Any]) -> str:
    pub_date = (
        article.get("MedlineCitation", {})
        .get("Article", {})
        .get("Journal", {})
        .get("JournalIssue", {})
        .get("PubDate", {})
    )
    year = str(pub_date.get("Year", "")).strip()
    month = str(pub_date.get("Month", "")).strip()
    day = str(pub_date.get("Day", "")).strip()
    return "-".join(part for part in [year, month, day] if part)


def load_pubmed(pmids: list[str]) -> list[Document]:
    """Fetch PubMed abstracts by PMID using Bio.Entrez."""

    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    try:
        records = Entrez.read(handle)
    finally:
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    documents: list[Document] = []
    for article in records.get("PubmedArticle", []):
        abstract = _article_abstract(article)
        if not abstract:
            continue

        pmid = _article_id(article)
        documents.append(
            Document(
                page_content=abstract,
                metadata={
                    "source": "pubmed",
                    "pmid": pmid,
                    "title": _article_title(article),
                    "journal": _article_journal(article),
                    "date": _article_date(article),
                    "doc_type": "abstract",
                    "authors": _article_authors(article),
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            )
        )

    return documents


def load_all(config: dict[str, Any]) -> list[Document]:
    """Load all configured PDF and PubMed sources with progress output."""

    console = Console()
    pdf_paths = [str(path) for path in config.get("pdf_paths", [])]
    pmids = [str(pmid) for pmid in config.get("pmids", [])]

    if email := config.get("entrez_email"):
        Entrez.email = str(email)

    documents: list[Document] = []
    total_tasks = len(pdf_paths) + (1 if pmids else 0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading healthcare sources", total=total_tasks)

        for pdf_path in pdf_paths:
            progress.update(task, description=f"Loading PDF: {pdf_path}")
            documents.extend(load_pdf(pdf_path))
            progress.advance(task)

        if pmids:
            progress.update(task, description=f"Loading PubMed: {len(pmids)} PMID(s)")
            documents.extend(load_pubmed(pmids))
            progress.advance(task)

    console.print(f"[green]Loaded {len(documents)} document section(s).[/green]")
    return documents
