"""Plain data types shared by the parsers, the chunker, the store and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["text", "heading", "table", "row", "code"]


@dataclass
class Block:
    """A piece of extracted text and where it came from."""

    text: str
    kind: BlockKind = "text"
    page: int | None = None  # 1-based page of a PDF or image
    sheet: str | None = None  # spreadsheet sheet name
    row: int | None = None  # spreadsheet row, as numbered in Excel
    level: int = 1  # heading depth for kind == "heading" (1 = top)
    label: str | None = None  # the record a spreadsheet row describes, e.g. "AC-2 Account Management"
    ocr_confidence: float | None = None  # mean Tesseract word confidence, 0-100


@dataclass
class ParsedDocument:
    title: str
    media_type: str  # pdf, docx, xlsx, csv, markdown, text, html, image
    blocks: list[Block]
    pages: int | None = None
    ocr_pages: list[int] = field(default_factory=list)
    language: str = "und"
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    doc_id: str
    ordinal: int
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    ocr_confidence: float | None = None
    language: str = "und"
    id: int | None = None  # assigned by the store; also the vector id in FAISS

    def location_label(self) -> str:
        """Human-readable citation, e.g. ``p. 12 · 3.1.1. Passwords`` or ``Sheet 'Controls', row 41``."""
        parts: list[str] = []
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pp. {self.page_start}–{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        if self.sheet is not None:
            rows = ""
            if self.row_start is not None:
                same = self.row_end in (None, self.row_start)
                rows = f", row {self.row_start}" if same else f", rows {self.row_start}–{self.row_end}"
            parts.append(f"Sheet '{self.sheet}'{rows}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)

    def indexed_text(self) -> str:
        """What BM25 and the reranker see: the section path gives headings extra weight and context."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def embedding_text(self, title: str) -> str:
        """What the embedding model sees: the title too, which tells it what the passage is about."""
        return f"{title}\n{self.indexed_text()}"


@dataclass
class Document:
    id: str
    title: str
    filename: str
    media_type: str
    sha256: str
    language: str = "und"
    pages: int | None = None
    ocr_pages: int = 0
    chunks: int = 0
    source_url: str | None = None
    license: str | None = None
    created_at: str = ""
