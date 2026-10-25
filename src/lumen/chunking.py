"""Structure-aware chunking.

Sentences are packed into chunks of about ``max_words`` words. A chunk never crosses a section heading
or a spreadsheet row, so every chunk has one precise citation, and consecutive chunks of a section
overlap by a sentence or two so an answer split across a boundary is still found whole.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise

from lumen.models import Chunk, ParsedDocument
from lumen.textproc import detect_language, split_sentences


@dataclass
class _Unit:
    text: str
    words: int
    section: str | None
    record: int  # a new value per section or spreadsheet row: units of different records never mix
    line: int  # source paragraph (or row line): a new one starts on a new line in the chunk text
    page: int | None
    sheet: str | None
    row: int | None
    ocr_confidence: float | None


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap, step)]


def _units(document: ParsedDocument, max_words: int, overlap: int) -> Iterator[_Unit]:
    stack: list[tuple[int, str]] = []
    record = 0
    for index, block in enumerate(document.blocks):
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            record += 1
            continue
        section = " › ".join(text for _, text in stack) or None
        if block.kind == "row":
            record += 1
            section = block.label
            lines = block.text.split("\n")  # one "Header: value" line per filled cell
        else:
            lines = [block.text]
        for number, line in enumerate(lines):
            if block.kind in ("table", "code"):
                pieces = _split_long(line, max_words, overlap)
            else:
                pieces = [p for s in split_sentences(line) for p in _split_long(s, max_words, overlap)]
            for piece in pieces:
                yield _Unit(
                    text=piece,
                    words=len(piece.split()),
                    section=section,
                    record=record,
                    line=index * 10_000 + number,
                    page=block.page,
                    sheet=block.sheet,
                    row=block.row,
                    ocr_confidence=block.ocr_confidence,
                )


def _tail(units: list[_Unit], overlap_words: int) -> list[_Unit]:
    """The last whole sentences of a chunk, up to ``overlap_words`` words, to start the next one."""
    tail: list[_Unit] = []
    size = 0
    for unit in reversed(units):
        if size + unit.words > overlap_words:
            break
        tail.insert(0, unit)
        size += unit.words
    return tail if len(tail) < len(units) else []


def _make_chunk(units: list[_Unit], doc_id: str, ordinal: int, fallback_language: str) -> Chunk:
    text = units[0].text
    for previous, unit in pairwise(units):
        text += (" " if unit.line == previous.line else "\n") + unit.text
    pages = [u.page for u in units if u.page is not None]
    rows = [u.row for u in units if u.row is not None]
    confidences = [u.ocr_confidence for u in units if u.ocr_confidence is not None]
    language = detect_language(text)
    return Chunk(
        doc_id=doc_id,
        ordinal=ordinal,
        text=text,
        section=units[0].section,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        sheet=units[0].sheet,
        row_start=min(rows) if rows else None,
        row_end=max(rows) if rows else None,
        ocr_confidence=round(sum(confidences) / len(confidences), 1) if confidences else None,
        language=fallback_language if language == "und" else language,
    )


def chunk_document(
    document: ParsedDocument, doc_id: str, *, max_words: int = 180, overlap_words: int = 30
) -> list[Chunk]:
    overlap_words = min(overlap_words, max_words // 2)
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    size = 0
    for unit in _units(document, max_words, overlap_words):
        if current:
            boundary = unit.record != current[-1].record
            if boundary or size + unit.words > max_words:
                chunks.append(_make_chunk(current, doc_id, len(chunks), document.language))
                current = [] if boundary else _tail(current, overlap_words)
                size = sum(u.words for u in current)
                if size + unit.words > max_words:
                    current, size = [], 0
        current.append(unit)
        size += unit.words
    if current:
        chunks.append(_make_chunk(current, doc_id, len(chunks), document.language))
    return chunks
