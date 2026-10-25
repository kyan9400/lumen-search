"""PDF text extraction with PDFium, falling back to OCR for scanned pages."""

from __future__ import annotations

import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium

from lumen.config import OcrMode
from lumen.ingest.errors import IngestError
from lumen.ingest.layout import (
    Heading,
    ends_broken,
    join_broken_word,
    lines_to_blocks,
    resolve_pdfium_breaks,
    strip_running_lines,
    vocabulary,
)
from lumen.ingest.ocr import OcrError, OcrPage, TesseractOcr
from lumen.models import Block, ParsedDocument
from lumen.textproc import detect_language

_WORD_CHARS = re.compile(r"[^\W_]", re.UNICODE)
# control characters PDFium leaves in text; a \x02 that ends a line (a hyphen at the line break)
# stays for lines_to_blocks to resolve
_CONTROL = re.compile(r"[\x00\x01\x03-\x08\x0b\x0c\x0e-\x1f]|\x02(?!$)")
_LEADING_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.?\s+")


def needs_ocr(text: str) -> bool:
    """True when a page has no usable text layer: an image-only scan or a broken text encoding."""
    letters = len(_WORD_CHARS.findall(text))
    if letters < 40:
        return True
    return text.count("�") > letters * 0.05


def _outline(pdf: pdfium.PdfDocument) -> list[tuple[str, int, int]]:
    """PDF bookmarks as ``(title, level, page index)``; titles in lower case with numbering removed."""
    entries = []
    for bookmark in pdf.get_toc():
        title = _LEADING_NUMBER.sub("", " ".join((bookmark.get_title() or "").split())).lower()
        dest = bookmark.get_dest()
        page = dest.get_index() if dest is not None else None
        if len(title) >= 3 and page is not None:
            entries.append((title, bookmark.level + 1, int(page)))
    return entries


def _page_outline(outline: list[tuple[str, int, int]], index: int) -> dict[str, int]:
    """Bookmarks that point at this page or a neighbour: a title elsewhere in the text is not a heading."""
    titles: dict[str, int] = {}
    for title, level, page in outline:
        if abs(page - index) <= 1:
            titles[title] = min(level, titles.get(title, level))
    return titles


def parse_pdf(
    path: Path,
    *,
    ocr: TesseractOcr | None,
    ocr_mode: OcrMode = "auto",
    dpi: int = 300,
    max_pages: int = 500,
    workers: int = 4,
) -> ParsedDocument:
    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise IngestError(f"cannot open PDF: {exc}") from exc
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise IngestError(f"PDF has {page_count} pages; the limit is {max_pages}")
        title = (pdf.get_metadata_value("Title") or "").strip() or path.stem
        outline = _outline(pdf)
        pages: list[list[str]] = []
        scanned: list[int] = []
        for index in range(page_count):
            page = pdf[index]
            text = page.get_textpage().get_text_bounded()
            pages.append(text.replace("\r\n", "\n").split("\n"))
            if ocr_mode == "always" or (ocr_mode == "auto" and needs_ocr(text)):
                scanned.append(index)
        warnings: list[str] = []
        confidences: dict[int, float] = {}
        if scanned and ocr is not None and ocr.available:
            done = []
            for index, result in _ocr_pages(pdf, scanned, ocr, dpi, workers):
                if isinstance(result, OcrError):
                    warnings.append(f"OCR failed on page {index + 1}: {result}")
                    continue
                pages[index] = result.lines
                confidences[index] = result.confidence
                done.append(index)
            scanned = done
        elif scanned:
            warnings.append(
                f"{len(scanned)} page(s) needed OCR but Tesseract is not installed; "
                "the PDF text layer was used where there is one"
            )
            scanned = []
    finally:
        pdf.close()

    blocks: list[Block] = []
    words = vocabulary(pages)
    pages = [[_CONTROL.sub("", resolve_pdfium_breaks(line, words)) for line in lines] for lines in pages]
    for index, lines in enumerate(strip_running_lines(pages)):
        confidence = confidences.get(index)
        for item in lines_to_blocks(lines, _page_outline(outline, index), words):
            if isinstance(item, Heading):
                blocks.append(Block(item.text, kind="heading", level=item.level, page=index + 1))
            else:
                blocks.append(Block(item, page=index + 1, ocr_confidence=confidence))
    blocks = join_across_pages(blocks, words)
    language = detect_language(" ".join(block.text for block in blocks[:400]))
    return ParsedDocument(
        title=title,
        media_type="pdf",
        blocks=blocks,
        pages=page_count,
        ocr_pages=[index + 1 for index in scanned],
        language=language,
        warnings=warnings,
    )


_SENTENCE_END = (".", "!", "?", ":", ";", "…", "»", '"', "”", ")")


def join_across_pages(blocks: list[Block], words: set[str]) -> list[Block]:
    """Rejoin a paragraph that a page break cut in two.

    Without this, the sentence that runs over the break becomes two fragments that no chunk holds whole.
    The joined paragraph keeps the page it starts on.
    """
    joined: list[Block] = []
    for block in blocks:
        previous = joined[-1] if joined else None
        if (
            previous is not None
            and previous.kind == block.kind == "text"
            and previous.page is not None
            and block.page == previous.page + 1
            and not previous.text.endswith(_SENTENCE_END)
            and block.text[:1].islower()
        ):
            if ends_broken(previous.text):
                previous.text = join_broken_word(previous.text, block.text, words)
            else:
                previous.text = f"{previous.text} {block.text}"
            continue
        joined.append(block)
    return joined


def _ocr_pages(
    pdf: pdfium.PdfDocument, indexes: list[int], ocr: TesseractOcr, dpi: int, workers: int
) -> list[tuple[int, OcrPage | OcrError]]:
    """Render pages one at a time (PDFium is not thread-safe) and OCR them in parallel.

    At most ``2 * workers`` rendered pages wait in memory: a 300 dpi A4 page is about 9 MB in grayscale.
    """
    results: list[tuple[int, OcrPage | OcrError]] = []
    pending: deque[tuple[int, Future[OcrPage]]] = deque()

    def collect() -> None:
        index, future = pending.popleft()
        try:
            results.append((index, future.result()))
        except OcrError as exc:  # one unreadable page should not fail the whole document
            results.append((index, exc))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index in indexes:
            image = pdf[index].render(scale=dpi / 72, grayscale=True).to_pil()
            pending.append((index, pool.submit(ocr.recognize, image)))
            while len(pending) >= 2 * workers:
                collect()
        while pending:
            collect()
    return results
