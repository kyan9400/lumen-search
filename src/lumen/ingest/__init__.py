"""Parse a file into blocks of text with locations (page, section, sheet and row)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image, ImageSequence, UnidentifiedImageError

from lumen.config import OcrMode
from lumen.ingest.errors import IngestError
from lumen.ingest.layout import Heading, lines_to_blocks, vocabulary
from lumen.ingest.markup import parse_html, parse_markdown, parse_text
from lumen.ingest.ocr import OcrError, TesseractOcr
from lumen.ingest.office import parse_csv, parse_docx, parse_xlsx
from lumen.ingest.pdf import parse_pdf
from lumen.models import Block, ParsedDocument
from lumen.textproc import detect_language

__all__ = ["SUPPORTED_EXTENSIONS", "IngestError", "TesseractOcr", "parse_file", "sniff_type"]

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}

_IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM", b"RIFF")


def sniff_type(path: Path) -> str:
    """The file's type from its extension, checked against its first bytes.

    Uploads are untrusted: a ``.pdf`` that is really a ZIP, or a ``.docx`` that is not an Office package,
    is rejected before any parser touches it.
    """
    kind = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if kind is None:
        raise IngestError(f"unsupported file type: {path.suffix or 'no extension'}")
    with path.open("rb") as handle:
        head = handle.read(8192)
    if not head:
        raise IngestError("the file is empty")
    if kind == "pdf" and not head.startswith(b"%PDF-"):
        raise IngestError("not a PDF file")
    if kind in ("docx", "xlsx"):
        if not zipfile.is_zipfile(path):
            raise IngestError(f"not an Office Open XML ({kind}) file")
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
        expected = "word/document.xml" if kind == "docx" else "xl/workbook.xml"
        if "[Content_Types].xml" not in names or expected not in names:
            raise IngestError(f"not an Office Open XML ({kind}) file")
    if kind == "image" and not head.startswith(_IMAGE_SIGNATURES):
        raise IngestError("not a supported image (PNG, JPEG, TIFF, BMP or WebP)")
    if kind in ("markdown", "text", "html", "csv") and b"\x00" in head:
        raise IngestError("binary data in a text file")
    return kind


def parse_image(path: Path, ocr: TesseractOcr | None) -> ParsedDocument:
    if ocr is None or not ocr.available:
        raise IngestError("images need OCR, and Tesseract is not installed")
    try:
        image = Image.open(path)
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError(f"cannot open image: {exc}") from exc
    blocks: list[Block] = []
    pages = 0
    with image:
        for pages, frame in enumerate(ImageSequence.Iterator(image), start=1):  # multi-page TIFF
            try:
                result = ocr.recognize(frame.copy())
            except OcrError as exc:
                raise IngestError(f"OCR failed: {exc}") from exc
            for item in lines_to_blocks(result.lines, words=vocabulary([result.lines])):
                if isinstance(item, Heading):
                    blocks.append(Block(item.text, kind="heading", level=item.level, page=pages))
                else:
                    blocks.append(Block(item, page=pages, ocr_confidence=result.confidence))
    if not blocks:
        raise IngestError("no text was recognised in the image")
    return ParsedDocument(
        title=path.stem,
        media_type="image",
        blocks=blocks,
        pages=pages,
        ocr_pages=list(range(1, pages + 1)),
        language=detect_language(" ".join(b.text for b in blocks)),
    )


def parse_file(
    path: Path,
    *,
    ocr: TesseractOcr | None = None,
    ocr_mode: OcrMode = "auto",
    ocr_dpi: int = 300,
    ocr_workers: int = 4,
    max_pages: int = 500,
) -> ParsedDocument:
    kind = sniff_type(path)
    if kind == "pdf":
        document = parse_pdf(
            path, ocr=ocr, ocr_mode=ocr_mode, dpi=ocr_dpi, max_pages=max_pages, workers=ocr_workers
        )
    elif kind == "docx":
        document = parse_docx(path)
    elif kind == "xlsx":
        document = parse_xlsx(path)
    elif kind == "csv":
        document = parse_csv(path)
    elif kind == "markdown":
        document = parse_markdown(path)
    elif kind == "html":
        document = parse_html(path)
    elif kind == "image":
        document = parse_image(path, ocr)
    else:
        document = parse_text(path)
    if not any(block.kind != "heading" and block.text.strip() for block in document.blocks):
        raise IngestError("no text could be extracted" + (" (is OCR installed?)" if kind == "pdf" else ""))
    return document
