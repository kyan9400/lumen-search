"""Markdown, plain text and HTML parsing."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from lumen.ingest.errors import IngestError
from lumen.ingest.layout import detect_heading
from lumen.models import Block, ParsedDocument
from lumen.textproc import detect_language

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_FENCE = re.compile(r"^(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_AUTOLINK = re.compile(r"<(https?://[^>]+)>")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_CODE_SPAN = re.compile(r"`([^`]*)`")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError("the file is not UTF-8 or Windows-1251 text")


def strip_inline_markdown(text: str) -> str:
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _REF_LINK.sub(r"\1", text)
    text = _AUTOLINK.sub(r"\1", text)
    text = _CODE_SPAN.sub(r"\1", text)
    text = _TAG.sub("", text)
    for _ in range(2):  # nested emphasis such as ***bold italic***
        text = _EMPHASIS.sub(r"\2", text)
    return text.replace("\\", "")


def parse_markdown(path: Path) -> ParsedDocument:
    lines = _read_text(path).splitlines()
    blocks: list[Block] = []
    paragraph: list[str] = []
    title: str | None = None

    def flush() -> None:
        if paragraph:
            text = strip_inline_markdown(" ".join(line.strip() for line in paragraph)).strip()
            if text:
                blocks.append(Block(text))
            paragraph.clear()

    def heading(text: str, level: int) -> None:
        nonlocal title
        flush()
        text = strip_inline_markdown(text).strip()
        if text:
            title = title or text
            blocks.append(Block(text, kind="heading", level=level))

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _FENCE.match(stripped):
            flush()
            fence, code = stripped[:3], []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index])
                index += 1
            if any(c.strip() for c in code):
                blocks.append(Block("\n".join(code).strip("\n"), kind="code"))
        elif match := _ATX.match(stripped):
            heading(match.group(2), len(match.group(1)))
        elif (
            stripped
            and index + 1 < len(lines)
            and _SETEXT.match(lines[index + 1].strip())
            and not paragraph
            and not _LIST_ITEM.match(line)
        ):
            heading(stripped, 1 if lines[index + 1].strip().startswith("=") else 2)
            index += 1
        elif stripped.startswith("|"):
            flush()
            if not _TABLE_RULE.match(stripped):
                cells = [strip_inline_markdown(c).strip() for c in stripped.strip("|").split("|")]
                blocks.append(Block(" | ".join(c for c in cells if c), kind="table"))
        elif not stripped:
            flush()
        elif _LIST_ITEM.match(line):
            flush()
            paragraph.append(_LIST_ITEM.sub("", line))
        else:
            paragraph.append(line.lstrip("> "))
        index += 1
    flush()
    if not blocks:
        raise IngestError("the Markdown file has no text")
    return ParsedDocument(
        title=title or path.stem,
        media_type="markdown",
        blocks=blocks,
        language=detect_language(" ".join(b.text for b in blocks[:400])),
    )


def _paragraph_block(text: str) -> Block:
    """A paragraph, or a heading when it reads like one ("Статья 5. Принципы...", "2.1 Scope")."""
    heading = detect_heading(text)
    return Block(text, kind="heading", level=heading.level) if heading else Block(text)


def parse_text(path: Path) -> ParsedDocument:
    text = _read_text(path)
    blocks = [_paragraph_block(" ".join(p.split())) for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not blocks:
        raise IngestError("the text file is empty")
    return ParsedDocument(
        title=path.stem,
        media_type="text",
        blocks=blocks,
        language=detect_language(text[:20_000]),
    )


class _HtmlText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "template", "nav", "footer", "svg", "head"})
    _BLOCK = frozenset(
        {"p", "div", "li", "td", "th", "tr", "section", "article", "blockquote", "pre", "br", "dd", "dt"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title = ""
        self._buffer: list[str] = []
        self._skip = 0
        self._heading: int | None = None
        self._in_title = False

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer.clear()
        if not text:
            return
        if self._heading is not None:
            self.blocks.append(Block(text, kind="heading", level=self._heading))
        else:
            self.blocks.append(_paragraph_block(text))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._heading = int(tag[1])
        elif tag in self._BLOCK:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._heading = None
        elif tag in self._BLOCK:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._buffer.append(data)


def parse_html(path: Path) -> ParsedDocument:
    parser = _HtmlText()
    parser.feed(_read_text(path))
    parser.close()
    parser._flush()
    if not parser.blocks:
        raise IngestError("the HTML page has no text")
    return ParsedDocument(
        title=" ".join(parser.title.split()) or path.stem,
        media_type="html",
        blocks=parser.blocks,
        language=detect_language(" ".join(b.text for b in parser.blocks[:400])),
    )
