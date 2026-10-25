"""From lines of page text to headings and paragraphs.

PDF text (born-digital or OCR) arrives as lines. This module removes running headers and footers, drops
table-of-contents leaders, joins hyphenated words, finds numbered headings and groups the rest into
paragraphs.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_TOC_LEADER = re.compile(r"(?:\.\s?){4,}\s*[\divxlc]+\s*$|…{2,}\s*\d+\s*$", re.IGNORECASE)
# a contents entry without leaders, as in older scans: "3.2 Length. 15", "APPENDIX C. Glossary. 26"
_TOC_ENTRY = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|appendix\s+[a-z]\.?)\s+\S.{0,80}?\.\s*\d{1,3}$", re.I)
_PAGE_NUMBER = re.compile(r"(?:page\s+)?\d{1,4}(?:\s*(?:of|/|из)\s*\d{1,4})?|[-–—]\s*\d{1,4}\s*[-–—]", re.I)
_BULLET = re.compile(r"^(?:[•◦▪●‣∙·*–—-]\s+|\(?\d{1,3}[.)]\s+|\(?[a-zа-я][.)]\s+)", re.IGNORECASE)
# a word broken at the line end: hyphen, soft hyphen, PDFium's generated hyphen (\x02) or "¬" from OCR
_HYPHEN_END = re.compile(r"[A-Za-zА-Яа-яЁё][-\N{SOFT HYPHEN}\x02¬]$")

# "3.1.1.2. Password Verifiers", "1.11 Password System", "Appendix A. CSF Core", "Статья 5. Принципы..."
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,5})\.?\s+(?P<title>\S.*)$")
_NAMED = re.compile(
    r"^(?P<kind>appendix|annex|chapter|part|section|статья|глава|раздел|приложение)\s+"
    r"(?P<num>[A-ZА-Я]|\d{1,3}(?:\.\d{1,3})*)\b\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_NAMED_LEVEL = {"part": 1, "chapter": 1, "раздел": 1, "глава": 1, "appendix": 1, "annex": 1, "приложение": 1}
_MINOR_WORDS = frozenset(
    "a an and as at by for from in into of on or the to via with и в во на по о об от для из к с со".split()
)


@dataclass
class Heading:
    text: str
    level: int


_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_LAST_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*$")
_FIRST_WORD = re.compile(r"^[^\W\d_]+(?:-[^\W\d_]+)*")
# PDFium rejoins a word hyphenated across lines itself and marks the hyphen with \x02: "hardware\x02protected"
_PDFIUM_BREAK = re.compile(r"([^\W\d_]+(?:-[^\W\d_]+)*)\x02([^\W\d_]+(?:-[^\W\d_]+)*)")


def vocabulary(pages: list[list[str]]) -> set[str]:
    """Lower-case words and hyphenated compounds of a document, for deciding how to join broken words.

    The fragments around a line-end hyphen ("authenti-" / "cation") are left out: they are not words.
    """
    words: set[str] = set()
    for lines in pages:
        broken = False
        for line in lines:
            found = _WORD.findall(_PDFIUM_BREAK.sub(" ", line.strip()))
            if broken and found:
                found = found[1:]
            broken = bool(_HYPHEN_END.search(line.strip()))
            if broken and found:
                found = found[:-1]
            words.update(w.lower() for w in found)
    return words


def ends_broken(text: str) -> bool:
    """True when the text ends in the first half of a word broken by a hyphen."""
    return bool(_HYPHEN_END.search(text))


def join_broken_word(left: str, right: str, words: set[str] | None) -> str:
    """Join a word split at a line end: "authenti-" + "cation", but "hardware-" + "protected".

    A soft hyphen always joins. A hyphen (or PDFium's \x02 mark) joins when the document uses the joined
    word, and is kept when either side is part of a compound, the document uses the hyphenated form, or
    both halves are words of their own.
    """
    head, mark = left[:-1], left[-1]
    if mark in "-\x02" and words:
        before = _LAST_WORD.search(head)
        after = _FIRST_WORD.match(right)
        if before and after:
            prefix, rest = before.group().lower(), after.group().lower()
            last, first = prefix.rsplit("-", 1)[-1], rest.split("-", 1)[0]
            if last + first not in words and (
                # "out-of-" + "band", "denial-" + "of-service": the break is at one of a compound's hyphens
                "-" in prefix
                or "-" in rest
                or f"{last}-{first}" in words
                or (last in words and first in words)
            ):
                return f"{head}-{right}"
    return head + right


def resolve_pdfium_breaks(line: str, words: set[str] | None) -> str:
    """Decide each "word\x02word" PDFium produced: "look-up" stays hyphenated, "authen-tication" joins."""
    return _PDFIUM_BREAK.sub(lambda m: join_broken_word(m.group(1) + "-", m.group(2), words), line)


def _key(line: str) -> str:
    return _SPACES.sub(" ", _DIGITS.sub("#", line)).strip().lower()


def strip_running_lines(pages: list[list[str]], *, share: float = 0.4, edge: int = 6) -> list[list[str]]:
    """Remove running headers, footers and page numbers.

    A line counts as running when, with digits masked, it appears near the top or bottom of at least
    ``share`` of the pages (and on three or more pages).
    """
    if len(pages) >= 3:
        counts: Counter[str] = Counter()
        for lines in pages:
            counts.update({_key(line) for line in lines[:edge] + lines[-edge:]})
        threshold = max(3, math.ceil(len(pages) * share))
        repeated = {key for key, count in counts.items() if key and count >= threshold}
    else:
        repeated = set()
    cleaned = []
    for lines in pages:
        last = len(lines) - edge
        kept = []
        for index, line in enumerate(lines):
            near_edge = index < edge or index >= last
            if near_edge and (_key(line) in repeated or _PAGE_NUMBER.fullmatch(line.strip())):
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned


def detect_heading(line: str, outline_titles: dict[str, int] | None = None) -> Heading | None:
    """Return a heading when the line looks like one (numbered, named, or listed in the PDF outline)."""
    text = _SPACES.sub(" ", line).strip()
    if not 2 < len(text) <= 160 or text.endswith((",", ";")):
        return None
    numbered = _NUMBERED.match(text)
    if outline_titles:
        bare = (numbered.group("title") if numbered else text).lower()
        if bare in outline_titles:
            # the line's own numbering ("3.1.1.") is the better guide to depth than the bookmark tree
            level = numbered.group("num").count(".") + 1 if numbered else outline_titles[bare]
            return Heading(text, level)
    named = _NAMED.match(text)
    if named:
        kind = named.group("kind").lower()
        title = named.group("title")
        # "Section 5 describes ..." and "Статья 5 настоящего закона" are prose: a heading's title is short
        # and starts with a capital letter
        if title and (title[0].islower() or len(title.split()) > 18 or title.endswith(",")):
            return None
        return Heading(text, _NAMED_LEVEL.get(kind, 2))
    if not numbered:
        return None
    title = numbered.group("title")
    depth = numbered.group("num").count(".") + 1
    words = title.split()
    if len(words) > 12 or title.endswith((".", ":")) or not title[0].isupper():
        return None
    significant = [w for w in words if w.lower() not in _MINOR_WORDS]
    capitalised = sum(1 for w in significant if w[:1].isupper() or not w[:1].isalpha())
    # "1. Verifiers SHALL..." is a list item; a single-number heading must read like a title
    needed = 0.6 if depth > 1 else 1.0
    if not significant or capitalised / len(significant) < needed:
        return None
    if depth == 1 and int(numbered.group("num")) > 30:
        return None  # "2006 Annual Report" or a figure from a table, not chapter 2006
    return Heading(text, depth)


def lines_to_blocks(
    lines: list[str], outline_titles: dict[str, int] | None = None, words: set[str] | None = None
) -> list[str | Heading]:
    """Group lines into paragraphs (``str``) and headings, in reading order.

    ``words`` is the document's :func:`vocabulary`, used to rejoin words broken across lines.
    """
    out: list[str | Heading] = []
    lengths = [len(line.strip()) for line in lines if len(line.strip()) >= 20]
    typical = statistics.median(lengths) if lengths else 0.0
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            out.append(current.strip())
        current = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if _TOC_LEADER.search(line) or (_TOC_ENTRY.match(line) and len(line.split()) <= 12):
            flush()
            continue
        heading = detect_heading(line, outline_titles)
        if heading is not None:
            flush()
            out.append(heading)
            continue
        if _BULLET.match(line):
            flush()
        if current and ends_broken(current) and line[:1].islower():
            current = join_broken_word(current, line, words)
        else:
            current = f"{current} {line}" if current else line
        if typical and len(line) < 0.7 * typical and line.endswith((".", ":", "!", "?", ";")):
            flush()  # a short line ending a sentence closes its paragraph
    flush()
    return out
