import pytest

from lumen.ingest.layout import (
    Heading,
    detect_heading,
    join_broken_word,
    lines_to_blocks,
    resolve_pdfium_breaks,
    strip_running_lines,
    vocabulary,
)


def test_strip_running_lines_removes_headers_footers_and_page_numbers() -> None:
    bodies = [
        "Passwords are hashed.",
        "Sessions time out.",
        "Keys are rotated.",
        "Logs are kept.",
        "Access is reviewed.",
    ]
    pages = [["NIST SP 800-63B-4", "July 2025", body, f"Page {n} of 5"] for n, body in enumerate(bodies, 1)]
    cleaned = strip_running_lines(pages)
    assert cleaned == [[body] for body in bodies]


def test_strip_running_lines_keeps_short_documents_untouched() -> None:
    pages = [["Title", "Text"], ["Title", "More text"]]
    assert strip_running_lines(pages) == pages


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("3.1.1.2. Password Verifiers", 4),
        ("1.11 Password System", 2),
        ("2. Authentication Assurance Levels", 1),
        ("Appendix A. CSF Core", 1),
        ("Статья 5. Принципы обработки персональных данных", 2),
        ("Глава 2. ПРИНЦИПЫ И УСЛОВИЯ ОБРАБОТКИ ПЕРСОНАЛЬНЫХ ДАННЫХ", 1),
    ],
)
def test_detect_heading_accepts_headings(line: str, level: int) -> None:
    heading = detect_heading(line)
    assert heading == Heading(line, level)


@pytest.mark.parametrize(
    "line",
    [
        "1. Verifiers and CSPs SHALL require passwords that are used as a single-factor",
        "2. Verifiers SHOULD permit a maximum length.",
        "Appendix A is the CSF Core — a set of cybersecurity outcomes arranged by Function, then",
        "Статья 5 настоящего Федерального закона применяется",
        "2006 Annual Report Summary",
        "15 characters in length.",
    ],
)
def test_detect_heading_rejects_prose_and_list_items(line: str) -> None:
    assert detect_heading(line) is None


def test_outline_titles_mark_unnumbered_headings() -> None:
    assert detect_heading("Reauthentication", {"reauthentication": 3}) == Heading("Reauthentication", 3)
    # the line's own numbering wins over the bookmark depth
    assert detect_heading("3.1.1. Passwords", {"passwords": 1}) == Heading("3.1.1. Passwords", 3)


def test_lines_to_blocks_builds_paragraphs_and_headings() -> None:
    lines = [
        "1.1 Scope",
        "This standard applies to every federal password system and every",
        "user of those systems in each agency.",
        "• First bullet item",
        "• Second bullet item",
        "Contents ......................... 12",
    ]
    blocks = lines_to_blocks(lines)
    assert blocks[0] == Heading("1.1 Scope", 2)
    assert (
        blocks[1] == "This standard applies to every federal password system and every user of those systems "
        "in each agency."
    )
    assert blocks[2:] == ["• First bullet item", "• Second bullet item"]


def test_hyphenated_line_breaks_use_the_document_vocabulary() -> None:
    words = vocabulary(
        [["the authentication of", "hardware and protected keys", "out-of-band device", "rate is limited"]]
    )
    assert join_broken_word("an authenti-", "cation step", words) == "an authentication step"
    assert join_broken_word("a hardware-", "protected key", words) == "a hardware-protected key"
    assert join_broken_word("out-of-", "band", words) == "out-of-band"
    assert join_broken_word("rate-", "limited", words) == "rate-limited"
    assert join_broken_word("персо\N{SOFT HYPHEN}", "нальных", words) == "персональных"


def test_vocabulary_skips_fragments_around_line_end_hyphens() -> None:
    words = vocabulary([["an authenti-", "cation step"]])
    assert "authenti" not in words
    assert "cation" not in words
    assert {"an", "step"} <= words


def test_pdfium_hyphen_marks_inside_a_line() -> None:
    words = vocabulary([["the authentication of look-up secrets", "hardware and protected keys"]])
    line = "authen\x02tication of look\x02up secrets in a hardware\x02protected module"
    assert (
        resolve_pdfium_breaks(line, words)
        == "authentication of look-up secrets in a hardware-protected module"
    )
    # the right-hand side continues as a compound, so the break was at one of its hyphens
    assert resolve_pdfium_breaks("a denial\x02of-service attack", words) == "a denial-of-service attack"
    assert resolve_pdfium_breaks("an out-of\x02band secret", words) == "an out-of-band secret"
    assert join_broken_word("a denial-", "of-service attack", words) == "a denial-of-service attack"
    # ...unless the document uses the joined word
    assert resolve_pdfium_breaks("authen\x02tication-based", words) == "authentication-based"
