from pathlib import Path

import pytest
from PIL import Image

from conftest import SCANNED_SENTENCES, FakeOcr, NoOcr, build_pdf
from lumen.ingest import IngestError, parse_file, sniff_type
from lumen.ingest.ocr import parse_tsv
from lumen.ingest.office import parse_csv
from lumen.ingest.pdf import join_across_pages, needs_ocr
from lumen.models import Block


def test_docx_headings_paragraphs_and_tables(docs: Path) -> None:
    parsed = parse_file(docs / "policy.docx")
    assert parsed.title == "Access Policy"
    assert parsed.media_type == "docx"
    kinds = [(b.kind, b.level if b.kind == "heading" else None, b.text) for b in parsed.blocks]
    assert kinds[:3] == [
        ("heading", 1, "1. Accounts"),
        (
            "text",
            None,
            "Accounts are disabled after 90 days without use. Owners review access every quarter.",
        ),
        ("heading", 2, "1.1 Administrators"),
    ]
    assert ("table", None, "Administrator | Monthly") in kinds


def test_xlsx_rows_keep_sheet_row_number_and_label(docs: Path) -> None:
    parsed = parse_file(docs / "controls.xlsx")
    rows = [(b.sheet, b.row, b.label) for b in parsed.blocks]
    assert rows == [
        ("Controls", 2, "AC-7 Unsuccessful Logon Attempts"),
        ("Controls", 4, "SC-28 Protection of Information at Rest"),
        ("Notes", 2, "Backups Backups are tested every month."),
    ]
    assert parsed.blocks[0].text.splitlines() == [
        "Control Identifier: AC-7",
        "Control Name: Unsuccessful Logon Attempts",
        "Control Text: Enforce a limit of consecutive invalid logon attempts.",
    ]


def test_csv_in_windows_1251_with_semicolons(tmp_path: Path) -> None:
    path = tmp_path / "staff.csv"
    path.write_bytes("Отдел;Ответственный\nИТ;Иванов И.И.\n".encode("cp1251"))
    parsed = parse_csv(path)
    assert parsed.blocks[0].text == "Отдел: ИТ\nОтветственный: Иванов И.И."
    assert parsed.blocks[0].row == 2


def test_markdown_headings_code_tables_and_inline_markup(docs: Path) -> None:
    parsed = parse_file(docs / "notes.md")
    assert parsed.title == "Password Storage"
    texts = [(b.kind, b.text) for b in parsed.blocks]
    assert ("text", "Intro paragraph about hashing and salts.") in texts
    assert ("code", "hash = argon2.hash(password)") in texts
    assert ("table", "Argon2id | 19 MiB") in texts
    assert ("text", "bcrypt has a maximum input length of 72 bytes.") in texts
    assert [b.text for b in parsed.blocks if b.kind == "heading"] == [
        "Password Storage",
        "Argon2id",
        "bcrypt",
    ]


def test_html_skips_scripts_and_navigation(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Guide</title><style>p{}</style></head><body><nav>Menu</nav>"
        "<h1>Sessions</h1><p>Idle sessions end after <b>15</b> minutes.</p>"
        "<script>x()</script></body></html>",
        encoding="utf-8",
    )
    parsed = parse_file(path)
    assert parsed.title == "Guide"
    assert [(b.kind, b.text) for b in parsed.blocks] == [
        ("heading", "Sessions"),
        ("text", "Idle sessions end after 15 minutes."),
    ]


def test_text_file_in_russian(docs: Path) -> None:
    parsed = parse_file(docs / "law.txt")
    assert parsed.language == "ru"
    assert parsed.blocks[0].text == "Статья 5. Принципы обработки персональных данных"


def test_pdf_text_headings_and_running_header_removed(docs: Path) -> None:
    parsed = parse_file(docs / "guide.pdf", ocr=NoOcr())
    assert parsed.title == "Remote Access Guide"
    assert parsed.pages == 3
    assert parsed.ocr_pages == []
    headings = [(b.text, b.page) for b in parsed.blocks if b.kind == "heading"]
    assert headings == [("1. Introduction", 1), ("2. Virtual Private Network", 2), ("3. Session Timeouts", 3)]
    text = " ".join(b.text for b in parsed.blocks)
    assert "ACME Remote Access Guide" not in text  # the running header
    assert "This guide explains remote access for staff. It applies to laptops and phones." in text


def test_scanned_pdf_pages_go_through_ocr(scanned_pdf: Path) -> None:
    ocr = FakeOcr()
    parsed = parse_file(scanned_pdf, ocr=ocr)
    assert ocr.calls == 1
    assert parsed.ocr_pages == [1]
    assert parsed.blocks[-1].text == "Passwords shall be changed after a suspected compromise."
    assert parsed.blocks[-1].ocr_confidence == 91.5


def test_scanned_pdf_without_ocr_fails_clearly(scanned_pdf: Path) -> None:
    with pytest.raises(IngestError, match="OCR"):
        parse_file(scanned_pdf, ocr=NoOcr())


def test_ocr_mode_always_and_never(docs: Path) -> None:
    ocr = FakeOcr()
    always = parse_file(docs / "guide.pdf", ocr=ocr, ocr_mode="always")
    assert ocr.calls == 3
    assert always.ocr_pages == [1, 2, 3]
    # "Scanned memo" heads every page, so it is dropped as a running header; the page text stays
    assert [b.text for b in always.blocks] == SCANNED_SENTENCES
    never = parse_file(docs / "guide.pdf", ocr=ocr, ocr_mode="never")
    assert never.ocr_pages == []
    assert ocr.calls == 3


def test_images_are_read_with_ocr(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    Image.new("RGB", (800, 600), "white").save(path)
    parsed = parse_file(path, ocr=FakeOcr())
    assert parsed.media_type == "image"
    assert parsed.ocr_pages == [1]
    with pytest.raises(IngestError, match="Tesseract"):
        parse_file(path, ocr=NoOcr())


def test_needs_ocr() -> None:
    assert needs_ocr("")
    assert needs_ocr("  12 \n")
    assert not needs_ocr("A page with a real text layer and enough words to be readable text.")
    assert needs_ocr("�" * 10 + "garbled text layer from a broken font encoding table")


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("fake.pdf", b"PK\x03\x04 not a pdf", "not a PDF"),
        ("fake.docx", b"%PDF-1.4 not a docx", "Office Open XML"),
        ("fake.png", b"GIF89a", "not a supported image"),
        ("fake.md", b"text\x00with\x00nul", "binary data"),
        ("empty.txt", b"", "empty"),
        ("archive.zip", b"PK\x03\x04", "unsupported file type"),
    ],
)
def test_sniff_type_rejects_disguised_files(tmp_path: Path, name: str, content: bytes, message: str) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(IngestError, match=message):
        sniff_type(path)


def test_page_limit(tmp_path: Path) -> None:
    path = tmp_path / "long.pdf"
    build_pdf(path, pages=5)
    with pytest.raises(IngestError, match="limit is 4"):
        parse_file(path, max_pages=4)


def test_parse_tsv_groups_words_into_lines_and_paragraphs() -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t96.0\tСтатья",
        "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t90.0\t1.",
        "5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t80.0\tТекст",
        "4\t1\t1\t2\t1\t0\t0\t0\t1\t1\t-1\t",
        "5\t1\t1\t2\t1\t1\t0\t0\t1\t1\t-1\tдалее",
    ]
    page = parse_tsv("\n".join([header, *rows]))
    assert page.lines == ["Статья 1.", "Текст", "", "далее"]
    assert page.confidence == pytest.approx(88.7)
    assert page.words == 3


def test_paragraphs_cut_by_a_page_break_are_rejoined() -> None:
    blocks = [
        Block("Passwords shall be compared against a block-", page=4),
        Block("list of compromised values.", page=5),
        Block("Next paragraph starts here.", page=5),
        Block("A finished sentence.", page=6),
        Block("lower-case text on a later page", page=8),
    ]
    joined = join_across_pages(blocks, {"blocklist"})
    assert [(b.text, b.page) for b in joined] == [
        ("Passwords shall be compared against a blocklist of compromised values.", 4),
        ("Next paragraph starts here.", 5),
        ("A finished sentence.", 6),
        ("lower-case text on a later page", 8),
    ]
