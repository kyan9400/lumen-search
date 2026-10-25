from itertools import pairwise

from lumen.chunking import chunk_document
from lumen.models import Block, ParsedDocument


def _doc(blocks: list[Block], media_type: str = "pdf") -> ParsedDocument:
    return ParsedDocument(title="Doc", media_type=media_type, blocks=blocks, language="en")


def _sentences(count: int, words: int = 10, start: int = 0) -> str:
    return " ".join(
        " ".join(f"word{i}x{j}" for j in range(words - 1)) + f" end{i}." for i in range(start, start + count)
    )


def test_chunks_never_cross_section_headings() -> None:
    doc = _doc(
        [
            Block("1. First", kind="heading", level=1, page=1),
            Block(_sentences(3), page=1),
            Block("1.1 Sub", kind="heading", level=2, page=1),
            Block(_sentences(2, start=10), page=2),
            Block("2. Second", kind="heading", level=1, page=3),
            Block(_sentences(2, start=20), page=3),
        ]
    )
    chunks = chunk_document(doc, "d", max_words=200, overlap_words=20)
    assert [c.section for c in chunks] == ["1. First", "1. First › 1.1 Sub", "2. Second"]
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_long_sections_split_with_sentence_overlap() -> None:
    doc = _doc([Block(_sentences(12), page=1)])
    chunks = chunk_document(doc, "d", max_words=40, overlap_words=10)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 40 for c in chunks)
    for previous, current in pairwise(chunks):
        last_sentence = previous.text.split(". ")[-1]
        assert current.text.startswith(last_sentence.rstrip("."))  # one sentence of overlap


def test_spreadsheet_rows_are_separate_chunks_labelled_by_row() -> None:
    doc = _doc(
        [
            Block(
                "Id: AC-7\nName: Unsuccessful Logon Attempts",
                kind="row",
                sheet="S",
                row=3,
                label="AC-7 Unsuccessful",
            ),
            Block(
                "Id: SC-28\nName: Information at Rest",
                kind="row",
                sheet="S",
                row=5,
                label="SC-28 Information",
            ),
        ],
        media_type="xlsx",
    )
    chunks = chunk_document(doc, "d")
    assert [(c.sheet, c.row_start, c.section) for c in chunks] == [
        ("S", 3, "AC-7 Unsuccessful"),
        ("S", 5, "SC-28 Information"),
    ]
    assert chunks[0].text == "Id: AC-7\nName: Unsuccessful Logon Attempts"
    assert chunks[0].location_label() == "Sheet 'S', row 3 · AC-7 Unsuccessful"


def test_an_oversized_sentence_is_split_by_words() -> None:
    doc = _doc([Block(" ".join(f"w{i}" for i in range(250)) + ".", page=1)])
    chunks = chunk_document(doc, "d", max_words=100, overlap_words=10)
    assert len(chunks) == 3
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_paragraph_breaks_survive_inside_a_chunk() -> None:
    doc = _doc([Block("First paragraph here.", page=1), Block("Second paragraph here.", page=1)])
    (chunk,) = chunk_document(doc, "d")
    assert chunk.text == "First paragraph here.\nSecond paragraph here."


def test_ocr_confidence_and_language_are_carried() -> None:
    doc = ParsedDocument(
        title="Скан",
        media_type="pdf",
        blocks=[
            Block(
                "Персональные данные хранятся не дольше, чем требуют цели обработки.",
                page=1,
                ocr_confidence=88.0,
            )
        ],
        language="ru",
    )
    (chunk,) = chunk_document(doc, "d")
    assert chunk.ocr_confidence == 88.0
    assert chunk.language == "ru"


def test_location_label_formats() -> None:
    doc = _doc([Block(_sentences(1), page=4), Block(_sentences(1, start=5), page=5)])
    (chunk,) = chunk_document(doc, "d")
    assert chunk.location_label() == "pp. 4–5"
