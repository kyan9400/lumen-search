"""OCR against the real Tesseract binary (CI installs it; skipped where it is missing)."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from lumen.ingest import parse_file
from lumen.ingest.ocr import TesseractOcr

pytestmark = pytest.mark.ocr
SCAN = Path(__file__).parents[1] / "corpus" / "files" / "ru-519-fz-2020-scan.pdf"


@pytest.fixture(scope="module")
def tesseract() -> TesseractOcr:
    ocr = TesseractOcr("eng+rus")
    if not ocr.available:
        pytest.skip("Tesseract is not installed")
    return ocr


def test_reads_a_rendered_english_page(tesseract: TesseractOcr) -> None:
    image = Image.new("L", (1700, 420), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=56)
    draw.text((60, 70), "Passwords shall be changed after", font=font, fill=0)
    draw.text((60, 200), "a suspected compromise.", font=font, fill=0)
    page = tesseract.recognize(image)
    text = " ".join(page.lines)
    assert "Passwords shall be changed" in text
    assert "compromise" in text
    assert page.confidence > 70


def test_reads_the_scanned_russian_law(tesseract: TesseractOcr) -> None:
    if "rus" not in tesseract.languages:
        pytest.skip("Russian language data for Tesseract is not installed")
    parsed = parse_file(SCAN, ocr=tesseract)
    assert parsed.ocr_pages == list(range(1, 9))
    assert parsed.language == "ru"
    text = " ".join(b.text for b in parsed.blocks)
    assert "О внесении изменений в Федеральный закон" in text
    assert "разрешенных субъектом персональных данных для распространения" in text
    assert min(b.ocr_confidence or 0 for b in parsed.blocks if b.kind == "text") > 80
