"""Tesseract OCR for scanned pages and images.

Tesseract is called directly (image on stdin, TSV on stdout) rather than through a wrapper library, so
its environment can be set per call: ``OMP_THREAD_LIMIT=1`` keeps each Tesseract process on one thread
while several pages run in parallel, without also limiting the embedding model's threads in this process.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cached_property

from PIL import Image, ImageOps

log = logging.getLogger(__name__)


@dataclass
class OcrPage:
    lines: list[str]
    confidence: float  # mean word confidence, 0-100
    words: int


class OcrError(RuntimeError):
    pass


class TesseractOcr:
    def __init__(
        self, languages: str = "eng+rus", tesseract_cmd: str | None = None, timeout: float = 180
    ) -> None:
        self.command = tesseract_cmd or shutil.which("tesseract") or "tesseract"
        self.requested = languages
        self.timeout = timeout
        self._env = {**os.environ, "OMP_THREAD_LIMIT": "1"}

    def _run(self, *args: str, data: bytes | None = None) -> str:
        try:
            result = subprocess.run(  # fixed executable, arguments built here
                [self.command, *args],
                input=data,
                capture_output=True,
                env=self._env,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OcrError(str(exc)) from exc
        if result.returncode != 0:
            raise OcrError(
                result.stderr.decode("utf-8", "replace").strip() or f"exit code {result.returncode}"
            )
        return result.stdout.decode("utf-8", "replace")

    @cached_property
    def available(self) -> bool:
        try:
            self._run("--version")
        except OcrError:
            return False
        return True

    @cached_property
    def languages(self) -> str:
        """The requested languages that are installed, e.g. ``eng+rus`` (falls back to ``eng``)."""
        if not self.available:
            return self.requested
        installed = set(self._run("--list-langs").split()[1:])  # first line is a header
        wanted = [lang for lang in self.requested.split("+") if lang in installed]
        missing = set(self.requested.split("+")) - set(wanted)
        if missing:
            log.warning("Tesseract language data missing: %s", ", ".join(sorted(missing)))
        return "+".join(wanted) or "eng"

    def recognize(self, image: Image.Image) -> OcrPage:
        image = ImageOps.exif_transpose(image).convert("L")
        if image.width < 1200:  # small screenshots and photos: Tesseract reads ~300 dpi text best
            scale = 1200 / image.width
            image = image.resize((1200, round(image.height * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        tsv = self._run(
            "stdin", "stdout", "-l", self.languages, "--oem", "1", "--psm", "3", "tsv", data=buffer.getvalue()
        )
        return parse_tsv(tsv)


def parse_tsv(tsv: str) -> OcrPage:
    """Tesseract TSV → lines in reading order, with a blank line between paragraphs."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        word = (row.get("text") or "").strip()
        if row.get("level") != "5" or not word:  # level 5 rows are words
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        lines.setdefault(key, []).append(word)
        confidence = float(row["conf"])
        if confidence >= 0:
            confidences.append(confidence)
    out: list[str] = []
    previous: tuple[int, int] | None = None
    for key, words in lines.items():  # dicts keep insertion order, which is Tesseract's reading order
        if previous is not None and key[:2] != previous:
            out.append("")
        out.append(" ".join(words))
        previous = key[:2]
    mean = sum(confidences) / len(confidences) if confidences else 0.0
    return OcrPage(lines=out, confidence=round(mean, 1), words=len(confidences))
