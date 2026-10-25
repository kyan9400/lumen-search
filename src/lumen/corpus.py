"""Corpus manifests: files with ids, titles, sources and licences, ingested in one go.

```json
{"documents": [{"id": "nist-sp-800-63b-4", "file": "files/NIST.SP.800-63B-4.pdf",
                "title": "...", "source_url": "https://...", "license": "Public domain", "ocr": "auto"}]}
```

File paths are relative to the manifest.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lumen.config import OcrMode

if TYPE_CHECKING:
    from lumen.engine import IngestReport, SearchEngine

log = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    id: str
    file: Path
    title: str | None = None
    source_url: str | None = None
    license: str | None = None
    ocr: OcrMode | None = None


def read_manifest(path: Path) -> list[ManifestEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in data["documents"]:
        ocr = item.get("ocr")
        if ocr not in (None, "auto", "always", "never"):
            raise ValueError(f"{item['id']}: ocr must be auto, always or never")
        entries.append(
            ManifestEntry(
                id=item["id"],
                file=(path.parent / item["file"]).resolve(),
                title=item.get("title"),
                source_url=item.get("source_url"),
                license=item.get("license"),
                ocr=cast("OcrMode | None", ocr),
            )
        )
    return entries


def ingest_manifest(
    engine: SearchEngine,
    path: Path,
    *,
    progress: Callable[[IngestReport], None] | None = None,
    failed: Callable[[str, str], None] | None = None,
) -> list[IngestReport]:
    """Ingest every entry not indexed yet. A file that cannot be parsed is logged and skipped."""
    from lumen.ingest import IngestError

    reports = []
    for entry in read_manifest(path):
        if engine.document(entry.id) is not None:
            log.info("%s is already indexed", entry.id)
            continue
        try:
            report = engine.ingest(
                entry.file,
                title=entry.title,
                doc_id=entry.id,
                source_url=entry.source_url,
                license=entry.license,
                ocr_mode=entry.ocr,
            )
        except IngestError as exc:
            log.error("skipped %s: %s", entry.id, exc)
            if failed is not None:
                failed(entry.id, str(exc))
            continue
        reports.append(report)
        if progress is not None:
            progress(report)
    return reports
