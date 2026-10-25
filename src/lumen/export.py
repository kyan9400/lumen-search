"""Export the index for the static browser demo.

Writes ``data/corpus.json`` (documents, chunks, stop words and retrieval parameters),
``data/vectors.bin`` (one int8 row per chunk), ``data/scales.bin`` (one float32 scale per row) and a copy
of every source file under ``files/``. The browser rebuilds BM25 from the chunk texts with the same
analyzer, so only the vectors need to be shipped in binary.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lumen.textproc import STOPWORDS

if TYPE_CHECKING:
    from lumen.engine import SearchEngine


def quantize(vectors: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]:
    """Symmetric int8 per row: row ≈ q * scale, with the largest component mapped to ±127."""
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, scales.astype(np.float32)


def export_demo(engine: SearchEngine, out: Path, *, model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the demo data under ``out``; ``model`` describes the browser model (id, query prefix, size)."""
    data = out / "data"
    files = out / "files"
    data.mkdir(parents=True, exist_ok=True)
    files.mkdir(parents=True, exist_ok=True)
    documents = engine.documents()
    position = {d.id: i for i, d in enumerate(documents)}
    chunks = sorted(engine.store.all_chunks(), key=lambda c: (position[c.doc_id], c.ordinal))
    # embed with the engine's embedder: for the demo that is the browser's own ONNX model (OnnxEmbedder)
    titles = {d.id: d.title for d in documents}
    vectors = engine.embedder.embed_passages([c.embedding_text(titles[c.doc_id]) for c in chunks])
    quantized, scales = quantize(vectors)
    quantized.tofile(data / "vectors.bin")
    scales.tofile(data / "scales.bin")

    exported_documents = []
    for document in documents:
        source = engine.document_file(document.id)
        entry: dict[str, Any] = {
            "id": document.id,
            "title": document.title,
            "media_type": document.media_type,
            "language": document.language,
            "pages": document.pages,
            "ocr_pages": document.ocr_pages,
            "chunks": document.chunks,
            "source_url": document.source_url,
            "license": document.license,
        }
        if source is not None:
            shutil.copyfile(source, files / source.name)
            entry["file"] = f"files/{source.name}"
        exported_documents.append(entry)

    def compact(chunk: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"d": position[chunk.doc_id], "t": chunk.text, "l": chunk.language}
        if chunk.section:
            item["s"] = chunk.section
        if chunk.page_start is not None:
            item["p"] = [chunk.page_start, chunk.page_end or chunk.page_start]
        if chunk.sheet is not None:
            item["sh"] = chunk.sheet
        if chunk.row_start is not None:
            item["r"] = [chunk.row_start, chunk.row_end or chunk.row_start]
        if chunk.ocr_confidence is not None:
            item["o"] = chunk.ocr_confidence
        return item

    settings = engine.settings
    corpus = {
        "format": 1,
        "generated": dt.date.today().isoformat(),
        "model": {
            "name": engine.embedder.name,
            "dim": int(vectors.shape[1]),
            "query_prefix": getattr(engine.embedder, "query_prefix", ""),
            **(model or {}),
        },
        "bm25": {"k1": settings.bm25_k1, "b": settings.bm25_b},
        "rrf_k": settings.rrf_k,
        "candidates": settings.candidates,
        "stopwords": sorted(STOPWORDS),
        "documents": exported_documents,
        "chunks": [compact(c) for c in chunks],
    }
    (data / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "corpus_mb": round((data / "corpus.json").stat().st_size / 1e6, 2),
        "vectors_mb": round((data / "vectors.bin").stat().st_size / 1e6, 2),
    }
