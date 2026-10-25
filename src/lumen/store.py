"""SQLite storage for documents, chunks and index metadata."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from lumen.models import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    language    TEXT NOT NULL DEFAULT 'und',
    pages       INTEGER,
    ocr_pages   INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    source_url  TEXT,
    license     TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    section         TEXT,
    page_start      INTEGER,
    page_end        INTEGER,
    sheet           TEXT,
    row_start       INTEGER,
    row_end         INTEGER,
    ocr_confidence  REAL,
    language        TEXT NOT NULL DEFAULT 'und',
    UNIQUE (doc_id, ordinal)
);
CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""

_CHUNK_COLUMNS = (
    "id, doc_id, ordinal, text, section, page_start, page_end, sheet, row_start, row_end, "
    "ocr_confidence, language"
)
_DOCUMENT_COLUMNS = (
    "id, title, filename, media_type, sha256, language, pages, ocr_pages, chunks, source_url, license, "
    "created_at"
)


def _chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        ordinal=row["ordinal"],
        text=row["text"],
        section=row["section"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        sheet=row["sheet"],
        row_start=row["row_start"],
        row_end=row["row_end"],
        ocr_confidence=row["ocr_confidence"],
        language=row["language"],
    )


def _document(row: sqlite3.Row) -> Document:
    return Document(**{key: row[key] for key in row.keys()})  # noqa: SIM118 - sqlite3.Row is not a dict


class Store:
    """A thread-safe wrapper around one SQLite connection (WAL mode, foreign keys on)."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    # --- metadata ------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # --- documents -----------------------------------------------------------------------------------

    def add_document(self, document: Document, chunks: Sequence[Chunk]) -> list[int]:
        """Insert a document and its chunks atomically; returns the new chunk ids in order."""
        with self.transaction() as db:
            db.execute(
                "INSERT INTO documents (id, title, filename, media_type, sha256, language, pages, ocr_pages, "
                "chunks, source_url, license) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.title,
                    document.filename,
                    document.media_type,
                    document.sha256,
                    document.language,
                    document.pages,
                    document.ocr_pages,
                    len(chunks),
                    document.source_url,
                    document.license,
                ),
            )
            ids = []
            for chunk in chunks:
                cursor = db.execute(
                    "INSERT INTO chunks (doc_id, ordinal, text, section, page_start, page_end, sheet, "
                    "row_start, row_end, ocr_confidence, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.sheet,
                        chunk.row_start,
                        chunk.row_end,
                        chunk.ocr_confidence,
                        chunk.language,
                    ),
                )
                ids.append(int(cursor.lastrowid or 0))
        return ids

    def delete_document(self, doc_id: str) -> list[int]:
        """Delete a document and its chunks; returns the deleted chunk ids."""
        with self.transaction() as db:
            ids = [r["id"] for r in db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return ids

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return None if row is None else _document(row)

    def find_by_hash(self, sha256: str) -> Document | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return None if row is None else _document(row)

    def documents(self) -> list[Document]:
        with self._lock:
            rows = self._db.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY created_at, id")
            return [_document(r) for r in rows]

    # --- chunks --------------------------------------------------------------------------------------

    def chunks(self, ids: Sequence[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        out: dict[int, Chunk] = {}
        with self._lock:
            for start in range(0, len(ids), 500):  # stay under SQLite's bound-parameter limit
                batch = list(ids[start : start + 500])
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id IN ({marks})", batch)
                out.update((r["id"], _chunk(r)) for r in rows)
        return out

    def all_chunks(self) -> Iterator[Chunk]:
        with self._lock:
            rows = self._db.execute(f"SELECT {_CHUNK_COLUMNS} FROM chunks ORDER BY id").fetchall()
        return (_chunk(r) for r in rows)

    def chunk_ids(self, doc_ids: Sequence[str] | None = None, language: str | None = None) -> set[int]:
        clauses: list[str] = []
        params: list[str] = []
        if doc_ids:
            clauses.append(f"doc_id IN ({','.join('?' * len(doc_ids))})")
            params.extend(doc_ids)
        if language:
            clauses.append("language = ?")
            params.append(language)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            return {r["id"] for r in self._db.execute(f"SELECT id FROM chunks{where}", params)}

    def neighbours(self, chunk: Chunk) -> tuple[Chunk | None, Chunk | None]:
        """The chunks just before and after ``chunk`` in its document."""
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE doc_id = ? AND ordinal IN (?, ?)",
                (chunk.doc_id, chunk.ordinal - 1, chunk.ordinal + 1),
            ).fetchall()
        by_ordinal = {r["ordinal"]: _chunk(r) for r in rows}
        return by_ordinal.get(chunk.ordinal - 1), by_ordinal.get(chunk.ordinal + 1)

    def counts(self) -> tuple[int, int]:
        with self._lock:
            documents = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(documents), int(chunks)
