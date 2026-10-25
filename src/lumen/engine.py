"""The search engine: ingestion, hybrid retrieval (BM25 + dense, fused with RRF), reranking, answers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lumen.answer import Answer, LlmClient, Source, extractive_answer, generated_answer
from lumen.bm25 import BM25Index
from lumen.cache import Cache, make_cache
from lumen.chunking import chunk_document
from lumen.config import OcrMode, SearchMode, Settings
from lumen.embeddings import Embedder, Vectors, load_embedder
from lumen.ingest import IngestError, TesseractOcr, parse_file
from lumen.models import Chunk, Document
from lumen.rerank import CrossEncoderReranker, Reranker
from lumen.store import Store
from lumen.textproc import analyze, highlight_spans
from lumen.vectors import VectorIndex

log = logging.getLogger(__name__)

_TRANSLIT = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщыэюя",
        "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh shch y e yu ya".split(),
        strict=True,
    ),
    ъ="",
    ь="",
)


def slugify(text: str, max_length: int = 48) -> str:
    """ASCII slug for document ids, with Russian transliterated.

    "О персональных данных" becomes "o-personalnykh-dannykh".
    """
    out = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    slug = re.sub(r"[^a-z0-9]+", "-", out).strip("-")
    return slug[:max_length].rstrip("-") or "document"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(d) = sum of weight / (k + rank of d), ranks counted from 1.

    RRF needs no score calibration between BM25 and cosine similarity, which live on unrelated scales.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], best[pair[0]], pair[0]))


@dataclass
class Hit:
    chunk: Chunk
    document: Document
    rank: int
    score: float
    fused: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    highlights: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SearchResult:
    query: str
    mode: SearchMode
    hits: list[Hit]
    took_ms: float
    cached: bool = False
    reranked: bool = False


@dataclass
class IngestReport:
    document: Document
    created: bool  # False when a file with the same SHA-256 was already indexed
    chunks: int
    ocr_pages: list[int]
    warnings: list[str]
    took_ms: float


@dataclass
class AnswerResult:
    question: str
    answer: Answer
    sources: list[Hit]
    took_ms: float


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        cache: Cache | None = None,
        reranker: Reranker | None = None,
        ocr: TesseractOcr | None = None,
        llm: LlmClient | None = None,
    ) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.files_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.db_path)
        self.embedder = embedder or load_embedder(
            settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        self._check_model()
        self.cache = cache or make_cache(settings.redis_url)
        if reranker is None and settings.reranker_model:
            reranker = CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
        self.reranker = reranker
        self.ocr = ocr or TesseractOcr(settings.ocr_languages, settings.tesseract_cmd)
        if llm is None and settings.llm_base_url and settings.llm_model:
            llm = LlmClient(
                settings.llm_base_url,
                settings.llm_model,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        self.llm = llm
        self._write_lock = threading.Lock()  # one ingestion or deletion at a time
        self._index_lock = threading.RLock()  # FAISS must not be searched while it is being modified
        self.vectors = VectorIndex.load(settings.vectors_path, self.embedder.dim, settings.faiss_index)
        self.bm25 = BM25Index(settings.bm25_k1, settings.bm25_b)
        self._terms: dict[int, list[str]] = {}
        self._documents: dict[str, Document] = {}
        self._load()

    # --- start-up ------------------------------------------------------------------------------------

    def _check_model(self) -> None:
        stored = self.store.get_meta("embedding_model")
        if stored is None:
            self.store.set_meta("embedding_model", self.embedder.name)
            self.store.set_meta("embedding_dim", str(self.embedder.dim))
        elif stored != self.embedder.name:
            raise RuntimeError(
                f"the index was built with {stored!r}, not {self.embedder.name!r}: "
                "run `lumen reindex` or use a different LUMEN_DATA_DIR"
            )

    def _load(self) -> None:
        self._documents = {d.id: d for d in self.store.documents()}
        chunks = list(self.store.all_chunks())
        self._terms = {c.id: analyze(c.indexed_text()) for c in chunks if c.id is not None}
        self._rebuild_bm25()
        # Keep FAISS in step with SQLite after a crash between the two writes.
        stored = set(self._terms)
        indexed = set(self.vectors.ids())
        missing = sorted(stored - indexed)
        extra = sorted(indexed - stored)
        if extra:
            self.vectors.remove(extra)
        if missing:
            log.warning("re-embedding %d chunk(s) missing from the vector index", len(missing))
            by_id = {c.id: c for c in chunks}
            texts = [by_id[i].embedding_text(self._documents[by_id[i].doc_id].title) for i in missing]
            self.vectors.add(missing, self.embedder.embed_passages(texts))
        if missing or extra:
            self.vectors.save(self.settings.vectors_path)

    def _rebuild_bm25(self) -> None:
        index = BM25Index(self.settings.bm25_k1, self.settings.bm25_b)
        ids = sorted(self._terms)
        index.build(ids, [self._terms[i] for i in ids])
        self.bm25 = index  # swapped in one assignment: searches see the old or the new index

    @property
    def index_version(self) -> int:
        return int(self.store.get_meta("index_version") or 0)

    def _bump_version(self) -> None:
        self.store.set_meta("index_version", str(self.index_version + 1))

    # --- documents -----------------------------------------------------------------------------------

    def documents(self) -> list[Document]:
        return list(self._documents.values())

    def document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def document_file(self, doc_id: str) -> Path | None:
        document = self._documents.get(doc_id)
        if document is None:
            return None
        path = self.settings.files_dir / f"{doc_id}{Path(document.filename).suffix.lower()}"
        return path if path.exists() else None

    def ingest(
        self,
        path: Path,
        *,
        title: str | None = None,
        doc_id: str | None = None,
        filename: str | None = None,
        source_url: str | None = None,
        license: str | None = None,
        ocr_mode: OcrMode | None = None,
    ) -> IngestReport:
        started = time.perf_counter()
        sha256 = file_sha256(path)
        existing = self.store.find_by_hash(sha256)
        if existing is not None:
            return IngestReport(existing, False, existing.chunks, [], [], _ms(started))
        parsed = parse_file(
            path,
            ocr=self.ocr,
            ocr_mode=ocr_mode or self.settings.ocr_mode,
            ocr_dpi=self.settings.ocr_dpi,
            ocr_workers=self.settings.ocr_workers,
            max_pages=self.settings.max_pages,
        )
        title = title or parsed.title
        doc_id = doc_id or f"{slugify(title)}-{sha256[:8]}"
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", doc_id):
            raise IngestError("document ids use lower-case letters, digits and hyphens")
        chunks = chunk_document(
            parsed,
            doc_id,
            max_words=self.settings.chunk_words,
            overlap_words=self.settings.chunk_overlap_words,
        )
        if not chunks:
            raise IngestError("no text could be extracted")
        vectors = self.embedder.embed_passages([c.embedding_text(title) for c in chunks])
        name = filename or path.name
        document = Document(
            id=doc_id,
            title=title,
            filename=name,
            media_type=parsed.media_type,
            sha256=sha256,
            language=parsed.language,
            pages=parsed.pages,
            ocr_pages=len(parsed.ocr_pages),
            chunks=len(chunks),
            source_url=source_url,
            license=license,
        )
        with self._write_lock:
            if self.store.get_document(doc_id) is not None:
                raise IngestError(f"a document with id {doc_id!r} already exists")
            target = self.settings.files_dir / f"{doc_id}{Path(name).suffix.lower()}"
            shutil.copyfile(path, target)
            try:
                ids = self.store.add_document(document, chunks)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            stored = self.store.get_document(doc_id)
            assert stored is not None
            with self._index_lock:
                self.vectors.add(ids, vectors)
                for chunk_id, chunk in zip(ids, chunks, strict=True):
                    self._terms[chunk_id] = analyze(chunk.indexed_text())
                self._rebuild_bm25()
                self._documents[doc_id] = stored
            self.vectors.save(self.settings.vectors_path)
            self._bump_version()
        log.info("indexed %s: %d chunks, %d OCR pages", doc_id, len(chunks), len(parsed.ocr_pages))
        return IngestReport(stored, True, len(chunks), parsed.ocr_pages, parsed.warnings, _ms(started))

    def delete(self, doc_id: str) -> bool:
        with self._write_lock:
            path = self.document_file(doc_id)
            if self.store.get_document(doc_id) is None:
                return False
            ids = self.store.delete_document(doc_id)
            with self._index_lock:
                self.vectors.remove(ids)
                for chunk_id in ids:
                    self._terms.pop(chunk_id, None)
                self._rebuild_bm25()
                self._documents.pop(doc_id, None)
            self.vectors.save(self.settings.vectors_path)
            self._bump_version()
            if path is not None:
                path.unlink(missing_ok=True)
        return True

    def chunk(self, chunk_id: int) -> tuple[Chunk, Chunk | None, Chunk | None] | None:
        found = self.store.chunks([chunk_id]).get(chunk_id)
        if found is None:
            return None
        before, after = self.store.neighbours(found)
        return found, before, after

    def stats(self) -> dict[str, int]:
        documents, chunks = self.store.counts()
        return {"documents": documents, "chunks": chunks, "vectors": len(self.vectors)}

    # --- search --------------------------------------------------------------------------------------

    def _query_vector(self, query: str) -> Vectors:
        key = "emb:" + hashlib.sha1(f"{self.embedder.name}\x00{query}".encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None and len(cached) == 4 * self.embedder.dim:
            restored: Vectors = np.frombuffer(cached, dtype=np.float32)
            return restored
        vector: Vectors = self.embedder.embed_queries([query])[0]
        self.cache.set(key, vector.astype(np.float32).tobytes(), self.settings.cache_ttl_seconds * 6)
        return vector

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
        rerank: bool | None = None,
    ) -> SearchResult:
        started = time.perf_counter()
        query = " ".join(query.split())
        use_rerank = self.reranker is not None and (rerank if rerank is not None else True)
        if not query:
            return SearchResult(query, mode, [], _ms(started))
        key = (
            "search:"
            + hashlib.sha1(
                json.dumps(
                    [self.index_version, query, k, mode, sorted(doc_ids or []), language, use_rerank],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        )
        cached = self.cache.get(key)
        if cached is not None:
            hits = self._hits_from_cache(json.loads(cached), query)
            if hits is not None:
                return SearchResult(query, mode, hits, _ms(started), cached=True, reranked=use_rerank)

        allowed = self.store.chunk_ids(doc_ids, language) if (doc_ids or language) else None
        depth = max(self.settings.candidates, k, self.settings.rerank_top if use_rerank else 0)
        query_terms = analyze(query)
        bm25 = self.bm25.search(query_terms, depth, allowed) if mode in ("hybrid", "bm25") else []
        dense: list[tuple[int, float]] = []
        if mode in ("hybrid", "dense"):
            vector = self._query_vector(query)
            with self._index_lock:
                dense = self.vectors.search(vector, depth, allowed)
        bm25_rank = {cid: (r, s) for r, (cid, s) in enumerate(bm25, start=1)}
        dense_rank = {cid: (r, s) for r, (cid, s) in enumerate(dense, start=1)}
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[cid for cid, _ in bm25], [cid for cid, _ in dense]], k=self.settings.rrf_k
            )
        else:
            fused = bm25 if mode == "bm25" else dense
        keep = max(k, self.settings.rerank_top) if use_rerank else k
        ranked = fused[:keep]
        chunks = self.store.chunks([cid for cid, _ in ranked])
        ranked = [(cid, s) for cid, s in ranked if cid in chunks and chunks[cid].doc_id in self._documents]

        rerank_scores: dict[int, float] = {}
        if use_rerank and ranked and self.reranker is not None:
            scores = self.reranker.score(query, [chunks[cid].indexed_text() for cid, _ in ranked])
            rerank_scores = {cid: s for (cid, _), s in zip(ranked, scores, strict=True)}
            ranked = sorted(ranked, key=lambda pair: -rerank_scores[pair[0]])

        terms = set(query_terms)
        hits = []
        for rank, (cid, fused_score) in enumerate(ranked[:k], start=1):
            chunk = chunks[cid]
            hits.append(
                Hit(
                    chunk=chunk,
                    document=self._documents[chunk.doc_id],
                    rank=rank,
                    score=rerank_scores.get(cid, fused_score),
                    fused=fused_score,
                    bm25_rank=bm25_rank[cid][0] if cid in bm25_rank else None,
                    bm25_score=bm25_rank[cid][1] if cid in bm25_rank else None,
                    dense_rank=dense_rank[cid][0] if cid in dense_rank else None,
                    dense_score=dense_rank[cid][1] if cid in dense_rank else None,
                    rerank_score=rerank_scores.get(cid),
                    highlights=highlight_spans(chunk.text, terms),
                )
            )
        self.cache.set(
            key, json.dumps([_hit_to_cache(h) for h in hits]).encode(), self.settings.cache_ttl_seconds
        )
        return SearchResult(query, mode, hits, _ms(started), reranked=use_rerank)

    def _hits_from_cache(self, payload: list[dict[str, float | int | None]], query: str) -> list[Hit] | None:
        chunks = self.store.chunks([int(item["id"] or 0) for item in payload])
        terms = set(analyze(query))
        hits = []
        for item in payload:
            chunk = chunks.get(int(item["id"] or 0))
            if chunk is None or chunk.doc_id not in self._documents:
                return None  # the index changed underneath the cached entry
            hits.append(
                Hit(
                    chunk=chunk,
                    document=self._documents[chunk.doc_id],
                    rank=int(item["rank"] or 0),
                    score=float(item["score"] or 0.0),
                    fused=_opt_float(item.get("fused")),
                    bm25_rank=_opt_int(item.get("bm25_rank")),
                    bm25_score=_opt_float(item.get("bm25_score")),
                    dense_rank=_opt_int(item.get("dense_rank")),
                    dense_score=_opt_float(item.get("dense_score")),
                    rerank_score=_opt_float(item.get("rerank_score")),
                    highlights=highlight_spans(chunk.text, terms),
                )
            )
        return hits

    # --- answers -------------------------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        mode: SearchMode = "hybrid",
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        result = self.search(question, k=k, mode=mode, doc_ids=doc_ids, language=language)
        sources = [Source(h.rank, h.chunk, h.document) for h in result.hits]
        answer: Answer
        if not sources:
            answer = extractive_answer(question, [], self.embedder)
        elif self.llm is not None:
            try:
                answer = generated_answer(question, sources, self.llm)
            except Exception as exc:  # network, HTTP or malformed-response errors: fall back, never fail
                log.warning("LLM answer failed, using the extractive answer: %s", exc)
                answer = extractive_answer(question, sources, self.embedder)
                answer.note = "The language model was unavailable; this is an extractive answer."
        else:
            answer = extractive_answer(question, sources, self.embedder)
        return AnswerResult(question, answer, result.hits, _ms(started))

    def close(self) -> None:
        if self.llm is not None:
            self.llm.close()
        self.store.close()


def _hit_to_cache(hit: Hit) -> dict[str, float | int | None]:
    return {
        "id": hit.chunk.id,
        "rank": hit.rank,
        "score": hit.score,
        "fused": hit.fused,
        "bm25_rank": hit.bm25_rank,
        "bm25_score": hit.bm25_score,
        "dense_rank": hit.dense_rank,
        "dense_score": hit.dense_score,
        "rerank_score": hit.rerank_score,
    }


def _opt_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: float | int | None) -> int | None:
    return None if value is None else int(value)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
