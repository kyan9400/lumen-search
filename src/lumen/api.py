"""HTTP API (FastAPI) and the bundled web UI."""

from __future__ import annotations

import hmac
import logging
import mimetypes
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from lumen import __version__
from lumen.config import OcrMode, SearchMode, Settings
from lumen.corpus import ingest_manifest
from lumen.engine import Hit, SearchEngine
from lumen.ingest import SUPPORTED_EXTENSIONS, IngestError
from lumen.metrics import Metrics
from lumen.models import Chunk, Document

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

# Python reads MIME types from the Windows registry, where .js is often "text/plain"; browsers refuse to
# run module scripts served like that, so register the web types explicitly.
for _type, _extension in (
    ("text/javascript", ".js"),
    ("text/javascript", ".mjs"),
    ("text/css", ".css"),
    ("image/svg+xml", ".svg"),
    ("application/wasm", ".wasm"),
):
    mimetypes.add_type(_type, _extension)

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


# --- response models ---------------------------------------------------------------------------------


class LocationOut(BaseModel):
    label: str = Field(description="Citation such as 'p. 25 · 3.1.1.2. Password Verifiers'")
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None


class ScoresOut(BaseModel):
    score: float = Field(description="Final ranking score: the reranker's when reranking ran, else RRF")
    fused: float | None = None
    bm25: float | None = None
    bm25_rank: int | None = None
    dense: float | None = Field(default=None, description="Cosine similarity of query and chunk")
    dense_rank: int | None = None
    rerank: float | None = None


class HitOut(BaseModel):
    rank: int
    chunk_id: int
    doc_id: str
    title: str
    media_type: str
    text: str
    highlights: list[tuple[int, int]] = Field(description="Character ranges of query terms in text")
    location: LocationOut
    language: str
    ocr_confidence: float | None = None
    scores: ScoresOut
    file_url: str | None = None


class SearchOut(BaseModel):
    query: str
    mode: SearchMode
    took_ms: float
    cached: bool
    reranked: bool
    hits: list[HitOut]


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    mode: SearchMode = "hybrid"
    doc: list[str] | None = None
    lang: Literal["en", "ru"] | None = None


class AnswerOut(BaseModel):
    question: str
    answer: str
    method: str = Field(description="extractive, llm, or none when nothing relevant was found")
    note: str | None = None
    citations: list[int] = Field(description="Ranks of the sources the answer cites")
    sources: list[HitOut]
    took_ms: float


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    media_type: str
    language: str
    pages: int | None
    ocr_pages: int
    chunks: int
    source_url: str | None
    license: str | None
    created_at: str
    file_url: str


class IngestOut(BaseModel):
    document: DocumentOut
    created: bool
    chunks: int
    ocr_pages: list[int]
    warnings: list[str]
    took_ms: float


class ChunkOut(BaseModel):
    chunk_id: int
    doc_id: str
    ordinal: int
    text: str
    location: LocationOut
    previous: str | None = None
    next: str | None = None


class HealthOut(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    vectors: int
    index_version: int
    embedding_model: str
    reranker: str | None
    cache: str
    cache_ok: bool
    ocr: bool
    ocr_languages: str | None
    llm: str | None
    uploads: bool


def _document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        title=document.title,
        filename=document.filename,
        media_type=document.media_type,
        language=document.language,
        pages=document.pages,
        ocr_pages=document.ocr_pages,
        chunks=document.chunks,
        source_url=document.source_url,
        license=document.license,
        created_at=document.created_at,
        file_url=f"/api/documents/{document.id}/file",
    )


def _location(chunk: Chunk) -> LocationOut:
    return LocationOut(
        label=chunk.location_label(),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
        sheet=chunk.sheet,
        row_start=chunk.row_start,
        row_end=chunk.row_end,
    )


def _hit_out(hit: Hit) -> HitOut:
    chunk = hit.chunk
    file_url = f"/api/documents/{hit.document.id}/file"
    if hit.document.media_type == "pdf" and chunk.page_start:
        file_url += f"#page={chunk.page_start}"
    return HitOut(
        rank=hit.rank,
        chunk_id=chunk.id or 0,
        doc_id=hit.document.id,
        title=hit.document.title,
        media_type=hit.document.media_type,
        text=chunk.text,
        highlights=hit.highlights,
        location=_location(chunk),
        language=chunk.language,
        ocr_confidence=chunk.ocr_confidence,
        scores=ScoresOut(
            score=round(hit.score, 6),
            fused=None if hit.fused is None else round(hit.fused, 6),
            bm25=None if hit.bm25_score is None else round(hit.bm25_score, 4),
            bm25_rank=hit.bm25_rank,
            dense=None if hit.dense_score is None else round(hit.dense_score, 4),
            dense_rank=hit.dense_rank,
            rerank=None if hit.rerank_score is None else round(hit.rerank_score, 4),
        ),
        file_url=file_url,
    )


# --- application -------------------------------------------------------------------------------------


def get_engine(request: Request) -> SearchEngine:
    engine: SearchEngine = request.app.state.engine
    return engine


EngineDep = Annotated[SearchEngine, Depends(get_engine)]


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None) -> FastAPI:
    settings = settings or (engine.settings if engine else Settings())
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine or SearchEngine(settings)
        if settings.bootstrap_manifest and not app.state.engine.documents():
            log.info("empty index: ingesting %s", settings.bootstrap_manifest)
            ingest_manifest(app.state.engine, settings.bootstrap_manifest)
        _refresh_gauges(app.state.engine)
        yield
        if engine is None:
            app.state.engine.close()

    app = FastAPI(
        title="Lumen Search",
        version=__version__,
        description="Hybrid document search with citations: BM25 + multilingual embeddings + FAISS, "
        "over PDF, Word, Excel, Markdown and scanned pages.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def _refresh_gauges(engine: SearchEngine) -> None:
        stats = engine.stats()
        metrics.documents.set(stats["documents"])
        metrics.chunks.set(stats["chunks"])

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if not settings.api_key:
            raise HTTPException(403, "Uploads and deletions are disabled: set LUMEN_API_KEY to enable them.")
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Missing or invalid X-API-Key header.")

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "static" if not request.url.path.startswith("/api") else "unmatched")
        metrics.requests.labels(path, str(response.status_code)).inc()
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith(("/docs", "/redoc", "/api", "/metrics")):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "frame-src 'self'; object-src 'none'; frame-ancestors 'self'; base-uri 'none'",
            )
        return response

    @app.get("/api/health", response_model=HealthOut, tags=["meta"])
    def health(engine: EngineDep) -> HealthOut:
        stats = engine.stats()
        return HealthOut(
            status="ok",
            version=__version__,
            documents=stats["documents"],
            chunks=stats["chunks"],
            vectors=stats["vectors"],
            index_version=engine.index_version,
            embedding_model=engine.embedder.name,
            reranker=engine.reranker.name if engine.reranker else None,
            cache=engine.cache.backend,
            cache_ok=engine.cache.ping(),
            ocr=engine.ocr.available,
            ocr_languages=engine.ocr.languages if engine.ocr.available else None,
            llm=engine.llm.model if engine.llm else None,
            uploads=bool(settings.api_key),
        )

    @app.get("/api/search", response_model=SearchOut, tags=["search"])
    def search(
        engine: EngineDep,
        q: Annotated[str, Query(min_length=1, max_length=500, description="The query")],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        mode: SearchMode = "hybrid",
        doc: Annotated[list[str] | None, Query(description="Restrict to these document ids")] = None,
        lang: Literal["en", "ru"] | None = None,
        rerank: bool | None = None,
    ) -> SearchOut:
        result = engine.search(q, k=k, mode=mode, doc_ids=doc, language=lang, rerank=rerank)
        metrics.search_seconds.labels(mode).observe(result.took_ms / 1000)
        metrics.cache.labels("hit" if result.cached else "miss").inc()
        return SearchOut(
            query=result.query,
            mode=result.mode,
            took_ms=result.took_ms,
            cached=result.cached,
            reranked=result.reranked,
            hits=[_hit_out(h) for h in result.hits],
        )

    @app.post("/api/answer", response_model=AnswerOut, tags=["search"])
    def answer(engine: EngineDep, body: AnswerIn) -> AnswerOut:
        result = engine.answer(body.question, k=body.k, mode=body.mode, doc_ids=body.doc, language=body.lang)
        return AnswerOut(
            question=result.question,
            answer=result.answer.text,
            method=result.answer.method,
            note=result.answer.note,
            citations=result.answer.cited,
            sources=[_hit_out(h) for h in result.sources],
            took_ms=result.took_ms,
        )

    @app.get("/api/documents", response_model=list[DocumentOut], tags=["documents"])
    def documents(engine: EngineDep) -> list[DocumentOut]:
        return [_document_out(d) for d in engine.documents()]

    @app.get("/api/documents/{doc_id}", response_model=DocumentOut, tags=["documents"])
    def document(engine: EngineDep, doc_id: str) -> DocumentOut:
        found = engine.document(doc_id)
        if found is None:
            raise HTTPException(404, "No such document.")
        return _document_out(found)

    @app.get("/api/documents/{doc_id}/file", tags=["documents"], response_class=FileResponse)
    def document_file(engine: EngineDep, doc_id: str) -> FileResponse:
        found = engine.document(doc_id)
        path = engine.document_file(doc_id)
        if found is None or path is None:
            raise HTTPException(404, "No such document.")
        suffix = path.suffix.lower()
        inline = suffix in _MEDIA_TYPES or SUPPORTED_EXTENSIONS.get(suffix) == "image"
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(suffix),
            filename=found.filename,
            content_disposition_type="inline" if inline else "attachment",
        )

    @app.get("/api/chunks/{chunk_id}", response_model=ChunkOut, tags=["documents"])
    def chunk(engine: EngineDep, chunk_id: int) -> ChunkOut:
        found = engine.chunk(chunk_id)
        if found is None:
            raise HTTPException(404, "No such chunk.")
        current, before, after = found
        return ChunkOut(
            chunk_id=chunk_id,
            doc_id=current.doc_id,
            ordinal=current.ordinal,
            text=current.text,
            location=_location(current),
            previous=before.text if before else None,
            next=after.text if after else None,
        )

    @app.post(
        "/api/documents",
        response_model=IngestOut,
        status_code=201,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
        responses={200: {"description": "The same file was already indexed"}},
    )
    def upload(
        engine: EngineDep,
        response: Response,
        file: Annotated[UploadFile, File(description="PDF, DOCX, XLSX, CSV, Markdown, text, HTML or image")],
        title: Annotated[str | None, Form(max_length=200)] = None,
        ocr: Annotated[OcrMode | None, Form()] = None,
    ) -> IngestOut:
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}.")
        limit = settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / f"upload{suffix}"
            size = 0
            with target.open("wb") as out:
                while block := file.file.read(1 << 20):
                    size += len(block)
                    if size > limit:
                        raise HTTPException(413, f"The file is larger than {settings.max_upload_mb} MB.")
                    out.write(block)
            try:
                report = engine.ingest(target, title=title, filename=name, ocr_mode=ocr)
            except IngestError as exc:
                raise HTTPException(422, str(exc)) from exc
        if report.created:
            metrics.ingested.labels(report.document.media_type).inc()
            metrics.ocr_pages.inc(len(report.ocr_pages))
            _refresh_gauges(engine)
        else:
            response.status_code = 200
        return IngestOut(
            document=_document_out(report.document),
            created=report.created,
            chunks=report.chunks,
            ocr_pages=report.ocr_pages,
            warnings=report.warnings,
            took_ms=report.took_ms,
        )

    @app.delete(
        "/api/documents/{doc_id}",
        status_code=204,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def delete(engine: EngineDep, doc_id: str) -> Response:
        if not engine.delete(doc_id):
            raise HTTPException(404, "No such document.")
        _refresh_gauges(engine)
        return Response(status_code=204)

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
