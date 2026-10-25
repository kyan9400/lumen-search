"""Command line: ``lumen ingest | search | answer | docs | delete | reindex | eval | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lumen.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    if args.model:
        overrides["embedding_model"] = args.model
    if getattr(args, "reranker", None) is not None:
        overrides["reranker_model"] = args.reranker
    return Settings(**overrides)


def _engine(args: argparse.Namespace) -> Any:
    from lumen.engine import SearchEngine

    return SearchEngine(_settings(args))


def _print_hits(result: Any) -> None:
    for hit in result.hits:
        where = hit.chunk.location_label()
        print(f"{hit.rank:>2}. {hit.document.title}" + (f" — {where}" if where else ""))
        ranks = []
        if hit.bm25_rank:
            ranks.append(f"bm25 #{hit.bm25_rank}")
        if hit.dense_rank:
            ranks.append(f"dense #{hit.dense_rank} ({hit.dense_score:.3f})")
        if hit.rerank_score is not None:
            ranks.append(f"rerank {hit.rerank_score:.3f}")
        snippet = " ".join(hit.chunk.text.split())
        print(f"    {snippet[:220]}{'…' if len(snippet) > 220 else ''}")
        print(f"    [{', '.join(ranks)}]")
    print(
        f"\n{len(result.hits)} result(s) in {result.took_ms:.1f} ms" + (" (cached)" if result.cached else "")
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from lumen.corpus import ingest_manifest
    from lumen.ingest import SUPPORTED_EXTENSIONS, IngestError

    engine = _engine(args)

    def show(report: Any) -> None:
        doc = report.document
        state = "indexed" if report.created else "already indexed"
        ocr = f", OCR on {len(report.ocr_pages)} page(s)" if report.ocr_pages else ""
        print(
            f"{state}: {doc.id} ({doc.media_type}, {report.chunks} chunks{ocr}) "
            f"in {report.took_ms / 1000:.1f} s"
        )
        for warning in report.warnings:
            print(f"  warning: {warning}")

    failures = 0

    def skipped(doc_id: str, reason: str) -> None:
        nonlocal failures
        failures += 1
        print(f"skipped {doc_id}: {reason}", file=sys.stderr)

    if args.manifest:
        ingest_manifest(engine, Path(args.manifest), progress=show, failed=skipped)
    for raw in args.paths:
        path = Path(raw)
        files = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if path.is_dir()
            else [path]
        )
        for file in files:
            try:
                show(engine.ingest(file, title=args.title, ocr_mode=args.ocr))
            except IngestError as exc:
                skipped(str(file), str(exc))
    engine.close()
    return 1 if failures else 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = _engine(args)
    result = engine.search(
        args.query, k=args.k, mode=args.mode, doc_ids=args.doc, language=args.lang, rerank=args.rerank
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "rank": h.rank,
                        "doc_id": h.document.id,
                        "location": h.chunk.location_label(),
                        "score": h.score,
                        "text": h.chunk.text,
                    }
                    for h in result.hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_hits(result)
    engine.close()
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    engine = _engine(args)
    result = engine.answer(args.question, k=args.k, mode=args.mode)
    print(result.answer.text)
    if result.answer.note:
        print(f"\nnote: {result.answer.note}")
    print()
    for hit in result.sources:
        if hit.rank in result.answer.cited:
            print(f"[{hit.rank}] {hit.document.title} — {hit.chunk.location_label()}")
    engine.close()
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    engine = _engine(args)
    for doc in engine.documents():
        pages = f", {doc.pages} pages" if doc.pages else ""
        ocr = f", {doc.ocr_pages} OCR" if doc.ocr_pages else ""
        print(f"{doc.id}  [{doc.media_type}, {doc.language}{pages}{ocr}, {doc.chunks} chunks]  {doc.title}")
    stats = engine.stats()
    print(f"\n{stats['documents']} document(s), {stats['chunks']} chunk(s)")
    engine.close()
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    engine = _engine(args)
    deleted = engine.delete(args.doc_id)
    print("deleted" if deleted else "no such document")
    engine.close()
    return 0 if deleted else 1


def cmd_reindex(args: argparse.Namespace) -> int:
    """Re-embed every stored chunk with the configured model (after changing LUMEN_EMBEDDING_MODEL)."""
    from lumen.embeddings import load_embedder
    from lumen.store import Store
    from lumen.vectors import VectorIndex

    settings = _settings(args)
    store = Store(settings.db_path)
    embedder = load_embedder(settings.embedding_model, device=settings.embedding_device)
    chunks = list(store.all_chunks())
    titles = {d.id: d.title for d in store.documents()}
    index = VectorIndex(embedder.dim, settings.faiss_index)
    for start in range(0, len(chunks), 256):
        batch = chunks[start : start + 256]
        index.add(
            [c.id for c in batch],
            embedder.embed_passages([c.embedding_text(titles[c.doc_id]) for c in batch]),
        )
        print(f"\r{min(start + 256, len(chunks))}/{len(chunks)} chunks", end="", flush=True)
    index.save(settings.vectors_path)
    store.set_meta("embedding_model", embedder.name)
    store.set_meta("embedding_dim", str(embedder.dim))
    store.set_meta("index_version", str(int(store.get_meta("index_version") or 0) + 1))
    store.close()
    print(f"\nre-embedded {len(chunks)} chunks with {embedder.name}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from lumen import evaluate

    engine = _engine(args)
    questions = evaluate.load_questions(Path(args.questions))
    reports = []
    for config in args.configs.split(","):
        mode, _, extra = config.strip().partition("+")
        if extra and engine.reranker is None:
            print(f"skipping {config}: no reranker configured (--reranker)", file=sys.stderr)
            continue
        reports.append(evaluate.run(engine, questions, mode=mode, rerank=bool(extra)))
    subsets: tuple[str | None, ...] = (None, "en", "ru", "cross-lingual")
    table = evaluate.to_markdown(reports, subsets)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(evaluate.to_json(reports), indent=2), encoding="utf-8")
    engine.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from lumen.api import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen", description="Hybrid document search with citations.")
    parser.add_argument("--data-dir", help="index location (default: $LUMEN_DATA_DIR or ./data)")
    parser.add_argument(
        "--model", help="embedding model (default: $LUMEN_EMBEDDING_MODEL); 'hash' needs no model"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index files or folders")
    p.add_argument("paths", nargs="*")
    p.add_argument("--manifest", help="corpus manifest (JSON) with ids, titles, sources and licences")
    p.add_argument("--title")
    p.add_argument("--ocr", choices=["auto", "always", "never"])
    p.set_defaults(func=cmd_ingest)

    for name, func, help_text in (
        ("search", cmd_search, "search the index"),
        ("answer", cmd_answer, "answer a question"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question" if name == "answer" else "query")
        p.add_argument("-k", type=int, default=5 if name == "answer" else 10)
        p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
        p.add_argument("--reranker", help="cross-encoder model; empty string disables")
        if name == "search":
            p.add_argument("--doc", action="append", help="restrict to a document id (repeatable)")
            p.add_argument("--lang", choices=["en", "ru"])
            p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
            p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    sub.add_parser("docs", help="list indexed documents").set_defaults(func=cmd_docs)
    p = sub.add_parser("delete", help="remove a document")
    p.add_argument("doc_id")
    p.set_defaults(func=cmd_delete)
    sub.add_parser("reindex", help="re-embed all chunks with the configured model").set_defaults(
        func=cmd_reindex
    )

    p = sub.add_parser("eval", help="measure retrieval quality on labelled questions")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument(
        "--configs", default="bm25,dense,hybrid", help="comma-separated; add '+rerank', e.g. hybrid+rerank"
    )
    p.add_argument("--reranker", help="cross-encoder model for the +rerank configurations")
    p.add_argument("--out", help="write the Markdown table here")
    p.add_argument("--json-out", help="write per-question results here")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API and web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic output on Windows consoles
    return int(args.func(args))
