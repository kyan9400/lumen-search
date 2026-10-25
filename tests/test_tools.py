"""The command line, corpus manifests, the evaluation harness and the static demo export."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import EngineFactory
from lumen.cli import main
from lumen.corpus import ingest_manifest, read_manifest
from lumen.engine import SearchEngine
from lumen.evaluate import Outcome, Question, Report, Target, load_questions, run, to_json, to_markdown
from lumen.export import export_demo, quantize
from lumen.models import Chunk


def _manifest(folder: Path, docs: Path) -> Path:
    path = folder / "manifest.json"
    entries = [
        {"id": "guide", "file": str(docs / "guide.pdf"), "title": "Remote Access Guide", "license": "test"},
        {"id": "law", "file": str(docs / "law.txt"), "source_url": "https://example.org/law", "ocr": "never"},
        {"id": "broken", "file": str(folder / "broken.pdf")},
    ]
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 truncated")
    path.write_text(json.dumps({"documents": entries}), encoding="utf-8")
    return path


def test_manifest_ingest_skips_broken_files_and_existing_ids(
    engine: SearchEngine, docs: Path, tmp_path: Path
) -> None:
    manifest = _manifest(tmp_path, docs)
    assert [e.id for e in read_manifest(manifest)] == ["guide", "law", "broken"]
    failures: list[str] = []
    reports = ingest_manifest(engine, manifest, failed=lambda doc_id, reason: failures.append(doc_id))
    assert [r.document.id for r in reports] == ["guide", "law"]
    assert failures == ["broken"]
    assert engine.document("law").source_url == "https://example.org/law"  # type: ignore[union-attr]
    assert ingest_manifest(engine, manifest, failed=lambda *_: None) == []  # already indexed


def test_manifest_rejects_unknown_ocr_modes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"documents": [{"id": "a", "file": "a.pdf", "ocr": "sometimes"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ocr must be"):
        read_manifest(path)


def _chunk(doc: str, **fields: object) -> Chunk:
    return Chunk(doc_id=doc, ordinal=0, text=str(fields.pop("text", "")), **fields)  # type: ignore[arg-type]


def test_targets_match_by_location() -> None:
    page = _chunk("d", page_start=4, page_end=5, section="3.1 Passwords", text="Minimum of 15 characters.")
    assert Target("d", pages=[5]).matches(page)
    assert not Target("d", pages=[6]).matches(page)
    assert Target("d", section=["Appendix", "3.1 passwords"]).matches(page)
    assert Target("d", text="minimum of 15").matches(page)
    assert not Target("other").matches(page)
    row = _chunk("x", sheet="S", row_start=41, row_end=41)
    assert Target("x", rows=[40, 41]).matches(row)
    assert not Target("x", rows=[42]).matches(row)


def test_metrics_credit_each_target_once() -> None:
    question = Question("q", "query", "en", "keyword", [Target("a"), Target("b")])
    outcome = Outcome(question, ranks=[2, 4], targets_found_at={0: 2, 1: 4}, latency_ms=10)
    assert outcome.hit_at(1) is False
    assert outcome.hit_at(5) is True
    assert outcome.recall_at(3) == 0.5
    assert outcome.reciprocal_rank() == 0.5
    ideal = 1 + 1 / np.log2(3)
    assert outcome.ndcg_at(10) == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)
    report = Report("hybrid", [outcome])
    metrics = report.metrics()
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr@10"] == 0.5
    assert report.metrics("ru") == {}


def test_evaluation_run_and_reports(engine: SearchEngine, docs: Path, tmp_path: Path) -> None:
    engine.ingest(docs / "guide.pdf", doc_id="guide")
    engine.ingest(docs / "notes.md", doc_id="notes")
    questions_file = tmp_path / "questions.jsonl"
    questions_file.write_text(
        "// comment lines are skipped\n"
        + json.dumps(
            {
                "id": "q1",
                "lang": "en",
                "query": "idle session timeout",
                "relevant": [{"doc": "guide", "pages": [3]}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "q2",
                "lang": "en",
                "kind": "paraphrase",
                "query": "bcrypt input limit",
                "relevant": [{"doc": "notes", "section": "bcrypt"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions = load_questions(questions_file)
    report = run(engine, questions, mode="hybrid")
    assert report.metrics()["hit@5"] == 1.0
    table = to_markdown([report], (None, "paraphrase"))
    assert "| hybrid | 100.0% |" in table
    assert "**paraphrase** (1)" in table
    assert to_json([report])["hybrid"]["questions"]["q1"]["ranks"]
    assert engine.cache.backend == "memory"  # the cache is restored after the run


def test_quantize_round_trip() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    quantized, scales = quantize(vectors)
    assert quantized.dtype == np.int8
    restored = quantized.astype(np.float32) * scales[:, None]
    assert np.abs(restored - vectors).max() < 0.01


def test_export_demo_writes_index_vectors_and_files(engine: SearchEngine, docs: Path, tmp_path: Path) -> None:
    for path in sorted(docs.iterdir()):
        engine.ingest(path)
    out = tmp_path / "site"
    summary = export_demo(engine, out, model={"id": "test-model", "download_mb": 1})
    corpus = json.loads((out / "data" / "corpus.json").read_text(encoding="utf-8"))
    assert summary["chunks"] == len(corpus["chunks"]) == engine.stats()["chunks"]
    assert corpus["model"] == {
        "name": "hash-256",
        "dim": 256,
        "query_prefix": "",
        "id": "test-model",
        "download_mb": 1,
    }
    assert set(corpus["stopwords"]) >= {"the", "и"}
    vectors = np.fromfile(out / "data" / "vectors.bin", dtype=np.int8).reshape(len(corpus["chunks"]), 256)
    scales = np.fromfile(out / "data" / "scales.bin", dtype=np.float32)
    assert len(scales) == len(vectors)
    for document in corpus["documents"]:
        assert (out / document["file"]).exists()
    xlsx = next(i for i, d in enumerate(corpus["documents"]) if d["media_type"] == "xlsx")
    row = next(c for c in corpus["chunks"] if c["d"] == xlsx)
    assert row["sh"] == "Controls"
    assert row["r"] == [2, 2]


def test_cli_end_to_end(docs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--data-dir", str(tmp_path / "cli-data"), "--model", "hash"]
    assert main([*base, "ingest", str(docs)]) == 0
    assert "indexed:" in capsys.readouterr().out
    assert main([*base, "docs"]) == 0
    assert "5 document(s)" in capsys.readouterr().out
    assert main([*base, "search", "virtual private network", "-k", "2"]) == 0
    assert "Remote Access Guide" in capsys.readouterr().out
    assert main([*base, "search", "audit logs", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["doc_id"]
    assert main([*base, "answer", "How long are audit logs kept?"]) == 0
    assert "one year" in capsys.readouterr().out
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps({"id": "q1", "lang": "en", "query": "vpn", "relevant": [{"doc": hits[0]["doc_id"]}]})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "eval.md"
    assert (
        main([*base, "eval", "--questions", str(questions), "--configs", "bm25,hybrid", "--out", str(out)])
        == 0
    )
    assert "| hybrid |" in out.read_text(encoding="utf-8")
    assert main([*base, "reindex"]) == 0
    assert "re-embedded" in capsys.readouterr().out
    assert main([*base, "delete", hits[0]["doc_id"]]) == 0
    assert main([*base, "delete", hits[0]["doc_id"]]) == 1


def test_cli_reports_files_it_cannot_parse(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert main(["--data-dir", str(tmp_path / "d"), "--model", "hash", "ingest", str(bad)]) == 1
    assert "skipped" in capsys.readouterr().err


def test_engine_factory_is_isolated(make_engine: EngineFactory) -> None:
    assert make_engine().stats() == {"documents": 0, "chunks": 0, "vectors": 0}
