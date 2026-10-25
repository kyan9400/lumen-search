import json
from pathlib import Path

import httpx
import pytest

from conftest import EngineFactory, FakeOcr
from lumen.answer import LlmClient
from lumen.embeddings import HashEmbedder
from lumen.engine import SearchEngine, slugify
from lumen.ingest import IngestError


def _ingest_all(engine: SearchEngine, docs: Path) -> None:
    for path in sorted(docs.iterdir()):
        engine.ingest(path)


def test_ingest_indexes_every_format(engine: SearchEngine, docs: Path) -> None:
    _ingest_all(engine, docs)
    kinds = sorted(d.media_type for d in engine.documents())
    assert kinds == ["docx", "markdown", "pdf", "text", "xlsx"]
    stats = engine.stats()
    assert stats["chunks"] == stats["vectors"] > 0
    assert engine.index_version == 5


def test_ingesting_the_same_file_twice_is_a_no_op(engine: SearchEngine, docs: Path) -> None:
    first = engine.ingest(docs / "notes.md")
    second = engine.ingest(docs / "notes.md", title="Another title")
    assert first.created and not second.created
    assert second.document.id == first.document.id
    assert len(engine.documents()) == 1


def test_document_ids_and_explicit_ids(engine: SearchEngine, docs: Path) -> None:
    report = engine.ingest(docs / "law.txt", doc_id="ru-law")
    assert report.document.id == "ru-law"
    assert engine.document_file("ru-law") is not None
    with pytest.raises(IngestError, match="already exists"):
        engine.ingest(docs / "notes.md", doc_id="ru-law")
    with pytest.raises(IngestError, match="lower-case"):
        engine.ingest(docs / "guide.pdf", doc_id="Bad Id")


def test_slugify_transliterates_russian() -> None:
    assert slugify("О персональных данных") == "o-personalnykh-dannykh"
    assert slugify("NIST SP 800-63B-4: Authentication") == "nist-sp-800-63b-4-authentication"
    assert slugify("!!!") == "document"


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_search_modes_find_the_right_document(engine: SearchEngine, docs: Path, mode: str) -> None:
    _ingest_all(engine, docs)
    result = engine.search("idle sessions inactivity timeout", k=3, mode=mode)  # type: ignore[arg-type]
    assert result.hits[0].document.title == "Remote Access Guide"
    assert result.hits[0].chunk.page_start == 3
    top = result.hits[0]
    if mode == "hybrid":
        assert top.bm25_rank is not None and top.dense_rank is not None
    assert top.highlights  # query terms are marked in the chunk text


def test_search_filters_by_document_and_language(engine: SearchEngine, docs: Path) -> None:
    _ingest_all(engine, docs)
    by_doc = engine.search("data", k=10, doc_ids=["law"])
    assert by_doc.hits == [] or {h.document.id for h in by_doc.hits} == {"law"}
    law_id = next(d.id for d in engine.documents() if d.media_type == "text")
    russian = engine.search("хранение персональных данных", k=10, language="ru")
    assert russian.hits and {h.document.id for h in russian.hits} == {law_id}
    assert russian.hits[0].chunk.section and "Статья 5" in russian.hits[0].chunk.section


def test_search_results_are_cached_until_the_index_changes(engine: SearchEngine, docs: Path) -> None:
    engine.ingest(docs / "guide.pdf")
    first = engine.search("virtual private network")
    again = engine.search("virtual private network")
    assert not first.cached and again.cached
    assert [h.chunk.id for h in again.hits] == [h.chunk.id for h in first.hits]
    engine.ingest(docs / "notes.md")  # bumps the index version, so the cached entry no longer applies
    assert not engine.search("virtual private network").cached


def test_delete_removes_chunks_vectors_and_file(engine: SearchEngine, docs: Path) -> None:
    report = engine.ingest(docs / "guide.pdf")
    engine.ingest(docs / "notes.md")
    doc_id = report.document.id
    file = engine.document_file(doc_id)
    assert file is not None
    assert engine.delete(doc_id)
    assert not engine.delete(doc_id)
    assert not file.exists()
    assert all(h.document.id != doc_id for h in engine.search("remote access vpn").hits)
    assert engine.stats()["vectors"] == engine.stats()["chunks"]


def test_restart_reloads_the_index(make_engine: EngineFactory, docs: Path) -> None:
    first = make_engine()
    first.ingest(docs / "guide.pdf")
    expected = [h.chunk.id for h in first.search("multi-factor vpn").hits]
    first.close()
    second = make_engine()
    assert [h.chunk.id for h in second.search("multi-factor vpn").hits] == expected


def test_vectors_missing_after_a_crash_are_rebuilt(make_engine: EngineFactory, docs: Path) -> None:
    first = make_engine()
    first.ingest(docs / "guide.pdf")
    first.close()
    first.settings.vectors_path.unlink()  # as if the process died between the SQLite and FAISS writes
    second = make_engine()
    assert second.stats()["vectors"] == second.stats()["chunks"] > 0


def test_changing_the_embedding_model_requires_a_reindex(make_engine: EngineFactory, docs: Path) -> None:
    make_engine().ingest(docs / "notes.md")
    with pytest.raises(RuntimeError, match="reindex"):
        make_engine(embedder=HashEmbedder(128))


def test_extractive_answer_cites_its_source(engine: SearchEngine, docs: Path) -> None:
    _ingest_all(engine, docs)
    result = engine.answer("How long are audit logs kept?")
    assert result.answer.method == "extractive"
    assert "one year" in result.answer.text
    cited = [h for h in result.sources if h.rank in result.answer.cited]
    assert cited and cited[0].document.title == "Access Policy"
    assert "[" in result.answer.text


def test_answer_with_nothing_indexed(engine: SearchEngine) -> None:
    result = engine.answer("anything at all?")
    assert result.answer.method == "none"
    assert result.sources == []


def _llm(handler: httpx.MockTransport) -> LlmClient:
    client = LlmClient("http://llm.test/v1", "test-model")
    client._client = httpx.Client(base_url="http://llm.test/v1", transport=handler)
    return client


def test_generated_answer_sends_numbered_sources(make_engine: EngineFactory, docs: Path) -> None:
    seen: dict[str, object] = {}

    def reply(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Logs are kept for one year [1]."}}]}
        )

    engine = make_engine(llm=_llm(httpx.MockTransport(reply)))
    _ingest_all(engine, docs)
    result = engine.answer("How long are audit logs kept?")
    assert result.answer.method == "llm"
    assert result.answer.cited == [1]
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert "[1] Access Policy" in messages[1]["content"]
    assert seen["model"] == "test-model"


def test_generated_answer_falls_back_when_the_llm_fails(make_engine: EngineFactory, docs: Path) -> None:
    engine = make_engine(llm=_llm(httpx.MockTransport(lambda request: httpx.Response(503))))
    _ingest_all(engine, docs)
    result = engine.answer("How long are audit logs kept?")
    assert result.answer.method == "extractive"
    assert result.answer.note and "unavailable" in result.answer.note


def test_ocr_pages_are_reported(make_engine: EngineFactory, scanned_pdf: Path) -> None:
    engine = make_engine(ocr=FakeOcr())
    report = engine.ingest(scanned_pdf)
    assert report.ocr_pages == [1]
    assert report.document.ocr_pages == 1
    hit = engine.search("suspected compromise").hits[0]
    assert hit.chunk.ocr_confidence == 91.5
