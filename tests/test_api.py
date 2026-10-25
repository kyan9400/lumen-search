from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import EngineFactory
from lumen.api import create_app
from lumen.engine import SearchEngine

KEY = "test-key-123"


@pytest.fixture
def indexed(make_engine: EngineFactory, docs: Path) -> SearchEngine:
    engine = make_engine()
    for path in sorted(docs.iterdir()):
        engine.ingest(path)
    return engine


@pytest.fixture
def client(indexed: SearchEngine) -> Iterator[TestClient]:
    indexed.settings.api_key = KEY
    with TestClient(create_app(engine=indexed)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 5
    assert body["chunks"] == body["vectors"]
    assert body["embedding_model"] == "hash-256"
    assert body["cache"] == "memory"
    assert body["uploads"] is True


def test_search_returns_cited_hits(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": "idle sessions inactivity", "k": 3})
    assert response.status_code == 200
    body = response.json()
    top = body["hits"][0]
    assert top["title"] == "Remote Access Guide"
    assert top["location"]["page_start"] == 3
    assert top["location"]["label"].startswith("p. 3")
    assert top["file_url"].endswith("/file#page=3")
    assert top["scores"]["bm25_rank"] is not None
    start, end = top["highlights"][0]
    assert top["text"][start:end].lower() in {"idle", "sessions", "inactivity"}


def test_search_validation(client: TestClient) -> None:
    assert client.get("/api/search").status_code == 422
    assert client.get("/api/search", params={"q": "x", "k": 500}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_search_is_cached(client: TestClient) -> None:
    params = {"q": "virtual private network"}
    assert client.get("/api/search", params=params).json()["cached"] is False
    assert client.get("/api/search", params=params).json()["cached"] is True


def test_answer(client: TestClient) -> None:
    body = client.post("/api/answer", json={"question": "How long are audit logs kept?"}).json()
    assert body["method"] == "extractive"
    assert "one year" in body["answer"]
    assert body["citations"]
    assert body["sources"][body["citations"][0] - 1]["title"] == "Access Policy"


def test_documents_file_and_chunk(client: TestClient) -> None:
    documents = client.get("/api/documents").json()
    assert {d["media_type"] for d in documents} == {"docx", "markdown", "pdf", "text", "xlsx"}
    pdf = next(d for d in documents if d["media_type"] == "pdf")
    assert client.get(f"/api/documents/{pdf['id']}").json()["pages"] == 3
    file = client.get(pdf["file_url"])
    assert file.status_code == 200
    assert file.headers["content-type"] == "application/pdf"
    assert file.content.startswith(b"%PDF")
    hit = client.get("/api/search", params={"q": "virtual private network"}).json()["hits"][0]
    chunk = client.get(f"/api/chunks/{hit['chunk_id']}").json()
    assert chunk["text"] == hit["text"]
    assert client.get("/api/documents/missing").status_code == 404
    assert client.get("/api/chunks/999999").status_code == 404


def test_upload_needs_the_api_key(client: TestClient, docs: Path) -> None:
    files = {"file": ("notes2.md", b"# Keys\n\nRotate keys every year.", "text/markdown")}
    assert client.post("/api/documents", files=files).status_code == 401
    assert client.post("/api/documents", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
    created = client.post("/api/documents", files=files, headers={"X-API-Key": KEY})
    assert created.status_code == 201
    body = created.json()
    assert body["created"] is True
    assert body["document"]["title"] == "Keys"
    again = client.post("/api/documents", files=files, headers={"X-API-Key": KEY})
    assert again.status_code == 200
    assert again.json()["created"] is False
    hits = client.get("/api/search", params={"q": "rotate keys"}).json()["hits"]
    assert hits[0]["doc_id"] == body["document"]["id"]


def test_upload_rejects_bad_files(client: TestClient) -> None:
    headers = {"X-API-Key": KEY}
    exe = {"file": ("tool.exe", b"MZ", "application/octet-stream")}
    assert client.post("/api/documents", files=exe, headers=headers).status_code == 415
    fake = {"file": ("fake.pdf", b"PK\x03\x04 not a pdf", "application/pdf")}
    response = client.post("/api/documents", files=fake, headers=headers)
    assert response.status_code == 422
    assert "not a PDF" in response.json()["detail"]


def test_upload_size_limit(client: TestClient, indexed: SearchEngine) -> None:
    indexed.settings.max_upload_mb = 1
    big = {"file": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")}
    assert client.post("/api/documents", files=big, headers={"X-API-Key": KEY}).status_code == 413


def test_delete_document(client: TestClient) -> None:
    doc_id = client.get("/api/documents").json()[0]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 401
    assert client.delete(f"/api/documents/{doc_id}", headers={"X-API-Key": KEY}).status_code == 204
    assert client.delete(f"/api/documents/{doc_id}", headers={"X-API-Key": KEY}).status_code == 404
    assert client.get("/api/health").json()["documents"] == 4


def test_writes_are_disabled_without_a_configured_key(indexed: SearchEngine) -> None:
    indexed.settings.api_key = None
    with TestClient(create_app(engine=indexed)) as client:
        files = {"file": ("a.md", b"# A\n\ntext", "text/markdown")}
        response = client.post("/api/documents", files=files, headers={"X-API-Key": "anything"})
        assert response.status_code == 403
        assert client.get("/api/health").json()["uploads"] is False


def test_metrics_and_security_headers(client: TestClient) -> None:
    client.get("/api/search", params={"q": "backup"})
    metrics = client.get("/metrics").text
    assert 'lumen_http_requests_total{route="/api/search",status="200"}' in metrics
    assert "lumen_search_seconds_bucket" in metrics
    assert "lumen_documents 5.0" in metrics
    script = client.get("/app.js")
    assert script.headers["content-type"].startswith("text/javascript")  # not text/plain from the registry
    page = client.get("/")
    assert page.status_code == 200
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
