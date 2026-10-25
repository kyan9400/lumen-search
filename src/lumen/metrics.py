"""Prometheus metrics, kept in a registry per app so tests can create many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "lumen_http_requests_total",
            "HTTP requests by route and status code",
            ["route", "status"],
            registry=self.registry,
        )
        self.search_seconds = Histogram(
            "lumen_search_seconds",
            "Search latency by mode (cache hits included)",
            ["mode"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry,
        )
        self.cache = Counter(
            "lumen_search_cache_total", "Search cache lookups", ["result"], registry=self.registry
        )
        self.ingested = Counter(
            "lumen_documents_ingested_total", "Documents indexed", ["media_type"], registry=self.registry
        )
        self.ocr_pages = Counter("lumen_ocr_pages_total", "Pages read with OCR", registry=self.registry)
        self.chunks = Gauge("lumen_chunks", "Chunks in the index", registry=self.registry)
        self.documents = Gauge("lumen_documents", "Documents in the index", registry=self.registry)
