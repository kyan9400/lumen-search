"""Runtime configuration, read from ``LUMEN_*`` environment variables or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

OcrMode = Literal["auto", "always", "never"]
SearchMode = Literal["hybrid", "bm25", "dense"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LUMEN_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # Retrieval models. "hash" selects the model-free HashEmbedder (tests, offline runs).
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str | None = None
    embedding_batch_size: int = 32
    reranker_model: str = ""  # e.g. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1; empty disables reranking
    # reranking the top 10 gave the same Hit@1 and MRR as the top 20 on eval/ at half the latency
    rerank_top: int = Field(default=10, ge=1, le=100)
    faiss_index: Literal["flat", "hnsw"] = "flat"

    # Chunking, in words; a Russian word is about two model tokens, so 180 words stay under 512 tokens.
    chunk_words: int = Field(default=180, ge=20)
    chunk_overlap_words: int = Field(default=30, ge=0)

    # OCR: "auto" reads the PDF text layer and OCRs only pages that have none (scans).
    ocr_mode: OcrMode = "auto"
    ocr_languages: str = "eng+rus"
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    ocr_workers: int = Field(default=4, ge=1)
    tesseract_cmd: str | None = None

    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    candidates: int = Field(default=50, ge=1, description="Results taken from each retriever before fusion")

    # Cache: Redis when a URL is set, otherwise an in-process LRU.
    redis_url: str | None = None
    cache_ttl_seconds: int = 600

    # Write endpoints (upload, delete) are disabled unless an API key is set.
    api_key: str | None = None
    max_upload_mb: int = 25
    max_pages: int = 500
    cors_origins: list[str] = []

    # Optional generated answers through any OpenAI-compatible endpoint (Ollama, vLLM, OpenAI...).
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: float = 30.0

    # Ingest this corpus manifest on start-up when the index is empty (used by the Docker image).
    bootstrap_manifest: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lumen.sqlite3"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.faiss"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"
