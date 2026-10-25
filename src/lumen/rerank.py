"""Optional second stage: a cross-encoder rescores the fused top candidates against the query."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str | None = None, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict(
            [(query, passage) for passage in passages], batch_size=16, show_progress_bar=False
        )
        return [float(s) for s in scores]
