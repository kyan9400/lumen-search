"""Text embedders: sentence-transformers models, and a model-free hashing embedder for tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from lumen.textproc import analyze, normalize

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_queries(self, texts: Sequence[str]) -> Vectors: ...

    def embed_passages(self, texts: Sequence[str]) -> Vectors: ...


def default_prefixes(model_name: str) -> tuple[str, str]:
    """Instruction prefixes the model was trained with (E5 expects "query: " and "passage: ")."""
    name = model_name.lower()
    if "e5" in name and "instruct" not in name:
        return "query: ", "passage: "
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # renamed to get_embedding_dimension in sentence-transformers 6
        get_dimension = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        dim = get_dimension()
        if dim is None:
            raise ValueError(f"{model_name} does not report an embedding size")
        self.dim = int(dim)
        self.batch_size = batch_size
        default_query, default_passage = default_prefixes(model_name)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.passage_prefix = default_passage if passage_prefix is None else passage_prefix

    def _encode(self, texts: Sequence[str]) -> Vectors:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self.query_prefix + text for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self.passage_prefix + text for text in texts])


class HashEmbedder:
    """Signed feature hashing of stemmed words and character trigrams, L2-normalised.

    It is lexical, not semantic, but deterministic and instant: tests and offline runs use it so they
    never download a model.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        terms = analyze(text)
        grams: list[str] = []
        for word in normalize(text).split():
            padded = f"#{word}#"
            grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return [f"w:{t}" for t in terms] + [f"c:{g}" for g in grams]

    def _embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                weight = 2.0 if feature.startswith("w:") else 1.0
                out[row, value % self.dim] += weight if value >> 63 else -weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts)


def load_embedder(model_name: str, *, device: str | None = None, batch_size: int = 32) -> Embedder:
    if model_name == "hash" or model_name.startswith("hash-"):
        dim = int(model_name.split("-", 1)[1]) if "-" in model_name else 512
        return HashEmbedder(dim)
    return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
