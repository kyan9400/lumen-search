"""Dense vector search with FAISS (inner product over L2-normalised vectors = cosine similarity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import faiss
import numpy as np
import numpy.typing as npt

IndexKind = Literal["flat", "hnsw"]


def _new_index(dim: int, kind: IndexKind) -> faiss.IndexIDMap2:
    base: faiss.Index
    if kind == "hnsw":
        # 32 neighbours per node: a common balance of recall, memory and build time
        base = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = 80
    else:
        base = faiss.IndexFlatIP(dim)  # exact search: fast enough up to a few hundred thousand chunks
    return faiss.IndexIDMap2(base)


def _search_params(kind: IndexKind, k: int, selector: Any = None) -> Any:
    # faiss's type stubs lack SearchParametersHNSW and the constructor keywords, so set attributes
    params: Any = faiss.SearchParametersHNSW() if kind == "hnsw" else faiss.SearchParameters()  # type: ignore[attr-defined]
    if kind == "hnsw":
        params.efSearch = max(64, k)
    if selector is not None:
        params.sel = selector
    return params


class VectorIndex:
    def __init__(self, dim: int, kind: IndexKind = "flat") -> None:
        self.dim = dim
        self.kind = kind
        self.index = _new_index(dim, kind)

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: npt.ArrayLike, vectors: npt.NDArray[np.float32]) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of size {self.dim}, got shape {vectors.shape}")
        if len(ids):
            self.index.add_with_ids(vectors, ids)

    def remove(self, ids: npt.ArrayLike) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        if self.kind == "flat":
            self.index.remove_ids(faiss.IDSelectorBatch(ids))
            return
        # HNSW graphs do not support deletion: rebuild from the vectors that remain
        doomed = set(ids.tolist())
        keep = np.asarray([i for i in self.ids() if i not in doomed])
        vectors = np.vstack([self.index.reconstruct(int(i)) for i in keep]) if len(keep) else None
        self.index = _new_index(self.dim, self.kind)
        if vectors is not None:
            self.add(keep, vectors)

    def search(
        self, vector: npt.NDArray[np.float32], k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top ``k`` ``(chunk_id, cosine)`` pairs, best first, optionally restricted to ``allowed`` ids."""
        if not len(self) or k <= 0:
            return []
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        selector = None
        if allowed is not None:
            if not allowed:
                return []
            selector = faiss.IDSelectorBatch(np.fromiter(allowed, dtype=np.int64))
        params = _search_params(self.kind, k, selector)  # keeps a reference to selector during the search
        scores, ids = self.index.search(query, min(k, len(self)), params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def ids(self) -> list[int]:
        return [int(i) for i in faiss.vector_to_array(self.index.id_map)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(self.index, str(temporary))
        temporary.replace(path)  # atomic: a crash never leaves a half-written index

    @classmethod
    def load(cls, path: Path, dim: int, kind: IndexKind = "flat") -> VectorIndex:
        instance = cls(dim, kind)
        if path.exists():
            index = faiss.read_index(str(path))
            if index.d != dim:
                raise ValueError(f"{path} holds {index.d}-d vectors; the embedding model makes {dim}-d ones")
            instance.index = cast(faiss.IndexIDMap2, index)
        return instance
