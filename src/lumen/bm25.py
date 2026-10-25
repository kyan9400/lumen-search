"""Okapi BM25 over a sparse term-weight matrix.

Weights are computed once per (chunk, term) at build time, so a query is one sparse matrix-vector
product: the score of chunk d is the sum, over query terms t, of

    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))

with Lucene's always-positive idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self.vocabulary: dict[str, int] = {}
        self.idf: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._weights: sparse.csc_matrix = sparse.csc_matrix((0, 0), dtype=np.float32)
        self._position: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def build(self, ids: Sequence[int], documents: Sequence[Sequence[str]]) -> None:
        """Index ``documents`` (lists of analysed terms) under the given chunk ids."""
        if len(ids) != len(documents):
            raise ValueError("ids and documents differ in length")
        vocabulary: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[float] = []
        lengths = np.zeros(len(documents), dtype=np.float32)
        for row, terms in enumerate(documents):
            lengths[row] = len(terms)
            for term, count in Counter(terms).items():
                rows.append(row)
                cols.append(vocabulary.setdefault(term, len(vocabulary)))
                tfs.append(count)
        n_docs = len(documents)
        tf = np.asarray(tfs, dtype=np.float32)
        row_index = np.asarray(rows, dtype=np.int64)
        col_index = np.asarray(cols, dtype=np.int64)
        df = np.bincount(col_index, minlength=len(vocabulary)).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        average = float(lengths.mean()) if n_docs else 0.0
        norm = self.k1 * (1 - self.b + self.b * lengths[row_index] / (average or 1.0))
        weights = idf[col_index] * tf * (self.k1 + 1) / (tf + norm)
        self._weights = sparse.csc_matrix(
            (weights, (row_index, col_index)), shape=(n_docs, len(vocabulary)), dtype=np.float32
        )
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vocabulary = vocabulary
        self.idf = idf
        self._position = {int(chunk_id): i for i, chunk_id in enumerate(self.ids)}

    def scores(self, query_terms: Sequence[str]) -> npt.NDArray[np.float32]:
        """BM25 score of every indexed chunk, in ``self.ids`` order."""
        counts = Counter(t for t in query_terms if t in self.vocabulary)
        if not counts or not len(self.ids):
            return np.zeros(len(self.ids), dtype=np.float32)
        columns = [self.vocabulary[t] for t in counts]
        weights = np.asarray(list(counts.values()), dtype=np.float32)
        result = self._weights[:, columns] @ weights
        return np.asarray(result, dtype=np.float32).ravel()

    def search(
        self, query_terms: Sequence[str], k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top ``k`` ``(chunk_id, score)`` pairs with a positive score, best first."""
        scores = self.scores(query_terms)
        if allowed is not None:
            mask = np.zeros(len(self.ids), dtype=bool)
            mask[[self._position[i] for i in allowed if i in self._position]] = True
            scores = np.where(mask, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > k:
            top = np.argpartition(-scores[candidates], k - 1)[:k]
            candidates = candidates[top]
        # stable order for ties: higher score first, then the earlier chunk
        order = sorted(candidates.tolist(), key=lambda i: (-scores[i], i))
        return [(int(self.ids[i]), float(scores[i])) for i in order]
