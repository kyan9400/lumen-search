import math

import numpy as np
import pytest

from lumen.bm25 import BM25Index
from lumen.embeddings import HashEmbedder, default_prefixes
from lumen.engine import reciprocal_rank_fusion
from lumen.textproc import analyze
from lumen.vectors import VectorIndex

DOCS = {
    10: "passwords must be at least fifteen characters long",
    11: "rotate encryption keys every year",
    12: "store passwords with a salted password hashing scheme",
    13: "audit logs are protected from modification",
}


def _bm25(k1: float = 1.2, b: float = 0.75) -> BM25Index:
    index = BM25Index(k1, b)
    index.build(list(DOCS), [analyze(t) for t in DOCS.values()])
    return index


def test_bm25_matches_the_formula() -> None:
    index = _bm25()
    terms = {i: analyze(t) for i, t in DOCS.items()}
    avg = sum(len(t) for t in terms.values()) / len(terms)
    df = sum("password" in t for t in terms.values())
    idf = math.log(1 + (len(terms) - df + 0.5) / (df + 0.5))

    def expected(doc: int) -> float:
        tf = terms[doc].count("password")
        return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(terms[doc]) / avg))

    scores = dict(zip(index.ids.tolist(), index.scores(["password"]).tolist(), strict=True))
    assert scores[10] == pytest.approx(expected(10), rel=1e-5)
    assert scores[12] == pytest.approx(expected(12), rel=1e-5)
    assert scores[11] == 0.0


def test_bm25_ranks_more_matching_terms_higher() -> None:
    hits = _bm25().search(analyze("salted password hashing"), k=5)
    assert [doc for doc, _ in hits] == [12, 10]
    assert hits[0][1] > hits[1][1] > 0


def test_bm25_filter_and_unknown_terms() -> None:
    index = _bm25()
    assert [d for d, _ in index.search(analyze("passwords"), k=5, allowed={10, 11})] == [10]
    assert index.search(analyze("kerberos"), k=5) == []
    assert index.search([], k=5) == []


def test_bm25_on_an_empty_index() -> None:
    index = BM25Index()
    index.build([], [])
    assert index.search(["password"], k=3) == []


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[4] == pytest.approx(1 / 63)
    assert [item for item, _ in fused] == [1, 3, 2, 4]


def test_rrf_breaks_ties_by_best_single_rank() -> None:
    # 5 and 6 tie on score; 6 was ranked first by one retriever, so it wins
    fused = reciprocal_rank_fusion([[6, 5], [5, 6]], k=60)
    assert fused[0][1] == fused[1][1]
    assert fused[0][0] == 5  # both have best rank 1, so the lower id is first: deterministic


def _vectors(kind: str) -> tuple[VectorIndex, HashEmbedder]:
    embedder = HashEmbedder(128)
    index = VectorIndex(embedder.dim, kind)  # type: ignore[arg-type]
    index.add(list(DOCS), embedder.embed_passages(list(DOCS.values())))
    return index, embedder


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_vector_index_search_filter_remove(kind: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    index, embedder = _vectors(kind)
    query = embedder.embed_queries(["salted password hashing"])[0]
    assert index.search(query, k=1)[0][0] == 12
    assert {i for i, _ in index.search(query, k=4, allowed={11, 13})} == {11, 13}
    assert index.search(query, k=3, allowed=set()) == []
    index.remove([12])
    assert len(index) == 3
    assert 12 not in {i for i, _ in index.search(query, k=4)}
    path = tmp_path / "v.faiss"
    index.save(path)
    loaded = VectorIndex.load(path, embedder.dim, kind)  # type: ignore[arg-type]
    assert len(loaded) == 3
    assert loaded.search(query, k=1) == index.search(query, k=1)


def test_vector_index_rejects_wrong_dimensions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    index = VectorIndex(8)
    with pytest.raises(ValueError, match="size 8"):
        index.add([1], np.zeros((1, 4), dtype=np.float32))
    index.add([1], np.ones((1, 8), dtype=np.float32))
    index.save(tmp_path / "v.faiss")
    with pytest.raises(ValueError, match="8-d"):
        VectorIndex.load(tmp_path / "v.faiss", 16)


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashEmbedder(64)
    a, b = embedder.embed_passages(["Password hashing", "Password hashing"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    close = embedder.embed_queries(["passwords hashed"])[0] @ a
    far = embedder.embed_queries(["backup schedule"])[0] @ a
    assert close > far


@pytest.mark.parametrize(
    ("model", "prefixes"),
    [
        ("intfloat/multilingual-e5-small", ("query: ", "passage: ")),
        ("BAAI/bge-small-en-v1.5", ("Represent this sentence for searching relevant passages: ", "")),
        ("sentence-transformers/all-MiniLM-L6-v2", ("", "")),
    ],
)
def test_default_prefixes(model: str, prefixes: tuple[str, str]) -> None:
    assert default_prefixes(model) == prefixes
