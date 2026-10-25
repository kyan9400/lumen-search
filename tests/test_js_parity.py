"""The static demo re-implements tokenisation, BM25 and fusion in JavaScript; both must agree."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import MARKDOWN, RUSSIAN_TEXT
from lumen.bm25 import BM25Index
from lumen.engine import reciprocal_rank_fusion
from lumen.textproc import STOPWORDS, analyze, split_sentences

NODE = shutil.which("node")
CLI = Path(__file__).parent / "js" / "engine-cli.mjs"

TEXTS = [
    "Verifiers SHALL NOT require subscribers to change passwords periodically (see Sec. 3.1.1.2).",
    "IA-5(1) Authenticator Management | Password-based Authentication — e.g. bcrypt, Argon2id.",
    "Хранение персональных данных должно осуществляться не дольше, чем этого требуют цели обработки.",
    "Оператор обязан ЁЛКУ, ёжиков и т.е. «кавычки» — № 152-ФЗ, ст. 18.1. Далее текст.",
    "Mixed: Статья 10.1 and Article 10.1; ﬁle ligatures, soft\N{SOFT HYPHEN}hyphens, ½ fractions.",
    MARKDOWN,
    RUSSIAN_TEXT,
]
QUERIES = [
    "minimum password length",
    "хранение персональных данных",
    "Argon2id bcrypt",
    "ст. 18.1",
    "nothing-matches",
]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_javascript_engine_matches_python() -> None:
    documents = TEXTS + [line for line in MARKDOWN.splitlines() if line.strip()]
    rankings = [[3, 1, 2, 9], [1, 3, 7], [8, 2]]
    request = {
        "stopwords": sorted(STOPWORDS),
        "texts": TEXTS,
        "documents": documents,
        "queries": QUERIES,
        "bm25": {"k1": 1.2, "b": 0.75},
        "rankings": rankings,
    }
    output = subprocess.run(
        [NODE, str(CLI)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    js = json.loads(output.stdout)

    assert js["analyze"] == [analyze(t) for t in TEXTS]
    assert js["sentences"] == [split_sentences(t) for t in TEXTS]

    index = BM25Index(1.2, 0.75)
    index.build(list(range(len(documents))), [analyze(d) for d in documents])
    for query, js_hits in zip(QUERIES, js["bm25"], strict=True):
        python_hits = index.search(analyze(query), k=5)
        assert [doc for doc, _ in js_hits] == [doc for doc, _ in python_hits], query
        for (_, js_score), (_, py_score) in zip(js_hits, python_hits, strict=True):
            assert js_score == pytest.approx(py_score, rel=1e-5)

    assert [[item, pytest.approx(score)] for item, score in reciprocal_rank_fusion(rankings)] == js["rrf"]
