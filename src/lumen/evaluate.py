"""Retrieval evaluation against hand-labelled questions.

Relevance is labelled by location, not by chunk id, so the labels survive changes to chunking: a target
names a document and, optionally, pages, a section, spreadsheet rows or a phrase the chunk must contain.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumen.cache import NullCache
from lumen.config import SearchMode
from lumen.engine import SearchEngine
from lumen.models import Chunk
from lumen.textproc import normalize


@dataclass
class Target:
    """One relevant location. Lists are alternatives: any listed page, row or section satisfies it."""

    doc: str
    pages: list[int] | None = None
    section: str | list[str] | None = None  # case-insensitive substring(s) of the chunk's section path
    rows: list[int] | None = None
    text: str | None = None  # normalised substring of the chunk text

    def matches(self, chunk: Chunk) -> bool:
        if chunk.doc_id != self.doc:
            return False
        if self.pages is not None:
            start = chunk.page_start or 0
            end = chunk.page_end or start
            if not any(start <= page <= end for page in self.pages):
                return False
        if self.section is not None:
            options = [self.section] if isinstance(self.section, str) else self.section
            path = (chunk.section or "").lower()
            if not any(option.lower() in path for option in options):
                return False
        if self.rows is not None:
            start = chunk.row_start or 0
            end = chunk.row_end or start
            if not any(start <= row <= end for row in self.rows):
                return False
        return self.text is None or normalize(self.text) in normalize(" ".join(chunk.text.split()))


@dataclass
class Question:
    id: str
    query: str
    lang: str
    kind: str  # "keyword", "paraphrase" or "cross-lingual"
    targets: list[Target]


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        item = json.loads(line)
        questions.append(
            Question(
                id=item["id"],
                query=item["query"],
                lang=item["lang"],
                kind=item.get("kind", "keyword"),
                targets=[Target(**t) for t in item["relevant"]],
            )
        )
    return questions


@dataclass
class Outcome:
    question: Question
    ranks: list[int]  # 1-based ranks of hits that credit a not-yet-found target
    targets_found_at: dict[int, int]  # target index -> first rank
    latency_ms: float

    def hit_at(self, k: int) -> bool:
        return any(rank <= k for rank in self.ranks)

    def recall_at(self, k: int) -> float:
        found = sum(1 for rank in self.targets_found_at.values() if rank <= k)
        return found / len(self.question.targets)

    def reciprocal_rank(self, k: int = 10) -> float:
        first = min((r for r in self.ranks if r <= k), default=None)
        return 0.0 if first is None else 1.0 / first

    def ndcg_at(self, k: int = 10) -> float:
        dcg = sum(1.0 / math.log2(rank + 1) for rank in self.ranks if rank <= k)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(self.question.targets), k) + 1))
        return dcg / ideal if ideal else 0.0


@dataclass
class Report:
    name: str
    outcomes: list[Outcome] = field(default_factory=list)

    def metrics(self, subset: str | None = None) -> dict[str, float]:
        outcomes = [o for o in self.outcomes if subset in (None, o.question.lang, o.question.kind)]
        if not outcomes:
            return {}
        latencies = sorted(o.latency_ms for o in outcomes)
        return {
            "questions": len(outcomes),
            "hit@1": statistics.fmean(o.hit_at(1) for o in outcomes),
            "hit@5": statistics.fmean(o.hit_at(5) for o in outcomes),
            "recall@10": statistics.fmean(o.recall_at(10) for o in outcomes),
            "mrr@10": statistics.fmean(o.reciprocal_rank(10) for o in outcomes),
            "ndcg@10": statistics.fmean(o.ndcg_at(10) for o in outcomes),
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)],
        }


def run(
    engine: SearchEngine, questions: list[Question], *, mode: SearchMode, rerank: bool = False, k: int = 10
) -> Report:
    """Run every question through one configuration, with the cache off so latency is honest."""
    name = mode + (" + rerank" if rerank else "")
    report = Report(name)
    saved, engine.cache = engine.cache, NullCache()
    try:
        engine.search(questions[0].query, k=k, mode=mode, rerank=rerank)  # warm-up: model load, JIT
        for question in questions:
            result = engine.search(question.query, k=k, mode=mode, rerank=rerank)
            ranks: list[int] = []
            found: dict[int, int] = {}
            for hit in result.hits:
                new = [i for i, t in enumerate(question.targets) if i not in found and t.matches(hit.chunk)]
                for i in new:
                    found[i] = hit.rank
                if new:
                    ranks.append(hit.rank)
            report.outcomes.append(Outcome(question, ranks, found, result.took_ms))
    finally:
        engine.cache = saved
    return report


def to_markdown(reports: list[Report], subsets: tuple[str | None, ...] = (None,)) -> str:
    lines = []
    for subset in subsets:
        rows = [(r.name, r.metrics(subset)) for r in reports]
        rows = [(name, m) for name, m in rows if m]
        if not rows:
            continue
        count = int(rows[0][1]["questions"])
        lines.append(f"**{subset or 'All questions'}** ({count})\n")
        lines.append("| Retrieval | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, m in rows:
            lines.append(
                f"| {name} | {m['hit@1']:.1%} | {m['hit@5']:.1%} | {m['recall@10']:.1%} | {m['mrr@10']:.3f} "
                f"| {m['ndcg@10']:.3f} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def to_json(reports: list[Report]) -> dict[str, Any]:
    return {
        r.name: {
            "metrics": r.metrics(),
            "questions": {
                o.question.id: {"ranks": o.ranks, "recall@10": o.recall_at(10), "latency_ms": o.latency_ms}
                for o in r.outcomes
            },
        }
        for r in reports
    }
