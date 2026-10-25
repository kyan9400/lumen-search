"""Answers with citations: extractive by default, generated when an LLM endpoint is configured."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx
import numpy as np

from lumen.embeddings import Embedder
from lumen.models import Chunk, Document
from lumen.textproc import analyze, split_sentences

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Source:
    """A retrieved chunk offered to the answerer, numbered from 1 as cited in the answer."""

    number: int
    chunk: Chunk
    document: Document


@dataclass
class Answer:
    text: str
    method: str  # "extractive", "llm", or "none" when nothing relevant was found
    cited: list[int] = field(default_factory=list)  # source numbers used in the text
    note: str | None = None


NOT_FOUND = {
    "en": "I could not find this in the indexed documents.",
    "ru": "В проиндексированных документах ответ не найден.",
}


def _language(question: str) -> str:
    return "ru" if re.search(r"[Ѐ-ӿ]", question) else "en"


def _shouting(sentence: str) -> bool:
    """Mostly capital letters: code listings, table headers, running titles; not an answer sentence."""
    letters = [c for c in sentence if c.isalpha()]
    return len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6


def extractive_answer(
    question: str, sources: Sequence[Source], embedder: Embedder, *, max_sentences: int = 2
) -> Answer:
    """Pick the sentences of the top sources that best answer the question.

    Each sentence is scored by embedding similarity to the question plus a small bonus for sharing its
    terms; near-duplicates are skipped. Every sentence keeps the number of the source it came from.
    """
    candidates: list[tuple[int, int, str]] = []  # (source number, position, sentence)
    for source in sources[:3]:
        position = 0
        for line in source.chunk.text.split("\n"):
            for sentence in split_sentences(line):
                # a fragment (lower-case start) or a code listing (capitals) makes a poor answer
                if len(sentence.split()) >= 4 and not sentence[0].islower() and not _shouting(sentence):
                    candidates.append((source.number, position, sentence))
                    position += 1
    if not candidates:
        return Answer(NOT_FOUND[_language(question)], method="none")
    query_vector = embedder.embed_queries([question])[0]
    vectors = embedder.embed_passages([sentence for _, _, sentence in candidates])
    similarity = vectors @ query_vector
    query_terms = set(analyze(question))
    scores = []
    for (number, _, sentence), cosine in zip(candidates, similarity, strict=True):
        terms = set(analyze(sentence))
        overlap = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        scores.append(float(cosine) + 0.15 * overlap - 0.02 * (number - 1))
    picked: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if len(picked) == max_sentences:
            break
        if any(float(vectors[index] @ vectors[p]) > 0.92 for p in picked):
            continue
        picked.append(int(index))
    picked.sort(key=lambda i: (candidates[i][0], candidates[i][1]))
    text = " ".join(f"{candidates[i][2]} [{candidates[i][0]}]" for i in picked)
    cited = sorted({candidates[i][0] for i in picked})
    return Answer(text, method="extractive", cited=cited)


SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. Cite every claim with the source "
    "number in square brackets, like [2]. If the sources do not contain the answer, say so plainly and do "
    "not guess. Answer in the language of the question, in at most five sentences."
)


def build_prompt(question: str, sources: Sequence[Source]) -> list[dict[str, str]]:
    blocks = []
    for source in sources:
        where = source.chunk.location_label()
        header = f"[{source.number}] {source.document.title}" + (f" ({where})" if where else "")
        blocks.append(f"{header}\n{source.chunk.text}")
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class LlmClient:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint (Ollama, vLLM, OpenAI)."""

    def __init__(
        self, base_url: str, model: str, *, api_key: str | None = None, timeout: float = 30.0
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.post(
            "/chat/completions", json={"model": self.model, "messages": messages, "temperature": 0.1}
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    def close(self) -> None:
        self._client.close()


def generated_answer(question: str, sources: Sequence[Source], client: LlmClient) -> Answer:
    text = client.complete(build_prompt(question, sources))
    valid = {source.number for source in sources}
    cited = sorted({int(n) for n in _CITATION.findall(text) if int(n) in valid})
    note = None if cited else "The model's answer cites no source; treat it with care."
    return Answer(text, method="llm", cited=cited, note=note)
