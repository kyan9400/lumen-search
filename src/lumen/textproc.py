"""Normalisation, tokenisation, stemming and sentence splitting for English and Russian text.

The browser demo re-implements ``analyze`` in JavaScript (``web/engine.js``) with the same stop words
and Snowball stemmers, so a query is tokenised identically on the server and in the page.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

# Letters and digits; underscores and punctuation split tokens ("IA-5(1)" -> ia, 5, 1).
_TOKEN = re.compile(r"[^\W_]+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)

_STEMMERS = {
    "en": snowballstemmer.stemmer("english"),
    "ru": snowballstemmer.stemmer("russian"),
}

STOPWORDS_EN = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most my
    myself no nor not of off on once only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your yours
    yourself yourselves must may might also""".split()
)

STOPWORDS_RU = frozenset(
    """а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до
    его ее ей ему если есть еще же за здесь и из или им их к как какая какие какой когда кто ли либо мне
    может мы на над нам нас не него нее нет ни них но ну о об однако он она они оно от очень по под при про
    с со так также такой там те тем то того тоже той только том ты у уже хотя чего чей чем что чтобы чье
    чья эта эти это этого этой этом этот эту я можно нужно надо нельзя сколько будет будут""".split()
)

STOPWORDS = STOPWORDS_EN | STOPWORDS_RU


def normalize(text: str) -> str:
    """NFKC, lower case, ё → е (Russian writes both), soft hyphens dropped."""
    text = unicodedata.normalize("NFKC", text).lower()
    return text.replace("ё", "е").replace("\N{SOFT HYPHEN}", "")


@lru_cache(maxsize=262_144)
def stem(token: str) -> str:
    """Snowball stem by script: Cyrillic → Russian, Latin → English, anything else unchanged."""
    if _CYRILLIC.search(token):
        return str(_STEMMERS["ru"].stemWord(token))
    if _LATIN.search(token):
        return str(_STEMMERS["en"].stemWord(token))
    return token


def analyze(text: str) -> list[str]:
    """Index and query terms: normalised, stop words removed, stemmed."""
    terms = []
    for match in _TOKEN.finditer(text):
        token = normalize(match.group())
        if token and token not in STOPWORDS:
            terms.append(stem(token))
    return terms


def term_spans(text: str) -> list[tuple[str, int, int]]:
    """``(term, start, end)`` for every indexable token, with offsets into the original text."""
    spans = []
    for match in _TOKEN.finditer(text):
        token = normalize(match.group())
        if token and token not in STOPWORDS:
            spans.append((stem(token), match.start(), match.end()))
    return spans


def highlight_spans(text: str, query_terms: set[str]) -> list[tuple[int, int]]:
    """Character ranges of ``text`` whose stems occur in the query."""
    return [(start, end) for term, start, end in term_spans(text) if term in query_terms]


def detect_language(text: str) -> str:
    """``ru``, ``en``, ``mixed`` or ``und`` from the share of Cyrillic and Latin letters."""
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    total = cyrillic + latin
    if total < 20:
        return "und"
    if cyrillic / total >= 0.6:
        return "ru"
    if latin / total >= 0.6:
        return "en"
    return "mixed"


# --- sentences -------------------------------------------------------------------------------------

_ABBREVIATIONS = frozenset(
    """e.g i.e etc vs fig figs no nos sec secs art arts p pp vol vols ch chap dr mr mrs ms st inc ltd co
    corp approx cf al ref refs eq eqs u.s u.k jan feb mar apr jun jul aug sep sept oct nov dec
    ст стр п пп ч гл т.е т.д т.п т.к т.ч др г гг им см рис табл руб коп тыс млн млрд ред изд прим""".split()
)
_BOUNDARY = re.compile(r"([.!?…]+)([\"'»”’)\]]*)(\s+)")
_WORD_BEFORE = re.compile(r"(\S+)$")
# a list marker standing alone before its text: "1.", "3.1.", "a.", "iv."
_MARKER = re.compile(r"\d+(?:\.\d+)*|[a-zа-я]|x{0,3}(?:ix|iv|v?i{0,3})", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations ("e.g.", "ст.") and numbering ("3.1.") intact."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        following = text[end : end + 1]
        if following and following.islower():
            continue  # "approx. ten", "т.е. если": a lower-case word continues the sentence
        if match.group(1) == ".":
            so_far = text[start : match.start()].strip()
            before = _WORD_BEFORE.search(so_far)
            word = before.group(1).lower().lstrip("([\"'«“") if before else ""
            if _MARKER.fullmatch(so_far) or word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue  # a list number, an abbreviation or an initial ("J. Smith")
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
