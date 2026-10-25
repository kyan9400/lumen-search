// The search algorithms of src/lumen in JavaScript, for the static demo that runs in the browser.
// tests/test_js_parity.py checks that analyze(), BM25 and RRF give the same results as the Python code.

import EnglishStemmer from "./vendor/snowball/english-stemmer.js";
import RussianStemmer from "./vendor/snowball/russian-stemmer.js";

const TOKEN = /[\p{L}\p{N}]+/gu; // Python: [^\W_]+
const CYRILLIC = /[Ѐ-ӿ]/;
const LATIN = /[a-z]/i;
const english = new EnglishStemmer();
const russian = new RussianStemmer();
const stemCache = new Map();

export function normalize(token) {
  return token.normalize("NFKC").toLowerCase().replaceAll("ё", "е").replaceAll("­", "");
}

export function stem(token) {
  let out = stemCache.get(token);
  if (out === undefined) {
    out = CYRILLIC.test(token) ? russian.stemWord(token) : LATIN.test(token) ? english.stemWord(token) : token;
    stemCache.set(token, out);
  }
  return out;
}

/** [term, start, end] for every indexable token; offsets are UTF-16 indexes into text. */
export function termSpans(text, stopwords) {
  const spans = [];
  for (const match of text.matchAll(TOKEN)) {
    const token = normalize(match[0]);
    if (token && !stopwords.has(token)) spans.push([stem(token), match.index, match.index + match[0].length]);
  }
  return spans;
}

export function analyze(text, stopwords) {
  return termSpans(text, stopwords).map((span) => span[0]);
}

/** Okapi BM25 with Lucene's idf; the same formula and parameters as src/lumen/bm25.py. */
export class BM25 {
  constructor(documents, { k1 = 1.2, b = 0.75 } = {}) {
    this.size = documents.length;
    const lengths = new Float32Array(this.size);
    const postings = new Map(); // term -> [doc indexes, term frequencies]
    documents.forEach((terms, doc) => {
      lengths[doc] = terms.length;
      const counts = new Map();
      for (const term of terms) counts.set(term, (counts.get(term) ?? 0) + 1);
      for (const [term, count] of counts) {
        let list = postings.get(term);
        if (!list) postings.set(term, (list = [[], []]));
        list[0].push(doc);
        list[1].push(count);
      }
    });
    const average = this.size ? lengths.reduce((a, x) => a + x, 0) / this.size : 0;
    this.index = new Map();
    for (const [term, [docs, tfs]] of postings) {
      const idf = Math.log1p((this.size - docs.length + 0.5) / (docs.length + 0.5));
      const weights = new Float32Array(docs.length);
      docs.forEach((doc, i) => {
        const tf = tfs[i];
        const norm = k1 * (1 - b + (b * lengths[doc]) / (average || 1));
        weights[i] = (idf * tf * (k1 + 1)) / (tf + norm);
      });
      this.index.set(term, [Int32Array.from(docs), weights]);
    }
  }

  /** Top k [doc, score] pairs with a positive score; allowed is an optional Uint8Array mask. */
  search(terms, k, allowed = null) {
    const counts = new Map();
    for (const term of terms) if (this.index.has(term)) counts.set(term, (counts.get(term) ?? 0) + 1);
    const scores = new Float64Array(this.size);
    for (const [term, count] of counts) {
      const [docs, weights] = this.index.get(term);
      for (let i = 0; i < docs.length; i++) scores[docs[i]] += weights[i] * count;
    }
    const hits = [];
    for (let doc = 0; doc < this.size; doc++) {
      if (scores[doc] > 0 && (!allowed || allowed[doc])) hits.push([doc, scores[doc]]);
    }
    return topK(hits, k);
  }
}

/** Cosine similarity against int8 passage vectors (row i ≈ vectors[i] * scales[i]). */
export function denseSearch(query, vectors, scales, dim, k, allowed = null) {
  const count = scales.length;
  const hits = [];
  for (let row = 0; row < count; row++) {
    if (allowed && !allowed[row]) continue;
    let dot = 0;
    const offset = row * dim;
    for (let j = 0; j < dim; j++) dot += query[j] * vectors[offset + j];
    hits.push([row, dot * scales[row]]);
  }
  return topK(hits, k);
}

/** Reciprocal rank fusion: score(d) = Σ 1 / (k + rank), ties broken by best single rank, then id. */
export function rrf(rankings, k = 60) {
  const scores = new Map();
  const best = new Map();
  for (const ranking of rankings) {
    ranking.forEach((item, index) => {
      const rank = index + 1;
      scores.set(item, (scores.get(item) ?? 0) + 1 / (k + rank));
      best.set(item, Math.min(best.get(item) ?? rank, rank));
    });
  }
  return [...scores].sort((a, b) => b[1] - a[1] || best.get(a[0]) - best.get(b[0]) || a[0] - b[0]);
}

function topK(hits, k) {
  hits.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return hits.slice(0, k);
}

// --- sentences, for the demo's extractive answer ----------------------------------------------------

const ABBREVIATIONS = new Set(
  `e.g i.e etc vs fig figs no nos sec secs art arts p pp vol vols ch chap dr mr mrs ms st inc ltd co
  corp approx cf al ref refs eq eqs u.s u.k jan feb mar apr jun jul aug sep sept oct nov dec
  ст стр п пп ч гл т.е т.д т.п т.к т.ч др г гг им см рис табл руб коп тыс млн млрд ред изд прим`.split(/\s+/),
);
const MARKER = /^(?:\d+(?:\.\d+)*|[a-zа-я]|x{0,3}(?:ix|iv|v?i{0,3}))$/i;

export function splitSentences(text) {
  text = text.replace(/\s+/g, " ").trim();
  const out = [];
  let start = 0;
  for (const match of text.matchAll(/([.!?…]+)(["'»”’)\]]*)(\s+)/g)) {
    const end = match.index + match[0].length;
    const following = text[end] ?? "";
    if (following && following !== following.toUpperCase() && following === following.toLowerCase()) continue;
    if (match[1] === ".") {
      const soFar = text.slice(start, match.index).trim();
      const word = (soFar.match(/(\S+)$/)?.[1] ?? "").toLowerCase().replace(/^[(["'«“]+/, "");
      if (MARKER.test(soFar) || ABBREVIATIONS.has(word) || (word.length === 1 && /\p{L}/u.test(word))) continue;
    }
    const sentence = text.slice(start, end).trim();
    if (sentence) out.push(sentence);
    start = end;
  }
  const tail = text.slice(start).trim();
  if (tail) out.push(tail);
  return out;
}
