// Lumen web UI. Served by the FastAPI app it talks to /api; on GitHub Pages it runs the same search in the
// browser over an exported index (see scripts/build_demo.py).

import { analyze, BM25, denseSearch, rrf, splitSentences, termSpans } from "./engine.js";

const $ = (selector, root = document) => root.querySelector(selector);
const TYPE_LABEL = { pdf: "PDF", docx: "Word", xlsx: "Excel", csv: "CSV", markdown: "Markdown", text: "Text", html: "HTML", image: "Image" };
const EXAMPLES = [
  "Is it OK to block passwords that appeared in data breaches?",
  "Can I use security questions to reset a password?",
  "IA-5(1) password-based authentication",
  "Argon2id parameters",
  "maximum lifetime of a password in 1985",
  "What are the six functions of the NIST CSF?",
  "Сколько можно хранить персональные данные?",
  "согласие на распространение персональных данных",
  "трансграничная передача персональных данных",
];

const state = { mode: "hybrid", lang: "", doc: "", query: "", rerank: true, run: 0, semantic: "off", progress: 0 };
let currentHits = [];

// --- helpers -----------------------------------------------------------------------------------------

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

/** Code-point offsets (from Python) → UTF-16 offsets (JavaScript strings). */
function toUtf16(text, spans) {
  if (!/[\uD800-\uDFFF]/.test(text)) return spans;
  const points = Array.from(text);
  const offsets = [0];
  for (const point of points) offsets.push(offsets.at(-1) + point.length);
  return spans.map(([a, b]) => [offsets[a], offsets[b]]);
}

/** Mostly capital letters: code listings, table headers, running titles; not an answer sentence. */
function shouting(sentence) {
  const letters = sentence.match(/\p{L}/gu) ?? [];
  const upper = letters.filter((c) => c !== c.toLowerCase()).length;
  return letters.length > 20 && upper / letters.length > 0.6;
}

function highlighted(text, spans, from = 0, to = text.length) {
  let html = "";
  let at = from;
  for (const [a, b] of spans) {
    if (b <= from || a >= to) continue;
    html += escapeHtml(text.slice(at, Math.max(a, from))) + `<mark>${escapeHtml(text.slice(Math.max(a, from), Math.min(b, to)))}</mark>`;
    at = Math.min(b, to);
  }
  return html + escapeHtml(text.slice(at, to));
}

/** A window of about 380 characters around the first highlight, cut at word boundaries. */
function snippetRange(text, spans) {
  if (text.length <= 420) return [0, text.length];
  const first = spans.length ? spans[0][0] : 0;
  let start = Math.max(0, first - 120);
  let end = Math.min(text.length, start + 380);
  if (start > 0) start = text.indexOf(" ", start) + 1 || start;
  if (end < text.length) end = text.lastIndexOf(" ", end) || end;
  return [start, end];
}

function locationLabel({ page_start, page_end, sheet, row_start, row_end, section }) {
  const parts = [];
  if (page_start != null) parts.push(page_end != null && page_end !== page_start ? `pp. ${page_start}–${page_end}` : `p. ${page_start}`);
  if (sheet != null) {
    const rows = row_start == null ? "" : row_end == null || row_end === row_start ? `, row ${row_start}` : `, rows ${row_start}–${row_end}`;
    parts.push(`Sheet '${sheet}'${rows}`);
  }
  if (section) parts.push(section);
  return parts.join(" · ");
}

function typeBadge(doc) {
  if (doc.ocr_pages) return `<span class="badge scan" title="Scanned: ${doc.ocr_pages} page(s) read with OCR">Scan · OCR</span>`;
  return `<span class="badge ${doc.media_type}">${TYPE_LABEL[doc.media_type] ?? doc.media_type}</span>`;
}

// --- backends ----------------------------------------------------------------------------------------

class ApiBackend {
  kind = "api";

  async init() {
    this.health = await getJSON("api/health");
    this.documents = await getJSON("api/documents");
  }

  describe() {
    const h = this.health;
    return `${h.chunks.toLocaleString("en")} passages from ${h.documents} documents · BM25 + ${h.embedding_model} + FAISS` +
      (h.reranker ? ` · reranked by ${h.reranker}` : "") + ` · cache: ${h.cache}`;
  }

  async search({ q, k, mode, doc, lang, rerank }) {
    const params = new URLSearchParams({ q, k, mode });
    if (doc) params.append("doc", doc);
    if (lang) params.set("lang", lang);
    if (!rerank) params.set("rerank", "false");
    const result = await getJSON(`api/search?${params}`);
    for (const hit of result.hits) hit.highlights = toUtf16(hit.text, hit.highlights);
    return result;
  }

  async answer({ q, mode, doc, lang }) {
    return getJSON("api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, k: 5, mode, doc: doc ? [doc] : null, lang: lang || null }),
    });
  }

  async context(hit) {
    return getJSON(`api/chunks/${hit.chunk_id}`);
  }
}

class StaticBackend {
  kind = "static";

  async init() {
    const [corpus, vectors, scales] = await Promise.all([
      getJSON("data/corpus.json"),
      fetch("data/vectors.bin").then((r) => r.arrayBuffer()),
      fetch("data/scales.bin").then((r) => r.arrayBuffer()),
    ]);
    this.corpus = corpus;
    this.documents = corpus.documents;
    this.vectors = new Int8Array(vectors);
    this.scales = new Float32Array(scales);
    this.stopwords = new Set(corpus.stopwords);
    // the same text the server indexes: section path, then the passage (Chunk.indexed_text)
    this.bm25 = new BM25(corpus.chunks.map((c) => analyze(c.s ? `${c.s}\n${c.t}` : c.t, this.stopwords)), corpus.bm25);
  }

  describe() {
    return `${this.corpus.chunks.length.toLocaleString("en")} passages from ${this.documents.length} documents · ` +
      `BM25 + ${this.corpus.model.name} (int8, running in this page) · reciprocal rank fusion`;
  }

  mask(doc, lang) {
    if (!doc && !lang) return null;
    const docIndex = doc ? this.documents.findIndex((d) => d.id === doc) : -1;
    return Uint8Array.from(this.corpus.chunks, (c) => (!doc || c.d === docIndex) && (!lang || c.l === lang));
  }

  search({ q, k, mode, doc, lang }, vector) {
    const started = performance.now();
    const allowed = this.mask(doc, lang);
    const depth = Math.max(this.corpus.candidates, k);
    const terms = analyze(q, this.stopwords);
    const bm25 = mode === "dense" ? [] : this.bm25.search(terms, depth, allowed);
    const dense = mode === "bm25" || !vector ? [] : denseSearch(vector, this.vectors, this.scales, this.corpus.model.dim, depth, allowed);
    const bm25Rank = new Map(bm25.map(([id, score], i) => [id, [i + 1, score]]));
    const denseRank = new Map(dense.map(([id, score], i) => [id, [i + 1, score]]));
    let ranked;
    if (mode === "hybrid" && vector) ranked = rrf([bm25.map((h) => h[0]), dense.map((h) => h[0])], this.corpus.rrf_k);
    else ranked = mode === "dense" ? dense : bm25;
    const querySet = new Set(terms);
    const hits = ranked.slice(0, k).map(([id, score], i) => {
      const c = this.corpus.chunks[id];
      const d = this.documents[c.d];
      const location = { page_start: c.p?.[0] ?? null, page_end: c.p?.[1] ?? null, section: c.s ?? null, sheet: c.sh ?? null, row_start: c.r?.[0] ?? null, row_end: c.r?.[1] ?? null };
      location.label = locationLabel(location);
      return {
        rank: i + 1,
        chunk_id: id,
        doc_id: d.id,
        title: d.title,
        media_type: d.media_type,
        text: c.t,
        highlights: termSpans(c.t, this.stopwords).filter((s) => querySet.has(s[0])).map((s) => [s[1], s[2]]),
        location,
        language: c.l,
        ocr_confidence: c.o ?? null,
        scores: {
          score,
          fused: mode === "hybrid" && vector ? score : null,
          bm25: bm25Rank.get(id)?.[1] ?? null,
          bm25_rank: bm25Rank.get(id)?.[0] ?? null,
          dense: denseRank.get(id)?.[1] ?? null,
          dense_rank: denseRank.get(id)?.[0] ?? null,
          rerank: null,
        },
        file_url: d.file + (d.media_type === "pdf" && location.page_start ? `#page=${location.page_start}` : ""),
      };
    });
    return { query: q, mode, took_ms: performance.now() - started, cached: false, reranked: false, hits, partial: mode === "hybrid" && !vector };
  }

  /** Extractive answer: the sentences of the top passages that share the most query terms. */
  answer({ q }, hits) {
    const terms = new Set(analyze(q, this.stopwords));
    const candidates = [];
    hits.slice(0, 3).forEach((hit) => {
      hit.text.split("\n").forEach((line) => {
        for (const sentence of splitSentences(line)) {
          // a fragment (lower-case start) or a code listing (capitals) makes a poor answer
          if (sentence.split(" ").length < 4 || /^\p{Ll}/u.test(sentence) || shouting(sentence)) continue;
          const own = new Set(analyze(sentence, this.stopwords));
          const overlap = terms.size ? [...terms].filter((t) => own.has(t)).length / terms.size : 0;
          candidates.push({ sentence, rank: hit.rank, score: overlap - 0.05 * (hit.rank - 1) });
        }
      });
    });
    candidates.sort((a, b) => b.score - a.score);
    const best = candidates[0]?.score ?? 0;
    // a second sentence only when it is nearly as good as the first
    const picked = candidates.filter((c, i) => c.score > 0 && (i === 0 || c.score >= 0.75 * best)).slice(0, 2);
    if (!picked.length) return { answer: "", method: "none", citations: [] };
    picked.sort((a, b) => a.rank - b.rank);
    return {
      answer: picked.map((c) => `${c.sentence} [${c.rank}]`).join(" "),
      method: "extractive",
      citations: [...new Set(picked.map((c) => c.rank))],
    };
  }

  context(hit) {
    const chunks = this.corpus.chunks;
    const current = chunks[hit.chunk_id];
    const sibling = (offset) => {
      const other = chunks[hit.chunk_id + offset];
      return other && other.d === current.d ? other.t : null;
    };
    return { text: current.t, previous: sibling(-1), next: sibling(1) };
  }
}

async function detectBackend() {
  // the GitHub Pages build marks itself static (scripts/build_demo.py), which saves a failing request
  if (document.querySelector('meta[name="lumen-backend"]')?.content === "static") return new StaticBackend();
  try {
    const response = await fetch("api/health", { headers: { Accept: "application/json" } });
    if (response.ok && response.headers.get("content-type")?.includes("json")) return new ApiBackend();
  } catch {
    /* no API here: the static demo */
  }
  return new StaticBackend();
}

// --- semantic model (static demo only) ---------------------------------------------------------------

let semantic = null;
let backend = null;

function setSemantic(status, progress = state.progress) {
  state.semantic = status;
  state.progress = progress;
  const pill = $("#model-status");
  if (!pill || backend?.kind !== "static") return;
  pill.hidden = false;
  pill.dataset.state = status;
  const size = backend.corpus.model.download_mb;
  pill.textContent = {
    off: `Semantic model not loaded (${size} MB)`,
    loading: `Loading semantic model… ${Math.round(progress * 100)}%`,
    ready: "Semantic model ready (in your browser)",
    failed: "Semantic model unavailable — keyword search only",
  }[status];
  pill.disabled = status !== "off" && status !== "failed";
}

async function startSemantic() {
  if (backend.kind !== "static" || state.semantic === "loading" || state.semantic === "ready") return;
  setSemantic("loading", 0);
  try {
    semantic ??= await import("./semantic.js");
    await semantic.loadModel(backend.corpus.model, (fraction) => setSemantic("loading", fraction));
    setSemantic("ready", 1);
    if (state.query && state.mode !== "bm25") search();
  } catch (error) {
    console.error(error);
    setSemantic("failed");
  }
}

// --- search and rendering ----------------------------------------------------------------------------

function readUrl() {
  const params = new URLSearchParams(location.search);
  state.query = params.get("q") ?? "";
  state.mode = ["hybrid", "bm25", "dense"].includes(params.get("mode")) ? params.get("mode") : "hybrid";
  state.lang = ["en", "ru"].includes(params.get("lang")) ? params.get("lang") : "";
  state.doc = params.get("doc") ?? "";
}

function writeUrl() {
  const params = new URLSearchParams();
  if (state.query) params.set("q", state.query);
  if (state.mode !== "hybrid") params.set("mode", state.mode);
  if (state.lang) params.set("lang", state.lang);
  if (state.doc) params.set("doc", state.doc);
  history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
}

function syncControls() {
  $("#q").value = state.query;
  for (const button of document.querySelectorAll("[data-mode]")) button.setAttribute("aria-checked", String(button.dataset.mode === state.mode));
  for (const button of document.querySelectorAll("[data-lang]")) button.setAttribute("aria-checked", String(button.dataset.lang === state.lang));
  $("#doc").value = state.doc;
}

async function search() {
  const run = ++state.run;
  const q = state.query.trim();
  writeUrl();
  if (!q) {
    renderEmpty();
    return;
  }
  const status = $("#status");
  status.textContent = "Searching…";
  const params = { q, k: 10, mode: state.mode, doc: state.doc, lang: state.lang, rerank: state.rerank };
  try {
    let result;
    let answer;
    if (backend.kind === "api") {
      [result, answer] = await Promise.all([backend.search(params), backend.answer(params)]);
    } else {
      let vector = null;
      if (state.mode !== "bm25") {
        if (state.semantic === "ready") vector = await semanticVector(q);
        else if (state.mode === "dense") {
          status.textContent = "Loading the semantic model…";
          startSemantic();
          vector = await semanticVector(q); // waits for the model
        } else startSemantic(); // hybrid: keyword results now; startSemantic searches again when ready
      }
      if (run !== state.run) return;
      result = backend.search(params, vector);
      answer = backend.answer(params, result.hits);
    }
    if (run !== state.run) return;
    renderResults(result, answer);
  } catch (error) {
    if (run !== state.run) return;
    console.error(error);
    status.textContent = `Search failed: ${error.message}`;
  }
}

async function semanticVector(q) {
  semantic ??= await import("./semantic.js");
  return semantic.embedQuery(backend.corpus.model, q);
}

function renderEmpty() {
  $("#status").textContent = "";
  $("#results").innerHTML = `<p class="hint">Search ${backend.documents.length} public documents — NIST and OWASP guidance in English,
    the Russian personal data law in Russian, a 1985 scan read with OCR — and see where every answer comes from.</p>`;
  $("#answer").hidden = true;
}

function scoreChips(scores) {
  const chips = [];
  if (scores.bm25_rank) chips.push(`<span class="chip" title="BM25 score ${scores.bm25.toFixed(2)}">Keyword #${scores.bm25_rank}</span>`);
  if (scores.dense_rank) chips.push(`<span class="chip" title="Cosine similarity">Semantic #${scores.dense_rank} · ${scores.dense.toFixed(3)}</span>`);
  if (scores.fused != null) chips.push(`<span class="chip" title="Reciprocal rank fusion score">RRF ${scores.fused.toFixed(4)}</span>`);
  if (scores.rerank != null) chips.push(`<span class="chip" title="Cross-encoder score">Rerank ${scores.rerank.toFixed(2)}</span>`);
  return chips.join("");
}

function renderResults(result, answer) {
  const docs = new Map(backend.documents.map((d) => [d.id, d]));
  const count = result.hits.length;
  const parts = [`${count} result${count === 1 ? "" : "s"}`, `${Math.max(1, Math.round(result.took_ms))} ms`, { hybrid: "hybrid", bm25: "keyword", dense: "semantic" }[result.mode]];
  if (result.cached) parts.push("cached");
  if (result.reranked) parts.push("reranked");
  if (result.partial) parts.push("keyword only until the semantic model loads");
  $("#status").textContent = parts.join(" · ");

  const list = $("#results");
  if (!count) {
    list.innerHTML = `<p class="hint">Nothing found. Try other words, another mode, or remove the filters.</p>`;
    $("#answer").hidden = true;
    return;
  }
  list.innerHTML = result.hits
    .map((hit) => {
      const doc = docs.get(hit.doc_id) ?? { media_type: hit.media_type, ocr_pages: 0 };
      const [from, to] = snippetRange(hit.text, hit.highlights);
      const clipped = from > 0 || to < hit.text.length;
      const ocr = hit.ocr_confidence != null ? `<span class="chip ocr" title="Mean Tesseract word confidence">OCR ${Math.round(hit.ocr_confidence)}%</span>` : "";
      const open = hit.file_url
        ? `<a class="link" href="${escapeHtml(hit.file_url)}" target="_blank" rel="noopener">${hit.media_type === "pdf" && hit.location.page_start ? `Open PDF at p. ${hit.location.page_start}` : "Open file"} ↗</a>`
        : "";
      return `<article class="hit" id="hit-${hit.rank}">
        <header class="hit-head"><span class="rank">${hit.rank}</span>${typeBadge(doc)}<h3>${escapeHtml(hit.title)}</h3></header>
        ${hit.location.label ? `<p class="where">${escapeHtml(hit.location.label)}</p>` : ""}
        <p class="text" data-full="false">${clipped && from > 0 ? "… " : ""}${highlighted(hit.text, hit.highlights, from, to)}${clipped && to < hit.text.length ? " …" : ""}</p>
        <footer class="hit-foot">${scoreChips(hit.scores)}${ocr}
          <span class="actions">${clipped ? `<button type="button" class="link" data-action="expand" data-rank="${hit.rank}">Full passage</button>` : ""}
          <button type="button" class="link" data-action="context" data-rank="${hit.rank}">Context</button>${open}</span>
        </footer></article>`;
    })
    .join("");
  currentHits = result.hits;
  renderAnswer(answer, result.hits);
}

function renderAnswer(answer, hits) {
  const box = $("#answer");
  if (!answer || !answer.answer || answer.method === "none") {
    box.hidden = true;
    return;
  }
  const label = { extractive: "Extractive answer", llm: "Generated answer" }[answer.method] ?? "Answer";
  const text = escapeHtml(answer.answer).replace(/\[(\d{1,2})\]/g, (marker, n) => {
    const hit = hits.find((h) => h.rank === Number(n));
    return hit ? `<a class="cite" href="#hit-${n}" title="${escapeHtml(hit.title)}">[${n}]</a>` : marker;
  });
  box.innerHTML = `<h2>${label}</h2><p>${text}</p>${answer.note ? `<p class="note">${escapeHtml(answer.note)}</p>` : ""}
    <p class="note">${answer.method === "llm" ? "Written by the configured language model from the cited passages." : "Sentences quoted from the top passages; numbers point to the results."}</p>`;
  box.hidden = false;
}

async function showContext(rank) {
  const hit = currentHits.find((h) => h.rank === rank);
  const dialog = $("#viewer");
  $("#viewer-title").textContent = hit.title;
  $("#viewer-where").textContent = hit.location.label;
  $("#viewer-body").innerHTML = "<p>Loading…</p>";
  dialog.showModal();
  const context = await backend.context(hit);
  const block = (text, cls) => (text ? `<p class="${cls}">${cls === "current" ? highlighted(text, hit.highlights) : escapeHtml(text)}</p>` : "");
  $("#viewer-body").innerHTML = block(context.previous, "around") + block(hit.text, "current") + block(context.next, "around");
  $("#viewer-open").innerHTML = hit.file_url ? `<a class="link" href="${escapeHtml(hit.file_url)}" target="_blank" rel="noopener">Open the source file ↗</a>` : "";
}

function renderCorpus() {
  const list = $("#corpus-list");
  list.innerHTML = backend.documents
    .map((d) => {
      const facts = [d.language === "ru" ? "Russian" : d.language === "en" ? "English" : d.language];
      if (d.pages) facts.push(`${d.pages} pages`);
      if (d.ocr_pages) facts.push(`${d.ocr_pages} read with OCR`);
      facts.push(`${d.chunks} passages`);
      const source = d.source_url ? ` · <a href="${escapeHtml(d.source_url)}" target="_blank" rel="noopener">source</a>` : "";
      return `<li>${typeBadge(d)} <span class="doc-title">${escapeHtml(d.title)}</span>
        <span class="doc-facts">${facts.join(" · ")}${source}</span></li>`;
    })
    .join("");
  const select = $("#doc");
  select.innerHTML = `<option value="">All documents</option>` + backend.documents.map((d) => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.title)}</option>`).join("");
  $("#engine-line").textContent = backend.describe();
}

function bind() {
  $("#search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.query = $("#q").value;
    search();
  });
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      syncControls();
      if (state.query) search();
    });
  }
  for (const button of document.querySelectorAll("[data-lang]")) {
    button.addEventListener("click", () => {
      state.lang = button.dataset.lang;
      syncControls();
      if (state.query) search();
    });
  }
  $("#doc").addEventListener("change", (event) => {
    state.doc = event.target.value;
    if (state.query) search();
  });
  $("#rerank").addEventListener("change", (event) => {
    state.rerank = event.target.checked;
    if (state.query) search();
  });
  $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="example">${escapeHtml(q)}</button>`).join("");
  $("#examples").addEventListener("click", (event) => {
    const button = event.target.closest(".example");
    if (!button) return;
    state.query = button.textContent;
    syncControls();
    search();
  });
  $("#results").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const rank = Number(button.dataset.rank);
    if (button.dataset.action === "context") showContext(rank);
    if (button.dataset.action === "expand") {
      const hit = currentHits.find((h) => h.rank === rank);
      button.closest(".hit").querySelector(".text").innerHTML = highlighted(hit.text, hit.highlights);
      button.remove();
    }
  });
  $("#viewer-close").addEventListener("click", () => $("#viewer").close());
  $("#model-status").addEventListener("click", startSemantic);
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("#q")) {
      event.preventDefault();
      $("#q").focus();
    }
  });
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    root.dataset.theme = current === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("lumen-theme", root.dataset.theme);
    } catch {
      /* private mode */
    }
  });
}

async function main() {
  try {
    const saved = localStorage.getItem("lumen-theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch {
    /* storage blocked */
  }
  bind();
  readUrl();
  syncControls();
  $("#status").textContent = "Loading the index…";
  backend = await detectBackend();
  document.body.dataset.backend = backend.kind;
  try {
    await backend.init();
  } catch (error) {
    $("#status").textContent = `Could not load the index: ${error.message}`;
    return;
  }
  renderCorpus();
  syncControls();
  $("#api-link").hidden = backend.kind !== "api";
  $("#rerank-toggle").hidden = !(backend.kind === "api" && backend.health.reranker);
  $("#demo-note").hidden = backend.kind !== "static";
  if (backend.kind === "static") {
    setSemantic("off");
    // fetch the model in the background unless the visitor asked browsers to save data
    if (!navigator.connection?.saveData) startSemantic();
  }
  if (state.query) search();
  else renderEmpty();
}

main();
