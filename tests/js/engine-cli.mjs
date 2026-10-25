// Runs the browser engine (src/lumen/web/engine.js) on a JSON request from stdin and prints the results,
// so tests/test_js_parity.py can compare them with the Python implementation.
import { readFileSync } from "node:fs";
import { analyze, BM25, rrf, splitSentences } from "../../src/lumen/web/engine.js";

const request = JSON.parse(readFileSync(0, "utf8"));
const stopwords = new Set(request.stopwords);
const index = new BM25(request.documents.map((d) => analyze(d, stopwords)), request.bm25);
const result = {
  analyze: request.texts.map((t) => analyze(t, stopwords)),
  sentences: request.texts.map((t) => splitSentences(t)),
  bm25: request.queries.map((q) => index.search(analyze(q, stopwords), 5)),
  rrf: rrf(request.rankings, 60),
};
process.stdout.write(JSON.stringify(result));
