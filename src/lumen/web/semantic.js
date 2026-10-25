// Query embeddings in the browser for the static demo: transformers.js runs the int8 ONNX model built by
// scripts/build_browser_model.py. Everything is served from this site; nothing is fetched from a CDN.

let loading = null;

/**
 * Load the model once. onProgress receives the fraction of bytes downloaded (0-1).
 * Files are kept in the browser's Cache Storage, so a second visit starts at once.
 */
export function loadModel(model, onProgress = () => {}) {
  loading ??= (async () => {
    const { pipeline, env } = await import("./vendor/transformers/transformers.web.min.js");
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.remoteHost = new URL("./models/", location.href).href;
    env.remotePathTemplate = "{model}/";
    env.useBrowserCache = true;
    env.backends.onnx.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
    env.backends.onnx.wasm.numThreads = 1; // GitHub Pages cannot send the headers threads need
    const files = new Map();
    const report = (event) => {
      if (event.status === "progress" && event.total) {
        files.set(event.file, [event.loaded, event.total]);
        let loaded = 0;
        let total = 0;
        for (const [l, t] of files.values()) {
          loaded += l;
          total += t;
        }
        onProgress(total ? loaded / total : 0);
      }
    };
    return pipeline("feature-extraction", model.id, { dtype: "q8", device: "wasm", progress_callback: report });
  })();
  loading.catch(() => {
    loading = null; // allow a retry after a network error
  });
  return loading;
}

export async function embedQuery(model, text) {
  const extractor = await loadModel(model);
  const output = await extractor(model.query_prefix + text, { pooling: "mean", normalize: true });
  return output.data;
}
