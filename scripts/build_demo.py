"""Assemble the static GitHub Pages demo.

    npm ci                                                   # transformers.js and ONNX Runtime Web
    python scripts/build_browser_model.py build/browser-model
    lumen --data-dir data ingest --manifest corpus/manifest.json
    python scripts/build_demo.py --data-dir data --model-dir build/browser-model --out build/demo

The site is the regular web UI (src/lumen/web) plus the exported index, the browser model and the two
libraries it needs, all served from the same origin: no CDN is involved, which matters in regions where
some CDNs are blocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lumen.config import Settings
from lumen.engine import SearchEngine
from lumen.export import export_demo
from lumen.onnx_embedder import OnnxEmbedder

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "lumen" / "web"
MODEL_ID = "multilingual-e5-small-en-ru"
VENDOR = {
    "vendor/transformers/transformers.web.min.js": "@huggingface/transformers/dist/transformers.web.min.js",
    # the WASM-only build of ONNX Runtime: 11 MB instead of 22 MB for the WebGPU-capable one
    "vendor/ort/ort.wasm.min.mjs": "onnxruntime-web/dist/ort.wasm.min.mjs",
    "vendor/ort/ort-wasm-simd-threaded.mjs": "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
    "vendor/ort/ort-wasm-simd-threaded.wasm": "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
    "vendor/LICENSE-transformers.txt": "@huggingface/transformers/LICENSE",
}
# the onnxruntime-web package declares MIT in package.json but ships no licence file
ORT_LICENSE = """ONNX Runtime Web - MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# transformers.web.min.js imports these bare specifiers; both resolve to the one ONNX Runtime build
IMPORT_MAP = {
    "imports": {
        "onnxruntime-web": "./vendor/ort/ort.wasm.min.mjs",
        "onnxruntime-common": "./vendor/ort/ort-common.mjs",
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static demo site.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "build" / "browser-model")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "demo")
    parser.add_argument("--node-modules", type=Path, default=ROOT / "node_modules")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(WEB, args.out)

    for target, source in VENDOR.items():
        path = args.out / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.node_modules / source, path)
    (args.out / "vendor/ort/ort-common.mjs").write_text(
        'export * from "./ort.wasm.min.mjs";\n', encoding="utf-8"
    )
    (args.out / "vendor/LICENSE-onnxruntime.txt").write_text(ORT_LICENSE, encoding="utf-8")
    shutil.copytree(args.model_dir, args.out / "models" / MODEL_ID)

    download = sum(
        p.stat().st_size
        for p in [*(args.out / "models" / MODEL_ID).rglob("*"), *(args.out / "vendor").rglob("*")]
        if p.is_file()
    )
    settings = Settings(data_dir=args.data_dir)
    engine = SearchEngine(settings, embedder=OnnxEmbedder(args.model_dir, name=settings.embedding_model))
    try:
        summary = export_demo(engine, args.out, model={"id": MODEL_ID, "download_mb": round(download / 1e6)})
    finally:
        engine.close()

    index = args.out / "index.html"
    html = index.read_text(encoding="utf-8")
    head = (
        '<meta name="lumen-backend" content="static">\n  '
        f'<script type="importmap">{json.dumps(IMPORT_MAP)}</script>\n  '
    )
    html = html.replace('<script type="module"', head + '<script type="module"', 1)
    index.write_text(html, encoding="utf-8")
    (args.out / ".nojekyll").write_text("", encoding="utf-8")
    print(json.dumps({**summary, "download_mb": round(download / 1e6), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
