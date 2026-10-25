"""Embeddings from an ONNX export, computed exactly as transformers.js does in the browser demo.

The static demo embeds queries in the browser with the int8 model built by
``scripts/build_browser_model.py``. Its passage vectors must come from the same model, quantization and
pooling, or query and passage would live in slightly different spaces. Needs ``pip install .[demo]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lumen.embeddings import Vectors


class OnnxEmbedder:
    def __init__(
        self,
        model_dir: Path,
        *,
        name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def _embed(self, texts: Sequence[str]) -> Vectors:
        out = []
        for start in range(0, len(texts), self.batch_size):
            encodings = self.tokenizer.encode_batch(list(texts[start : start + self.batch_size]))
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
            # mean over real tokens, then L2: transformers.js pooling "mean" with normalize: true
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed([self.query_prefix + t for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._embed([self.passage_prefix + t for t in texts])
