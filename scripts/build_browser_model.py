"""Build the browser copy of multilingual-e5-small used by the static demo.

    python scripts/build_browser_model.py demo/models/multilingual-e5-small-en-ru

The int8 ONNX export of the model (Xenova/multilingual-e5-small, 118 MB) spends 96 MB on the word
embeddings of a 250,000-piece vocabulary for 100 languages. The demo searches English and Russian, so
this script keeps only the vocabulary pieces made of ASCII, Cyrillic and common typographic characters.

The result is exact, not approximate: SentencePiece segments a text only into pieces that occur in it,
so a text written in those characters can only ever use pieces that were kept. The script checks this on
sample texts by comparing the hidden states of both models, which must be bit-identical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import numpy_helper

SOURCE = "Xenova/multilingual-e5-small"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
EMBEDDINGS = "embeddings.word_embeddings.weight_quantized"
EXTRA = set("–—‘’‚“”„«»…•·№°±×§©®™€₽‰′″")
SAMPLES = [
    "query: What is the minimum password length?",
    "passage: Verifiers and CSPs SHALL require passwords to be a minimum of 15 characters in length.",
    "query: Какая минимальная длина пароля?",
    "passage: Обработка персональных данных допускается в следующих случаях: «согласие» — № 152-ФЗ…",
    "passage: IA-5(1) Authenticator Management | Password-based Authentication [Assignment: org-defined]",
]


WORD_START = chr(0x2581)  # "▁": SentencePiece's marker for the start of a word


def allowed(piece: str) -> bool:
    """True for pieces made only of ASCII, Cyrillic (U+0400-U+04FF) and common typographic characters."""
    return all(
        0x20 <= ord(c) <= 0x7E or 0x400 <= ord(c) <= 0x4FF or c in EXTRA
        for c in piece.replace(WORD_START, "")
    )


def build(target: Path) -> None:
    files = {name: Path(hf_hub_download(SOURCE, name)) for name in FILES}
    files["model"] = Path(hf_hub_download(SOURCE, "onnx/model_quantized.onnx"))
    tokenizer = json.loads(files["tokenizer.json"].read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    special = {token["id"] for token in tokenizer["added_tokens"]}
    keep = [i for i, (piece, _) in enumerate(vocab) if i in special or allowed(piece)]
    new_id = {old: new for new, old in enumerate(keep)}
    print(f"keeping {len(keep):,} of {len(vocab):,} vocabulary pieces")

    # tokenizer: same pieces and scores, renumbered; specials 0-3 keep their ids
    tokenizer["model"]["vocab"] = [vocab[i] for i in keep]
    tokenizer["model"]["unk_id"] = new_id[tokenizer["model"]["unk_id"]]
    for token in tokenizer["added_tokens"]:
        token["id"] = new_id[token["id"]]
    for entry in tokenizer["post_processor"].get("special_tokens", {}).values():
        entry["ids"] = [new_id[i] for i in entry["ids"]]
    if tokenizer.get("padding"):
        tokenizer["padding"]["pad_id"] = new_id[tokenizer["padding"]["pad_id"]]
    assert all(new_id[i] == i for i in range(4)), "<s>, <pad>, </s> and <unk> must keep ids 0-3"

    # model: slice the rows of the int8 embedding matrix (one scale and zero point for the whole matrix)
    model = onnx.load(str(files["model"]))
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == EMBEDDINGS:
            matrix = numpy_helper.to_array(initializer)
            sliced = numpy_helper.from_array(np.ascontiguousarray(matrix[keep]), EMBEDDINGS)
            model.graph.initializer[index].CopyFrom(sliced)
            break
    else:
        raise SystemExit(f"{EMBEDDINGS} not found in the ONNX graph")

    if target.exists():
        shutil.rmtree(target)
    (target / "onnx").mkdir(parents=True)
    onnx.save(model, str(target / "onnx" / "model_quantized.onnx"))
    config = json.loads(files["config.json"].read_text(encoding="utf-8"))
    config["vocab_size"] = len(keep)
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(files[name], target / name)
    (target / "README.md").write_text(
        "# multilingual-e5-small, English and Russian vocabulary\n\n"
        f"Built by `scripts/build_browser_model.py` from [{SOURCE}](https://huggingface.co/{SOURCE}), "
        "the int8 ONNX export of "
        "[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) "
        f"(MIT License). Only the vocabulary was reduced, to {len(keep):,} pieces; weights are unchanged.\n",
        encoding="utf-8",
    )
    size = (target / "onnx" / "model_quantized.onnx").stat().st_size / 1e6
    print(f"wrote {target} ({size:.1f} MB model, was {files['model'].stat().st_size / 1e6:.1f} MB)")
    verify(files["model"], files["tokenizer.json"], target)


def verify(original_model: Path, original_tokenizer: Path, target: Path) -> None:
    """Both models must give bit-identical hidden states for English and Russian text."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    def run(model: Path, tokenizer: Path, text: str) -> np.ndarray:
        encoding = Tokenizer.from_file(str(tokenizer)).encode(text)
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros((1, len(encoding.ids)), dtype=np.int64),
        }
        return np.asarray(session.run(None, feed)[0])

    for text in SAMPLES:
        before = run(original_model, original_tokenizer, text)
        after = run(target / "onnx" / "model_quantized.onnx", target / "tokenizer.json", text)
        if not np.array_equal(before, after):
            raise SystemExit(f"outputs differ for {text!r}")
    print(f"verified: identical hidden states on {len(SAMPLES)} English and Russian samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", type=Path)
    build(parser.parse_args().target)
    sys.exit(0)
