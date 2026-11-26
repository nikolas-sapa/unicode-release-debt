from __future__ import annotations

import json
import sys
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf
import tokscan


def load_backend(carrier_id: str) -> dict:
    directory = cf.TOKENIZER_DIR / carrier_id
    tokenizer_json = directory / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"{carrier_id}: missing frozen tokenizer.json; run freeze_confirmatory_tokenizers.py")
    raw = json.loads(tokenizer_json.read_text())
    fast = Tokenizer.from_file(str(tokenizer_json))
    model_path = directory / "tokenizer.model"
    sp = None
    piece_bytes = []
    meta = {}
    strip_prepend = False
    if model_path.exists():
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        piece_bytes = tokscan.sentencepiece_piece_bytes(model_path)
        if len(piece_bytes) != sp.get_piece_size():
            raise ValueError(f"{carrier_id}: protobuf piece count != sentencepiece piece count")
        proto = model_path.read_bytes()
        normalizer_name = tokscan.embedded(proto, 3, 1, 2) or b""
        precompiled = tokscan.embedded(proto, 3, 2, 2) or b""
        meta = {
            "model_type_enum": tokscan.embedded(proto, 2, 3, 0),
            "trainer_vocab_size": tokscan.embedded(proto, 2, 4, 0),
            "normalizer_name": normalizer_name.decode(errors="replace"),
            "precompiled_charsmap_sha256": __import__("hashlib").sha256(precompiled).hexdigest() if precompiled else None,
            "add_dummy_prefix": bool(tokscan.embedded(proto, 3, 3, 0)),
        }
        strip_prepend = meta["add_dummy_prefix"]
    bytelevel = "ByteLevel" in json.dumps(raw.get("decoder"), separators=(",", ":")) or "ByteLevel" in json.dumps(raw.get("pre_tokenizer"), separators=(",", ":"))
    return {
        "tokenizer_id": carrier_id,
        "directory": directory,
        "raw": raw,
        "fast": fast,
        "bytelevel": bytelevel,
        "sp": sp,
        "piece_bytes": piece_bytes,
        "sp_meta": meta,
        "strip_prepend": strip_prepend,
        "backend": "sentencepiece-native+tokenizer.json-crosscheck" if sp else "tokenizers-tokenizer.json-raw",
    }


def effective_encode(backend: dict, text: str):
    ids, pieces, decoded = tokscan.backend_encode(backend, text)
    eff_ids, eff_pieces, eff_bytes, dummy_removed, dummy_stripped = tokscan.strip_dummy_prefix(backend, text, ids, pieces)
    fast_enc = backend["fast"].encode(text, add_special_tokens=False)
    return {
        "token_ids": eff_ids,
        "token_count": len(eff_ids),
        "raw_token_count": len(ids),
        "tokens": eff_pieces,
        "decoded": decoded,
        "token_bytes": eff_bytes,
        "dummy_prefix_tokens_removed": dummy_removed,
        "dummy_prefix_stripped": int(dummy_stripped),
        "native_backend_ids_match": None if backend["sp"] is None else int(fast_enc.ids == ids),
    }
