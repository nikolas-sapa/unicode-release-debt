from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import regex
import sentencepiece as spm
import tokenizers as tokenizers_pkg
import unicodedata2
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT, DATA, RESULTS = ucd.ROOT, ucd.DATA, ucd.RESULTS
BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
SPACING_MARK = "▁".encode()


def byte_reverse():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_REV = byte_reverse()
canon = lambda obj: json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
sha_text = lambda text: hashlib.sha256(text.encode()).hexdigest()


def varint(buf: bytes, pos: int):
    value = shift = 0
    while True:
        byte = buf[pos]; pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, pos
        shift += 7


def wire_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = buf[pos:pos + size]; pos += size
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = buf[pos:pos + size]; pos += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def embedded(buf: bytes, section_field: int, scalar_field: int, want_wire: int):
    for field, wire, section in wire_fields(buf):
        if field == section_field and wire == 2:
            for subfield, subwire, value in wire_fields(section):
                if subfield == scalar_field and subwire == want_wire:
                    return value
    return None


def sentencepiece_piece_bytes(path: Path) -> list[bytes]:
    pieces = []
    for field, wire, record in wire_fields(path.read_bytes()):
        if field == 1 and wire == 2:
            pieces.append(next(value for f, w, value in wire_fields(record) if f == 1 and w == 2))
    return pieces


def tokenizer_meta(tid: str):
    directory = DATA / "hf" / tid
    path = directory / "tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ucd.HF[tid]["files"]["tokenizer.json"]:
        raise ValueError(f"{tid} tokenizer.json hash mismatch")
    raw = json.loads(path.read_text())
    decoder, normalizer = canon(raw.get("decoder")), canon(raw.get("normalizer"))
    bytelevel = "ByteLevel" in decoder or "ByteLevel" in canon(raw.get("pre_tokenizer"))
    return directory, raw, bytelevel, '"type":"Prepend"' in normalizer and "▁" in normalizer


def load_backend(tid: str):
    directory, raw, bytelevel, strip_prepend = tokenizer_meta(tid)
    fast = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model_path, sp, piece_bytes, meta = directory / "tokenizer.model", None, [], {}
    if model_path.exists():
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        proto = model_path.read_bytes()
        piece_bytes = sentencepiece_piece_bytes(model_path)
        if len(piece_bytes) != sp.get_piece_size():
            raise ValueError(f"{tid}: protobuf piece count != sentencepiece piece count")
        normalizer_name = embedded(proto, 3, 1, 2) or b""
        precompiled = embedded(proto, 3, 2, 2) or b""
        meta = {"model_type_enum": embedded(proto, 2, 3, 0), "trainer_vocab_size": embedded(proto, 2, 4, 0),
                "normalizer_name": normalizer_name.decode(errors="replace"), "precompiled_charsmap_sha256": hashlib.sha256(precompiled).hexdigest() if precompiled else None,
                "add_dummy_prefix": bool(embedded(proto, 3, 3, 0))}
        strip_prepend = meta["add_dummy_prefix"]
    return {"tokenizer_id": tid, "directory": directory, "raw": raw, "fast": fast, "bytelevel": bytelevel,
            "sp": sp, "piece_bytes": piece_bytes, "sp_meta": meta, "strip_prepend": strip_prepend,
            "backend": "sentencepiece-native+tokenizer.json-crosscheck" if sp else "tokenizers-tokenizer.json-raw"}


def semantic_piece_bytes(backend, token_id: int, token: str):
    sp = backend["sp"]
    if sp is not None:
        if sp.is_byte(token_id):
            match = BYTE_RE.match(token)
            if not match:
                raise ValueError(f"byte piece without <0xNN> spelling: {token!r}")
            return bytes([int(match.group(1), 16)])
        return backend["piece_bytes"][token_id].replace(SPACING_MARK, b" ")
    match = BYTE_RE.match(token)
    if match and not backend["bytelevel"]:
        return bytes([int(match.group(1), 16)])
    if backend["bytelevel"]:
        try:
            return bytes(BYTE_REV[c] for c in token)
        except KeyError:
            return None
    try:
        return token.replace("▁", " ").encode()
    except UnicodeEncodeError:
        return None


def strip_dummy_prefix(backend, text: str, ids, pieces):
    piece_bytes = [semantic_piece_bytes(backend, i, piece) for i, piece in zip(ids, pieces)]
    if not (backend["strip_prepend"] and pieces and pieces[0].startswith("▁") and not text.startswith(" ")):
        return ids, pieces, piece_bytes, 0, False
    if pieces[0] == "▁":
        return ids[1:], pieces[1:], piece_bytes[1:], 1, True
    stripped = piece_bytes[0]
    if stripped is not None:
        if not stripped.startswith(b" "):
            raise ValueError("dummy-prefix piece bytes do not start with decoded SentencePiece space")
        stripped = stripped[1:]
    return ids, [pieces[0][1:]] + pieces[1:], [stripped] + piece_bytes[1:], 0, True


def utf8_scalar_spans(data: bytes):
    spans, i = [], 0
    while i < len(data):
        b0 = data[i]
        n = 1 if b0 < 0x80 else 2 if 0xC2 <= b0 <= 0xDF else 3 if 0xE0 <= b0 <= 0xEF else 4 if 0xF0 <= b0 <= 0xF4 else 0
        if not n or i + n > len(data):
            return None
        try:
            data[i:i + n].decode()
        except UnicodeDecodeError:
            return None
        spans.append((i, i + n)); i += n
    return spans


def split_codepoints(token_bytes):
    if token_bytes is None or any(x is None for x in token_bytes):
        return None, None, None, None
    lengths, data = [len(x) for x in token_bytes], b"".join(token_bytes)
    scalar_spans = utf8_scalar_spans(data)
    if scalar_spans is None:
        return None, data, lengths, None
    token_spans, pos = [], 0
    for length in lengths:
        token_spans.append((pos, pos + length)); pos += length
    split = sum(sum(a < end and b > start for a, b in token_spans) > 1 for start, end in scalar_spans)
    partial = 0
    for piece in token_bytes:
        try:
            piece.decode()
        except UnicodeDecodeError:
            partial += 1
    return split, data, lengths, partial


def backend_encode(backend, text: str):
    if backend["sp"] is not None:
        ids = backend["sp"].encode(text)
        return ids, [backend["sp"].id_to_piece(i) for i in ids], backend["sp"].decode(ids)
    enc = backend["fast"].encode(text, add_special_tokens=False)
    return enc.ids, enc.tokens, backend["fast"].decode(enc.ids, skip_special_tokens=False)


def scan_text(backend, text: str):
    ids, pieces, decoded = backend_encode(backend, text)
    fast_enc = backend["fast"].encode(text, add_special_tokens=False)
    eff_ids, eff_pieces, eff_bytes, dummy_removed, dummy_stripped = strip_dummy_prefix(backend, text, ids, pieces)
    split, concatenated, _, partial = split_codepoints(eff_bytes)
    nfc, nfd = unicodedata2.normalize("NFC", text), unicodedata2.normalize("NFD", text)
    nfc_ids, _, nfc_decoded = backend_encode(backend, nfc)
    nfd_ids, _, nfd_decoded = backend_encode(backend, nfd)
    clusters = len(regex.findall(r"\X", text))
    return {
        "token_count": len(eff_ids), "raw_token_count": len(ids), "dummy_prefix_tokens_removed": dummy_removed,
        "dummy_prefix_stripped": int(dummy_stripped), "grapheme_clusters": clusters,
        "tokens_per_cluster": len(eff_ids) / clusters if clusters else None, "whole_sequence_token": int(len(eff_ids) == 1),
        "token_ids": eff_ids, "tokens": eff_pieces, "tokenizer_roundtrip_exact": int(decoded == text),
        "decoded_text": decoded if decoded != text else None,
        "explicit_byte_fallback_count": sum(backend["sp"].is_byte(i) for i in eff_ids) if backend["sp"] is not None else 0,
        "byte_level_single_byte_atom_count": sum(len(x) == 1 for x in eff_bytes if x is not None) if backend["bytelevel"] else None,
        "partial_utf8_token_count": partial, "split_codepoint_count": split,
        "token_bytes_concat_equals_input": None if concatenated is None else int(concatenated == text.encode()),
        "nfc_nfd_canonical_distinct": int(nfc != nfd), "nfc_token_count": len(nfc_ids), "nfd_token_count": len(nfd_ids),
        "nfc_nfd_token_count_difference": len(nfc_ids) - len(nfd_ids), "nfc_nfd_token_ids_differ": int(nfc_ids != nfd_ids),
        "nfc_roundtrip_exact": int(nfc_decoded == nfc), "nfd_roundtrip_exact": int(nfd_decoded == nfd),
        "native_backend_ids_match": None if backend["sp"] is None else int(fast_enc.ids == ids), "backend": backend["backend"],
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "normalization_unicodedata_version": unicodedata2.unidata_version, "tokenizers_library_version": tokenizers_pkg.__version__,
        "sentencepiece_library_version": spm.__version__ if backend["sp"] is not None else None,
    }


def norm_merges(merges):
    return [tuple(x.split()) if isinstance(x, str) else tuple(x) for x in merges or []]


def vocab_token_bytes(token: str, bytelevel: bool):
    if BYTE_RE.match(token):
        return None
    if bytelevel:
        try:
            return bytes(BYTE_REV[c] for c in token)
        except KeyError:
            return None
    try:
        return token.replace("▁", " ").encode()
    except UnicodeEncodeError:
        return None


def vocab_bytes_map(backend):
    out = {}
    if backend["sp"] is not None:
        sp = backend["sp"]
        for token_id in range(sp.get_piece_size()):
            if not (sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id)):
                data = semantic_piece_bytes(backend, token_id, sp.id_to_piece(token_id))
                if data:
                    out.setdefault(data, sp.id_to_piece(token_id))
        return out
    for token in backend["raw"]["model"]["vocab"]:
        data = vocab_token_bytes(token, backend["bytelevel"])
        if data:
            out.setdefault(data, token)
    return out


def inventory_row(tid: str, ages, chars, backend):
    cfg, raw, model = ucd.HF[tid], backend["raw"], backend["raw"]["model"]
    added = {x["content"] for x in raw.get("added_tokens", [])}
    regular_vocab = {k: v for k, v in model["vocab"].items() if k not in added}
    merges, represented = norm_merges(model.get("merges")), set()
    for data in vocab_bytes_map(backend):
        try:
            text = data.decode()
        except UnicodeDecodeError:
            continue
        if len(text) == 1:
            represented.add(ord(text))
    by_version = defaultdict(lambda: [0, 0])
    for cp in chars:
        version = ages.get(cp)
        if version:
            by_version[version][1] += 1
            by_version[version][0] += cp in represented
    uvkey = lambda v: tuple(int(x) for x in v.split("."))
    coverage = {v: {"represented": a, "assigned": b, "fraction": a / b if b else None} for v, (a, b) in sorted(by_version.items(), key=lambda x: uvkey(x[0]))}
    primary, model_file = backend["directory"] / "tokenizer.json", backend["directory"] / "tokenizer.model"
    return {
        "tokenizer_id": tid, "family": cfg["family"], "artifact_url": f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/tokenizer.json",
        "repository": cfg["repo"], "revision": cfg["revision"], "artifact_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(), "artifact_size": primary.stat().st_size,
        "sentencepiece_model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest() if model_file.exists() else None,
        "license_status": cfg["license_status"], "first_public_date": cfg["release_date"], "first_public_evidence_url": cfg["evidence"],
        "release_date": cfg["release_date"], "construction_window_start": "unknown", "construction_window_end": cfg["release_date"],
        "vintage_confidence": "medium" if cfg["license_status"] == "open" else "low: mirror used because canonical repository is gated",
        "algorithm": model["type"], "backend": backend["backend"], "sentencepiece_metadata": backend["sp_meta"] or None,
        "vocab_size": len(model["vocab"]), "regular_vocab_size": len(regular_vocab),
        "regular_vocab_sha256": sha_text(canon(regular_vocab)), "merges_sha256": sha_text(canon(merges)), "merge_count": len(merges),
        "normalizer": canon(raw.get("normalizer")), "pre_tokenizer_hash": sha_text(canon(raw.get("pre_tokenizer"))),
        "byte_fallback_mode": "byte-level-bpe" if backend["bytelevel"] else ("explicit-byte-fallback" if backend["sp"] is not None else "none/unknown"),
        "unicode_table_version": f"unknown embedded in tokenizers {tokenizers_pkg.__version__}; host Python unicodedata {unicodedata.unidata_version}; diagnostics use unicodedata2 {unicodedata2.unidata_version}",
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "normalization_unicodedata_version": unicodedata2.unidata_version,
        "max_atomic_age": max((v for v, x in coverage.items() if x["represented"]), key=uvkey, default=None), "coverage_by_unicode_version": coverage,
        "notes": "tokenizer files only; no model weights downloaded; SentencePiece scans use native ids and raw protobuf piece bytes with tokenizer.json cross-checks",
    }


def newest_cohort(item):
    return (item["emoji_version"] == "E15.1" and item["sequence_type"] == "zwj") or (item["emoji_version"] == "E17.0" and item["sequence_type"] == "zwj") or (item["emoji_version"] in ("E16.0", "E17.0", "E18.0") and item["sequence_type"] == "atomic")


def scan_rows(input_path: Path, output_path: Path):
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    out, backend_mismatches = [], {}
    for tid in sorted(ucd.HF):
        backend, mismatches = load_backend(tid), 0
        for row in rows:
            metrics = scan_text(backend, row["text"])
            mismatches += metrics["native_backend_ids_match"] == 0
            result = {k: v for k, v in row.items() if k != "text"} | metrics | {"text": row["text"], "tokenizer_id": tid, "family": ucd.HF[tid]["family"]}
            out.append(result)
        backend_mismatches[tid] = mismatches
    with output_path.open("w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    meta = {"input": str(input_path), "output": str(output_path), "rows": len(out), "native_backend_id_mismatches": backend_mismatches,
            "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
            "normalization_unicodedata_version": unicodedata2.unidata_version, "tokenizers_library_version": tokenizers_pkg.__version__, "sentencepiece_library_version": spm.__version__}
    output_path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({"rows": len(out), "output": str(output_path), "native_backend_id_mismatches": backend_mismatches}, indent=2))


def build_inventory():
    ages, _ = ucd.parse_age(DATA / "unicode" / "DerivedAge.txt")
    chars, _ = ucd.parse_unicode_data(DATA / "unicode" / "UnicodeData.txt")
    rows, _, _ = ucd.parse_emoji_test(DATA / "unicode" / "emoji-test.txt")
    universe, _ = ucd.build_emoji_universe(rows)
    inventory = []
    for tid in sorted(ucd.HF):
        backend, row = load_backend(tid), None
        row = inventory_row(tid, ages, chars, backend)
        vocab_bytes = vocab_bytes_map(backend)
        def new_curve():
            return {"n": 0, "single": 0, "vocab_exact": 0, "atomic_n": 0, "atomic_single": 0, "atomic_vocab_exact": 0, "atomic_split": [], "zwj_n": 0, "zwj_single": 0, "zwj_vocab_exact": 0, "zwj_tpc": [], "zwj_split": []}
        curves = defaultdict(new_curve)
        newest = {"families": 0, "pipeline_single_token_count": 0, "exact_vocab_entry_count": 0, "exact_vocab_entries": [], "pipeline_single_tokens": []}
        native_mismatches = 0
        for item in universe:
            text = "".join(chr(x) for x in item["codepoints"])
            metrics = scan_text(backend, text)
            native_mismatches += metrics["native_backend_ids_match"] == 0
            exact_vocab = int(text.encode() in vocab_bytes)
            x = curves[item["emoji_version"]]
            x["n"] += 1; x["single"] += metrics["whole_sequence_token"]; x["vocab_exact"] += exact_vocab
            typ = item["sequence_type"]
            if typ in ("atomic", "zwj"):
                x[f"{typ}_n"] += 1; x[f"{typ}_single"] += metrics["whole_sequence_token"]; x[f"{typ}_vocab_exact"] += exact_vocab
                if metrics["split_codepoint_count"] is not None:
                    x[f"{typ}_split"].append(metrics["split_codepoint_count"])
                if typ == "zwj":
                    x["zwj_tpc"].append(metrics["tokens_per_cluster"])
            if newest_cohort(item):
                newest["families"] += 1
                newest["pipeline_single_token_count"] += metrics["whole_sequence_token"]
                newest["exact_vocab_entry_count"] += exact_vocab
                if exact_vocab:
                    newest["exact_vocab_entries"].append({"emoji_version": item["emoji_version"], "name": item["name"], "vocab_piece": vocab_bytes[text.encode()]})
                if metrics["whole_sequence_token"]:
                    newest["pipeline_single_tokens"].append({"emoji_version": item["emoji_version"], "name": item["name"]})
        row["native_backend_id_mismatches_in_rgi_scan"] = native_mismatches
        row["pilot_newest_cohort_exactness"] = newest
        row["rgi_single_token_rate_by_emoji_version"] = {}
        mean = lambda z: sum(z) / len(z) if z else None
        for version, x in sorted(curves.items(), key=lambda z: list(ucd.EMOJI_DATES).index(z[0])):
            row["rgi_single_token_rate_by_emoji_version"][version] = {
                "families": x["n"], "single_token_count": x["single"], "single_token_rate": x["single"] / x["n"],
                "exact_vocab_count": x["vocab_exact"], "exact_vocab_rate": x["vocab_exact"] / x["n"],
                "atomic_families": x["atomic_n"], "atomic_single_token_count": x["atomic_single"], "atomic_single_token_rate": x["atomic_single"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_exact_vocab_count": x["atomic_vocab_exact"], "atomic_exact_vocab_rate": x["atomic_vocab_exact"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_mean_split_codepoint_count": mean(x["atomic_split"]),
                "zwj_families": x["zwj_n"], "zwj_single_token_count": x["zwj_single"], "zwj_single_token_rate": x["zwj_single"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_exact_vocab_count": x["zwj_vocab_exact"], "zwj_exact_vocab_rate": x["zwj_vocab_exact"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_mean_tokens_per_cluster": mean(x["zwj_tpc"]), "zwj_mean_split_codepoint_count": mean(x["zwj_split"]),
            }
        inventory.append(row)
    continuity = []
    for a, b in (("qwen2.5-0.5b-instruct", "qwen3-0.6b"), ("tinyllama-1.1b-chat", "llama2-7b-hf-mirror")):
        ra = next(x for x in inventory if x["tokenizer_id"] == a); rb = next(x for x in inventory if x["tokenizer_id"] == b)
        continuity.append({"left": a, "right": b, "regular_vocab_equal": ra["regular_vocab_sha256"] == rb["regular_vocab_sha256"], "merges_equal": ra["merges_sha256"] == rb["merges_sha256"], "left_vocab": ra["regular_vocab_size"], "right_vocab": rb["regular_vocab_size"], "left_merges": ra["merge_count"], "right_merges": rb["merge_count"]})
    (RESULTS / "tokenizer_inventory.json").write_text(json.dumps({"tokenizers": inventory, "continuity": continuity}, indent=2, ensure_ascii=False))
    print(json.dumps({"tokenizers": len(inventory), "continuity": continuity}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "scan"])
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.mode == "inventory":
        build_inventory()
    elif args.input and args.output:
        scan_rows(args.input, args.output)
    else:
        raise SystemExit("scan requires input and output paths")


if __name__ == "__main__":
    main()
