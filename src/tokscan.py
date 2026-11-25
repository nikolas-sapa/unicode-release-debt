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

ROOT = ucd.ROOT
DATA = ucd.DATA
RESULTS = ucd.RESULTS
BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
SPACING_MARK = "▁".encode("utf-8")


def byte_reverse():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_REV = byte_reverse()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def varint(buf: bytes, pos: int):
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
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
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = buf[pos:pos + size]
            pos += size
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def first_embedded_scalar(buf: bytes, section_field: int, scalar_field: int):
    for field, wire, section in wire_fields(buf):
        if field == section_field and wire == 2:
            for subfield, subwire, value in wire_fields(section):
                if subfield == scalar_field and subwire == 0:
                    return value
    return None


def piece_bytes_from_record(record: bytes) -> bytes:
    for field, wire, value in wire_fields(record):
        if field == 1 and wire == 2:
            return value
    raise ValueError("SentencePiece piece record without piece bytes")


def sentencepiece_piece_bytes(path: Path) -> list[bytes]:
    return [piece_bytes_from_record(record[2]) for record in wire_fields(path.read_bytes()) if record[0] == 1 and record[1] == 2]


def tokenizer_meta(tid: str):
    directory = DATA / "hf" / tid
    path = directory / "tokenizer.json"
    expected = ucd.HF[tid]["files"]["tokenizer.json"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise ValueError(f"{tid} tokenizer.json hash mismatch")
    raw = json.loads(path.read_text())
    decoder = canon(raw.get("decoder"))
    normalizer = canon(raw.get("normalizer"))
    bytelevel = "ByteLevel" in decoder or "ByteLevel" in canon(raw.get("pre_tokenizer"))
    strip_prepend = '"type":"Prepend"' in normalizer and '▁' in normalizer
    return directory, raw, bytelevel, strip_prepend


def load_backend(tid: str):
    directory, raw, bytelevel, strip_prepend_json = tokenizer_meta(tid)
    fast = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model_path = directory / "tokenizer.model"
    sp = None
    piece_bytes = []
    sp_meta = {}
    strip_prepend = strip_prepend_json
    if model_path.exists():
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        proto = model_path.read_bytes()
        piece_bytes = sentencepiece_piece_bytes(model_path)
        if len(piece_bytes) != sp.get_piece_size():
            raise ValueError(f"{tid}: protobuf piece count {len(piece_bytes)} != sentencepiece piece count {sp.get_piece_size()}")
        sp_meta = {
            "model_type_enum": first_embedded_scalar(proto, 2, 3),
            "trainer_vocab_size": first_embedded_scalar(proto, 2, 4),
            "add_dummy_prefix": bool(first_embedded_scalar(proto, 3, 3)),
        }
        strip_prepend = sp_meta["add_dummy_prefix"]
    return {
        "tokenizer_id": tid, "directory": directory, "raw": raw, "fast": fast, "bytelevel": bytelevel,
        "sp": sp, "piece_bytes": piece_bytes, "sp_meta": sp_meta, "strip_prepend": strip_prepend,
        "backend": "sentencepiece-native+tokenizer.json-crosscheck" if sp else "tokenizers-tokenizer.json-raw",
    }


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
        return token.replace("▁", " ").encode("utf-8")
    except UnicodeEncodeError:
        return None


def strip_dummy_prefix(backend, text: str, ids, pieces):
    piece_bytes = [semantic_piece_bytes(backend, i, piece) for i, piece in zip(ids, pieces)]
    if not (backend["strip_prepend"] and pieces and pieces[0].startswith("▁") and not text.startswith(" ")):
        return ids, pieces, piece_bytes, 0, False
    if pieces[0] == "▁":
        return ids[1:], pieces[1:], piece_bytes[1:], 1, True
    stripped_piece = pieces[0][1:]
    stripped_bytes = piece_bytes[0]
    if stripped_bytes is not None:
        if not stripped_bytes.startswith(b" "):
            raise ValueError("dummy-prefix piece bytes do not start with decoded SentencePiece space")
        stripped_bytes = stripped_bytes[1:]
    return ids, [stripped_piece] + pieces[1:], [stripped_bytes] + piece_bytes[1:], 0, True


def utf8_scalar_spans(data: bytes):
    spans, i = [], 0
    while i < len(data):
        b0 = data[i]
        if b0 < 0x80:
            n = 1
        elif 0xC2 <= b0 <= 0xDF:
            n = 2
        elif 0xE0 <= b0 <= 0xEF:
            n = 3
        elif 0xF0 <= b0 <= 0xF4:
            n = 4
        else:
            return None
        if i + n > len(data):
            return None
        chunk = data[i:i + n]
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            return None
        spans.append((i, i + n))
        i += n
    return spans


def split_codepoints(token_bytes):
    if token_bytes is None or any(x is None for x in token_bytes):
        return None, None, None, None
    lengths = [len(x) for x in token_bytes]
    data = b"".join(token_bytes)
    scalar_spans = utf8_scalar_spans(data)
    if scalar_spans is None:
        return None, data, lengths, None
    token_spans, pos = [], 0
    for length in lengths:
        token_spans.append((pos, pos + length))
        pos += length
    split = 0
    for start, end in scalar_spans:
        overlapping = [i for i, (a, b) in enumerate(token_spans) if a < end and b > start]
        if len(overlapping) > 1:
            split += 1
    partial = 0
    for piece in token_bytes:
        try:
            piece.decode("utf-8")
        except UnicodeDecodeError:
            partial += 1
    return split, data, lengths, partial


def backend_encode(backend, text: str):
    if backend["sp"] is not None:
        ids = backend["sp"].encode(text)
        pieces = [backend["sp"].id_to_piece(i) for i in ids]
        decoded = backend["sp"].decode(ids)
    else:
        enc = backend["fast"].encode(text, add_special_tokens=False)
        ids, pieces = enc.ids, enc.tokens
        decoded = backend["fast"].decode(ids, skip_special_tokens=False)
    return ids, pieces, decoded


def scan_text(backend, text: str):
    ids, pieces, decoded = backend_encode(backend, text)
    fast_enc = backend["fast"].encode(text, add_special_tokens=False)
    native_match = None if backend["sp"] is None else int(fast_enc.ids == ids)
    raw_token_count = len(ids)
    dummy_result = strip_dummy_prefix(backend, text, ids, pieces)
    eff_ids, eff_pieces, eff_bytes, dummy_tokens_removed, dummy_stripped = dummy_result
    split, concatenated, lengths, partial = split_codepoints(eff_bytes)
    nfc = unicodedata2.normalize("NFC", text)
    nfd = unicodedata2.normalize("NFD", text)
    nfc_ids, _, nfc_decoded = backend_encode(backend, nfc)
    nfd_ids, _, nfd_decoded = backend_encode(backend, nfd)
    clusters = len(regex.findall(r"\X", text))
    explicit_fallback = sum(backend["sp"].is_byte(i) for i in eff_ids) if backend["sp"] is not None else 0
    return {
        "token_count": len(eff_ids), "raw_token_count": raw_token_count, "dummy_prefix_tokens_removed": dummy_tokens_removed,
        "dummy_prefix_stripped": int(dummy_stripped), "grapheme_clusters": clusters,
        "tokens_per_cluster": len(eff_ids) / clusters if clusters else None,
        "whole_sequence_token": int(len(eff_ids) == 1), "token_ids": eff_ids, "tokens": eff_pieces,
        "tokenizer_roundtrip_exact": int(decoded == text), "decoded_text": decoded if decoded != text else None,
        "explicit_byte_fallback_count": explicit_fallback,
        "byte_level_single_byte_atom_count": sum(len(x) == 1 for x in eff_bytes if x is not None) if backend["bytelevel"] else None,
        "partial_utf8_token_count": partial, "split_codepoint_count": split,
        "token_bytes_concat_equals_input": None if concatenated is None else int(concatenated == text.encode()),
        "nfc_nfd_canonical_distinct": int(nfc != nfd), "nfc_token_count": len(nfc_ids), "nfd_token_count": len(nfd_ids),
        "nfc_nfd_token_count_difference": len(nfc_ids) - len(nfd_ids), "nfc_nfd_token_ids_differ": int(nfc_ids != nfd_ids),
        "nfc_roundtrip_exact": int(nfc_decoded == nfc), "nfd_roundtrip_exact": int(nfd_decoded == nfd),
        "native_backend_ids_match": native_match, "backend": backend["backend"],
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "normalization_unicodedata_version": unicodedata2.unidata_version, "tokenizers_library_version": tokenizers_pkg.__version__,
        "sentencepiece_library_version": spm.__version__ if backend["sp"] is not None else None,
    }


def norm_merges(merges):
    out = []
    for merge in merges or []:
        if isinstance(merge, str):
            out.append(tuple(merge.split()))
        else:
            out.append(tuple(merge))
    return out


def vocab_token_bytes(token: str, bytelevel: bool):
    if BYTE_RE.match(token):
        return None
    if bytelevel:
        try:
            return bytes(BYTE_REV[c] for c in token)
        except KeyError:
            return None
    try:
        return token.replace("▁", " ").encode("utf-8")
    except UnicodeEncodeError:
        return None


def vocab_bytes_map(backend):
    out = {}
    if backend["sp"] is not None:
        sp = backend["sp"]
        for token_id in range(sp.get_piece_size()):
            if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id):
                continue
            piece = sp.id_to_piece(token_id)
            data = semantic_piece_bytes(backend, token_id, piece)
            if data:
                out.setdefault(data, piece)
        return out
    for token in backend["raw"]["model"]["vocab"]:
        data = vocab_token_bytes(token, backend["bytelevel"])
        if data:
            out.setdefault(data, token)
    return out


def inventory_row(tid: str, ages, chars, backend):
    cfg = ucd.HF[tid]
    raw = backend["raw"]
    model = raw["model"]
    added = {x["content"] for x in raw.get("added_tokens", [])}
    regular_vocab = {k: v for k, v in model["vocab"].items() if k not in added}
    merges = norm_merges(model.get("merges"))
    represented = {}
    for data in vocab_bytes_map(backend):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) == 1:
            represented[ord(text)] = True
    by_version = defaultdict(lambda: [0, 0])
    for cp in chars:
        version = ages.get(cp)
        if version:
            by_version[version][1] += 1
            if cp in represented:
                by_version[version][0] += 1
    uvkey = lambda v: tuple(int(x) for x in v.split("."))
    coverage = {v: {"represented": a, "assigned": b, "fraction": a / b if b else None} for v, (a, b) in sorted(by_version.items(), key=lambda x: uvkey(x[0]))}
    primary = backend["directory"] / "tokenizer.json"
    model_file = backend["directory"] / "tokenizer.model"
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
    if item["emoji_version"] == "E15.1" and item["sequence_type"] == "zwj":
        return True
    if item["emoji_version"] == "E17.0" and item["sequence_type"] == "zwj":
        return True
    return item["emoji_version"] in ("E16.0", "E17.0", "E18.0") and item["sequence_type"] == "atomic"


def scan_rows(input_path: Path, output_path: Path):
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    out = []
    backend_mismatches = {}
    for tid in sorted(ucd.HF):
        backend = load_backend(tid)
        mismatches = 0
        for row in rows:
            metrics = scan_text(backend, row["text"])
            mismatches += metrics["native_backend_ids_match"] == 0
            result = {k: v for k, v in row.items() if k != "text"}
            result.update(metrics)
            result["text"] = row["text"]
            result["tokenizer_id"] = tid
            result["family"] = ucd.HF[tid]["family"]
            out.append(result)
        backend_mismatches[tid] = mismatches
    with output_path.open("w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    meta = {
        "input": str(input_path), "output": str(output_path), "rows": len(out), "native_backend_id_mismatches": backend_mismatches,
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "normalization_unicodedata_version": unicodedata2.unidata_version, "tokenizers_library_version": tokenizers_pkg.__version__, "sentencepiece_library_version": spm.__version__,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({"rows": len(out), "output": str(output_path), "native_backend_id_mismatches": backend_mismatches}, indent=2))


def build_inventory():
    ages, _ = ucd.parse_age(DATA / "unicode" / "DerivedAge.txt")
    chars, _ = ucd.parse_unicode_data(DATA / "unicode" / "UnicodeData.txt")
    rows, _, _ = ucd.parse_emoji_test(DATA / "unicode" / "emoji-test.txt")
    universe, _ = ucd.build_emoji_universe(rows)
    inventory = []
    for tid in sorted(ucd.HF):
        backend = load_backend(tid)
        row = inventory_row(tid, ages, chars, backend)
        vocab_bytes = vocab_bytes_map(backend)
        curves = defaultdict(lambda: {"n": 0, "single": 0, "vocab_exact": 0, "atomic_n": 0, "atomic_single": 0, "atomic_vocab_exact": 0, "atomic_split": [], "zwj_n": 0, "zwj_single": 0, "zwj_vocab_exact": 0, "zwj_tpc": [], "zwj_split": []})
        newest = {"families": 0, "pipeline_single_token_count": 0, "exact_vocab_entry_count": 0, "exact_vocab_entries": [], "pipeline_single_tokens": []}
        native_mismatches = 0
        for item in universe:
            text = "".join(chr(x) for x in item["codepoints"])
            metrics = scan_text(backend, text)
            native_mismatches += metrics["native_backend_ids_match"] == 0
            exact_vocab = int(text.encode() in vocab_bytes)
            version = item["emoji_version"]
            curves[version]["n"] += 1
            curves[version]["single"] += metrics["whole_sequence_token"]
            curves[version]["vocab_exact"] += exact_vocab
            if item["sequence_type"] == "atomic":
                curves[version]["atomic_n"] += 1
                curves[version]["atomic_single"] += metrics["whole_sequence_token"]
                curves[version]["atomic_vocab_exact"] += exact_vocab
                if metrics["split_codepoint_count"] is not None:
                    curves[version]["atomic_split"].append(metrics["split_codepoint_count"])
            if item["sequence_type"] == "zwj":
                curves[version]["zwj_n"] += 1
                curves[version]["zwj_single"] += metrics["whole_sequence_token"]
                curves[version]["zwj_vocab_exact"] += exact_vocab
                curves[version]["zwj_tpc"].append(metrics["tokens_per_cluster"])
                if metrics["split_codepoint_count"] is not None:
                    curves[version]["zwj_split"].append(metrics["split_codepoint_count"])
            if newest_cohort(item):
                newest["families"] += 1
                newest["pipeline_single_token_count"] += metrics["whole_sequence_token"]
                newest["exact_vocab_entry_count"] += exact_vocab
                if exact_vocab:
                    newest["exact_vocab_entries"].append({"emoji_version": version, "name": item["name"], "vocab_piece": vocab_bytes[text.encode()]})
                if metrics["whole_sequence_token"]:
                    newest["pipeline_single_tokens"].append({"emoji_version": version, "name": item["name"]})
        row["native_backend_id_mismatches_in_rgi_scan"] = native_mismatches
        row["pilot_newest_cohort_exactness"] = newest
        row["rgi_single_token_rate_by_emoji_version"] = {}
        for version, x in sorted(curves.items(), key=lambda z: list(ucd.EMOJI_DATES).index(z[0])):
            row["rgi_single_token_rate_by_emoji_version"][version] = {
                "families": x["n"], "single_token_count": x["single"], "single_token_rate": x["single"] / x["n"],
                "exact_vocab_count": x["vocab_exact"], "exact_vocab_rate": x["vocab_exact"] / x["n"],
                "atomic_families": x["atomic_n"], "atomic_single_token_count": x["atomic_single"], "atomic_single_token_rate": x["atomic_single"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_exact_vocab_count": x["atomic_vocab_exact"], "atomic_exact_vocab_rate": x["atomic_vocab_exact"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_mean_split_codepoint_count": sum(x["atomic_split"]) / len(x["atomic_split"]) if x["atomic_split"] else None,
                "zwj_families": x["zwj_n"], "zwj_single_token_count": x["zwj_single"], "zwj_single_token_rate": x["zwj_single"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_exact_vocab_count": x["zwj_vocab_exact"], "zwj_exact_vocab_rate": x["zwj_vocab_exact"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_mean_tokens_per_cluster": sum(x["zwj_tpc"]) / len(x["zwj_tpc"]) if x["zwj_tpc"] else None,
                "zwj_mean_split_codepoint_count": sum(x["zwj_split"]) / len(x["zwj_split"]) if x["zwj_split"] else None,
            }
        inventory.append(row)
    continuity = []
    for a, b in (("qwen2.5-0.5b-instruct", "qwen3-0.6b"), ("tinyllama-1.1b-chat", "llama2-7b-hf-mirror")):
        ra, rb = (x for x in inventory if x["tokenizer_id"] in (a, b))
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
    else:
        if not args.input or not args.output:
            raise SystemExit("scan requires input and output paths")
        scan_rows(args.input, args.output)


if __name__ == "__main__":
    main()
