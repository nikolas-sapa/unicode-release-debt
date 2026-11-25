from __future__ import annotations

import json
import platform
import re
import sys
import unicodedata
from pathlib import Path

import regex
import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT = ucd.ROOT
DATA = ucd.DATA
RESULTS = ucd.RESULTS


def parse_break_test(line: str):
    tokens = line.split("#", 1)[0].strip().split()
    chars, breaks = [], []
    for token in tokens:
        if token in ("÷", "×"):
            breaks.append(token == "÷")
        else:
            chars.append(chr(int(token, 16)))
    clusters, current = [], []
    for i, char in enumerate(chars):
        if i and breaks[i]:
            clusters.append(current)
            current = []
        current.append(char)
    if current or chars:
        clusters.append(current)
    return "".join(chars), ["".join(x) for x in clusters]


def grapheme_validation(path: Path):
    total = failures = broken_failures = 0
    first_failure = None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        text, expected = parse_break_test(line)
        got = regex.findall(r"\X", text)
        total += 1
        if got != expected:
            failures += 1
            first_failure = first_failure or {"line": line_no, "expected": expected, "got": got}
        if list(text) != expected:
            broken_failures += 1
    tr29 = (DATA / "unicode" / "tr29.html").read_text(errors="replace")
    revision_match = re.search(r"Revision\s+(\d+)", tr29)
    revision = revision_match.group(1) if revision_match else "unknown"
    return {
        "implementation": f"regex {regex.__version__} \\X",
        "uax29_revision": revision, "unicode_version": "18.0.0", "cluster_type": "extended grapheme clusters",
        "revision_evidence": "pinned data/unicode/tr29.html revision parsed dynamically; implementation validated against pinned Unicode 18.0.0 GraphemeBreakTest.txt",
        "total": total, "passed": total - failures, "failures": failures, "pass_rate": (total - failures) / total,
        "first_failure": first_failure, "broken_split_every_codepoint_failures": broken_failures,
        "mutation_control_detects_wrong_boundaries": broken_failures > 0,
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_items():
    copy_rows = read_jsonl(RESULTS / "copy_pilot.jsonl")
    scan_units = read_jsonl(RESULTS / "scan_units.jsonl") if (RESULTS / "scan_units.jsonl").exists() else []
    delimiters = ["<IN>", "</IN>", "<OUT>", "</OUT>"]
    texts = [x["text"] for x in copy_rows]
    bad_utf8 = []
    for row in copy_rows:
        try:
            if row["text"].encode().decode() != row["text"] or any(0xD800 <= ord(c) <= 0xDFFF for c in row["text"]):
                bad_utf8.append(row["item_id"])
        except UnicodeError:
            bad_utf8.append(row["item_id"])
    return {
        "copy_rows": len(copy_rows), "copy_texts_byte_distinct": len(set(texts)) == len(texts),
        "utf8_and_scalar_valid_count": len(copy_rows) - len(bad_utf8), "bad_utf8": bad_utf8,
        "delimiter_collisions": [x["item_id"] for x in copy_rows if any(d in x["text"] for d in delimiters)],
        "boundary_whitespace": [x["item_id"] for x in copy_rows if x["text"] != x["text"].strip()],
        "scan_units": len(scan_units), "scan_unit_texts_nonempty": all(x["text"] for x in scan_units),
    }


def token_scan_invariants():
    import tokscan
    whole = tokscan.split_codepoints([bytes.fromhex("f09f9880")])[0]
    split = tokscan.split_codepoints([bytes.fromhex(x) for x in ("f0", "9f", "98", "80")])[0]
    checks = {"synthetic_whole_scalar_zero": whole == 0, "synthetic_byte_split_scalar_one": split == 1}
    for tid in sorted(ucd.HF):
        backend = tokscan.load_backend(tid)
        ascii_metrics = tokscan.scan_text(backend, "abc")
        literal_metrics = tokscan.scan_text(backend, "<0xF0>")
        checks[tid + "_ascii_roundtrip"] = ascii_metrics["tokenizer_roundtrip_exact"] == 1
        checks[tid + "_ascii_bytes_align"] = ascii_metrics["token_bytes_concat_equals_input"] == 1
        checks[tid + "_ascii_no_split"] = ascii_metrics["split_codepoint_count"] == 0
        checks[tid + "_literal_byte_spelling_not_fallback"] = literal_metrics["explicit_byte_fallback_count"] == 0
        if backend["sp"] is not None:
            checks[tid + "_native_fast_ids_match"] = ascii_metrics["native_backend_ids_match"] == 1
        if backend["strip_prepend"]:
            emoji_metrics = tokscan.scan_text(backend, "😀")
            checks[tid + "_dummy_prefix_removed_from_count"] = emoji_metrics["raw_token_count"] == emoji_metrics["token_count"] + emoji_metrics["dummy_prefix_tokens_removed"]
            checks[tid + "_dummy_prefix_bytes_align"] = emoji_metrics["token_bytes_concat_equals_input"] == 1
    for tid in ("qwen2.5-0.5b-instruct", "qwen3-0.6b"):
        metrics = tokscan.scan_text(tokscan.load_backend(tid), "café")
        checks[tid + "_nfd_loss_inversion_detected"] = metrics["nfc_nfd_token_ids_differ"] == 0 and metrics["nfc_roundtrip_exact"] == 1 and metrics["nfd_roundtrip_exact"] == 0
    return checks


def main():
    weight_suffixes = {".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt", ".h5", ".msgpack"}
    weight_files = [str(p.relative_to(ROOT)) for p in (DATA / "hf").rglob("*") if p.is_file() and p.suffix.lower() in weight_suffixes]
    report = {
        "grapheme": grapheme_validation(DATA / "unicode" / "GraphemeBreakTest.txt"),
        "items": validate_items(), "token_scan_invariants": token_scan_invariants(),
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "host_platform": platform.platform(), "host_machine": platform.machine(),
        "normalization_unicodedata_version": unicodedata2.unidata_version, "model_weight_files_present": weight_files,
    }
    gates = {
        "grapheme_100_percent": report["grapheme"]["failures"] == 0,
        "grapheme_mutation_control": report["grapheme"]["mutation_control_detects_wrong_boundaries"],
        "copy_rows_182": report["items"]["copy_rows"] == 182,
        "copy_texts_byte_distinct": report["items"]["copy_texts_byte_distinct"],
        "no_delimiter_collisions": not report["items"]["delimiter_collisions"],
        "no_boundary_whitespace": not report["items"]["boundary_whitespace"],
        "token_scan_invariants": all(report["token_scan_invariants"].values()),
        "tokenizer_files_only_no_model_weights": not weight_files,
    }
    report["gates"] = gates
    (RESULTS / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"grapheme": f"{report['grapheme']['passed']}/{report['grapheme']['total']}", "mutation_control": report["grapheme"]["mutation_control_detects_wrong_boundaries"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
