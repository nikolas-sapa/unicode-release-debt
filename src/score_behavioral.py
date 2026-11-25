from __future__ import annotations

import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import ucd_build as ucd

RESULTS = ucd.RESULTS
BEHAVIORAL = bp.BEHAVIORAL
RAW_PATH = BEHAVIORAL / "behavioral_raw.jsonl"
SCORED_PATH = BEHAVIORAL / "behavioral_scored.jsonl"
OUT_RE = re.compile(r"<OUT>(.*?)</OUT>", re.S)
DECOMP_RE = re.compile(r"U\+[0-9A-F]{4,6}(?:\s+U\+[0-9A-F]{4,6})*")
INT_RE = re.compile(r"[+-]?\d+")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_copy(text: str) -> dict:
    matches = list(OUT_RE.finditer(text))
    unique = len(matches) == 1 and text.count("<OUT>") == 1 and text.count("</OUT>") == 1
    if not unique:
        return {"extracted_text": None, "format_failure": True, "outside_wrapper_text": None, "wrapper_match_count": len(matches)}
    match = matches[0]
    outside = text[:match.start()] + text[match.end():]
    return {"extracted_text": match.group(1), "format_failure": False, "outside_wrapper_text": outside, "wrapper_match_count": 1}


def edit_counts(a: str, b: str) -> dict:
    counts = {"inserted_codepoints": 0, "deleted_codepoints": 0, "replaced_codepoints": 0}
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, list(a), list(b), autojunk=False).get_opcodes():
        if op == "insert":
            counts["inserted_codepoints"] += b2 - b1
        elif op == "delete":
            counts["deleted_codepoints"] += a2 - a1
        elif op == "replace":
            counts["replaced_codepoints"] += max(a2 - a1, b2 - b1)
    return counts


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def byte_diff(target: str, output: str) -> dict:
    a, b = target.encode(), output.encode()
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return {
        "target_utf8_bytes": len(a), "output_utf8_bytes": len(b), "byte_length_difference": len(b) - len(a),
        "common_prefix_bytes": common, "first_byte_mismatch_offset": common if a != b else None,
        "target_diff_hex_context": a[common:common + 16].hex(), "output_diff_hex_context": b[common:common + 16].hex(),
    }


def copy_taxonomy(target: str, output: str) -> dict:
    exact = output.encode() == target.encode()
    edits = edit_counts(target, output)
    canonical_equivalent = unicodedata2.normalize("NFC", target) == unicodedata2.normalize("NFC", output) or unicodedata2.normalize("NFD", target) == unicodedata2.normalize("NFD", output)
    no_zwj_target = target.replace("\u200d", "")
    whitespaceless_output = re.sub(r"\s+", "", output)
    classes = {
        "normalization_only_mismatch": (not exact) and canonical_equivalent,
        "missing_zwj": target.count("\u200d") > output.count("\u200d"),
        "missing_vs16": target.count("\ufe0f") > output.count("\ufe0f"),
        "decomposed_rgi_sequence": "\u200d" in target and (output == no_zwj_target or whitespaceless_output == no_zwj_target),
        "truncation": len(output.encode()) < len(target.encode()) and target.encode().startswith(output.encode()),
        "extra_text_after_exact_target": len(output.encode()) > len(target.encode()) and output.encode().startswith(target.encode()),
        "insertion": edits["inserted_codepoints"] > 0,
        "deletion": edits["deleted_codepoints"] > 0,
        "substitution": edits["replaced_codepoints"] > 0,
        "replacement_character_emitted": "\ufffd" in output,
        "reordering": len(target) == len(output) and sorted(target) == sorted(output) and target != output,
    }
    informative = ["normalization_only_mismatch", "missing_zwj", "missing_vs16", "decomposed_rgi_sequence", "truncation", "extra_text_after_exact_target", "insertion", "deletion", "substitution", "replacement_character_emitted", "reordering"]
    classes["other_unclassified_mismatch"] = (not exact) and not any(classes[k] for k in informative)
    return {
        **classes, **edits, "codepoint_levenshtein_distance": levenshtein(target, output),
        "canonical_equivalent": canonical_equivalent, **byte_diff(target, output),
    }


def parse_count(text: str) -> dict:
    stripped = text.strip()
    ok = bool(INT_RE.fullmatch(stripped))
    return {"parsed_answer": int(stripped) if ok else None, "format_failure": not ok, "parser": "full-strip-integer"}


def parse_decomp(text: str) -> dict:
    stripped = text.strip()
    ok = bool(DECOMP_RE.fullmatch(stripped))
    vals = stripped.split() if ok else []
    return {"parsed_answer": vals if ok else None, "format_failure": not ok, "parser": "full-strip-uppercase-U+hex-token-list"}


def parse_compare(text: str) -> dict:
    stripped = text.strip()
    ok = stripped in ("SAME", "REORDERED", "DIFFERENT")
    return {"parsed_answer": stripped if ok else None, "format_failure": not ok, "parser": "full-strip-exact-label"}


def probe_lookup(model_id: str) -> dict:
    cfg = bp.model_config(model_id)
    return {x["probe_key"]: x for x in bp.probe_rows(cfg["scan_tokenizer_id"])}


def score_row(raw: dict, probe: dict) -> dict:
    typ = raw["probe_type"]
    base = {
        "completion_key": raw["completion_key"], "model_id": raw["model_id"], "probe_key": raw["probe_key"], "probe_type": typ,
        "batch_id": raw["batch_id"], "raw_text": raw["raw_text"], "output_token_count": raw["output_token_count"], "finish_reason_inferred": raw["finish_reason_inferred"],
        "hf_repo": raw["hf_repo"], "model_revision": raw["model_revision"], "upstream_repo": raw["upstream_repo"], "upstream_revision": raw["upstream_revision"],
        "scan_tokenizer_id": raw["scan_tokenizer_id"], "dtype": raw["dtype"], "decode_params": raw["decode_params"],
        "host_python_version": raw["host_python_version"], "host_unicodedata_version": raw["host_unicodedata_version"],
    }
    for key in ("cohort", "copy_role", "item_id", "item_type", "era", "fragmentation_group", "pair_id", "scan_tokenizer_id"):
        if key in probe:
            base[key] = probe[key]
    if typ == "copy":
        extraction = extract_copy(raw["raw_text"])
        target = probe["expected_text"]
        if extraction["format_failure"]:
            taxonomy = {}
            exact = False
            content_failure = None
        else:
            taxonomy = copy_taxonomy(target, extraction["extracted_text"])
            exact = extraction["extracted_text"].encode() == target.encode()
            content_failure = not exact
        base.update({
            "target": target, "expected_text": target, **extraction, "extracted_answer": extraction["extracted_text"],
            "task_exact": exact, "copy_exact": exact, "content_failure_given_format": content_failure,
            "total_failure": not exact, **taxonomy,
        })
    elif typ == "count":
        parsed = parse_count(raw["raw_text"])
        exact = parsed["parsed_answer"] == probe["expected_count"] if not parsed["format_failure"] else False
        base.update({
            "target": probe["target"], "expected_count": probe["expected_count"], **parsed, "extracted_answer": parsed["parsed_answer"],
            "task_exact": exact, "count_exact": exact, "content_failure_given_format": (not exact) if not parsed["format_failure"] else None,
            "total_failure": not exact,
        })
    elif typ == "decomposition":
        parsed = parse_decomp(raw["raw_text"])
        exact = parsed["parsed_answer"] == probe["expected_codepoints"] if not parsed["format_failure"] else False
        base.update({
            "target": probe["target"], "expected_codepoints": probe["expected_codepoints"], **parsed, "extracted_answer": parsed["parsed_answer"],
            "task_exact": exact, "decomp_exact": exact, "content_failure_given_format": (not exact) if not parsed["format_failure"] else None,
            "total_failure": not exact,
        })
    elif typ == "comparison":
        parsed = parse_compare(raw["raw_text"])
        exact = parsed["parsed_answer"] == probe["expected_label"] if not parsed["format_failure"] else False
        base.update({
            "a_text": probe["a_text"], "b_text": probe["b_text"], "expected_label": probe["expected_label"], **parsed, "extracted_answer": parsed["parsed_answer"],
            "task_exact": exact, "compare_exact": exact, "content_failure_given_format": (not exact) if not parsed["format_failure"] else None,
            "total_failure": not exact,
        })
    else:
        raise ValueError(typ)
    return base


def mutation_negative_control() -> dict:
    cfg = bp.model_configs()[0]
    copy_rows = [x for x in bp.probe_rows(cfg["scan_tokenizer_id"]) if x["probe_type"] == "copy"]
    content_probe = next((x for x in copy_rows if x.get("copy_role") == "target"), copy_rows[0])
    ascii_probe = next((x for x in copy_rows if x.get("cohort") == "ASCII"), copy_rows[0])

    def fake_raw(probe, text):
        return {
            "completion_key": "mutation-negative-control", "model_id": cfg["model_id"], "probe_key": probe["probe_key"], "probe_type": probe["probe_type"],
            "batch_id": "mutation-negative-control", "raw_text": text, "output_token_count": 0, "finish_reason_inferred": "synthetic",
            "hf_repo": cfg["hf_repo"], "model_revision": cfg["revision"], "upstream_repo": cfg.get("upstream_repo"), "upstream_revision": cfg.get("upstream_revision"),
            "scan_tokenizer_id": cfg["scan_tokenizer_id"], "dtype": "synthetic", "decode_params": {}, "host_python_version": "synthetic", "host_unicodedata_version": "synthetic",
        }

    wrong_content = score_row(fake_raw(content_probe, "<OUT>mutation-negative-control</OUT>"), content_probe)
    wrong_ascii = score_row(fake_raw(ascii_probe, "<OUT>mutation-negative-control</OUT>"), ascii_probe)
    broken_wrapper = score_row(fake_raw(ascii_probe, "mutation-negative-control"), ascii_probe)
    checks = {
        "content_mutation_rejected": wrong_content["task_exact"] is False and wrong_content["total_failure"] is True,
        "ascii_content_mutation_rejected": wrong_ascii["task_exact"] is False and wrong_ascii["total_failure"] is True,
        "missing_wrapper_rejected": broken_wrapper["format_failure"] is True and broken_wrapper["task_exact"] is False,
    }
    return {"checks": checks, "mutation_negative_control_pass": all(checks.values()), "content_mutation_exact": wrong_content["task_exact"], "ascii_mutation_exact": wrong_ascii["task_exact"], "missing_wrapper_format_failure": broken_wrapper["format_failure"]}


def main():
    raw = [x for x in read_jsonl(RAW_PATH) if x.get("status") == "ok"]
    keys = [x["completion_key"] for x in raw]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate completion keys in append-only raw file; manual correction batch required")
    lookups = {}
    for row in raw:
        lookups.setdefault(row["model_id"], probe_lookup(row["model_id"]))
    missing = [x["completion_key"] for x in raw if x["probe_key"] not in lookups[x["model_id"]]]
    if missing:
        raise RuntimeError(f"raw rows without probe definitions: {missing[:5]}")
    scored = [score_row(x, lookups[x["model_id"]][x["probe_key"]]) for x in raw]
    bp.write_jsonl(SCORED_PATH, scored)
    mutation = mutation_negative_control()
    (BEHAVIORAL / "behavioral_mutation_check.json").write_text(json.dumps(mutation, indent=2))
    print(json.dumps({"raw_rows": len(raw), "scored_rows": len(scored), "output": str(SCORED_PATH), "mutation_negative_control_pass": mutation["mutation_negative_control_pass"], "host_unicodedata_version": unicodedata.unidata_version, "normalization_unicodedata_version": unicodedata2.unidata_version}, indent=2))


if __name__ == "__main__":
    main()
