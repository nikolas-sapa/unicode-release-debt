from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2
from score_behavioral import parse_compare, parse_count, parse_decomp

ROUND2 = r2.ROUND2
RAW_PATH = ROUND2 / "round2_raw.jsonl"
SCORED_PATH = ROUND2 / "round2_scored.jsonl"
INT_RE = re.compile(r"[+-]?\d+")


def extract_copy_round2(text: str) -> dict:
    open_count = text.count("<OUT>")
    close_count = text.count("</OUT>")
    start = text.find("<OUT>") if open_count == 1 else -1
    end = text.find("</OUT>", start + len("<OUT>")) if start >= 0 and close_count == 1 else -1
    if open_count != 1 or close_count != 1 or start < 0 or end < 0:
        return {"extracted_text": None, "format_failure": True, "outside_wrapper_text": None, "wrapper_match_count": 0, "format_failure_reason": "wrapper count or ordering mismatch"}
    prefix = text[:start]
    suffix = text[end + len("</OUT>"):]
    if prefix.strip() or suffix.strip():
        return {"extracted_text": None, "format_failure": True, "outside_wrapper_text": prefix + suffix, "wrapper_match_count": 1, "format_failure_reason": "non-whitespace outside wrapper"}
    return {
        "extracted_text": text[start + len("<OUT>"):end],
        "format_failure": False,
        "outside_wrapper_text": prefix + suffix,
        "outside_wrapper_whitespace_only": True,
        "wrapper_match_count": 1,
        "format_failure_reason": None,
    }


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


def copy_taxonomy_round2(target: str, output: str) -> dict:
    exact = output.encode() == target.encode()
    edits = edit_counts(target, output)
    canonical_equivalent = unicodedata2.normalize("NFC", target) == unicodedata2.normalize("NFC", output) or unicodedata2.normalize("NFD", target) == unicodedata2.normalize("NFD", output)
    no_zwj_target = target.replace("\u200d", "")
    whitespaceless_output = re.sub(r"\s+", "", output)
    newline_stripped = output.strip("\r\n")
    newline_padding = (not exact) and newline_stripped == target and len(output) != len(target)
    classes = {
        "normalization_only_mismatch": (not exact) and canonical_equivalent,
        "nfc_nfd_variant": (not exact) and canonical_equivalent,
        "missing_zwj": target.count("\u200d") > output.count("\u200d"),
        "missing_vs16": target.count("\ufe0f") > output.count("\ufe0f"),
        "decomposed_rgi_sequence": "\u200d" in target and (output == no_zwj_target or whitespaceless_output == no_zwj_target),
        "truncation": len(output.encode()) < len(target.encode()) and target.encode().startswith(output.encode()),
        "newline_padding": newline_padding,
        "extra_text_after_exact_target": len(output.encode()) > len(target.encode()) and output.encode().startswith(target.encode()) and not newline_padding,
        "insertion": edits["inserted_codepoints"] > 0,
        "deletion": edits["deleted_codepoints"] > 0,
        "substitution": edits["replaced_codepoints"] > 0,
        "replacement_character_emitted": "\ufffd" in output,
        "reordering": len(target) == len(output) and sorted(target) == sorted(output) and target != output,
    }
    informative = ["normalization_only_mismatch", "missing_zwj", "missing_vs16", "decomposed_rgi_sequence", "truncation", "newline_padding", "extra_text_after_exact_target", "insertion", "deletion", "substitution", "replacement_character_emitted", "reordering"]
    classes["other_unclassified_mismatch"] = (not exact) and not any(classes[k] for k in informative)
    return {
        **classes, **edits, "codepoint_levenshtein_distance": levenshtein(target, output),
        "canonical_equivalent": canonical_equivalent, **byte_diff(target, output),
    }


def probe_lookup(model_id: str, copy_protocol: str | None) -> dict:
    cfg = r2.model_config(model_id)
    return {x["probe_key"]: x for x in r2.probe_rows(cfg["scan_tokenizer_id"], copy_protocol or "fewshot_copy_v1")}


def score_row(raw: dict, probe: dict) -> dict:
    typ = raw["probe_type"]
    base = {
        "completion_key": raw["completion_key"], "model_id": raw["model_id"], "probe_key": raw["probe_key"], "probe_type": typ,
        "copy_protocol": raw.get("copy_protocol"), "batch_id": raw["batch_id"], "raw_text": raw["raw_text"], "output_token_count": raw["output_token_count"], "finish_reason_inferred": raw["finish_reason_inferred"],
        "hf_repo": raw["hf_repo"], "model_revision": raw["model_revision"], "upstream_repo": raw["upstream_repo"], "upstream_revision": raw["upstream_revision"],
        "scan_tokenizer_id": raw["scan_tokenizer_id"], "dtype": raw["dtype"], "quantization_class": raw.get("quantization_class"), "decode_params": raw["decode_params"],
        "host_python_version": raw["host_python_version"], "host_unicodedata_version": raw["host_unicodedata_version"],
    }
    for key in ("cohort", "copy_role", "item_id", "item_type", "era", "fragmentation_group", "pair_id", "scan_tokenizer_id"):
        if key in probe:
            base[key] = probe[key]
    if typ == "copy":
        extraction = extract_copy_round2(raw["raw_text"])
        target = probe["expected_text"]
        if extraction["format_failure"]:
            taxonomy = {}
            exact = False
            content_failure = None
        else:
            taxonomy = copy_taxonomy_round2(target, extraction["extracted_text"])
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


def synthetic_probe(target: str) -> dict:
    return {
        "probe_key": "copy:synthetic-mutation", "probe_type": "copy", "item_id": "synthetic-mutation",
        "target": target, "expected_text": target, "copy_role": "ascii_control", "cohort": "ASCII", "item_type": "synthetic",
    }


def synthetic_raw(text: str) -> dict:
    return {
        "completion_key": "mutation-negative-control", "model_id": "tinyllama-1.1b-chat-v1.0-bf16", "probe_key": "copy:synthetic-mutation", "probe_type": "copy",
        "copy_protocol": "fewshot_copy_v1", "batch_id": "mutation-negative-control", "raw_text": text, "output_token_count": 0, "finish_reason_inferred": "synthetic",
        "hf_repo": "synthetic", "model_revision": "synthetic", "upstream_repo": "synthetic", "upstream_revision": "synthetic",
        "scan_tokenizer_id": "tinyllama-1.1b-chat", "dtype": "synthetic", "quantization_class": "synthetic", "decode_params": {}, "host_python_version": "synthetic", "host_unicodedata_version": "synthetic",
    }


def extractor_mutation_checks() -> dict:
    target = "A😀\u200d♂️X"
    probe = synthetic_probe(target)

    def score(text):
        return score_row(synthetic_raw(text), probe)

    exact = score(f"<OUT>{target}</OUT>")
    byte_mutated = score(f"<OUT>{target[:-1]}Y</OUT>")
    missing_zwj = score(f"<OUT>{target.replace(chr(0x200d), '')}</OUT>")
    missing_vs16 = score(f"<OUT>{target.replace(chr(0xfe0f), '')}</OUT>")
    newline_padding = score(f"<OUT>\n{target}\n</OUT>")
    nfc_target = "é"
    nfc_probe = synthetic_probe(nfc_target)
    nfd_mutated = score_row(synthetic_raw("<OUT>e\u0301</OUT>"), nfc_probe)
    missing_wrapper = score(target)
    duplicate_wrapper = score(f"<OUT>{target}</OUT><OUT>{target}</OUT>")
    echoed_prompt = score(f"Input: <IN>{target}</IN>\nOutput: <OUT>{target}</OUT>")
    prefix_text = score(f"answer: <OUT>{target}</OUT>")
    suffix_text = score(f"<OUT>{target}</OUT> done")
    truncation = score(f"<OUT>{target[:-1]}</OUT>")
    checks = {
        "byte_exact_copy_accepted": exact["task_exact"] is True and exact["format_failure"] is False,
        "single_codepoint_substitution_rejected": byte_mutated["task_exact"] is False and byte_mutated["format_failure"] is False,
        "missing_zwj_rejected_and_classified": missing_zwj["task_exact"] is False and missing_zwj["missing_zwj"] is True,
        "missing_vs16_rejected_and_classified": missing_vs16["task_exact"] is False and missing_vs16["missing_vs16"] is True,
        "newline_padding_rejected_and_classified": newline_padding["task_exact"] is False and newline_padding["newline_padding"] is True,
        "nfc_nfd_mutation_rejected_and_classified": nfd_mutated["task_exact"] is False and nfd_mutated["normalization_only_mismatch"] is True,
        "missing_wrapper_rejected": missing_wrapper["format_failure"] is True,
        "duplicate_wrapper_rejected": duplicate_wrapper["format_failure"] is True,
        "echoed_prompt_rejected": echoed_prompt["format_failure"] is True,
        "non whitespace_prefix_rejected": prefix_text["format_failure"] is True,
        "non whitespace_suffix_rejected": suffix_text["format_failure"] is True,
        "truncation_rejected_and_classified": truncation["task_exact"] is False and truncation["truncation"] is True,
    }
    return {
        "checks": checks,
        "extractor_mutation_checks_pass": all(checks.values()),
        "strict_extraction_rule": "exactly one <OUT> and one </OUT>; no non-whitespace outside wrappers; extracted bytes are compared without stripping, normalization, or repair",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--scored-path", type=Path, default=SCORED_PATH)
    parser.add_argument("--mutation-path", type=Path, default=ROUND2 / "round2_extractor_mutation_check.json")
    args = parser.parse_args()
    raw = [x for x in r2.read_jsonl(args.raw_path) if x.get("status") == "ok"]
    keys = [x["completion_key"] for x in raw]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate completion keys in append-only raw file; manual correction batch required")
    lookups = {}
    for row in raw:
        key = (row["model_id"], row.get("copy_protocol") or "fewshot_copy_v1")
        lookups.setdefault(key, probe_lookup(row["model_id"], row.get("copy_protocol")))
    missing = [x["completion_key"] for x in raw if x["probe_key"] not in lookups[(x["model_id"], x.get("copy_protocol") or "fewshot_copy_v1")]]
    if missing:
        raise RuntimeError(f"raw rows without probe definitions: {missing[:5]}")
    scored = [score_row(x, lookups[(x["model_id"], x.get("copy_protocol") or "fewshot_copy_v1")][x["probe_key"]]) for x in raw]
    r2.write_jsonl(args.scored_path, scored)
    mutation = extractor_mutation_checks()
    args.mutation_path.parent.mkdir(parents=True, exist_ok=True)
    args.mutation_path.write_text(json.dumps(mutation, indent=2, ensure_ascii=False))
    print(json.dumps({"raw_rows": len(raw), "scored_rows": len(scored), "output": str(args.scored_path), "extractor_mutation_checks_pass": mutation["extractor_mutation_checks_pass"], "host_unicodedata_version": unicodedata.unidata_version, "normalization_unicodedata_version": unicodedata2.unidata_version}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
