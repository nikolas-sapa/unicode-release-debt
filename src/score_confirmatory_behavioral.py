from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf
from score_behavioral import parse_count, parse_decomp
from score_behavioral_round2 import byte_diff, copy_taxonomy_round2, edit_counts, extract_copy_round2, levenshtein

RAW_PATH = cf.CONFIRMATORY / "confirmatory_raw.jsonl"
SCORED_PATH = cf.CONFIRMATORY / "confirmatory_scored.jsonl"
INT_RE = re.compile(r"[+-]?\d+")


def wrapper_mimicry(raw_text: str) -> bool:
    return any(x in raw_text for x in ("<IN>", "</IN>", "Input:", "Output:", "Copy the input"))


def copy_near_miss_class(row: dict) -> str:
    if row["format_failure"]:
        return "format_failure_wrapper_mimicry" if wrapper_mimicry(row["raw_text"]) else "format_failure_other"
    if row["task_exact"]:
        return "exact_copy"
    if row.get("normalization_only_mismatch"):
        return "normalization_only_mismatch"
    if row.get("decomposed_rgi_sequence") or row.get("missing_zwj"):
        return "decomposed_rgi_sequence_or_missing_zwj"
    if row.get("missing_vs16"):
        return "missing_vs16"
    if row.get("truncation") and row.get("common_prefix_bytes", 0) >= 0.8 * row.get("target_utf8_bytes", 1):
        return "truncation_near_miss"
    if row.get("truncation"):
        return "truncation_other"
    if row.get("extra_text_after_exact_target"):
        return "extra_text_after_exact_target"
    if row.get("replacement_character_emitted"):
        return "replacement_character_emitted"
    if row.get("codepoint_levenshtein_distance") == 1:
        return "single_codepoint_edit"
    if row.get("codepoint_levenshtein_distance") == 2:
        return "two_codepoint_edits"
    if row.get("reordering"):
        return "reordering"
    if row.get("codepoint_levenshtein_distance", 0) > 2:
        return "multi_codepoint_corruption"
    return "other_unclassified_mismatch"


def copy_is_near_miss(row: dict) -> bool:
    if row["format_failure"] or row["task_exact"]:
        return False
    return row["copy_near_miss_class"] in {
        "normalization_only_mismatch",
        "decomposed_rgi_sequence_or_missing_zwj",
        "missing_vs16",
        "truncation_near_miss",
        "extra_text_after_exact_target",
        "single_codepoint_edit",
        "two_codepoint_edits",
    }


def scalar_error_class(parsed_tokens, expected_tokens, raw_text: str, target: str, format_failure: bool) -> str:
    if format_failure:
        return "format_failure"
    if len(parsed_tokens) != len(expected_tokens):
        return "wrong_number_of_scalar_tokens"
    wrong = sum(a != b for a, b in zip(parsed_tokens, expected_tokens))
    if wrong == 1:
        return "correct_count_one_wrong_scalar"
    if wrong > 1:
        return "correct_count_multiple_wrong_scalars"
    if wrong == 0:
        return "exact"
    if "\ufffd" in raw_text or "U+FFFD" in parsed_tokens:
        return "placeholder_or_replacement_character_output"
    nfc = cf.scalar_text(unicodedata2.normalize("NFC", target))
    nfd = cf.scalar_text(unicodedata2.normalize("NFD", target))
    if parsed_tokens != expected_tokens and (parsed_tokens == nfc.split() or parsed_tokens == nfd.split()):
        return "normalization_substitution"
    return "other"


def probe_lookup() -> dict:
    return {(x["carrier_id"], x["probe_key"]): x for x in cf.read_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl")}


def score_row(raw: dict, probe: dict) -> dict:
    typ = raw["probe_type"]
    base = {
        "completion_key": raw["completion_key"], "carrier_id": raw["carrier_id"], "model_id": raw["carrier_id"], "probe_key": raw["probe_key"], "probe_type": typ,
        "probe_scope": raw["probe_scope"], "batch_id": raw["batch_id"], "raw_text": raw["raw_text"], "output_token_count": raw["output_token_count"],
        "finish_reason_inferred": raw["finish_reason_inferred"], "hf_repo": raw["hf_repo"], "model_revision": raw["model_revision"],
        "upstream_repo": raw["upstream_repo"], "upstream_revision": raw["upstream_revision"], "dtype": raw["dtype"], "quantization_class": raw.get("quantization_class"),
        "decode_params": raw["decode_params"], "host_python_version": raw["host_python_version"], "host_unicodedata_version": raw["host_unicodedata_version"],
        "prompt_sha256": raw["prompt_sha256"], "user_content_sha256": raw["user_content_sha256"],
    }
    for key in (
        "era", "item_type", "requested_item_type", "item_type_substituted", "inventory_cell", "fragmentation_level", "expected_count", "slot_index", "match_pair_id",
        "tokens_per_cluster", "whole_probe_token_count", "utf8_bytes", "utf8_bytes_per_cluster", "codepoint_count", "codepoints_per_cluster", "mean_sequence_complexity",
        "supplementary_cluster_count", "supplementary_cluster_fraction", "emoji_frequency_row_mean", "emoji_frequency_any_missing", "candidate_id", "control_slot",
    ):
        if key in probe:
            base[key] = probe[key]
    if typ == "copy":
        extraction = extract_copy_round2(raw["raw_text"])
        target = probe["target"]
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
        base["copy_near_miss_class"] = copy_near_miss_class(base)
        base["copy_near_miss"] = copy_is_near_miss(base)
    elif typ == "count":
        parsed = parse_count(raw["raw_text"])
        exact = parsed["parsed_answer"] == probe["expected_count"] if not parsed["format_failure"] else False
        signed = parsed["parsed_answer"] - probe["expected_count"] if not parsed["format_failure"] else None
        base.update({
            "target": probe["target"], "expected_count": probe["expected_count"], **parsed, "extracted_answer": parsed["parsed_answer"],
            "task_exact": exact, "count_exact": exact, "content_failure_given_format": (not exact) if not parsed["format_failure"] else None,
            "total_failure": not exact, "count_signed_error": signed, "count_near_miss": abs(signed) == 1 if signed is not None else False,
        })
    elif typ == "scalar_listing":
        parsed = parse_decomp(raw["raw_text"])
        expected_tokens = probe["expected_text"].split()
        exact = parsed["parsed_answer"] == expected_tokens if not parsed["format_failure"] else False
        tokens = parsed["parsed_answer"] or []
        wrong_count = sum(a != b for a, b in zip(tokens, expected_tokens)) if len(tokens) == len(expected_tokens) else None
        err = scalar_error_class(tokens, expected_tokens, raw["raw_text"], probe["target"], parsed["format_failure"])
        base.update({
            "target": probe["target"], "expected_text": probe["expected_text"], "expected_codepoints": expected_tokens, **parsed, "extracted_answer": parsed["parsed_answer"],
            "task_exact": exact, "scalar_exact": exact, "content_failure_given_format": (not exact) if not parsed["format_failure"] else None,
            "total_failure": not exact, "scalar_error_class": err,
            "scalar_wrong_token_count": wrong_count, "scalar_near_miss": err == "correct_count_one_wrong_scalar",
        })
    else:
        raise ValueError(typ)
    return base


def synthetic_probe(probe_type: str, target: str, expected_count=None) -> dict:
    row = {"probe_key": f"{probe_type}:synthetic", "probe_type": probe_type, "probe_scope": "ascii", "carrier_id": "synthetic", "target": target}
    if probe_type == "copy":
        row["expected_text"] = target
    elif probe_type == "count":
        row["expected_count"] = expected_count if expected_count is not None else cf.grapheme_count(target)
    else:
        row["expected_text"] = cf.scalar_text(target)
    return row


def synthetic_raw(probe_type: str, text: str) -> dict:
    return {
        "completion_key": "synthetic", "carrier_id": "synthetic", "probe_key": f"{probe_type}:synthetic", "probe_type": probe_type, "probe_scope": "ascii",
        "batch_id": "synthetic", "raw_text": text, "output_token_count": 0, "finish_reason_inferred": "synthetic", "hf_repo": "synthetic", "model_revision": "synthetic",
        "upstream_repo": "synthetic", "upstream_revision": "synthetic", "dtype": "synthetic", "quantization_class": "synthetic", "decode_params": {},
        "host_python_version": "synthetic", "host_unicodedata_version": "synthetic", "prompt_sha256": "synthetic", "user_content_sha256": "synthetic",
    }


def extractor_mutation_checks() -> dict:
    target = "A😀\u200d♂️X"
    copy_probe = synthetic_probe("copy", target)
    score_copy = lambda text: score_row(synthetic_raw("copy", text), copy_probe)
    exact = score_copy(f"<OUT>{target}</OUT>")
    duplicate = score_copy(f"<OUT>{target}</OUT><OUT>{target}</OUT>")
    outside = score_copy(f"<OUT>{target}</OUT> done")
    missing_zwj = score_copy(f"<OUT>{target.replace(chr(0x200d), '')}</OUT>")
    long_target = "abcdefghij"
    long_probe = synthetic_probe("copy", long_target)
    trunc85 = score_row(synthetic_raw("copy", f"<OUT>{long_target[:8] + 'x'}</OUT>"), long_probe)
    trunc85 = score_row(synthetic_raw("copy", f"<OUT>{long_target[:9][:-1]}</OUT>"), long_probe)
    trunc85 = score_row(synthetic_raw("copy", f"<OUT>{long_target[:8]}</OUT>"), long_probe)
    trunc50 = score_row(synthetic_raw("copy", f"<OUT>{long_target[:5]}</OUT>"), long_probe)
    count_probe = synthetic_probe("count", "abc", 3)
    count3 = score_row(synthetic_raw("count", "3"), count_probe)
    count3dot = score_row(synthetic_raw("count", "3."), count_probe)
    scalar_probe = synthetic_probe("scalar_listing", "Ab")
    scalar_ok = score_row(synthetic_raw("scalar_listing", "U+0041 U+0062"), scalar_probe)
    scalar_lower = score_row(synthetic_raw("scalar_listing", "u+0041 u+0062"), scalar_probe)
    scalar_one_wrong = score_row(synthetic_raw("scalar_listing", "U+0041 U+0063"), scalar_probe)
    checks = {
        "correct_wrapped_copy_passes": exact["task_exact"] is True,
        "two_wrappers_fail": duplicate["format_failure"] is True,
        "non_whitespace_outside_wrapper_fails": outside["format_failure"] is True,
        "integer_3_passes": count3["task_exact"] is True,
        "integer_3dot_fails": count3dot["format_failure"] is True,
        "uppercase_scalar_list_passes": scalar_ok["task_exact"] is True,
        "lowercase_scalar_list_fails": scalar_lower["format_failure"] is True,
        "missing_zwj_precedence_class": missing_zwj["copy_near_miss_class"] == "decomposed_rgi_sequence_or_missing_zwj",
        "truncation_80_percent_is_near_miss": trunc85["copy_near_miss_class"] == "truncation_near_miss" and trunc85["copy_near_miss"] is True,
        "truncation_50_percent_not_near_miss": trunc50["copy_near_miss_class"] == "truncation_other" and trunc50["copy_near_miss"] is False,
        "one_token_scalar_error_is_near_miss": scalar_one_wrong["scalar_near_miss"] is True and scalar_one_wrong["scalar_error_class"] == "correct_count_one_wrong_scalar",
    }
    return {"checks": checks, "extractor_mutation_checks_pass": all(checks.values()), "details": {"trunc85_class": trunc85["copy_near_miss_class"], "missing_zwj_class": missing_zwj["copy_near_miss_class"]}}


def completion_audit(raw: list[dict], carriers: list[str] | None = None) -> dict:
    plan = cf.read_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl")
    errors = []
    carriers = carriers or [x["carrier_id"] for x in cf.carrier_configs(include_descriptive=True)]
    keys = [x["completion_key"] for x in raw]
    if len(keys) != len(set(keys)):
        errors.append({"check": "duplicate_completion_keys", "count": len(keys) - len(set(keys))})
    required_fields = ["raw_text", "finish_reason_inferred", "output_token_count", "prompt_sha256", "model_revision", "dtype", "batch_id", "batch_elapsed_seconds", "decode_params", "completion_key"]
    for carrier in carriers:
        expected = {(x["probe_key"]) for x in plan if x["carrier_id"] == carrier}
        got_rows = [x for x in raw if x["carrier_id"] == carrier]
        got = {x["probe_key"] for x in got_rows}
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing:
            errors.append({"carrier_id": carrier, "check": "missing_probe_keys", "count": len(missing), "examples": missing[:5]})
        if extra:
            errors.append({"carrier_id": carrier, "check": "extra_probe_keys", "count": len(extra), "examples": extra[:5]})
        for row in got_rows:
            absent = [x for x in required_fields if x not in row]
            if absent:
                errors.append({"carrier_id": carrier, "check": "required_raw_fields_missing", "probe_key": row.get("probe_key"), "fields": absent})
                break
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "carriers_checked": carriers, "raw_rows": len(raw), "raw_sha256_before_scoring": cf.sha_file(RAW_PATH) if RAW_PATH.exists() else None, "errors": errors, "completion_audit_pass": not errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--carrier", action="append")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    raw = [x for x in cf.read_jsonl(RAW_PATH) if x.get("status") == "ok"]
    audit = completion_audit(raw, args.carrier)
    if args.allow_incomplete:
        audit["completion_audit_pass"] = not [x for x in audit["errors"] if x["check"] not in ("missing_probe_keys",)]
    cf.write_json(cf.CONFIRMATORY / "raw_completion_audit.json", audit)
    if args.audit_only:
        print(json.dumps({"completion_audit_pass": audit["completion_audit_pass"], "errors": len(audit["errors"])}, indent=2))
        if not audit["completion_audit_pass"]:
            raise SystemExit(1)
        return
    lookup = probe_lookup()
    missing = [x["completion_key"] for x in raw if (x["carrier_id"], x["probe_key"]) not in lookup]
    if missing:
        raise RuntimeError(f"raw rows without probe definitions: {missing[:5]}")
    scored = [score_row(x, lookup[(x["carrier_id"], x["probe_key"])]) for x in raw]
    cf.write_jsonl(SCORED_PATH, scored)
    mutation = extractor_mutation_checks()
    cf.write_json(cf.CONFIRMATORY / "extractor_mutation_checks.json", mutation)
    scoring_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "raw_rows": len(raw), "scored_rows": len(scored),
        "scored_sha256": cf.sha_file(SCORED_PATH), "task_counts": dict(Counter(f"{x['carrier_id']}/{x['probe_type']}/{x['probe_scope']}" for x in scored)),
        "extractor_mutation_checks_pass": mutation["extractor_mutation_checks_pass"],
    }
    cf.write_json(cf.CONFIRMATORY / "scoring_audit.json", scoring_audit)
    print(json.dumps({"raw_rows": len(raw), "scored_rows": len(scored), "completion_audit_pass": audit["completion_audit_pass"], "extractor_mutation_checks_pass": mutation["extractor_mutation_checks_pass"]}, indent=2))
    if not mutation["extractor_mutation_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
