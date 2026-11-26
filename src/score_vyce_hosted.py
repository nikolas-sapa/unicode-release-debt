from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import score_behavioral_round2 as round2_scoring
import vyce_common as vc
from score_behavioral import parse_compare, parse_count, parse_decomp

RAW_PATH = vc.VYCE / "vyce_raw.jsonl"
SCORED_PATH = vc.VYCE / "vyce_scored.jsonl"
PLAN_PATH = vc.VYCE / "vyce_probe_plan.jsonl"
VALIDATION_PATH = vc.VYCE / "vyce_scoring_validation.json"
ASCII_GATE_PATH = vc.VYCE / "vyce_ascii_gate.json"


def score_vyce_row(raw: dict, probe: dict) -> dict:
    typ = raw["probe_type"]
    base = {
        "completion_key": raw["completion_key"], "model_id": raw["model_id"], "probe_key": raw["probe_key"], "probe_type": typ,
        "copy_protocol": raw.get("copy_protocol"), "batch_id": raw["batch_id"], "raw_text": raw["raw_text"],
        "output_token_count": raw["output_token_count"], "input_token_count": raw["input_token_count"],
        "finish_reason": raw.get("finish_reason"), "finish_reason_inferred": raw["finish_reason_inferred"],
        "endpoint": raw["endpoint"], "api_model_returned": raw.get("api_model_returned"), "request_sha256": raw["request_sha256"],
        "raw_response_body_sha256": raw["raw_response_body_sha256"], "decode_params": raw["decode_params"],
        "host_python_version": raw["host_python_version"], "host_unicodedata_version": raw["host_unicodedata_version"],
        "relay_artifacts": raw.get("relay_artifacts", []), "refusal": raw.get("refusal"), "reasoning_field_present": raw.get("reasoning_field_present"),
        "estimated_cost_usd": raw.get("estimated_cost_usd"), "cap_cost_usd": raw.get("cap_cost_usd"),
    }
    for key in ("cohort", "copy_role", "item_id", "item_type", "era", "fragmentation_group", "pair_id", "vyce_selection_stratum"):
        if key in probe:
            base[key] = probe[key]
    if typ == "copy":
        extraction = round2_scoring.extract_copy_round2(raw["raw_text"])
        target = probe["expected_text"]
        if extraction["format_failure"]:
            taxonomy = {}
            exact = False
            content_failure = None
        else:
            taxonomy = round2_scoring.copy_taxonomy_round2(target, extraction["extracted_text"])
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


def mutate_copy_response(scored_row: dict) -> dict:
    if not scored_row.get("copy_exact") or not scored_row.get("extracted_text"):
        return {"ran": False, "reason": "no byte-exact copy response available"}
    target = scored_row["expected_text"]
    extracted = scored_row["extracted_text"]
    if len(extracted) < 2:
        mutated = extracted + "X"
    else:
        replacement = "X" if extracted[-1] != "X" else "Y"
        mutated = extracted[:-1] + replacement
    probe = {"expected_text": target, "probe_type": "copy"}
    raw = {"raw_text": f"<OUT>{mutated}</OUT>"}
    extraction = round2_scoring.extract_copy_round2(raw["raw_text"])
    exact = extraction["extracted_text"].encode() == target.encode() if not extraction["format_failure"] else False
    return {
        "ran": True,
        "source_completion_key": scored_row["completion_key"],
        "original_copy_exact": scored_row["copy_exact"],
        "mutated_extracted_text": mutated,
        "mutated_task_exact": exact,
        "mutation_detected": exact is False,
    }


def corrupted_negative_control() -> dict:
    target = "A😀\u200d♂️X"
    corrupted = target.replace("\u200d", "")
    extraction = round2_scoring.extract_copy_round2(f"<OUT>{corrupted}</OUT>")
    taxonomy = round2_scoring.copy_taxonomy_round2(target, extraction["extracted_text"]) if not extraction["format_failure"] else {}
    exact = extraction["extracted_text"].encode() == target.encode() if not extraction["format_failure"] else False
    return {
        "target": target,
        "corrupted_response": f"<OUT>{corrupted}</OUT>",
        "corrupted_response_exact": exact,
        "missing_zwj_classified": bool(taxonomy.get("missing_zwj")),
        "negative_control_pass": exact is False and bool(taxonomy.get("missing_zwj")),
    }


def ascii_gates(scored: list[dict]) -> dict:
    models = {}
    for model in sorted({x["model_id"] for x in scored}):
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
        correct = sum(bool(x["task_exact"]) for x in rows)
        format_failures = sum(bool(x["format_failure"]) for x in rows)
        models[model] = {
            "n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None,
            "format_failures": format_failures,
            "pass": len(rows) == 24 and correct >= 23,
            "gate_rule": "all 24 frozen ASCII controls present and at least 23/24 byte-exact",
        }
    return {"generated_utc": vc.utcnow(), "models": models, "adaptation": "round-2 chat messages passed directly to OpenAI-compatible chat API; no system prompt and no template rewrite"}


def score_file(raw_path: Path = RAW_PATH, scored_path: Path = SCORED_PATH, validation_path: Path = VALIDATION_PATH) -> dict:
    raw_all = vc.read_jsonl(raw_path)
    raw = [x for x in raw_all if x.get("status") == "ok"]
    keys = [x["completion_key"] for x in raw]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate completion keys in append-only vyce raw file; manual correction batch required")
    plan = {x["probe_key"]: x for x in vc.read_jsonl(PLAN_PATH)}
    missing = [x["completion_key"] for x in raw if x["probe_key"] not in plan]
    if missing:
        raise RuntimeError(f"raw rows without vyce probe definitions: {missing[:5]}")
    scored = [score_vyce_row(x, plan[x["probe_key"]]) for x in raw]
    vc.write_jsonl(scored_path, scored)
    extractor = round2_scoring.extractor_mutation_checks()
    first_exact = next((x for x in scored if x.get("copy_exact")), None)
    response_mutation = mutate_copy_response(first_exact) if first_exact else {"ran": False, "reason": "no scored byte-exact copy response yet"}
    negative = corrupted_negative_control()
    validation = {
        "generated_utc": vc.utcnow(),
        "extractor_mutation_checks_pass": extractor["extractor_mutation_checks_pass"],
        "extractor_mutation_detail": extractor,
        "scored_response_mutation": response_mutation,
        "corrupted_negative_control": negative,
        "scoring_validation_pass": extractor["extractor_mutation_checks_pass"] and (not response_mutation.get("ran") or response_mutation.get("mutation_detected")) and negative["negative_control_pass"],
        "host_unicodedata_version": unicodedata.unidata_version,
        "normalization_unicodedata_version": unicodedata2.unidata_version,
    }
    vc.write_json(validation_path, validation)
    gates = ascii_gates(scored)
    vc.write_json(ASCII_GATE_PATH, gates)
    return {
        "raw_rows_total": len(raw_all), "raw_rows_scored": len(raw), "error_or_incomplete_rows_unscored": len(raw_all) - len(raw),
        "scored_rows": len(scored), "scored_path": str(scored_path), "ascii_gates": gates,
        "scoring_validation_pass": validation["scoring_validation_pass"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--scored-path", type=Path, default=SCORED_PATH)
    parser.add_argument("--validation-path", type=Path, default=VALIDATION_PATH)
    args = parser.parse_args()
    print(json.dumps(score_file(args.raw_path, args.scored_path, args.validation_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
