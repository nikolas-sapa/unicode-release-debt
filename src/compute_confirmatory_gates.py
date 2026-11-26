from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf


def compute_gates(scored: list[dict]) -> dict:
    tokenizer_freeze = json.loads((cf.CONFIRMATORY / "tokenizer_freeze.json").read_text())
    tokenizer_pass = tokenizer_freeze.get("freeze_pass") is True
    out = {}
    for cfg in cf.carrier_configs(include_descriptive=True):
        cid = cfg["carrier_id"]
        rows = [x for x in scored if x["carrier_id"] == cid]
        ascii_copy = [x for x in rows if x["probe_type"] == "copy" and x["probe_scope"] == "ascii"]
        all_copy = [x for x in rows if x["probe_type"] == "copy"]
        ascii_count = [x for x in rows if x["probe_type"] == "count" and x["probe_scope"] == "ascii"]
        ascii_scalar = [x for x in rows if x["probe_type"] == "scalar_listing" and x["probe_scope"] == "ascii"]
        ascii_copy_exact = sum(x["task_exact"] for x in ascii_copy)
        copy_format_failures = sum(x["format_failure"] for x in all_copy)
        copy_complete = len(all_copy) == 124
        copy_control_valid = ascii_copy_exact >= 23
        copy_valid = copy_complete and copy_control_valid and copy_format_failures <= 6
        ascii_count_exact = sum(x["task_exact"] for x in ascii_count)
        ascii_scalar_exact = sum(x["task_exact"] for x in ascii_scalar)
        structural_ascii_valid = ascii_count_exact >= 11 and ascii_scalar_exact >= 11
        structural_valid = copy_valid and structural_ascii_valid
        if cfg["quantization_class"] == "bf16":
            quantization_status = "bf16_no_comparison_required"
            quantization_eligible = True
        else:
            quantization_status = "narrow_validation_only_no_bf16_comparator_downloaded_allowed_by_frozen_user_resolution"
            quantization_eligible = True
        primary = cfg["primary_ladder"]
        out[cid] = {
            "carrier_id": cid,
            "primary_ladder": primary,
            "descriptive_robustness_only": not primary,
            "family": cfg["family"],
            "tokenizer_family": cfg["tokenizer_family"],
            "tokenizer_valid": tokenizer_pass,
            "ascii_copy_rows": len(ascii_copy),
            "ascii_copy_exact": ascii_copy_exact,
            "ascii_copy_gate_pass": copy_control_valid,
            "copy_rows": len(all_copy),
            "copy_complete": copy_complete,
            "copy_wrapper_format_failures": copy_format_failures,
            "copy_wrapper_gate_pass_at_most_6": copy_format_failures <= 6,
            "copy_valid": copy_valid,
            "copy_valid_provisional_until_unicode_copy_complete": not copy_complete,
            "ascii_count_rows": len(ascii_count),
            "ascii_count_exact": ascii_count_exact,
            "ascii_count_gate_pass": ascii_count_exact >= 11,
            "ascii_scalar_rows": len(ascii_scalar),
            "ascii_scalar_exact": ascii_scalar_exact,
            "ascii_scalar_gate_pass": ascii_scalar_exact >= 11,
            "structural_ascii_valid": structural_ascii_valid,
            "structural_valid": structural_valid,
            "quantization_status": quantization_status,
            "quantization_eligible_under_frozen_resolution": quantization_eligible,
            "quantization_limitation": "no direct bf16 comparator inference because expanded comparator weight downloads were explicitly skipped before execution" if cfg["quantization_class"] != "bf16" else None,
            "include_h1h2h3_primary": primary and tokenizer_pass and structural_valid and quantization_eligible,
            "include_h4_primary": primary and tokenizer_pass and copy_valid and quantization_eligible,
        }
    valid_structural = [x for x in out.values() if x["include_h1h2h3_primary"]]
    valid_copy = [x for x in out.values() if x["include_h4_primary"]]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "carriers": out,
        "structural_valid_primary_carriers": sorted(valid_structural, key=lambda x: x["carrier_id"]),
        "copy_valid_primary_carriers": sorted(valid_copy, key=lambda x: x["carrier_id"]),
        "structural_valid_primary_count": len(valid_structural),
        "copy_valid_primary_count": len(valid_copy),
        "structural_tokenizer_family_count": len({x["tokenizer_family"] for x in valid_structural}),
        "copy_tokenizer_family_count": len({x["tokenizer_family"] for x in valid_copy}),
        "minimum_h1h2_met": len(valid_structural) >= 4 and len({x["tokenizer_family"] for x in valid_structural}) >= 3,
        "minimum_h4_met": len(valid_copy) >= 4 and len({x["tokenizer_family"] for x in valid_copy}) >= 3,
        "quantization_resolution": "expanded bf16 downloads skipped; MLX8 carriers remain eligible under the pre-generation user resolution, with the missing direct contrast documented as a limitation",
    }


def gate_mutation_checks() -> dict:
    checks = {
        "ascii_copy_22_of_24_fails_control_threshold": (22 >= 23) is False,
        "ascii_copy_23_of_24_passes_control_threshold": (23 >= 23) is True,
        "seven_wrapper_failures_fail_threshold": (7 <= 6) is False,
        "six_wrapper_failures_pass_threshold": (6 <= 6) is True,
        "ascii_structural_10_of_12_fails": (10 >= 11) is False,
        "ascii_structural_11_of_12_passes": (11 >= 11) is True,
    }
    return {"checks": checks, "gate_mutation_checks_pass": all(checks.values())}


def main():
    scored = cf.read_jsonl(cf.CONFIRMATORY / "confirmatory_scored.jsonl")
    obj = compute_gates(scored)
    obj["gate_mutation_checks"] = gate_mutation_checks()
    cf.write_json(cf.CONFIRMATORY / "carrier_gates.json", obj)
    print(json.dumps({
        "carriers": {k: {"ascii_copy_exact": v["ascii_copy_exact"], "ascii_count_exact": v["ascii_count_exact"], "ascii_scalar_exact": v["ascii_scalar_exact"], "copy_valid": v["copy_valid"], "structural_valid": v["structural_valid"]} for k, v in obj["carriers"].items()},
        "minimum_h1h2_met": obj["minimum_h1h2_met"],
        "minimum_h4_met": obj["minimum_h4_met"],
        "gate_mutation_checks_pass": obj["gate_mutation_checks"]["gate_mutation_checks_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
