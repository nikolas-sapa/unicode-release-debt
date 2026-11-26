from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_vyce_hosted as runner
import score_behavioral_round2 as round2_scoring
import score_vyce_hosted as vyce_scoring
import vyce_common as vc

VALIDATION_PATH = vc.VYCE / "vyce_harness_validation.json"
VALIDATION_LEDGER = vc.VYCE / "cost_ledger_validation_tiny_cap.jsonl"


def tiny_cap_batch_check() -> dict:
    plan = vc.read_jsonl(runner.PLAN_PATH)
    batch = plan[:2]
    before = len(vc.read_jsonl(VALIDATION_LEDGER))
    try:
        runner.ensure_batch_budget("gpt-oss-120b", batch, 0.000000001, VALIDATION_LEDGER, "tiny_cap_validation_batch")
        aborted = False
        error_text = "no abort"
    except RuntimeError as exc:
        aborted = True
        error_text = str(exc)
    rows = vc.read_jsonl(VALIDATION_LEDGER)
    new_rows = rows[before:]
    return {
        "cap_usd": 0.000000001,
        "batch_rows": len(batch),
        "aborted_before_any_api_call": aborted,
        "error_text": error_text,
        "new_validation_ledger_rows": len(new_rows),
        "budget_abort_rows": sum(x.get("event") == "budget_abort" for x in new_rows),
        "api_call_rows": sum(x.get("event") == "api_call" for x in new_rows),
        "pass": aborted and new_rows and all(x.get("event") == "budget_abort" for x in new_rows),
    }


def ledger_integrity() -> dict:
    rows = vc.read_jsonl(vc.LEDGER_PATH)
    problems = []
    last_est = 0.0
    last_cap = 0.0
    api_calls = 0
    for i, row in enumerate(rows, 1):
        if "entry_sha256" not in row:
            problems.append(f"row {i}: missing entry hash")
        if row.get("event") != "api_call":
            continue
        api_calls += 1
        for key in ("purpose", "input_tokens", "output_tokens", "estimated_cost_usd", "cap_cost_usd", "cumulative_cap_cost_after_usd"):
            if key not in row:
                problems.append(f"row {i}: missing {key}")
        if (row.get("estimated_cost_usd") or 0) < 0 or (row.get("cap_cost_usd") or 0) < 0:
            problems.append(f"row {i}: negative cost")
        after_cap = row.get("cumulative_cap_cost_after_usd")
        after_est = row.get("cumulative_estimated_cost_after_usd")
        if after_cap is not None and after_cap + 1e-12 < last_cap:
            problems.append(f"row {i}: cap cumulative decreased")
        if after_est is not None and after_est + 1e-12 < last_est:
            problems.append(f"row {i}: estimated cumulative decreased")
        last_cap = after_cap if after_cap is not None else last_cap
        last_est = after_est if after_est is not None else last_est
    return {"rows": len(rows), "api_calls": api_calls, "problems": problems, "pass": not problems}


def scored_response_mutation_status() -> dict:
    validation_path = vc.VYCE / "vyce_scoring_validation.json"
    if not validation_path.exists():
        return {"ran": False, "pass": False, "reason": "vyce_scoring_validation.json missing"}
    data = json.loads(validation_path.read_text())
    row = data.get("scored_response_mutation", {})
    if not row.get("ran"):
        return {"ran": False, "pass": False, "reason": row.get("reason", "no scored response mutation row") }
    return {**row, "pass": bool(row.get("mutation_detected"))}


def plan_integrity() -> dict:
    manifest = json.loads((vc.VYCE / "vyce_selection_manifest.json").read_text())
    rows = vc.read_jsonl(vc.VYCE / "vyce_probe_plan.jsonl")
    identity = [{"probe_key": x["probe_key"], "user_content_sha256": x["user_content_sha256"], "round2_item_identity_sha256": x["vyce_round2_item_identity_sha256"]} for x in rows]
    digest = vc.sha_text(vc.canon(identity))
    return {
        "rows": len(rows),
        "expected_rows": 104,
        "sha256": digest,
        "manifest_sha256": manifest["probe_plan_sha256"],
        "pass": len(rows) == 104 and digest == manifest["probe_plan_sha256"] and manifest["round2_exact_reuse_verification"]["user_content_sha256_exact_for_all_selected_rows"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    tiny = tiny_cap_batch_check()
    ledger = ledger_integrity()
    extractor = round2_scoring.extractor_mutation_checks()
    negative = vyce_scoring.corrupted_negative_control()
    mutation = scored_response_mutation_status()
    plan = plan_integrity()
    validation = {
        "generated_utc": vc.utcnow(),
        "tiny_cap_batch_check": tiny,
        "production_ledger_integrity": ledger,
        "extractor_mutation_checks_pass": extractor["extractor_mutation_checks_pass"],
        "corrupted_negative_control": negative,
        "scored_response_mutation": mutation,
        "probe_plan_integrity": plan,
    }
    validation["harness_validation_pass"] = all([
        tiny["pass"], ledger["pass"], extractor["extractor_mutation_checks_pass"], negative["negative_control_pass"], plan["pass"], mutation.get("pass") is True,
    ])
    validation["scored_response_mutation_required_before_completion"] = not mutation.get("ran")
    vc.write_json(VALIDATION_PATH, validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
