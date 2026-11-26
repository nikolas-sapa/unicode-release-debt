from __future__ import annotations

import argparse
import json
import platform
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

RAW_PATH = vc.VYCE / "vyce_raw.jsonl"
RUNTIME_PATH = vc.VYCE / "vyce_runtime.jsonl"
PLAN_PATH = vc.VYCE / "vyce_probe_plan.jsonl"
ASCII_GATE_PATH = vc.VYCE / "vyce_ascii_gate.json"
AVAILABILITY_SUMMARY_PATH = vc.VYCE / "availability_summary.json"
COPY_PROTOCOL = "fewshot_copy_v5_multiturn_cueless"
LOGICAL_BATCH_SIZE = 8


def completion_key(model: str, row: dict) -> str:
    decode = {"temperature": 0.0, "max_tokens": vc.MAX_TOKENS, "seed": None, "response_format": None}
    obj = {"endpoint": vc.ENDPOINT, "model": model, "probe_key": row["probe_key"], "messages": row["messages"], "decode": decode, "plan_item_sha256": row["vyce_round2_item_identity_sha256"]}
    return vc.sha_text(vc.canon(obj))


def load_plan() -> list[dict]:
    rows = vc.read_jsonl(PLAN_PATH)
    if len(rows) != 104:
        raise RuntimeError(f"vyce probe plan has {len(rows)} rows, expected 104; run build_vyce_plan.py")
    return rows


def require_availability(model: str):
    if not AVAILABILITY_SUMMARY_PATH.exists():
        raise RuntimeError("availability summary missing; run probe_vyce_availability.py before inference")
    summary = json.loads(AVAILABILITY_SUMMARY_PATH.read_text())
    if model not in summary.get("working_models", []):
        raise RuntimeError(f"{model} is not recorded as a working availability probe; refusing spend on full battery")


def require_ascii_gate(model: str):
    if not ASCII_GATE_PATH.exists():
        raise RuntimeError("ASCII gate file missing; score the ASCII-only phase before full inference")
    gates = json.loads(ASCII_GATE_PATH.read_text())
    row = gates.get("models", {}).get(model)
    if not row or not row.get("pass"):
        raise RuntimeError(f"{model} has not passed the frozen ASCII adaptation gate; refusing Unicode/structural spend")


def select_rows(rows: list[dict], ascii_only: bool) -> list[dict]:
    if ascii_only:
        return [x for x in rows if x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
    return rows


def batch_chunks(rows: list[dict], n: int):
    for i in range(0, len(rows), n):
        yield i // n, rows[i:i + n]


def batch_projection(model: str, rows: list[dict]) -> dict:
    parts = [vc.projected_cap_cost(model, x["messages"], vc.MAX_TOKENS) for x in rows]
    return {
        "input_tokens_upper_bound": sum(x["input_tokens_upper_bound"] for x in parts),
        "output_tokens_upper_bound": sum(x["output_tokens_upper_bound"] for x in parts),
        "projected_cap_cost_usd": sum(x["projected_cap_cost_usd"] for x in parts),
        "cost_bound_method": "sum of per-call UTF-8 byte/overhead input upper bounds plus max_tokens output upper bounds",
    }


def ensure_batch_budget(model: str, rows: list[dict], cap_usd: float, ledger_path: Path, purpose: str):
    projection = batch_projection(model, rows)
    totals = vc.ledger_totals(ledger_path)
    if totals["cap_cost_usd"] + projection["projected_cap_cost_usd"] > cap_usd:
        raise vc.budget_abort("projected conservative cost for logical batch would exceed hard cap", model, projection, cap_usd, ledger_path, purpose)
    return projection


def runtime_event(event: str, model: str, extra: dict) -> dict:
    return {
        "event": event, "utc": vc.utcnow(), "model_id": model,
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "arm_label": "EXPLORATORY hosted scale/diversity extension; not confirmatory",
        **extra,
    }


def run_model(model: str, ascii_only: bool, cap_usd: float, ledger_path: Path, raw_path: Path, runtime_path: Path, limit: int | None = None, dry_run: bool = False) -> dict:
    require_availability(model)
    if not ascii_only:
        require_ascii_gate(model)
    rows = select_rows(load_plan(), ascii_only)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no probes selected")
    keys = {x["completion_key"]: x for x in vc.read_jsonl(raw_path) if x.get("status") == "ok"}
    phase = "ascii_gate" if ascii_only else "full_after_ascii_gate"
    selected = []
    skipped = 0
    for row in rows:
        key = completion_key(model, row)
        if key in keys:
            skipped += 1
        else:
            selected.append((row, key))
    projection = batch_projection(model, [x for x, _ in selected]) if selected else {"input_tokens_upper_bound": 0, "output_tokens_upper_bound": 0, "projected_cap_cost_usd": 0.0}
    if dry_run:
        return {"model": model, "phase": phase, "planned_rows": len(rows), "already_complete": skipped, "remaining_rows": len(selected), "projection": projection, "ledger_totals": vc.ledger_totals(ledger_path), "dry_run": True}
    invocation = runtime_event("run_started", model, {
        "phase": phase, "selected_probe_rows": len(rows), "remaining_rows": len(selected), "already_complete_rows": skipped,
        "decode_params": {"temperature": 0.0, "max_tokens": vc.MAX_TOKENS, "seed": None, "response_format": None},
        "copy_protocol": COPY_PROTOCOL, "endpoint": vc.ENDPOINT, "cost_cap_usd": cap_usd,
        "projection_before_run": projection, "ledger_totals_before_run": vc.ledger_totals(ledger_path),
    })
    vc.append_jsonl(runtime_path, [invocation])
    generated = 0
    errors = 0
    for batch_index, batch in batch_chunks(selected, LOGICAL_BATCH_SIZE):
        batch_rows = [x for x, _ in batch]
        batch_projection_row = ensure_batch_budget(model, batch_rows, cap_usd, ledger_path, phase)
        batch_id = f"vyce-{model}-{phase}-batch{batch_index:03d}"
        vc.append_jsonl(runtime_path, [runtime_event("logical_batch_started", model, {"batch_id": batch_id, "rows": len(batch_rows), "projection": batch_projection_row})])
        for row, key in batch:
            result = vc.post_chat_completion(model, row["messages"], max_tokens=vc.MAX_TOKENS, temperature=0.0, purpose=phase, cap_usd=cap_usd, ledger_path=ledger_path, timeout=180)
            valid_content = isinstance(result["content"], str)
            status = "ok" if result["http_status"] == 200 and valid_content else "error"
            artifacts = list(result["relay_artifacts"])
            if result["http_status"] == 200 and not valid_content:
                artifacts.append("content_non_string_or_missing")
            message_obj = result["message_obj"] or {}
            reasoning_present = any(k in message_obj for k in ("reasoning", "reasoning_content", "thinking"))
            if reasoning_present:
                artifacts.append("reasoning_field_present")
            raw_row = {
                "status": status,
                "completion_key": key,
                "batch_id": batch_id,
                "model_id": model,
                "probe_key": row["probe_key"],
                "probe_type": row["probe_type"],
                "copy_protocol": row.get("copy_protocol"),
                "vyce_reference_model_id": row["vyce_reference_model_id"],
                "vyce_reference_scan_tokenizer_id": row["vyce_reference_scan_tokenizer_id"],
                "endpoint": vc.ENDPOINT,
                "api_model_returned": result["parsed"].get("model") if isinstance(result["parsed"], dict) else None,
                "request_sha256": result["request_sha256"],
                "messages_sha256": vc.sha_text(vc.canon(row["messages"])),
                "user_content_sha256": row["user_content_sha256"],
                "decode_params": invocation["decode_params"],
                "http_status": result["http_status"],
                "network_error": result["network_error"],
                "request_id": (result["parsed"] or {}).get("request_id") if isinstance(result["parsed"], dict) else None,
                "response_headers": result["headers"],
                "raw_text": result["content"] if valid_content else None,
                "raw_response_body_text": result["body_text"],
                "raw_response_body_base64": result["body_base64"],
                "raw_response_body_sha256": result["body_sha256"],
                "output_token_count": result["tokens"]["output_tokens"],
                "input_token_count": result["tokens"]["input_tokens"],
                "total_token_count": result["tokens"]["total_tokens"],
                "usage": result["tokens"]["usage"],
                "usage_missing": result["tokens"]["usage_missing"],
                "finish_reason": result["finish_reason"],
                "finish_reason_inferred": "length" if result["finish_reason"] == "length" else ("stop" if result["finish_reason"] == "stop" else result["finish_reason"]),
                "refusal": result["refusal"],
                "reasoning_field_present": reasoning_present,
                "relay_artifacts": sorted(set(artifacts)),
                "estimated_cost_usd": result["costs"]["estimated_cost_usd"],
                "cap_cost_usd": result["costs"]["cap_cost_usd"],
                "batch_started_utc": result["started_utc"],
                "batch_finished_utc": result["finished_utc"],
                "latency_seconds": result["latency_seconds"],
                "host_python_version": platform.python_version(),
                "host_unicodedata_version": unicodedata.unidata_version,
            }
            vc.append_jsonl(raw_path, [raw_row])
            if status == "ok":
                generated += 1
            else:
                errors += 1
        vc.append_jsonl(runtime_path, [runtime_event("logical_batch_finished", model, {"batch_id": batch_id, "rows": len(batch_rows), "ledger_totals": vc.ledger_totals(ledger_path)})])
    vc.append_jsonl(runtime_path, [runtime_event("run_finished", model, {"phase": phase, "generated_ok_rows_this_invocation": generated, "error_rows_this_invocation": errors, "ledger_totals": vc.ledger_totals(ledger_path)})])
    return {"model": model, "phase": phase, "generated_ok_rows": generated, "error_rows": errors, "skipped_existing": skipped, "remaining_before_run": len(selected), "ledger_totals": vc.ledger_totals(ledger_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=vc.CANDIDATE_MODELS)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--ascii-only", action="store_true")
    phase.add_argument("--full", action="store_true")
    parser.add_argument("--cost-cap", type=float, default=vc.HARD_CAP_USD)
    parser.add_argument("--ledger-path", type=Path, default=vc.LEDGER_PATH)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--runtime-path", type=Path, default=RUNTIME_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_model(args.model, args.ascii_only, args.cost_cap, args.ledger_path, args.raw_path, args.runtime_path, args.limit, args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
