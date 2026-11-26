from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

AVAILABILITY_PATH = vc.VYCE / "availability_probe.jsonl"
SUMMARY_PATH = vc.VYCE / "availability_summary.json"
MODELS_METADATA_PATH = vc.VYCE / "models_list_response.json"


def availability_key(model: str) -> str:
    return vc.sha_text("vyce-availability-v1\0" + model)


def probe_models(cap_usd: float, ledger_path: Path, retry_failed: bool = False) -> dict:
    existing = vc.read_jsonl(AVAILABILITY_PATH)
    done = {x["availability_key"]: x for x in existing}
    metadata_result = vc.get_models(cap_usd, ledger_path)
    metadata_row = {
        "event": "models_list",
        "utc": metadata_result["finished_utc"],
        "http_status": metadata_result["http_status"],
        "network_error": metadata_result["network_error"],
        "response_body_sha256": metadata_result["body_sha256"],
        "response_body_text": metadata_result["body_text"],
        "response_body_base64": metadata_result["body_base64"],
        "latency_seconds": metadata_result["latency_seconds"],
    }
    vc.write_json(MODELS_METADATA_PATH, metadata_row)
    rows = []
    for model in vc.CANDIDATE_MODELS:
        key = availability_key(model)
        if key in done and (done[key].get("responds") or not retry_failed):
            rows.append(done[key])
            continue
        messages = [{"role": "user", "content": "Reply with exactly: OK"}]
        result = vc.post_chat_completion(model, messages, max_tokens=16, temperature=0.0, purpose="availability_probe", cap_usd=cap_usd, ledger_path=ledger_path, timeout=90)
        row = {
            "event": "availability_probe",
            "availability_key": key,
            "utc": result["finished_utc"],
            "model": model,
            "http_status": result["http_status"],
            "network_error": result["network_error"],
            "request_sha256": result["request_sha256"],
            "response_body_sha256": result["body_sha256"],
            "response_body_text": result["body_text"],
            "response_body_base64": result["body_base64"],
            "content": result["content"],
            "finish_reason": result["finish_reason"],
            "usage": result["tokens"]["usage"],
            "input_tokens": result["tokens"]["input_tokens"],
            "output_tokens": result["tokens"]["output_tokens"],
            "estimated_cost_usd": result["costs"]["estimated_cost_usd"],
            "cap_cost_usd": result["costs"]["cap_cost_usd"],
            "latency_seconds": result["latency_seconds"],
            "relay_artifacts": result["relay_artifacts"],
            "returned_model": result["parsed"].get("model") if isinstance(result["parsed"], dict) else None,
        }
        row["responds"] = result["http_status"] == 200 and isinstance(result["content"], str) and len(result["content"]) > 0
        vc.append_jsonl(AVAILABILITY_PATH, [row])
        rows.append(row)
    listed = []
    if isinstance(metadata_result["parsed"], dict):
        listed = [x.get("id") for x in metadata_result["parsed"].get("data", []) if isinstance(x, dict)]
    summary = {
        "arm_label": "EXPLORATORY hosted scale/diversity extension; not confirmatory",
        "generated_utc": vc.utcnow(),
        "models_endpoint_status": metadata_result["http_status"],
        "models_listed": listed,
        "probe_rows": [{k: x.get(k) for k in ("model", "http_status", "content", "finish_reason", "responds", "returned_model", "input_tokens", "output_tokens", "cap_cost_usd", "relay_artifacts", "network_error")} for x in rows],
        "working_models": [x["model"] for x in rows if x.get("responds")],
        "nonworking_models": [x["model"] for x in rows if not x.get("responds")],
        "ledger_totals": vc.ledger_totals(ledger_path),
    }
    vc.write_json(SUMMARY_PATH, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-cap", type=float, default=vc.HARD_CAP_USD)
    parser.add_argument("--ledger-path", type=Path, default=vc.LEDGER_PATH)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    summary = probe_models(args.cost_cap, args.ledger_path, args.retry_failed)
    print(json.dumps({"working_models": summary["working_models"], "nonworking_models": summary["nonworking_models"], "ledger_totals": summary["ledger_totals"]}, indent=2))


if __name__ == "__main__":
    main()
