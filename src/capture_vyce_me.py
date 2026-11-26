from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

SNAPSHOTS = vc.VYCE / "v1_me_snapshots.jsonl"
FIRST = vc.VYCE / "v1_me_sanitized.json"
DELTA = vc.VYCE / "v1_me_usage_delta.json"
URL = "https://vyceai.com/v1/me"


def sanitize(parsed):
    # Public-artifact allowlist. In particular, do not publish the key name,
    # balance, lifetime spend, timestamps, or per-model lifetime usage counters.
    if not isinstance(parsed, dict):
        return {}
    return {k: parsed.get(k) for k in ["enabled", "tier", "verified", "rateLimit"] if k in parsed}


def capture() -> dict:
    key = vc.load_key()
    req = urllib.request.Request(URL, headers={"Authorization": "Bearer " + key, "User-Agent": vc.USER_AGENT})
    status, body, err = None, b"", None
    tic = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except Exception as e:
        err = type(e).__name__ + ": " + str(e)[:200]
    latency = time.perf_counter() - tic
    body, secret_redacted = vc.redacted_body(body, key)
    _, parsed, _ = vc.parse_response_body(body)
    snap = {
        "utc": vc.utcnow(), "status": status, "network_error": err,
        "response_body_sha256": vc.sha_bytes(body), "sanitized": sanitize(parsed),
        "secret_redacted": secret_redacted, "latency_seconds": latency,
    }
    vc.append_jsonl(SNAPSHOTS, [snap])
    if not FIRST.exists():
        vc.write_json(FIRST, snap)
    tot = vc.ledger_totals(vc.LEDGER_PATH)
    vc.append_ledger({
        "event": "api_call", "utc": snap["utc"], "purpose": "v1_me_metadata", "model": None,
        "endpoint": URL, "http_status": status, "network_error": err, "request_sha256": None,
        "response_body_sha256": snap["response_body_sha256"], "request_id": None,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "usage_source": "non-completion key metadata endpoint; zero token cost assumed",
        "usage_missing": True, "usage_estimation_notes": [],
        "price_input_per_million_usd": None, "price_output_per_million_usd": None,
        "pricing_source": "non-completion metadata endpoint", "estimated_cost_usd": 0.0,
        "cap_input_per_million_usd": vc.CAP_INPUT_PER_MILLION, "cap_output_per_million_usd": vc.CAP_OUTPUT_PER_MILLION,
        "cap_cost_usd": 0.0, "cumulative_estimated_cost_before_usd": tot["estimated_cost_usd"],
        "cumulative_cap_cost_before_usd": tot["cap_cost_usd"],
        "cumulative_estimated_cost_after_usd": tot["estimated_cost_usd"],
        "cumulative_cap_cost_after_usd": tot["cap_cost_usd"], "cap_usd": vc.HARD_CAP_USD,
        "max_tokens": 0, "temperature": None, "latency_seconds": latency, "relay_artifacts": [],
    }, vc.LEDGER_PATH)
    return snap


def write_delta(latest: dict) -> dict:
    first = json.loads(FIRST.read_text()) if FIRST.exists() else latest
    old, new = first.get("sanitized", {}), latest.get("sanitized", {})
    diff = {}
    for m in vc.CANDIDATE_MODELS + ["wave"]:
        a = (old.get("modelUsage") or {}).get(m, {"cost": 0, "inputTokens": 0, "outputTokens": 0})
        b = (new.get("modelUsage") or {}).get(m, {"cost": 0, "inputTokens": 0, "outputTokens": 0})
        diff[m] = {k: b.get(k, 0) - a.get(k, 0) for k in ["cost", "inputTokens", "outputTokens"]}
    out = {"generated_utc": vc.utcnow(), "baseline_utc": first.get("utc"), "latest_utc": latest.get("utc"), "delta": diff}
    vc.write_json(DELTA, out)
    return out


def main():
    snap = capture()
    delta = write_delta(snap)
    s = snap.get("sanitized", {})
    print(json.dumps({"status": snap["status"], "verified": s.get("verified"), "tier": s.get("tier"), "target_usage_delta": delta["delta"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
