from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

RAW = cf.CONFIRMATORY / "confirmatory_raw.jsonl"
BASELINE = cf.CONFIRMATORY / "resume_baseline_before_gemmaresume.json"
OUT = cf.CONFIRMATORY / "resume_no_regeneration_audit.json"
GEMMA_CHECKPOINT_COMMIT = "7129d98"
QWEN_CHECKPOINT_COMMIT = "3d5ec04"


def git_file(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=cf.ROOT)


def main():
    baseline = json.loads(BASELINE.read_text())
    raw_bytes = RAW.read_bytes()
    rows = [json.loads(x) for x in raw_bytes.splitlines() if x.strip()]
    plan = [json.loads(x) for x in (cf.CONFIRMATORY / "probe_plan.jsonl").read_text().splitlines() if x.strip()]
    counts = Counter(r["carrier_id"] for r in rows if r.get("status") == "ok")
    plan_counts = Counter(r["carrier_id"] for r in plan)
    completion_keys = [r["completion_key"] for r in rows]
    carrier_probe_keys = [(r["carrier_id"], r["probe_key"]) for r in rows]
    increments = {c: counts.get(c, 0) - baseline["counts"].get(c, 0) for c in sorted(plan_counts)}
    expected_increments = {
        "tinyllama-1.1b-chat-v1.0-bf16": 0,
        "qwen2.5-3b-instruct-bf16": 0,
        "llama32-3b-instruct-mlx8": 0,
        "gemma3-4b-it-mlx8-mirror": 140,
        "qwen3-4b-instruct-2507-mlx8": 500,
        "mistral7b-instruct-v03-mlx8": 460,
    }
    gemma_raw = git_file(GEMMA_CHECKPOINT_COMMIT, "results/confirmatory/confirmatory_raw.jsonl")
    qwen_raw = git_file(QWEN_CHECKPOINT_COMMIT, "results/confirmatory/confirmatory_raw.jsonl")
    gemma_lines = gemma_raw.splitlines(keepends=True)
    baseline_prefix = b"".join(gemma_lines[:baseline["raw_rows"]])
    evidence = {
        "gemma_checkpoint_commit": GEMMA_CHECKPOINT_COMMIT,
        "qwen_checkpoint_commit": QWEN_CHECKPOINT_COMMIT,
        "baseline_prefix_hash_match": hashlib.sha256(baseline_prefix).hexdigest() == baseline["raw_sha256"],
        "gemma_checkpoint_rows": len(gemma_lines),
        "qwen_checkpoint_rows": len(qwen_raw.splitlines()),
        "current_startswith_gemma_checkpoint_raw": raw_bytes.startswith(gemma_raw),
        "current_startswith_qwen_checkpoint_raw": raw_bytes.startswith(qwen_raw),
    }
    final = {
        "raw_rows": len(rows),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "counts": dict(sorted(counts.items())),
        "plan_counts": dict(sorted(plan_counts.items())),
        "all_plan_rows_complete_exactly_once": counts == plan_counts and len(carrier_probe_keys) == len(set(carrier_probe_keys)),
        "unique_completion_keys": len(set(completion_keys)),
        "duplicate_completion_keys": len(completion_keys) - len(set(completion_keys)),
        "duplicate_carrier_probe_keys": len(carrier_probe_keys) - len(set(carrier_probe_keys)),
    }
    checks = {
        "final_row_count_is_2880": len(rows) == 2880,
        "counts_match_probe_plan_exactly": counts == plan_counts,
        "no_completion_key_duplicates": final["duplicate_completion_keys"] == 0,
        "no_carrier_probe_key_duplicates": final["duplicate_carrier_probe_keys"] == 0,
        "increments_match_expected_resume_rows": increments == expected_increments,
        "no_carrier_count_regressed": all(counts.get(c, 0) >= baseline["counts"].get(c, 0) for c in plan_counts),
        "baseline_prefix_hash_match": evidence["baseline_prefix_hash_match"],
        "append_only_since_gemma_checkpoint": evidence["current_startswith_gemma_checkpoint_raw"],
        "append_only_since_qwen_checkpoint": evidence["current_startswith_qwen_checkpoint_raw"],
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "final": final,
        "increments": increments,
        "expected_increments": expected_increments,
        "append_only_evidence": evidence,
        "checks": checks,
        "no_regeneration_audit_pass": all(checks.values()),
        "interpretation": "row counts increased exactly by the missing 140+500+460 rows, completion keys are unique, and the retained raw file is byte-prefix append-only relative to committed checkpoints; no completed row was regenerated",
    }
    cf.write_json(OUT, result)
    print(json.dumps({"no_regeneration_audit_pass": result["no_regeneration_audit_pass"], "checks": checks, "increments": increments}, indent=2))
    if not result["no_regeneration_audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
