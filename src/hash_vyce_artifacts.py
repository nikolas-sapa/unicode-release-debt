from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

OUT = vc.VYCE / "vyce_artifact_hashes.json"

ARTIFACTS = [
    "results/vyce_hosted/vyce_probe_plan.jsonl",
    "results/vyce_hosted/vyce_selection_manifest.json",
    "results/vyce_hosted/models_list_response.json",
    "results/vyce_hosted/availability_probe.jsonl",
    "results/vyce_hosted/availability_summary.json",
    "results/vyce_hosted/diagnostic_probe.jsonl",
    "results/vyce_hosted/v1_me_sanitized.json",
    "results/vyce_hosted/v1_me_snapshots.jsonl",
    "results/vyce_hosted/v1_me_usage_delta.json",
    "results/vyce_hosted/cost_ledger.jsonl",
    "results/vyce_hosted/cost_ledger_validation_tiny_cap.jsonl",
    "results/vyce_hosted/vyce_raw.jsonl",
    "results/vyce_hosted/vyce_runtime.jsonl",
    "results/vyce_hosted/vyce_scored.jsonl",
    "results/vyce_hosted/vyce_scoring_validation.json",
    "results/vyce_hosted/vyce_harness_validation.json",
    "results/vyce_hosted/vyce_copy_cohort_summary.csv",
    "results/vyce_hosted/vyce_copy_newest_old_summary.csv",
    "results/vyce_hosted/vyce_structural_summary.csv",
    "results/vyce_hosted/vyce_count_fragmentation_summary.csv",
    "results/vyce_hosted/vyce_count_tokenizer_covariates.csv",
    "results/vyce_hosted/vyce_analysis.json",
    "results/vyce_hosted/vyce_cost_summary.json",
    "results/vyce_hosted/vyce_relay_artifact_summary.json",
    "results/vyce_hosted/tokenizer_extension_manifest.json",
    "results/vyce_hosted/tokenizer_scan_extension.jsonl",
    "results/vyce_hosted/tokenizer_scan_extension.metadata.json",
    "results/vyce_hosted/tokenizer_inventory_extension.json",
    "results/VYCE_ARM_REPORT.md",
]

SCRIPTS = [
    "src/vyce_common.py",
    "src/build_vyce_plan.py",
    "src/probe_vyce_availability.py",
    "src/probe_vyce_diagnostics.py",
    "src/capture_vyce_me.py",
    "src/run_vyce_hosted.py",
    "src/score_vyce_hosted.py",
    "src/analyze_vyce_hosted.py",
    "src/vyce_tokenizer_scan.py",
    "src/validate_vyce_harness.py",
    "src/hash_vyce_artifacts.py",
]


def hash_row(path: Path, kind: str) -> dict:
    return {"kind": kind, "path": str(path), "exists": path.exists(), "sha256": vc.file_sha256(path) if path.exists() else None, "size": path.stat().st_size if path.exists() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows = [hash_row(vc.ROOT / x, "artifact") for x in ARTIFACTS] + [hash_row(vc.ROOT / x, "script") for x in SCRIPTS]
    payload = {
        "generated_utc": vc.utcnow(),
        "scope": "VYCE exploratory hosted-arm artifact and script hashes; excludes cost_ledger key material because no key material is ever written",
        "rows": rows,
        "all_listed_artifacts_exist": all(x["exists"] or x["path"].endswith(("vyce_raw.jsonl", "vyce_runtime.jsonl")) for x in rows if x["kind"] == "artifact"),
    }
    vc.write_json(OUT, payload)
    print(json.dumps({"output": str(OUT), "rows": len(rows), "missing": [x["path"] for x in rows if not x["exists"]]}, indent=2))


if __name__ == "__main__":
    main()
