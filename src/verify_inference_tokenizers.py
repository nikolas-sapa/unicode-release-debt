from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf


def main():
    freeze = json.loads((cf.CONFIRMATORY / "tokenizer_freeze.json").read_text())
    frozen = {}
    for row in freeze["carriers"]:
        frozen[row["carrier_id"]] = {x["file"]: x["sha256"] for x in row["files"] if x.get("sha256")}
    runtime = [json.loads(x) for x in open(cf.CONFIRMATORY / "confirmatory_runtime.jsonl")]
    checks = {}
    errors = []
    for cfg in cf.carrier_configs(include_descriptive=True):
        cid = cfg["carrier_id"]
        events = [x for x in runtime if x.get("event") == "model_loaded" and x.get("carrier_id") == cid]
        if not events:
            checks[cid] = {"checked": False, "error": "no model_loaded events"}
            continue
        event = events[-1]
        manifest = {x["file"]: x.get("sha256") for x in event["file_manifest"]}
        prep = event.get("preparation", {})
        comparisons = []
        for filename, expected in frozen[cid].items():
            if prep.get("overlay_used") and filename in {x["file"] for x in prep.get("overlay_files", [])}:
                actual = next((x["sha256"] for x in prep["overlay_files"] if x["file"] == filename), None)
                source = "canonical_overlay"
            else:
                actual = manifest.get(filename)
                source = "model_snapshot"
            ok = actual == expected
            comparisons.append({"file": filename, "source": source, "expected_sha256": expected, "actual_sha256": actual, "match": ok})
            if not ok:
                errors.append({"carrier_id": cid, "file": filename, "expected": expected, "actual": actual})
        checks[cid] = {"checked": True, "model_loaded_events": len(events), "overlay_used": prep.get("overlay_used", False), "comparisons": comparisons, "all_match": all(x["match"] for x in comparisons)}
    obj = {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "errors": errors, "inference_tokenizer_verification_pass": bool(checks) and not errors and all(x.get("checked") and x.get("all_match") for x in checks.values())}
    cf.write_json(cf.CONFIRMATORY / "inference_tokenizer_verification.json", obj)
    print(json.dumps({"pass": obj["inference_tokenizer_verification_pass"], "errors": errors}, indent=2))
    if not obj["inference_tokenizer_verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
