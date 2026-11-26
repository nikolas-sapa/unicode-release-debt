from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf
import score_confirmatory_behavioral as scorer


def main():
    raw = [json.loads(x) for x in open(cf.CONFIRMATORY / "confirmatory_raw.jsonl")]
    lookup = scorer.probe_lookup()
    exact_row = None
    for row in raw:
        if row["probe_type"] != "copy":
            continue
        probe = lookup[(row["carrier_id"], row["probe_key"])]
        if scorer.score_row(row, probe)["task_exact"]:
            exact_row = row
            break
    if exact_row is None:
        result = {"created_utc": datetime.now(timezone.utc).isoformat(), "mutation_test_pass": False, "error": "no exact copy row available to corrupt"}
    else:
        probe = lookup[(exact_row["carrier_id"], exact_row["probe_key"])]
        before = scorer.score_row(exact_row, probe)
        corrupted = dict(exact_row)
        text = corrupted["raw_text"]
        pos = text.find("<OUT>") + len("<OUT>")
        corrupted["raw_text"] = text[:pos] + "X" + text[pos + 1:]
        after = scorer.score_row(corrupted, probe)
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test": "copy one exact raw response in memory, corrupt one output byte, rescore; retained raw jsonl is never mutated",
            "completion_key": exact_row["completion_key"],
            "before_exact": before["task_exact"],
            "after_exact": after["task_exact"],
            "before_hash": cf.sha_obj(exact_row["raw_text"]),
            "after_hash": cf.sha_obj(corrupted["raw_text"]),
            "scoring_detected_corruption": before["task_exact"] is True and after["task_exact"] is False,
            "mutation_test_pass": before["task_exact"] is True and after["task_exact"] is False,
        }
    cf.write_json(cf.CONFIRMATORY / "raw_response_scoring_mutation_check.json", result)
    print(json.dumps(result, indent=2))
    if not result["mutation_test_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
