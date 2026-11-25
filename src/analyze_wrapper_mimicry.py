"""Post-hoc analysis: copy outputs wrapped in wrong-direction <IN> tags whose inner content is byte-exact.

Strict protocol treats these as format failures (they are, and gates account them as such).
This artifact quantifies how much apparent copy degradation is wrapper-direction mimicry
rather than byte-copy failure, because the distinction changes the scientific interpretation.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2

RESULTS = r2.ROOT / "results" / "behavioral_round2"


def main():
    import score_behavioral_round2 as s
    out = {}
    for model in r2.MODELS:
        mid = model["model_id"]
        lookup = s.probe_lookup(mid, "fewshot_copy_v5_multiturn_cueless")
        rows, seen = [], set()
        if not (RESULTS / "round2_raw.jsonl").exists():
            continue
        for line in open(RESULTS / "round2_raw.jsonl"):
            x = json.loads(line)
            if x["model_id"] == mid and x["probe_type"] == "copy" and x["probe_key"] not in seen:
                seen.add(x["probe_key"])
                rows.append((x, s.score_row(x, lookup[x["probe_key"]])))
        if not rows:
            continue
        mimic_rows = []
        for x, sc in rows:
            raw = x["raw_text"]
            if sc["task_exact"]:
                continue
            if raw.startswith("<IN>") and raw.endswith("</IN>") and raw.count("<IN>") == 1 and raw.count("</IN>") == 1:
                inner = raw[len("<IN>"):-len("</IN>")]
                target = lookup[x["probe_key"]]["target"]
                mimic_rows.append({
                    "probe_key": x["probe_key"], "cohort": lookup[x["probe_key"]].get("cohort"),
                    "inner_byte_exact": inner == target,
                })
        cohort_counts = Counter(m["cohort"] for m in mimic_rows)
        out[mid] = {
            "copy_rows": len(rows),
            "in_wrapper_mimicry_failures": len(mimic_rows),
            "mimicry_inner_byte_exact": sum(1 for m in mimic_rows if m["inner_byte_exact"]),
            "mimicry_inner_byte_diff": sum(1 for m in mimic_rows if not m["inner_byte_exact"]),
            "mimicry_by_cohort": dict(cohort_counts),
            "mimicry_probe_keys": [m["probe_key"] for m in mimic_rows],
        }
    path = RESULTS / "round2_wrapper_mimicry_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: v[kk] for kk in ("in_wrapper_mimicry_failures", "mimicry_inner_byte_exact")} for k, v in out.items()}, indent=2))
    print("output:", path)


if __name__ == "__main__":
    main()
