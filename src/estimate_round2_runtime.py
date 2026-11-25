from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2

OUT = r2.ROUND2 / "runtime_estimate.json"
PARAMETERS_B = {
    "qwen2.5-3b-instruct-bf16": 3.0,
    "qwen3-4b-instruct-2507-mlx8": 4.0,
    "qwen2.5-7b-instruct-mlx8": 7.0,
    "gemma3-4b-it-mlx8-mirror": 4.0,
    "tinyllama-1.1b-chat-v1.0-bf16": 1.1,
}


def main():
    runtime = r2.read_jsonl(r2.ROUND2 / "quant_ascii_runtime.jsonl")
    batches = [x for x in runtime if x.get("event") == "batch_completed" and x.get("model_id") == "quantcheck-qwen2.5-1.5b-instruct-bf16"]
    if not batches:
        raise SystemExit("quant ASCII runtime missing; run quantization sensitivity first")
    observed = sum(x["elapsed_seconds"] for x in batches) / sum(x["probe_count"] for x in batches)
    rows = []
    total = 0.0
    for cfg in sorted(r2.model_configs(), key=lambda x: x["priority"]):
        params = PARAMETERS_B[cfg["model_id"]]
        quant_factor = 0.85 if cfg["quantization_class"] == "mlx8" else 1.0
        estimated = observed * (params / 1.5) * quant_factor * 290 * 1.20
        conservative = estimated * 2.0
        total += conservative
        rows.append({
            "model_id": cfg["model_id"], "priority": cfg["priority"], "approx_parameters_b": params,
            "estimated_seconds_conservative": conservative, "estimated_minutes_conservative": conservative / 60,
        })
    ordered = sorted(rows, key=lambda x: x["priority"], reverse=True)
    kept = list(rows)
    trimmed = []
    while total > 6 * 3600 and ordered:
        drop = ordered.pop(0)
        kept = [x for x in kept if x["model_id"] != drop["model_id"]]
        trimmed.append({"model_id": drop["model_id"], "reason": "projected runtime exceeded six hours; trimmed from bottom of priority list"})
        total -= drop["estimated_seconds_conservative"]
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "observed Qwen2.5-1.5B bf16 ASCII few-shot seconds/probe scaled by approximate parameter count, adjusted for q8 memory bandwidth, multiplied by 290 probes, 1.20 structural-prompt overhead, and a 2x conservative safety factor",
        "observed_qwen15b_bf16_seconds_per_ascii_probe": observed,
        "per_model_estimates": rows,
        "projected_total_seconds": total,
        "projected_total_minutes": total / 60,
        "threshold_minutes": 360,
        "model_count": len(kept),
        "kept_model_ids": [x["model_id"] for x in sorted(kept, key=lambda x: x["priority"])],
        "trimmed_models": trimmed,
        "trimming_required": bool(trimmed),
    }
    r2.ROUND2.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"projected_total_minutes": out["projected_total_minutes"], "trimming_required": out["trimming_required"], "model_count": out["model_count"]}, indent=2))


if __name__ == "__main__":
    main()
