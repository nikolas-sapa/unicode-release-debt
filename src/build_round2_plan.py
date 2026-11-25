from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import round2_common as r2

ROUND1_PLAN = bp.BEHAVIORAL / "probe_plan.jsonl"
ROUND2_PLAN = r2.ROUND2 / "probe_plan_round2.jsonl"
ROUND2_AUDIT = r2.ROUND2 / "probe_identity_audit.json"

ROUND1_REFERENCE_BY_SCAN = {
    "tinyllama-1.1b-chat": "tinyllama-1.1b-chat-v1.0-bf16",
    "qwen2.5-0.5b-instruct": "qwen2.5-1.5b-instruct-bf16",
    "qwen3-0.6b": "qwen3-1.7b-bf16",
    "gemma3-1b-it-mirror": "gemma3-1b-it-bf16-mirror",
}


def build_plan(copy_protocol: str):
    round1 = r2.read_jsonl(ROUND1_PLAN)
    audits = []
    all_rows = []
    for cfg in r2.model_configs():
        rows = r2.probe_rows(cfg["scan_tokenizer_id"], copy_protocol)
        reference_model = ROUND1_REFERENCE_BY_SCAN[cfg["scan_tokenizer_id"]]
        reference = [x for x in round1 if x["model_id"] == reference_model]
        round1_identity = [r2.item_identity(x) for x in reference]
        round2_identity = [r2.item_identity(x) for x in rows]
        identical = round1_identity == round2_identity
        audits.append({
            "model_id": cfg["model_id"], "scan_tokenizer_id": cfg["scan_tokenizer_id"], "round1_reference_model": reference_model,
            "round1_rows": len(reference), "round2_rows": len(rows), "item_identity_exactly_matches_round1": identical,
            "round1_item_identity_sha256": r2.sha_text(r2.canon(round1_identity)),
            "round2_item_identity_sha256": r2.sha_text(r2.canon(round2_identity)),
        })
        if not identical:
            raise RuntimeError(f"round-2 item identity mismatch for {cfg['model_id']} against {reference_model}")
        for row in rows:
            all_rows.append({"model_id": cfg["model_id"], **{k: v for k, v in row.items() if k not in ("item", "source_probe")}})
    r2.ROUND2.mkdir(parents=True, exist_ok=True)
    r2.write_jsonl(ROUND2_PLAN, all_rows)
    audit = {
        "copy_protocol": copy_protocol,
        "probe_rows": len(all_rows),
        "rows_per_model": 290,
        "models": len(r2.model_configs()),
        "expected_rows": 290 * len(r2.model_configs()),
        "audits": audits,
        "all_item_identities_match_round1": all(x["item_identity_exactly_matches_round1"] for x in audits),
        "exemplar_overlap_audit": r2.exemplar_overlap_audit(),
        "copy_protocol_templates": r2.protocol_template_metadata(),
    }
    ROUND2_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-protocol", default="fewshot_copy_v1", choices=sorted(r2.COPY_PROTOCOLS))
    args = parser.parse_args()
    audit = build_plan(args.copy_protocol)
    print(json.dumps({"output": str(ROUND2_PLAN), "rows": audit["probe_rows"], "all_item_identities_match_round1": audit["all_item_identities_match_round1"]}, indent=2))


if __name__ == "__main__":
    main()
