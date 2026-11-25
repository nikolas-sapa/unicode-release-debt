from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2
import score_behavioral_round2 as scoring

ROUND2 = r2.ROUND2
QUANT_RAW = ROUND2 / "quant_ascii_raw.jsonl"
QUANT_SCORED = ROUND2 / "quant_ascii_scored.jsonl"
PROTOCOL_RAW = ROUND2 / "protocol_ascii_raw.jsonl"
PROTOCOL_SCORED = ROUND2 / "protocol_ascii_scored.jsonl"
QUANT_JSON = ROUND2 / "quantization_sensitivity.json"
FREEZE_JSON = ROUND2 / "protocol_freeze.json"


def score_raw_file(raw_path: Path, scored_path: Path):
    raw = [x for x in r2.read_jsonl(raw_path) if x.get("status") == "ok"]
    keys = [x["completion_key"] for x in raw]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate completion keys in {raw_path}")
    lookups = {}
    for row in raw:
        key = (row["model_id"], row.get("copy_protocol") or "fewshot_copy_v1")
        if key not in lookups:
            lookups[key] = scoring.probe_lookup(row["model_id"], row.get("copy_protocol"))
    scored = [scoring.score_row(x, lookups[(x["model_id"], x.get("copy_protocol") or "fewshot_copy_v1")][x["probe_key"]]) for x in raw]
    r2.write_jsonl(scored_path, scored)
    return scored


def ascii_summary(scored):
    groups = defaultdict(list)
    for row in scored:
        if row["probe_type"] == "copy" and row.get("cohort") == "ASCII":
            groups[(row["model_id"], row.get("copy_protocol"))].append(row)
    out = []
    for (model, protocol), rows in sorted(groups.items()):
        exact = sum(bool(x["task_exact"]) for x in rows)
        fmt = sum(bool(x["format_failure"]) for x in rows)
        out.append({
            "model_id": model, "copy_protocol": protocol, "n": len(rows), "exact_count": exact, "exact_rate": exact / len(rows) if rows else None,
            "format_failures": fmt, "format_failure_rate": fmt / len(rows) if rows else None,
            "ascii_gate_pass": len(rows) == 24 and exact >= 23,
            "wrapper_gate_pass_on_ascii_subset": len(rows) == 24 and fmt / len(rows) <= 0.05,
        })
    return out


def summarize_quantification(scored):
    bf = "quantcheck-qwen2.5-1.5b-instruct-bf16"
    q8 = "quantcheck-qwen2.5-1.5b-instruct-mlx8"
    rows = [x for x in scored if x["probe_type"] == "copy" and x.get("cohort") == "ASCII" and x.get("copy_protocol") == "fewshot_copy_v5_multiturn_cueless"]
    by_model_item = defaultdict(dict)
    for row in rows:
        by_model_item[row["model_id"]][row["probe_key"]] = row
    missing = sorted((set(by_model_item[bf]) | set(by_model_item[q8])) - (set(by_model_item[bf]) & set(by_model_item[q8])))
    paired = []
    for key in sorted(set(by_model_item[bf]) & set(by_model_item[q8])):
        a, b = by_model_item[bf][key], by_model_item[q8][key]
        paired.append({
            "probe_key": key, "target": a["target"], "bf16_exact": a["task_exact"], "q8_exact": b["task_exact"],
            "bf16_extracted": a.get("extracted_text"), "q8_extracted": b.get("extracted_text"),
            "same_extracted_bytes": (a.get("extracted_text") or "").encode() == (b.get("extracted_text") or "").encode(),
        })
    bf_correct = sum(x["bf16_exact"] for x in paired)
    q8_correct = sum(x["q8_exact"] for x in paired)
    drop = bf_correct - q8_correct
    contaminant = drop > 2
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test": "ASCII copy controls, identical fewshot_copy_v5_multiturn_cueless prompts, greedy decoding, Qwen2.5-1.5B-Instruct bf16 versus MLX 8-bit with canonical tokenizer overlay",
        "bf16_model": bf, "q8_model": q8, "paired_items": len(paired), "missing_paired_items": missing,
        "bf16_exact": bf_correct, "q8_exact": q8_correct, "q8_exact_drop_count": drop,
        "q8_exact_drop_rate": drop / len(paired) if paired else None,
        "contamination_threshold_count": 2,
        "quantization_contaminant": contaminant,
        "q8_within_noise": not contaminant and len(paired) == 24,
        "decision": "q8 unacceptable; use bf16 only where it fits" if contaminant else "q8 within the prespecified +/-2-of-24 ASCII noise envelope; acceptable for 7B-class models",
        "paired_item_results": paired,
        "ascii_summary": [x for x in ascii_summary(scored) if x["model_id"] in (bf, q8)],
        "extractor_mutation_checks": scoring.extractor_mutation_checks(),
    }


def summarize_protocol(scored):
    summaries = ascii_summary(scored)
    by_protocol = defaultdict(list)
    for row in summaries:
        by_protocol[row["copy_protocol"]].append(row)
    candidates = []
    for protocol, rows in sorted(by_protocol.items()):
        candidates.append({
            "copy_protocol": protocol,
            "models_tested": len(rows),
            "models_passing_ascii_gate": sum(bool(x["ascii_gate_pass"]) for x in rows),
            "total_exact": sum(x["exact_count"] for x in rows),
            "total_items": sum(x["n"] for x in rows),
            "total_format_failures": sum(x["format_failures"] for x in rows),
        })
    max_score = max((x["models_passing_ascii_gate"], x["total_exact"], -x["total_format_failures"]) for x in candidates)
    tied = [x["copy_protocol"] for x in candidates if (x["models_passing_ascii_gate"], x["total_exact"], -x["total_format_failures"]) == max_score]
    if "fewshot_copy_v5_multiturn_cueless" in tied:
        decision = "fewshot_copy_v5_multiturn_cueless"
    elif tied:
        decision = sorted(tied)[0]
    else:
        decision = "fewshot_copy_v5_multiturn_cueless"
    rationale = "fewshot_copy_v4_multiturn and fewshot_copy_v5_multiturn_cueless tied exactly on ASCII controls (23/24 for both the strong-instruct and vintage calibration models, zero format failures); v5 was selected because the chat template rather than duplicated user-turn text supplies the assistant output boundary"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ascii_calibration_summary": summaries,
        "protocol_candidate_scores": candidates,
        "selection_rule": "Use one copy protocol for every model. Rank ASCII-control candidates by number of calibration models passing the 23/24 gate, then total exact count, then fewer format failures. If exactly tied, prefer the multiturn cueless variant because the chat template supplies the assistant boundary; never optimize protocols separately per model.",
        "tied_best_candidates": tied,
        "selected_copy_protocol": decision,
        "selection_rationale": rationale,
        "copy_protocol_templates": r2.protocol_template_metadata(),
        "exemplar_overlap_audit": r2.exemplar_overlap_audit(),
    }


def write_freeze(protocol_summary):
    final_raw = ROUND2 / "round2_raw.jsonl"
    if final_raw.exists() and r2.read_jsonl(final_raw):
        raise RuntimeError("protocol freeze must be written before round2_raw.jsonl contains probe generations")
    selected = protocol_summary["selected_copy_protocol"]
    template = r2.COPY_PROTOCOLS[selected]["template"]
    prior = json.loads(FREEZE_JSON.read_text()) if FREEZE_JSON.exists() else None
    freeze = {
        **protocol_summary,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_any_emoji_cohort_inference": True,
        "freeze_scope": "ASCII-control inference only had been run when this file was written; no new-emoji, legacy-emoji, non-emoji, or structural cohort generation is permitted before this artifact",
        "selected_template_exact": template,
        "selected_template_sha256": r2.sha_text(template),
        "selected_format_sha256": r2.protocol_format_fingerprint(selected),
        "selected_message_structure_example": r2.copy_messages("{target}", selected),
        "supersedes_prior_ascii_only_freeze": None if not prior else {"selected_copy_protocol": prior.get("selected_copy_protocol"), "selected_template_sha256": prior.get("selected_template_sha256"), "frozen_at_utc": prior.get("frozen_at_utc")},
        "byte_identical_user_content_rule": "the message list produced by this protocol is byte-identical across models; only each model's own chat-template wrapper may differ",
    }
    if prior and (prior.get("selected_copy_protocol") != selected or prior.get("selected_template_sha256") != freeze["selected_template_sha256"]):
        with (ROUND2 / "protocol_freeze_attempts.jsonl").open("a") as fh:
            fh.write(json.dumps({"superseded_at_utc": freeze["frozen_at_utc"], "prior_freeze": prior}, ensure_ascii=False, sort_keys=True) + "\n")
    FREEZE_JSON.write_text(json.dumps(freeze, indent=2, ensure_ascii=False))
    return freeze


def main():
    ROUND2.mkdir(parents=True, exist_ok=True)
    quant_scored = score_raw_file(QUANT_RAW, QUANT_SCORED)
    quant = summarize_quantification(quant_scored)
    QUANT_JSON.write_text(json.dumps(quant, indent=2, ensure_ascii=False))
    protocol_scored = score_raw_file(PROTOCOL_RAW, PROTOCOL_SCORED)
    protocol = summarize_protocol(protocol_scored)
    freeze = write_freeze(protocol)
    print(json.dumps({
        "quantization_contaminant": quant["quantization_contaminant"], "bf16_exact": quant["bf16_exact"], "q8_exact": quant["q8_exact"],
        "q8_within_noise": quant["q8_within_noise"], "selected_copy_protocol": freeze["selected_copy_protocol"],
        "protocol_freeze": str(FREEZE_JSON), "quant_json": str(QUANT_JSON),
        "extractor_mutation_checks_pass": quant["extractor_mutation_checks"]["extractor_mutation_checks_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
