from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2
import score_behavioral_round2 as scoring
import ucd_build as ucd

RESULTS = ucd.RESULTS
ROUND2 = r2.ROUND2
SCORED_PATH = ROUND2 / "round2_scored.jsonl"
RUNTIME_PATH = ROUND2 / "round2_runtime.jsonl"
COHORT_ORDER = ["ASCII", "legacy emoji", "legacy non-emoji", "E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji", "new non-emoji"]
NEW_EMOJI_ORDER = ["E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji"]
SYSTEMATIC_CLASSES = ["missing_zwj", "missing_vs16", "normalization_only_mismatch", "replacement_character_emitted", "decomposed_rgi_sequence"]
TAXONOMY_CLASSES = SYSTEMATIC_CLASSES + ["nfc_nfd_variant", "truncation", "newline_padding", "extra_text_after_exact_target", "insertion", "deletion", "substitution", "reordering", "other_unclassified_mismatch"]


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rate(num, den):
    return num / den if den else None


def pct(x):
    return "NA" if x is None else f"{100 * x:.1f}%"


def pp(x):
    return "NA" if x is None else f"{x:.1f} pp"


def md_table(headers, rows) -> str:
    def cell(x):
        if isinstance(x, float):
            x = f"{x:.3f}"
        return str(x).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(x) for x in row) for row in rows]
    return "\n".join(lines)


def ascii_gate(rows):
    ascii_rows = [x for x in rows if x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
    correct = sum(bool(x["task_exact"]) for x in ascii_rows)
    return {"n": len(ascii_rows), "correct": correct, "accuracy": rate(correct, len(ascii_rows)), "pass": len(ascii_rows) == 24 and correct >= 23}


def wrapper_gate(rows):
    copy_rows = [x for x in rows if x["probe_type"] == "copy"]
    failures = sum(bool(x["format_failure"]) for x in copy_rows)
    return {"n": len(copy_rows), "format_failures": failures, "format_failure_rate": rate(failures, len(copy_rows)), "pass": len(copy_rows) == 182 and rate(failures, len(copy_rows)) <= 0.05}


def gate_mutation_checks():
    synthetic_ascii = [{"probe_type": "copy", "cohort": "ASCII", "task_exact": True, "format_failure": False} for _ in range(24)]
    mutated_ascii = [dict(x, task_exact=i >= 2) for i, x in enumerate(synthetic_ascii)]
    synthetic_wrapper = [{"probe_type": "copy", "cohort": "legacy emoji", "task_exact": True, "format_failure": False} for _ in range(182)]
    mutated_wrapper = [dict(x, format_failure=i < 10) for i, x in enumerate(synthetic_wrapper)]
    extractor = scoring.extractor_mutation_checks()
    return {
        "ascii_gate_mutation_detects_failure": (not ascii_gate(mutated_ascii)["pass"]) and ascii_gate(synthetic_ascii)["pass"],
        "wrapper_gate_mutation_detects_failure": (not wrapper_gate(mutated_wrapper)["pass"]) and wrapper_gate(synthetic_wrapper)["pass"],
        "extractor_mutation_checks_pass": extractor["extractor_mutation_checks_pass"],
        "extractor_mutation_detail": extractor,
        "ascii_mutation_detail": {"baseline": ascii_gate(synthetic_ascii), "mutated": ascii_gate(mutated_ascii)},
        "wrapper_mutation_detail": {"baseline": wrapper_gate(synthetic_wrapper), "mutated": wrapper_gate(mutated_wrapper)},
    }


def copy_scan_metrics():
    rows = r2.read_jsonl(RESULTS / "tokenizer_scan.jsonl")
    out = {}
    for row in rows:
        if row.get("probe_type") == "copy" and row.get("item_id"):
            out[(row["tokenizer_id"], row["item_id"])] = row
    return out


def summarize_copy(scored, scanmap, model_ids):
    groups = defaultdict(list)
    for row in scored:
        if row["probe_type"] == "copy":
            groups[(row["model_id"], row["cohort"])].append(row)
    summaries = []
    for model in model_ids:
        scan_tid = r2.model_config(model)["scan_tokenizer_id"]
        for cohort in COHORT_ORDER:
            rows = groups.get((model, cohort), [])
            if not rows:
                continue
            scans = [scanmap.get((scan_tid, x["item_id"])) for x in rows]
            scans = [x for x in scans if x]
            exact = sum(bool(x["task_exact"]) for x in rows)
            formatted = [x for x in rows if not x["format_failure"]]
            summaries.append({
                "model_id": model, "cohort": cohort, "n": len(rows), "exact_count": exact, "exact_rate": rate(exact, len(rows)),
                "failure_rate": rate(len(rows) - exact, len(rows)), "format_failures": sum(bool(x["format_failure"]) for x in rows),
                "format_failure_rate": rate(sum(bool(x["format_failure"]) for x in rows), len(rows)),
                "content_failures_given_format": sum(bool(x["content_failure_given_format"]) for x in formatted),
                "content_failure_rate_given_format": rate(sum(bool(x["content_failure_given_format"]) for x in formatted), len(formatted)),
                "mean_tokens_per_cluster": rate(sum(x["tokens_per_cluster"] for x in scans), len(scans)),
                "mean_split_codepoint_count": rate(sum(x["split_codepoint_count"] or 0 for x in scans), len(scans)),
                "tokenizer_roundtrip_exact_rate": rate(sum(x["tokenizer_roundtrip_exact"] for x in scans), len(scans)),
            })
    return summaries


def newest_old_copy(scored, model_ids):
    out = []
    for model in model_ids:
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy"]
        newest = [x for x in rows if x.get("copy_role") in ("new_emoji", "new_nonemoji")]
        old = [x for x in rows if x.get("copy_role") in ("old_emoji_control", "old_nonemoji_control")]
        ncorrect = sum(bool(x["task_exact"]) for x in newest)
        ocorrect = sum(bool(x["task_exact"]) for x in old)
        nfail = rate(len(newest) - ncorrect, len(newest))
        ofail = rate(len(old) - ocorrect, len(old))
        diff = nfail - ofail if nfail is not None and ofail is not None else None
        diff_pp = None if diff is None else 100 * diff
        out.append({
            "model_id": model, "newest_n": len(newest), "newest_correct": ncorrect, "newest_exact_rate": rate(ncorrect, len(newest)), "newest_failure_rate": nfail,
            "old_control_n": len(old), "old_control_correct": ocorrect, "old_control_exact_rate": rate(ocorrect, len(old)), "old_control_failure_rate": ofail,
            "newest_minus_old_failure_rate": diff, "newest_minus_old_failure_pp": diff_pp,
            "meets_operational_hard_kill_conditions": len(newest) == 94 and ncorrect >= 92 and diff is not None and diff <= 0.02,
            "proceed_rule1_model_signal": diff is not None and diff >= 0.05,
        })
    return out


def systematic_counts(scored, model_ids):
    rows = []
    for model in model_ids:
        subset = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("copy_role") in ("new_emoji", "new_nonemoji") and not x["task_exact"]]
        row = {"model_id": model}
        for cls in SYSTEMATIC_CLASSES:
            row[cls] = len({x["item_id"] for x in subset if x.get(cls)})
        row["any_systematic_class_at_least_5"] = any(row[cls] >= 5 for cls in SYSTEMATIC_CLASSES)
        rows.append(row)
    return rows


def taxonomy_summary(scored):
    groups = defaultdict(list)
    for row in scored:
        if row["probe_type"] == "copy":
            groups[(row["model_id"], row["cohort"])].append(row)
    out = []
    for (model, cohort), rows in sorted(groups.items(), key=lambda x: (x[0][0], COHORT_ORDER.index(x[0][1]))):
        row = {"model_id": model, "cohort": cohort, "n": len(rows), "failures": sum(not x["task_exact"] for x in rows), "format_failures": sum(x["format_failure"] for x in rows)}
        for cls in TAXONOMY_CLASSES:
            row[cls] = sum(bool(x.get(cls)) for x in rows)
        out.append(row)
    return out


def summarize_structural(scored, model_ids):
    out = []
    count_frag = []
    for model in model_ids:
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] != "copy"]
        for typ in ("decomposition", "count", "comparison"):
            subsets = defaultdict(list)
            for row in [x for x in rows if x["probe_type"] == typ]:
                key = (row.get("era"), row.get("fragmentation_group") if typ == "count" else "all")
                subsets[key].append(row)
            for (era, frag), sub in sorted(subsets.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
                exact = sum(bool(x["task_exact"]) for x in sub)
                formatted = [x for x in sub if not x["format_failure"]]
                out.append({
                    "model_id": model, "probe_type": typ, "era": era, "fragmentation_group": frag, "n": len(sub), "exact_count": exact,
                    "exact_rate": rate(exact, len(sub)), "failure_rate": rate(len(sub) - exact, len(sub)),
                    "format_failures": sum(bool(x["format_failure"]) for x in sub), "format_failure_rate": rate(sum(bool(x["format_failure"]) for x in sub), len(sub)),
                    "content_failure_rate_given_format": rate(sum(bool(x["content_failure_given_format"]) for x in formatted), len(formatted)),
                })
        for scope in ("pooled", "newest", "legacy"):
            sub = [x for x in rows if x["probe_type"] == "count" and (scope == "pooled" or x.get("era") == scope)]
            entry = {"model_id": model, "scope": scope}
            for frag in ("low", "high"):
                fr = [x for x in sub if x.get("fragmentation_group") == frag]
                exact = sum(bool(x["task_exact"]) for x in fr)
                entry[f"{frag}_n"] = len(fr)
                entry[f"{frag}_exact"] = exact
                entry[f"{frag}_failure_rate"] = rate(len(fr) - exact, len(fr))
            entry["high_minus_low_failure_rate"] = entry["high_failure_rate"] - entry["low_failure_rate"] if entry["high_failure_rate"] is not None and entry["low_failure_rate"] is not None else None
            entry["high_minus_low_failure_pp"] = None if entry["high_minus_low_failure_rate"] is None else 100 * entry["high_minus_low_failure_rate"]
            entry["proceed_rule2_model_signal"] = entry["high_minus_low_failure_rate"] is not None and entry["high_minus_low_failure_rate"] >= 0.15
            count_frag.append(entry)
    return out, count_frag


def structural_kill_rows(scored, count_frag, model_ids):
    out = []
    for model in model_ids:
        rows = [x for x in scored if x["model_id"] == model]
        nd = [x for x in rows if x["probe_type"] == "decomposition" and x.get("era") == "newest"]
        nc = [x for x in rows if x["probe_type"] == "count" and x.get("era") == "newest"]
        pooled = next((x for x in count_frag if x["model_id"] == model and x["scope"] == "pooled"), None)
        nd_correct = sum(bool(x["task_exact"]) for x in nd)
        nc_correct = sum(bool(x["task_exact"]) for x in nc)
        gap = pooled["high_minus_low_failure_rate"] if pooled else None
        out.append({
            "model_id": model, "newest_decomp_n": len(nd), "newest_decomp_correct": nd_correct, "newest_decomp_exact_rate": rate(nd_correct, len(nd)),
            "newest_count_n": len(nc), "newest_count_correct": nc_correct, "newest_count_exact_rate": rate(nc_correct, len(nc)),
            "count_high_minus_low_failure_rate": gap,
            "meets_structural_hard_kill_conditions": len(nd) == 24 and nd_correct >= 22 and len(nc) == 18 and nc_correct >= 17 and gap is not None and gap <= 0.02,
        })
    return out


def trend_rows(copy_summary, model_ids):
    out = []
    for model in model_ids:
        rows = [x for x in copy_summary if x["model_id"] == model]
        bycohort = {x["cohort"]: x for x in rows}
        seq = [bycohort[x] for x in NEW_EMOJI_ORDER if x in bycohort]
        legacy = bycohort.get("legacy emoji")
        rates = [x["exact_rate"] for x in seq]
        failures = [x["failure_rate"] for x in seq]
        xs = list(range(len(rates)))
        slope = None
        if len(set(xs)) > 1 and len(xs) == len(failures):
            xbar = sum(xs) / len(xs)
            ybar = sum(failures) / len(failures)
            den = sum((x - xbar) ** 2 for x in xs)
            slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, failures)) / den if den else None
        with_legacy = ([legacy["exact_rate"]] if legacy else []) + rates
        out.append({
            "model_id": model, "new_emoji_exact_rates": rates, "new_emoji_failure_rates": failures,
            "new_emoji_exact_monotone_nonincreasing": len(rates) == 4 and all(b <= a + 1e-12 for a, b in zip(rates, rates[1:])),
            "legacy_plus_new_exact_monotone_nonincreasing": len(with_legacy) == 5 and all(b <= a + 1e-12 for a, b in zip(with_legacy, with_legacy[1:])),
            "cohort_index_failure_slope_pp_per_step": None if slope is None else 100 * slope,
            "ascii_exact_rate": bycohort.get("ASCII", {}).get("exact_rate"),
            "legacy_emoji_exact_rate": legacy.get("exact_rate") if legacy else None,
        })
    return out


def runtime_summary(model_ids):
    rows = r2.read_jsonl(RUNTIME_PATH)
    out = []
    for model in model_ids:
        batches = [x for x in rows if x.get("model_id") == model and x.get("event") == "batch_completed"]
        loads = [x for x in rows if x.get("model_id") == model and x.get("event") == "model_loaded"]
        out.append({
            "model_id": model, "batches": len(batches), "probe_rows_completed": sum(x.get("probe_count", 0) for x in batches),
            "batch_elapsed_seconds": sum(x.get("elapsed_seconds", 0) for x in batches), "load_seconds": sum(x.get("load_seconds", 0) for x in loads),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in batches), "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in batches),
            "peak_memory_gb_max": max([x.get("peak_memory_gb") or 0 for x in batches] or [0]),
            "process_rss_gb_max": max([x.get("process_rss_gb_after_batch") or 0 for x in batches] or [0]),
        })
    return out


def completeness_rows(scored, model_ids):
    out = []
    expected = Counter({"copy": 182, "count": 36, "decomposition": 48, "comparison": 24})
    for model in model_ids:
        rows = [x for x in scored if x["model_id"] == model]
        counts = Counter(x["probe_type"] for x in rows)
        out.append({"model_id": model, "rows": len(rows), "expected_rows": 290, **counts, "complete": len(rows) == 290 and counts == expected})
    return out


def vintage_contrast(newest, copy_summary, valid_models):
    cfg_by_model = {x["model_id"]: x for x in r2.model_configs()}
    rows = []
    for row in newest:
        model = row["model_id"]
        cohort = [x for x in copy_summary if x["model_id"] == model]
        bycohort = {x["cohort"]: x for x in cohort}
        rows.append({
            "model_id": model, "vintage_group": cfg_by_model[model]["vintage_group"], "valid_for_interpretation": model in valid_models,
            "ascii_exact_rate": bycohort.get("ASCII", {}).get("exact_rate"),
            "legacy_emoji_exact_rate": bycohort.get("legacy emoji", {}).get("exact_rate"),
            "legacy_nonemoji_exact_rate": bycohort.get("legacy non-emoji", {}).get("exact_rate"),
            "newest_exact_rate": row["newest_exact_rate"], "old_control_exact_rate": row["old_control_exact_rate"],
            "newest_minus_old_failure_pp": row["newest_minus_old_failure_pp"],
        })
    groups = {}
    for group in sorted({x["vintage_group"] for x in rows}):
        valid = [x for x in rows if x["vintage_group"] == group and x["valid_for_interpretation"]]
        allrows = [x for x in rows if x["vintage_group"] == group]
        groups[group] = {
            "models": [x["model_id"] for x in allrows], "valid_models": [x["model_id"] for x in valid],
            "mean_newest_minus_old_failure_pp_valid": rate(sum(x["newest_minus_old_failure_pp"] for x in valid), len(valid)),
            "mean_newest_exact_rate_valid": rate(sum(x["newest_exact_rate"] for x in valid), len(valid)),
            "mean_old_control_exact_rate_valid": rate(sum(x["old_control_exact_rate"] for x in valid), len(valid)),
        }
    vintage = groups.get("vintage_llama2_tokenizer", {})
    newer = groups.get("newer_tokenizer", {})
    vgap = vintage.get("mean_newest_minus_old_failure_pp_valid")
    ngap = newer.get("mean_newest_minus_old_failure_pp_valid")
    return {
        "per_model": rows, "groups": groups,
        "newer_minus_vintage_failure_gap_difference_pp": None if vgap is None or ngap is None else ngap - vgap,
        "interpretable": bool(vintage.get("valid_models")) and bool(newer.get("valid_models")),
        "interpretation_note": "Positive difference means newer tokenizers have a larger newest-minus-old failure gap than the Llama-2-vintage carrier. Validity-gate failures make this descriptive rather than interpretable.",
    }


def build_verdict(scored):
    model_ids = [x["model_id"] for x in r2.model_configs()]
    gates = {}
    for model in model_ids:
        rows = [x for x in scored if x["model_id"] == model]
        gates[model] = {"ascii": ascii_gate(rows), "wrapper": wrapper_gate(rows)}
    mutations = gate_mutation_checks()
    scanmap = copy_scan_metrics()
    copy_summary = summarize_copy(scored, scanmap, model_ids)
    newest = newest_old_copy(scored, model_ids)
    systematic = systematic_counts(scored, model_ids)
    structural, count_frag = summarize_structural(scored, model_ids)
    structural_kill = structural_kill_rows(scored, count_frag, model_ids)
    trends = trend_rows(copy_summary, model_ids)
    complete = completeness_rows(scored, model_ids)
    valid_models = [model for model, gate in gates.items() if gate["ascii"]["pass"] and gate["wrapper"]["pass"]]
    invalid_models = [model for model in gates if model not in valid_models]
    eval_newest = [x for x in newest if x["model_id"] in valid_models]
    eval_structural_kill = [x for x in structural_kill if x["model_id"] in valid_models]
    copy_hard_kill_models = [x["model_id"] for x in eval_newest if x["meets_operational_hard_kill_conditions"]]
    structural_hard_kill_models = [x["model_id"] for x in eval_structural_kill if x["meets_structural_hard_kill_conditions"]]
    copy_hard_kill = len(valid_models) >= 2 and len(copy_hard_kill_models) == len(valid_models)
    structural_hard_kill = len(valid_models) >= 2 and len(structural_hard_kill_models) == len(valid_models)
    rule1_models = [x["model_id"] for x in eval_newest if x["proceed_rule1_model_signal"]]
    pooled_frag = [x for x in count_frag if x["scope"] == "pooled" and x["model_id"] in valid_models]
    rule2_models = [x["model_id"] for x in pooled_frag if x["proceed_rule2_model_signal"]]
    rule3_models = [x["model_id"] for x in systematic if x["model_id"] in valid_models and x["any_systematic_class_at_least_5"]]
    scan_verdict = json.loads((RESULTS / "scan_verdict.json").read_text()) if (RESULTS / "scan_verdict.json").exists() else {}
    rule4 = bool(scan_verdict.get("proceed_rule4_fires"))
    vintage = vintage_contrast(newest, copy_summary, valid_models)
    behavioral_proceed = len(rule1_models) >= 2 or len(rule2_models) >= 2 or bool(rule3_models)
    if len(valid_models) < 2:
        interpretation = "inconclusive_validity_gate_failure"
    elif copy_hard_kill and structural_hard_kill:
        interpretation = "whole_project_hard_kill"
    elif behavioral_proceed:
        interpretation = "proceed"
    else:
        interpretation = "inconclusive_no_behavioral_proceed_or_hard_kill_signal"
    verdict = {
        "spec_version": "SPEC.md sections 8.1-8.3; round-2 implementation corrects round-1 analysis defects: operational/structural hard kills require the spec's high-success thresholds, and proceed rule 1 uses 5 percentage points rather than 20",
        "hard_kill_adaptation_note": "the spec's 'every one of four models' language is applied to every gate-valid round-2 model and requires at least two gate-valid models; models failing validity gates are not allowed to establish either a hard kill or a proceed signal",
        "completeness": complete, "all_models_complete": all(x["complete"] for x in complete),
        "validity_gates": gates, "gate_mutation_checks": mutations,
        "all_gate_mutation_checks_pass": mutations["ascii_gate_mutation_detects_failure"] and mutations["wrapper_gate_mutation_detects_failure"] and mutations["extractor_mutation_checks_pass"],
        "ascii_gates_pass": all(x["ascii"]["pass"] for x in gates.values()),
        "wrapper_gates_pass": all(x["wrapper"]["pass"] for x in gates.values()),
        "valid_models": valid_models, "invalid_models": invalid_models, "valid_model_count": len(valid_models), "behavioral_round2_conclusive": len(valid_models) >= 2,
        "copy_hard_kill_models": copy_hard_kill_models, "structural_hard_kill_models": structural_hard_kill_models,
        "copy_summary": copy_summary, "newest_old_copy": newest, "systematic_error_counts": systematic,
        "structural_summary": structural, "count_fragmentation": count_frag, "structural_hard_kill_rows": structural_kill,
        "cohort_trends": trends, "vintage_contrast": vintage,
        "hard_kill_operational_arm": copy_hard_kill, "hard_kill_structural_arm": structural_hard_kill,
        "whole_project_hard_kill": copy_hard_kill and structural_hard_kill,
        "behavioral_proceed_rule1_fires": len(rule1_models) >= 2, "behavioral_proceed_rule1_models": rule1_models,
        "behavioral_proceed_rule2_fires": len(rule2_models) >= 2, "behavioral_proceed_rule2_models": rule2_models,
        "behavioral_proceed_rule3_fires": bool(rule3_models), "behavioral_proceed_rule3_models": rule3_models,
        "tokenizer_scan_proceed_rule4_fires": rule4,
        "overall_proceed_rule_fires": len(rule1_models) >= 2 or len(rule2_models) >= 2 or bool(rule3_models) or rule4,
        "behavioral_only_proceed_rule_fires": behavioral_proceed,
        "final_behavioral_interpretation": interpretation,
        "regression_stop_rule": "not needed for round-2 decision because tokenizer-scan rule 4 already fires" if rule4 else "not evaluated in descriptive round-2 analysis",
    }
    return verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, mutations, complete, model_ids


def write_outputs(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, runtime):
    write_csv(ROUND2 / "round2_copy_cohort_summary.csv", copy_summary, ["model_id", "cohort", "n", "exact_count", "exact_rate", "failure_rate", "format_failures", "format_failure_rate", "content_failures_given_format", "content_failure_rate_given_format", "mean_tokens_per_cluster", "mean_split_codepoint_count", "tokenizer_roundtrip_exact_rate"])
    write_csv(ROUND2 / "round2_copy_newest_old_summary.csv", newest, list(newest[0]))
    write_csv(ROUND2 / "round2_copy_systematic_error_counts.csv", systematic, list(systematic[0]))
    write_csv(ROUND2 / "round2_structural_summary.csv", structural, ["model_id", "probe_type", "era", "fragmentation_group", "n", "exact_count", "exact_rate", "failure_rate", "format_failures", "format_failure_rate", "content_failure_rate_given_format"])
    write_csv(ROUND2 / "round2_count_fragmentation_summary.csv", count_frag, list(count_frag[0]))
    write_csv(ROUND2 / "round2_structural_hard_kill_summary.csv", structural_kill, list(structural_kill[0]))
    write_csv(ROUND2 / "round2_copy_error_taxonomy_summary.csv", taxonomy_summary(r2.read_jsonl(SCORED_PATH)), ["model_id", "cohort", "n", "failures", "format_failures"] + TAXONOMY_CLASSES)
    write_csv(ROUND2 / "round2_runtime_summary.csv", runtime, ["model_id", "batches", "probe_rows_completed", "batch_elapsed_seconds", "load_seconds", "generation_tokens", "prompt_tokens", "peak_memory_gb_max", "process_rss_gb_max"])
    (ROUND2 / "round2_verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def model_manifest(runtime):
    return {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "models": r2.model_configs(), "skipped_models": r2.SKIPPED_MODELS,
        "copy_protocol_templates": r2.protocol_template_metadata(),
        "protocol_freeze": load_json(ROUND2 / "protocol_freeze.json"),
        "quantization_sensitivity": load_json(ROUND2 / "quantization_sensitivity.json"),
        "probe_identity_audit": load_json(ROUND2 / "probe_identity_audit.json"),
        "probe_content_discipline": "one user message per probe; no explicit system message; identical user content across models; each model's own chat template, including any template-injected default wrapper text, may differ",
        "decode_params": {"sampler": "greedy/argmax", "temperature": 0.0, "max_tokens": r2.MAX_TOKENS, "repetition_penalty": None},
        "inference_backend": "mlx-lm batch_generate on Apple Silicon; no vLLM; one model process at a time",
        "host": {"python_version": platform.python_version(), "unicodedata_version": unicodedata.unidata_version, "platform": platform.platform(), "machine": platform.machine(), "normalization_unicodedata2_version": unicodedata2.unidata_version},
        "runtime": runtime,
        "tokenizer_linkage": load_json(ROUND2 / "tokenizer_linkage_round2.json"),
        "count_bundle_note": "count probes are tokenizer-specific; each behavioral model uses the 36 bundles constructed for its linked scanned tokenizer artifact; item identities were verified unchanged from round 1",
    }


def manual_failure_examples(scored, model_ids):
    lines = []
    for model in model_ids:
        ascii_failure = next((x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("cohort") == "ASCII" and not x["task_exact"]), None)
        if ascii_failure:
            lines.append(f"- {model} ASCII near-miss: target={json.dumps(ascii_failure['target'], ensure_ascii=False)}; extracted={json.dumps(ascii_failure.get('extracted_answer'), ensure_ascii=False)}; newline_padding={ascii_failure.get('newline_padding')}; first_byte_mismatch={ascii_failure.get('first_byte_mismatch_offset')}.")
        newest_failure = next((x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("copy_role") in ("new_emoji", "new_nonemoji") and not x["task_exact"]), None)
        if newest_failure:
            classes = [c for c in TAXONOMY_CLASSES if newest_failure.get(c)]
            lines.append(f"- {model} newest copy failure example: item={newest_failure.get('item_id')}; target={json.dumps(newest_failure['target'], ensure_ascii=False)}; extracted={json.dumps(newest_failure.get('extracted_answer'), ensure_ascii=False)}; classes={','.join(classes) or 'format_failure'}; raw_prefix={json.dumps(newest_failure['raw_text'][:90], ensure_ascii=False)}.")
    return lines[:12]


def write_report(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, complete, runtime, scored, model_ids):
    quant = load_json(ROUND2 / "quantization_sensitivity.json") or {}
    freeze = load_json(ROUND2 / "protocol_freeze.json") or {}
    linkage = load_json(ROUND2 / "tokenizer_linkage_round2.json") or {}
    estimate = load_json(ROUND2 / "runtime_estimate.json") or {}
    lines = [
        "# Unicode release debt: behavioral round 2 report", "",
        "Scope: deterministic local behavioral round 2 after the round-1 validity-gate failure. Probe item identities are byte-for-byte unchanged from round 1; only the copy prompt/extraction instrument and model slate changed. Inference used MLX `batch_generate`, greedy/argmax decoding, max 64 output tokens, no repetition penalty, no vLLM, and one model process at a time. Raw responses are append-only in `results/behavioral_round2/round2_raw.jsonl`.", "",
        "## Instrument fixes and freeze order", "",
    ]
    if quant:
        lines += [
            f"- Quantization sensitivity was run first, before calibration or cohort inference: Qwen2.5-1.5B ASCII exact was {quant.get('bf16_exact')}/24 bf16 versus {quant.get('q8_exact')}/24 MLX 8-bit; drop={quant.get('q8_exact_drop_count')}/24. Verdict: **{quant.get('decision')}**.",
        ]
    if freeze:
        lines += [
            f"- Copy protocol frozen before any emoji-cohort or structural generation in `protocol_freeze.json`: `{freeze.get('selected_copy_protocol')}`, template SHA-256 `{freeze.get('selected_template_sha256')}`, full-format SHA-256 `{freeze.get('selected_format_sha256')}`. The four exemplars are held-out ASCII filler strings, with zero exact overlap against copy probes.",
            "- Protocol selection: eight candidates were calibrated on 48 ASCII control items across two harness-test models before any emoji-cohort inference. Single-turn few-shot and instruction variants failed the ASCII gate badly (9-18/48 exact). `fewshot_copy_v4_multiturn` and `fewshot_copy_v5_multiturn_cueless` tied numerically (46/48 exact, 0 format failures, ASCII gate pass on both test models). v5 was selected over v4 because its probe user turn carries no output cue: the `<OUT>` answer format must transfer from the exemplar pairs alone, so format compliance measures exemplar-driven task inference rather than cue echoing, the more conservative instrument. v6 (minimal, no prose) regressed to 43/48 and was rejected. The v1 freeze attempt was superseded before any emoji-cohort inference; supersession is recorded in `protocol_freeze_attempts.jsonl`.",
            "- Strict extraction: exactly one `<OUT>`/`</OUT>` pair, no non-whitespace outside the wrapper, and no stripping/normalization/repair of extracted bytes. Newline padding inside the wrapper is a byte-copy failure, not a valid copy.",
        ]
    if estimate:
        lines += [f"- Runtime estimate before cohort inference: {estimate.get('projected_total_minutes')} minutes projected for {estimate.get('model_count')} models; below the 6-hour trimming threshold.", ""]
    else:
        lines += [""]
    lines += ["## Models, provenance, and tokenizer linkage", ""]
    mrows = []
    link_rows = {x["model_id"]: x for x in linkage.get("rows", [])}
    for cfg in r2.model_configs():
        rt = next((x for x in runtime if x["model_id"] == cfg["model_id"]), {})
        lr = link_rows.get(cfg["model_id"], {})
        mrows.append([
            cfg["model_id"], cfg["hf_repo"] + "@" + cfg["revision"][:12], cfg["upstream_repo"] + "@" + cfg["upstream_revision"][:12], cfg["quantization_class"],
            lr.get("all_scan_files_byte_identical"), lr.get("linkage_acceptable_for_scan_covariates"), rt.get("probe_rows_completed", 0), f"{rt.get('batch_elapsed_seconds', 0) / 60:.1f} min", f"{rt.get('peak_memory_gb_max', 0):.1f} GB",
        ])
    lines += [md_table(["model", "inference artifact", "upstream artifact", "quant", "strict tokenizer files identical", "scan linkage acceptable", "rows", "generation time", "peak MLX memory"], mrows), ""]
    lines += [
        "- Canonical `meta-llama/Llama-2-7b-chat-hf` was attempted first and returned HTTP 403 gated-repository access. The accessible NousResearch mirror has tokenizer files matching the scanned llama2 mirror but no weight-hash-equivalence documentation; per the round-2 rule it was not used. TinyLlama is therefore the vintage Llama-2-tokenizer carrier.",
        "- Strict per-file tokenizer byte identity is reported separately from scan-covariate usability. MLX-community Qwen2.5 8-bit repositories were found to package a non-byte-identical Qwen3-like tokenizer, so canonical Qwen2.5 tokenizer files were overlaid before accepted q8 inference; wrong-tokenizer quant-control rows are preserved only as superseded artifacts. Qwen3-4B-Instruct-2507 and Gemma-3-4B differ in auxiliary tokenizer/config files but share core `tokenizer.json`; probe-text token-id equality is explicit in `tokenizer_linkage_round2.json`.",
        "- Chat-template verification (`qwen3_template_rendering_check.json`, `gemma_template_rendering_check.json`): the Qwen3 q8 artifact's `chat_template.jinja` file differs textually from the canonical embedded template, but both render the round-2 multi-turn message structure byte-identically, matching the actual inference prompt hash. Gemma's artifact template is byte-identical to the canonical embedded template. Rendered prompts, not just token IDs, are therefore verified.",
        "- Gemma-3-4B inference used the mlx-community multimodal artifact because no clean official text-only Gemma-3-4B MLX 8-bit artifact exists; only the text backbone was exercised (no vision inputs). Its card states conversion from `google/gemma-3-4b-it` with mlx-vlm 0.1.18; canonical upstream is license-gated, so the unsloth mirror (bf46152c) served as the accessible canonical reference for tokenizer/template comparison. Weight-hash equivalence to canonical cannot be independently verified for any MLX conversion; pinned revisions are the artifact identity.",
        "",
        "## Harness-validity gates", "",
    ]
    grows = []
    for model, g in gates.items():
        grows.append([model, f"{g['ascii']['correct']}/{g['ascii']['n']}", pct(g["ascii"]["accuracy"]), g["ascii"]["pass"], f"{g['wrapper']['format_failures']}/{g['wrapper']['n']}", pct(g["wrapper"]["format_failure_rate"]), g["wrapper"]["pass"]])
    lines += [md_table(["model", "ASCII exact", "ASCII rate", "ASCII gate", "wrapper failures", "wrapper rate", "wrapper gate"], grows), ""]
    mut = verdict["gate_mutation_checks"]
    lines += [f"Mutation controls: ASCII-gate mutation detected={mut['ascii_gate_mutation_detects_failure']}; wrapper-gate mutation detected={mut['wrapper_gate_mutation_detects_failure']}; extractor mutation battery passed={mut['extractor_mutation_checks_pass']}. Completeness: " + "; ".join(f"{x['model_id']} {x['rows']}/{x['expected_rows']}" for x in complete) + ".", ""]
    lines += ["## Copy exact-match rates by cohort", ""]
    crows = []
    for model in model_ids:
        row = [model]
        for cohort in COHORT_ORDER:
            hit = next((x for x in copy_summary if x["model_id"] == model and x["cohort"] == cohort), None)
            row.append("NA" if not hit else f"{hit['exact_count']}/{hit['n']} ({pct(hit['exact_rate'])})")
        crows.append(row)
    lines += [md_table(["model"] + COHORT_ORDER, crows), ""]
    lines += ["## Newest-bin versus old-copy controls", ""]
    nrows = [[x["model_id"], f"{x['newest_correct']}/{x['newest_n']}", pct(x["newest_exact_rate"]), f"{x['old_control_correct']}/{x['old_control_n']}", pct(x["old_control_exact_rate"]), pp(x["newest_minus_old_failure_pp"]), x["meets_operational_hard_kill_conditions"], x["proceed_rule1_model_signal"]] for x in newest]
    lines += [md_table(["model", "newest exact", "newest rate", "old exact", "old rate", "failure gap", "meets operational hard-kill conditions", "rule-1 signal (>=5 pp)"], nrows), ""]
    lines += ["## Cohort-trend check", ""]
    trows = [[x["model_id"], ", ".join(f"{pct(y)}" for y in x["new_emoji_exact_rates"]), x["new_emoji_exact_monotone_nonincreasing"], x["legacy_plus_new_exact_monotone_nonincreasing"], "NA" if x["cohort_index_failure_slope_pp_per_step"] is None else f"{x['cohort_index_failure_slope_pp_per_step']:.1f}", pct(x["ascii_exact_rate"])] for x in trends]
    lines += [md_table(["model", "E15.1/E16/E17/E18 exact rates", "new-only monotone non-increasing", "legacy+new monotone", "failure slope pp/step", "ASCII exact"], trows), "",
        "Interpretation: positive slope means failure rises from E15.1 to E18. These raw cohort contrasts are descriptive and item-type-confounded because E15.1/E17 are mostly ZWJ families while E16/E18 are atomic; they are not the adjusted confirmatory regression.", ""]
    lines += ["## Cross-model vintage contrast", ""]
    vc = verdict["vintage_contrast"]
    vrows = [[x["model_id"], x["vintage_group"], x["valid_for_interpretation"], pct(x["newest_exact_rate"]), pct(x["old_control_exact_rate"]), pp(x["newest_minus_old_failure_pp"])] for x in vc["per_model"]]
    lines += [md_table(["model", "vintage group", "gate-valid", "newest exact", "old exact", "newest-old failure gap"], vrows), ""]
    lines += [f"Group summary: vintage valid mean gap={pp(vc['groups'].get('vintage_llama2_tokenizer', {}).get('mean_newest_minus_old_failure_pp_valid'))}; newer valid mean gap={pp(vc['groups'].get('newer_tokenizer', {}).get('mean_newest_minus_old_failure_pp_valid'))}; newer-minus-vintage difference={pp(vc.get('newer_minus_vintage_failure_gap_difference_pp'))}. Interpretable={vc.get('interpretable')}.", ""]
    lines += ["## Structural probes", ""]
    srows = []
    for x in structural_kill:
        frag = next((y for y in count_frag if y["model_id"] == x["model_id"] and y["scope"] == "pooled"), {})
        srows.append([x["model_id"], f"{x['newest_decomp_correct']}/{x['newest_decomp_n']}", pct(x["newest_decomp_exact_rate"]), f"{x['newest_count_correct']}/{x['newest_count_n']}", pct(x["newest_count_exact_rate"]), pp(frag.get("high_minus_low_failure_pp")), x["meets_structural_hard_kill_conditions"], frag.get("proceed_rule2_model_signal")])
    lines += [md_table(["model", "newest decomp", "decomp rate", "newest count", "count rate", "high-low count failure gap", "meets structural hard-kill conditions", "rule-2 signal"], srows), ""]
    detail_rows = []
    for x in structural:
        if x["probe_type"] in ("decomposition", "count", "comparison") and x["era"] in ("newest", "legacy") and (x["probe_type"] != "count" or x["fragmentation_group"] in ("low", "high")):
            detail_rows.append([x["model_id"], x["probe_type"], x["era"], x["fragmentation_group"], f"{x['exact_count']}/{x['n']}", pct(x["exact_rate"]), pct(x["format_failure_rate"])])
    lines += [md_table(["model", "probe", "era", "fragmentation", "exact", "rate", "format-failure rate"], detail_rows), ""]
    lines += ["## Copy byte-error taxonomy", ""]
    tax = taxonomy_summary(scored)
    newest_tax = [x for x in tax if x["cohort"] in ("E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji", "new non-emoji")]
    tax_rows = [[x["model_id"], x["cohort"], x["failures"], x["missing_zwj"], x["missing_vs16"], x["nfc_nfd_variant"], x["truncation"], x["newline_padding"], x["other_unclassified_mismatch"]] for x in newest_tax]
    lines += [md_table(["model", "cohort", "failures", "ZWJ dropped", "VS16 dropped", "NFC/NFD variant", "truncation", "newline padding", "other"], tax_rows), ""]
    lines += ["## Systematic copy-error classes in newest items", ""]
    erows = [[x["model_id"]] + [x[c] for c in SYSTEMATIC_CLASSES] + [x["any_systematic_class_at_least_5"]] for x in systematic]
    lines += [md_table(["model"] + SYSTEMATIC_CLASSES + ["rule-3 signal"], erows), ""]
    lines += ["## Manual failure inspection", ""] + manual_failure_examples(scored, model_ids) + ["", "## Wrapper-direction mimicry analysis (post-hoc)", ""]
    mimicry_path = ROUND2 / "round2_wrapper_mimicry_analysis.json"
    if mimicry_path.exists():
        mimicry = json.loads(mimicry_path.read_text())
        mrows = []
        for model in model_ids:
            m = mimicry.get(model)
            if m:
                mrows.append([model, m["in_wrapper_mimicry_failures"], m["mimicry_inner_byte_exact"], m["mimicry_inner_byte_diff"], json.dumps(m["mimicry_by_cohort"])])
        lines += [md_table(["model", "IN-tag mimicry failures", "inner byte-exact", "inner byte-diff", "by cohort"], mrows), ""]
        lines += [
            "Strict-protocol extraction treats `<IN>...</IN>` outputs as format failures, and the validity gates account them as such; that accounting is unchanged. This post-hoc check asks a separate scientific question: is the byte-copy operation itself intact when the wrapper direction is wrong? For qwen2.5-7b-instruct-mlx8, **all 28 mimicry failures contain byte-perfect copies of the target inside the wrong-direction wrapper**. The model mirrors the user-turn `<IN>` tag instead of the assistant exemplar `<OUT>` tag, concentrated in emoji cohorts (E15.1=7, E16=3, E17=9, E18=6, legacy emoji=3; zero in ASCII, legacy non-emoji, and new non-emoji). No other model produced any mimicry output.",
            "",
            "Interpretation consequence: the qwen2.5-7b newest-minus-old copy gap of 20.3 pp visible in the tables above is **not evidence of byte-copy release debt**; the underlying copied bytes are exact in every mimicry case. The model is gate-invalid (15.4% wrapper failure) and therefore excluded from every proceed/hard-kill rule, and the gap is reported descriptively only. The single non-mimicry 7B copy failure (`copy-072`) emitted a replacement character plus non-target text (`<OUT>\\ufffdfindViewById</OUT>`), the only true byte-level copy corruption observed in any model outside single-character ASCII substitutions.",
            "",
        ]
    lines += ["## Proceed/kill verdict under spec 8.3", ""]
    lines += [
        f"- Behavioral round 2 conclusive: **{verdict['behavioral_round2_conclusive']}** ({verdict['valid_model_count']} gate-valid models; at least two required). Final behavioral interpretation: **{verdict['final_behavioral_interpretation']}**.",
        f"- Operational hard kill among gate-valid models: **{verdict['hard_kill_operational_arm']}**. Structural hard kill among gate-valid models: **{verdict['hard_kill_structural_arm']}**. Whole-project hard kill: **{verdict['whole_project_hard_kill']}**.",
        f"- Behavioral proceed rule 1 (newest-minus-old copy failure >=5 pp in >=2 gate-valid models): **{verdict['behavioral_proceed_rule1_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule1_models']) or 'none'}.",
        f"- Behavioral proceed rule 2 (high-minus-low fragmentation structural failure >=15 pp in >=2 models): **{verdict['behavioral_proceed_rule2_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule2_models']) or 'none'}.",
        f"- Behavioral proceed rule 3 (systematic edit class in >=5 distinct newest items in any model): **{verdict['behavioral_proceed_rule3_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule3_models']) or 'none'}.",
        f"- Tokenizer-scan proceed rule 4: **{verdict['tokenizer_scan_proceed_rule4_fires']}** (already established). Overall proceed-rule status: **{verdict['overall_proceed_rule_fires']}**.",
        f"- Regression-based stop rule: {verdict['regression_stop_rule']}.", "",
        "## Synthesis", "",
        "The two arms of the study separate cleanly in round 2. The **operational copy arm is hard-killed**: every gate-valid model, including the vintage Llama-2-tokenizer TinyLlama-1.1B, copies every newest-cohort item byte-exactly (94/94 newest vs 64/64 legacy controls in each valid model, zero missing-ZWJ/VS16/normalization errors anywhere). Unicode release recency has no measurable effect on byte-copy fidelity, and the only copy failures observed anywhere are single-character substitutions inside adversarial ASCII control strings plus the qwen2.5-7b wrapper-direction mimicry artifact, whose inner bytes are nevertheless exact. Byte-level copying of unfamiliar Unicode is attention-mediated and tokenizer-release-independent at these model scales.",
        "",
        "The **structural arm survives and carries the project forward**: decomposition is an era-independent floor failure (0/48 everywhere, legacy included), while the count probe shows the prespecified fragmentation contrast, high-fragmentation failure exceeding low-fragmentation failure by 50.0 pp (qwen2.5-3b) and 22.2 pp (qwen3-4b) pooled over eras, firing behavioral proceed rule 2 in two gate-valid models. Count errors overcount ZWJ sequences and combining-mark clusters as multiple characters, i.e. models count code points rather than grapheme clusters in both eras. Because the same fragmentation pattern appears in legacy items, round-2 behavioral evidence supports a fragmentation-sensitivity mechanism rather than a release-recency mechanism; the release-cohort contrast survives only in the tokenizer-scan covariates (rule 4) and must be tested in the regression-adjusted full-study analysis, not in copy behavior.",
        "",
        "## Interpretation caveats", "",
        "- The round-1 analysis implementation inverted the spec's high-success hard-kill logic and used 20 pp instead of the spec's 5 pp for proceed rule 1. Round-2 machine verdicts use the spec thresholds directly; this correction is analysis-only and does not change probe items or raw responses.",
        "- Validity gates remain the primary guard against confusing instruction-following failure with Unicode release debt. Invalid models are tabulated descriptively but excluded from proceed/hard-kill rule evaluation.",
        "- Copy attention can succeed even when structural operations fail; the copy and structural arms are therefore reported separately.",
        "- MLX 8-bit inference is accepted only because the prespecified ASCII sensitivity check was within noise; this does not prove quantization neutrality for every Unicode item. In particular, qwen2.5-7b ran only in MLX 8-bit (bf16 7B exceeds the 16 GB host budget), so its wrapper-direction mimicry cannot be isolated from a possible scale-by-quantization interaction; the q8-acceptance control was measured on Qwen2.5-1.5B, not at 7B scale.",
        "- Decomposition exact accuracy is 0/48 in every model for both eras, including legacy emoji sequences that predate every tokenizer here. The decomposition arm therefore cannot discriminate release cohorts at all: it is an era-independent floor failure (models hallucinate plausible-looking but wrong codepoint sequences even for legacy emoji). Fragmentation sensitivity is instead visible in the count arm, where errors systematically count code points rather than grapheme clusters (ZWJ families and VS16/combining-mark sequences overcounted), in both eras.",
        "- TinyLlama structural results are uninterpretable beyond format compliance: 100% format failure on decomposition, count, and comparison (the model does not emit the required answer shapes at all), while its copy arm passes both gates with 181/182 byte-exact copies. The vintage contrast is therefore a copy-arm contrast only.",
        "- Probe-text token-id equality between scan and behavior tokenizers held on every checked probe text for every model (243 unique texts for the Qwen/Gemma scan tokenizers, 242 for TinyLlama; the count difference is a dedup artifact of per-scan-tokenizer probe construction, not an encoding failure).",
        "- The regression-adjusted full-study analysis is not replaced by these round-2 tables.",
        "", "Machine-readable outputs: `round2_manifest.json`, `probe_plan_round2.jsonl`, `round2_raw.jsonl`, `round2_scored.jsonl`, `round2_verdict.json`, `quantization_sensitivity.json`, `protocol_freeze.json`, `tokenizer_linkage_round2.json`, and round2 CSV summaries.",
    ]
    (RESULTS / "ROUND2_REPORT.md").write_text("\n".join(lines) + "\n")


def freeze_round2_hashes():
    names = ["probe_plan_round2.jsonl", "probe_identity_audit.json", "tokenizer_linkage_round2.json", "protocol_freeze.json", "quantization_sensitivity.json", "runtime_estimate.json", "round2_manifest.json", "round2_raw.jsonl", "round2_runtime.jsonl", "round2_scored.jsonl", "round2_verdict.json", "round2_wrapper_mimicry_analysis.json", "qwen3_template_rendering_check.json", "gemma_template_rendering_check.json", "round2_copy_cohort_summary.csv", "round2_copy_newest_old_summary.csv", "round2_copy_systematic_error_counts.csv", "round2_copy_error_taxonomy_summary.csv", "round2_structural_summary.csv", "round2_count_fragmentation_summary.csv", "round2_structural_hard_kill_summary.csv", "round2_runtime_summary.csv"]
    rows = {}
    for name in names:
        path = ROUND2 / name
        if path.exists():
            rows["behavioral_round2/" + name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    report = RESULTS / "ROUND2_REPORT.md"
    if report.exists():
        rows["ROUND2_REPORT.md"] = {"sha256": hashlib.sha256(report.read_bytes()).hexdigest(), "size": report.stat().st_size}
    code_paths = sorted((ucd.ROOT / "src").glob("*round2*.py"))
    rows["round2_code"] = {str(p.relative_to(ucd.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths if p.exists()}
    (ROUND2 / "round2_artifact_hashes.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


def main():
    scored = r2.read_jsonl(SCORED_PATH)
    if not scored:
        raise SystemExit("no scored rows; run run_behavioral_round2.py then score_behavioral_round2.py")
    verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, mutations, complete, model_ids = build_verdict(scored)
    runtime = runtime_summary(model_ids)
    write_outputs(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, runtime)
    (ROUND2 / "round2_manifest.json").write_text(json.dumps(model_manifest(runtime), indent=2, ensure_ascii=False))
    write_report(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, complete, runtime, scored, model_ids)
    freeze_round2_hashes()
    print(json.dumps({
        "rows": len(scored), "all_models_complete": verdict["all_models_complete"],
        "ascii_gates_pass": verdict["ascii_gates_pass"], "wrapper_gates_pass": verdict["wrapper_gates_pass"],
        "valid_models": verdict["valid_models"],
        "hard_kill_operational_arm": verdict["hard_kill_operational_arm"], "hard_kill_structural_arm": verdict["hard_kill_structural_arm"],
        "behavioral_rule1": verdict["behavioral_proceed_rule1_fires"], "behavioral_rule2": verdict["behavioral_proceed_rule2_fires"], "behavioral_rule3": verdict["behavioral_proceed_rule3_fires"],
        "overall_proceed": verdict["overall_proceed_rule_fires"], "mutation_checks": verdict["all_gate_mutation_checks_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
