from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import score_behavioral as scoring
import ucd_build as ucd

RESULTS = ucd.RESULTS
BEHAVIORAL = bp.BEHAVIORAL
SCORED_PATH = BEHAVIORAL / "behavioral_scored.jsonl"
RAW_PATH = BEHAVIORAL / "behavioral_raw.jsonl"
RUNTIME_PATH = BEHAVIORAL / "behavioral_runtime.jsonl"
COHORT_ORDER = ["ASCII", "legacy emoji", "legacy non-emoji", "E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji", "new non-emoji"]
NEW_EMOJI_ORDER = ["E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji"]
SYSTEMATIC_CLASSES = ["missing_zwj", "missing_vs16", "normalization_only_mismatch", "replacement_character_emitted", "decomposed_rgi_sequence"]
TAXONOMY_CLASSES = SYSTEMATIC_CLASSES + ["truncation", "extra_text_after_exact_target", "insertion", "deletion", "substitution", "reordering", "other_unclassified_mismatch"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def ascii_gate(rows):
    ascii_rows = [x for x in rows if x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
    correct = sum(bool(x["task_exact"]) for x in ascii_rows)
    return {"n": len(ascii_rows), "correct": correct, "accuracy": rate(correct, len(ascii_rows)), "pass": len(ascii_rows) == 24 and correct >= 23}


def wrapper_gate(rows):
    copy_rows = [x for x in rows if x["probe_type"] == "copy"]
    failures = sum(bool(x["format_failure"]) for x in copy_rows)
    return {"n": len(copy_rows), "format_failures": failures, "format_failure_rate": rate(failures, len(copy_rows)), "pass": len(copy_rows) == 182 and rate(failures, len(copy_rows)) <= 0.05}


def mutation_checks():
    synthetic_ascii = [{"probe_type": "copy", "cohort": "ASCII", "task_exact": True, "format_failure": False} for _ in range(24)]
    mutated_ascii = [dict(x, task_exact=i >= 2) for i, x in enumerate(synthetic_ascii)]
    synthetic_wrapper = [{"probe_type": "copy", "cohort": "legacy emoji", "task_exact": True, "format_failure": False} for _ in range(182)]
    mutated_wrapper = [dict(x, format_failure=i < 10) for i, x in enumerate(synthetic_wrapper)]
    good_extraction = scoring.extract_copy("<OUT>😀</OUT>")
    bad_extraction = scoring.extract_copy("<OUT>😀</OUT> trailing <OUT>😀</OUT>")
    return {
        "ascii_gate_mutation_detects_failure": (not ascii_gate(mutated_ascii)["pass"]) and ascii_gate(synthetic_ascii)["pass"],
        "wrapper_gate_mutation_detects_failure": (not wrapper_gate(mutated_wrapper)["pass"]) and wrapper_gate(synthetic_wrapper)["pass"],
        "duplicate_wrapper_mutation_detected": (not good_extraction["format_failure"]) and bad_extraction["format_failure"],
        "ascii_mutation_detail": {"baseline": ascii_gate(synthetic_ascii), "mutated": ascii_gate(mutated_ascii)},
        "wrapper_mutation_detail": {"baseline": wrapper_gate(synthetic_wrapper), "mutated": wrapper_gate(mutated_wrapper)},
    }


def copy_scan_metrics():
    rows = read_jsonl(RESULTS / "tokenizer_scan.jsonl")
    out = {}
    for row in rows:
        if row.get("probe_type") == "copy" and row.get("item_id"):
            out[(row["tokenizer_id"], row["item_id"])] = row
    return out


def summarize_copy(scored, scanmap):
    groups = defaultdict(list)
    for row in scored:
        if row["probe_type"] == "copy":
            groups[(row["model_id"], row["cohort"])].append(row)
    summaries = []
    for model in [x["model_id"] for x in bp.MODELS]:
        scan_tid = bp.model_config(model)["scan_tokenizer_id"]
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


def newest_old_copy(scored):
    out = []
    for model in [x["model_id"] for x in bp.MODELS]:
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy"]
        newest = [x for x in rows if x.get("copy_role") in ("new_emoji", "new_nonemoji")]
        old = [x for x in rows if x.get("copy_role") in ("old_emoji_control", "old_nonemoji_control")]
        ncorrect = sum(bool(x["task_exact"]) for x in newest)
        ocorrect = sum(bool(x["task_exact"]) for x in old)
        nfail = rate(len(newest) - ncorrect, len(newest))
        ofail = rate(len(old) - ocorrect, len(old))
        diff = nfail - ofail if nfail is not None and ofail is not None else None
        out.append({
            "model_id": model, "newest_n": len(newest), "newest_correct": ncorrect, "newest_exact_rate": rate(ncorrect, len(newest)), "newest_failure_rate": nfail,
            "old_control_n": len(old), "old_control_correct": ocorrect, "old_control_exact_rate": rate(ocorrect, len(old)), "old_control_failure_rate": ofail,
            "newest_minus_old_failure_rate": diff, "newest_minus_old_failure_pp": None if diff is None else 100 * diff,
            "hard_kill_copy_prong": rate(ncorrect, len(newest)) is not None and rate(ncorrect, len(newest)) <= 0.40 and rate(ocorrect, len(old)) is not None and rate(ocorrect, len(old)) <= 0.40,
            "proceed_rule1_model_signal": diff is not None and diff >= 0.20,
        })
    return out


def systematic_counts(scored):
    rows = []
    for model in [x["model_id"] for x in bp.MODELS]:
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


def summarize_structural(scored):
    out = []
    count_frag = []
    for model in [x["model_id"] for x in bp.MODELS]:
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


def structural_kill_rows(scored, count_frag):
    out = []
    for model in [x["model_id"] for x in bp.MODELS]:
        rows = [x for x in scored if x["model_id"] == model]
        nd = [x for x in rows if x["probe_type"] == "decomposition" and x.get("era") == "newest"]
        nc = [x for x in rows if x["probe_type"] == "count" and x.get("era") == "newest"]
        pooled = next(x for x in count_frag if x["model_id"] == model and x["scope"] == "pooled")
        nd_correct = sum(bool(x["task_exact"]) for x in nd)
        nc_correct = sum(bool(x["task_exact"]) for x in nc)
        out.append({
            "model_id": model, "newest_decomp_n": len(nd), "newest_decomp_correct": nd_correct, "newest_decomp_exact_rate": rate(nd_correct, len(nd)),
            "newest_count_n": len(nc), "newest_count_correct": nc_correct, "newest_count_exact_rate": rate(nc_correct, len(nc)),
            "count_high_minus_low_failure_rate": pooled["high_minus_low_failure_rate"],
            "structural_hard_kill_model": rate(nd_correct, len(nd)) is not None and rate(nd_correct, len(nd)) <= 0.40 and rate(nc_correct, len(nc)) is not None and rate(nc_correct, len(nc)) <= 0.40,
        })
    return out


def trend_rows(copy_summary):
    out = []
    for model in [x["model_id"] for x in bp.MODELS]:
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
            "new_emoji_exact_monotone_nonincreasing": all(b <= a + 1e-12 for a, b in zip(rates, rates[1:])),
            "legacy_plus_new_exact_monotone_nonincreasing": all(b <= a + 1e-12 for a, b in zip(with_legacy, with_legacy[1:])),
            "cohort_index_failure_slope_pp_per_step": None if slope is None else 100 * slope,
            "ascii_exact_rate": bycohort.get("ASCII", {}).get("exact_rate"),
            "legacy_emoji_exact_rate": legacy.get("exact_rate") if legacy else None,
        })
    return out


def runtime_summary():
    rows = read_jsonl(RUNTIME_PATH)
    out = []
    for model in [x["model_id"] for x in bp.MODELS]:
        batches = [x for x in rows if x.get("model_id") == model and x.get("event") == "batch_completed"]
        loads = [x for x in rows if x.get("model_id") == model and x.get("event") == "model_loaded"]
        out.append({
            "model_id": model, "batches": len(batches), "probe_rows_completed": sum(x.get("probe_count", 0) for x in batches),
            "batch_elapsed_seconds": sum(x.get("elapsed_seconds", 0) for x in batches), "load_seconds": sum(x.get("load_seconds", 0) for x in loads),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in batches), "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in batches),
            "peak_memory_gb_max": max([x.get("peak_memory_gb") or 0 for x in batches] or [0]),
        })
    return out


def completeness_rows(scored):
    out = []
    for cfg in bp.MODELS:
        rows = [x for x in scored if x["model_id"] == cfg["model_id"]]
        counts = Counter(x["probe_type"] for x in rows)
        out.append({"model_id": cfg["model_id"], "rows": len(rows), "expected_rows": 290, **counts, "complete": len(rows) == 290 and counts == Counter({"copy": 182, "count": 36, "decomposition": 48, "comparison": 24})})
    return out


def build_verdict(scored):
    gates = {}
    for model in [x["model_id"] for x in bp.MODELS]:
        rows = [x for x in scored if x["model_id"] == model]
        gates[model] = {"ascii": ascii_gate(rows), "wrapper": wrapper_gate(rows)}
    mutations = mutation_checks()
    scanmap = copy_scan_metrics()
    copy_summary = summarize_copy(scored, scanmap)
    newest = newest_old_copy(scored)
    systematic = systematic_counts(scored)
    structural, count_frag = summarize_structural(scored)
    structural_kill = structural_kill_rows(scored, count_frag)
    trends = trend_rows(copy_summary)
    complete = completeness_rows(scored)
    valid_models = [model for model, gate in gates.items() if gate["ascii"]["pass"] and gate["wrapper"]["pass"]]
    invalid_models = [model for model in gates if model not in valid_models]
    eval_newest = [x for x in newest if x["model_id"] in valid_models]
    eval_structural_kill = [x for x in structural_kill if x["model_id"] in valid_models]
    copy_hard_kill_models = [x["model_id"] for x in eval_newest if x["hard_kill_copy_prong"]]
    structural_hard_kill_models = [x["model_id"] for x in eval_structural_kill if x["structural_hard_kill_model"]]
    copy_hard_kill = len(copy_hard_kill_models) >= 2
    structural_hard_kill = len(structural_hard_kill_models) >= 2
    rule1_models = [x["model_id"] for x in eval_newest if x["proceed_rule1_model_signal"]]
    pooled_frag = [x for x in count_frag if x["scope"] == "pooled" and x["model_id"] in valid_models]
    rule2_models = [x["model_id"] for x in pooled_frag if x["proceed_rule2_model_signal"]]
    rule3_models = [x["model_id"] for x in systematic if x["model_id"] in valid_models and x["any_systematic_class_at_least_5"]]
    scan_verdict = json.loads((RESULTS / "scan_verdict.json").read_text()) if (RESULTS / "scan_verdict.json").exists() else {}
    rule4 = bool(scan_verdict.get("proceed_rule4_fires"))
    verdict = {
        "completeness": complete, "all_models_complete": all(x["complete"] for x in complete),
        "validity_gates": gates, "gate_mutation_checks": mutations,
        "all_gate_mutation_checks_pass": all(v for k, v in mutations.items() if k.endswith("detects_failure") or k == "duplicate_wrapper_mutation_detected"),
        "ascii_gates_pass": all(x["ascii"]["pass"] for x in gates.values()),
        "wrapper_gates_pass": all(x["wrapper"]["pass"] for x in gates.values()),
        "valid_models": valid_models, "invalid_models": invalid_models, "valid_model_count": len(valid_models), "behavioral_pilot_conclusive": len(valid_models) >= 2,
        "copy_hard_kill_models": copy_hard_kill_models, "structural_hard_kill_models": structural_hard_kill_models,
        "copy_summary": copy_summary, "newest_old_copy": newest, "systematic_error_counts": systematic,
        "structural_summary": structural, "count_fragmentation": count_frag, "structural_hard_kill_rows": structural_kill,
        "cohort_trends": trends,
        "hard_kill_operational_arm": copy_hard_kill, "hard_kill_structural_arm": structural_hard_kill,
        "whole_project_hard_kill": copy_hard_kill and structural_hard_kill,
        "behavioral_proceed_rule1_fires": len(rule1_models) >= 2, "behavioral_proceed_rule1_models": rule1_models,
        "behavioral_proceed_rule2_fires": len(rule2_models) >= 2, "behavioral_proceed_rule2_models": rule2_models,
        "behavioral_proceed_rule3_fires": bool(rule3_models), "behavioral_proceed_rule3_models": rule3_models,
        "tokenizer_scan_proceed_rule4_fires": rule4,
        "overall_proceed_rule_fires": len(rule1_models) >= 2 or len(rule2_models) >= 2 or bool(rule3_models) or rule4,
        "behavioral_only_proceed_rule_fires": len(rule1_models) >= 2 or len(rule2_models) >= 2 or bool(rule3_models),
        "final_behavioral_interpretation": "inconclusive_validity_gate_failure" if len(valid_models) < 2 else ("proceed" if (len(rule1_models) >= 2 or len(rule2_models) >= 2 or bool(rule3_models)) and not copy_hard_kill and not structural_hard_kill else "no_behavioral_proceed_signal"),
        "regression_stop_rule": "not needed for pilot decision because tokenizer-scan rule 4 already fires" if rule4 else "not evaluated in descriptive pilot analysis",
    }
    return verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, mutations, complete


def write_outputs(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends):
    write_csv(BEHAVIORAL / "copy_cohort_summary.csv", copy_summary, ["model_id", "cohort", "n", "exact_count", "exact_rate", "failure_rate", "format_failures", "format_failure_rate", "content_failures_given_format", "content_failure_rate_given_format", "mean_tokens_per_cluster", "mean_split_codepoint_count", "tokenizer_roundtrip_exact_rate"])
    write_csv(BEHAVIORAL / "copy_newest_old_summary.csv", newest, list(newest[0]))
    write_csv(BEHAVIORAL / "copy_systematic_error_counts.csv", systematic, list(systematic[0]))
    write_csv(BEHAVIORAL / "structural_summary.csv", structural, ["model_id", "probe_type", "era", "fragmentation_group", "n", "exact_count", "exact_rate", "failure_rate", "format_failures", "format_failure_rate", "content_failure_rate_given_format"])
    write_csv(BEHAVIORAL / "count_fragmentation_summary.csv", count_frag, list(count_frag[0]))
    write_csv(BEHAVIORAL / "structural_hard_kill_summary.csv", structural_kill, list(structural_kill[0]))
    write_csv(BEHAVIORAL / "copy_error_taxonomy_summary.csv", taxonomy_summary(read_jsonl(SCORED_PATH)), ["model_id", "cohort", "n", "failures", "format_failures"] + TAXONOMY_CLASSES)
    write_csv(BEHAVIORAL / "runtime_summary.csv", runtime_summary(), ["model_id", "batches", "probe_rows_completed", "batch_elapsed_seconds", "load_seconds", "generation_tokens", "prompt_tokens", "peak_memory_gb_max"])
    (BEHAVIORAL / "behavioral_verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))


def model_manifest(runtime):
    return {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "models": bp.model_configs(), "skipped_models": bp.SKIPPED_MODELS,
        "prompt_templates": bp.PROMPT_TEMPLATES,
        "probe_content_discipline": "one user message per probe; no explicit system message; identical user content across models; each model's own chat template, including any template-injected default wrapper text; Qwen3 uses its native enable_thinking=false template switch",
        "decode_params": {"sampler": "greedy/argmax", "temperature": 0.0, "max_tokens": bp.MAX_TOKENS, "repetition_penalty": None},
        "inference_backend": "mlx-lm batch_generate on Apple Silicon; no vLLM",
        "host": {"python_version": platform.python_version(), "unicodedata_version": unicodedata.unidata_version, "platform": platform.platform(), "machine": platform.machine(), "normalization_unicodedata2_version": unicodedata2.unidata_version},
        "runtime": runtime,
        "tokenizer_linkage": json.loads((BEHAVIORAL / "tokenizer_linkage.json").read_text()) if (BEHAVIORAL / "tokenizer_linkage.json").exists() else None,
        "count_bundle_note": "count probes are tokenizer-specific; each behavioral model uses the 36 bundles constructed for its byte-identical scanned tokenizer artifact",
    }


def manual_failure_examples():
    scored = read_jsonl(SCORED_PATH)

    def row(model, probe_type, item_id=None):
        for x in scored:
            if x["model_id"] == model and x["probe_type"] == probe_type and (item_id is None or x.get("item_id") == item_id):
                return x
        return None

    specs = [
        ("TinyLlama ASCII echo", row("tinyllama-1.1b-chat-v1.0-bf16", "copy", "copy-159")),
        ("Qwen2.5 ASCII punctuation loss", row("qwen2.5-1.5b-instruct-bf16", "copy", "copy-159")),
        ("Qwen3 non-emoji newline padding", row("qwen3-1.7b-bf16", "copy", "copy-127")),
        ("Gemma missing copy wrapper", row("gemma3-1b-it-bf16-mirror", "copy", "copy-001")),
        ("Qwen2.5 decomposition hallucination", next((x for x in scored if x["model_id"] == "qwen2.5-1.5b-instruct-bf16" and x["probe_type"] == "decomposition"), None)),
    ]
    lines = []
    for label, x in specs:
        if x is None:
            continue
        target = x.get("target", x.get("a_text", ""))
        lines.append(f"- {label}: target={json.dumps(target, ensure_ascii=False)}; extracted/parsed={json.dumps(x.get('extracted_answer'), ensure_ascii=False)}; raw_prefix={json.dumps(x['raw_text'][:100], ensure_ascii=False)}.")
    return lines


def write_report(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, complete, runtime):
    lines = [
        "# Unicode release debt: behavioral pilot report", "",
        "Scope: deterministic local behavioral pilot only. Inputs are the frozen 182-item copy pilot and already-constructed structural probes. Inference used MLX `batch_generate` with greedy/argmax decoding, max 64 output tokens, no repetition penalty, and no vLLM. Raw responses are append-only in `results/behavioral/behavioral_raw.jsonl`.", "",
        "## Models and protocol deviations", "",
    ]
    mrows = []
    for cfg in bp.MODELS:
        rt = next((x for x in runtime if x["model_id"] == cfg["model_id"]), {})
        mrows.append([cfg["model_id"], cfg["hf_repo"] + "@" + cfg["revision"][:12], cfg["upstream_repo"] + "@" + cfg["upstream_revision"][:12], cfg["quantization"], rt.get("probe_rows_completed", 0), f"{rt.get('batch_elapsed_seconds', 0) / 60:.1f} min"])
    lines += [md_table(["model", "inference artifact", "upstream artifact", "dtype/quantization", "rows", "generation time"], mrows), ""]
    lines += [
        "- Llama 3.2: canonical `meta-llama/Llama-3.2-1B-Instruct` and 3B repositories are manual-license-gated; `config.json` download returned HTTP 403 for this account. Per instruction, Llama 3.2 was noted and skipped rather than substituted with a mirror. Consequently the behavioral pilot has four models and the originally expected 1,160 generations; the tokenizer scan was not extended with Llama 3.2 in this branch.",
        "- Unquantized bfloat16 safetensors were used. Qwen2.5 uses the 1.5B rather than 3B variant and Qwen3 uses 1.7B rather than 4B under the pilot compute/memory budget; exact repository revision SHAs are pinned.",
        "- Qwen2.5-1.5B and Qwen3-1.7B tokenizer files were verified byte-identical to the already-scanned Qwen2.5-0.5B and Qwen3-0.6B tokenizer artifacts respectively, so the existing tokenizer-specific count bundles and scan covariates apply directly.",
        "- Qwen3 generation used `enable_thinking=False`, a switch in its own chat template. No explicit system message was supplied; each model's own template was allowed to inject its default wrapper text (notably Qwen2.5's default system text). User message content was byte-identical across models.", "",
        "## Harness-validity gates", "",
    ]
    grows = []
    for model, g in gates.items():
        grows.append([model, f"{g['ascii']['correct']}/{g['ascii']['n']}", pct(g["ascii"]["accuracy"]), g["ascii"]["pass"], f"{g['wrapper']['format_failures']}/{g['wrapper']['n']}", pct(g["wrapper"]["format_failure_rate"]), g["wrapper"]["pass"]])
    lines += [md_table(["model", "ASCII exact", "ASCII rate", "ASCII gate", "wrapper failures", "wrapper rate", "wrapper gate"], grows), ""]
    mut = verdict["gate_mutation_checks"]
    lines += [f"Mutation controls: ASCII-gate mutation detected={mut['ascii_gate_mutation_detects_failure']}; wrapper-gate mutation detected={mut['wrapper_gate_mutation_detects_failure']}; duplicate-wrapper parser mutation detected={mut['duplicate_wrapper_mutation_detected']}. Completeness: " + "; ".join(f"{x['model_id']} {x['rows']}/{x['expected_rows']}" for x in complete) + ".", ""]
    lines += ["## Copy exact-match rates by cohort", ""]
    crows = []
    for model in [x["model_id"] for x in bp.MODELS]:
        row = [model]
        for cohort in COHORT_ORDER:
            hit = next((x for x in copy_summary if x["model_id"] == model and x["cohort"] == cohort), None)
            row.append("NA" if not hit else f"{hit['exact_count']}/{hit['n']} ({pct(hit['exact_rate'])})")
        crows.append(row)
    lines += [md_table(["model"] + COHORT_ORDER, crows), ""]
    lines += ["## Newest-bin versus old-copy controls", ""]
    nrows = [[x["model_id"], f"{x['newest_correct']}/{x['newest_n']}", pct(x["newest_exact_rate"]), f"{x['old_control_correct']}/{x['old_control_n']}", pct(x["old_control_exact_rate"]), pp(x["newest_minus_old_failure_pp"]), x["hard_kill_copy_prong"], x["proceed_rule1_model_signal"]] for x in newest]
    lines += [md_table(["model", "newest exact", "newest rate", "old exact", "old rate", "failure gap", "copy hard-kill prong", "rule-1 signal"], nrows), ""]
    lines += ["## Cohort-trend check", ""]
    trows = [[x["model_id"], ", ".join(f"{pct(y)}" for y in x["new_emoji_exact_rates"]), x["new_emoji_exact_monotone_nonincreasing"], x["legacy_plus_new_exact_monotone_nonincreasing"], "NA" if x["cohort_index_failure_slope_pp_per_step"] is None else f"{x['cohort_index_failure_slope_pp_per_step']:.1f}", pct(x["ascii_exact_rate"])] for x in trends]
    lines += [md_table(["model", "E15.1/E16/E17/E18 exact rates", "new-only monotone non-increasing", "legacy+new monotone", "failure slope pp/step", "ASCII exact"], trows), "",
        "Interpretation: positive slope means failure rises from E15.1 to E18. These raw cohort contrasts are descriptive and item-type-confounded because E15.1/E17 are mostly ZWJ families while E16/E18 are atomic; they are not the adjusted confirmatory regression.", ""]
    lines += ["## Structural probes", ""]
    srows = []
    for x in structural_kill:
        frag = next(y for y in count_frag if y["model_id"] == x["model_id"] and y["scope"] == "pooled")
        srows.append([x["model_id"], f"{x['newest_decomp_correct']}/{x['newest_decomp_n']}", pct(x["newest_decomp_exact_rate"]), f"{x['newest_count_correct']}/{x['newest_count_n']}", pct(x["newest_count_exact_rate"]), pp(frag["high_minus_low_failure_pp"]), x["structural_hard_kill_model"], frag["proceed_rule2_model_signal"]])
    lines += [md_table(["model", "newest decomp", "decomp rate", "newest count", "count rate", "high-low count failure gap", "structural hard-kill", "rule-2 signal"], srows), ""]
    detail_rows = []
    for x in structural:
        if x["probe_type"] in ("decomposition", "count", "comparison") and x["era"] in ("newest", "legacy") and (x["probe_type"] != "count" or x["fragmentation_group"] in ("low", "high")):
            detail_rows.append([x["model_id"], x["probe_type"], x["era"], x["fragmentation_group"], f"{x['exact_count']}/{x['n']}", pct(x["exact_rate"]), pct(x["format_failure_rate"])])
    lines += [md_table(["model", "probe", "era", "fragmentation", "exact", "rate", "format-failure rate"], detail_rows), ""]
    lines += ["## Systematic copy-error classes in newest items", ""]
    erows = [[x["model_id"]] + [x[c] for c in SYSTEMATIC_CLASSES] + [x["any_systematic_class_at_least_5"]] for x in systematic]
    lines += [md_table(["model"] + SYSTEMATIC_CLASSES + ["rule-3 signal"], erows), ""]
    lines += ["## Manual failure inspection", ""] + manual_failure_examples() + ["", "## Proceed/kill verdict under spec 8.3", ""]
    lines += [
        f"- Behavioral pilot conclusive: **{verdict['behavioral_pilot_conclusive']}** ({verdict['valid_model_count']} gate-valid models; at least two required). Final behavioral interpretation: **{verdict['final_behavioral_interpretation']}**.",
        f"- Operational hard kill among gate-valid models: **{verdict['hard_kill_operational_arm']}**. Structural hard kill among gate-valid models: **{verdict['hard_kill_structural_arm']}**. Whole-project hard kill: **{verdict['whole_project_hard_kill']}**.",
        f"- Behavioral proceed rule 1 (newest-minus-old copy failure >=20 pp in >=2 gate-valid models): **{verdict['behavioral_proceed_rule1_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule1_models']) or 'none'}.",
        f"- Behavioral proceed rule 2 (high-minus-low fragmentation structural failure >=15 pp in >=2 models): **{verdict['behavioral_proceed_rule2_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule2_models']) or 'none'}.",
        f"- Behavioral proceed rule 3 (systematic edit class in >=5 distinct newest items in any model): **{verdict['behavioral_proceed_rule3_fires']}**; models: {', '.join(verdict['behavioral_proceed_rule3_models']) or 'none'}.",
        f"- Tokenizer-scan proceed rule 4: **{verdict['tokenizer_scan_proceed_rule4_fires']}** (already established). Overall proceed-rule status: **{verdict['overall_proceed_rule_fires']}**.",
        f"- Regression-based stop rule: {verdict['regression_stop_rule']}.", "",
        "## Interpretation caveats", "",
        "- ASCII and wrapper gates determine whether cohort behavior is interpretable as Unicode-specific representation/content failure. Models failing either gate are reported but flagged rather than treated as clean evidence. TinyLlama's apparent wrapper pass is an extraction artifact: it often echoed the instruction, whose literal wrapper mention was then parsed as a wrapper; its ASCII exact score was nevertheless 0/24.",
        "- Copy attention can succeed even when structural operations fail; the copy and structural arms are therefore reported separately.",
        "- Small-model instruction following and batch execution can raise absolute failure rates but are uniform across cohorts. Format-conditional rates are in the CSV artifacts.",
        "- The regression-adjusted full-study analysis is not replaced by these pilot tables.",
        "", "Machine-readable outputs: `behavioral_manifest.json`, `probe_plan.jsonl`, `behavioral_raw.jsonl`, `behavioral_scored.jsonl`, `behavioral_verdict.json`, `copy_cohort_summary.csv`, `copy_newest_old_summary.csv`, `copy_error_taxonomy_summary.csv`, `structural_summary.csv`, `count_fragmentation_summary.csv`, and `runtime_summary.csv`.",
    ]
    (RESULTS / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n")


def freeze_behavioral_hashes():
    names = ["probe_plan.jsonl", "tokenizer_linkage.json", "behavioral_manifest.json", "behavioral_raw.jsonl", "behavioral_runtime.jsonl", "behavioral_scored.jsonl", "behavioral_verdict.json", "copy_cohort_summary.csv", "copy_newest_old_summary.csv", "copy_systematic_error_counts.csv", "copy_error_taxonomy_summary.csv", "structural_summary.csv", "count_fragmentation_summary.csv", "structural_hard_kill_summary.csv", "runtime_summary.csv"]
    rows = {}
    for name in names:
        path = BEHAVIORAL / name
        if path.exists():
            rows["behavioral/" + name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    pilot = RESULTS / "PILOT_REPORT.md"
    if pilot.exists():
        rows["PILOT_REPORT.md"] = {"sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(), "size": pilot.stat().st_size}
    code_paths = sorted(set((ucd.ROOT / "src").glob("*behavioral*.py")) | {ucd.ROOT / "src" / "run_behavioral.py", ucd.ROOT / "src" / "build_tokenizer_linkage.py", ucd.ROOT / "run_behavioral.sh"})
    rows["behavioral_code"] = {str(p.relative_to(ucd.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths if p.exists()}
    (BEHAVIORAL / "behavioral_artifact_hashes.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


def main():
    scored = read_jsonl(SCORED_PATH)
    if not scored:
        raise SystemExit("no scored rows; run run_behavioral.py then score_behavioral.py")
    verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, mutations, complete = build_verdict(scored)
    write_outputs(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends)
    runtime = runtime_summary()
    (BEHAVIORAL / "behavioral_manifest.json").write_text(json.dumps(model_manifest(runtime), indent=2, ensure_ascii=False))
    write_report(verdict, copy_summary, newest, systematic, structural, count_frag, structural_kill, trends, gates, complete, runtime)
    freeze_behavioral_hashes()
    print(json.dumps({
        "rows": len(scored), "all_models_complete": verdict["all_models_complete"],
        "ascii_gates_pass": verdict["ascii_gates_pass"], "wrapper_gates_pass": verdict["wrapper_gates_pass"],
        "hard_kill_operational_arm": verdict["hard_kill_operational_arm"], "hard_kill_structural_arm": verdict["hard_kill_structural_arm"],
        "behavioral_rule1": verdict["behavioral_proceed_rule1_fires"], "behavioral_rule2": verdict["behavioral_proceed_rule2_fires"], "behavioral_rule3": verdict["behavioral_proceed_rule3_fires"],
        "overall_proceed": verdict["overall_proceed_rule_fires"],
    }, indent=2))


if __name__ == "__main__":
    main()
