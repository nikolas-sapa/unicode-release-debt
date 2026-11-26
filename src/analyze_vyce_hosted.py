from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

SCORED_PATH = vc.VYCE / "vyce_scored.jsonl"
RAW_PATH = vc.VYCE / "vyce_raw.jsonl"
PLAN_PATH = vc.VYCE / "vyce_probe_plan.jsonl"
ANALYSIS_PATH = vc.VYCE / "vyce_analysis.json"
COST_PATH = vc.VYCE / "vyce_cost_summary.json"
ARTIFACTS_PATH = vc.VYCE / "vyce_relay_artifact_summary.json"
REPORT_PATH = Path("results/VYCE_ARM_REPORT.md")
EXPECTED_COPY = 64
EXPECTED_STRUCTURAL = 40
EXPECTED_TOTAL = 104


def rate(num, den):
    return num / den if den else None


def pct(x):
    return "NA" if x is None else f"{100 * x:.1f}%"


def pp(x):
    return "NA" if x is None else f"{x:.1f} pp"


def money(x):
    return f"${x:.6f}"


def md_table(headers, rows) -> str:
    def cell(x):
        if isinstance(x, float):
            x = f"{x:.3f}"
        if x is None:
            x = "NA"
        return str(x).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(x) for x in row) for row in rows]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_ids(scored: list[dict], raw: list[dict]) -> list[str]:
    order = vc.MODEL_PRIORITY
    present = sorted({x["model_id"] for x in scored} | {x["model_id"] for x in raw})
    return [x for x in order if x in present] + [x for x in present if x not in order]


def summarize_copy(scored: list[dict], models: list[str]) -> list[dict]:
    out = []
    cohorts = ["ASCII", "legacy emoji", "legacy non-emoji", "E15.1 emoji", "E16.0 emoji", "E17.0 emoji", "E18.0 emoji", "new non-emoji"]
    for model in models:
        for cohort in cohorts:
            rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("cohort") == cohort]
            if not rows:
                continue
            exact = sum(bool(x["task_exact"]) for x in rows)
            formatted = [x for x in rows if not x["format_failure"]]
            out.append({
                "model_id": model, "cohort": cohort, "n": len(rows), "exact_count": exact, "exact_rate": rate(exact, len(rows)),
                "format_failures": sum(bool(x["format_failure"]) for x in rows),
                "format_failure_rate": rate(sum(bool(x["format_failure"]) for x in rows), len(rows)),
                "content_failures_given_format": sum(bool(x["content_failure_given_format"]) for x in formatted),
                "content_failure_rate_given_format": rate(sum(bool(x["content_failure_given_format"]) for x in formatted), len(formatted)),
            })
    return out


def newest_old_copy(scored: list[dict], models: list[str]) -> list[dict]:
    out = []
    for model in models:
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy"]
        newest = [x for x in rows if x.get("copy_role") in ("new_emoji", "new_nonemoji")]
        old = [x for x in rows if x.get("copy_role") in ("old_emoji_control", "old_nonemoji_control")]
        ncorrect = sum(bool(x["task_exact"]) for x in newest)
        ocorrect = sum(bool(x["task_exact"]) for x in old)
        nfail = rate(len(newest) - ncorrect, len(newest))
        ofail = rate(len(old) - ocorrect, len(old))
        diff = nfail - ofail if nfail is not None and ofail is not None else None
        out.append({
            "model_id": model, "newest_n": len(newest), "newest_correct": ncorrect, "newest_exact_rate": rate(ncorrect, len(newest)),
            "old_control_n": len(old), "old_control_correct": ocorrect, "old_control_exact_rate": rate(ocorrect, len(old)),
            "newest_minus_old_failure_rate": diff, "newest_minus_old_failure_pp": None if diff is None else 100 * diff,
            "subsample_note": "28 newest and 12 legacy non-ASCII controls in the exploratory stratified subsample; not comparable in power to the 94/64 round-2 cells",
        })
    return out


def summarize_structural(scored: list[dict], models: list[str]) -> tuple[list[dict], list[dict]]:
    rows_out = []
    frag_out = []
    for model in models:
        rows = [x for x in scored if x["model_id"] == model and x["probe_type"] in ("count", "decomposition")]
        for typ in ("decomposition", "count"):
            subsets = defaultdict(list)
            for row in [x for x in rows if x["probe_type"] == typ]:
                subsets[(row.get("era"), row.get("fragmentation_group") if typ == "count" else "all")].append(row)
            for (era, frag), sub in sorted(subsets.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
                exact = sum(bool(x["task_exact"]) for x in sub)
                formatted = [x for x in sub if not x["format_failure"]]
                rows_out.append({
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
            frag_out.append(entry)
    return rows_out, frag_out


def gates(scored: list[dict], models: list[str]) -> list[dict]:
    out = []
    for model in models:
        ascii_rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
        copy_rows = [x for x in scored if x["model_id"] == model and x["probe_type"] == "copy"]
        ascii_correct = sum(bool(x["task_exact"]) for x in ascii_rows)
        wrapper_failures = sum(bool(x["format_failure"]) for x in copy_rows)
        wrapper_rate = rate(wrapper_failures, len(copy_rows))
        ascii_pass = len(ascii_rows) == 24 and ascii_correct >= 23
        wrapper_complete = len(copy_rows) == EXPECTED_COPY
        wrapper_pass = wrapper_complete and wrapper_rate is not None and wrapper_rate <= 0.05
        out.append({
            "model_id": model, "ascii_n": len(ascii_rows), "ascii_correct": ascii_correct, "ascii_rate": rate(ascii_correct, len(ascii_rows)), "ascii_gate_pass": ascii_pass,
            "copy_rows_scored": len(copy_rows), "copy_rows_expected": EXPECTED_COPY, "wrapper_failures": wrapper_failures,
            "wrapper_failure_rate": wrapper_rate, "wrapper_gate_pass": wrapper_pass, "gate_valid_exploratory_subsample": ascii_pass and wrapper_pass,
        })
    return out


def artifact_summary(raw: list[dict], availability_rows: list[dict], diagnostic_rows: list[dict], models: list[str]) -> dict:
    rows = []
    for model in models:
        subset = [x for x in raw if x["model_id"] == model]
        counter = Counter()
        for row in subset:
            counter.update(row.get("relay_artifacts", []))
        rows.append({
            "model_id": model,
            "raw_rows": len(subset),
            "ok_rows": sum(x.get("status") == "ok" for x in subset),
            "error_rows": sum(x.get("status") != "ok" for x in subset),
            "artifact_counts": dict(sorted(counter.items())),
            "finish_length_rows": sum(x.get("finish_reason") == "length" for x in subset),
            "usage_missing_rows": sum(bool(x.get("usage_missing")) for x in subset),
            "returned_model_mismatch_rows": sum(bool(x.get("api_model_returned")) and x.get("api_model_returned") != model for x in subset),
            "refusal_rows": sum(bool(x.get("refusal")) for x in subset),
            "reasoning_field_rows": sum(bool(x.get("reasoning_field_present")) for x in subset),
        })
    avail = []
    for row in availability_rows:
        slim = {k: row.get(k) for k in ("utc", "model", "http_status", "responds", "content", "finish_reason", "relay_artifacts", "network_error", "returned_model")}
        try:
            slim["error_message"] = (json.loads(row.get("response_body_text") or "{}").get("error") or {}).get("message")
        except Exception:
            slim["error_message"] = None
        avail.append(slim)
    diagnostics = []
    for row in diagnostic_rows:
        diagnostics.append({k: row.get(k) for k in ("utc", "purpose", "model", "endpoint", "http_status", "content", "finish_reason", "relay_artifacts", "returned_model", "note")})
    return {"per_model": rows, "availability_attempts": avail, "diagnostic_probes": diagnostics}


def cost_summary() -> dict:
    rows = [x for x in vc.read_jsonl(vc.LEDGER_PATH) if x.get("event") == "api_call"]
    by_model = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cap_cost_usd": 0.0})
    by_purpose = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0, "cap_cost_usd": 0.0})
    for row in rows:
        for key, group in ((row.get("model") or "non_completion_endpoint", by_model), (row.get("purpose") or "unknown", by_purpose)):
            group[key]["calls"] += 1
            group[key]["input_tokens"] += row.get("input_tokens") or 0
            group[key]["output_tokens"] += row.get("output_tokens") or 0
            group[key]["estimated_cost_usd"] += row.get("estimated_cost_usd") or 0.0
            group[key]["cap_cost_usd"] += row.get("cap_cost_usd") or 0.0
    totals = vc.ledger_totals(vc.LEDGER_PATH)
    out = {
        "generated_utc": vc.utcnow(),
        "hard_cap_usd": vc.HARD_CAP_USD,
        "total_api_calls": totals["api_calls"],
        "total_estimated_cost_usd": totals["estimated_cost_usd"],
        "total_conservative_cap_cost_usd": totals["cap_cost_usd"],
        "cap_remaining_usd": vc.HARD_CAP_USD - totals["cap_cost_usd"],
        "pricing_caveat": "vyce exposes no public price-schedule endpoint. Its /v1/me endpoint exposes key usage/balance metadata, but exact per-run prepaid-credit debit is only measurable through adjacent before/after snapshots; estimated cost uses external published proxy prices where available, while cap cost conservatively prices every model at $1/$3 per million input/output tokens.",
        "by_model": dict(sorted(by_model.items())),
        "by_purpose": dict(sorted(by_purpose.items())),
    }
    vc.write_json(COST_PATH, out)
    return out


def count_tokenizer_covariates() -> list[dict]:
    scan_path = vc.VYCE / "tokenizer_scan_extension.jsonl"
    if not scan_path.exists():
        return []
    plan = [x for x in vc.read_jsonl(PLAN_PATH) if x["probe_type"] == "count"]
    scan = vc.read_jsonl(scan_path)
    scanmap = {(x["tokenizer_id"], x["scan_id"]): x for x in scan}
    rows = []
    for tid in sorted({x["tokenizer_id"] for x in scan}):
        for frag in ("low", "high"):
            vals = []
            for probe in [x for x in plan if x["fragmentation_group"] == frag]:
                row = scanmap.get((tid, probe["probe_key"]))
                if row:
                    vals.append(row)
            if not vals:
                continue
            rows.append({
                "tokenizer_id": tid, "frozen_fragmentation_group": frag, "n": len(vals),
                "mean_tokens_per_cluster": sum(x["tokens_per_cluster"] for x in vals) / len(vals),
                "min_tokens_per_cluster": min(x["tokens_per_cluster"] for x in vals),
                "max_tokens_per_cluster": max(x["tokens_per_cluster"] for x in vals),
                "mean_split_codepoint_count": sum(x["split_codepoint_count"] or 0 for x in vals) / len(vals),
            })
    write_csv(vc.VYCE / "vyce_count_tokenizer_covariates.csv", rows, ["tokenizer_id", "frozen_fragmentation_group", "n", "mean_tokens_per_cluster", "min_tokens_per_cluster", "max_tokens_per_cluster", "mean_split_codepoint_count"])
    return rows


def tokenizer_summary() -> dict:
    path = vc.VYCE / "tokenizer_inventory_extension.json"
    if not path.exists():
        return {"ran": False}
    inv = json.loads(path.read_text())
    rows = []
    for tok in inv["tokenizers"]:
        e151 = tok["rgi_single_token_rate_by_emoji_version"].get("E15.1", {})
        e17 = tok["rgi_single_token_rate_by_emoji_version"].get("E17.0", {})
        rows.append({
            "tokenizer_id": tok["tokenizer_id"], "family": tok["family"], "algorithm": tok["algorithm"], "vocab_size": tok["vocab_size"],
            "newest_families": tok["pilot_newest_cohort_exactness"]["families"],
            "newest_pipeline_single_token_count": tok["pilot_newest_cohort_exactness"]["pipeline_single_token_count"],
            "newest_exact_vocab_entry_count": tok["pilot_newest_cohort_exactness"]["exact_vocab_entry_count"],
            "e151_zwj_mean_tokens_per_cluster": e151.get("zwj_mean_tokens_per_cluster"),
            "e17_zwj_mean_tokens_per_cluster": e17.get("zwj_mean_tokens_per_cluster"),
            "max_atomic_age": tok["max_atomic_age"],
        })
    return {"ran": True, "rows": rows, "continuity": inv.get("continuity", [])}


def relay_account_metadata() -> dict:
    first_path = vc.VYCE / "v1_me_sanitized.json"
    delta_path = vc.VYCE / "v1_me_usage_delta.json"
    snapshots_path = vc.VYCE / "v1_me_snapshots.jsonl"
    out = {"available": False}
    if first_path.exists():
        first = json.loads(first_path.read_text())
        sanitized = first.get("sanitized", {})
        out.update({
            "available": True,
            "verified": sanitized.get("verified"),
            "tier": sanitized.get("tier"),
            "rateLimit": sanitized.get("rateLimit"),
            "enabled": sanitized.get("enabled"),
            "response_body_sha256": first.get("response_body_sha256"),
        })
    if delta_path.exists():
        out["usage_delta"] = json.loads(delta_path.read_text())
    if snapshots_path.exists():
        rows = vc.read_jsonl(snapshots_path)
        out["snapshot_count"] = len(rows)
        if rows:
            out["latest_snapshot_sha256"] = rows[-1].get("response_body_sha256")
    return out


def analysis_payload() -> dict:
    scored = vc.read_jsonl(SCORED_PATH) if SCORED_PATH.exists() else []
    raw = vc.read_jsonl(RAW_PATH) if RAW_PATH.exists() else []
    availability = vc.read_jsonl(vc.VYCE / "availability_probe.jsonl") if (vc.VYCE / "availability_probe.jsonl").exists() else []
    diagnostics = vc.read_jsonl(vc.VYCE / "diagnostic_probe.jsonl") if (vc.VYCE / "diagnostic_probe.jsonl").exists() else []
    models = model_ids(scored, raw)
    copy_rows = summarize_copy(scored, models)
    newold = newest_old_copy(scored, models)
    structural, count_frag = summarize_structural(scored, models)
    gate_rows = gates(scored, models)
    artifacts = artifact_summary(raw, availability, diagnostics, models)
    cost = cost_summary()
    tok = tokenizer_summary()
    count_covariates = count_tokenizer_covariates()
    payload = {
        "generated_utc": vc.utcnow(),
        "arm_label": "EXPLORATORY hosted scale/diversity extension; not confirmatory and not a replacement for the local confirmatory scale-up",
        "expected_rows_per_model": EXPECTED_TOTAL,
        "models_with_any_raw_rows": models,
        "gates": gate_rows,
        "copy_cohort_summary": copy_rows,
        "copy_newest_old_summary": newold,
        "structural_summary": structural,
        "count_fragmentation_summary": count_frag,
        "relay_artifacts": artifacts,
        "cost_summary": cost,
        "relay_account_metadata": relay_account_metadata(),
        "harness_validation": json.loads((vc.VYCE / "vyce_harness_validation.json").read_text()) if (vc.VYCE / "vyce_harness_validation.json").exists() else {},
        "tokenizer_extension_summary": tok,
        "count_tokenizer_covariates": count_covariates,
        "scored_rows": len(scored),
        "raw_rows": len(raw),
    }
    vc.write_json(ANALYSIS_PATH, payload)
    vc.write_json(ARTIFACTS_PATH, artifacts)
    write_csv(vc.VYCE / "vyce_copy_cohort_summary.csv", copy_rows, ["model_id", "cohort", "n", "exact_count", "exact_rate", "format_failures", "format_failure_rate", "content_failures_given_format", "content_failure_rate_given_format"])
    write_csv(vc.VYCE / "vyce_copy_newest_old_summary.csv", newold, ["model_id", "newest_n", "newest_correct", "newest_exact_rate", "old_control_n", "old_control_correct", "old_control_exact_rate", "newest_minus_old_failure_pp"])
    write_csv(vc.VYCE / "vyce_structural_summary.csv", structural, ["model_id", "probe_type", "era", "fragmentation_group", "n", "exact_count", "exact_rate", "format_failures", "format_failure_rate", "content_failure_rate_given_format"])
    write_csv(vc.VYCE / "vyce_count_fragmentation_summary.csv", count_frag, ["model_id", "scope", "low_n", "low_exact", "low_failure_rate", "high_n", "high_exact", "high_failure_rate", "high_minus_low_failure_pp"])
    return payload


def find_cell(rows: list[dict], model: str, probe_type: str, era: str, frag: str = "all") -> dict | None:
    return next((x for x in rows if x["model_id"] == model and x["probe_type"] == probe_type and x["era"] == era and x["fragmentation_group"] == frag), None)


def latest_availability(availability: list[dict]) -> list[dict]:
    latest = {}
    for row in availability:
        latest[row["model"]] = row
    return sorted(latest.values(), key=lambda x: vc.MODEL_PRIORITY.index(x["model"]) if x["model"] in vc.MODEL_PRIORITY else 99)


def report_text(payload: dict) -> str:
    gates_rows = payload["gates"]
    newold = payload["copy_newest_old_summary"]
    countfrag = [x for x in payload["count_fragmentation_summary"] if x["scope"] == "pooled"]
    structural = payload["structural_summary"]
    availability = payload["relay_artifacts"]["availability_attempts"]
    cost = payload["cost_summary"]
    tok_rows = payload["tokenizer_extension_summary"].get("rows", []) if payload["tokenizer_extension_summary"].get("ran") else []
    gate_table = md_table(["model", "ASCII exact", "ASCII gate", "copy rows", "wrapper failures", "wrapper gate", "exploratory valid"], [[x["model_id"], f"{x['ascii_correct']}/{x['ascii_n']} ({pct(x['ascii_rate'])})", x["ascii_gate_pass"], f"{x['copy_rows_scored']}/{x['copy_rows_expected']}", f"{x['wrapper_failures']} ({pct(x['wrapper_failure_rate'])})", x["wrapper_gate_pass"], x["gate_valid_exploratory_subsample"]] for x in gates_rows])
    copy_table = md_table(["model", "newest exact", "legacy-control exact", "failure gap"], [[x["model_id"], f"{x['newest_correct']}/{x['newest_n']} ({pct(x['newest_exact_rate'])})", f"{x['old_control_correct']}/{x['old_control_n']} ({pct(x['old_control_exact_rate'])})", pp(x["newest_minus_old_failure_pp"])] for x in newold])
    count_table = md_table(["model", "low-frag failure", "high-frag failure", "high-minus-low gap"], [[x["model_id"], f"{x['low_n'] - x['low_exact']}/{x['low_n']} ({pct(x['low_failure_rate'])})", f"{x['high_n'] - x['high_exact']}/{x['high_n']} ({pct(x['high_failure_rate'])})", pp(x["high_minus_low_failure_pp"])] for x in countfrag])
    decomp_rows = []
    for model in [x["model_id"] for x in gates_rows]:
        newest = find_cell(structural, model, "decomposition", "newest")
        legacy = find_cell(structural, model, "decomposition", "legacy")
        decomp_rows.append([model, f"{newest['exact_count']}/{newest['n']} ({pct(newest['exact_rate'])})" if newest else "missing", f"{legacy['exact_count']}/{legacy['n']} ({pct(legacy['exact_rate'])})" if legacy else "missing"])
    decomp_table = md_table(["model", "newest decomposition exact", "legacy decomposition exact"], decomp_rows)
    latest_avail = latest_availability(availability)
    def avail_error(row: dict) -> str:
        return str(row.get("error_message") or "")[:160]
    avail_table = md_table(["model", "HTTP", "responds", "error", "artifacts"], [[x["model"], x["http_status"], x["responds"], avail_error(x), ", ".join(x.get("relay_artifacts") or []) or "none"] for x in latest_avail])
    cost_table = md_table(["model", "calls", "input tokens", "output tokens", "proxy-estimated cost", "conservative cap cost"], [[k, v["calls"], v["input_tokens"], v["output_tokens"], money(v["estimated_cost_usd"]), money(v["cap_cost_usd"])] for k, v in cost["by_model"].items()])
    tok_table = md_table(["tokenizer", "family", "vocab", "newest exact/pipeline single", "E15.1 ZWJ TPC", "E17 ZWJ TPC"], [[x["tokenizer_id"], x["family"], x["vocab_size"], f"{x['newest_exact_vocab_entry_count']}/{x['newest_pipeline_single_token_count']} of {x['newest_families']}", x["e151_zwj_mean_tokens_per_cluster"], x["e17_zwj_mean_tokens_per_cluster"]] for x in tok_rows])
    covariate_table = md_table(["tokenizer", "frozen stratum", "n", "mean TPC", "TPC range", "mean split CP"], [[x["tokenizer_id"], x["frozen_fragmentation_group"], x["n"], round(x["mean_tokens_per_cluster"], 3), f"{x['min_tokens_per_cluster']:.3f}-{x['max_tokens_per_cluster']:.3f}", round(x["mean_split_codepoint_count"], 3)] for x in payload.get("count_tokenizer_covariates", [])])
    artifact_rows = []
    for row in payload["relay_artifacts"]["per_model"]:
        artifact_rows.append([row["model_id"], row["raw_rows"], row["ok_rows"], row["error_rows"], row["finish_length_rows"], row["usage_missing_rows"], row["returned_model_mismatch_rows"], row["reasoning_field_rows"], json.dumps(row["artifact_counts"], ensure_ascii=False, sort_keys=True)])
    artifact_table = md_table(["model", "raw rows", "ok", "error", "finish=length", "usage missing", "model mismatch", "reasoning field", "artifact counts"], artifact_rows)
    working = [x["model"] for x in latest_avail if x.get("responds")]
    nonworking = [x["model"] for x in latest_avail if not x.get("responds")]
    diagnostics = payload["relay_artifacts"].get("diagnostic_probes", [])
    diagnostic_table = md_table(["purpose", "model", "HTTP", "content", "note"], [[x["purpose"], x["model"], x["http_status"], repr(x["content"]), x.get("note") or ""] for x in diagnostics])
    account = payload.get("relay_account_metadata", {})
    if account.get("available"):
        delta = (account.get("usage_delta") or {}).get("delta", {})
        target_delta = {m: delta.get(m) for m in vc.CANDIDATE_MODELS if m in delta}
        account_text = f"The relay's sanitized `/v1/me` metadata reports tier=`{account.get('tier')}`, verified=`{account.get('verified')}`, enabled=`{account.get('enabled')}`, rateLimit=`{account.get('rateLimit')}`. No account verification or tier change was attempted. The measured target-model usage delta after the first metadata snapshot was `{json.dumps(target_delta, sort_keys=True)}`."
    else:
        account_text = "No sanitized `/v1/me` relay account metadata snapshot is available."
    hv = payload.get("harness_validation", {})
    validation_table = md_table(["check", "pass", "detail"], [
        ["tiny-cap preflight abort", (hv.get("tiny_cap_batch_check") or {}).get("pass"), (hv.get("tiny_cap_batch_check") or {}).get("error_text") or ""],
        ["cost-ledger consistency", (hv.get("production_ledger_integrity") or {}).get("pass"), f"rows={(hv.get('production_ledger_integrity') or {}).get('rows')}, problems={len((hv.get('production_ledger_integrity') or {}).get('problems') or [])}"],
        ["synthetic scorer mutations", hv.get("extractor_mutation_checks_pass"), ""],
        ["corrupted-response negative control", (hv.get("corrupted_negative_control") or {}).get("negative_control_pass"), ""],
        ["live scored-response mutation", (hv.get("scored_response_mutation") or {}).get("pass"), (hv.get("scored_response_mutation") or {}).get("reason") or ""],
        ["overall harness validation", hv.get("harness_validation_pass"), "overall is false until a live scored byte-exact copy response exists"],
    ])
    if not payload["models_with_any_raw_rows"]:
        outcome = "No hosted battery rows were run because no target model passed the live availability probe. The scale hypothesis is therefore untested in this arm, not supported and not refuted."
    elif any(x["model_id"] == "gpt-oss-120b" and x["gate_valid_exploratory_subsample"] for x in gates_rows):
        g = next(x for x in newold if x["model_id"] == "gpt-oss-120b")
        outcome = f"gpt-oss-120b passed the exploratory gates. Its newest copy exactness was {g['newest_correct']}/{g['newest_n']} versus {g['old_control_correct']}/{g['old_control_n']} on legacy controls; inspect the structural table before drawing any scale conclusion."
    else:
        outcome = "Hosted rows exist, but no gate-valid gpt-oss-120b battery is available. The scale hypothesis remains unresolved in this exploratory arm."
    return f"""# Vyce hosted scale/diversity arm report

**Arm status: EXPLORATORY scale/diversity extension. This is not confirmatory evidence and does not replace the local confirmatory scale-up.**

This arm ports the frozen round-2 probe items, prompt wrappers, few-shot copy protocol, and byte-exact scoring to the OpenAI-compatible vyce relay. It uses a stratified 104-probe subsample per model: all 24 ASCII gate controls, 40 additional copy probes, and 40 structural probes (24 count + 16 decomposition). The count high/low labels are frozen round-2 strata measured under `qwen2.5-0.5b-instruct`; no hosted-endpoint tokenizer relabeling is assumed. Scoring remains byte-exact and deterministic. No normalization, stripping, repair, fuzzy matching, or model judging is used.

## Availability and relay access

Working target models at analysis time: {working or "none"}. Nonworking or unverified target models: {nonworking or "none"}.

{avail_table}

{account_text}

## Harness-validity gates

{gate_table}

## Copy arm: newest versus legacy controls

{copy_table}

## Structural arm: decomposition

{decomp_table}

## Structural arm: count fragmentation

{count_table}

## Relay artifacts and untrusted observability

The vyce relay is untrusted observability: hidden preprocessing, injected instructions, safety layers, upstream substitution, detokenization, caching, and endpoint drift cannot be excluded. Raw response bodies are preserved verbatim as base64 plus SHA-256 hashes before scoring. Format failures are therefore reported as measurement outcomes, not silently repaired. Usage tokens are relay-reported when present and conservatively upper-bounded when absent.

{artifact_table}

Diagnostic relay probes, including non-target infrastructure controls and alias checks, are first-class relay artifacts rather than scientific battery rows:

{diagnostic_table}

## Cost accounting

{ cost_table }

Total proxy-estimated cost: **{money(cost['total_estimated_cost_usd'])}**. Total conservative cap-accounted cost: **{money(cost['total_conservative_cap_cost_usd'])}** against the hard cap of **$8.000000**; remaining conservative headroom: **{money(cost['cap_remaining_usd'])}**. These are conservative accounting bounds, not observed debit. The relay's `/v1/me` usage meter showed zero target-model token/cost delta after the first metadata snapshot; because all target calls returned HTTP errors before generation and the non-target wave control showed no metered token/cost delta, observed incremental spend is consistent with **$0**. Exact debit for the earliest diagnostics cannot be isolated from the key's large pre-existing lifetime usage, so the conservative cap ledger remains the binding audit trail.

## Harness validation evidence

{validation_table}

## Tokenizer scan extension

Open tokenizer artifacts were pinned by exact Hugging Face revision and SHA-256, then scanned locally with the existing scanner. This scan is free of relay API cost.

{tok_table}

DeepSeek V4 Flash and V4.1 Flash tokenizer continuity: `{json.dumps(payload['tokenizer_extension_summary'].get('continuity', []), ensure_ascii=False)}`.

Frozen count-stratum covariates under the extension tokenizers (TPC = tokens per grapheme cluster; split CP = split codepoints):

{covariate_table}

## Interpretation

{outcome}

## Protocol and verification notes

- Copy prompts use the frozen `fewshot_copy_v5_multiturn_cueless` round-2 protocol, including the same four held-out ASCII exemplars as alternating user/assistant chat pairs. Structural probes use the same single-user prompt text. No system prompt is added.
- Temperature is 0.0 and maximum output is 64 tokens where the relay accepts them.
- The adaptation is frozen on ASCII controls first: a model requires at least 23/24 ASCII byte-exact copies before Unicode/structural spend. The wrapper gate uses the same <=5% failure-rate logic as round 2, applied to the 64-copy exploratory subsample rather than the full 182-copy battery.
- Raw rows are append-only JSONL. Each row stores the response body verbatim as base64, a SHA-256 body hash, usage, finish reason, latency, relay artifacts, cost ledger linkage fields, and the exact chat-message hash.
- Mutation controls verify that exact copies are accepted, byte/codepoint corruption is rejected, missing ZWJ/VS16 are classified, NFC/NFD alterations are rejected, newline padding is rejected, echoed prompts and extra text are rejected, and truncations are rejected.
- This subsample cannot reproduce the round-2 hard-kill cell sizes (94 newest copy items, 24 newest decomposition probes, 18 newest count probes). It is a scale/diversity screen only; inferential use is descriptive and exploratory.

## Honest limitations

- Hosted endpoints are black-box systems. Tokenizer scans of open artifacts do not prove that the relay serves those exact weights, tokenizers, preprocessing, or detokenization paths.
- `deepseek-v4-flash-lr` has no confidently matching public tokenizer artifact; DeepSeek V4/V4.1 scans are tokenizer-family evidence only.
- Relay-reported usage may include hidden prompt or reasoning tokens and cannot be independently audited.
- Small stratified cells are fragile: 28 newest copy items, 12 legacy copy controls, 12 low/high count rows pooled over eras, and 8 decomposition rows per era per model.
- Absence of a working endpoint is a relay availability result, not a model capability result.

Machine-readable outputs: `results/vyce_hosted/vyce_probe_plan.jsonl`, `vyce_selection_manifest.json`, `availability_probe.jsonl`, `cost_ledger.jsonl`, `vyce_raw.jsonl`, `vyce_scored.jsonl`, `vyce_analysis.json`, `vyce_cost_summary.json`, `vyce_scoring_validation.json`, `tokenizer_scan_extension.jsonl`, `tokenizer_inventory_extension.json`, and `vyce_artifact_hashes.json`.
"""


def write_report(payload: dict):
    REPORT_PATH.write_text(report_text(payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload = analysis_payload()
    write_report(payload)
    print(json.dumps({"analysis": str(ANALYSIS_PATH), "report": str(REPORT_PATH), "models": payload["models_with_any_raw_rows"], "cost": payload["cost_summary"]["total_conservative_cap_cost_usd"]}, indent=2))


if __name__ == "__main__":
    main()
