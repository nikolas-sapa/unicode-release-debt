from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

OUT = cf.CONFIRMATORY
REPORT = cf.ROOT / "results" / "CONFIRMATORY_REPORT.md"


def load(name):
    return json.loads((OUT / name).read_text())


def optional_load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "NA"
        return f"{x:.{nd}f}"
    return str(x)


def pp(x, nd=2):
    if x is None:
        return "NA"
    return f"{100 * x:.{nd}f} pp"


def rate_text(stats):
    if not stats or stats.get("n") in (None, 0):
        return "NA"
    return f"{stats['exact']}/{stats['n']} ({100 * stats['rate']:.1f}%)"


def ci_text(ci, scale=1.0, nd=4):
    if not ci or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return "NA"
    return f"[{ci[0] * scale:.{nd}f}, {ci[1] * scale:.{nd}f}]"


def bool_text(x):
    return "pass" if x else "FAIL"


def first_events():
    runtime = [json.loads(x) for x in open(OUT / "confirmatory_runtime.jsonl")]
    smoke = [json.loads(x) for x in open(OUT / "memory_smoke.jsonl")]
    batches = sorted((x for x in runtime if x.get("event") == "batch_completed"), key=lambda x: x["utc"])
    return {
        "first_smoke_started": min(x["started_utc"] for x in smoke) if smoke else None,
        "first_batch_completed": batches[0]["utc"] if batches else None,
        "last_batch_completed": batches[-1]["utc"] if batches else None,
    }


def commit_time():
    return subprocess.check_output(["git", "show", "-s", "--format=%cI", "d06208a555d43ef42419d3387297350fe20838e2"], cwd=cf.ROOT, text=True).strip()


def hypothesis_stat(key, v):
    if key == "H1":
        if v.get("beta") is None:
            return "primary beta and bootstrap CI not estimable: minimum structural-validity gate unmet"
        return f"beta={fmt(v['beta'])}, OR={fmt(math.exp(v['beta']))}, bootstrap CI={v['bootstrap_ci95']}, Holm p={fmt(v['holm_adjusted_p'])}"
    if key == "H2":
        if v.get("delta") is None:
            return "primary Delta and equivalence interval not estimable: minimum structural-validity gate unmet"
        return f"Delta={pp(v['delta'])}, bootstrap CI={v['bootstrap_ci95']}, Holm TOST p={fmt(v['holm_adjusted_tost_p'])}"
    return json.dumps({k: x for k, x in v.items() if k.startswith("every") or k == "minimum_met"}, sort_keys=True)


def main():
    analysis = load("confirmatory_analysis.json")
    verdict = load("confirmatory_verdict.json")
    gates = analysis["gates"]
    h1 = analysis["h1"]
    h2 = analysis["h2"]
    hh = analysis["h3_h4"]
    exploratory = analysis.get("exploratory_descriptive", {}).get("carriers", {})
    freeze = load("inventory_freeze.json")
    events = first_events()
    artifacts = load("artifact_hash_check.json")
    extractor = load("extractor_mutation_checks.json")
    mutation_impl = analysis["implementation_mutation_checks"]
    scoring = load("scoring_audit.json")
    completion = load("raw_completion_audit.json")
    artifact_mutation = optional_load("artifact_hash_mutation_check.json")
    raw_mutation = optional_load("raw_response_scoring_mutation_check.json")
    resume_audit = optional_load("resume_no_regeneration_audit.json")
    reproducibility = optional_load("final_reproducibility_checks.json")
    scored = cf.read_jsonl(OUT / "confirmatory_scored.jsonl")

    lines = []
    a = lines.append
    a("# Confirmatory report: Unicode release debt or tokenizer fragmentation?")
    a("")
    a(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    a("")
    a("This report executes the frozen decision rules in `docs/CONFIRMATORY_PREREGISTRATION.md`. It does not loosen copy, structural-ASCII, tokenizer, or quantization gates after seeing outcomes. Descriptive observations are separated from confirmatory inference.")
    a("")
    a("## Frozen verdicts")
    a("")
    a("| Hypothesis | Verdict | Key result |")
    a("|---|---|---|")
    for key, label in [("H1", "fragmentation dose-response"), ("H2", "era equivalence at matched fragmentation"), ("H3", "scalar-listing exact floor"), ("H4", "copy byte-exact ceiling")]:
        v = verdict["hypotheses"][key]
        a(f"| {key}: {label} | **{v['verdict'].upper()}** | {hypothesis_stat(key, v)} |")
    a("")
    h2v = verdict["hypotheses"]["H2"]["verdict"]
    if h2v == "confirmed":
        a("**Pivot claim:** H2 era-equivalence within the frozen ±10 pp band is confirmed.")
    elif h2v == "refuted":
        a("**Pivot claim:** H2 era-equivalence within the frozen ±10 pp band is refuted.")
    else:
        a("**Pivot claim:** H2 era-equivalence is **not confirmed**. The verdict is inconclusive because the frozen minimum structural-validity gate was not met; this run cannot either confirm or refute the paper's pivot claim.")
    a("")
    a("## Freeze-order and completion evidence")
    a("")
    a(f"VERIFY_REAL: inventory freeze artifact created_utc={freeze['created_utc']}")
    a(f"VERIFY_REAL: inventory freeze commit d06208a555d43ef42419d3387297350fe20838e2 committer_time={commit_time()}")
    a(f"VERIFY_REAL: first post-freeze memory smoke started={events['first_smoke_started']} (infrastructure smoke only, excluded from confirmatory rows)")
    a(f"VERIFY_REAL: first inventory probe batch completed={events['first_batch_completed']}")
    a(f"VERIFY_REAL: last inventory probe batch completed={events['last_batch_completed']}")
    a(f"VERIFY_REAL: inventory freeze predates first inventory probe generation: {freeze['created_utc'] < events['first_batch_completed']}")
    a(f"VERIFY_REAL: frozen artifact hash verification pass={artifacts['artifact_hash_verification_pass']}")
    a(f"VERIFY_REAL: raw completion audit pass={completion['completion_audit_pass']}")
    a(f"VERIFY_REAL: extractor mutation checks pass={extractor['extractor_mutation_checks_pass']}")
    a(f"VERIFY_REAL: analysis implementation mutation checks pass={mutation_impl['implementation_mutation_checks_pass']}")
    if raw_mutation:
        a(f"VERIFY_REAL: raw-response corruption mutation detected={raw_mutation['mutation_test_pass']} before_exact={raw_mutation['before_exact']} after_exact={raw_mutation['after_exact']}")
    if artifact_mutation:
        a(f"VERIFY_REAL: corrupted-artifact hash mutation detected={artifact_mutation['mutation_test_pass']} target_error_detected={artifact_mutation['target_error_detected']}")
    if resume_audit:
        a(f"VERIFY_REAL: no completed row regenerated={resume_audit['no_regeneration_audit_pass']} rows_before_resume={resume_audit['baseline']['raw_rows']} rows_after_completion={resume_audit['final']['raw_rows']} duplicate_completion_keys={resume_audit['final']['duplicate_completion_keys']}")
        a(f"VERIFY_REAL: baseline prefix hash match={resume_audit['append_only_evidence']['baseline_prefix_hash_match']} current-startswith-gemma-checkpoint={resume_audit['append_only_evidence']['current_startswith_gemma_checkpoint_raw']}")
    a("")
    a("The candidate universe, tokenizer scan, inventory, probe plan, environment freeze, tokenizer freeze, design documents, and inventory-construction scripts were hash-pinned before confirmatory generation. Raw responses were appended to `results/confirmatory/confirmatory_raw.jsonl`; scoring was run only after completion.")
    a("")
    a("## Execution completion and resume audit")
    a("")
    a("| Carrier | Planned rows | Completed rows | Scope |")
    a("|---|---:|---:|---|")
    final_counts = (resume_audit or {}).get("final", {}).get("counts", {})
    for cfg in cf.carrier_configs(include_descriptive=True):
        c = cfg["carrier_id"]
        planned = len([x for x in cf.read_jsonl(OUT / "probe_plan.jsonl") if x["carrier_id"] == c])
        done = final_counts.get(c, len([x for x in scored if x["carrier_id"] == c]))
        scope = "primary" if cfg["primary_ladder"] else "descriptive only"
        a(f"| `{c}` | {planned} | {done} | {scope} |")
    a("")
    if resume_audit:
        a(f"Resume audit: `{resume_audit['no_regeneration_audit_pass']}`. The pre-resume baseline contained {resume_audit['baseline']['raw_rows']} rows; the final raw file contains {resume_audit['final']['raw_rows']}. Actual increments were {json.dumps(resume_audit['increments'], sort_keys=True)}. Completion-key duplicates: {resume_audit['final']['duplicate_completion_keys']}; carrier/probe-key duplicates: {resume_audit['final']['duplicate_carrier_probe_keys']}.")
        a("")
    a("## Frozen scope, resolutions, and execution deviations")
    a("")
    a("- Mistral-7B was demoted to descriptive robustness only in the pre-generation frozen judgment resolutions. It never enters H1-H4 primary carrier minima or verdicts.")
    a("- Expanded bf16 comparator downloads were skipped before generation under the same frozen resolution. MLX8 carriers are therefore not automatically excluded, but the intended direct quantization contrast is absent and remains a limitation.")
    a("- The pre-inference resolutions otherwise kept the strict gates unchanged: ASCII copy at least 23/24, wrapper failures at most 6/124, ASCII count at least 11/12, ASCII scalar listing at least 11/12.")
    a("- During the interrupted prior execution, a Qwen3 load overlapped the tail of Gemma generation, contrary to the intended sequential-loading operations rule. That Qwen3 batch failed before writing any Qwen3 rows. Existing Gemma checkpoint rows were retained under the append-only no-regeneration rule; all work after the interruption was strictly carrier-sequential.")
    a("- After that memory error, Qwen3 was rerun at batch size 8 rather than 16, following the predefined downward batch ladder. Batch size is scheduling metadata, not a semantic decode parameter.")
    a("- Before final analysis, an implementation-only draft CI-width criterion was removed from the H2 verdict code because it was not part of the frozen ±10 pp decision rule. This correction cannot affect the final classification: the minimum structural-validity gate was already unmet.")
    a("")
    a("## T1. Carrier validity and quantization gates")
    a("")
    a("| Carrier | Primary | ASCII copy | Wrapper failures | ASCII count | ASCII scalar | Copy valid | Structural valid | Quantization status | H1-H3 | H4 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for cfg in cf.carrier_configs(include_descriptive=True):
        c = cfg["carrier_id"]
        x = gates["carriers"][c]
        a(f"| `{c}` | {x['primary_ladder']} | {x['ascii_copy_exact']}/24 | {x['copy_wrapper_format_failures']}/124 | {x['ascii_count_exact']}/12 | {x['ascii_scalar_exact']}/12 | {x['copy_valid']} | {x['structural_valid']} | {x['quantization_status']} | {x['include_h1h2h3_primary']} | {x['include_h4_primary']} |")
    a("")
    a(f"Minimum H1/H2 carrier gate met: **{gates['minimum_h1h2_met']}** ({gates['structural_valid_primary_count']} structurally valid primary carriers across {gates['structural_tokenizer_family_count']} tokenizer families). Minimum H4 gate met: **{gates['minimum_h4_met']}** ({gates['copy_valid_primary_count']} copy-valid primary carriers across {gates['copy_tokenizer_family_count']} tokenizer families).")
    a("")
    a("Gate outcomes are data. In particular, Llama-3.2 fails the copy-wrapper gate with 16 format failures, and Gemma fails the strict structural-ASCII gate despite passing copy validity. No threshold was adjusted.")
    a("")
    a("## H1. Fragmentation dose-response")
    a("")
    p = h1["pooled"]
    a(f"Primary pooled estimand: `beta_frag` for ordinal fragmentation level in the covariate-adjusted logistic model. Primary rows available to the confirmatory fit: {p['n_rows']} from {p['n_carriers']} structurally valid carriers.")
    a("")
    a(f"- beta: **{fmt(p.get('point_beta'))}**")
    a(f"- odds ratio per level: **{fmt(p.get('point_odds_ratio'))}**")
    a(f"- percentile 95% bootstrap CI: **{p.get('bootstrap_ci95')}**")
    a(f"- valid bootstrap replicates: **{p.get('bootstrap_valid')}/{p.get('bootstrap_replicates')}**")
    a(f"- count-based one-sided bootstrap p: {fmt(p.get('bootstrap_one_sided_p_beta_gt_0'))}")
    a(f"- normal-approximate one-sided bootstrap p: {fmt(p.get('bootstrap_normal_approx_one_sided_p_beta_gt_0'))}")
    a(f"- Holm-adjusted p: {fmt(verdict['hypotheses']['H1'].get('holm_adjusted_p'))}")
    a(f"- nonnegative carrier-specific coefficients: {h1['estimable_nonnegative_carrier_count']} across {h1['nonnegative_tokenizer_family_count']} tokenizer families")
    a("")
    if p.get("insufficient_before_estimation"):
        a(f"Primary estimation was not attempted because the frozen minimum was unmet: {p['insufficient_before_estimation']}")
        a("")
    a(f"Verdict: **{verdict['hypotheses']['H1']['verdict'].upper()}**. This is a gate-driven inconclusive result, not evidence against fragmentation.")
    a("")
    a("## H2. Era equivalence at matched fragmentation")
    a("")
    a(f"Complete matched pairs in the primary population: {h2.get('pairs')}. Incomplete pair groups excluded without replacement: {h2.get('incomplete_overlap_group_count')}.")
    a("")
    a(f"- Delta (newest minus legacy failure): **{fmt(h2.get('delta_newest_minus_legacy'))}**")
    a(f"- percentile 95% bootstrap CI: **{h2.get('bootstrap_ci95')}**")
    a(f"- bootstrap SE: {fmt(h2.get('bootstrap_se'))}")
    a(f"- TOST component p-values: upper={fmt(h2.get('tost_p_upper'))}, lower={fmt(h2.get('tost_p_lower'))}, max={fmt(h2.get('tost_p'))}")
    a(f"- Holm-adjusted TOST p: {fmt(verdict['hypotheses']['H2'].get('holm_adjusted_tost_p'))}")
    a(f"- exact-zero sensitivity p (descriptive): {fmt(h2.get('exact_zero_difference_p'))}")
    a(f"- raw bootstrap tail proportions (diagnostic, not primary): <=-10 pp {fmt(h2.get('bootstrap_tail_lower_prop_le_minus_10pp'))}, >=+10 pp {fmt(h2.get('bootstrap_tail_upper_prop_ge_plus_10pp'))}")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H2']['verdict'].upper()}** against the frozen ±10 percentage-point band. The era-equivalence pivot is therefore not confirmed by this run.")
    a("")
    a("## H3. Scalar-listing exact floor")
    a("")
    a("| Eligible carrier | newest exact | legacy exact |")
    a("|---|---:|---:|")
    for c, x in sorted(hh["h3"]["by_carrier"].items()):
        a(f"| `{c}` | {x['newest']['exact']}/{x['newest']['n']} | {x['legacy']['exact']}/{x['legacy']['n']} |")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H3']['verdict'].upper()}**. The cross-carrier exact-floor hypothesis is inconclusive when fewer than four structurally valid quantization-eligible primary carriers exist.")
    a("")
    a("## H4. Copy byte-exact ceiling")
    a("")
    a("| Copy-valid carrier | newest byte-exact | legacy byte-exact |")
    a("|---|---:|---:|")
    for c, x in sorted(hh["h4"]["by_carrier"].items()):
        a(f"| `{c}` | {x['newest']['exact']}/{x['newest']['n']} | {x['legacy']['exact']}/{x['legacy']['n']} |")
    a("")
    copyvalid = set(hh["h4"]["eligible_carriers"])
    failures = [r for r in scored if r["carrier_id"] in copyvalid and r["probe_type"] == "copy" and r["probe_scope"] == "unicode" and not r["task_exact"]]
    if failures:
        a("Byte-level failures inside copy-valid primary carriers:")
        a("")
        a("| Carrier | Probe | Class | First byte offset | Target hex context | Output hex context |")
        a("|---|---|---|---:|---|---|")
        for r in failures[:20]:
            a(f"| `{r['carrier_id']}` | `{r['probe_key']}` | `{r['copy_near_miss_class']}` | {fmt(r.get('first_byte_mismatch_offset'))} | `{r.get('target_diff_hex_context')}` | `{r.get('output_diff_hex_context')}` |")
        a("")
    a(f"Verdict: **{verdict['hypotheses']['H4']['verdict'].upper()}** under the exact-ceiling rule. Any byte-level failure in a copy-valid eligible carrier refutes the exact universal form, even when the failure is isolated.")
    a("")
    a("## Required tables and figures")
    a("")
    a("Primary CSV artifacts in `results/confirmatory/tables/`:")
    a("")
    for name in ["T1_carrier_validity_quantization_gates.csv", "T2_inventory_cells.csv", "T3_h1_coefficients.csv", "T4_h2_matched_pairs.csv", "T5_copy_ceiling.csv", "T6_scalar_floor.csv", "T7_count_errors.csv"]:
        a(f"- `{name}`")
    a("")
    a("Primary figures in `results/confirmatory/figures/`:")
    a("")
    for name in ["F1_failure_heatmap.png", "F2_fragmentation_dose_response.png", "F3_era_effect_forest.png", "F4_copy_count_scalar_triad.png", "F5_copy_near_miss_taxonomy.png"]:
        a(f"- `{name}`")
    a("")
    a("T2 prints requested and achieved inventory cells, including structurally empty newest-L1 cells and item-type substitutions. Because no carrier entered the primary structural population, H1/H2 adjusted curves and pooled forest estimates are marked not estimable rather than visually invented.")
    a("")
    a("## Pre-registered sensitivity analyses")
    a("")
    a("Sensitivity outputs are stored under `sensitivity_analyses` in `results/confirmatory/confirmatory_analysis.json`. The planned sensitivities include continuous tokens-per-cluster, level fixed effects, legacy-only fits, within-type fits, era interaction, linear probability, pair-mismatch regression, logistic-era average marginal effect, exact-zero testing, and quantization exclusion. With zero structurally valid primary carriers, the structural sensitivities have no confirmatory population to estimate. The quantization-exclusion sensitivity is additionally limited by the pre-generation decision not to download bf16 comparator weights; it cannot estimate the originally intended direct quantization contrast.")
    a("")
    a("## Exploratory observations, not confirmation")
    a("")
    a("**Everything in this section is descriptive.** Every primary carrier failed at least one strict structural-ASCII validity gate, so these numbers must not be reinterpreted as confirmatory structural evidence.")
    a("")
    a("| Carrier | Unicode copy exact | Overlap count exact | Unicode scalar exact | Exploratory matched Delta [bootstrap CI] | Exploratory fragmentation beta [normal CI] |")
    a("|---|---:|---:|---:|---:|---:|")
    for cfg in cf.carrier_configs(include_descriptive=True):
        c = cfg["carrier_id"]
        d = exploratory.get(c, {})
        pair_delta = d.get("exploratory_matched_pair_delta", {})
        delta = pair_delta.get("delta")
        delta_ci = ci_text(pair_delta.get("bootstrap_ci95"), scale=100, nd=2) if pair_delta.get("estimable") else "NA"
        fit = d.get("exploratory_logistic_fragmentation_fit", {})
        beta = fit.get("beta") if fit.get("estimable") else None
        beta_ci = ci_text(fit.get("normal_ci95"), nd=3) if fit.get("estimable") else "NA"
        delta_text = f"{pp(delta)} {delta_ci}" if delta is not None else "NA"
        beta_text = f"{fmt(beta)} {beta_ci}" if beta is not None else "NA"
        a(f"| `{c}` | {rate_text(d.get('copy_unicode_exact', {}))} | {rate_text(d.get('count_overlap_exact', {}))} | {rate_text(d.get('scalar_unicode_exact', {}))} | {delta_text} | {beta_text} |")
    a("")
    a("The descriptive pattern is strong byte-copy performance in several carriers alongside near-total scalar-listing failure and heterogeneous count performance. That pattern motivated the confirmatory design, but the frozen structural-gate failure means these observations remain hypothesis-generating here. Copy failures and near misses are classified in T5/F5; scalar error classes are in T6; count signed errors are in T7.")
    a("")
    a("## Mutation and integrity tests")
    a("")
    a(f"- Scorer extraction mutations: **{extractor['extractor_mutation_checks_pass']}** (`extractor_mutation_checks.json`)")
    a(f"- Frozen artifact hash verification: **{artifacts['artifact_hash_verification_pass']}** (`artifact_hash_check.json`)")
    if artifact_mutation:
        a(f"- Corrupted-artifact detection mutation: **{artifact_mutation['mutation_test_pass']}**; one byte was flipped in a temporary copy and hash verification reported `{artifact_mutation['target_error'][0]['error'] if artifact_mutation.get('target_error') else 'NA'}`")
    if raw_mutation:
        a(f"- Corrupted raw-response scoring mutation: **{raw_mutation['mutation_test_pass']}**; one output byte corruption changed task_exact from {raw_mutation['before_exact']} to {raw_mutation['after_exact']} without touching the retained raw file")
    a(f"- Logistic/TOST/Holm implementation mutation checks: **{mutation_impl['implementation_mutation_checks_pass']}**")
    a(f"- Raw completion audit: **{completion['completion_audit_pass']}**; raw SHA-256 before scoring `{completion.get('raw_sha256_before_scoring')}`")
    if reproducibility:
        a(f"- Power-analysis validation and mutation self-test: **{reproducibility.get('power_self_test_pass')}**")
        a(f"- Power-analysis deterministic rerun match after removing timestamps: **{reproducibility.get('power_deterministic_output_match')}**")
        a(f"- Inventory audit/mutation reproducibility: **{reproducibility.get('inventory_audit_pass')}**")
        a(f"- Chat-template smoke/rendering checks: **{reproducibility.get('template_rendering_checks_pass')}**")
        a(f"- Inference-tokenizer hash verification: **{reproducibility.get('inference_tokenizer_verification_pass')}**")
    a("")
    a("## Limitations")
    a("")
    a("1. **No confirmatory structural population:** no primary carrier passed both strict ASCII structural gates while also passing copy validity. H1, H2, and H3 are therefore inconclusive by design, not negative evidence.")
    a("2. **Strict-gate tradeoff:** the ASCII gates intentionally exclude carriers that cannot reliably count short ASCII strings or print exact scalar notation. This protects interpretability but means most measured Unicode structural behavior is descriptive only.")
    a("3. **Quantization contrast missing:** direct bf16 comparator inference for MLX8 mirrors was not run under the frozen pre-generation resolution. Quantization cannot be ruled out as a contributor to MLX8 exactness differences.")
    a("4. **Sequential-loading deviation in interrupted work:** the prior agent briefly overlapped a Qwen3 load with Gemma generation. No Qwen3 row was written during that failed attempt, and all resumed work was sequential. Existing checkpoints were preserved exactly as required.")
    a("5. **Structural empty cells:** newest low-fragmentation complex/ZWJ cells are intrinsically impossible in frozen byte-fallback vocabularies. H2 uses only the L2-L4 overlap core.")
    a("6. **Prompt scope:** results apply to the frozen prompts, strict extraction rules, deterministic decoding, local MLX artifacts, and small instruct models tested here. They do not generalize automatically to other prompts, APIs, sampling, or larger models.")
    a("7. **Host Unicode stack:** host Python Unicode data is 15.0.0; Unicode 16/17/18 behavior comes from pinned `unicodedata2` 18.0.0 and the explicit segmentation implementation.")
    a("8. **Constituent dependence:** probe-level bootstrap does not fully model repeated constituent use across probes, a limitation accepted in the frozen judgment resolutions.")
    a("")
    a("## Reproducibility")
    a("")
    a("Key artifacts:")
    a("")
    for name in ["environment_freeze.json", "tokenizer_freeze.json", "inventory_freeze.json", "inventory_audit.json", "probe_plan.jsonl", "confirmatory_raw.jsonl", "raw_completion_audit.json", "confirmatory_scored.jsonl", "carrier_gates.json", "confirmatory_analysis.json", "confirmatory_verdict.json", "resume_no_regeneration_audit.json"]:
        p = OUT / name
        if p.exists():
            a(f"- `results/confirmatory/{name}` sha256 `{cf.sha_file(p)}`")
    a("")
    a(f"Scored rows: {scoring['scored_rows']}; raw rows: {scoring['raw_rows']}. Runtime, memory, prompt, revision, dtype, template, batch, and finish metadata are in `confirmatory_runtime.jsonl` and `memory_smoke.jsonl`.")
    a("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
