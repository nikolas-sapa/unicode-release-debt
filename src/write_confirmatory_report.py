from __future__ import annotations

import json
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


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


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


def main():
    analysis = load("confirmatory_analysis.json")
    verdict = load("confirmatory_verdict.json")
    gates = analysis["gates"]
    h1 = analysis["h1"]
    h2 = analysis["h2"]
    hh = analysis["h3_h4"]
    freeze = load("inventory_freeze.json")
    events = first_events()
    artifacts = load("artifact_hash_check.json")
    extractor = load("extractor_mutation_checks.json")
    mutation_impl = analysis["implementation_mutation_checks"]
    scoring = load("scoring_audit.json")
    completion = load("raw_completion_audit.json")
    lines = []
    a = lines.append
    a("# Confirmatory report: Unicode release debt or tokenizer fragmentation?")
    a("")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    a("")
    a("## Verdicts")
    a("")
    a("| Hypothesis | Verdict | Key result |")
    a("|---|---:|---|")
    for key, label in [("H1", "fragmentation dose-response"), ("H2", "era equivalence at matched fragmentation"), ("H3", "scalar-listing exact floor"), ("H4", "copy byte-exact ceiling")]:
        v = verdict["hypotheses"][key]
        if key == "H1":
            stat = f"beta={fmt(v['beta'])}, bootstrap CI={v['bootstrap_ci95']}, Holm p={fmt(v['holm_adjusted_p'])}"
        elif key == "H2":
            stat = f"Delta={fmt(v['delta'])}, bootstrap CI={v['bootstrap_ci95']}, Holm TOST p={fmt(v['holm_adjusted_tost_p'])}"
        else:
            stat = json.dumps({k: x for k, x in v.items() if k.startswith('every') or k == 'minimum_met'}, sort_keys=True)
        a(f"| {key}: {label} | **{v['verdict'].upper()}** | {stat} |")
    a("")
    a("## Freeze-order evidence")
    a("")
    a("These are the audit lines tying the frozen inventory to the first model output:")
    a("")
    a(f"VERIFY_REAL: inventory freeze artifact created_at_utc={freeze['created_at_utc']}")
    a(f"VERIFY_REAL: inventory freeze commit d06208a555d43ef42419d3387297350fe20838e2 committer_time={commit_time()}")
    a(f"VERIFY_REAL: first post-freeze memory smoke started={events['first_smoke_started']} (not an inventory probe and excluded from confirmatory_raw.jsonl)")
    a(f"VERIFY_REAL: first inventory probe batch completed={events['first_batch_completed']}")
    a(f"VERIFY_REAL: inventory freeze predates first inventory probe generation: {freeze['created_at_utc'] < events['first_batch_completed']}")
    a(f"VERIFY_REAL: frozen artifact hash verification pass={artifacts['artifact_hash_verification_pass']}")
    a(f"VERIFY_REAL: raw completion audit pass={completion['completion_audit_pass']}")
    a(f"VERIFY_REAL: extractor mutation checks pass={extractor['extractor_mutation_checks_pass']}")
    a(f"VERIFY_REAL: analysis implementation mutation checks pass={mutation_impl['implementation_mutation_checks_pass']}")
    a("")
    a("The candidate universe, tokenizer scan, inventory, probe plan, environment freeze, tokenizer freeze, design documents, and inventory-construction scripts were hash-pinned in commit `d06208a555d43ef42419d3387297350fe20838e2` before any confirmatory generation. Raw model responses were appended verbatim to `results/confirmatory/confirmatory_raw.jsonl`; extraction and scoring were implemented separately and run only through the checkpointed scorer.")
    a("")
    a("## Confirmatory scope and frozen judgment calls")
    a("")
    a("- Primary carriers: TinyLlama-1.1B bf16, Qwen2.5-3B bf16, Llama-3.2-3B MLX8, Gemma-3-4B MLX8 mirror, and Qwen3-4B MLX8. Mistral-7B MLX8 is descriptive robustness only, as resolved before freezing.")
    a("- Expanded bf16 comparator downloads were skipped before execution. MLX8 carriers are therefore not automatically excluded, but the missing direct bf16 quantization contrast is a frozen limitation and is not reinterpreted post hoc.")
    a("- ASCII structural gates are strict at 11/12 for count and scalar-listing; copy control is at least 23/24 exact with no more than six wrapper-format failures across all 124 copy rows.")
    a("- H1 and H2 form the only Holm-adjusted alpha family. H3 and H4 are exact replication tests. Everything beyond the pre-registered sensitivity list is descriptive.")
    a("")
    a("## Carrier validity")
    a("")
    a("| Carrier | Primary | ASCII copy exact | ASCII count exact | ASCII scalar exact | Copy valid | Structural valid | H1-H3 inclusion | H4 inclusion |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c, x in sorted(gates["carriers"].items()):
        a(f"| `{c}` | {x['primary_ladder']} | {x['ascii_copy_exact']}/24 | {x['ascii_count_exact']}/12 | {x['ascii_scalar_exact']}/12 | {x['copy_valid']} | {x['structural_valid']} | {x['include_h1h2h3_primary']} | {x['include_h4_primary']} |")
    a("")
    a(f"Minimum H1/H2 validity met: **{gates['minimum_h1h2_met']}**. Minimum H4 validity met: **{gates['minimum_h4_met']}**.")
    a("")
    a("## H1: fragmentation dose-response")
    a("")
    p = h1["pooled"]
    a(f"The primary pooled logistic fit used {p['n_rows']} Unicode count probes from {p['n_carriers']} structurally valid carriers. The design-matrix order is recorded in `confirmatory_analysis.json`; no covariate level was silently dropped. Conditional newest-L1 probes were excluded even where constructible, exactly as pre-registered.")
    a("")
    a(f"- `beta_frag`: **{fmt(p['point_beta'])}**")
    a(f"- odds ratio per ordinal fragmentation level: **{fmt(p['point_odds_ratio'])}**")
    a(f"- percentile 95% bootstrap CI: **{p['bootstrap_ci95']}**")
    a(f"- valid bootstrap replicates: **{p['bootstrap_valid']}/{p['bootstrap_replicates']}**; separation/singular: {p['bootstrap_separation_or_singular']}; nonconvergence: {p['bootstrap_nonconvergence']}")
    a(f"- count-based one-sided bootstrap p: {fmt(p['bootstrap_one_sided_p_beta_gt_0'])}")
    a(f"- normal-approximate one-sided bootstrap p: {fmt(p.get('bootstrap_normal_approx_one_sided_p_beta_gt_0'))}; bootstrap SE {fmt(p.get('bootstrap_se'))}")
    a(f"- Holm-adjusted p within H1/H2: {fmt(verdict['hypotheses']['H1']['holm_adjusted_p'])}")
    a(f"- nonnegative estimable carrier-specific coefficients: {h1['estimable_nonnegative_carrier_count']} across {h1['nonnegative_tokenizer_family_count']} tokenizer families")
    a("")
    a("| Carrier | beta | odds ratio | bootstrap valid | estimable |")
    a("|---|---:|---:|---:|---:|")
    for c, x in sorted(h1["carrier_specific"].items()):
        a(f"| `{c}` | {fmt(x.get('point_beta'))} | {fmt(x.get('point_odds_ratio'))} | {x.get('bootstrap_valid')} | {x.get('estimable')} |")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H1']['verdict'].upper()}**. The material-effect threshold is log(1.25)={math_log125():.6f}.")
    a("")
    a("## H2: era equivalence at matched fragmentation")
    a("")
    a(f"The overlap core contained {h2.get('pairs')} complete matched pairs after excluding incomplete pairs without replacement. `Delta` is newest minus legacy count failure, equally weighting levels within carriers and carriers overall.")
    a("")
    a(f"- Delta: **{fmt(h2.get('delta_newest_minus_legacy'))}**")
    a(f"- bootstrap SE: {fmt(h2.get('bootstrap_se'))}")
    a(f"- percentile 95% bootstrap CI: **{h2.get('bootstrap_ci95')}**")
    a(f"- CI width: {fmt(h2.get('bootstrap_ci_width'))}")
    a(f"- TOST component p-values: upper={fmt(h2.get('tost_p_upper'))}, lower={fmt(h2.get('tost_p_lower'))}, max={fmt(h2.get('tost_p'))}")
    a(f"- Holm-adjusted TOST p: {fmt(verdict['hypotheses']['H2']['holm_adjusted_tost_p'])}")
    a(f"- exact-zero sensitivity p (descriptive): {fmt(h2.get('exact_zero_difference_p'))}")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H2']['verdict'].upper()}** against the frozen ±10 percentage-point equivalence band.")
    a("")
    a("## H3 and H4: exact replication tests")
    a("")
    a("### H3 scalar-listing exact floor")
    a("")
    a("| Carrier | newest exact | legacy exact |")
    a("|---|---:|---:|")
    for c, x in sorted(hh["h3"]["by_carrier"].items()):
        a(f"| `{c}` | {x['newest']['exact']}/{x['newest']['n']} | {x['legacy']['exact']}/{x['legacy']['n']} |")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H3']['verdict'].upper()}**.")
    a("")
    a("### H4 copy byte-exact ceiling")
    a("")
    a("| Carrier | newest byte-exact | legacy byte-exact |")
    a("|---|---:|---:|")
    for c, x in sorted(hh["h4"]["by_carrier"].items()):
        a(f"| `{c}` | {x['newest']['exact']}/{x['newest']['n']} | {x['legacy']['exact']}/{x['legacy']['n']} |")
    a("")
    a(f"Verdict: **{verdict['hypotheses']['H4']['verdict'].upper()}**.")
    a("")
    a("## Pre-registered sensitivity analyses")
    a("")
    a("Sensitivity outputs are stored under `sensitivity_analyses` in `results/confirmatory/confirmatory_analysis.json` and do not overwrite the primary estimands. They include continuous tokens-per-cluster, level fixed effects, legacy-only fragmentation, within-item-type fits, era interaction, linear-probability, unadjusted breakdowns, exact-zero H2 testing, and the quantization-exclusion limitation. Because expanded comparator weight downloads were explicitly skipped before generation, the quantization sensitivity cannot estimate the originally intended direct bf16/MLX8 contrast; this is reported as a limitation rather than repaired post hoc.")
    a("")
    a("## Tables and figures")
    a("")
    a("Tables T1-T7 are in `results/confirmatory/tables/`; figures F1-F5 are in `results/confirmatory/figures/`. Figure interpretations are descriptive companions to the inferential statistics above, not additional hypothesis tests.")
    a("")
    a("## Mutation and integrity tests")
    a("")
    a(f"- Scorer extraction mutations: **{extractor['extractor_mutation_checks_pass']}** (`extractor_mutation_checks.json`)")
    a(f"- Artifact hash verification: **{artifacts['artifact_hash_verification_pass']}** (`artifact_hash_check.json`)")
    artifact_mutation = (OUT / 'artifact_hash_mutation_check.json')
    if artifact_mutation.exists():
        am = load('artifact_hash_mutation_check.json')
        a(f"- Corrupted-artifact detection mutation: **{am['mutation_test_pass']}** (`artifact_hash_mutation_check.json`)")
    raw_mutation = (OUT / 'raw_response_scoring_mutation_check.json')
    if raw_mutation.exists():
        rm = load('raw_response_scoring_mutation_check.json')
        a(f"- Corrupted raw-response scoring mutation: **{rm['mutation_test_pass']}** (`raw_response_scoring_mutation_check.json`)")
    a(f"- Logistic/TOST/Holm implementation mutation checks: **{mutation_impl['implementation_mutation_checks_pass']}**")
    a(f"- Raw completion audit: **{completion['completion_audit_pass']}**; raw SHA-256 before scoring `{completion.get('raw_sha256_before_scoring')}`")
    a("")
    a("## Limitations")
    a("")
    a("1. The host Python Unicode database is 15.0.0; Unicode 16/17 behavior is supplied by `unicodedata2` 18.0.0 and the explicit segmentation implementation. This host-library mismatch is a documented boundary condition, not hidden uncertainty.")
    a("2. Newest-L1 is structurally impossible for byte-fallback tokenizers. It is therefore excluded from primary H1/H2 even when Qwen/Llama-3.2 tokenizers make it constructible; boundary L1 analyses are sensitivity-only.")
    a("3. Direct bf16 quantization contrasts for MLX8 mirrors were not downloaded under the frozen user resolution. Quantization therefore cannot be ruled out as a narrow implementation-level contributor for MLX8 rows, but MLX8 carriers remain eligible exactly as resolved before generation.")
    a("4. The results concern small instruct/chat models under deterministic decoding and the frozen prompt formats. They do not estimate broader human-perceived Unicode support, API tokenizer behavior, or performance under sampling.")
    a("5. Mistral-7B is descriptive robustness only and never enters primary confirmatory verdicts.")
    a("")
    a("## Reproducibility")
    a("")
    a("Key artifacts:")
    a("")
    for name in ["environment_freeze.json", "tokenizer_freeze.json", "inventory_freeze.json", "inventory_audit.json", "probe_plan.jsonl", "confirmatory_raw.jsonl", "raw_completion_audit.json", "confirmatory_scored.jsonl", "carrier_gates.json", "confirmatory_analysis.json", "confirmatory_verdict.json"]:
        p = OUT / name
        if p.exists():
            a(f"- `results/confirmatory/{name}` sha256 `{cf.sha_file(p)}`")
    a("")
    a("Primary execution order was carrier-sequential with ASCII rows before Unicode rows for each carrier. Runtime, memory, prompt, revision, dtype, template, and batch metadata are in `confirmatory_runtime.jsonl` and `memory_smoke.jsonl`.")
    a("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


def math_log125():
    import math
    return math.log(1.25)


if __name__ == "__main__":
    main()
