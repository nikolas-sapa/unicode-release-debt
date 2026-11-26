from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "results" / "behavioral_round2" / "round2_scored.jsonl"
DEFAULT_OUT = ROOT / "results" / "confirmatory_power_analysis.json"

GATE_VALID_MODELS = [
    "qwen2.5-3b-instruct-bf16",
    "qwen3-4b-instruct-2507-mlx8",
    "gemma3-4b-it-mlx8-mirror",
    "tinyllama-1.1b-chat-v1.0-bf16",
]

NEW_COPY_ROLES = {"new_emoji", "new_nonemoji"}
OLD_COPY_ROLES = {"old_emoji_control", "old_nonemoji_control"}
Z_BONFERRONI_025 = NormalDist().inv_cdf(0.975)

# Failure probabilities by ordinal fragmentation level L1..L4.  The moderate
# scenario is constructed so mean failure in L1/L2 is exactly 12/18 and mean
# failure in L3/L4 is exactly 16/18, matching the conservative 22.2 pp
# high-minus-low pilot gap.  The strong scenario is the 50 pp qwen2.5 pilot
# pattern with the upper boundary capped at 0.97 to avoid complete separation.
FRAGMENTATION_SCENARIOS = {
    "pilot_moderate_22p2pp": [11 / 18, 13 / 18, 15 / 18, 17 / 18],
    "pilot_strong_50pp_capped": [0.375, 0.625, 0.875, 0.97],
    "flat_null": [0.70, 0.70, 0.70, 0.70],
}

MEASURED_SECONDS_PER_PROBE = {
    "tinyllama-1.1b-chat-v1.0-bf16": 114.10850920609664 / 290,
    "qwen2.5-3b-instruct-bf16": 231.10806741693523 / 290,
    "qwen3-4b-instruct-2507-mlx8": 389.889050710015 / 290,
    "gemma3-4b-it-mlx8-mirror": 283.1208251664648 / 290,
    # Conservative proxy: round-2 qwen2.5-7b q8 was the slowest measured carrier.
    "mistral7b-instruct-v03-mlx8": 591.9522585838567 / 290,
    # Conservative proxy: Llama-3.2-3B q8 is assigned the qwen3-4b q8 rate.
    "llama32-3b-instruct-mlx8": 389.889050710015 / 290,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def failure_rate(rows: list[dict]) -> float:
    return 1.0 - sum(bool(x["task_exact"]) for x in rows) / len(rows)


def pilot_statistics(rows: list[dict]) -> dict:
    models = sorted({x["model_id"] for x in rows})
    per_model = Counter(x["model_id"] for x in rows)
    per_model_type = {
        model: Counter(x["probe_type"] for x in rows if x["model_id"] == model)
        for model in models
    }
    copy = {}
    decomp = {}
    count = {}
    for model in models:
        mrows = [x for x in rows if x["model_id"] == model]
        copy_rows = [x for x in mrows if x["probe_type"] == "copy"]
        newest = [x for x in copy_rows if x.get("copy_role") in NEW_COPY_ROLES]
        legacy = [x for x in copy_rows if x.get("copy_role") in OLD_COPY_ROLES]
        copy[model] = {
            "newest_exact": sum(bool(x["task_exact"]) for x in newest),
            "newest_n": len(newest),
            "legacy_exact": sum(bool(x["task_exact"]) for x in legacy),
            "legacy_n": len(legacy),
            "copy_format_failures": sum(bool(x["format_failure"]) for x in copy_rows),
            "copy_n": len(copy_rows),
        }
        drows = [x for x in mrows if x["probe_type"] == "decomposition"]
        decomp[model] = {
            "exact": sum(bool(x["task_exact"]) for x in drows),
            "n": len(drows),
        }
        crows = [x for x in mrows if x["probe_type"] == "count"]
        entry = {}
        for group in ("low", "high"):
            sub = [x for x in crows if x.get("fragmentation_group") == group]
            entry[f"{group}_exact"] = sum(bool(x["task_exact"]) for x in sub)
            entry[f"{group}_n"] = len(sub)
            entry[f"{group}_failure_rate"] = failure_rate(sub) if sub else None
        if entry["low_n"] and entry["high_n"]:
            entry["high_minus_low_failure_rate"] = entry["high_failure_rate"] - entry["low_failure_rate"]
            entry["high_minus_low_failure_pp"] = 100 * entry["high_minus_low_failure_rate"]
        count[model] = entry
    return {
        "row_count": len(rows),
        "model_count": len(models),
        "models": models,
        "rows_per_model": dict(sorted(per_model.items())),
        "probe_type_counts_per_model": {m: dict(c) for m, c in sorted(per_model_type.items())},
        "copy_exact_by_role": copy,
        "decomposition_exact": decomp,
        "count_fragmentation": count,
    }


def validate_pilot_statistics(stats: dict) -> dict:
    checks = []

    def check(name: str, actual, expected, pass_: bool):
        checks.append({"name": name, "actual": actual, "expected": expected, "pass": bool(pass_)})

    check("round2 row count", stats["row_count"], 1450, stats["row_count"] == 1450)
    check("round2 model count", stats["model_count"], 5, stats["model_count"] == 5)
    check("rows per model", stats["rows_per_model"], {m: 290 for m in stats["models"]}, all(v == 290 for v in stats["rows_per_model"].values()))
    expected_types = {"copy": 182, "decomposition": 48, "count": 36, "comparison": 24}
    type_ok = all(stats["probe_type_counts_per_model"][m] == expected_types for m in stats["models"])
    check("probe-type inventory per model", stats["probe_type_counts_per_model"], expected_types, type_ok)
    for model in GATE_VALID_MODELS:
        c = stats["copy_exact_by_role"][model]
        check(f"{model} newest copy exact", c["newest_exact"], 94, c["newest_exact"] == 94 and c["newest_n"] == 94)
        check(f"{model} legacy copy exact", c["legacy_exact"], 64, c["legacy_exact"] == 64 and c["legacy_n"] == 64)
        d = stats["decomposition_exact"][model]
        check(f"{model} decomposition floor", d, {"exact": 0, "n": 48}, d["exact"] == 0 and d["n"] == 48)
    q25_gap = stats["count_fragmentation"]["qwen2.5-3b-instruct-bf16"]["high_minus_low_failure_rate"]
    q3_gap = stats["count_fragmentation"]["qwen3-4b-instruct-2507-mlx8"]["high_minus_low_failure_rate"]
    gemma_gap = stats["count_fragmentation"]["gemma3-4b-it-mlx8-mirror"]["high_minus_low_failure_rate"]
    check("qwen2.5-3b count fragmentation failure gap", q25_gap, 0.5, abs(q25_gap - 0.5) < 1e-12)
    check("qwen3-4b count fragmentation failure gap", q3_gap, 2 / 9, abs(q3_gap - 2 / 9) < 1e-12)
    check("gemma3-4b count fragmentation failure gap", gemma_gap, 1 / 18, abs(gemma_gap - 1 / 18) < 1e-12)
    q7_copy = stats["copy_exact_by_role"]["qwen2.5-7b-instruct-mlx8"]
    q7_format_failures = q7_copy["copy_format_failures"]
    check("qwen2.5-7b copy wrapper failures", {"format_failures": q7_format_failures, "n": q7_copy["copy_n"]}, {"format_failures": 28, "n": 182}, q7_format_failures == 28 and q7_copy["copy_n"] == 182)
    tiny_count_exact = stats["count_fragmentation"]["tinyllama-1.1b-chat-v1.0-bf16"]["low_exact"] + stats["count_fragmentation"]["tinyllama-1.1b-chat-v1.0-bf16"]["high_exact"]
    tiny_count_n = stats["count_fragmentation"]["tinyllama-1.1b-chat-v1.0-bf16"]["low_n"] + stats["count_fragmentation"]["tinyllama-1.1b-chat-v1.0-bf16"]["high_n"]
    check("tinyllama count floor", {"exact": tiny_count_exact, "n": tiny_count_n}, {"exact": 0, "n": 36}, tiny_count_exact == 0 and tiny_count_n == 36)
    return {"all_pass": all(x["pass"] for x in checks), "checks": checks}


def mutation_check(rows: list[dict]) -> dict:
    stats = pilot_statistics(rows)
    baseline_gap = stats["count_fragmentation"]["qwen2.5-3b-instruct-bf16"]["high_minus_low_failure_rate"]
    mutated = []
    changed = False
    for row in rows:
        row = dict(row)
        if (
            not changed
            and row["model_id"] == "qwen2.5-3b-instruct-bf16"
            and row["probe_type"] == "count"
            and row.get("fragmentation_group") == "high"
            and not row["task_exact"]
        ):
            row["task_exact"] = True
            changed = True
        mutated.append(row)
    mutated_gap = pilot_statistics(mutated)["count_fragmentation"]["qwen2.5-3b-instruct-bf16"]["high_minus_low_failure_rate"]
    return {
        "mutation_description": "flip exactly one qwen2.5-3b high-fragmentation count failure to success",
        "baseline_gap": baseline_gap,
        "mutated_gap": mutated_gap,
        "mutation_detected": changed and abs(mutated_gap - baseline_gap) > 1e-12,
    }


def fit_logistic_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-5)
        grad = X.T @ (y - p)
        hess = (X.T * w) @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    w = np.maximum(p * (1 - p), 1e-5)
    hess = (X.T * w) @ X
    cov = np.linalg.pinv(hess)
    return beta, np.sqrt(np.maximum(np.diag(cov), 0))


def h1_design(carriers: int, per_cell: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for carrier in range(carriers):
        for era in (0, 1):
            levels = (1, 2, 3, 4) if era == 0 else (2, 3, 4)
            for level in levels:
                for j in range(per_cell):
                    rows.append((carrier, era, level, (j % 5) + 1))
    design = np.asarray(rows, dtype=float)
    cols = [np.ones(len(design)), design[:, 1], design[:, 2], design[:, 3] - 3]
    for carrier in range(1, carriers):
        cols.append((design[:, 0] == carrier).astype(float))
    return design, np.column_stack(cols)


def scenario_probabilities(name: str, carriers: int) -> list[list[float]]:
    if name != "heterogeneous_pilot_mix":
        probs = FRAGMENTATION_SCENARIOS[name]
        return [probs for _ in range(carriers)]
    strong = FRAGMENTATION_SCENARIOS["pilot_strong_50pp_capped"]
    moderate = FRAGMENTATION_SCENARIOS["pilot_moderate_22p2pp"]
    flat = FRAGMENTATION_SCENARIOS["flat_null"]
    out = []
    for carrier in range(carriers):
        if carrier in (0, 1):
            out.append(strong)
        elif carrier == 2:
            out.append(moderate)
        else:
            out.append(flat)
    return out


def simulate_h1_power(carriers: int, per_cell: int, scenario: str, nsim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    design, X = h1_design(carriers, per_cell)
    probs_by_carrier = scenario_probabilities(scenario, carriers)
    hits = 0
    betas = []
    ses = []
    for _ in range(nsim):
        offsets = rng.uniform(-0.25, 0.25, carriers)
        mu = np.empty(len(design))
        for i, (carrier, _era, level, _count) in enumerate(design):
            p = probs_by_carrier[int(carrier)][int(level) - 1]
            mu[i] = math.log(p / (1 - p)) + offsets[int(carrier)]
        prob = 1 / (1 + np.exp(-mu))
        y = (rng.random(len(prob)) < prob).astype(float)
        beta, se = fit_logistic_irls(X, y)
        beta_level = float(beta[2])
        se_level = float(se[2])
        betas.append(beta_level)
        ses.append(se_level)
        # H1 is a material dose-response hypothesis: the lower confidence
        # bound must exceed log(1.25), not merely zero.
        if beta_level - Z_BONFERRONI_025 * se_level > math.log(1.25):
            hits += 1
    return {
        "scenario": scenario,
        "carriers": carriers,
        "per_era_level_cell": per_cell,
        "unicode_count_rows_simulated": int(len(design)),
        "simulations": nsim,
        "criterion": f"lower Wald 95% bound for beta_fragmentation_level exceeds log(1.25), using z={Z_BONFERRONI_025:.6f}; conservative fixed alpha=0.025 approximation",
        "power": hits / nsim,
        "mean_beta_per_level_log_or": float(np.mean(betas)),
        "mean_wald_se": float(np.mean(ses)),
        "mean_odds_ratio_per_level": float(np.exp(np.mean(betas))),
    }


def simulate_h2(carriers: int, per_cell: int, delta: float, nsim: int, seed: int, margin: float = 0.10) -> dict:
    rng = np.random.default_rng(seed)
    pairs_per_carrier = 3 * per_cell
    n_pairs = carriers * pairs_per_carrier
    base_by_level = {2: 13 / 18, 3: 15 / 18, 4: 17 / 18}
    level_cycle = np.tile(np.array([2, 3, 4]), math.ceil(n_pairs / 3))[:n_pairs]
    confirm = 0
    refute = 0
    estimates = []
    ses = []
    for _ in range(nsim):
        diffs = np.empty(n_pairs)
        for i, level in enumerate(level_cycle):
            base = base_by_level[int(level)]
            logit = math.log(base / (1 - base)) + rng.normal(0.0, 0.5)
            p0 = 1 / (1 + np.exp(-logit))
            p1 = min(0.99, max(0.01, p0 + delta))
            legacy = rng.random() < p0
            newest = rng.random() < p1
            diffs[i] = float(newest) - float(legacy)
        estimate = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / math.sqrt(n_pairs))
        lo = estimate - Z_BONFERRONI_025 * se
        hi = estimate + Z_BONFERRONI_025 * se
        if lo > -margin and hi < margin and -margin < estimate < margin:
            confirm += 1
        if lo > margin or hi < -margin:
            refute += 1
        estimates.append(estimate)
        ses.append(se)
    return {
        "carriers": carriers,
        "per_era_level_cell": per_cell,
        "matched_pairs": n_pairs,
        "input_probability_shift_before_ceiling": delta,
        "equivalence_margin_pp": 100 * margin,
        "simulations": nsim,
        "pair_difficulty_logit_noise_sd": 0.5,
        "criterion": f"confirmation: paired difference-in-means 95% normal CI inside +/-{margin:.2f}; refutation: 95% CI entirely outside +/-{margin:.2f}; conservative fixed alpha=0.025 approximation",
        "equivalence_confirmation_power": confirm / nsim,
        "meaningful_effect_refutation_power": refute / nsim,
        "realized_mean_risk_difference": float(np.mean(estimates)),
        "mean_se": float(np.mean(ses)),
    }


def inventory_totals(per_cell: int) -> dict:
    overlap = 6 * per_cell
    boundary_legacy = per_cell
    boundary_newest_optional = per_cell
    count_ascii = 12
    copy_unicode = 100
    copy_ascii = 24
    decomp_unicode = 32
    decomp_ascii = 12
    minimum = copy_unicode + copy_ascii + decomp_unicode + decomp_ascii + overlap + boundary_legacy + count_ascii
    maximum = minimum + boundary_newest_optional
    return {
        "per_era_level_cell": per_cell,
        "copy_unicode": copy_unicode,
        "copy_ascii": copy_ascii,
        "decomposition_unicode": decomp_unicode,
        "decomposition_ascii": decomp_ascii,
        "count_overlap_l2_l4_both_eras": overlap,
        "count_boundary_l1_legacy": boundary_legacy,
        "count_boundary_l1_newest_if_structurally_feasible": boundary_newest_optional,
        "count_ascii": count_ascii,
        "per_carrier_minimum": minimum,
        "per_carrier_maximum": maximum,
        "six_carrier_minimum": 6 * minimum,
        "six_carrier_maximum": 6 * maximum,
    }


def runtime_estimate(per_carrier_rows: int) -> dict:
    rows = []
    total = 0.0
    for model, sec_per_probe in MEASURED_SECONDS_PER_PROBE.items():
        minutes = sec_per_probe * per_carrier_rows / 60
        total += minutes
        rows.append({"model_id": model, "assumed_rows": per_carrier_rows, "estimated_generation_minutes": minutes, "seconds_per_probe_source": sec_per_probe})
    return {"per_model": rows, "generation_only_total_minutes": total, "generation_only_total_hours": total / 60}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible confirmatory power analysis and round-2 pilot audit; pure stdlib+numpy.")
    parser.add_argument("--pilot", type=Path, default=PILOT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-cell", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--self-test", action="store_true", help="run validation and mutation checks without writing output")
    args = parser.parse_args()

    rows = read_jsonl(args.pilot)
    stats = pilot_statistics(rows)
    validation = validate_pilot_statistics(stats)
    mutation = mutation_check(rows)

    if args.self_test:
        print(json.dumps({"validation_all_pass": validation["all_pass"], "mutation_detected": mutation["mutation_detected"]}, indent=2, sort_keys=True))
        return 0 if validation["all_pass"] and mutation["mutation_detected"] else 1

    h1 = []
    for scenario in ("pilot_moderate_22p2pp", "pilot_strong_50pp_capped", "heterogeneous_pilot_mix", "flat_null"):
        for carriers in ((5, 4) if scenario != "heterogeneous_pilot_mix" else (5,)):
            h1.append(simulate_h1_power(carriers, args.per_cell, scenario, args.simulations, args.seed + 101 + len(h1)))

    h2 = []
    for carriers in (5, 4, 3):
        h2.append(simulate_h2(carriers, args.per_cell, 0.0, args.simulations, args.seed + 1000 + carriers))
        h2.append(simulate_h2(carriers, args.per_cell, 0.15, args.simulations, args.seed + 2000 + carriers))
        h2.append(simulate_h2(carriers, args.per_cell, 0.25, args.simulations, args.seed + 3000 + carriers))

    inv = inventory_totals(args.per_cell)
    script_path = Path(__file__).resolve()
    output = {
        "script": str(script_path.name),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "seed": args.seed,
        "simulations": args.simulations,
        "design": inv,
        "pilot_audit": stats,
        "pilot_validation": validation,
        "mutation_check": mutation,
        "h1_power": h1,
        "h2_power": h2,
        "generation_runtime_estimate_max_rows": runtime_estimate(inv["per_carrier_maximum"]),
        "notes": [
            "Power is a design approximation over stimulus sampling, not a claim that deterministic greedy decoding is random.",
            "H1 simulation omits item-type fixed effects and therefore is optimistic; the inventory uses homogeneous matched probes to limit that omission.",
            "H2 assumes pair-level outcomes conditionally independent given matched difficulty; positive within-pair dependence would improve precision, while unmodeled stimulus dependence would worsen it.",
            "The flat_null H1 row is a false-positive/check scenario, not a power scenario.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"pilot validation all_pass={validation['all_pass']} mutation_detected={mutation['mutation_detected']}")
    print("H1 power:")
    for row in h1:
        print(f"  {row['scenario']} carriers={row['carriers']}: {row['power']:.3f}")
    print("H2 equivalence confirmation power under true delta=0:")
    for row in h2:
        if row["input_probability_shift_before_ceiling"] == 0.0:
            print(f"  carriers={row['carriers']}: {row['equivalence_confirmation_power']:.3f}")
    return 0 if validation["all_pass"] and mutation["mutation_detected"] else 1


if __name__ == "__main__":
    sys.exit(main())
