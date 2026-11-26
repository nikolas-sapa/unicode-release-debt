from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf
import compute_confirmatory_gates as gatesmod

OUT = cf.CONFIRMATORY
BOOT = 5000
SEED = 20261127
LEVELS = ["L1", "L2", "L3", "L4"]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1 / (1 + np.exp(-x))


def loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def fit_logistic(X, y, maxiter=100, tol=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    ll = loglik(X, y, beta)
    converged = False
    for _ in range(maxiter):
        eta = X @ beta
        p = sigmoid(eta)
        w = np.maximum(p * (1 - p), 1e-10)
        grad = X.T @ (y - p)
        h = (X.T * w) @ X
        try:
            step = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h, grad, rcond=None)[0]
        scale = 1.0
        improved = False
        for _half in range(30):
            cand = beta + scale * step
            cll = loglik(X, y, cand)
            if cll >= ll - 1e-10:
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
        delta = cand - beta
        beta = cand
        ll = cll
        if float(np.max(np.abs(delta / (np.abs(beta) + 1e-8)))) < tol:
            converged = True
            break
    eta = X @ beta
    p = sigmoid(eta)
    w = np.maximum(p * (1 - p), 1e-10)
    h = (X.T * w) @ X
    try:
        cov = np.linalg.inv(h)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
        se = np.full(X.shape[1], np.nan)
    separation = bool(np.max(np.abs(beta)) > 25 or np.max(np.abs(eta)) > 34)
    return {"beta": beta, "se": se, "cov": cov, "converged": converged, "loglik": ll, "separation": separation, "valid": converged and not separation}


def design(rows, carriers, continuous_spec=None, level_fixed=False, include_carrier=True, include_era=True, include_type=True, continuous_tpc=False, interaction=False):
    carriers = sorted(carriers)
    base_continuous = [] if continuous_tpc else ["utf8_bytes_per_cluster", "codepoints_per_cluster", "mean_sequence_complexity", "supplementary_cluster_fraction", "emoji_frequency_row_mean"]
    if continuous_tpc:
        base_continuous = ["tokens_per_cluster", "utf8_bytes_per_cluster", "codepoints_per_cluster", "mean_sequence_complexity", "supplementary_cluster_fraction", "emoji_frequency_row_mean"]
    if continuous_spec is None:
        vals = {}
        for key in base_continuous:
            arr = np.array([(0.0 if key == "emoji_frequency_row_mean" and r.get("item_type") not in ("atomic_emoji", "zwj_emoji") else float(r.get(key) or 0)) for r in rows], dtype=float)
            sd = float(arr.std())
            vals[key] = (float(arr.mean()), sd if sd > 0 else 1.0)
        continuous_spec = vals
    names = ["intercept"]
    cols = [np.ones(len(rows))]
    if include_carrier:
        for c in carriers[1:]:
            names.append(f"carrier:{c}")
            cols.append(np.array([r["carrier_id"] == c for r in rows], dtype=float))
    frag_ord = {"L1": 1.0, "L2": 2.0, "L3": 3.0, "L4": 4.0}
    if continuous_tpc:
        names.append("tokens_per_cluster_continuous")
        z = continuous_spec["tokens_per_cluster"]
        cols.append(np.array([(float(r.get("tokens_per_cluster") or 0) - z[0]) / z[1] for r in rows]))
        frag_col = len(names) - 1
    elif level_fixed:
        frag_col = []
        for lev in LEVELS[1:]:
            names.append(f"fragmentation:{lev}")
            cols.append(np.array([r["fragmentation_level"] == lev for r in rows], dtype=float))
            frag_col.append(len(names) - 1)
    else:
        names.append("fragmentation_ordinal")
        cols.append(np.array([frag_ord[r["fragmentation_level"]] for r in rows], dtype=float))
        frag_col = len(names) - 1
    era_col = None
    if include_era:
        names.append("era_newest")
        cols.append(np.array([r["era"] == "newest" for r in rows], dtype=float))
        era_col = len(names) - 1
    for n in [2, 3, 4, 5]:
        names.append(f"expected_count:{n}")
        cols.append(np.array([r.get("expected_count") == n for r in rows], dtype=float))
    if include_type:
        for typ in ["atomic_emoji", "complex_nonemoji", "zwj_emoji"]:
            names.append(f"item_type:{typ}")
            cols.append(np.array([r.get("item_type") == typ for r in rows], dtype=float))
    cont_cols = []
    for key in base_continuous:
        if continuous_tpc and key == "tokens_per_cluster":
            continue
        z = continuous_spec[key]
        names.append(key)
        vals = []
        for r in rows:
            value = float(r.get(key) or 0)
            if key == "emoji_frequency_row_mean" and r.get("item_type") not in ("atomic_emoji", "zwj_emoji"):
                value = 0.0
            vals.append((value - z[0]) / z[1])
        cols.append(np.array(vals))
        cont_cols.append(len(names) - 1)
    names.append("emoji_frequency_any_missing")
    cols.append(np.array([r.get("item_type") in ("atomic_emoji", "zwj_emoji") and bool(r.get("emoji_frequency_any_missing")) for r in rows], dtype=float))
    if interaction and include_era and not level_fixed and not continuous_tpc:
        names.append("fragmentation_x_era")
        cols.append(cols[frag_col] * cols[era_col])
    X = np.column_stack(cols)
    return X, names, continuous_spec, frag_col, era_col


def h1_rows(scored, valid_carriers):
    rows = []
    for r in scored:
        if r["carrier_id"] not in valid_carriers or r["probe_type"] != "count" or r["probe_scope"] != "unicode":
            continue
        if r["fragmentation_level"] == "L1" and r["era"] == "newest":
            continue
        rows.append(r)
    return rows


def bootstrap_h1(rows, carriers, design_kwargs=None, nboot=BOOT, seed=SEED):
    design_kwargs = design_kwargs or {}
    X0, names, spec, frag_col, era_col = design(rows, carriers, **design_kwargs)
    y0 = np.array([not r["task_exact"] for r in rows], dtype=float)
    fit0 = fit_logistic(X0, y0)
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r["carrier_id"], r["era"], r["fragmentation_level"])].append(i)
    keys = sorted(strata)
    rng = np.random.default_rng(seed)
    vals = []
    valid = 0
    separation_count = 0
    nonconvergence_count = 0
    for _ in range(nboot):
        idx = []
        for key in keys:
            pool = strata[key]
            idx.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
        Xb = X0[idx]
        yb = y0[idx]
        fit = fit_logistic(Xb, yb, maxiter=60)
        if fit["valid"]:
            valid += 1
            if isinstance(frag_col, list):
                vals.append(fit["beta"][frag_col].copy())
            else:
                vals.append(fit["beta"][frag_col])
        elif fit["separation"]:
            separation_count += 1
        else:
            nonconvergence_count += 1
    arr = np.array(vals, dtype=float)
    if arr.ndim == 1:
        ci = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else [None, None]
        one_sided = (1 + int(np.sum(arr <= 0))) / (1 + len(arr)) if len(arr) else None
        boot_se = float(arr.std(ddof=1)) if len(arr) > 1 else None
        med = float(np.median(arr)) if len(arr) else None
    else:
        ci = {LEVELS[i + 1]: [float(np.quantile(arr[:, i], 0.025)), float(np.quantile(arr[:, i], 0.975))] for i in range(arr.shape[1])}
        one_sided = None
        med = [float(x) for x in np.median(arr, axis=0)]
    b = fit0["beta"][frag_col] if not isinstance(frag_col, list) else fit0["beta"][frag_col]
    se = fit0["se"][frag_col] if not isinstance(frag_col, list) else fit0["se"][frag_col]
    beta_scalar = b if not isinstance(frag_col, list) else None
    normal_boot_p = None
    if beta_scalar is not None and boot_se and boot_se > 0 and math.isfinite(float(beta_scalar)):
        normal_boot_p = 1 - phi(float(beta_scalar) / boot_se)
    return {
        "n_rows": len(rows), "n_carriers": len(set(r["carrier_id"] for r in rows)), "feature_names": names, "continuous_scaling": spec,
        "point_beta": b.tolist() if hasattr(b, "tolist") else float(b), "point_odds_ratio": np.exp(b).tolist() if hasattr(np.exp(b), "tolist") else float(np.exp(b)),
        "normal_se": se.tolist() if hasattr(se, "tolist") else float(se), "fit_converged": fit0["converged"], "fit_separation": fit0["separation"],
        "bootstrap_replicates": nboot, "bootstrap_valid": valid, "bootstrap_separation_or_singular": separation_count, "bootstrap_nonconvergence": nonconvergence_count,
        "bootstrap_median": med, "bootstrap_ci95": ci, "bootstrap_se": boot_se, "bootstrap_one_sided_p_beta_gt_0": one_sided, "bootstrap_normal_approx_one_sided_p_beta_gt_0": normal_boot_p,
        "seed": seed, "strata": ["carrier x era x fragmentation_level"], "design_kwargs": design_kwargs,
    }


def fit_carrier_h1(rows, carrier, nboot=BOOT):
    rr = [r for r in rows if r["carrier_id"] == carrier]
    if len({x["era"] for x in rr}) < 2 or len(rr) < 20:
        return {"carrier_id": carrier, "estimable": False, "reason": "insufficient carrier rows"}
    out = bootstrap_h1(rr, [carrier], {"include_carrier": False}, nboot=nboot, seed=SEED)
    out["carrier_id"] = carrier
    out["estimable"] = True
    return out


def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj



def pair_records(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["carrier_id"], r["fragmentation_level"], str(r.get("expected_count")), r["item_type"], str(r.get("supplementary_cluster_count")), r.get("match_pair_id"))].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        if len(vals) != 2 or {x["era"] for x in vals} != {"newest", "legacy"}:
            continue
        nxt = next(x for x in vals if x["era"] == "newest")
        leg = next(x for x in vals if x["era"] == "legacy")
        out.append({
            "carrier_id": nxt["carrier_id"], "fragmentation_level": nxt["fragmentation_level"], "item_type": nxt["item_type"],
            "failure_difference": int(not nxt["task_exact"]) - int(not leg["task_exact"]),
            "byte_mismatch": float(nxt.get("utf8_bytes_per_cluster") or 0) - float(leg.get("utf8_bytes_per_cluster") or 0),
            "codepoint_mismatch": float(nxt.get("codepoints_per_cluster") or 0) - float(leg.get("codepoints_per_cluster") or 0),
            "complexity_mismatch": float(nxt.get("mean_sequence_complexity") or 0) - float(leg.get("mean_sequence_complexity") or 0),
            "frequency_mismatch": float(nxt.get("emoji_frequency_row_mean") or 0) - float(leg.get("emoji_frequency_row_mean") or 0),
        })
    return out


def equal_weighted_pair_delta(pairs):
    by_cl = defaultdict(list)
    for p in pairs:
        by_cl[(p["carrier_id"], p["fragmentation_level"])].append(p["failure_difference"])
    cms = []
    detail = {}
    for c in sorted({x[0] for x in by_cl}):
        lms = []
        for lev in ("L2", "L3", "L4"):
            vals = by_cl.get((c, lev), [])
            if vals:
                lm = float(np.mean(vals))
                lms.append(lm)
                detail[f"{c}|{lev}"] = {"pairs": len(vals), "delta": lm}
        if len(lms) == 3:
            cms.append(float(np.mean(lms)))
    return {"delta": float(np.mean(cms)) if cms else None, "carrier_count": len(cms), "carrier_level_detail": detail}

def analyze_h1(scored, valid_carriers, do_carrier_boot=True):
    rows = h1_rows(scored, valid_carriers)
    carrier_results = {}
    for c in sorted(valid_carriers):
        carrier_results[c] = fit_carrier_h1(rows, c, BOOT if do_carrier_boot else 100)
    coefficient_valid = {c: bool(x.get("estimable") and x.get("fit_converged") and not x.get("fit_separation") and math.isfinite(x.get("point_beta", math.nan))) for c, x in carrier_results.items()}
    finite_nonneg = sum(1 for c, x in carrier_results.items() if coefficient_valid.get(c) and x["point_beta"] >= 0)
    families = {cf.CARRIERS[c]["tokenizer_family"] for c, x in carrier_results.items() if coefficient_valid.get(c) and x["point_beta"] >= 0}
    if len(valid_carriers) < 4 or len({cf.CARRIERS[c]["tokenizer_family"] for c in valid_carriers}) < 3:
        pooled = {
            "n_rows": len(rows), "n_carriers": len(valid_carriers), "point_beta": None, "point_odds_ratio": None, "normal_se": None,
            "fit_converged": False, "fit_separation": False, "bootstrap_replicates": BOOT, "bootstrap_valid": 0,
            "bootstrap_separation_or_singular": 0, "bootstrap_nonconvergence": 0, "bootstrap_median": None, "bootstrap_ci95": [None, None],
            "bootstrap_se": None, "bootstrap_one_sided_p_beta_gt_0": None, "bootstrap_normal_approx_one_sided_p_beta_gt_0": None,
            "normal_one_sided_p_beta_gt_0": None, "seed": SEED, "strata": ["carrier x era x fragmentation_level"],
            "insufficient_before_estimation": "fewer than four structurally valid carriers or fewer than three tokenizer families; primary H1 is inconclusive without pooling",
        }
    else:
        pooled = bootstrap_h1(rows, valid_carriers)
        z = pooled["point_beta"] / pooled["normal_se"] if pooled["normal_se"] and pooled["normal_se"] > 0 else float("nan")
        pooled["normal_one_sided_p_beta_gt_0"] = 1 - phi(z) if math.isfinite(z) else None
    return {
        "pooled": pooled, "carrier_specific": carrier_results,
        "estimable_nonnegative_carrier_count": finite_nonneg, "nonnegative_tokenizer_family_count": len(families),
        "estimable_nonnegative_carriers": sorted(c for c, x in carrier_results.items() if coefficient_valid.get(c) and x["point_beta"] >= 0),
        "carrier_coefficient_valid": coefficient_valid,
        "count_rows": len(rows), "count_row_hash": cf.sha_text(json.dumps([[r["completion_key"], r["task_exact"]] for r in rows], ensure_ascii=False, separators=(",", ":"))),
    }


def overlap_pair_key(r):
    return (r["carrier_id"], r["fragmentation_level"], str(r.get("expected_count")), r["item_type"], str(r.get("supplementary_cluster_count")), r.get("match_pair_id"))


def h2_overlap_rows(scored, valid_carriers):
    rows = [r for r in scored if r["carrier_id"] in valid_carriers and r["probe_type"] == "count" and r["probe_scope"] == "unicode" and r["fragmentation_level"] in ("L2", "L3", "L4") and r.get("match_pair_id")]
    groups = defaultdict(list)
    for r in rows:
        groups[overlap_pair_key(r)].append(r)
    complete = []
    incomplete = []
    for key, vals in groups.items():
        eras = {x["era"] for x in vals}
        if eras == {"newest", "legacy"} and len(vals) == 2:
            complete.extend(vals)
        else:
            incomplete.append({"key": list(key), "eras": sorted(eras), "n": len(vals)})
    return complete, incomplete


def bootstrap_h2(rows, nboot=BOOT, seed=20261128):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["carrier_id"], r["fragmentation_level"], str(r.get("expected_count")), r["item_type"], str(r.get("supplementary_cluster_count")), r.get("match_pair_id"))].append(r)
    pair_groups = list(groups.values())
    by_cl = defaultdict(list)
    for pair in pair_groups:
        f = {x["era"]: int(not x["task_exact"]) for x in pair}
        by_cl[(pair[0]["carrier_id"], pair[0]["fragmentation_level"])].append(f["newest"] - f["legacy"])
    carrier_level = {}
    for key, vals in sorted(by_cl.items()):
        carrier_level["|".join(key)] = {"pairs": len(vals), "mean_newest_minus_legacy_failure": float(np.mean(vals))}
    carrier_means = {}
    for c in sorted({x[0] for x in by_cl}):
        ds = []
        levels = []
        for (cc, lev), vals in sorted(by_cl.items()):
            if cc == c:
                ds.append(float(np.mean(vals)))
                levels.append(lev)
        if levels == ["L2", "L3", "L4"]:
            carrier_means[c] = float(np.mean(ds))
    delta = float(np.mean(list(carrier_means.values()))) if carrier_means else None
    rng = np.random.default_rng(seed)
    reps = []
    pair_arrays = []
    carrier_levels = sorted(by_cl)
    for key in carrier_levels:
        pair_arrays.append(np.array(by_cl[key], dtype=float))
    for _ in range(nboot):
        cms = []
        for c in sorted(carrier_means):
            levelmeans = []
            for key, arr in zip(carrier_levels, pair_arrays):
                if key[0] != c:
                    continue
                idx = rng.integers(0, len(arr), len(arr))
                levelmeans.append(float(np.mean(arr[idx])))
            if len(levelmeans) == 3:
                cms.append(float(np.mean(levelmeans)))
        if cms:
            reps.append(float(np.mean(cms)))
    arr = np.array(reps)
    pair_n = len(pair_groups)
    se = float(arr.std(ddof=1)) if len(arr) else None
    ci = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else [None, None]
    width = float(np.quantile(arr, 0.975) - np.quantile(arr, 0.025)) if len(arr) else None
    if delta is not None and se and se > 0:
        z_upper = (delta - 0.10) / se
        z_lower = (delta + 0.10) / se
        p_upper = phi(z_upper)
        p_lower = 1 - phi(z_lower)
        tost_p = max(p_upper, p_lower)
        exact_zero_p = 2 * (1 - phi(abs(delta) / se))
    else:
        z_upper = z_lower = p_upper = p_lower = tost_p = exact_zero_p = None
    return {
        "pairs": pair_n, "pair_hash": cf.sha_text(json.dumps([[x["completion_key"], x["task_exact"]] for x in rows], ensure_ascii=False, separators=(",", ":"))),
        "carrier_level_deltas": carrier_level, "carrier_mean_deltas": carrier_means, "delta_newest_minus_legacy": delta,
        "bootstrap_replicates": nboot, "bootstrap_se": se, "bootstrap_ci95": ci, "bootstrap_ci_width": width,
        "bootstrap_tail_lower_prop_le_minus_10pp": float(np.mean(arr <= -0.10)) if len(arr) else None,
        "bootstrap_tail_upper_prop_ge_plus_10pp": float(np.mean(arr >= 0.10)) if len(arr) else None,
        "z_upper": z_upper, "z_lower": z_lower, "tost_p_upper": p_upper, "tost_p_lower": p_lower, "tost_p": tost_p,
        "exact_zero_difference_p": exact_zero_p, "seed": seed,
    }


def rates(rows, keys):
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        key = tuple(r.get(k) for k in keys)
        groups[key][0] += int(not r["task_exact"])
        groups[key][1] += 1
    return {"|".join(str(x) for x in key): {"failures": v[0], "n": v[1], "failure_rate": v[0] / v[1]} for key, v in sorted(groups.items())}


def h3_h4(scored, gates):
    structural = [x["carrier_id"] for x in gates["carriers"].values() if x["include_h1h2h3_primary"]]
    copyvalid = [x["carrier_id"] for x in gates["carriers"].values() if x["include_h4_primary"]]
    scalar_rows = [r for r in scored if r["carrier_id"] in structural and r["probe_type"] == "scalar_listing" and r["probe_scope"] == "unicode"]
    copy_rows = [r for r in scored if r["carrier_id"] in copyvalid and r["probe_type"] == "copy" and r["probe_scope"] == "unicode"]
    h3_by_carrier = {}
    for c in sorted(structural):
        vals = {}
        for era in ("newest", "legacy"):
            rr = [r for r in scalar_rows if r["carrier_id"] == c and r["era"] == era]
            vals[era] = {"exact": sum(r["task_exact"] for r in rr), "n": len(rr), "rate": sum(r["task_exact"] for r in rr) / len(rr) if rr else None}
        h3_by_carrier[c] = vals
    h4_by_carrier = {}
    for c in sorted(copyvalid):
        vals = {}
        for era in ("newest", "legacy"):
            rr = [r for r in copy_rows if r["carrier_id"] == c and r["era"] == era]
            vals[era] = {"exact": sum(r["task_exact"] for r in rr), "n": len(rr), "rate": sum(r["task_exact"] for r in rr) / len(rr) if rr else None}
        h4_by_carrier[c] = vals
    return {
        "h3": {"eligible_carriers": sorted(structural), "by_carrier": h3_by_carrier, "every_newest_zero_exact": bool(structural) and all(v["newest"]["n"] == 16 and v["newest"]["exact"] == 0 for v in h3_by_carrier.values()), "every_legacy_zero_exact": bool(structural) and all(v["legacy"]["n"] == 16 and v["legacy"]["exact"] == 0 for v in h3_by_carrier.values()), "minimum_met": len(structural) >= 4, "error_classes": dict(Counter(r.get("scalar_error_class") for r in scalar_rows)), "signed_or_token_errors_precluded": True},
        "h4": {"eligible_carriers": sorted(copyvalid), "by_carrier": h4_by_carrier, "every_newest_byte_exact": bool(copyvalid) and all(v["newest"]["n"] == 60 and v["newest"]["exact"] == 60 for v in h4_by_carrier.values()), "every_legacy_byte_exact": bool(copyvalid) and all(v["legacy"]["n"] == 40 and v["legacy"]["exact"] == 40 for v in h4_by_carrier.values()), "minimum_met": len(copyvalid) >= 4, "copy_failure_classes": dict(Counter(r.get("copy_near_miss_class") for r in copy_rows if not r["task_exact"])), "byte_exact_by_item_type_era": rates(copy_rows, ["item_type", "era"]), "near_miss_by_item_type": rates([r for r in copy_rows if r.get("copy_near_miss")], ["item_type"])},
    }


def sensitivity_analyses(scored, gates):
    structural = [x["carrier_id"] for x in gates["carriers"].values() if x["include_h1h2h3_primary"]]
    rows = h1_rows(scored, structural)
    out = {}
    if len(rows) >= 100 and len(structural) >= 2:
        try:
            out["continuous_tokens_per_cluster"] = bootstrap_h1(rows, structural, {"continuous_tpc": True}, nboot=1000, seed=SEED + 20)
        except Exception as e:
            out["continuous_tokens_per_cluster"] = {"error": str(e)}
        try:
            out["level_fixed_effects"] = bootstrap_h1(rows, structural, {"level_fixed": True}, nboot=1000, seed=SEED + 21)
        except Exception as e:
            out["level_fixed_effects"] = {"error": str(e)}
        legacy = [r for r in rows if r["era"] == "legacy"]
        if legacy:
            try:
                out["legacy_only_fragmentation"] = bootstrap_h1(legacy, structural, {"include_era": False}, nboot=1000, seed=SEED + 22)
            except Exception as e:
                out["legacy_only_fragmentation"] = {"error": str(e)}
        within = {}
        for typ in sorted({r["item_type"] for r in rows}):
            rr = [r for r in rows if r["item_type"] == typ]
            try:
                within[typ] = bootstrap_h1(rr, structural, {"include_type": False}, nboot=500, seed=SEED + 30 + len(typ))
            except Exception as e:
                within[typ] = {"error": str(e)}
        out["within_item_type"] = within
        try:
            out["fragmentation_by_era_interaction"] = bootstrap_h1(rows, structural, {"interaction": True}, nboot=1000, seed=SEED + 40)
        except Exception as e:
            out["fragmentation_by_era_interaction"] = {"error": str(e)}
        X, names, spec, frag, era = design(rows, structural)
        y = np.array([not r["task_exact"] for r in rows])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        out["linear_probability_fragmentation_coefficient"] = {"coefficient": float(coef[frag]), "note": "unregularized OLS sensitivity; inferential uncertainty not used for confirmation"}
        out["unadjusted_failure_by_level_era"] = rates(rows, ["fragmentation_level", "era"])
    out["quantization_exclusion_sensitivity"] = {"note": "expanded bf16 comparator downloads were skipped pre-generation; this sensitivity therefore cannot estimate the intended direct quantization contrast and is reported as a frozen limitation", "bf16_only_carriers": sorted(c for c in structural if cf.CARRIERS[c]["quantization_class"] == "bf16")}
    overlap_rows, _ = h2_overlap_rows(scored, structural)
    pairs = pair_records(overlap_rows)
    h2s = {"pair_count": len(pairs), "overall_unweighted_implementation_check": equal_weighted_pair_delta(pairs)}
    for typ in ("complex_nonemoji", "zwj_emoji"):
        tt = [r for r in overlap_rows if r["item_type"] == typ]
        try:
            h2s[f"type_restricted_{typ}"] = bootstrap_h2(tt, nboot=1000, seed=20261128 + len(typ)) if tt else {"error": "no rows"}
        except Exception as e:
            h2s[f"type_restricted_{typ}"] = {"error": str(e)}
    if pairs:
        Xp = np.column_stack([np.ones(len(pairs)), [p["byte_mismatch"] for p in pairs], [p["codepoint_mismatch"] for p in pairs], [p["complexity_mismatch"] for p in pairs], [p["frequency_mismatch"] for p in pairs]])
        yp = np.array([p["failure_difference"] for p in pairs], dtype=float)
        coef = np.linalg.lstsq(Xp, yp, rcond=None)[0]
        h2s["pair_difference_mismatch_regression"] = {"coefficient_names": ["intercept", "byte_mismatch", "codepoint_mismatch", "complexity_mismatch", "frequency_mismatch"], "coefficients": [float(x) for x in coef], "note": "OLS sensitivity only; no confirmatory p-value"}
    if rows:
        try:
            Xe, names_e, spec_e, frag_e, era_e = design(rows, structural)
            ye = np.array([not r["task_exact"] for r in rows], dtype=float)
            fe = fit_logistic(Xe, ye)
            if era_e is not None:
                Xnew = Xe.copy(); Xnew[:, era_e] = 1.0
                Xleg = Xe.copy(); Xleg[:, era_e] = 0.0
                ame = float(np.mean(sigmoid(Xnew @ fe["beta"]) - sigmoid(Xleg @ fe["beta"])))
                h2s["logistic_era_coefficient_average_marginal_effect"] = {"era_coefficient": float(fe["beta"][era_e]), "era_coefficient_se": float(fe["se"][era_e]), "average_marginal_effect_newest_minus_legacy": ame, "fit_valid": fe["valid"], "note": "covariate-adjusted sensitivity, not the matched-pair primary estimand"}
        except Exception as e:
            h2s["logistic_era_coefficient_average_marginal_effect"] = {"error": str(e)}
    h2s["newest_l1_pair_sensitivity"] = {"estimable": False, "reason": "the frozen inventory contains match_pair_id values only for L2-L4; newest-L1 pairs were not constructed at freeze time, so this sensitivity cannot be run without violating the freeze"}
    out["h2_sensitivities"] = h2s
    out["h2_exact_zero_difference_reported_in_main_h2"] = True
    out["quantitative_descriptive_breakdowns"] = {
        "count_failure_by_carrier_level_era": rates(rows, ["carrier_id", "fragmentation_level", "era"]),
        "count_failure_by_item_type_level": rates(rows, ["item_type", "fragmentation_level"]),
    }
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def exact_stats(rows):
    exact = sum(1 for r in rows if r.get("task_exact"))
    return {"exact": exact, "n": len(rows), "rate": exact / len(rows) if rows else None}


def failure_stats(rows):
    failures = sum(1 for r in rows if not r.get("task_exact"))
    return {"failures": failures, "n": len(rows), "failure_rate": failures / len(rows) if rows else None}


def bootstrap_single_carrier_pair_delta(rows, carrier_id, nboot=1000, seed=20261128):
    count_rows = [r for r in rows if r["carrier_id"] == carrier_id and r["probe_type"] == "count" and r["probe_scope"] == "unicode" and r.get("match_pair_id") and r["fragmentation_level"] in ("L2", "L3", "L4")]
    pairs = pair_records(count_rows)
    by_level = defaultdict(list)
    for p in pairs:
        by_level[p["fragmentation_level"]].append(p["failure_difference"])
    if sorted(by_level) != ["L2", "L3", "L4"]:
        return {"carrier_id": carrier_id, "pairs": len(pairs), "estimable": False, "reason": "missing one or more overlap levels"}
    observed = float(np.mean([np.mean(by_level[level]) for level in ("L2", "L3", "L4")]))
    rng = np.random.default_rng(seed)
    reps = []
    arrays = [np.array(by_level[level], dtype=float) for level in ("L2", "L3", "L4")]
    for _ in range(nboot):
        reps.append(float(np.mean([np.mean(arr[rng.integers(0, len(arr), len(arr))]) for arr in arrays])))
    arr = np.array(reps)
    return {
        "carrier_id": carrier_id, "pairs": len(pairs), "estimable": True, "delta": observed,
        "bootstrap_ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "bootstrap_se": float(arr.std(ddof=1)), "bootstrap_replicates": nboot,
        "level_deltas": {level: {"pairs": len(by_level[level]), "delta": float(np.mean(by_level[level]))} for level in ("L2", "L3", "L4")},
    }


def exploratory_descriptive(scored):
    out = {"carriers": {}, "warning": "exploratory descriptive output only; carriers failing frozen copy or structural ASCII gates are not interpretable confirmatory structural measurements"}
    for cfg in cf.carrier_configs(include_descriptive=True):
        c = cfg["carrier_id"]
        rows = [r for r in scored if r["carrier_id"] == c]
        count_rows = [r for r in rows if r["probe_type"] == "count" and r["probe_scope"] == "unicode"]
        copy_rows = [r for r in rows if r["probe_type"] == "copy" and r["probe_scope"] == "unicode"]
        scalar_rows = [r for r in rows if r["probe_type"] == "scalar_listing" and r["probe_scope"] == "unicode"]
        level_era = {}
        for level in LEVELS:
            for era in ("legacy", "newest"):
                rr = [r for r in count_rows if r["fragmentation_level"] == level and r["era"] == era]
                st = failure_stats(rr)
                level_era[f"{level}|{era}"] = {**st, "exact": st["n"] - st["failures"], "format_failures": sum(1 for r in rr if r.get("format_failure")), "wrong_by_one": sum(1 for r in rr if r.get("count_near_miss"))}
        h1_like = h1_rows(rows, [c])
        logistic = {"estimable": False, "reason": "insufficient rows"}
        if len(h1_like) >= 20:
            try:
                X, names, spec, frag_col, era_col = design(h1_like, [c], include_carrier=False)
                y = np.array([not r["task_exact"] for r in h1_like], dtype=float)
                fit = fit_logistic(X, y)
                beta = float(fit["beta"][frag_col])
                se = float(fit["se"][frag_col]) if math.isfinite(float(fit["se"][frag_col])) else None
                logistic = {
                    "estimable": bool(fit["valid"] and math.isfinite(beta)), "beta": beta, "odds_ratio": float(math.exp(beta)) if math.isfinite(beta) and abs(beta) < 700 else None,
                    "normal_se": se, "normal_ci95": [beta - 1.96 * se, beta + 1.96 * se] if se is not None else [None, None],
                    "fit_converged": fit["converged"], "fit_separation": fit["separation"], "rows": len(h1_like),
                    "note": "single-carrier covariate-adjusted descriptive fit without bootstrap; not a confirmatory coefficient when the carrier fails structural gates",
                }
            except Exception as exc:
                logistic = {"estimable": False, "reason": str(exc), "rows": len(h1_like)}
        overlap = [r for r in count_rows if r.get("match_pair_id") and r["fragmentation_level"] in ("L2", "L3", "L4")]
        pair_delta = bootstrap_single_carrier_pair_delta(rows, c)
        out["carriers"][c] = {
            "count_failure_by_level_era": level_era,
            "copy_unicode_exact": exact_stats(copy_rows),
            "scalar_unicode_exact": exact_stats(scalar_rows),
            "count_overlap_exact": exact_stats([r for r in count_rows if r["fragmentation_level"] in ("L2", "L3", "L4")]),
            "exploratory_logistic_fragmentation_fit": logistic,
            "exploratory_matched_pair_delta": pair_delta,
            "copy_wrapper_format_failures": sum(1 for r in rows if r["probe_type"] == "copy" and r.get("format_failure")),
        }
    return out


def make_tables_figures(scored, gates, h1, h2, hh, analysis):
    tables = OUT / "tables"
    figures = OUT / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    blue = "#006bff"
    dark = "#1f2937"
    gray = "#6b7280"
    light_gray = "#d1d5db"
    carriers = [x["carrier_id"] for x in cf.carrier_configs(include_descriptive=True)]

    t1_rows = []
    for cfg in cf.carrier_configs(include_descriptive=True):
        c = cfg["carrier_id"]
        x = gates["carriers"][c]
        t1_rows.append({
            "carrier_id": c, "revision": cfg["revision"], "primary_ladder": x["primary_ladder"], "tokenizer_valid": x["tokenizer_valid"],
            "ascii_copy_exact": f"{x['ascii_copy_exact']}/24", "copy_wrapper_format_failures": x["copy_wrapper_format_failures"],
            "ascii_count_exact": f"{x['ascii_count_exact']}/12", "ascii_scalar_exact": f"{x['ascii_scalar_exact']}/12",
            "copy_valid": x["copy_valid"], "structural_valid": x["structural_valid"], "quantization_status": x["quantization_status"],
            "include_h1h2h3_primary": x["include_h1h2h3_primary"], "include_h4_primary": x["include_h4_primary"],
        })
    write_csv(tables / "T1_carrier_validity_quantization_gates.csv", t1_rows, ["carrier_id", "revision", "primary_ladder", "tokenizer_valid", "ascii_copy_exact", "copy_wrapper_format_failures", "ascii_count_exact", "ascii_scalar_exact", "copy_valid", "structural_valid", "quantization_status", "include_h1h2h3_primary", "include_h4_primary"])

    with open(OUT / "inventory_cell_table.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        inventory_rows = list(reader)
        inventory_fields = reader.fieldnames
    write_csv(tables / "T2_inventory_cells.csv", inventory_rows, inventory_fields)

    pooled = h1["pooled"]
    pooled_ci = pooled.get("bootstrap_ci95") or [None, None]
    t3_rows = [{
        "row_type": "pooled_primary", "carrier_id": "pooled", "beta": pooled.get("point_beta"), "odds_ratio": pooled.get("point_odds_ratio"),
        "bootstrap_ci_low": pooled_ci[0], "bootstrap_ci_high": pooled_ci[1], "bootstrap_p_beta_gt_0": pooled.get("bootstrap_one_sided_p_beta_gt_0"),
        "holm_adjusted_p": analysis["verdict"]["hypotheses"]["H1"].get("holm_adjusted_p"), "bootstrap_valid": pooled.get("bootstrap_valid"),
        "bootstrap_replicates": pooled.get("bootstrap_replicates"), "estimable": pooled.get("point_beta") is not None,
        "reason": pooled.get("insufficient_before_estimation", ""),
    }]
    for c, x in sorted(h1["carrier_specific"].items()):
        ci = x.get("bootstrap_ci95") or [None, None]
        t3_rows.append({
            "row_type": "carrier_specific_primary", "carrier_id": c, "beta": x.get("point_beta"), "odds_ratio": x.get("point_odds_ratio"),
            "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1], "bootstrap_p_beta_gt_0": x.get("bootstrap_one_sided_p_beta_gt_0"),
            "holm_adjusted_p": "", "bootstrap_valid": x.get("bootstrap_valid"), "bootstrap_replicates": x.get("bootstrap_replicates"),
            "estimable": h1.get("carrier_coefficient_valid", {}).get(c, x.get("estimable")), "reason": x.get("reason", ""),
        })
    write_csv(tables / "T3_h1_coefficients.csv", t3_rows, ["row_type", "carrier_id", "beta", "odds_ratio", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_p_beta_gt_0", "holm_adjusted_p", "bootstrap_valid", "bootstrap_replicates", "estimable", "reason"])

    h2_ci = h2.get("bootstrap_ci95") or [None, None]
    t4_rows = [{
        "row_type": "pooled_primary", "carrier_or_level": "pooled", "pairs": h2.get("pairs"), "delta": h2.get("delta_newest_minus_legacy"),
        "bootstrap_ci_low": h2_ci[0], "bootstrap_ci_high": h2_ci[1], "tost_p": h2.get("tost_p"),
        "holm_adjusted_tost_p": analysis["verdict"]["hypotheses"]["H2"].get("holm_adjusted_tost_p"), "note": h2.get("error", ""),
    }]
    for c, delta in sorted(h2.get("carrier_mean_deltas", {}).items()):
        t4_rows.append({"row_type": "carrier_primary", "carrier_or_level": c, "pairs": "", "delta": delta, "bootstrap_ci_low": "", "bootstrap_ci_high": "", "tost_p": "", "holm_adjusted_tost_p": "", "note": ""})
    for key, x in sorted(h2.get("carrier_level_deltas", {}).items()):
        t4_rows.append({"row_type": "carrier_level_primary", "carrier_or_level": key, "pairs": x["pairs"], "delta": x["mean_newest_minus_legacy_failure"], "bootstrap_ci_low": "", "bootstrap_ci_high": "", "tost_p": "", "holm_adjusted_tost_p": "", "note": ""})
    write_csv(tables / "T4_h2_matched_pairs.csv", t4_rows, ["row_type", "carrier_or_level", "pairs", "delta", "bootstrap_ci_low", "bootstrap_ci_high", "tost_p", "holm_adjusted_tost_p", "note"])

    copy_unicode = [r for r in scored if r["probe_type"] == "copy" and r["probe_scope"] == "unicode"]
    t5_rows = []
    for c in carriers:
        for era in ("newest", "legacy"):
            for typ in ("atomic_nonemoji", "atomic_emoji", "complex_nonemoji", "zwj_emoji"):
                rr = [r for r in copy_unicode if r["carrier_id"] == c and r["era"] == era and r["item_type"] == typ]
                st = exact_stats(rr)
                t5_rows.append({"carrier_id": c, "era": era, "item_type": typ, **st, "near_misses": sum(1 for r in rr if r.get("copy_near_miss")), "format_failures": sum(1 for r in rr if r.get("format_failure")), "include_h4_primary": gates["carriers"][c]["include_h4_primary"]})
    write_csv(tables / "T5_copy_ceiling.csv", t5_rows, ["carrier_id", "era", "item_type", "exact", "n", "rate", "near_misses", "format_failures", "include_h4_primary"])

    scalar_unicode = [r for r in scored if r["probe_type"] == "scalar_listing" and r["probe_scope"] == "unicode"]
    t6_rows = []
    for c in carriers:
        for era in ("newest", "legacy"):
            for typ in ("atomic_nonemoji", "atomic_emoji", "complex_nonemoji", "zwj_emoji"):
                rr = [r for r in scalar_unicode if r["carrier_id"] == c and r["era"] == era and r["item_type"] == typ]
                st = exact_stats(rr)
                t6_rows.append({"carrier_id": c, "era": era, "item_type": typ, **st, "near_misses": sum(1 for r in rr if r.get("scalar_near_miss")), "error_classes": json.dumps(dict(Counter(r.get("scalar_error_class") for r in rr)), sort_keys=True), "include_h1h2h3_primary": gates["carriers"][c]["include_h1h2h3_primary"]})
    write_csv(tables / "T6_scalar_floor.csv", t6_rows, ["carrier_id", "era", "item_type", "exact", "n", "rate", "near_misses", "error_classes", "include_h1h2h3_primary"])

    count_unicode = [r for r in scored if r["probe_type"] == "count" and r["probe_scope"] == "unicode"]
    t7_rows = []
    for c in carriers:
        for level in LEVELS:
            for era in ("legacy", "newest"):
                rr = [r for r in count_unicode if r["carrier_id"] == c and r["fragmentation_level"] == level and r["era"] == era]
                st = exact_stats(rr)
                signed = Counter(str(r.get("count_signed_error")) for r in rr if r.get("count_signed_error") is not None)
                t7_rows.append({"carrier_id": c, "fragmentation_level": level, "era": era, **st, "wrong_by_one": sum(1 for r in rr if r.get("count_near_miss")), "format_failures": sum(1 for r in rr if r.get("format_failure")), "signed_error_distribution": json.dumps(dict(sorted(signed.items())), sort_keys=True), "include_h1h2h3_primary": gates["carriers"][c]["include_h1h2h3_primary"]})
    write_csv(tables / "T7_count_errors.csv", t7_rows, ["carrier_id", "fragmentation_level", "era", "exact", "n", "rate", "wrong_by_one", "format_failures", "signed_error_distribution", "include_h1h2h3_primary"])

    copy_examples = []
    for r in copy_unicode:
        if not r["task_exact"]:
            copy_examples.append({"carrier_id": r["carrier_id"], "probe_key": r["probe_key"], "era": r.get("era"), "item_type": r.get("item_type"), "near_miss_class": r.get("copy_near_miss_class"), "first_byte_mismatch_offset": r.get("first_byte_mismatch_offset"), "target_diff_hex_context": r.get("target_diff_hex_context"), "output_diff_hex_context": r.get("output_diff_hex_context"), "raw_text": r.get("raw_text"), "target": r.get("target")})
    write_csv(tables / "supplementary_copy_near_miss_examples.csv", copy_examples, ["carrier_id", "probe_key", "era", "item_type", "near_miss_class", "first_byte_mismatch_offset", "target_diff_hex_context", "output_diff_hex_context", "raw_text", "target"])

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#f3f4f6")
    columns = [(era, level) for era in ("legacy", "newest") for level in LEVELS]
    z = np.full((len(carriers), len(columns)), np.nan)
    for i, c in enumerate(carriers):
        for j, (era, level) in enumerate(columns):
            rr = [r for r in count_unicode if r["carrier_id"] == c and r["era"] == era and r["fragmentation_level"] == level]
            if rr:
                z[i, j] = np.mean([not r["task_exact"] for r in rr])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(np.ma.masked_invalid(z), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)), [f"{era}\n{level}" for era, level in columns])
    ax.set_yticks(range(len(carriers)), [c.replace("-instruct", "-instr") for c in carriers], fontsize=7)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if z[i, j] > 0.55 else dark)
    ax.set_title("Count-probe failure rate by era and fragmentation level")
    fig.colorbar(im, ax=ax, label="failure rate")
    plt.tight_layout()
    plt.savefig(figures / "F1_failure_heatmap.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, carriers):
        rr = [r for r in count_unicode if r["carrier_id"] == c and not (r["fragmentation_level"] == "L1" and r["era"] == "newest")]
        for era, color in (("legacy", blue), ("newest", gray)):
            groups = defaultdict(list)
            for r in rr:
                if r["era"] == era and r.get("tokens_per_cluster") is not None:
                    groups[round(float(r["tokens_per_cluster"]) * 2) / 2].append(not r["task_exact"])
            xs = sorted(groups)
            ax.scatter(xs, [np.mean(groups[x]) for x in xs], s=[12 + 2 * len(groups[x]) for x in xs], color=color, alpha=0.85, label=era)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(c.replace("-instruct", "-instr"), fontsize=8)
        ax.text(0.02, 0.03, "adjusted primary curve not estimable" if not gates["minimum_h1h2_met"] else "observed binned means", transform=ax.transAxes, fontsize=6, color=gray)
    for ax in axes[-1]:
        ax.set_xlabel("measured tokens per cluster")
    for ax in axes[:, 0]:
        ax.set_ylabel("count failure rate")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Observed fragmentation dose-response (binned; descriptive when gates fail)", y=0.995)
    plt.tight_layout()
    plt.savefig(figures / "F2_fragmentation_dose_response.png", dpi=180)
    plt.close()

    descriptive = analysis.get("exploratory_descriptive", {}).get("carriers", {})
    entries = [descriptive[c]["exploratory_matched_pair_delta"] for c in carriers if c in descriptive and descriptive[c]["exploratory_matched_pair_delta"].get("estimable")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvspan(-0.10, 0.10, color="#eff6ff", zorder=0)
    ax.axvline(-0.10, color="#dc2626", linestyle="--", linewidth=1)
    ax.axvline(0.10, color="#dc2626", linestyle="--", linewidth=1)
    ax.axvline(0, color=dark, linewidth=0.8)
    ylabels = []
    for y, x in enumerate(entries):
        lo, hi = x["bootstrap_ci95"]
        ax.errorbar(x["delta"], y, xerr=[[x["delta"] - lo], [hi - x["delta"]]], fmt="o", color=blue, ecolor=light_gray, capsize=3, markersize=4)
        ylabels.append(x["carrier_id"].replace("-instruct", "-instr") + " (expl.)")
    if h2.get("delta_newest_minus_legacy") is not None and h2.get("bootstrap_ci95") and h2["bootstrap_ci95"][0] is not None:
        y = len(entries)
        lo, hi = h2["bootstrap_ci95"]
        ax.errorbar(h2["delta_newest_minus_legacy"], y, xerr=[[h2["delta_newest_minus_legacy"] - lo], [hi - h2["delta_newest_minus_legacy"]]], fmt="D", color=dark, ecolor=dark, capsize=3, markersize=4)
        ylabels.append("pooled primary")
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_xlabel("newest minus legacy count failure (matched pairs)")
    ax.set_title("Matched era-effect forest plot; shaded band is ±10 pp")
    if not gates["minimum_h1h2_met"]:
        ax.text(0.01, 0.98, "No confirmatory pooled estimate: fewer than four structurally valid primary carriers", transform=ax.transAxes, va="top", fontsize=7, color="#991b1b")
    plt.tight_layout()
    plt.savefig(figures / "F3_era_effect_forest.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xpos = np.arange(len(carriers))
    width = 0.25
    series = []
    for label, probe_type in (("copy unicode", "copy"), ("count overlap", "count"), ("scalar unicode", "scalar_listing")):
        vals = []
        for c in carriers:
            if probe_type == "count":
                rr = [r for r in count_unicode if r["carrier_id"] == c and r["fragmentation_level"] in ("L2", "L3", "L4")]
            else:
                rr = [r for r in scored if r["carrier_id"] == c and r["probe_type"] == probe_type and r["probe_scope"] == "unicode"]
            st = exact_stats(rr)
            vals.append(st["rate"] if st["n"] else np.nan)
        series.append((label, vals))
    for idx, ((label, vals), color) in enumerate(zip(series, (blue, gray, dark))):
        ax.bar(xpos + (idx - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(xpos, [c.replace("-instruct", "-instr") for c in carriers], rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exact rate")
    ax.set_title("Copy/count/scalar exactness triad")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures / "F4_copy_count_scalar_triad.png", dpi=180)
    plt.close()

    taxonomy = ["format_failure_wrapper_mimicry", "format_failure_other", "normalization_only_mismatch", "decomposed_rgi_sequence_or_missing_zwj", "missing_vs16", "truncation_near_miss", "truncation_other", "extra_text_after_exact_target", "replacement_character_emitted", "single_codepoint_edit", "two_codepoint_edits", "reordering", "multi_codepoint_corruption", "other_unclassified_mismatch"]
    row_keys = [(c, typ) for c in carriers for typ in ("atomic_nonemoji", "atomic_emoji", "complex_nonemoji", "zwj_emoji")]
    mat = np.zeros((len(row_keys), len(taxonomy)))
    for i, (c, typ) in enumerate(row_keys):
        rr = [r for r in copy_unicode if r["carrier_id"] == c and r["item_type"] == typ and not r["task_exact"]]
        counts = Counter(r.get("copy_near_miss_class") for r in rr)
        for j, cls in enumerate(taxonomy):
            mat[i, j] = counts.get(cls, 0)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(len(taxonomy)), taxonomy, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_keys)), [f"{c.replace('-instruct', '-instr')} | {typ}" for c, typ in row_keys], fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=6, color="white" if mat[i, j] > max(1, mat.max() / 2) else dark)
    ax.set_title("Copy failure near-miss taxonomy by carrier and item type")
    fig.colorbar(im, ax=ax, label="rows")
    plt.tight_layout()
    plt.savefig(figures / "F5_copy_near_miss_taxonomy.png", dpi=180)
    plt.close()
    return {"tables": sorted(str(x) for x in tables.glob("*.csv")), "figures": sorted(str(x) for x in figures.glob("*.png"))}


def implementation_mutation_checks():
    rng = np.random.default_rng(1)
    n = 200
    x = np.linspace(-2, 2, n)
    intercept = 1.5
    slope = 0.8
    p = sigmoid(intercept + slope * x)
    y = (rng.random(n) < p).astype(float)
    X = np.column_stack([np.ones(n), x])
    fit = fit_logistic(X, y)
    xsep = np.array([-2., -1., 1., 2.])
    ysep = np.array([0., 0., 1., 1.])
    Xsep = np.column_stack([np.ones(4), xsep])
    fsep = fit_logistic(Xsep, ysep)
    checks = {
        "logistic_recovers_positive_slope": bool(fit["valid"] and fit["beta"][1] > 0),
        "logistic_intercept_reasonable": bool(fit["valid"] and abs(fit["beta"][0] - intercept) < 1.5),
        "complete_separation_flagged": fsep["separation"] or not fsep["valid"],
        "tost_zero_delta_is_small_p": max(phi((0 - 0.10) / 0.02), 1 - phi((0 + 0.10) / 0.02)) < 0.01,
        "tost_large_delta_not_significant": max(1 - phi((0.2 - 0.10) / 0.02), phi((0.2 + 0.10) / 0.02)) > 0.05,
        "holm_monotonic_example": holm_adjust([0.01, 0.04]) == [0.02, 0.04],
    }
    yflat = (rng.random(n) < 0.5).astype(float)
    fflat = fit_logistic(X, yflat)
    checks["flat_synthetic_slope_near_zero"] = bool(fflat["valid"] and abs(fflat["beta"][1]) < 0.75)
    stratum = np.arange(40)
    draw = rng.choice(stratum, size=len(stratum), replace=True)
    checks["bootstrap_stratum_draw_size_preserved"] = bool(len(draw) == 40)
    synth_rows = []
    pairn = 0
    for lev in ("L2", "L3", "L4"):
        for i in range(40):
            pairn += 1
            key = {"carrier_id": "fakec", "fragmentation_level": lev, "expected_count": 1, "item_type": "atomic_nonemoji", "supplementary_cluster_count": 0, "match_pair_id": f"p{pairn}", "probe_type": "count", "probe_scope": "unicode"}
            synth_rows.append({**key, "era": "newest", "task_exact": True, "completion_key": f"n{pairn}"})
            synth_rows.append({**key, "era": "legacy", "task_exact": True, "completion_key": f"l{pairn}"})
    sp0 = equal_weighted_pair_delta(pair_records(synth_rows))["delta"]
    synth_rows[0]["task_exact"] = False
    sp1 = equal_weighted_pair_delta(pair_records(synth_rows))["delta"]
    checks["h2_equal_outcomes_delta_zero"] = bool(sp0 == 0)
    checks["h2_one_flip_exact_weight"] = bool(abs(sp1 - 1 / 120) < 1e-12)
    fake_gates = {"carriers": {f"fake{i}": {"carrier_id": f"fake{i}", "include_h1h2h3_primary": True, "include_h4_primary": True} for i in range(4)}}
    scalar_rows = []
    copy_rows = []
    for c in fake_gates["carriers"]:
        for era, n in (("newest", 16), ("legacy", 16)):
            scalar_rows.extend({"carrier_id": c, "probe_type": "scalar_listing", "probe_scope": "unicode", "era": era, "task_exact": False, "scalar_error_class": "other"} for _ in range(n))
        for era, n in (("newest", 60), ("legacy", 40)):
            copy_rows.extend({"carrier_id": c, "probe_type": "copy", "probe_scope": "unicode", "era": era, "task_exact": True, "copy_near_miss_class": "exact_copy", "item_type": "atomic_nonemoji"} for _ in range(n))
    exact_hh = h3_h4(scalar_rows + copy_rows, fake_gates)
    flipped_scalar = [dict(x) for x in scalar_rows]
    flipped_scalar[0]["task_exact"] = True
    flipped_h3 = h3_h4(flipped_scalar + copy_rows, fake_gates)
    flipped_copy = [dict(x) for x in copy_rows]
    flipped_copy[0]["task_exact"] = False
    flipped_h4 = h3_h4(scalar_rows + flipped_copy, fake_gates)
    checks["synthetic_exact_scalar_floor_detected"] = bool(exact_hh["h3"]["every_newest_zero_exact"] and exact_hh["h3"]["every_legacy_zero_exact"])
    checks["one_scalar_flip_breaks_exact_floor"] = bool(not flipped_h3["h3"]["every_newest_zero_exact"])
    checks["synthetic_exact_copy_ceiling_detected"] = bool(exact_hh["h4"]["every_newest_byte_exact"] and exact_hh["h4"]["every_legacy_byte_exact"])
    checks["one_copy_flip_breaks_exact_ceiling"] = bool(not flipped_h4["h4"]["every_newest_byte_exact"])
    return {"checks": checks, "implementation_mutation_checks_pass": all(checks.values()), "synthetic_fit_beta": fit["beta"].tolist(), "flat_fit_beta": fflat["beta"].tolist(), "separation_beta": fsep["beta"].tolist()}


def verdicts(gates, h1, h2, hh):
    h1p = h1["pooled"].get("bootstrap_one_sided_p_beta_gt_0")
    h2p = h2.get("tost_p")
    can_adjust = h1p is not None and h2p is not None and gates["minimum_h1h2_met"] and h1["pooled"].get("bootstrap_valid", 0) >= 4000
    if can_adjust:
        h1adj, h2adj = holm_adjust([h1p, h2p])
    else:
        h1adj = h2adj = None
    lo, hi = h1["pooled"].get("bootstrap_ci95", [None, None])
    h1_rules = {
        "minimum_validity_met": gates["minimum_h1h2_met"],
        "at_least_4000_valid_bootstrap_replicates": h1["pooled"].get("bootstrap_valid", 0) >= 4000,
        "holm_adjusted_count_significant": h1adj is not None and h1adj < 0.05,
        "bootstrap_lower_ci_exceeds_log_1_25": lo is not None and lo > math.log(1.25),
        "at_least_3_carriers_2_families_nonnegative": h1["estimable_nonnegative_carrier_count"] >= 3 and h1["nonnegative_tokenizer_family_count"] >= 2,
    }
    if not gates["minimum_h1h2_met"] or h1["pooled"].get("bootstrap_valid", 0) < 4000:
        h1v = "inconclusive"
    elif all(h1_rules.values()):
        h1v = "confirmed"
    elif hi is not None and hi < math.log(1.25) or hi is not None and hi < 0:
        h1v = "refuted"
    else:
        h1v = "inconclusive"
    h2_rules = {
        "minimum_validity_met": gates["minimum_h1h2_met"],
        "observed_delta_inside_band": h2.get("delta_newest_minus_legacy") is not None and -0.10 <= h2["delta_newest_minus_legacy"] <= 0.10,
        "holm_adjusted_tost_significant": h2adj is not None and h2adj < 0.05,
        "ci_within_band": bool(h2.get("bootstrap_ci95")) and h2["bootstrap_ci95"][0] is not None and h2["bootstrap_ci95"][0] >= -0.10 and h2["bootstrap_ci95"][1] <= 0.10,
    }
    if not gates["minimum_h1h2_met"]:
        h2v = "inconclusive"
    elif all(h2_rules.values()):
        h2v = "confirmed"
    elif h2.get("bootstrap_ci95") and h2["bootstrap_ci95"][0] is not None and (h2["bootstrap_ci95"][0] > 0.10 or h2["bootstrap_ci95"][1] < -0.10):
        h2v = "refuted"
    else:
        h2v = "inconclusive"
    if not hh["h3"]["minimum_met"]:
        h3v = "inconclusive"
    elif hh["h3"]["every_newest_zero_exact"] and hh["h3"]["every_legacy_zero_exact"]:
        h3v = "confirmed"
    else:
        h3v = "refuted"
    if not hh["h4"]["minimum_met"]:
        h4v = "inconclusive"
    elif hh["h4"]["every_newest_byte_exact"] and hh["h4"]["every_legacy_byte_exact"]:
        h4v = "confirmed"
    else:
        h4v = "refuted"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "hypotheses": {
            "H1": {"verdict": h1v, "confirmation_rules": h1_rules, "raw_count_p": h1p, "holm_adjusted_p": h1adj, "beta": h1["pooled"].get("point_beta"), "bootstrap_ci95": h1["pooled"].get("bootstrap_ci95"), "material_threshold_log_or": math.log(1.25), "refutation_logic": "upper CI below material threshold or entirely below zero"},
            "H2": {"verdict": h2v, "confirmation_rules": h2_rules, "raw_tost_p": h2p, "holm_adjusted_tost_p": h2adj, "delta": h2.get("delta_newest_minus_legacy"), "bootstrap_ci95": h2.get("bootstrap_ci95"), "ci_width": h2.get("bootstrap_ci_width"), "equivalence_band": [-0.10, 0.10]},
            "H3": {"verdict": h3v, "every_newest_zero_exact": hh["h3"]["every_newest_zero_exact"], "every_legacy_zero_exact": hh["h3"]["every_legacy_zero_exact"], "minimum_met": hh["h3"]["minimum_met"]},
            "H4": {"verdict": h4v, "every_newest_byte_exact": hh["h4"]["every_newest_byte_exact"], "every_legacy_byte_exact": hh["h4"]["every_legacy_byte_exact"], "minimum_met": hh["h4"]["minimum_met"]},
        },
        "holms_family": "H1 one-sided fragmentation count test and H2 two-sided TOST count test only; H3/H4 are exact replication tests outside the alpha family",
        "exploratory_not_confirmatory": "all subtype, formatting, scalar-error, timing, post-Unicode-17, and tokenizer-property analyses are descriptive unless explicitly pre-registered as sensitivity analyses",
    }


def main():
    global BOOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=BOOT)
    parser.add_argument("--skip-carrier-boot", action="store_true")
    args = parser.parse_args()
    BOOT = args.boot
    scored = cf.read_jsonl(OUT / "confirmatory_scored.jsonl")
    gates = gatesmod.compute_gates(scored)
    gates["gate_mutation_checks"] = gatesmod.gate_mutation_checks()
    cf.write_json(OUT / "carrier_gates.json", gates)
    valid_structural = [x["carrier_id"] for x in gates["carriers"].values() if x["include_h1h2h3_primary"]]
    h1 = analyze_h1(scored, valid_structural, not args.skip_carrier_boot)
    overlap, incomplete = h2_overlap_rows(scored, valid_structural)
    h2 = bootstrap_h2(overlap) if overlap else {"pairs": 0, "error": "no complete overlap pairs"}
    h2["incomplete_overlap_groups"] = incomplete[:20]
    h2["incomplete_overlap_group_count"] = len(incomplete)
    hh = h3_h4(scored, gates)
    sensitivity = sensitivity_analyses(scored, gates)
    exploratory = exploratory_descriptive(scored)
    mutation = implementation_mutation_checks()
    verdict = verdicts(gates, h1, h2, hh)
    analysis = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "pre_registration": "docs/CONFIRMATORY_PREREGISTRATION.md", "random_seed": SEED,
        "gates": gates, "h1": h1, "h2": h2, "h3_h4": hh, "sensitivity_analyses": sensitivity,
        "exploratory_descriptive": exploratory,
        "implementation_mutation_checks": mutation, "verdict": verdict,
    }
    cf.write_json(OUT / "confirmatory_analysis.json", analysis)
    artifacts = make_tables_figures(scored, gates, h1, h2, hh, analysis)
    cf.write_json(OUT / "confirmatory_verdict.json", verdict)
    print(json.dumps({"h1": verdict["hypotheses"]["H1"]["verdict"], "h2": verdict["hypotheses"]["H2"]["verdict"], "h3": verdict["hypotheses"]["H3"]["verdict"], "h4": verdict["hypotheses"]["H4"]["verdict"], "implementation_mutation_checks_pass": mutation["implementation_mutation_checks_pass"], "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
