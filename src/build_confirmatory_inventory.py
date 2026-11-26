from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_backend as cb
import confirmatory_common as cf

BUNDLE_CACHE = cf.SCAN_DIR / "bundle_pools.jsonl"
BUNDLE_META = cf.CONFIRMATORY / "bundle_scan.metadata.json"
GENERATOR_VERSION = "confirmatory-bundle-generator-v1"
POOL_TARGET = 300
MAX_ATTEMPTS_FACTOR = 80


def seed_for(*parts) -> int:
    return int(cf.sha_text("\x1f".join(str(x) for x in parts))[:16], 16) % (2**32)


def load_candidates():
    rows = cf.read_jsonl(cf.CONFIRMATORY / "candidate_universe.jsonl")
    if not rows:
        raise RuntimeError("candidate universe missing")
    return rows, {x["candidate_id"]: x for x in rows}


def load_isolated_scan():
    out = defaultdict(dict)
    path = cf.SCAN_DIR / "candidate_scan.jsonl"
    if not path.exists():
        raise RuntimeError("candidate scan missing")
    for line in path.open():
        x = json.loads(line)
        out[x["carrier_id"]][x["candidate_id"]] = {
            "token_count": x["token_count"],
            "tokens_per_cluster": x["tokens_per_cluster"],
            "roundtrip_exact": bool(x["tokenizer_roundtrip_exact"]),
        }
    return out


def level_integer_range(level: str, count: int) -> tuple[int, int | None]:
    lo, hi = cf.LEVELS[level]
    lo_tokens = math.ceil(lo * count - 1e-12)
    if math.isinf(hi):
        return lo_tokens, None
    hi_tokens = math.ceil(hi * count - 1e-12) - 1
    return lo_tokens, hi_tokens


def frequency_stats(items: list[dict]) -> tuple[float | None, bool]:
    vals = [x.get("frequency_row") for x in items if x.get("frequency_row") is not None]
    missing = any(x.get("frequency_row") is None for x in items) and any(x["item_type"].endswith("emoji") for x in items)
    return (sum(vals) / len(vals) if vals else None), missing


def bundle_record(carrier_id: str, era: str, item_type: str, items: list[dict], isolated_counts: list[int], token_count: int, generation_method: str) -> dict | None:
    text = "".join(x["text"] for x in items)
    count = len(items)
    if any(x in text for x in ("<IN>", "</IN>", "<OUT>", "</OUT>", "\n", "\r")):
        return None
    clusters = cf.grapheme_count(text)
    if clusters != count:
        return None
    tpc = token_count / clusters
    level = cf.fragmentation_level(tpc)
    if level is None:
        return None
    cps = [ord(c) for c in text]
    freq_mean, freq_missing = frequency_stats(items)
    supplementary = sum(int(x.get("supplementary_cluster_count", 0)) for x in items)
    return {
        "bundle_id": cf.sha_text("\x1f".join([carrier_id, era, item_type] + [x["candidate_id"] for x in items]))[:24],
        "carrier_id": carrier_id,
        "era": era,
        "item_type": item_type,
        "expected_count": count,
        "text": text,
        "text_sha256": cf.sha_text(text),
        "constituent_ids": [x["candidate_id"] for x in items],
        "constituent_texts": [x["text"] for x in items],
        "isolated_token_counts": isolated_counts,
        "isolated_tokens_per_cluster_sum": sum(isolated_counts) / count,
        "whole_probe_token_count": token_count,
        "tokens_per_cluster": tpc,
        "fragmentation_level": level,
        "utf8_bytes": len(text.encode()),
        "utf8_bytes_per_cluster": len(text.encode()) / count,
        "codepoint_count": len(cps),
        "codepoints_per_cluster": len(cps) / count,
        "mean_sequence_complexity": sum(x["complexity"] for x in items) / count,
        "supplementary_cluster_count": supplementary,
        "supplementary_cluster_fraction": supplementary / count,
        "emoji_frequency_row_mean": freq_mean if freq_mean is not None else 0,
        "emoji_frequency_any_missing": freq_missing,
        "frequency_rows": [x.get("frequency_row") for x in items],
        "primary_release_versions": [x.get("primary_release_version") for x in items],
        "emoji_versions": [x.get("emoji_version") for x in items],
        "generation_method": generation_method,
    }


def sample_composition(total: int, count: int, values: list[int], rng: random.Random) -> list[int] | None:
    values = sorted(set(values))
    if count == 1:
        return [total] if total in values else None
    random.shuffle(values)
    for first in values:
        rest = total - first
        if rest < min(values) * (count - 1) or rest > max(values) * (count - 1):
            continue
        tail = sample_composition(rest, count - 1, values, rng)
        if tail is not None:
            return [first] + tail
    return None


def direct_count1_pool(carrier_id: str, era: str, item_type: str, level: str, candidates: list[dict], scan: dict, limit: int) -> list[dict]:
    rows = []
    for cand in sorted(candidates, key=lambda x: x["candidate_id"]):
        sc = scan.get(cand["candidate_id"])
        if not sc or not sc["roundtrip_exact"]:
            continue
        tpc = sc["tokens_per_cluster"]
        if cf.fragmentation_level(tpc) != level:
            continue
        row = bundle_record(carrier_id, era, item_type, [cand], [sc["token_count"]], sc["token_count"], "direct-count1-candidate-scan")
        if row:
            rows.append(row)
    rows.sort(key=lambda x: (x["tokens_per_cluster"], x["text_sha256"]))
    if len(rows) > limit:
        rng = random.Random(seed_for("count1", carrier_id, era, item_type, level))
        rows = sorted(rng.sample(rows, limit), key=lambda x: x["text_sha256"])
    return rows


def generate_pool(carrier_id: str, backend, era: str, item_type: str, count: int, level: str, candidates: list[dict], scan: dict, pool_target: int = POOL_TARGET) -> list[dict]:
    if count == 1:
        return direct_count1_pool(carrier_id, era, item_type, level, candidates, scan, pool_target)
    by_tokens = defaultdict(list)
    for cand in candidates:
        sc = scan.get(cand["candidate_id"])
        if sc and sc["roundtrip_exact"]:
            by_tokens[sc["token_count"]].append(cand)
    if not by_tokens:
        return []
    values = sorted(by_tokens)
    min_avg = min(values)
    max_avg = max(values)
    lo, hi = cf.LEVELS[level]
    if max_avg < lo or min_avg >= hi + count:
        return []
    lo_tokens, hi_tokens = level_integer_range(level, count)
    rng = random.Random(seed_for("bundle", carrier_id, era, item_type, count, level))
    out, seen_texts, attempts = [], set(), 0
    max_attempts = pool_target * MAX_ATTEMPTS_FACTOR
    while len(out) < pool_target and attempts < max_attempts:
        attempts += 1
        if math.isinf(hi):
            target_whole = lo_tokens + rng.randrange(0, max(2, count * 2))
        elif hi_tokens is not None and hi_tokens >= lo_tokens:
            target_whole = rng.randint(lo_tokens, hi_tokens)
        else:
            continue
        merge_allowance = rng.randrange(0, count + 1)
        comp = sample_composition(target_whole + merge_allowance, count, values, rng)
        if comp is None and math.isinf(hi):
            comp = [rng.choice(values) for _ in range(count)]
        if comp is None:
            continue
        items = []
        used_indices = set()
        impossible = False
        for token_value in comp:
            pool = by_tokens.get(token_value) or by_tokens[min(values, key=lambda v: abs(v - token_value))]
            if len(pool) < count and len(pool) == len(used_indices):
                impossible = True
                break
            for _ in range(20):
                idx = rng.randrange(len(pool))
                key = (token_value, idx)
                if key not in used_indices or len(pool) == 1:
                    used_indices.add(key)
                    items.append(pool[idx])
                    break
            else:
                impossible = True
                break
        if impossible or len(items) != count or len({x["candidate_id"] for x in items}) != count:
            continue
        text = "".join(x["text"] for x in items)
        if text in seen_texts:
            continue
        encoded = cb.effective_encode(backend, text)
        row = bundle_record(carrier_id, era, item_type, items, comp, encoded["token_count"], "guided-random-homogeneous-concatenation")
        if row and row["fragmentation_level"] == level:
            seen_texts.add(text)
            out.append(row)
    out.sort(key=lambda x: (x["tokens_per_cluster"], x["utf8_bytes"], x["text_sha256"]))
    return out


def plausible_type_level(item_type: str, level: str, count: int) -> bool:
    if item_type == "atomic_nonemoji" and level in ("L3", "L4"):
        return False
    if item_type == "complex_nonemoji" and level == "L1":
        return False
    if item_type == "atomic_emoji" and level == "L4":
        return False
    return True


def build_bundle_pools(candidates_by_id: dict, scan_by_carrier: dict, reuse_cache: bool = False) -> tuple[dict, dict]:
    if reuse_cache and BUNDLE_CACHE.exists():
        pools = defaultdict(list)
        for line in BUNDLE_CACHE.open():
            row = json.loads(line)
            key = (row["carrier_id"], row["era"], row["item_type"], row["expected_count"], row["fragmentation_level"])
            pools[key].append(row)
        meta = json.loads(BUNDLE_META.read_text()) if BUNDLE_META.exists() else {}
        return pools, meta
    candidates = list(candidates_by_id.values())
    grouped = defaultdict(list)
    for row in candidates:
        grouped[(row["era"], row["item_type"])].append(row)
    pools = {}
    diagnostics = {}
    all_rows = []
    started = time.perf_counter()
    for cfg in cf.carrier_configs(include_descriptive=True):
        carrier_id = cfg["carrier_id"]
        backend = cb.load_backend(carrier_id)
        scan = scan_by_carrier[carrier_id]
        for era in ("newest", "legacy"):
            for item_type in cf.ITEM_TYPE_CYCLE:
                for count in range(1, 6):
                    for level in ("L1", "L2", "L3", "L4"):
                        if not plausible_type_level(item_type, level, count):
                            diagnostics[f"{carrier_id}/{era}/{item_type}/{count}/{level}"] = {"pool_size": 0, "skipped": "structurally implausible isolated-token envelope"}
                            continue
                        pool = generate_pool(carrier_id, backend, era, item_type, count, level, grouped[(era, item_type)], scan)
                        key = (carrier_id, era, item_type, count, level)
                        pools[key] = pool
                        diagnostics["/".join(str(x) for x in key)] = {"pool_size": len(pool)}
                        all_rows.extend(pool)
        print(json.dumps({"carrier_id": carrier_id, "bundle_rows_so_far": len(all_rows)}), flush=True)
    cf.SCAN_DIR.mkdir(parents=True, exist_ok=True)
    cf.write_jsonl(BUNDLE_CACHE, all_rows)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "pool_target_per_cell": POOL_TARGET,
        "bundle_cache_path": str(BUNDLE_CACHE),
        "bundle_cache_sha256": cf.sha_file(BUNDLE_CACHE),
        "bundle_cache_size": BUNDLE_CACHE.stat().st_size,
        "bundle_rows": len(all_rows),
        "diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    cf.write_json(BUNDLE_META, meta)
    return pools, meta


def choose_copy_rows(candidates: list[dict]) -> tuple[list[dict], dict]:
    rng = random.Random(cf.SEED + 10)
    round2_texts = {x["text"] for x in cf.read_jsonl(cf.RESULTS / "copy_pilot.jsonl")}
    quotas = {
        ("newest", "atomic_nonemoji"): 10, ("newest", "atomic_emoji"): 20, ("newest", "complex_nonemoji"): 10, ("newest", "zwj_emoji"): 20,
        ("legacy", "atomic_nonemoji"): 10, ("legacy", "atomic_emoji"): 10, ("legacy", "complex_nonemoji"): 10, ("legacy", "zwj_emoji"): 10,
    }
    rows, audit = [], {}
    for (era, item_type), quota in quotas.items():
        pool = [x for x in candidates if x["era"] == era and x["item_type"] == item_type]
        overlap = sorted((x for x in pool if x["text"] in round2_texts), key=lambda x: (x["primary_release_version"] or "", x["candidate_id"]))
        selected = overlap[:quota]
        if len(selected) < quota:
            remaining = sorted((x for x in pool if x not in selected), key=lambda x: x["candidate_id"])
            rng.shuffle(remaining)
            selected.extend(remaining[:quota - len(selected)])
        if len(selected) != quota:
            raise RuntimeError(f"copy quota deficit {era}/{item_type}: {len(selected)}/{quota}")
        rows.extend(selected)
        audit[f"{era}/{item_type}"] = {"quota": quota, "round2_overlap_selected": min(len(overlap), quota), "round2_overlap_available": len(overlap)}
    return rows, audit


def choose_scalar_rows(candidates: list[dict], excluded_texts: set[str] | None = None) -> tuple[list[dict], dict]:
    rng = random.Random(cf.SEED + 11)
    excluded_texts = set(excluded_texts or set())
    rows, audit = [], {}
    used_texts = set()
    broad_groups = {
        "newest/atomic": (["atomic_nonemoji", "atomic_emoji"], 8),
        "newest/complex_or_zwj": (["complex_nonemoji", "zwj_emoji"], 8),
        "legacy/atomic": (["atomic_nonemoji", "atomic_emoji"], 8),
        "legacy/complex_or_zwj": (["complex_nonemoji", "zwj_emoji"], 8),
    }
    for label, (types, quota) in broad_groups.items():
        era = label.split("/", 1)[0]
        selected = []
        desired = quota // len(types)
        for item_type in types:
            pool = sorted((x for x in candidates if x["era"] == era and x["item_type"] == item_type and x["text"] not in excluded_texts and x["text"] not in used_texts), key=lambda x: x["candidate_id"])
            take = min(desired, len(pool))
            chosen = rng.sample(pool, take) if take else []
            selected.extend(chosen)
            for row in chosen:
                used_texts.add(row["text"])
        if len(selected) < quota:
            pool = sorted((x for x in candidates if x["era"] == era and x["item_type"] in types and x["text"] not in excluded_texts and x["text"] not in used_texts and x not in selected), key=lambda x: x["candidate_id"])
            take = quota - len(selected)
            chosen = rng.sample(pool, take) if len(pool) >= take else []
            selected.extend(chosen)
            for row in chosen:
                used_texts.add(row["text"])
        if len(selected) != quota:
            raise RuntimeError(f"scalar quota deficit {label}: {len(selected)}/{quota}")
        rows.extend(sorted(selected, key=lambda x: x["candidate_id"]))
        audit[label] = {"quota": quota, "selected_by_type": dict(Counter(x["item_type"] for x in selected)), "copy_excluded_texts": len(excluded_texts)}
    return rows, audit


def pair_frequency_difference(new: dict, old: dict) -> float:
    a, b = new.get("emoji_frequency_row_mean"), old.get("emoji_frequency_row_mean")
    if not a and not b:
        return 0.0
    if not a or not b:
        return 100000.0
    return abs(a - b)


def pair_compatible(new: dict, old: dict) -> bool:
    return (
        new["expected_count"] == old["expected_count"]
        and new["item_type"] == old["item_type"]
        and new["fragmentation_level"] == old["fragmentation_level"]
        and new["supplementary_cluster_count"] == old["supplementary_cluster_count"]
        and abs(new["tokens_per_cluster"] - old["tokens_per_cluster"]) <= 0.5 + 1e-12
        and abs(new["utf8_bytes"] - old["utf8_bytes"]) <= 4
        and abs(new["codepoint_count"] - old["codepoint_count"]) <= 2
        and abs(new["mean_sequence_complexity"] - old["mean_sequence_complexity"]) <= 2
    )


def pair_objective(new: dict, old: dict, used_constituents: Counter) -> tuple:
    reuse = sum(used_constituents[x] for x in new["constituent_ids"] + old["constituent_ids"])
    canon = cf.canon({"new": new["text_sha256"], "old": old["text_sha256"]})
    return (
        abs(new["tokens_per_cluster"] - old["tokens_per_cluster"]),
        abs(new["utf8_bytes"] - old["utf8_bytes"]),
        abs(new["codepoint_count"] - old["codepoint_count"]),
        abs(new["mean_sequence_complexity"] - old["mean_sequence_complexity"]),
        pair_frequency_difference(new, old),
        reuse,
        cf.sha_text(canon),
    )


def find_best_pair(pools: dict, carrier_id: str, level: str, count: int, item_type: str, used_targets: set, used_constituents: Counter) -> tuple[dict, dict] | None:
    new_pool = pools.get((carrier_id, "newest", item_type, count, level), [])
    old_pool = pools.get((carrier_id, "legacy", item_type, count, level), [])
    best = None
    for new in new_pool:
        if new["text"] in used_targets:
            continue
        for old in old_pool:
            if old["text"] in used_targets or new["text"] == old["text"] or not pair_compatible(new, old):
                continue
            obj = pair_objective(new, old, used_constituents)
            if best is None or obj < best[0]:
                best = (obj, new, old)
    if best is None:
        return None
    return best[1], best[2]


def cyclic_types(preferred: str) -> list[str]:
    i = cf.ITEM_TYPE_CYCLE.index(preferred)
    return [cf.ITEM_TYPE_CYCLE[(i + j) % len(cf.ITEM_TYPE_CYCLE)] for j in range(len(cf.ITEM_TYPE_CYCLE))]


def choose_unpaired_probe(pools: dict, carrier_id: str, era: str, level: str, count: int, preferred_type: str, used_targets: set, used_constituents: Counter) -> tuple[dict, dict] | None:
    for item_type in cyclic_types(preferred_type):
        pool = pools.get((carrier_id, era, item_type, count, level), [])
        options = []
        for row in pool:
            if row["text"] in used_targets:
                continue
            reuse = sum(used_constituents[x] for x in row["constituent_ids"])
            options.append(((reuse, row["text_sha256"]), row))
        if options:
            row = min(options, key=lambda x: x[0])[1]
            return row, {"requested_type": preferred_type, "selected_type": item_type, "substituted": item_type != preferred_type, "type_attempt_order": cyclic_types(preferred_type)}
    return None


def inventory_base_row(carrier_id: str, probe_type: str, probe_scope: str, target: str, serial: int, **kwargs) -> dict:
    probe_id = f"{carrier_id}:{probe_type}:{probe_scope}:{serial:04d}"
    row = {
        "probe_id": probe_id,
        "probe_key": probe_id,
        "carrier_id": carrier_id,
        "probe_type": probe_type,
        "probe_scope": probe_scope,
        "target": target,
        "target_sha256": cf.sha_text(target),
        **kwargs,
    }
    if probe_type == "copy":
        row["expected_text"] = target
    return row


def probe_user_fields(row: dict) -> dict:
    if row["probe_type"] == "copy":
        user = cf.canon(cf.copy_messages(row["target"]))
        template_hash = cf.sha_text(cf.COPY_USER_TEMPLATE)
    elif row["probe_type"] == "count":
        user = cf.count_user_content(row["target"])
        template_hash = cf.sha_text(cf.COUNT_TEMPLATE)
    else:
        user = cf.scalar_user_content(row["target"])
        template_hash = cf.sha_text(cf.SCALAR_TEMPLATE)
    return {"user_content": user, "user_content_sha256": cf.sha_text(user), "prompt_template_sha256": template_hash}


def build_count_inventory_for_carrier(cfg: dict, pools: dict, reserved_targets: set[str] | None = None) -> tuple[list[dict], dict]:
    carrier_id = cfg["carrier_id"]
    rows, audit = [], {"overlap_substitutions": Counter(), "empty_cells": [], "pair_matching": Counter(), "newest_l1_feasible": None}
    used_targets, used_constituents = set(reserved_targets or set()), Counter()
    serial = 1
    pair_serial = 1
    for level in ("L2", "L3", "L4"):
        cell_rows = []
        for slot in range(40):
            count = (slot % 5) + 1
            preferred = cf.ITEM_TYPE_CYCLE[slot % 4]
            found = None
            selected_type = None
            attempts = []
            for item_type in cyclic_types(preferred):
                attempts.append(item_type)
                found = find_best_pair(pools, carrier_id, level, count, item_type, used_targets, used_constituents)
                if found:
                    selected_type = item_type
                    break
            if not found:
                audit["empty_cells"].append({"carrier_id": carrier_id, "level": level, "slot": slot, "expected_count": count, "preferred_item_type": preferred, "reason": "no newest/legacy pair inside frozen hard caps"})
                continue
            new, old = found
            pair_id = f"{carrier_id}:h2-{level}-{pair_serial:04d}"
            pair_serial += 1
            for era, bundle in (("newest", new), ("legacy", old)):
                row = inventory_base_row(
                    carrier_id, "count", "unicode", bundle["text"], serial,
                    era=era, item_type=selected_type, requested_item_type=preferred,
                    item_type_substituted=selected_type != preferred, type_attempt_order=attempts,
                    inventory_cell=f"count-{level}-{era}", fragmentation_level=level,
                    expected_count=count, slot_index=slot, match_pair_id=pair_id,
                    constituent_ids=bundle["constituent_ids"], isolated_token_counts=bundle["isolated_token_counts"],
                    whole_probe_token_count=bundle["whole_probe_token_count"], tokens_per_cluster=bundle["tokens_per_cluster"],
                    utf8_bytes=bundle["utf8_bytes"], utf8_bytes_per_cluster=bundle["utf8_bytes_per_cluster"],
                    codepoint_count=bundle["codepoint_count"], codepoints_per_cluster=bundle["codepoints_per_cluster"],
                    mean_sequence_complexity=bundle["mean_sequence_complexity"],
                    supplementary_cluster_count=bundle["supplementary_cluster_count"], supplementary_cluster_fraction=bundle["supplementary_cluster_fraction"],
                    emoji_frequency_row_mean=bundle["emoji_frequency_row_mean"], emoji_frequency_any_missing=bundle["emoji_frequency_any_missing"],
                    frequency_rows=bundle["frequency_rows"], primary_release_versions=bundle["primary_release_versions"], emoji_versions=bundle["emoji_versions"],
                    homogeneous_item_type=True,
                )
                row["paired_side"] = "newest" if era == "newest" else "legacy"
                cell_rows.append(row)
                serial += 1
            for bundle in (new, old):
                used_targets.add(bundle["text"])
                used_constituents.update(bundle["constituent_ids"])
            audit["pair_matching"]["pairs_constructed"] += 1
            if selected_type != preferred:
                audit["overlap_substitutions"][f"{preferred}->{selected_type}"] += 1
        rows.extend(cell_rows)
    l1_rows = []
    for slot in range(40):
        count = (slot % 5) + 1
        preferred = cf.ITEM_TYPE_CYCLE[slot % 4]
        found = choose_unpaired_probe(pools, carrier_id, "legacy", "L1", count, preferred, used_targets, used_constituents)
        if not found:
            audit["empty_cells"].append({"carrier_id": carrier_id, "level": "L1", "era": "legacy", "slot": slot, "expected_count": count, "preferred_item_type": preferred, "reason": "no unpaired legacy-L1 probe in level"})
            continue
        bundle, meta = found
        row = inventory_base_row(
            carrier_id, "count", "unicode", bundle["text"], serial,
            era="legacy", item_type=meta["selected_type"], requested_item_type=preferred,
            item_type_substituted=meta["substituted"], type_attempt_order=meta["type_attempt_order"],
            inventory_cell="count-L1-legacy", fragmentation_level="L1", expected_count=count, slot_index=slot,
            match_pair_id=None, constituent_ids=bundle["constituent_ids"], isolated_token_counts=bundle["isolated_token_counts"],
            whole_probe_token_count=bundle["whole_probe_token_count"], tokens_per_cluster=bundle["tokens_per_cluster"],
            utf8_bytes=bundle["utf8_bytes"], utf8_bytes_per_cluster=bundle["utf8_bytes_per_cluster"],
            codepoint_count=bundle["codepoint_count"], codepoints_per_cluster=bundle["codepoints_per_cluster"],
            mean_sequence_complexity=bundle["mean_sequence_complexity"], supplementary_cluster_count=bundle["supplementary_cluster_count"],
            supplementary_cluster_fraction=bundle["supplementary_cluster_fraction"], emoji_frequency_row_mean=bundle["emoji_frequency_row_mean"],
            emoji_frequency_any_missing=bundle["emoji_frequency_any_missing"], frequency_rows=bundle["frequency_rows"],
            primary_release_versions=bundle["primary_release_versions"], emoji_versions=bundle["emoji_versions"], homogeneous_item_type=True,
        )
        l1_rows.append(row)
        serial += 1
        used_targets.add(bundle["text"])
        used_constituents.update(bundle["constituent_ids"])
    rows.extend(l1_rows)

    newest_l1_possible = []
    for count in range(1, 6):
        available = sum(1 for typ in cf.ITEM_TYPE_CYCLE for x in pools.get((carrier_id, "newest", typ, count, "L1"), []) if x["text"] not in used_targets)
        newest_l1_possible.append(available >= 8)
    audit["newest_l1_feasible"] = all(newest_l1_possible)
    if audit["newest_l1_feasible"]:
        for slot in range(40):
            count = (slot % 5) + 1
            preferred = cf.ITEM_TYPE_CYCLE[slot % 4]
            found = choose_unpaired_probe(pools, carrier_id, "newest", "L1", count, preferred, used_targets, used_constituents)
            if not found:
                audit["empty_cells"].append({"carrier_id": carrier_id, "level": "L1", "era": "newest", "slot": slot, "expected_count": count, "preferred_item_type": preferred, "reason": "conditional newest-L1 slot unfilled despite feasibility"})
                continue
            bundle, meta = found
            row = inventory_base_row(
                carrier_id, "count", "unicode", bundle["text"], serial,
                era="newest", item_type=meta["selected_type"], requested_item_type=preferred,
                item_type_substituted=meta["substituted"], type_attempt_order=meta["type_attempt_order"],
                inventory_cell="count-L1-newest", fragmentation_level="L1", expected_count=count, slot_index=slot,
                match_pair_id=None, constituent_ids=bundle["constituent_ids"], isolated_token_counts=bundle["isolated_token_counts"],
                whole_probe_token_count=bundle["whole_probe_token_count"], tokens_per_cluster=bundle["tokens_per_cluster"],
                utf8_bytes=bundle["utf8_bytes"], utf8_bytes_per_cluster=bundle["utf8_bytes_per_cluster"],
                codepoint_count=bundle["codepoint_count"], codepoints_per_cluster=bundle["codepoints_per_cluster"],
                mean_sequence_complexity=bundle["mean_sequence_complexity"], supplementary_cluster_count=bundle["supplementary_cluster_count"],
                supplementary_cluster_fraction=bundle["supplementary_cluster_fraction"], emoji_frequency_row_mean=bundle["emoji_frequency_row_mean"],
                emoji_frequency_any_missing=bundle["emoji_frequency_any_missing"], frequency_rows=bundle["frequency_rows"],
                primary_release_versions=bundle["primary_release_versions"], emoji_versions=bundle["emoji_versions"], homogeneous_item_type=True,
            )
            rows.append(row)
            serial += 1
            used_targets.add(bundle["text"])
            used_constituents.update(bundle["constituent_ids"])
    else:
        for count in range(1, 6):
            audit["empty_cells"].append({"carrier_id": carrier_id, "level": "L1", "era": "newest", "expected_count": count, "reason": "structurally infeasible: fewer than eight unique valid probes available"})
    audit["overlap_substitutions"] = dict(audit["overlap_substitutions"])
    audit["count_rows"] = len(rows)
    audit["overlap_pairs"] = audit["pair_matching"]["pairs_constructed"]
    return rows, audit


def build_static_inventories(candidates: list[dict]) -> tuple[list[dict], list[dict], dict]:
    copy_items, copy_audit = choose_copy_rows(candidates)
    scalar_items, scalar_audit = choose_scalar_rows(candidates, {x["text"] for x in copy_items})
    return copy_items, scalar_items, {"copy_selection": copy_audit, "scalar_selection": scalar_audit}


def make_static_probe_rows(cfg: dict, copy_items: list[dict], scalar_items: list[dict]) -> list[dict]:
    rows = []
    serial = 1
    for control in cf.round2_ascii_copy_controls():
        rows.append(inventory_base_row(cfg["carrier_id"], "copy", "ascii", control["target"], serial, era="ASCII", item_type="ascii_control", control_slot=control["control_slot"], round2_item_id=control["round2_item_id"]))
        serial += 1
    for item in sorted(copy_items, key=lambda x: (x["era"] != "newest", x["era"], x["item_type"], x["candidate_id"])):
        rows.append(inventory_base_row(
            cfg["carrier_id"], "copy", "unicode", item["text"], serial,
            era=item["era"], item_type=item["item_type"], candidate_id=item["candidate_id"], name=item["name"],
            utf8_bytes=item["utf8_bytes"], codepoint_count=item["codepoint_count"], complexity=item["complexity"],
            primary_release_version=item["primary_release_version"], emoji_version=item.get("emoji_version"), frequency_row=item.get("frequency_row"),
        ))
        serial += 1
    for control in cf.ascii_scalar_controls():
        rows.append(inventory_base_row(cfg["carrier_id"], "scalar_listing", "ascii", control["target"], serial, era="ASCII", item_type="ascii_control", control_slot=control["control_slot"], expected_text=control["expected_text"], expected_scalars=control["expected_scalars"]))
        serial += 1
    for item in sorted(scalar_items, key=lambda x: (x["era"] != "newest", x["era"], x["item_type"], x["candidate_id"])):
        rows.append(inventory_base_row(
            cfg["carrier_id"], "scalar_listing", "unicode", item["text"], serial,
            era=item["era"], item_type=item["item_type"], candidate_id=item["candidate_id"], name=item["name"],
            expected_text=cf.scalar_text(item["text"]), expected_scalars=len(item["text"]),
            utf8_bytes=item["utf8_bytes"], codepoint_count=item["codepoint_count"], complexity=item["complexity"],
            primary_release_version=item["primary_release_version"], emoji_version=item.get("emoji_version"), frequency_row=item.get("frequency_row"),
        ))
        serial += 1
    for control in cf.ascii_count_controls():
        rows.append(inventory_base_row(cfg["carrier_id"], "count", "ascii", control["target"], serial, era="ASCII", item_type="ascii_control", control_slot=control["control_slot"], expected_count=control["expected_count"], inventory_cell="count-ascii"))
        serial += 1
    return rows


def execution_sort_key(row: dict) -> tuple:
    order = {
        ("copy", "ascii"): 1,
        ("copy", "unicode"): 2,
        ("scalar_listing", "ascii"): 3,
        ("scalar_listing", "unicode"): 4,
        ("count", "ascii"): 5,
        ("count", "unicode"): 6,
    }
    level = row.get("fragmentation_level") or ""
    return (order[(row["probe_type"], row["probe_scope"])], level, row.get("slot_index") if row.get("slot_index") is not None else -1, row["probe_id"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-bundle-cache", action="store_true")
    args = parser.parse_args()
    candidates, candidates_by_id = load_candidates()
    scan = load_isolated_scan()
    copy_items, scalar_items, static_audit = build_static_inventories(candidates)
    pools, bundle_meta = build_bundle_pools(candidates_by_id, scan, reuse_cache=args.reuse_bundle_cache)
    inventory, plan, carrier_audits = [], [], {}
    for cfg in cf.carrier_configs(include_descriptive=True):
        static_rows = make_static_probe_rows(cfg, copy_items, scalar_items)
        count_rows, count_audit = build_count_inventory_for_carrier(cfg, pools, {x["target"] for x in static_rows})
        rows = static_rows + count_rows
        for row in rows:
            row.update(probe_user_fields(row))
        inventory.extend(rows)
        rows_sorted = sorted(rows, key=execution_sort_key)
        for execution_index, row in enumerate(rows_sorted, 1):
            out = dict(row)
            out["execution_index"] = execution_index
            plan.append(out)
        carrier_audits[cfg["carrier_id"]] = {
            **count_audit,
            "static_rows": len(static_rows),
            "inventory_rows": len(rows),
            "copy_rows": sum(x["probe_type"] == "copy" for x in rows),
            "scalar_rows": sum(x["probe_type"] == "scalar_listing" for x in rows),
            "ascii_count_rows": sum(x["probe_type"] == "count" and x["probe_scope"] == "ascii" for x in rows),
        }
        print(json.dumps({"carrier_id": cfg["carrier_id"], "inventory_rows": len(rows), "overlap_pairs": count_audit["overlap_pairs"], "empty_cells": len(count_audit["empty_cells"]), "newest_l1_feasible": count_audit["newest_l1_feasible"]}), flush=True)
    cf.write_jsonl(cf.CONFIRMATORY / "inventory.jsonl", inventory)
    cf.write_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl", plan)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cf.SEED,
        "static_selection": static_audit,
        "carrier_audits": carrier_audits,
        "inventory_rows": len(inventory),
        "probe_plan_rows": len(plan),
        "inventory_sha256": cf.sha_file(cf.CONFIRMATORY / "inventory.jsonl"),
        "probe_plan_sha256": cf.sha_file(cf.CONFIRMATORY / "probe_plan.jsonl"),
    }
    cf.write_json(cf.CONFIRMATORY / "inventory_build_summary.json", audit)
    print(json.dumps({"inventory_rows": len(inventory), "probe_plan_rows": len(plan), "carriers": carrier_audits}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
