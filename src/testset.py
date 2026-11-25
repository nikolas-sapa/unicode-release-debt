from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT = ucd.ROOT
DATA = ucd.DATA
RESULTS = ucd.RESULTS
SEED = 20261125
MARK = 0x20D0
EV_ORDER = {v: i for i, v in enumerate(["E0.6", "E0.7", "E1.0", "E2.0", "E3.0", "E4.0", "E5.0", "E11.0", "E12.0", "E12.1", "E13.0", "E13.1", "E14.0", "E15.0", "E15.1", "E16.0", "E17.0", "E18.0"])}
UV_ORDER = {v: i for i, v in enumerate(["1.1", "2.0", "2.1", "3.0", "3.1", "3.2", "4.0", "4.1", "5.0", "5.1", "5.2", "6.0", "6.1", "6.2", "6.3", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "12.1", "13.0", "14.0", "15.0", "15.1", "16.0", "17.0", "18.0"])}


def jwrite(path: Path, rows):
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cps_text(cps) -> str:
    return "".join(chr(x) for x in cps)


def cphex(cps) -> list[str]:
    return [f"U+{x:04X}" for x in cps]


def date_precision(value):
    if value is None:
        return "unknown"
    return "day" if len(value) == 10 else ("month" if len(value) == 7 else "unknown")


def uvkey(version: str):
    return UV_ORDER.get(version, -1)


def grapheme_count(text: str) -> int:
    return len(regex.findall(r"\X", text))


def load_all():
    ages, _ = ucd.parse_age(DATA / "unicode" / "DerivedAge.txt")
    chars, _ = ucd.parse_unicode_data(DATA / "unicode" / "UnicodeData.txt")
    props, _ = ucd.parse_emoji_data(DATA / "unicode" / "emoji-data.txt")
    rows, _, _ = ucd.parse_emoji_test(DATA / "unicode" / "emoji-test.txt")
    universe, _ = ucd.build_emoji_universe(rows)
    return ages, chars, props, universe, ucd.parse_frequency(DATA / "unicode" / "emoji-frequency.html"), ucd.unicode_release_dates(DATA / "unicode" / "publicationdates.html")


def max_age(cps, ages):
    return max((ages.get(cp, "1.1") for cp in cps), key=uvkey)


def complexity(cps, pictographs):
    picts = {cp for cp in cps if cp in pictographs}
    return cps.count(0x200D) + sum(cp in ucd.SKIN for cp in cps) + sum(cp in range(0xFE00, 0xFE10) for cp in cps) + sum(cp == 0x20E3 or 0xE0020 <= cp <= 0xE007F for cp in cps) + max(0, len(picts) - 1)


def concept_key(name: str) -> str:
    x = re.sub(r":\s*(light|medium-light|medium|medium-dark|dark) skin tone.*$", "", name.lower())
    x = re.sub(r"\b(man|woman|person|people)\b", "person", x)
    return re.sub(r"\s+", " ", x).strip()


def base_fields(cps, ages, props, freq, udates):
    text = cps_text(cps)
    nfc = regex.sub("", "", text)
    return {
        "text": text, "codepoints": cphex(cps), "utf8_bytes": len(text.encode()),
        "contains_supplementary": any(cp > 0xFFFF for cp in cps), "plane_mix": "supplementary" if any(cp > 0xFFFF for cp in cps) else "bmp-only",
        "codepoint_count": len(cps), "grapheme_clusters": grapheme_count(text), "complexity": complexity(tuple(cps), props["Extended_Pictographic"]),
        "codepoint_max_version": max_age(tuple(cps), ages), "codepoint_max_date": udates.get(max_age(tuple(cps), ages)), "codepoint_max_date_precision": date_precision(udates.get(max_age(tuple(cps), ages))),
        "frequency_row": freq.get(text), "source_file_version": "Unicode 18.0.0 / Emoji 18.0",
    }


def emoji_item(row, ages, props, freq, udates):
    cps = row["codepoints"]
    out = base_fields(cps, ages, props, freq, udates)
    atomic = row["sequence_type"] == "atomic"
    out.update({
        "item_type": f"{row['sequence_type']}_emoji", "name": row["name"], "source_line": row["source_line"],
        "surface_family_key": row["surface_family_key"], "concept_cluster_key": concept_key(row["name"]),
        "emoji_version": row["emoji_version"], "emoji_release_date": ucd.EMOJI_DATES[row["emoji_version"]],
        "emoji_codepoint_age_mismatch": atomic and row["emoji_version"].lstrip("E") != out["codepoint_max_version"],
        "primary_release_version": out["codepoint_max_version"] if atomic else row["emoji_version"],
        "primary_release_date": out["codepoint_max_date"] if atomic else ucd.EMOJI_DATES[row["emoji_version"]], "primary_release_date_precision": date_precision(out["codepoint_max_date"] if atomic else ucd.EMOJI_DATES[row["emoji_version"]]),
        "variant_count": row["variant_count"],
    })
    return out


def nonemoji_item(cps, version, name, line, complex_item, ages, props, freq, udates):
    out = base_fields(cps, ages, props, freq, udates)
    out.update({
        "item_type": "complex_nonemoji" if complex_item else "atomic_nonemoji", "name": name + (" + COMBINING LEFT HARPOON ABOVE" if complex_item else ""),
        "source_line": line, "surface_family_key": " ".join(cphex(cps)), "concept_cluster_key": " ".join(cphex(cps)),
        "unicode_version": version, "primary_release_version": version, "primary_release_date": udates.get(version), "primary_release_date_precision": date_precision(udates.get(version)),
        "emoji_version": None, "emoji_release_date": None, "emoji_codepoint_age_mismatch": False, "variant_count": 1,
    })
    return out


def match_distance(new, old):
    fd = 0 if new.get("frequency_row") is None and old.get("frequency_row") is None else (99 if new.get("frequency_row") is None or old.get("frequency_row") is None else abs(new["frequency_row"] - old["frequency_row"]))
    return (abs(new["codepoint_count"] - old["codepoint_count"]), abs(new["grapheme_clusters"] - old["grapheme_clusters"]), abs(new["complexity"] - old["complexity"]), fd, [ord(c) for c in old["text"]])


def acceptable(dist):
    return dist[0] <= 1 and dist[1] <= 1 and dist[2] <= 2


def greedy_matches(refs, controls, needed, exact=True):
    used, out, unmatched = set(), [], []
    for new in refs:
        best = None
        for i, old in enumerate(controls):
            if i in used or old["item_type"] != new["item_type"]:
                continue
            if exact and (old["utf8_bytes"] != new["utf8_bytes"] or old["contains_supplementary"] != new["contains_supplementary"]):
                continue
            dist = match_distance(new, old)
            if best is None or dist < best[0]:
                best = (dist, i, old)
        if best and acceptable(best[0]):
            used.add(best[1])
            row = dict(best[2])
            row["matched_new_name"] = new["name"]
            row["match_distance"] = list(best[0][:4])
            out.append(row)
            if len(out) == needed:
                break
        else:
            unmatched.append(new["name"])
    return out, unmatched


def stratified_refs(items, quotas, rng):
    refs = []
    for version, quota in quotas.items():
        pool = sorted((x for x in items if x.get("emoji_version") == version or x.get("unicode_version") == version), key=lambda x: x["codepoints"])
        refs.extend(sorted(rng.sample(pool, min(quota, len(pool))), key=lambda x: x["codepoints"]))
    return refs


def eligible_chars(chars, ages, emoji_props, versions=None, supplementary=None):
    out = []
    for cp, meta in chars.items():
        version = ages.get(cp)
        if versions is not None and version not in versions:
            continue
        if supplementary is not None and (cp > 0xFFFF) != supplementary:
            continue
        if cp in emoji_props["Emoji"] or meta["category"][0] not in "LNPS" or meta["category"][0] == "M":
            continue
        out.append((cp, version, meta))
    return sorted(out)


def make_ascii(rng):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'()*+,-./:;=?@[]^_`{|}~ "
    rows, seen = [], set()
    while len(rows) < 24:
        s = "".join(rng.choice(alphabet) for _ in range(16)).strip()
        if len(s) != 16 or s in seen or any(x in s for x in ["<IN>", "</IN>", "<OUT>", "</OUT>"]):
            continue
        seen.add(s)
        rows.append({"text": s, "codepoints": cphex(tuple(ord(c) for c in s)), "item_type": "ascii_control", "name": "ASCII control", "utf8_bytes": 16, "contains_supplementary": False, "plane_mix": "bmp-only", "codepoint_count": 16, "grapheme_clusters": 16, "complexity": 0, "primary_release_version": "ASCII", "primary_release_date": None, "primary_release_date_precision": "unknown", "codepoint_max_version": "ASCII", "codepoint_max_date": None, "emoji_version": None, "emoji_release_date": None, "surface_family_key": "ASCII", "concept_cluster_key": "ASCII", "frequency_row": None, "source_line": None, "source_file_version": "generated", "variant_count": 1, "emoji_codepoint_age_mismatch": False, "copy_role": "ascii_control"})
    return rows


def reorder_zwj(cps):
    blocks, cur = [], []
    for cp in cps:
        if cp == 0x200D:
            blocks.append(cur)
            cur = []
        else:
            cur.append(cp)
    blocks.append(cur)
    if len(blocks) < 2 or any(not x for x in blocks):
        return None
    out = []
    for i, block in enumerate(reversed(blocks)):
        if i:
            out.append(0x200D)
        out.extend(block)
    return tuple(out)


def compare_label(a, b):
    if a == b:
        return "SAME"
    return "REORDERED" if sorted(a) == sorted(b) else "DIFFERENT"


def build_base():
    ages, chars, props, universe, freq, udates = load_all()
    rng = random.Random(SEED)
    emoji = [emoji_item(r, ages, props, freq, udates) for r in universe]
    new_emoji = [x for x in emoji if (x["emoji_version"] == "E15.1" and x["item_type"] == "zwj_emoji") or (x["emoji_version"] == "E17.0" and x["item_type"] == "zwj_emoji") or (x["emoji_version"] in ("E16.0", "E17.0", "E18.0") and x["item_type"] == "atomic_emoji")]
    old_emoji = [x for x in emoji if x["item_type"] in ("atomic_emoji", "zwj_emoji") and EV_ORDER[x["emoji_version"]] <= EV_ORDER["E15.0"]]
    atomic_refs = stratified_refs([x for x in new_emoji if x["item_type"] == "atomic_emoji"], {"E16.0": 5, "E17.0": 5, "E18.0": 6}, rng)
    zwj_refs = stratified_refs([x for x in new_emoji if x["item_type"] == "zwj_emoji"], {"E15.1": 11, "E17.0": 5}, rng)
    atomic_controls, atomic_unmatched = greedy_matches(atomic_refs, old_emoji, 16)
    zwj_controls, zwj_unmatched = greedy_matches(zwj_refs + [x for x in new_emoji if x["item_type"] == "zwj_emoji" and x not in zwj_refs], old_emoji, 16)
    if len(atomic_controls) != 16 or len(zwj_controls) != 16:
        raise RuntimeError(f"emoji control matching failed: atomic={len(atomic_controls)}, zwj={len(zwj_controls)}")

    new_versions = ["15.1", "16.0", "17.0", "18.0"]
    atomic_new, complex_new = [], []
    complex_bases_used = set()
    for version in new_versions:
        pool_all = eligible_chars(chars, ages, props, [version])
        supp = [x for x in pool_all if x[0] > 0xFFFF]
        pool = supp if len(supp) >= 8 else pool_all
        chosen = rng.sample(pool, 4)
        atomic_new.extend(nonemoji_item((cp,), version, meta["name"], meta["source_line"], False, ages, props, freq, udates) for cp, _, meta in chosen)
        rest = [x for x in pool if x[0] not in {c[0] for c in chosen} and x[0] not in complex_bases_used]
        chosen_complex = rng.sample(rest, 4)
        complex_bases_used.update(x[0] for x in chosen_complex)
        complex_new.extend(nonemoji_item((cp, MARK), version, meta["name"], meta["source_line"], True, ages, props, freq, udates) for cp, _, meta in chosen_complex)

    old_pool = eligible_chars(chars, ages, props, [v for v in UV_ORDER if uvkey(v) <= uvkey("15.0") and v != "1.1"] + ["1.1"])
    supplementary_old = [x for x in old_pool if x[0] > 0xFFFF][:2000]
    old_atomic_candidates = [nonemoji_item((cp,), ages[cp], meta["name"], meta["source_line"], False, ages, props, freq, udates) for cp, _, meta in supplementary_old]
    old_complex_candidates = [nonemoji_item((cp, MARK), ages[cp], meta["name"], meta["source_line"], True, ages, props, freq, udates) for cp, _, meta in supplementary_old]
    old_atomic, un_a = greedy_matches(atomic_new, old_atomic_candidates, 16)
    old_complex, un_c = greedy_matches(complex_new, old_complex_candidates, 16)
    if len(old_atomic) != 16 or len(old_complex) != 16:
        raise RuntimeError(f"nonemoji control matching failed: atomic={len(old_atomic)}, complex={len(old_complex)}")

    copy_rows = []
    for row in new_emoji:
        row["copy_role"] = "new_emoji"
    for row in atomic_controls + zwj_controls:
        row["copy_role"] = "old_emoji_control"
    for row in atomic_new + complex_new:
        row["copy_role"] = "new_nonemoji"
    for row in old_atomic + old_complex:
        row["copy_role"] = "old_nonemoji_control"
    copy_rows = new_emoji + atomic_controls + zwj_controls + atomic_new + complex_new + old_atomic + old_complex + make_ascii(rng)
    for i, row in enumerate(copy_rows, 1):
        row["item_id"] = f"copy-{i:03d}"
        row["panel"] = "copy_pilot"
    if len(copy_rows) != 182:
        raise RuntimeError(f"copy pilot has {len(copy_rows)} rows, expected 182")

    e151 = sorted((x for x in new_emoji if x["emoji_version"] == "E15.1" and x["item_type"] == "zwj_emoji"), key=lambda x: x["codepoints"])
    e17 = sorted((x for x in new_emoji if x["emoji_version"] == "E17.0" and x["item_type"] == "zwj_emoji"), key=lambda x: x["codepoints"])
    newest_decomp = rng.sample(e151, 13) + e17
    legacy_zwj = [x for x in emoji if x["item_type"] == "zwj_emoji" and EV_ORDER[x["emoji_version"]] <= EV_ORDER["E13.1"]]
    decomp_controls, decomp_unmatched = greedy_matches(newest_decomp, legacy_zwj, 24, exact=False)
    if len(decomp_controls) != 24:
        raise RuntimeError("decomposition control matching failed")
    structural = []
    for i, (new, old) in enumerate(zip(newest_decomp, decomp_controls), 1):
        for era, row in (("newest", new), ("legacy", old)):
            structural.append({"probe_type": "decomposition", "probe_id": f"decomp-{i:02d}-{era}", "era": era, "target": row["text"], "expected_codepoints": row["codepoints"], "source_item_name": row["name"], "pair_id": f"decomp-{i:02d}"})

    comp_rows = []
    for i, (new, old) in enumerate(zip(newest_decomp[:12], decomp_controls[:12]), 1):
        for era, row in (("newest", new), ("legacy", old)):
            a = tuple(int(x[2:], 16) for x in row["codepoints"])
            desired = ["SAME", "REORDERED", "DIFFERENT"][(i - 1) % 3]
            if desired == "SAME":
                b = a
            elif desired == "REORDERED":
                b = reorder_zwj(a)
                if b is None or b == a:
                    b = tuple(x if j else (0x1F600 if x not in (0x1F600,) else 0x1F601) for j, x in enumerate(a))
            else:
                replacement = 0x1F600 if 0x1F600 not in a else 0x1F601
                b = (replacement,) + a[1:]
            label = compare_label(a, b)
            comp_rows.append({"probe_type": "comparison", "probe_id": f"compare-{i:02d}-{era}", "era": era, "a_text": cps_text(a), "b_text": cps_text(b), "expected_label": label, "requested_label": desired, "source_item_name": row["name"], "pair_id": f"compare-{i:02d}"})
    structural.extend(comp_rows)

    # Count candidates are scanned before final count-bundle construction.
    legacy_complex_pool = []
    legacy_base_pool = [x for x in old_pool if ages[x[0]] and uvkey(ages[x[0]]) <= uvkey("13.0")]
    for cp, _, meta in rng.sample(legacy_base_pool, 60):
        legacy_complex_pool.append(nonemoji_item((cp, MARK), ages[cp], meta["name"], meta["source_line"], True, ages, props, freq, udates))
    legacy_emoji_pool = [x for x in emoji if x["item_type"] in ("atomic_emoji", "zwj_emoji") and EV_ORDER[x["emoji_version"]] <= EV_ORDER["E13.1"] and x["grapheme_clusters"] == 1]
    legacy_emoji_pool = rng.sample(sorted(legacy_emoji_pool, key=lambda x: x["codepoints"]), min(180, len(legacy_emoji_pool)))
    candidates = []
    for era, pool in (("new", new_emoji + complex_new), ("legacy", legacy_emoji_pool + legacy_complex_pool)):
        for j, row in enumerate(sorted(pool, key=lambda x: (x["item_type"], x["codepoints"])), 1):
            if row["grapheme_clusters"] == 1:
                candidates.append({"candidate_id": f"{era}-{j:04d}", "era": era, "text": row["text"], "codepoints": row["codepoints"], "item_type": row["item_type"], "utf8_bytes": row["utf8_bytes"], "name": row["name"], "grapheme_clusters": 1})
    jwrite(RESULTS / "count_candidates.jsonl", candidates)
    jwrite(RESULTS / "copy_pilot.jsonl", copy_rows)
    jwrite(RESULTS / "structural_base.jsonl", structural)
    audit = {
        "seed": SEED, "copy_count": len(copy_rows), "copy_roles": dict(Counter(x["copy_role"] for x in copy_rows)),
        "new_emoji_counts": {f"{k[0]}/{k[1]}": v for k, v in Counter((x["emoji_version"], x["item_type"]) for x in new_emoji).items()},
        "new_nonemoji_counts": {f"{k[0]}/{k[1]}": v for k, v in Counter((x["unicode_version"], x["item_type"]) for x in atomic_new + complex_new).items()},
        "emoji_matching": {"atomic_reference_rule": "stratified random quotas 5/5/6 from E16/E17/E18", "zwj_reference_rule": "stratified random quotas 11/5 from E15.1/E17, then deterministic fallback over remaining new items", "atomic_unmatched": atomic_unmatched, "zwj_unmatched": zwj_unmatched},
        "nonemoji_matching": {"atomic_unmatched": un_a, "complex_unmatched": un_c},
        "decomp_unmatched": decomp_unmatched, "count_candidates": dict(Counter(x["era"] for x in candidates)),
        "strict_atomic_e16_e18_age_mismatches": [x["name"] for x in new_emoji if x["item_type"] == "atomic_emoji" and x["emoji_codepoint_age_mismatch"]],
    }
    (RESULTS / "testset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=list))
    write_copy_csv(copy_rows)
    print(json.dumps({"copy_rows": len(copy_rows), "count_candidates": len(candidates), "structural_base": len(structural)}, indent=2))


def write_copy_csv(rows):
    fields = ["item_id", "copy_role", "item_type", "name", "text", "utf8_bytes", "plane_mix", "codepoint_count", "grapheme_clusters", "complexity", "primary_release_version", "primary_release_date", "emoji_version", "frequency_row", "matched_new_name"]
    with (RESULTS / "copy_pilot.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_count_bundles():
    candidates = {x["candidate_id"]: x for x in read_jsonl(RESULTS / "count_candidates.jsonl")}
    scans = read_jsonl(RESULTS / "count_candidate_scan.jsonl")
    structural = read_jsonl(RESULTS / "structural_base.jsonl")
    rng = random.Random(SEED)
    count_rows = []
    pattern = [1, 2, 3, 4, 5, 2, 3, 4, 5]
    for tokenizer_id in sorted({x["tokenizer_id"] for x in scans}):
        rows = [x for x in scans if x["tokenizer_id"] == tokenizer_id]
        bundle_options = {}
        diagnostics = {}
        for era in ("new", "legacy"):
            era_rows = sorted((x for x in rows if x["era"] == era), key=lambda x: x["candidate_id"])
            median = statistics.median(x["tokens_per_cluster"] for x in era_rows)
            pools = {"low": [x for x in era_rows if x["tokens_per_cluster"] <= median], "high": [x for x in era_rows if x["tokens_per_cluster"] > median]}
            fallback = False
            if len(pools["low"]) < 5 or len(pools["high"]) < 5:
                ordered = sorted(era_rows, key=lambda x: (x["tokens_per_cluster"], x["candidate_id"]))
                half = len(ordered) // 2
                pools = {"low": ordered[:half], "high": ordered[half:]}
                fallback = True
            diagnostics[era] = {"candidate_median_tokens_per_cluster": median, "pool_sizes": {k: len(v) for k, v in pools.items()}, "rank_fallback_used": fallback}
            for frag, pool in pools.items():
                for slot, n in enumerate(pattern, 1):
                    options, seen = [], set()
                    for _ in range(240):
                        if len(pool) < n:
                            break
                        chosen = rng.sample(pool, n)
                        text = "".join(x["text"] for x in chosen)
                        if text in seen or grapheme_count(text) != n:
                            continue
                        seen.add(text)
                        options.append({"text": text, "utf8_bytes": len(text.encode()), "constituent_ids": [x["candidate_id"] for x in chosen], "mean_constituent_tpc": sum(x["tokens_per_cluster"] for x in chosen) / n})
                        if len(options) == 40:
                            break
                    if not options:
                        raise RuntimeError(f"no valid count bundle for {tokenizer_id} {era} {frag} slot {slot}")
                    bundle_options[(era, frag, slot)] = options
        for frag in ("low", "high"):
            for slot, n in enumerate(pattern, 1):
                best = None
                for new_opt, old_opt in product(bundle_options[("new", frag, slot)], bundle_options[("legacy", frag, slot)]):
                    key = (abs(new_opt["utf8_bytes"] - old_opt["utf8_bytes"]), new_opt["text"], old_opt["text"])
                    if best is None or key < best[0]:
                        best = (key, new_opt, old_opt)
                pair_id = f"count-{tokenizer_id}-{frag}-{slot:02d}"
                for era, opt in (("newest", best[1]), ("legacy", best[2])):
                    count_rows.append({"probe_type": "count", "probe_id": f"{pair_id}-{era}", "pair_id": pair_id, "tokenizer_id": tokenizer_id, "era": era, "fragmentation_group": frag, "target": opt["text"], "expected_count": n, "utf8_bytes": opt["utf8_bytes"], "pair_byte_length_difference": abs(best[1]["utf8_bytes"] - best[2]["utf8_bytes"]), "constituent_ids": opt["constituent_ids"], "construction_mean_tokens_per_cluster": opt["mean_constituent_tpc"], "candidate_pool_diagnostics": diagnostics["new" if era == "newest" else "legacy"]})
    structural.extend(count_rows)
    scan_units = make_scan_units(read_jsonl(RESULTS / "copy_pilot.jsonl"), structural)
    jwrite(RESULTS / "structural_probes.jsonl", structural)
    jwrite(RESULTS / "scan_units.jsonl", scan_units)
    print(json.dumps({"structural_probes": len(structural), "count_bundles": len(count_rows), "scan_units": len(scan_units)}, indent=2))


def make_scan_units(copy_rows, structural):
    units = []
    for row in copy_rows:
        units.append({"scan_id": "copy:" + row["item_id"], "probe_type": "copy", "item_id": row["item_id"], "text": row["text"], "item_type": row["item_type"], "era": "newest" if row["copy_role"].startswith("new_") else ("ascii" if row["item_type"] == "ascii_control" else "old_control"), "release_version": row["primary_release_version"], "emoji_version": row["emoji_version"], "copy_role": row["copy_role"], "grapheme_clusters": row["grapheme_clusters"], "utf8_bytes": row["utf8_bytes"], "codepoints": row["codepoints"]})
    for row in structural:
        if row["probe_type"] in ("decomposition", "count"):
            target = row["target"]
            units.append({"scan_id": row["probe_type"] + ":" + row["probe_id"], "probe_type": row["probe_type"], "probe_id": row["probe_id"], "text": target, "item_type": "structural_bundle" if row["probe_type"] == "count" else "zwj_emoji", "era": row["era"], "release_version": "mixed", "emoji_version": "mixed", "grapheme_clusters": grapheme_count(target), "utf8_bytes": len(target.encode()), "codepoints": cphex(tuple(ord(c) for c in target))})
        elif row["probe_type"] == "comparison":
            for side in ("a", "b"):
                target = row[side + "_text"]
                units.append({"scan_id": f"comparison:{row['probe_id']}:{side.upper()}", "probe_type": "comparison", "probe_id": row["probe_id"], "side": side.upper(), "text": target, "item_type": "zwj_emoji", "era": row["era"], "release_version": "mixed", "emoji_version": "mixed", "grapheme_clusters": grapheme_count(target), "utf8_bytes": len(target.encode()), "codepoints": cphex(tuple(ord(c) for c in target))})
    for i, target in enumerate(["café", "Ångström", "naïve", "žluťoučký", "한"], 1):
        units.append({"scan_id": f"normalization:nfc-{i:02d}", "probe_type": "normalization_diagnostic", "probe_id": f"nfc-{i:02d}", "text": target, "item_type": "canonical_pair_diagnostic", "era": "diagnostic", "release_version": "legacy", "emoji_version": None, "grapheme_clusters": grapheme_count(target), "utf8_bytes": len(target.encode()), "codepoints": cphex(tuple(ord(c) for c in target))})
    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["base", "final"])
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    build_base() if args.stage == "base" else build_count_bundles()


if __name__ == "__main__":
    main()
