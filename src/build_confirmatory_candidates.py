from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf
import testset
import ucd_build as ucd

MARK = 0x20D0
NEWEST_COMPLEX_QUOTA = 2500
LEGACY_ATOMIC_QUOTA = 3000
LEGACY_COMPLEX_QUOTA = 1500
DEPRECATED_SCALARS = {0x0149, 0x0673, 0x0F77, 0x0F79, 0x17A3, 0x17A4, 0x2329, 0x232A}
COMPLEX_QUOTAS = {"newest": NEWEST_COMPLEX_QUOTA, "legacy": LEGACY_COMPLEX_QUOTA}
ATOMIC_QUOTAS = {"legacy": LEGACY_ATOMIC_QUOTA}


def uvkey(version: str) -> int:
    return testset.uvkey(version)


def evkey(version: str) -> int:
    return testset.EV_ORDER[version]


def is_noncharacter(cp: int) -> bool:
    return 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE


def eligible_scalar(cp: int, meta: dict, ages: dict[int, str], emoji_props) -> bool:
    if cp in emoji_props["Emoji"] or cp in DEPRECATED_SCALARS:
        return False
    if 0xD800 <= cp <= 0xDFFF or is_noncharacter(cp):
        return False
    if meta["category"][0] not in "LNPS":
        return False
    if unicodedata2.normalize("NFC", chr(cp)) != chr(cp):
        return False
    return ages.get(cp) is not None


def stratified_sample(entries: list[tuple[int, str, dict]], quota: int, rng: random.Random) -> list[tuple[int, str, dict]]:
    if len(entries) <= quota:
        return sorted(entries)
    groups = defaultdict(list)
    for row in entries:
        cp, version, meta = row
        groups[(version, cp > 0xFFFF)].append(row)
    total = len(entries)
    allocations = {}
    remainders = []
    used = 0
    for key, rows in sorted(groups.items()):
        exact = quota * len(rows) / total
        n = min(len(rows), int(exact))
        allocations[key] = n
        used += n
        remainders.append((exact - int(exact), key))
    for _, key in sorted(remainders, reverse=True):
        if used >= quota:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            used += 1
    out = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows)
        take = allocations[key]
        out.extend(rng.sample(rows, take) if take < len(rows) else rows)
    return sorted(out)


def nonemoji_row(cp: int, complex_item: bool, ages, chars, props, freq, udates) -> dict:
    meta = chars[cp]
    version = ages[cp]
    cps = (cp, MARK) if complex_item else (cp,)
    row = testset.nonemoji_item(cps, version, meta["name"], meta["source_line"], complex_item, ages, props, freq, udates)
    row["base_codepoint"] = cp
    row["combining_mark"] = MARK if complex_item else None
    return row


def finalize_candidate(row: dict, era: str, candidate_id: str) -> dict:
    cps = [int(x[2:], 16) for x in row["codepoints"]]
    supplementary = any(cp > 0xFFFF for cp in cps)
    out = {
        "candidate_id": candidate_id,
        "era": era,
        "item_type": row["item_type"],
        "text": row["text"],
        "text_sha256": cf.sha_text(row["text"]),
        "codepoints": row["codepoints"],
        "codepoint_count": row["codepoint_count"],
        "utf8_bytes": row["utf8_bytes"],
        "grapheme_clusters": row["grapheme_clusters"],
        "complexity": row["complexity"],
        "supplementary_cluster_count": int(supplementary),
        "plane_mix": row["plane_mix"],
        "frequency_row": row.get("frequency_row"),
        "name": row["name"],
        "source_line": row["source_line"],
        "surface_family_key": row["surface_family_key"],
        "concept_cluster_key": row["concept_cluster_key"],
        "emoji_version": row.get("emoji_version"),
        "emoji_release_date": row.get("emoji_release_date"),
        "unicode_version": row.get("unicode_version"),
        "primary_release_version": row.get("primary_release_version"),
        "primary_release_date": row.get("primary_release_date"),
        "primary_release_date_precision": row.get("primary_release_date_precision"),
        "codepoint_max_version": row.get("codepoint_max_version"),
        "codepoint_max_date": row.get("codepoint_max_date"),
        "emoji_codepoint_age_mismatch": row.get("emoji_codepoint_age_mismatch"),
        "variant_count": row.get("variant_count"),
    }
    return out


def build():
    ages, chars, props, emoji_universe, freq, udates = testset.load_all()
    rng = random.Random(cf.SEED)
    eligible = sorted((cp, ages[cp], meta) for cp, meta in chars.items() if eligible_scalar(cp, meta, ages, props))
    newest_entries = [x for x in eligible if x[1] in cf.NEWEST_UNICODE_VERSIONS]
    legacy_entries = [x for x in eligible if uvkey(x[1]) <= uvkey("15.0")]
    newest_complex_entries = stratified_sample(newest_entries, NEWEST_COMPLEX_QUOTA, rng)
    legacy_atomic_entries = stratified_sample(legacy_entries, LEGACY_ATOMIC_QUOTA, rng)
    legacy_complex_entries = stratified_sample(legacy_entries, LEGACY_COMPLEX_QUOTA, rng)

    nonemoji_rows = []
    nonemoji_rows.extend((nonemoji_row(cp, False, ages, chars, props, freq, udates), "newest") for cp, _, _ in newest_entries)
    nonemoji_rows.extend((nonemoji_row(cp, True, ages, chars, props, freq, udates), "newest") for cp, _, _ in newest_complex_entries)
    nonemoji_rows.extend((nonemoji_row(cp, False, ages, chars, props, freq, udates), "legacy") for cp, _, _ in legacy_atomic_entries)
    nonemoji_rows.extend((nonemoji_row(cp, True, ages, chars, props, freq, udates), "legacy") for cp, _, _ in legacy_complex_entries)

    emoji_rows = []
    for raw in emoji_universe:
        if raw["sequence_type"] not in ("atomic", "zwj"):
            continue
        row = testset.emoji_item(raw, ages, props, freq, udates)
        era = "newest" if row["emoji_version"] in cf.NEWEST_EMOJI_VERSIONS else "legacy"
        emoji_rows.append((row, era))

    grouped = defaultdict(list)
    for row, era in nonemoji_rows + emoji_rows:
        grouped[(era, row["item_type"])].append(row)

    candidates = []
    for (era, item_type), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: (x["codepoints"], x["text"]))
        seen = set()
        serial = 1
        for row in rows:
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            candidates.append(finalize_candidate(row, era, f"{era}-{item_type}-{serial:05d}"))
            serial += 1

    counts = Counter((x["era"], x["item_type"]) for x in candidates)
    audit = {
        "seed": cf.SEED,
        "candidate_rows": len(candidates),
        "composition": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "quotas": {
            "newest_atomic_nonemoji": "all eligible newest scalars",
            "newest_complex_nonemoji": NEWEST_COMPLEX_QUOTA,
            "legacy_atomic_nonemoji": LEGACY_ATOMIC_QUOTA,
            "legacy_complex_nonemoji": LEGACY_COMPLEX_QUOTA,
            "emoji": "all fully-qualified atomic and ZWJ surface families",
        },
        "input_rows": {
            "eligible_scalars": len(eligible),
            "newest_eligible_scalars": len(newest_entries),
            "legacy_eligible_scalars": len(legacy_entries),
            "newest_complex_bases": len(newest_complex_entries),
            "legacy_atomic_bases": len(legacy_atomic_entries),
            "legacy_complex_bases": len(legacy_complex_entries),
        },
        "exclusion_rules": [
            "emoji-property scalars excluded from non-emoji candidates",
            "general categories outside L/N/P/S excluded",
            "surrogates, noncharacters, unassigned scalars, whitespace/control/mark categories excluded by construction",
            "documented deprecated scalar exclusions applied",
            "non-NFC-stable scalars excluded because byte-exact target identity is otherwise altered by tokenizer NFC normalization",
        ],
        "design_reference_comparison": "execution uses a superset of the design scan for legacy emoji and non-emoji pools; quotas were fixed before model execution",
    }
    cf.CONFIRMATORY.mkdir(parents=True, exist_ok=True)
    cf.write_jsonl(cf.CONFIRMATORY / "candidate_universe.jsonl", candidates)
    cf.write_json(cf.CONFIRMATORY / "candidate_universe_build_summary.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    build()
