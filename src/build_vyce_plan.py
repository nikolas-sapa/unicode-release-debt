from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2
import vyce_common as vc

REFERENCE_MODEL = "qwen2.5-3b-instruct-bf16"
REFERENCE_SCAN_TOKENIZER = "qwen2.5-0.5b-instruct"
COPY_PROTOCOL = "fewshot_copy_v5_multiturn_cueless"
PLAN_PATH = vc.VYCE / "vyce_probe_plan.jsonl"
MANIFEST_PATH = vc.VYCE / "vyce_selection_manifest.json"


def item_num(item_id: str) -> int:
    return int(item_id.rsplit("-", 1)[1])


def even_select(rows: list[dict], n: int) -> list[dict]:
    rows = sorted(rows, key=lambda x: x.get("item_id") or x.get("probe_id"))
    if len(rows) < n:
        raise RuntimeError(f"selection quota {n} exceeds available rows {len(rows)}")
    if len(rows) == n:
        return rows
    indexes = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [0]
    if len(indexes) != n:
        raise RuntimeError("even selection index collision")
    return [rows[i] for i in indexes]


def quota_select(copy_rows: list[dict], copy_role: str, item_type: str, n: int, emoji_version: str | None = None) -> list[dict]:
    rows = [x for x in copy_rows if x["copy_role"] == copy_role and x["item_type"] == item_type]
    if emoji_version is not None:
        rows = [x for x in rows if x.get("item", {}).get("emoji_version") == emoji_version]
    return even_select(rows, n)


def select_copy(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    copy_rows = [x for x in rows if x["probe_type"] == "copy"]
    selected = []
    strata = []
    ascii_rows = sorted([x for x in copy_rows if x["copy_role"] == "ascii_control"], key=lambda x: item_num(x["item_id"]))
    if len(ascii_rows) != 24:
        raise RuntimeError(f"expected 24 ASCII controls, got {len(ascii_rows)}")
    for row in ascii_rows:
        row["vyce_selection_stratum"] = "ascii_control_all"
    selected.extend(ascii_rows)
    strata.append({"stratum": "ascii_control_all", "n": len(ascii_rows), "item_ids": [x["item_id"] for x in ascii_rows]})
    quotas = [
        ("new_emoji_E15.1_zwj", "new_emoji", "zwj_emoji", "E15.1", 7),
        ("new_emoji_E16_atomic", "new_emoji", "atomic_emoji", "E16.0", 3),
        ("new_emoji_E17_zwj", "new_emoji", "zwj_emoji", "E17.0", 4),
        ("new_emoji_E17_atomic", "new_emoji", "atomic_emoji", "E17.0", 2),
        ("new_emoji_E18_atomic", "new_emoji", "atomic_emoji", "E18.0", 4),
        ("new_nonemoji_atomic", "new_nonemoji", "atomic_nonemoji", None, 4),
        ("new_nonemoji_complex", "new_nonemoji", "complex_nonemoji", None, 4),
        ("old_emoji_atomic", "old_emoji_control", "atomic_emoji", None, 3),
        ("old_emoji_zwj", "old_emoji_control", "zwj_emoji", None, 3),
        ("old_nonemoji_atomic", "old_nonemoji_control", "atomic_nonemoji", None, 3),
        ("old_nonemoji_complex", "old_nonemoji_control", "complex_nonemoji", None, 3),
    ]
    for label, role, typ, version, n in quotas:
        rows_sel = quota_select(copy_rows, role, typ, n, version)
        for row in rows_sel:
            row["vyce_selection_stratum"] = label
        selected.extend(rows_sel)
        strata.append({"stratum": label, "n": len(rows_sel), "item_ids": [x["item_id"] for x in rows_sel]})
    if len(selected) != 64:
        raise RuntimeError(f"copy subsample has {len(selected)} rows, expected 64")
    if len({x["probe_key"] for x in selected}) != len(selected):
        raise RuntimeError("duplicate copy probe keys in vyce selection")
    zwj_count = sum("\u200d" in x["target"] for x in selected if x["copy_role"] != "ascii_control")
    vs16_count = sum("\ufe0f" in x["target"] for x in selected if x["copy_role"] != "ascii_control")
    strata.append({"stratum": "copy_selected_composition", "zwj_target_count": zwj_count, "vs16_target_count": vs16_count})
    return selected, strata


def select_decomposition(rows: list[dict]) -> tuple[list[dict], dict]:
    decomp = [x for x in rows if x["probe_type"] == "decomposition"]
    pairs = defaultdict(list)
    for row in decomp:
        pairs[row["pair_id"]].append(row)
    pair_ids = sorted(pairs, key=lambda x: int(x.rsplit("-", 1)[1]))
    if len(pair_ids) != 24 or any(len(pairs[x]) != 2 for x in pair_ids):
        raise RuntimeError("decomposition pair structure mismatch")
    chosen_ids = [pair_ids[i] for i in sorted({round(i * (len(pair_ids) - 1) / 7) for i in range(8)})]
    selected = []
    for pair_id in chosen_ids:
        selected.extend(sorted(pairs[pair_id], key=lambda x: x["era"], reverse=True))
    for row in selected:
        row["vyce_selection_stratum"] = "decomposition_matched_pair"
    return selected, {"pair_ids": chosen_ids, "rows": len(selected), "note": "8 matched newest/legacy pairs, evenly spaced across the 24 frozen round-2 pairs"}


def select_count(rows: list[dict]) -> tuple[list[dict], dict]:
    count = [x for x in rows if x["probe_type"] == "count" and x.get("tokenizer_id") == REFERENCE_SCAN_TOKENIZER]
    pairs = defaultdict(list)
    for row in count:
        pairs[(row["fragmentation_group"], row["pair_id"])].append(row)
    selected = []
    detail = {}
    for frag in ("low", "high"):
        keys = sorted([x for x in pairs if x[0] == frag], key=lambda x: int(x[1].rsplit("-", 1)[1]))
        if len(keys) != 9:
            raise RuntimeError(f"{frag} count pair count {len(keys)} != 9")
        idx = sorted({round(i * (len(keys) - 1) / 5) for i in range(6)})
        chosen = [keys[i] for i in idx]
        detail[frag] = [x[1] for x in chosen]
        for key in chosen:
            rows_pair = sorted(pairs[key], key=lambda x: x["era"], reverse=True)
            if len(rows_pair) != 2 or {x["era"] for x in rows_pair} != {"newest", "legacy"}:
                raise RuntimeError(f"count pair structure mismatch for {key}")
            selected.extend(rows_pair)
    for row in selected:
        row["vyce_selection_stratum"] = "count_" + row["fragmentation_group"] + "_matched_pair"
    return selected, {"pair_ids_by_fragmentation": detail, "rows": len(selected), "reference_scan_tokenizer": REFERENCE_SCAN_TOKENIZER, "note": "6 low plus 6 high fragmentation pairs, each containing one newest and one legacy probe, from the frozen qwen2.5 round-2 count strata"}


def verify_against_round2(selected: list[dict]) -> dict:
    reference = [x for x in r2.read_jsonl(r2.ROUND2 / "probe_plan_round2.jsonl") if x["model_id"] == REFERENCE_MODEL]
    by_key = {x["probe_key"]: x for x in reference}
    mismatches = []
    for row in selected:
        ref = by_key.get(row["probe_key"])
        if ref is None:
            mismatches.append({"probe_key": row["probe_key"], "problem": "missing_from_round2_plan"})
            continue
        for key in ("user_content_sha256", "target", "probe_type", "copy_protocol"):
            if row.get(key) != ref.get(key):
                mismatches.append({"probe_key": row["probe_key"], "problem": f"{key} mismatch"})
    if mismatches:
        raise RuntimeError(json.dumps(mismatches[:5], ensure_ascii=False))
    return {"reference_model": REFERENCE_MODEL, "reference_rows": len(reference), "selected_rows_verified": len(selected), "all_selected_probe_keys_exist_in_round2": True, "user_content_sha256_exact_for_all_selected_rows": True}


def build_plan() -> dict:
    rows = r2.probe_rows(REFERENCE_SCAN_TOKENIZER, COPY_PROTOCOL)
    copy_rows, copy_strata = select_copy(rows)
    decomp_rows, decomp_detail = select_decomposition(rows)
    count_rows, count_detail = select_count(rows)
    selected = copy_rows + decomp_rows + count_rows
    for row in selected:
        row["vyce_reference_model_id"] = REFERENCE_MODEL
        row["vyce_reference_scan_tokenizer_id"] = REFERENCE_SCAN_TOKENIZER
        row["vyce_round2_item_identity_sha256"] = r2.sha_text(r2.canon(r2.item_identity(row)))
    verification = verify_against_round2(selected)
    vc.write_jsonl(PLAN_PATH, selected)
    full_identity = [{"probe_key": x["probe_key"], "user_content_sha256": x["user_content_sha256"], "round2_item_identity_sha256": x["vyce_round2_item_identity_sha256"]} for x in selected]
    manifest = {
        "arm_label": "EXPLORATORY hosted scale/diversity extension; not confirmatory and not a replacement for the local confirmatory scale-up",
        "created_utc": vc.utcnow(),
        "reference_model": REFERENCE_MODEL,
        "reference_scan_tokenizer": REFERENCE_SCAN_TOKENIZER,
        "copy_protocol": COPY_PROTOCOL,
        "copy_protocol_template_sha256": r2.protocol_template_metadata()[COPY_PROTOCOL]["sha256"],
        "copy_protocol_format_sha256": r2.protocol_template_metadata()[COPY_PROTOCOL]["format_sha256"],
        "rows_per_hosted_model": len(selected),
        "copy_rows": len(copy_rows),
        "structural_rows": len(decomp_rows) + len(count_rows),
        "decomposition_rows": len(decomp_rows),
        "count_rows": len(count_rows),
        "copy_selection": copy_strata,
        "decomposition_selection": decomp_detail,
        "count_selection": count_detail,
        "round2_exact_reuse_verification": verification,
        "probe_plan_sha256": vc.sha_text(vc.canon(full_identity)),
        "probe_plan_path": str(PLAN_PATH),
        "sampling_method": "deterministic evenly spaced indexes within frozen round-2 strata; no randomness and no item rewriting",
        "fragmentation_caveat": "count high/low labels are frozen round-2 strata measured under qwen2.5-0.5b-instruct; hosted-endpoint tokenizer covariates are measured separately where open artifacts exist and are not silently relabeled",
    }
    vc.write_json(MANIFEST_PATH, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    manifest = build_plan()
    print(json.dumps({"plan": str(PLAN_PATH), "rows_per_hosted_model": manifest["rows_per_hosted_model"], "probe_plan_sha256": manifest["probe_plan_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
