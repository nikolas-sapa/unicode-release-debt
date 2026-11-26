from __future__ import annotations

import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

WRAPPER_STRINGS = ("<IN>", "</IN>", "<OUT>", "</OUT>")


def load_inputs():
    rows = cf.read_jsonl(cf.CONFIRMATORY / "inventory.jsonl")
    candidates = {x["candidate_id"]: x for x in cf.read_jsonl(cf.CONFIRMATORY / "candidate_universe.jsonl")}
    return rows, candidates


def validate_inventory(rows: list[dict], candidates_by_id: dict) -> list[dict]:
    errors = []
    carriers = [x["carrier_id"] for x in cf.carrier_configs(include_descriptive=True)]
    ids = [x.get("probe_id") for x in rows]
    if len(ids) != len(set(ids)):
        errors.append({"check": "unique_probe_ids", "detail": "duplicate probe_id values", "count": len(ids) - len(set(ids))})
    keys = [x.get("probe_key") for x in rows]
    if len(keys) != len(set(keys)):
        errors.append({"check": "unique_probe_keys", "detail": "duplicate probe_key values"})
    expected_prompt_hashes = {
        "copy": cf.sha_text(cf.COPY_USER_TEMPLATE),
        "count": cf.sha_text(cf.COUNT_TEMPLATE),
        "scalar_listing": cf.sha_text(cf.SCALAR_TEMPLATE),
    }
    ascii_copy_targets = [x["target"] for x in cf.round2_ascii_copy_controls()]
    for carrier in carriers:
        cr = [x for x in rows if x.get("carrier_id") == carrier]
        if not cr:
            errors.append({"check": "carrier_present", "carrier_id": carrier, "detail": "no inventory rows"})
            continue
        copy_rows = [x for x in cr if x["probe_type"] == "copy"]
        scalar_rows = [x for x in cr if x["probe_type"] == "scalar_listing"]
        count_ascii = [x for x in cr if x["probe_type"] == "count" and x["probe_scope"] == "ascii"]
        if len(copy_rows) != 124:
            errors.append({"check": "copy_count", "carrier_id": carrier, "expected": 124, "actual": len(copy_rows)})
        if len(scalar_rows) != 44:
            errors.append({"check": "scalar_count", "carrier_id": carrier, "expected": 44, "actual": len(scalar_rows)})
        if len(count_ascii) != 12:
            errors.append({"check": "ascii_count_control_count", "carrier_id": carrier, "expected": 12, "actual": len(count_ascii)})
        if sum(x["probe_type"] == "copy" and x["probe_scope"] == "ascii" for x in cr) != 24:
            errors.append({"check": "ascii_copy_control_count", "carrier_id": carrier, "expected": 24})
        actual_ascii_copy = [x["target"] for x in copy_rows if x["probe_scope"] == "ascii"]
        if actual_ascii_copy != ascii_copy_targets:
            errors.append({"check": "ascii_copy_round2_exactness", "carrier_id": carrier, "detail": "ASCII copy controls differ from round-2 order/content"})
        targets = [x["target"] for x in cr]
        dup_targets = [x for x, n in Counter(targets).items() if n > 1]
        if dup_targets:
            errors.append({"check": "cross_cell_duplicate_targets", "carrier_id": carrier, "count": len(dup_targets), "examples": dup_targets[:3]})
        for row in cr:
            base = {"carrier_id": carrier, "probe_id": row.get("probe_id")}
            if row.get("target_sha256") != cf.sha_text(row.get("target", "")):
                errors.append({**base, "check": "target_hash", "detail": "target_sha256 does not match target"})
            if any(x in row.get("target", "") for x in WRAPPER_STRINGS) or "\n" in row.get("target", "") or "\r" in row.get("target", ""):
                errors.append({**base, "check": "target_forbidden_delimiter", "detail": "target contains newline or wrapper substring"})
            if any(0xD800 <= ord(c) <= 0xDFFF for c in row.get("target", "")):
                errors.append({**base, "check": "target_scalar_validity", "detail": "target contains surrogate"})
            if row.get("prompt_template_sha256") != expected_prompt_hashes[row["probe_type"]]:
                errors.append({**base, "check": "prompt_template_hash", "expected": expected_prompt_hashes[row["probe_type"]], "actual": row.get("prompt_template_sha256")})
            expected_user = cf.canon(cf.copy_messages(row["target"])) if row["probe_type"] == "copy" else (cf.count_user_content(row["target"]) if row["probe_type"] == "count" else cf.scalar_user_content(row["target"]))
            if row.get("user_content_sha256") != cf.sha_text(expected_user):
                errors.append({**base, "check": "user_content_hash", "detail": "user content hash does not recompute"})
            if row["probe_type"] == "count":
                if cf.grapheme_count(row["target"]) != row.get("expected_count") and row["probe_scope"] == "unicode":
                    errors.append({**base, "check": "regex_grapheme_count", "expected": row.get("expected_count"), "actual": cf.grapheme_count(row["target"])})
                if row["probe_scope"] == "unicode":
                    if cf.fragmentation_level(row.get("tokens_per_cluster", -1)) != row.get("fragmentation_level"):
                        errors.append({**base, "check": "fragmentation_level_consistency", "level": row.get("fragmentation_level"), "tokens_per_cluster": row.get("tokens_per_cluster")})
                    constituents = row.get("constituent_ids") or []
                    types = {candidates_by_id[x]["item_type"] for x in constituents if x in candidates_by_id}
                    if len(constituents) != row.get("expected_count") or len(types) != 1 or next(iter(types), None) != row.get("item_type"):
                        errors.append({**base, "check": "homogeneous_item_type", "constituent_types": sorted(types), "expected_type": row.get("item_type"), "constituent_count": len(constituents)})
            if row["probe_type"] == "scalar_listing" and row.get("expected_text") != cf.scalar_text(row["target"]):
                errors.append({**base, "check": "scalar_expected_exact", "expected": cf.scalar_text(row["target"]), "actual": row.get("expected_text")})
        for level in ("L2", "L3", "L4"):
            for era in ("newest", "legacy"):
                cell = [x for x in cr if x["probe_type"] == "count" and x["probe_scope"] == "unicode" and x.get("fragmentation_level") == level and x.get("era") == era]
                if len(cell) != 40:
                    errors.append({"check": "unicode_count_cell_size", "carrier_id": carrier, "level": level, "era": era, "expected": 40, "actual": len(cell)})
                counts = Counter(x.get("expected_count") for x in cell)
                if any(counts.get(n, 0) != 8 for n in range(1, 6)):
                    errors.append({"check": "expected_count_slots", "carrier_id": carrier, "level": level, "era": era, "counts": dict(counts)})
        legacy_l1 = [x for x in cr if x["probe_type"] == "count" and x["probe_scope"] == "unicode" and x.get("fragmentation_level") == "L1" and x.get("era") == "legacy"]
        newest_l1 = [x for x in cr if x["probe_type"] == "count" and x["probe_scope"] == "unicode" and x.get("fragmentation_level") == "L1" and x.get("era") == "newest"]
        if len(legacy_l1) != 40:
            errors.append({"check": "legacy_l1_size", "carrier_id": carrier, "expected": 40, "actual": len(legacy_l1)})
        if len(newest_l1) not in (0, 40):
            errors.append({"check": "conditional_newest_l1_size", "carrier_id": carrier, "expected": "0 or 40", "actual": len(newest_l1)})
        for label, cell in (("legacy_l1", legacy_l1), ("newest_l1", newest_l1)):
            counts = Counter(x.get("expected_count") for x in cell)
            if cell and any(counts.get(n, 0) != 8 for n in range(1, 6)):
                errors.append({"check": "l1_expected_count_slots", "carrier_id": carrier, "cell": label, "counts": dict(counts)})
        pair_groups = defaultdict(list)
        for row in cr:
            if row["probe_type"] == "count" and row.get("match_pair_id"):
                pair_groups[row["match_pair_id"]].append(row)
        if len(pair_groups) != 120:
            errors.append({"check": "h2_pair_count", "carrier_id": carrier, "expected": 120, "actual": len(pair_groups)})
        for pair_id, members in pair_groups.items():
            base = {"check": "h2_pair_constraints", "carrier_id": carrier, "match_pair_id": pair_id}
            if len(members) != 2 or sorted(x["era"] for x in members) != ["legacy", "newest"]:
                errors.append({**base, "detail": "pair does not contain exactly one newest and one legacy probe", "member_count": len(members)})
                continue
            new = next(x for x in members if x["era"] == "newest")
            old = next(x for x in members if x["era"] == "legacy")
            equal_keys = ["carrier_id", "fragmentation_level", "expected_count", "item_type", "supplementary_cluster_count"]
            for key in equal_keys:
                if new.get(key) != old.get(key):
                    errors.append({**base, "detail": f"{key} mismatch", "newest": new.get(key), "legacy": old.get(key)})
            caps = {
                "tokens_per_cluster": 0.5,
                "utf8_bytes": 4,
                "codepoint_count": 2,
                "mean_sequence_complexity": 2,
            }
            for key, cap in caps.items():
                diff = abs(new.get(key, 10**9) - old.get(key, -10**9))
                if diff > cap + 1e-12:
                    errors.append({**base, "detail": f"{key} hard cap violated", "difference": diff, "cap": cap})
    return errors


def mutation_rows(rows: list[dict], kind: str) -> list[dict]:
    mutated = copy.deepcopy(rows)
    target = next(x for x in mutated if x["probe_type"] == "count" and x["probe_scope"] == "unicode" and x.get("match_pair_id"))
    if kind == "expected_count":
        target["expected_count"] += 1
    elif kind == "target_text_hash":
        target["target"] += "X"
    elif kind == "pair_era_label":
        target["era"] = "legacy" if target["era"] == "newest" else "newest"
    elif kind == "duplicate_probe_id":
        mutated.append(copy.deepcopy(target))
    elif kind == "mixed_type_constituents":
        others = []
        for row in mutated:
            for cid in row.get("constituent_ids") or []:
                others.append((row, cid))
        changed = False
        all_candidates = {x["candidate_id"]: x for x in cf.read_jsonl(cf.CONFIRMATORY / "candidate_universe.jsonl")}
        for row in mutated:
            if row["probe_type"] == "count" and row["probe_scope"] == "unicode":
                current = row["constituent_ids"][0]
                replacement = next(x for x in all_candidates.values() if x["era"] == row["era"] and x["item_type"] != row["item_type"])
                row["constituent_ids"][0] = replacement["candidate_id"]
                changed = True
                break
        if not changed:
            raise RuntimeError("mixed-type mutation setup failed")
    elif kind == "fragmentation_boundary":
        level = target["fragmentation_level"]
        target["tokens_per_cluster"] = {"L1": 2.5, "L2": 4.5, "L3": 6.5, "L4": 6.49}[level]
    elif kind == "pair_tpc_cap":
        members = [x for x in mutated if x.get("match_pair_id") == target["match_pair_id"]]
        other = next(x for x in members if x is not target)
        other["tokens_per_cluster"] = target["tokens_per_cluster"] + 0.6
    else:
        raise ValueError(kind)
    return mutated


def write_cell_table(rows: list[dict], path: Path):
    fields = ["carrier_id", "era", "fragmentation_level", "expected_count", "item_type", "requested_preferred_probes", "achieved_probes", "type_substitution_net", "cell_achieved_total", "cell_structurally_empty"]
    out = []
    for cfg in cf.carrier_configs(include_descriptive=True):
        carrier = cfg["carrier_id"]
        cr = [x for x in rows if x["carrier_id"] == carrier and x["probe_type"] == "count" and x["probe_scope"] == "unicode"]
        newest_l1 = any(x["era"] == "newest" and x.get("fragmentation_level") == "L1" for x in cr)
        for level in ("L1", "L2", "L3", "L4"):
            for era in ("newest", "legacy"):
                cell_present = level != "L1" or era == "legacy" or newest_l1
                if not cell_present:
                    out.append({"carrier_id": carrier, "era": era, "fragmentation_level": level, "expected_count": "all", "item_type": "all", "requested_preferred_probes": 0, "achieved_probes": 0, "type_substitution_net": 0, "cell_achieved_total": 0, "cell_structurally_empty": 1})
                    continue
                for count in range(1, 6):
                    cell_total = sum(1 for x in cr if x["era"] == era and x.get("fragmentation_level") == level and x.get("expected_count") == count)
                    for item_type in cf.ITEM_TYPE_CYCLE:
                        achieved = sum(1 for x in cr if x["era"] == era and x.get("fragmentation_level") == level and x.get("expected_count") == count and x.get("item_type") == item_type)
                        requested = 2
                        out.append({
                            "carrier_id": carrier, "era": era, "fragmentation_level": level, "expected_count": count, "item_type": item_type,
                            "requested_preferred_probes": requested, "achieved_probes": achieved, "type_substitution_net": achieved - requested,
                            "cell_achieved_total": cell_total, "cell_structurally_empty": int(cell_total == 0),
                        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(out)


def main():
    rows, candidates = load_inputs()
    errors = validate_inventory(rows, candidates)
    mutations = {}
    for kind in ("expected_count", "target_text_hash", "pair_era_label", "duplicate_probe_id", "mixed_type_constituents", "fragmentation_boundary", "pair_tpc_cap"):
        mutated_errors = validate_inventory(mutation_rows(rows, kind), candidates)
        mutations[kind] = {"audit_rejected_mutation": len(mutated_errors) > 0, "error_count": len(mutated_errors), "first_errors": mutated_errors[:3]}
    write_cell_table(rows, cf.CONFIRMATORY / "inventory_cell_table.csv")
    tokenizer_freeze = json.loads((cf.CONFIRMATORY / "tokenizer_freeze.json").read_text())
    scan_meta = json.loads((cf.CONFIRMATORY / "candidate_scan.metadata.json").read_text())
    obj = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_rows": len(rows),
        "inventory_sha256": cf.sha_file(cf.CONFIRMATORY / "inventory.jsonl"),
        "probe_plan_sha256": cf.sha_file(cf.CONFIRMATORY / "probe_plan.jsonl"),
        "errors": errors,
        "audit_pass": not errors,
        "mutation_checks": mutations,
        "mutation_checks_pass": all(x["audit_rejected_mutation"] for x in mutations.values()),
        "tokenizer_freeze_pass": tokenizer_freeze.get("freeze_pass"),
        "candidate_scan_audit_pass": scan_meta.get("audit_pass"),
        "candidate_scan_sha256": scan_meta.get("scan_sha256"),
    }
    obj["freeze_ready"] = obj["audit_pass"] and obj["mutation_checks_pass"] and obj["tokenizer_freeze_pass"] and obj["candidate_scan_audit_pass"]
    cf.write_json(cf.CONFIRMATORY / "inventory_audit.json", obj)
    print(json.dumps({"audit_pass": obj["audit_pass"], "mutation_checks_pass": obj["mutation_checks_pass"], "freeze_ready": obj["freeze_ready"], "error_count": len(errors)}, indent=2))
    if errors:
        print(json.dumps(errors[:20], indent=2, ensure_ascii=False))
    if not obj["freeze_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
