from __future__ import annotations

import copy
import json
from pathlib import Path

import confirmatory_common as cf
from hosted_frontier_common import HostedFrontierError, canon, sha_text

DEFAULT_REFERENCE_CARRIER = "qwen2.5-3b-instruct-bf16"
REFERENCE_EXCLUDED_FIELDS = {"carrier_id", "probe_key", "probe_id"}


def hosted_carrier_id(provider: str, model: str) -> str:
    return f"hosted-{provider}-{model}"


def load_reference_plan(reference_carrier: str = DEFAULT_REFERENCE_CARRIER) -> list[dict]:
    rows = [x for x in cf.read_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl") if x["carrier_id"] == reference_carrier]
    if not rows:
        raise HostedFrontierError(f"no frozen confirmatory probe-plan rows for reference carrier {reference_carrier}")
    rows = sorted(rows, key=lambda x: x["execution_index"])
    if len(rows) not in (460, 500):
        raise HostedFrontierError(f"reference carrier {reference_carrier} has {len(rows)} rows, outside frozen 460/500 range")
    return rows


def probe_suffix(reference_carrier: str, probe_key: str) -> str:
    prefix = reference_carrier + ":"
    if not probe_key.startswith(prefix):
        raise HostedFrontierError(f"probe key {probe_key} does not start with reference carrier prefix")
    return probe_key[len(prefix):]


def transform_plan(provider: str, model: str, reference_carrier: str = DEFAULT_REFERENCE_CARRIER) -> tuple[list[dict], dict]:
    reference_rows = load_reference_plan(reference_carrier)
    carrier = hosted_carrier_id(provider, model)
    transformed = []
    mismatches = []
    for ref in reference_rows:
        row = copy.deepcopy(ref)
        suffix = probe_suffix(reference_carrier, ref["probe_key"])
        row["carrier_id"] = carrier
        row["probe_key"] = carrier + ":" + suffix
        row["probe_id"] = row["probe_key"]
        row["hosted_reference_carrier_id"] = reference_carrier
        row["reference_probe_key"] = ref["probe_key"]
        row["reference_probe_id"] = ref.get("probe_id", ref["probe_key"])
        for key, value in ref.items():
            if key in REFERENCE_EXCLUDED_FIELDS:
                continue
            if row.get(key) != value:
                mismatches.append({"probe_key": ref["probe_key"], "field": key})
        messages = cf.prompt_messages(ref)
        logical = cf.canon(messages) if ref["probe_type"] == "copy" else messages[0]["content"]
        if sha_text(logical) != ref["user_content_sha256"]:
            mismatches.append({"probe_key": ref["probe_key"], "field": "user_content_sha256_recompute"})
        transformed.append(row)
    if mismatches:
        raise HostedFrontierError(json.dumps({"reference_inventory_mismatch": mismatches[:5]}, sort_keys=True))
    identity = [
        {
            "reference_probe_key": x["reference_probe_key"],
            "hosted_probe_key": x["probe_key"],
            "target_sha256": x["target_sha256"],
            "user_content_sha256": x["user_content_sha256"],
        }
        for x in transformed
    ]
    manifest = {
        "reference_carrier_id": reference_carrier,
        "reference_probe_plan_sha256": cf.sha_file(cf.CONFIRMATORY / "probe_plan.jsonl"),
        "hosted_carrier_id": carrier,
        "rows": len(transformed),
        "semantic_probe_inventory_exact": True,
        "semantic_comparison_excludes_only": sorted(REFERENCE_EXCLUDED_FIELDS),
        "hosted_probe_identity_sha256": sha_text(canon(identity)),
        "fragmentation_covariate_caveat": "count-probe fragmentation levels and token covariates remain measurements of the frozen reference-carrier tokenizer; hosted providers do not expose inference tokenization, so these labels are fixed-stimulus covariates for hosted runs, not verified hosted-carrier fragmentation measurements",
    }
    return transformed, manifest


def phase_for_probe(row: dict) -> str:
    if row["probe_scope"] == "ascii":
        return "ascii_gate"
    if row["probe_type"] == "copy":
        return "unicode_copy_gate"
    if row["probe_type"] == "scalar_listing":
        return "unicode_scalar"
    if row["probe_type"] == "count":
        return "unicode_count"
    raise HostedFrontierError(f"unmapped probe type/scope: {row['probe_type']}/{row['probe_scope']}")


def phase_ordered_rows(rows: list[dict]) -> list[dict]:
    ascii_type_order = {"copy": 0, "scalar_listing": 1, "count": 2}
    phase_order = {"ascii_gate": 0, "unicode_copy_gate": 1, "unicode_scalar": 2, "unicode_count": 3}

    def key(row: dict):
        phase_rank = phase_order[phase_for_probe(row)]
        type_rank = ascii_type_order.get(row["probe_type"], 0) if row["probe_scope"] == "ascii" else 0
        return (phase_rank, type_rank, row["execution_index"])

    out = sorted(rows, key=key)
    for index, row in enumerate(out, 1):
        row["hosted_execution_order"] = index
    return out


def phase_rows(rows: list[dict], phase: str) -> list[dict]:
    return [x for x in phase_ordered_rows(rows) if phase_for_probe(x) == phase]
