from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ROUND2 = RESULTS / "behavioral_round2"
FIGURES = ROOT / "docs" / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    if not path.exists():
        raise AssertionError(f"missing required artifact: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def jsonl_count(path: Path) -> int:
    if not path.exists():
        raise AssertionError(f"missing required artifact: {path.relative_to(ROOT)}")
    return sum(1 for line in path.open() if line.strip())


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_inputs() -> dict:
    manifest = load_json(RESULTS / "manifest.json")
    rows = manifest["inputs"]
    require(len(rows) == 32, f"expected 32 pinned inputs, found {len(rows)}")
    missing, mismatched = [], []
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            missing.append(row["path"])
        elif sha256(path) != row["sha256"]:
            mismatched.append(row["path"])
    require(not missing, f"pinned inputs missing: {missing[:3]}")
    require(not mismatched, f"pinned input hash mismatch: {mismatched[:3]}")
    require(all(row.get("independent_download_sha256_match") is True for row in rows), "independent input hash replication is not clean")
    return {"pinned_inputs": len(rows), "input_hashes_verified": True}


def check_scan() -> dict:
    validation = load_json(RESULTS / "validation.json")
    verdict = load_json(RESULTS / "scan_verdict.json")
    require(all(validation["gates"].values()), f"scan validation gate failure: {validation['gates']}")
    require(validation["grapheme"]["passed"] == validation["grapheme"]["total"] == 853, "grapheme conformance is not 853/853")
    require(validation["grapheme"]["broken_split_every_codepoint_failures"] == 516, "grapheme negative-control count changed")
    require(verdict["tokenizer_gate_pass"] is True, "tokenizer metric completeness gate failed")
    require(verdict["native_backend_gate_pass"] is True, "native tokenizer backend gate failed")
    require(verdict["grapheme_gate_pass"] is True, "scan verdict grapheme gate failed")
    require(verdict["proceed_rule4_fires"] is True, "tokenizer scan proceed rule 4 does not fire")
    require(set(verdict["firing_families"]) == {"Gemma", "Llama", "Qwen"}, "rule-4 firing families changed")
    require(verdict["any_newest_vocab_entry"] is False and verdict["any_newest_pipeline_single"] is False, "newest exact-token absence claim changed")
    require(all(x["pipeline_single_token_count"] == 0 and x["exact_vocab_entry_count"] == 0 and x["families"] == 62 for x in verdict["newest_cohort_exactness"]), "0/62 newest-cohort exactness check changed")
    return {"scan_rule4_fires": True, "newest_exact_single_token": "0/62 per tokenizer"}


def check_round2() -> dict:
    verdict = load_json(ROUND2 / "round2_verdict.json")
    mimicry = load_json(ROUND2 / "round2_wrapper_mimicry_analysis.json")
    expected_valid = [
        "qwen2.5-3b-instruct-bf16",
        "qwen3-4b-instruct-2507-mlx8",
        "gemma3-4b-it-mlx8-mirror",
        "tinyllama-1.1b-chat-v1.0-bf16",
    ]
    require(jsonl_count(ROUND2 / "round2_raw.jsonl") == 1450, "round2 raw row count is not 1450")
    require(jsonl_count(ROUND2 / "round2_scored.jsonl") == 1450, "round2 scored row count is not 1450")
    require(verdict["all_models_complete"] is True, "round2 completeness failed")
    require(verdict["all_gate_mutation_checks_pass"] is True, "round2 mutation checks failed")
    require(verdict["valid_models"] == expected_valid, "round2 valid-model slate changed")
    require(verdict["invalid_models"] == ["qwen2.5-7b-instruct-mlx8"], "round2 invalid-model slate changed")
    require(verdict["hard_kill_operational_arm"] is True, "operational copy arm hard kill missing")
    require(verdict["hard_kill_structural_arm"] is False, "structural arm unexpectedly hard-killed")
    require(verdict["behavioral_proceed_rule1_fires"] is False, "copy proceed rule 1 unexpectedly fires")
    require(verdict["behavioral_proceed_rule2_fires"] is True, "structural fragmentation proceed rule 2 missing")
    require(verdict["behavioral_proceed_rule2_models"] == ["qwen2.5-3b-instruct-bf16", "qwen3-4b-instruct-2507-mlx8"], "rule-2 model list changed")
    require(verdict["behavioral_proceed_rule3_fires"] is False, "systematic copy-error rule 3 unexpectedly fires")
    require(verdict["final_behavioral_interpretation"] == "proceed", "final behavioral interpretation changed")
    for row in verdict["newest_old_copy"]:
        if row["model_id"] in expected_valid:
            require((row["newest_correct"], row["newest_n"], row["old_control_correct"], row["old_control_n"]) == (94, 94, 64, 64), f"copy hard-kill counts changed for {row['model_id']}")
    q7 = mimicry["qwen2.5-7b-instruct-mlx8"]
    require(q7["in_wrapper_mimicry_failures"] == 28 and q7["mimicry_inner_byte_exact"] == 28 and q7["mimicry_inner_byte_diff"] == 0, "qwen2.5-7b wrapper-mimicry accounting changed")
    return {"round2_rows": 1450, "valid_models": len(expected_valid), "mimicry_inner_byte_exact": "28/28"}


def check_figures() -> dict:
    manifest_path = FIGURES / "figure_manifest.json"
    manifest = load_json(manifest_path)
    outputs = manifest["outputs"]
    require(len(outputs) == 8, f"expected 8 figure outputs, found {len(outputs)}")
    missing, mismatched = [], []
    for rel, meta in outputs.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != meta["sha256"]:
            mismatched.append(rel)
    require(not missing, f"figure outputs missing: {missing}")
    require(not mismatched, f"figure output hash mismatch: {mismatched}")
    require((ROOT / "docs" / "assets" / "fonts" / "OFL.txt").exists(), "Geist OFL license missing")
    return {"figure_outputs_verified": len(outputs)}


def main():
    checks = {
        "inputs": check_inputs(),
        "scan": check_scan(),
        "behavioral_round2": check_round2(),
        "figures": check_figures(),
    }
    print(json.dumps({"artifact_check": "pass", **checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(json.dumps({"artifact_check": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
