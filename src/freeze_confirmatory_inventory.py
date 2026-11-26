from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

SCRIPT_FILES = [
    "src/confirmatory_common.py",
    "src/build_confirmatory_candidates.py",
    "src/scan_confirmatory_candidates.py",
    "src/confirmatory_backend.py",
    "src/freeze_confirmatory_environment.py",
    "src/freeze_confirmatory_tokenizers.py",
    "src/build_confirmatory_inventory.py",
    "src/audit_confirmatory_inventory.py",
    "src/freeze_confirmatory_inventory.py",
]

ARTIFACT_FILES = [
    "results/confirmatory/environment_freeze.json",
    "results/confirmatory/tokenizer_freeze.json",
    "results/confirmatory/candidate_universe.jsonl",
    "results/confirmatory/candidate_universe_build_summary.json",
    "results/confirmatory/candidate_scan.metadata.json",
    "results/confirmatory/bundle_scan.metadata.json",
    "results/confirmatory/inventory.jsonl",
    "results/confirmatory/probe_plan.jsonl",
    "results/confirmatory/inventory_audit.json",
    "results/confirmatory/inventory_cell_table.csv",
    "results/confirmatory/inventory_build_summary.json",
]

RESOLUTIONS = [
    "Mistral-7B is demoted from the primary confirmatory ladder to descriptive robustness only; it is excluded from H1/H2/H3/H4 primary carrier requirements and verdicts.",
    "Expanded bf16 quantization-control weight downloads are skipped. Quantization validation is narrower: no new bf16 comparator weights may be downloaded; MLX8 carriers are not automatically excluded solely for missing bf16 comparators, and the absent direct quantization contrast is a frozen limitation rather than a post-hoc rescue. Strict tokenizer, ASCII copy, ASCII count, and ASCII scalar-listing gates remain binding.",
    "Scalar-floor and copy-ceiling exact replication rules are accepted exactly as pre-registered.",
    "Matching hard caps are accepted exactly as pre-registered: <=0.5 tokens/cluster, <=4 UTF-8 bytes, <=2 codepoints, and <=2 mean sequence-complexity units per bundle.",
    "Probe-level bootstrap is accepted exactly as pre-registered despite constituent reuse dependence.",
    "ASCII structural gates at 11/12 are accepted exactly as pre-registered.",
]


def git(args: list[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=cf.ROOT, text=True).strip()


def artifact_entry(relpath: str) -> dict:
    path = cf.ROOT / relpath
    if not path.exists():
        raise FileNotFoundError(relpath)
    return {"path": relpath, "sha256": cf.sha_file(path), "size": path.stat().st_size, "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def main():
    audit = json.loads((cf.CONFIRMATORY / "inventory_audit.json").read_text())
    if not audit.get("freeze_ready"):
        raise RuntimeError("inventory audit is not freeze-ready")
    model_output_candidates = [
        cf.CONFIRMATORY / "confirmatory_raw.jsonl",
        cf.CONFIRMATORY / "raw.jsonl",
        cf.CONFIRMATORY / "scored.jsonl",
    ]
    existing_outputs = [str(x) for x in model_output_candidates if x.exists()]
    scripts = [artifact_entry(x) for x in SCRIPT_FILES]
    artifacts = [artifact_entry(x) for x in ARTIFACT_FILES]
    docs = [artifact_entry("docs/CONFIRMATORY_PREREGISTRATION.md"), artifact_entry("docs/CONFIRMATORY_IMPLEMENTATION_PLAN.md"), artifact_entry("results/confirmatory_power_analysis.json")]
    obj = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_schema": "confirmatory-inventory-freeze-v1",
        "git": {"branch": git(["branch", "--show-current"]), "commit_before_freeze_commit": git(["rev-parse", "HEAD"])},
        "pre_generation_assertion": {
            "confirmatory_model_output_files_present": existing_outputs,
            "no_confirmatory_model_output_exists": not existing_outputs,
            "assertion_time_utc": datetime.now(timezone.utc).isoformat(),
        },
        "judgment_call_resolutions": RESOLUTIONS,
        "implementation_notes": [
            "Constituent reuse penalty is operationalized as the number of already-selected count-inventory constituent uses across both sides of a candidate pair; deterministic construction order makes this objective reproducible.",
            "Non-NFC-stable scalar candidates are excluded before inventory construction because byte-exact target identity is otherwise altered by tokenizer NFC normalization; this stricter measurement-validity exclusion is recorded in the candidate audit.",
            "Newest-L1 is present only for carriers whose frozen feasibility audit constructs eight unique valid probes at every expected count 1-5; structurally impossible cells are explicit in the audit and report.",
        ],
        "prompt_template_sha256": {
            "copy_user_template": cf.sha_text(cf.COPY_USER_TEMPLATE),
            "count_template": cf.sha_text(cf.COUNT_TEMPLATE),
            "scalar_listing_template": cf.sha_text(cf.SCALAR_TEMPLATE),
            "copy_format_sha256_input": cf.sha_text(cf.canon({"exemplars": cf.FEWSHOT_EXEMPLARS, "template": cf.COPY_USER_TEMPLATE, "assistant_output_template": "<OUT>{exemplar}</OUT>", "message_style": "assistant_exemplar_pairs"})),
        },
        "artifacts": artifacts,
        "scripts": scripts,
        "design_documents": docs,
        "inventory_audit_sha256_at_freeze": audit["inventory_sha256"],
        "freeze_ready": True,
    }
    if existing_outputs:
        raise RuntimeError(f"confirmatory output exists before freeze: {existing_outputs}")
    cf.write_json(cf.CONFIRMATORY / "inventory_freeze.json", obj)
    print(json.dumps({"wrote": str(cf.CONFIRMATORY / "inventory_freeze.json"), "freeze_ready": True, "no_confirmatory_model_output_exists": True}, indent=2))


if __name__ == "__main__":
    main()
