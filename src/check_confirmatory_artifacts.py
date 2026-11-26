from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else cf.ROOT / path


def verify_entries(entries: list[dict]) -> list[dict]:
    errors = []
    for entry in entries:
        path = resolve_path(entry["path"])
        if not path.exists():
            errors.append({"path": entry["path"], "error": "missing"})
            continue
        got = cf.sha_file(path)
        if got != entry["sha256"]:
            errors.append({"path": entry["path"], "error": "sha256_mismatch", "expected": entry["sha256"], "actual": got})
        elif entry.get("size") is not None and path.stat().st_size != entry["size"]:
            errors.append({"path": entry["path"], "error": "size_mismatch", "expected": entry["size"], "actual": path.stat().st_size})
    return errors


def load_freeze() -> dict:
    return json.loads((cf.CONFIRMATORY / "inventory_freeze.json").read_text())


def verify_frozen_artifacts() -> dict:
    freeze = load_freeze()
    sections = {}
    all_errors = []
    for section in ("artifacts", "scripts", "design_documents"):
        errors = verify_entries(freeze[section])
        sections[section] = {"checked": len(freeze[section]), "errors": errors, "pass": not errors}
        all_errors.extend({**x, "section": section} for x in errors)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "freeze_schema": freeze.get("freeze_schema"), "sections": sections, "errors": all_errors, "artifact_hash_verification_pass": not all_errors}


def mutation_test() -> dict:
    freeze = load_freeze()
    entries = copy.deepcopy(freeze["artifacts"])
    target = next(x for x in entries if x["path"].endswith("inventory.jsonl"))
    original_path = resolve_path(target["path"])
    data = bytearray(original_path.read_bytes())
    data[len(data) // 2] ^= 1
    mutation_dir = cf.SCAN_DIR / "artifact_mutation_test"
    mutation_dir.mkdir(parents=True, exist_ok=True)
    mutated_path = mutation_dir / "inventory.corrupted-for-mutation-test.jsonl"
    mutated_path.write_bytes(bytes(data))
    mutated_entries = copy.deepcopy(entries)
    for row in mutated_entries:
        if row["path"] == target["path"]:
            row["path"] = str(mutated_path)
    errors = verify_entries(mutated_entries)
    relevant = [x for x in errors if x["path"] == str(mutated_path)]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test": "corrupt one byte in a temporary copy of frozen inventory.jsonl; retained artifact is never mutated",
        "target_original_path": target["path"],
        "mutated_copy_path": str(mutated_path),
        "frozen_sha256": target["sha256"],
        "mutated_sha256": cf.sha_file(mutated_path),
        "verification_error_count": len(errors),
        "target_error_detected": bool(relevant),
        "target_error": relevant[:1],
        "mutation_test_pass": bool(relevant) and cf.sha_file(mutated_path) != target["sha256"],
    }
    cf.write_json(cf.CONFIRMATORY / "artifact_hash_mutation_check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation-test", action="store_true")
    args = parser.parse_args()
    verification = verify_frozen_artifacts()
    cf.write_json(cf.CONFIRMATORY / "artifact_hash_check.json", verification)
    out = {"artifact_hash_verification_pass": verification["artifact_hash_verification_pass"], "error_count": len(verification["errors"])}
    if args.mutation_test:
        mutation = mutation_test()
        out["artifact_hash_mutation_test_pass"] = mutation["mutation_test_pass"]
    print(json.dumps(out, indent=2))
    if not verification["artifact_hash_verification_pass"] or (args.mutation_test and not out["artifact_hash_mutation_test_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
