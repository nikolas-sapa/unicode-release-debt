from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

COMPARATORS = {
    "gemma3-4b-it-mlx8-mirror": {
        "repo": "unsloth/gemma-3-4b-it",
        "revision": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        "files": ["added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
    },
    "llama32-3b-instruct-mlx8": {
        "repo": "unsloth/Llama-3.2-3B-Instruct",
        "revision": "006f5dcd1393c3add266de40994ba96225e9689d",
        "files": ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
    },
    "mistral7b-instruct-v03-mlx8": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "files": ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
    },
}

AUDIT_TEXTS = [
    "test",
    "ASCII_123",
    "é",
    "e\u0301",
    "😀",
    "😀\u200d😀",
    "🫨",
    "🧑\u200d🧑\u200d🧒\u200d🧒",
    "a\u20d0",
    "\u20d0",
    "<IN>wrapper</IN><OUT>wrapper</OUT>",
]


def download_file(repo: str, revision: str, filename: str, dest_dir: Path) -> dict:
    source = Path(hf_hub_download(repo, filename, revision=revision))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    data = source.read_bytes()
    if dest.exists() and dest.read_bytes() != data:
        raise RuntimeError(f"tokenizer destination hash conflict: {dest}")
    if not dest.exists():
        dest.write_bytes(data)
    return {
        "file": filename,
        "repo": repo,
        "revision": revision,
        "cache_source": str(source),
        "path": str(dest),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def tokenizer_json_metadata(path: Path) -> dict:
    raw = json.loads(path.read_text())
    model = raw.get("model") or {}
    return {
        "model_type": model.get("type"),
        "vocab_size": len(model.get("vocab") or {}),
        "normalizer": cf.canon(raw.get("normalizer")),
        "pre_tokenizer": cf.canon(raw.get("pre_tokenizer")),
        "decoder": cf.canon(raw.get("decoder")),
        "byte_fallback_or_bytelevel": "ByteLevel" in cf.canon(raw.get("pre_tokenizer")) or "byte_fallback" in cf.canon(model),
    }


def mutation_check(path: Path) -> dict:
    original = path.read_bytes()
    with tempfile.TemporaryDirectory() as td:
        mutated = Path(td) / path.name
        data = bytearray(original)
        data[0] ^= 1
        mutated.write_bytes(bytes(data))
        original_hash = hashlib.sha256(original).hexdigest()
        mutated_hash = hashlib.sha256(mutated.read_bytes()).hexdigest()
    return {"file": path.name, "original_sha256": original_hash, "mutated_sha256": mutated_hash, "hash_mismatch_detected": original_hash != mutated_hash}


def main():
    rows = []
    errors = []
    for cfg in cf.carrier_configs(include_descriptive=True):
        carrier_dir = cf.TOKENIZER_DIR / cfg["carrier_id"]
        files = []
        for filename in cfg["tokenizer_files"]:
            try:
                row = download_file(cfg["tokenizer_source_repo"], cfg["tokenizer_source_revision"], filename, carrier_dir)
                expected = cfg.get("expected_tokenizer_sha256", {}).get(filename)
                row["expected_sha256"] = expected
                row["expected_hash_match"] = None if expected is None else row["sha256"] == expected
                if expected is not None and not row["expected_hash_match"]:
                    errors.append({"carrier_id": cfg["carrier_id"], "file": filename, "expected": expected, "got": row["sha256"]})
                files.append(row)
            except Exception as exc:
                errors.append({"carrier_id": cfg["carrier_id"], "file": filename, "error": f"{type(exc).__name__}: {exc}"})
        tjson = carrier_dir / "tokenizer.json"
        meta = tokenizer_json_metadata(tjson) if tjson.exists() else {}
        rows.append({
            "carrier_id": cfg["carrier_id"],
            "primary_ladder": cfg["primary_ladder"],
            "tokenizer_source_repo": cfg["tokenizer_source_repo"],
            "tokenizer_source_revision": cfg["tokenizer_source_revision"],
            "files": files,
            "all_expected_hashes_match": all(x.get("expected_hash_match") is not False for x in files),
            **meta,
            "chat_template_source": "tokenizer_config.json or artifact chat template inspected at runner load time",
            "canonical_overlay_required": bool(cfg.get("tokenizer_overlay")),
        })
    comparator_rows = []
    for carrier_id, spec in COMPARATORS.items():
        dest = cf.TOKENIZER_DIR / "comparators" / carrier_id
        files = []
        for filename in spec["files"]:
            try:
                files.append(download_file(spec["repo"], spec["revision"], filename, dest))
            except Exception as exc:
                files.append({"file": filename, "error": f"{type(exc).__name__}: {exc}"})
        comparator_rows.append({"carrier_id": carrier_id, **spec, "files": files, "purpose": "tokenizer-only behavioral comparison; no comparator weight download"})
    mutations = []
    for cfg in cf.carrier_configs(include_descriptive=True):
        path = cf.TOKENIZER_DIR / cfg["carrier_id"] / "tokenizer.json"
        if path.exists():
            mutations.append(mutation_check(path))
    obj = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "carriers": rows,
        "comparator_tokenizer_files": comparator_rows,
        "fixed_unicode_audit_texts": AUDIT_TEXTS,
        "hash_mutation_checks": mutations,
        "hash_mutation_checks_pass": all(x["hash_mismatch_detected"] for x in mutations),
        "errors": errors,
        "freeze_pass": not errors and all(x["all_expected_hashes_match"] for x in rows) and all(x["hash_mismatch_detected"] for x in mutations),
    }
    cf.write_json(cf.CONFIRMATORY / "tokenizer_freeze.json", obj)
    print(json.dumps({"freeze_pass": obj["freeze_pass"], "errors": errors, "carriers": len(rows)}, indent=2))
    if not obj["freeze_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
