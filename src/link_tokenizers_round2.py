from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import transformers
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2
import ucd_build as ucd

OUT = r2.ROUND2 / "tokenizer_linkage_round2.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def behavior_file_source(cfg: dict, filename: str) -> tuple[str, str]:
    overlay = cfg.get("tokenizer_overlay")
    if overlay and filename in overlay["files"]:
        return overlay["repo"], overlay["revision"]
    return cfg["hf_repo"], cfg["revision"]


def unique_probe_texts(scan_tokenizer_id: str) -> list[str]:
    rows = r2.probe_rows(scan_tokenizer_id, "fewshot_copy_v1")
    texts = []
    for row in rows:
        for key in ("target", "a_text", "b_text"):
            if key in row and row[key] not in texts:
                texts.append(row[key])
    return texts


def tokenizer_encoding_equivalence(cfg: dict, scan_cfg: dict) -> dict:
    try:
        scan_path = r2.ROOT / "data" / "hf" / cfg["scan_tokenizer_id"]
        scan_tok = transformers.AutoTokenizer.from_pretrained(str(scan_path), local_files_only=True)
        overlay = cfg.get("tokenizer_overlay")
        behavior_repo = overlay["repo"] if overlay else cfg["hf_repo"]
        behavior_revision = overlay["revision"] if overlay else cfg["revision"]
        behavior_tok = transformers.AutoTokenizer.from_pretrained(behavior_repo, revision=behavior_revision)
        texts = unique_probe_texts(cfg["scan_tokenizer_id"])
        mismatches = []
        for text in texts:
            a = scan_tok.encode(text, add_special_tokens=False)
            b = behavior_tok.encode(text, add_special_tokens=False)
            if a != b:
                mismatches.append({"text": text, "scan_ids": a, "behavior_ids": b})
                if len(mismatches) >= 5:
                    break
        return {
            "checked_texts": len(texts), "token_ids_equal_on_all_probe_texts": not mismatches, "first_mismatches": mismatches,
            "scan_tokenizer_class": type(scan_tok).__name__, "behavior_tokenizer_class": type(behavior_tok).__name__,
            "behavior_tokenizer_repo": behavior_repo, "behavior_tokenizer_revision": behavior_revision,
        }
    except Exception as exc:
        return {"checked_texts": 0, "token_ids_equal_on_all_probe_texts": None, "error": type(exc).__name__ + ": " + str(exc)[:500]}


def main():
    rows = []
    for cfg in r2.model_configs():
        scan_cfg = ucd.HF[cfg["scan_tokenizer_id"]]
        comparisons = []
        for filename, scan_hash in sorted(scan_cfg["files"].items()):
            scan_path = r2.ROOT / "data" / hf_filename(cfg["scan_tokenizer_id"], filename)
            row = {"file": filename, "scan_artifact_sha256": scan_hash, "scan_file_present": scan_path.exists()}
            repo, revision = behavior_file_source(cfg, filename)
            try:
                behavior_path = Path(hf_hub_download(repo, filename, revision=revision))
                behavior_hash = sha_file(behavior_path)
                row.update({
                    "behavior_source_repo": repo, "behavior_source_revision": revision,
                    "behavior_file_sha256": behavior_hash, "byte_identical_to_scan_file": behavior_hash == scan_hash,
                    "access": "downloaded-or-cache-verified",
                })
            except Exception as exc:
                row.update({
                    "behavior_source_repo": repo, "behavior_source_revision": revision,
                    "behavior_file_sha256": None, "byte_identical_to_scan_file": None,
                    "access": type(exc).__name__ + ": " + str(exc)[:300],
                })
            comparisons.append(row)
        encoding = tokenizer_encoding_equivalence(cfg, scan_cfg)
        strict_all = all(x["byte_identical_to_scan_file"] is True for x in comparisons)
        core_json = next((x for x in comparisons if x["file"] == "tokenizer.json"), {})
        rows.append({
            "model_id": cfg["model_id"], "behavior_repo": cfg["hf_repo"], "behavior_revision": cfg["revision"],
            "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
            "scan_tokenizer_id": cfg["scan_tokenizer_id"], "scan_repo": scan_cfg["repo"], "scan_revision": scan_cfg["revision"],
            "files": comparisons, "all_scan_files_byte_identical": strict_all,
            "core_tokenizer_json_byte_identical": core_json.get("byte_identical_to_scan_file") is True,
            "tokenizer_encoding_equivalence": encoding,
            "linkage_acceptable_for_scan_covariates": core_json.get("byte_identical_to_scan_file") is True and encoding.get("token_ids_equal_on_all_probe_texts") is True,
            "tokenizer_overlay": cfg.get("tokenizer_overlay"),
        })
    out = {
        "rows": rows,
        "all_models_strictly_byte_identical_on_every_scan_file": all(x["all_scan_files_byte_identical"] for x in rows),
        "all_models_acceptable_core_tokenizer_linkage": all(x["linkage_acceptable_for_scan_covariates"] for x in rows),
        "note": "Strict per-file byte identity is reported separately from scan-covariate linkage. Qwen3-4B-Instruct-2507 and Gemma-3-4B differ in tokenizer_config or auxiliary files while sharing the core tokenizer.json; probe-text token-id equality is therefore explicitly tested. The Qwen3 MLX artifact's missing chat template is overlaid from canonical upstream for inference.",
        "skipped_models": r2.SKIPPED_MODELS,
    }
    r2.ROUND2.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({
        "rows": len(rows),
        "strict": [x["all_scan_files_byte_identical"] for x in rows],
        "acceptable": [x["linkage_acceptable_for_scan_covariates"] for x in rows],
        "output": str(OUT),
    }, indent=2))


def hf_filename(scan_tokenizer_id: str, filename: str) -> str:
    return scan_tokenizer_id + "/" + filename


if __name__ == "__main__":
    main()
