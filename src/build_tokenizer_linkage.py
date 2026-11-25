from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
BEHAVIORAL = bp.BEHAVIORAL


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    for cfg in bp.model_configs():
        scan_cfg = ucd.HF[cfg["scan_tokenizer_id"]]
        comparisons = []
        for filename, scan_hash in sorted(scan_cfg["files"].items()):
            scan_path = ROOT / "data" / "hf" / cfg["scan_tokenizer_id"] / filename
            row = {"file": filename, "scan_artifact_sha256": scan_hash, "scan_file_present": scan_path.exists()}
            try:
                behavior_path = Path(hf_hub_download(cfg["hf_repo"], filename, revision=cfg["revision"]))
                behavior_hash = sha_file(behavior_path)
                row.update({"behavior_file_sha256": behavior_hash, "byte_identical_to_scan_file": behavior_hash == scan_hash, "access": "downloaded-or-cache-verified"})
            except Exception as exc:
                row.update({"behavior_file_sha256": None, "byte_identical_to_scan_file": None, "access": type(exc).__name__})
            comparisons.append(row)
        rows.append({
            "model_id": cfg["model_id"], "behavior_repo": cfg["hf_repo"], "behavior_revision": cfg["revision"],
            "scan_tokenizer_id": cfg["scan_tokenizer_id"], "scan_repo": scan_cfg["repo"], "scan_revision": scan_cfg["revision"],
            "files": comparisons, "all_common_files_byte_identical": all(x["byte_identical_to_scan_file"] is True for x in comparisons),
        })
    out = {"rows": rows, "note": "Qwen comparisons establish exact byte identity across model-size variants; Gemma compares the accessible mirror artifact already used by the scan because canonical access is gated."}
    BEHAVIORAL.mkdir(parents=True, exist_ok=True)
    (BEHAVIORAL / "tokenizer_linkage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"rows": len(rows), "all": [x["all_common_files_byte_identical"] for x in rows]}, indent=2))


if __name__ == "__main__":
    main()
