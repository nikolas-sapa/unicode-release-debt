from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import regex
import sentencepiece as spm
import tokenizers as tokenizers_pkg
import unicodedata2

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_backend as cb
import confirmatory_common as cf
import tokscan

SCAN_PATH = cf.SCAN_DIR / "candidate_scan.jsonl"


def scan_one(backend: dict, row: dict) -> dict:
    text = row["text"]
    encoded = cb.effective_encode(backend, text)
    split, concatenated, _, partial = tokscan.split_codepoints(encoded["token_bytes"])
    clusters = len(regex.findall(r"\X", text))
    return {
        "candidate_id": row["candidate_id"],
        "carrier_id": backend["tokenizer_id"],
        "era": row["era"],
        "item_type": row["item_type"],
        "text": text,
        "text_sha256": row["text_sha256"],
        "token_ids": encoded["token_ids"],
        "token_count": encoded["token_count"],
        "raw_token_count": encoded["raw_token_count"],
        "dummy_prefix_tokens_removed": encoded["dummy_prefix_tokens_removed"],
        "dummy_prefix_stripped": encoded["dummy_prefix_stripped"],
        "grapheme_clusters": clusters,
        "tokens_per_cluster": encoded["token_count"] / clusters if clusters else None,
        "tokenizer_roundtrip_exact": int(encoded["decoded"] == text),
        "native_backend_ids_match": encoded["native_backend_ids_match"],
        "partial_utf8_token_count": partial,
        "split_codepoint_count": split,
        "token_bytes_concat_equals_input": None if concatenated is None else int(concatenated == text.encode()),
        "backend": backend["backend"],
    }


def main():
    candidates = cf.read_jsonl(cf.CONFIRMATORY / "candidate_universe.jsonl")
    if not candidates:
        raise RuntimeError("candidate universe missing; run build_confirmatory_candidates.py")
    cf.SCAN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts = Counter()
    mismatches = Counter()
    roundtrip_failures = Counter()
    concat_mismatches = Counter()
    rows_written = 0
    with SCAN_PATH.open("w") as fh:
        for cfg in cf.carrier_configs(include_descriptive=True):
            backend = cb.load_backend(cfg["carrier_id"])
            tic = time.perf_counter()
            for row in candidates:
                out = scan_one(backend, row)
                fh.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += 1
                counts[cfg["carrier_id"]] += 1
                if out["native_backend_ids_match"] == 0:
                    mismatches[cfg["carrier_id"]] += 1
                if not out["tokenizer_roundtrip_exact"]:
                    roundtrip_failures[cfg["carrier_id"]] += 1
                if out["token_bytes_concat_equals_input"] == 0:
                    concat_mismatches[cfg["carrier_id"]] += 1
            print(json.dumps({"carrier_id": cfg["carrier_id"], "rows": counts[cfg["carrier_id"]], "seconds": round(time.perf_counter() - tic, 2)}), flush=True)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scan_path": str(SCAN_PATH),
        "scan_sha256": cf.sha_file(SCAN_PATH),
        "scan_size": SCAN_PATH.stat().st_size,
        "candidate_rows": len(candidates),
        "carrier_rows": dict(counts),
        "rows": rows_written,
        "expected_rows": len(candidates) * len(cf.carrier_configs(include_descriptive=True)),
        "row_count_pass": rows_written == len(candidates) * len(cf.carrier_configs(include_descriptive=True)),
        "native_backend_id_mismatches": dict(mismatches),
        "roundtrip_failures": dict(roundtrip_failures),
        "token_bytes_concat_mismatches": dict(concat_mismatches),
        "backend_versions": {
            "tokenizers": tokenizers_pkg.__version__,
            "sentencepiece": spm.__version__,
            "regex": regex.__version__,
            "unicodedata2": unicodedata2.unidata_version,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    audit["audit_pass"] = (
        audit["row_count_pass"]
        and not any(mismatches.values())
        and not any(roundtrip_failures.values())
    )
    cf.write_json(cf.CONFIRMATORY / "candidate_scan.metadata.json", audit)
    print(json.dumps({"rows": rows_written, "audit_pass": audit["audit_pass"], "elapsed_seconds": round(audit["elapsed_seconds"], 2), "scan_sha256": audit["scan_sha256"]}, indent=2))
    if not audit["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
