from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tokscan
import ucd_build as ucd
import vyce_common as vc

SCAN_OUT = vc.VYCE / "tokenizer_scan_extension.jsonl"
SCAN_META_OUT = vc.VYCE / "tokenizer_scan_extension.metadata.json"
INVENTORY_OUT = vc.VYCE / "tokenizer_inventory_extension.json"
MANIFEST_OUT = vc.VYCE / "tokenizer_extension_manifest.json"

TOKENIZER_EXTENSIONS = {
    "openai-gpt-oss-120b": {
        "repo": "openai/gpt-oss-120b",
        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "family": "OpenAI-gpt-oss",
        "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    },
    "deepseek-v4-flash": {
        "repo": "deepseek-ai/DeepSeek-V4-Flash",
        "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "family": "DeepSeek",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "deepseek-v4.1-flash": {
        "repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": "314113f8a7dadb388215396ba5fbee2c610a04d6",
        "family": "DeepSeek",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    "glm-5.3-flash": {
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
        "family": "GLM",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def urlopen_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "unicode-release-debt-research-artifact"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def hf_api_metadata(repo: str) -> dict:
    with urllib.request.urlopen("https://huggingface.co/api/models/" + repo, timeout=60) as response:
        data = json.loads(response.read().decode())
    return {k: data.get(k) for k in ("id", "sha", "createdAt", "lastModified", "private", "gated", "downloads")}


def download_extension_artifacts() -> dict:
    previous = json.loads(MANIFEST_OUT.read_text()) if MANIFEST_OUT.exists() else None
    previous_files = {}
    if previous:
        for row in previous.get("artifacts", []):
            previous_files[(row["tokenizer_id"], row["file"], row["revision"])] = row
    artifact_rows = []
    api_rows = {}
    for tid, cfg in TOKENIZER_EXTENSIONS.items():
        api = hf_api_metadata(cfg["repo"])
        api_rows[tid] = api
        if api.get("sha") != cfg["revision"]:
            raise RuntimeError(f"{tid}: pinned revision {cfg['revision']} no longer matches HF api sha {api.get('sha')}; refusing unpinned drift")
        files = {}
        for filename in cfg["files"]:
            url = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/{filename}"
            dest = ucd.DATA / "hf" / tid / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            pin_key = (tid, filename, cfg["revision"])
            expected = previous_files.get(pin_key, {}).get("sha256")
            if dest.exists():
                digest = sha256(dest.read_bytes())
                if expected and digest != expected:
                    raise RuntimeError(f"{tid}/{filename}: cached artifact hash differs from extension manifest")
                if not expected:
                    data = urlopen_bytes(url)
                    if sha256(data) != digest:
                        raise RuntimeError(f"{tid}/{filename}: unpinned cache does not match pinned remote revision")
            else:
                data = urlopen_bytes(url)
                digest = sha256(data)
                if expected and digest != expected:
                    raise RuntimeError(f"{tid}/{filename}: downloaded hash differs from extension manifest")
                dest.write_bytes(data)
            digest = sha256(dest.read_bytes())
            independent_digest = previous_files.get(pin_key, {}).get("independent_download_sha256")
            independent_match = previous_files.get(pin_key, {}).get("independent_download_sha256_match")
            if independent_digest is None:
                independent_digest = sha256(urlopen_bytes(url))
                independent_match = independent_digest == digest
            files[filename] = digest
            artifact_rows.append({
                "tokenizer_id": tid, "repo": cfg["repo"], "revision": cfg["revision"], "file": filename, "url": url,
                "path": str(dest.relative_to(ucd.ROOT)), "sha256": digest, "size": dest.stat().st_size,
                "independent_download_sha256": independent_digest, "independent_download_sha256_match": independent_match,
                "retrieval_date": datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc).isoformat(),
            })
        created = (api.get("createdAt") or "")[:10] or "unknown"
        ucd.HF[tid] = {
            "repo": cfg["repo"], "revision": cfg["revision"], "family": cfg["family"], "files": files,
            "license_status": "open", "release_date": created, "evidence": "https://huggingface.co/api/models/" + cfg["repo"],
        }
    if not all(x["independent_download_sha256_match"] for x in artifact_rows):
        raise RuntimeError("independent tokenizer artifact download hash mismatch")
    manifest = {
        "generated_utc": vc.utcnow(),
        "scope": "VYCE exploratory hosted-arm tokenizer extension; local scan only, no hosted API cost",
        "artifacts": artifact_rows,
        "hf_api_metadata": api_rows,
        "revision_policy": "exact HF commit revision pinned; tokenizer file bytes SHA-256 pinned; independent second download must match",
    }
    vc.write_json(MANIFEST_OUT, manifest)
    return manifest


def scan_extension() -> dict:
    rows = [json.loads(line) for line in (ucd.RESULTS / "scan_units.jsonl").read_text().splitlines() if line.strip()]
    out = []
    mismatches = {}
    for tid in sorted(TOKENIZER_EXTENSIONS):
        backend = tokscan.load_backend(tid)
        bad = 0
        for row in rows:
            metrics = tokscan.scan_text(backend, row["text"])
            bad += metrics["native_backend_ids_match"] == 0
            out.append({k: v for k, v in row.items() if k != "text"} | metrics | {"text": row["text"], "tokenizer_id": tid, "family": ucd.HF[tid]["family"]})
        mismatches[tid] = bad
    vc.write_jsonl(SCAN_OUT, out)
    meta = {
        "input": "results/scan_units.jsonl", "output": str(SCAN_OUT), "rows": len(out),
        "tokenizers": sorted(TOKENIZER_EXTENSIONS), "native_backend_id_mismatches": mismatches,
        "host_python_version": sys.version.split()[0],
        "tokenizers_library_version": tokscan.tokenizers_pkg.__version__,
        "sentencepiece_library_version": tokscan.spm.__version__,
    }
    vc.write_json(SCAN_META_OUT, meta)
    return meta


def inventory_extension() -> dict:
    ages, _ = ucd.parse_age(ucd.DATA / "unicode" / "DerivedAge.txt")
    chars, _ = ucd.parse_unicode_data(ucd.DATA / "unicode" / "UnicodeData.txt")
    rows, _, _ = ucd.parse_emoji_test(ucd.DATA / "unicode" / "emoji-test.txt")
    universe, _ = ucd.build_emoji_universe(rows)
    inventory = []
    for tid in sorted(TOKENIZER_EXTENSIONS):
        backend = tokscan.load_backend(tid)
        row = tokscan.inventory_row(tid, ages, chars, backend)
        vocab_bytes = tokscan.vocab_bytes_map(backend)
        def new_curve():
            return {"n": 0, "single": 0, "vocab_exact": 0, "atomic_n": 0, "atomic_single": 0, "atomic_vocab_exact": 0, "atomic_split": [], "zwj_n": 0, "zwj_single": 0, "zwj_vocab_exact": 0, "zwj_tpc": [], "zwj_split": []}
        curves = defaultdict(new_curve)
        newest = {"families": 0, "pipeline_single_token_count": 0, "exact_vocab_entry_count": 0, "exact_vocab_entries": [], "pipeline_single_tokens": []}
        native_mismatches = 0
        for item in universe:
            text = "".join(chr(x) for x in item["codepoints"])
            metrics = tokscan.scan_text(backend, text)
            native_mismatches += metrics["native_backend_ids_match"] == 0
            exact_vocab = int(text.encode() in vocab_bytes)
            x = curves[item["emoji_version"]]
            x["n"] += 1
            x["single"] += metrics["whole_sequence_token"]
            x["vocab_exact"] += exact_vocab
            typ = item["sequence_type"]
            if typ in ("atomic", "zwj"):
                x[f"{typ}_n"] += 1
                x[f"{typ}_single"] += metrics["whole_sequence_token"]
                x[f"{typ}_vocab_exact"] += exact_vocab
                if metrics["split_codepoint_count"] is not None:
                    x[f"{typ}_split"].append(metrics["split_codepoint_count"])
                if typ == "zwj":
                    x["zwj_tpc"].append(metrics["tokens_per_cluster"])
            if tokscan.newest_cohort(item):
                newest["families"] += 1
                newest["pipeline_single_token_count"] += metrics["whole_sequence_token"]
                newest["exact_vocab_entry_count"] += exact_vocab
                if exact_vocab:
                    newest["exact_vocab_entries"].append({"emoji_version": item["emoji_version"], "name": item["name"], "vocab_piece": vocab_bytes[text.encode()]})
                if metrics["whole_sequence_token"]:
                    newest["pipeline_single_tokens"].append({"emoji_version": item["emoji_version"], "name": item["name"]})
        row["native_backend_id_mismatches_in_rgi_scan"] = native_mismatches
        row["pilot_newest_cohort_exactness"] = newest
        row["rgi_single_token_rate_by_emoji_version"] = {}
        mean = lambda z: sum(z) / len(z) if z else None
        for version, x in sorted(curves.items(), key=lambda z: list(ucd.EMOJI_DATES).index(z[0])):
            row["rgi_single_token_rate_by_emoji_version"][version] = {
                "families": x["n"], "single_token_count": x["single"], "single_token_rate": x["single"] / x["n"],
                "exact_vocab_count": x["vocab_exact"], "exact_vocab_rate": x["vocab_exact"] / x["n"],
                "atomic_families": x["atomic_n"], "atomic_single_token_count": x["atomic_single"], "atomic_single_token_rate": x["atomic_single"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_exact_vocab_count": x["atomic_vocab_exact"], "atomic_exact_vocab_rate": x["atomic_vocab_exact"] / x["atomic_n"] if x["atomic_n"] else None,
                "atomic_mean_split_codepoint_count": mean(x["atomic_split"]),
                "zwj_families": x["zwj_n"], "zwj_single_token_count": x["zwj_single"], "zwj_single_token_rate": x["zwj_single"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_exact_vocab_count": x["zwj_vocab_exact"], "zwj_exact_vocab_rate": x["zwj_vocab_exact"] / x["zwj_n"] if x["zwj_n"] else None,
                "zwj_mean_tokens_per_cluster": mean(x["zwj_tpc"]), "zwj_mean_split_codepoint_count": mean(x["zwj_split"]),
            }
        inventory.append(row)
    continuity = []
    for a, b in (("deepseek-v4-flash", "deepseek-v4.1-flash"),):
        ra = next(x for x in inventory if x["tokenizer_id"] == a)
        rb = next(x for x in inventory if x["tokenizer_id"] == b)
        continuity.append({"left": a, "right": b, "regular_vocab_equal": ra["regular_vocab_sha256"] == rb["regular_vocab_sha256"], "merges_equal": ra["merges_sha256"] == rb["merges_sha256"], "left_vocab": ra["regular_vocab_size"], "right_vocab": rb["regular_vocab_size"], "left_merges": ra["merge_count"], "right_merges": rb["merge_count"]})
    payload = {"generated_utc": vc.utcnow(), "scope": "VYCE exploratory hosted-arm tokenizer extension", "tokenizers": inventory, "continuity": continuity}
    vc.write_json(INVENTORY_OUT, payload)
    return {"tokenizers": len(inventory), "continuity": continuity}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    manifest = download_extension_artifacts()
    scan_meta = scan_extension()
    inventory_meta = inventory_extension()
    print(json.dumps({"artifacts": len(manifest["artifacts"]), "scan_rows": scan_meta["rows"], "inventory_tokenizers": inventory_meta["tokenizers"], "continuity": inventory_meta["continuity"]}, indent=2))


if __name__ == "__main__":
    main()
