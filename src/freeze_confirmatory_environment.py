from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf

PINNED = {
    "numpy": "2.5.5",
    "regex": "2026.11.24",
    "unicodedata2": "18.0.0",
    "tokenizers": "0.23.3",
    "sentencepiece": "0.2.3",
    "transformers": "5.24.1",
    "mlx": "0.32.5",
    "mlx-lm": "0.32.2",
}


def git(args: list[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=cf.ROOT, text=True).strip()


def main():
    package_versions = {}
    mismatches = {}
    for package, expected in PINNED.items():
        try:
            got = metadata.version(package)
        except metadata.PackageNotFoundError:
            got = None
        package_versions[package] = {"expected": expected, "installed": got, "match": got == expected}
        if got != expected:
            mismatches[package] = {"expected": expected, "installed": got}
    unicode_files = {}
    for path in sorted((cf.DATA / "unicode").iterdir()):
        if path.is_file():
            unicode_files[path.name] = {"sha256": cf.sha_file(path), "size": path.stat().st_size}
    disk = shutil.disk_usage(cf.ROOT)
    hf_cache = Path.home() / ".cache" / "huggingface"
    obj = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git(["branch", "--show-current"]), "commit": git(["rev-parse", "HEAD"]), "status_porcelain": git(["status", "--porcelain"])},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "executable": sys.executable},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "mac_ver": platform.mac_ver(), "unicodedata_version": unicodedata.unidata_version},
        "packages": package_versions,
        "package_pin_mismatches": mismatches,
        "unicode_input_files": unicode_files,
        "hf_cache_path": str(hf_cache),
        "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
        "disk_gate": {"free_gib": disk.free / 2**30, "minimum_required_gib_without_bf16_comparator_downloads": 35, "pass": disk.free / 2**30 >= 35},
        "user_resolution_quantization": "expanded bf16 comparator downloads skipped before execution; quantization validation is narrower and no new bf16 comparator weight download is permitted",
    }
    cf.write_json(cf.CONFIRMATORY / "environment_freeze.json", obj)
    if mismatches:
        raise SystemExit(json.dumps({"package_pin_mismatches": mismatches}, indent=2))
    if not obj["disk_gate"]["pass"]:
        raise SystemExit(json.dumps({"disk_gate": obj["disk_gate"]}, indent=2))
    print(json.dumps({"wrote": str(cf.CONFIRMATORY / "environment_freeze.json"), "packages_match": True, "free_gib": obj["disk_gate"]["free_gib"]}, indent=2))


if __name__ == "__main__":
    main()
