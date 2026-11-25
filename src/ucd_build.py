from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PINS = json.loads((Path(__file__).parent / "pinned_inputs.json").read_text())
UCD = {k: tuple(v) for k, v in PINS["ucd"].items()}
HF = PINS["hf"]
EMOJI_ORDER = ["E0.6", "E0.7", "E1.0", "E2.0", "E3.0", "E4.0", "E5.0", "E11.0", "E12.0", "E12.1", "E13.0", "E13.1", "E14.0", "E15.0", "E15.1", "E16.0", "E17.0", "E18.0"]
EMOJI_DATES = {key: PINS["emoji_dates"][key] for key in EMOJI_ORDER}
SKIN = set(range(0x1F3FB, 0x1F400))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, path: Path, expected: str) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path.read_bytes()) == expected:
        return path.stat().st_size, "cached-hash-verified"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if sha256(data) != expected:
        raise ValueError(f"SHA256 mismatch for {url}: got {sha256(data)}, expected {expected}")
    path.write_bytes(data)
    return len(data), "downloaded-hash-verified"


def replicate_hashes(entries: list[dict]) -> dict:
    cache_path = RESULTS / "hash_replication.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out = {}
    for row in entries:
        cached = cache.get(row["url"])
        if cached and cached.get("sha256") == row["sha256"] and cached.get("matches_pinned_copy") is True:
            out[row["url"]] = cached
            continue
        with urllib.request.urlopen(row["url"], timeout=120) as response:
            data = response.read()
        digest = sha256(data)
        out[row["url"]] = {"sha256": digest, "matches_pinned_copy": digest == row["sha256"]}
    cache_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def download_all() -> list[dict]:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for filename, (url, expected) in UCD.items():
        path = DATA / "unicode" / filename
        size, status = fetch(url, path, expected)
        rows.append({"kind": "unicode", "file": filename, "url": url, "path": f"data/unicode/{filename}", "sha256": expected, "size": size, "status": status})
    for tid, cfg in HF.items():
        for filename, expected in cfg["files"].items():
            url = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/{filename}"
            path = DATA / "hf" / tid / filename
            size, status = fetch(url, path, expected)
            rows.append({"kind": "tokenizer", "tokenizer_id": tid, "repo": cfg["repo"], "revision": cfg["revision"], "file": filename, "url": url, "path": f"data/hf/{tid}/{filename}", "sha256": expected, "size": size, "status": status})
    replication = replicate_hashes(rows)
    generated = datetime.now(timezone.utc).isoformat()
    for row in rows:
        path = ROOT / row["path"]
        row["retrieval_date"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        row["independent_download_sha256_match"] = replication[row["url"]]["matches_pinned_copy"]
    (RESULTS / "manifest.json").write_text(json.dumps({"generated_utc": generated, "inputs": rows}, indent=2))
    return rows


def rng_points(text: str) -> tuple[int, int]:
    bits = text.strip().split("..")
    return int(bits[0], 16), int(bits[-1], 16)


def data_rows(path: Path):
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line and not line.startswith("#"):
            yield line_no, line.split("#", 1)[0]


def parse_age(path: Path) -> tuple[dict[int, str], int]:
    ages, rows = {}, 0
    for _, line in data_rows(path):
        rows += 1
        span, version = [x.strip() for x in line.split(";")[:2]]
        lo, hi = rng_points(span)
        ages.update(dict.fromkeys(range(lo, hi + 1), version))
    return ages, rows


def parse_unicode_data(path: Path) -> tuple[dict[int, dict], int]:
    chars, rows, pending = {}, 0, None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        rows += 1
        fields = line.split(";")
        cp, name, category = int(fields[0], 16), fields[1], fields[2]
        if name.endswith(", First>"):
            pending = (cp, name, category, line_no)
        elif name.endswith(", Last>") and pending:
            first, first_name, first_category, first_line = pending
            base = first_name[1:].replace(", First>", "")
            for x in range(first, cp + 1):
                chars[x] = {"name": base, "category": first_category, "range": True, "source_line": line_no}
            pending = None
        else:
            chars[cp] = {"name": name, "category": category, "range": False, "source_line": line_no}
    return chars, rows


def parse_emoji_data(path: Path) -> tuple[dict[str, set[int]], int]:
    props, rows = defaultdict(set), 0
    for _, line in data_rows(path):
        rows += 1
        span, prop = [x.strip() for x in line.split(";")[:2]]
        lo, hi = rng_points(span)
        props[prop].update(range(lo, hi + 1))
    return props, rows


def surface_key(cps: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(cp for cp in cps if cp not in SKIN)


def strict_atomic(cps: tuple[int, ...]) -> bool:
    return len(cps) == 1 or (len(cps) == 2 and cps[1] == 0xFE0F)


def parse_emoji_test(path: Path) -> tuple[list[dict], int, list[str]]:
    rows, errors = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        try:
            left, comment = line.split("#", 1)
            cp_text, status = [x.strip() for x in left.split(";")[:2]]
            match = re.match(r"^\S+\s+(E[0-9.]+)\s+(.*)$", comment.strip())
            if not match:
                raise ValueError("missing emoji version/comment")
            cps = tuple(int(x, 16) for x in cp_text.split())
            rows.append({"source_line": line_no, "codepoints": cps, "status": status, "emoji_version": match.group(1), "name": match.group(2)})
        except Exception as exc:
            errors.append(f"line {line_no}: {exc}")
    return rows, len(rows) + len(errors), errors


def parse_frequency(path: Path) -> dict[str, int]:
    text = path.read_text(errors="replace")
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", "", cell)).split()) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)]
        if len(cells) >= 2 and cells[0].isdigit():
            for emoji in cells[1].split():
                out[emoji] = int(cells[0])
    return out


def unicode_release_dates(path: Path) -> dict[str, str]:
    text = re.sub(r"\s+", " ", path.read_text(errors="replace"))
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    out = {}
    for version, year, month, day in re.findall(r"<td>(\d+\.\d+(?:\.\d+)?)</td> <td>(\d{4})</td> <td>([A-Za-z]+)(?: (\d+))?</td>", text):
        key = ".".join(version.split(".")[:2])
        out[key] = f"{year}-{months[month]:02d}-{int(day):02d}" if day else f"{year}-{months[month]:02d}"
    return out


def parse_emoji_dates(tr51_path: Path, unicode_dates: dict[str, str]) -> dict[str, str]:
    text = tr51_path.read_text(errors="replace")
    out = {"E" + v: d for v, d in re.findall(r"<td>Emoji\s*([0-9.]+)</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>", text)}
    out["E0.6"], out["E0.7"] = unicode_dates.get("6.0"), unicode_dates.get("7.0")
    return {key: out[key] for key in EMOJI_DATES}


def build_emoji_universe(rows: list[dict]) -> tuple[list[dict], dict]:
    groups, audit = defaultdict(list), defaultdict(lambda: {"raw": 0, "families": 0})
    for row in [r for r in rows if r["status"] == "fully-qualified"]:
        cps = row["codepoints"]
        typ = "atomic" if strict_atomic(cps) else ("zwj" if 0x200D in cps else "other")
        if typ == "other":
            continue
        audit[(row["emoji_version"], typ)]["raw"] += 1
        groups[(row["emoji_version"], typ, surface_key(cps))].append(row)
    universe = []
    for (version, typ, key), members in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        first = sorted(members, key=lambda r: r["source_line"])[0]
        audit[(version, typ)]["families"] += 1
        universe.append({**first, "sequence_type": typ, "surface_family_key": " ".join(f"U+{x:04X}" for x in key), "variant_count": len(members)})
    return universe, {f"{v}/{t}": dict(vals) for (v, t), vals in sorted(audit.items())}


def main():
    rows = download_all()
    ages, age_rows = parse_age(DATA / "unicode" / "DerivedAge.txt")
    chars, unicode_rows = parse_unicode_data(DATA / "unicode" / "UnicodeData.txt")
    emoji_props, emoji_data_rows = parse_emoji_data(DATA / "unicode" / "emoji-data.txt")
    emoji_rows, emoji_test_lines, parse_errors = parse_emoji_test(DATA / "unicode" / "emoji-test.txt")
    universe, audit = build_emoji_universe(emoji_rows)
    frequency = parse_frequency(DATA / "unicode" / "emoji-frequency.html")
    release_dates = unicode_release_dates(DATA / "unicode" / "publicationdates.html")
    parsed_emoji_dates = parse_emoji_dates(DATA / "unicode" / "tr51.html", release_dates)
    if parsed_emoji_dates != EMOJI_DATES:
        raise ValueError(f"emoji date table mismatch: {parsed_emoji_dates}")
    surface_versions = defaultdict(set)
    for item in universe:
        if item["sequence_type"] == "zwj":
            surface_versions[item["surface_family_key"]].add(item["emoji_version"])
    collisions = {key: sorted(vals) for key, vals in surface_versions.items() if len(vals) > 1}
    with (RESULTS / "emoji_universe.jsonl").open("w") as fh:
        for row in universe:
            out = dict(row)
            out["codepoints"] = [f"U+{x:04X}" for x in row["codepoints"]]
            out["text"] = "".join(chr(x) for x in row["codepoints"])
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    cldr_text = (DATA / "unicode" / "cldr-en-annotations.xml").read_text(errors="replace")
    summary = {
        "input_files": len(rows), "all_hashes_verified": all(r["sha256"] for r in rows),
        "independent_downloads_verified": all(r["independent_download_sha256_match"] for r in rows),
        "derivedage_rows": age_rows, "unicodedata_rows": unicode_rows, "assigned_expanded_count": len(chars),
        "emoji_data_rows": emoji_data_rows, "emoji_test_data_lines": emoji_test_lines,
        "emoji_test_parse_errors": parse_errors, "fully_qualified_rows": sum(1 for r in emoji_rows if r["status"] == "fully-qualified"),
        "audit_counts": audit, "cldr_annotation_rows": len(re.findall(r"<annotation\s", cldr_text)),
        "frequency_rows_mapped": len(frequency), "unicode_release_dates": {k: release_dates.get(k) for k in ["13.0", "14.0", "15.0", "15.1", "16.0", "17.0", "18.0"]},
        "emoji_dates_source": "pinned tr51.html Emoji Versions table", "emoji_dates": EMOJI_DATES,
        "emoji_dates_match_tr51_parse": True, "cross_version_zwj_surface_key_collisions": collisions,
    }
    (RESULTS / "ucd_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"inputs": len(rows), "parse_errors": len(parse_errors), "audit_E15.1_zwj": audit.get("E15.1/zwj"), "audit_E17_zwj": audit.get("E17.0/zwj")}, indent=2))


if __name__ == "__main__":
    main()
