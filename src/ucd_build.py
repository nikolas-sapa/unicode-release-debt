from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
STAMP = "2026-11-25"

UCD = {
    "DerivedAge.txt": ("https://www.unicode.org/Public/18.0.0/ucd/DerivedAge.txt", "58b04334e3a31f9612694bbc3f6bff4a302b7d424565bca9c47e30aa74f8dbac"),
    "UnicodeData.txt": ("https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt", "0736451de439ae7baf1425136617da495e09ee5afbe6e394374db7009ea08950"),
    "GraphemeBreakTest.txt": ("https://www.unicode.org/Public/18.0.0/ucd/auxiliary/GraphemeBreakTest.txt", "b0cf047ee94485bbdc846de2b902f5f8a815f6b674f9d04223cddadd91c9df31"),
    "GraphemeBreakProperty.txt": ("https://www.unicode.org/Public/18.0.0/ucd/auxiliary/GraphemeBreakProperty.txt", "0839dcb79e4ac639ecd538b1abf7c9d22e3f9dd265b7e182d33627aa4d75b45a"),
    "emoji-data.txt": ("https://www.unicode.org/Public/18.0.0/ucd/emoji/emoji-data.txt", "80d00f8e616a0ef27fd6b8de3b758c06383b5d917e2977709578e68baf733bf1"),
    "emoji-test.txt": ("https://www.unicode.org/Public/18.0.0/emoji/emoji-test.txt", "8f3735cda1f92a779d78af67cf86066bb1f07143dc22f2ac29394d9bc57ab21a"),
    "cldr-en-annotations.xml": ("https://raw.githubusercontent.com/unicode-org/cldr/release-49/common/annotations/en.xml", "14d0e9bb5003c085250df4ac7bd7ba1e26d92e74174d7a3a2f42dc16e07fb155"),
    "publicationdates.html": ("https://www.unicode.org/history/publicationdates.html", "20617b10bfefb97d947cc85a6824d360607e1f1cae2084eb5e846c3f61ceb234"),
    "emoji-frequency.html": ("https://www.unicode.org/emoji/frequency.html", "84d1c48a5d2eca87d7b41ce47b90d9158f86ce5b05d5ea6bc6943edf3876129b"),
    "tr29.html": ("https://unicode.org/reports/tr29/", "60fb49ee640a154d9e40468306c6b283edcef73be3af5ede778107b4787bdec8"),
    "tr51.html": ("https://unicode.org/reports/tr51/", "a12a8c986f50b97c2d82cbdabfe18928ac16375e74b6eb61b54654e132966658"),
}

HF = {
    "qwen2.5-0.5b-instruct": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775", "family": "Qwen",
        "license_status": "open", "release_date": "2024-09-19", "evidence": "https://qwenlm.github.io/blog/qwen2.5/",
        "files": {
            "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
            "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        },
    },
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "family": "Qwen",
        "license_status": "open", "release_date": "2025-04-29", "evidence": "https://qwenlm.github.io/blog/qwen3/",
        "files": {
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        },
    },
    "tinyllama-1.1b-chat": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6", "family": "Llama",
        "license_status": "open", "release_date": "2023-07-18", "evidence": "https://github.com/meta-llama/llama-models/blob/main/models/llama2/MODEL_CARD.md",
        "files": {
            "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
            "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
            "tokenizer_config.json": "7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a",
            "special_tokens_map.json": "82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db",
        },
    },
    "llama2-7b-hf-mirror": {
        "repo": "NousResearch/Llama-2-7b-hf", "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d", "family": "Llama",
        "license_status": "mirror; canonical meta-llama repository is license-gated (401 without acceptance)", "release_date": "2023-07-18", "evidence": "https://github.com/meta-llama/llama-models/blob/main/models/llama2/MODEL_CARD.md",
        "files": {
            "tokenizer.json": "f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7",
            "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
            "tokenizer_config.json": "37951d5a1e0c6a77f88c61f79972f17e889b62258aadff4c61eacd3fb567fe72",
            "special_tokens_map.json": "2ea88203b955648951a454e73a73967ff77ba4e79ae8ceb4400d4ab2cb040777",
        },
    },
    "gemma3-1b-it-mirror": {
        "repo": "unsloth/gemma-3-1b-it", "revision": "5b11413a10db4e486ef16a20101fd028f8f2499c", "family": "Gemma",
        "license_status": "mirror; canonical google/gemma-3-1b-it is license-gated (401 without acceptance)", "release_date": "2025-03-12", "evidence": "https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/",
        "files": {
            "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
            "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
            "tokenizer_config.json": "a622f3103725cffe1f19deed01e8675cea4443bc3eff27680f9484e5dc4acf7b",
            "special_tokens_map.json": "45a857d8a2495d0be30a5d2d6de03278195eb028b6e0b8efc248bfa697d65f05",
            "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
        },
    },
}

SKIN = set(range(0x1F3FB, 0x1F400))
EMOJI_DATES = {
    "E0.6": "2010-10-11", "E0.7": "2014-06-16", "E1.0": "2015-06-09", "E2.0": "2015-11-12",
    "E3.0": "2016-06-03", "E4.0": "2016-11-22", "E5.0": "2017-06-20", "E11.0": "2018-05-21",
    "E12.0": "2019-03-05", "E12.1": "2019-10-21", "E13.0": "2020-03-10", "E13.1": "2020-09-15",
    "E14.0": "2021-09-14", "E15.0": "2022-09-13", "E15.1": "2023-09-12", "E16.0": "2024-09-10",
    "E17.0": "2025-09-09", "E18.0": "2026-09-15",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, path: Path, expected: str) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path.read_bytes()) == expected:
        return path.stat().st_size, "cached-hash-verified"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    got = sha256(data)
    if got != expected:
        raise ValueError(f"SHA256 mismatch for {url}: got {got}, expected {expected}")
    path.write_bytes(data)
    return len(data), "downloaded-hash-verified"


def replicate_hashes(entries: list[dict]) -> dict:
    cache_path = RESULTS / "hash_replication.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    out = {}
    for row in entries:
        with urllib.request.urlopen(row["url"], timeout=120) as response:
            data = response.read()
        out[row["url"]] = {"sha256": sha256(data), "matches_pinned_copy": sha256(data) == row["sha256"]}
    cache_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def download_all() -> list[dict]:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for filename, (url, expected) in UCD.items():
        size, status = fetch(url, DATA / "unicode" / filename, expected)
        rows.append({"kind": "unicode", "file": filename, "url": url, "path": f"data/unicode/{filename}", "sha256": expected, "size": size, "status": status})
    for tid, cfg in HF.items():
        for filename, expected in cfg["files"].items():
            url = f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/{filename}"
            size, status = fetch(url, DATA / "hf" / tid / filename, expected)
            rows.append({"kind": "tokenizer", "tokenizer_id": tid, "repo": cfg["repo"], "revision": cfg["revision"], "file": filename, "url": url, "path": f"data/hf/{tid}/{filename}", "sha256": expected, "size": size, "status": status})
    retrieval = datetime.now(timezone.utc).isoformat()
    replication = replicate_hashes(rows)
    for row in rows:
        row["retrieval_date"] = retrieval
        row["independent_download_sha256_match"] = replication[row["url"]]["matches_pinned_copy"]
    (RESULTS / "manifest.json").write_text(json.dumps({"generated_utc": retrieval, "inputs": rows}, indent=2))
    return rows


def rng_points(text: str) -> tuple[int, int]:
    bits = text.strip().split("..")
    return int(bits[0], 16), int(bits[-1], 16)


def parse_age(path: Path) -> tuple[dict[int, str], int]:
    ages, rows = {}, 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        rows += 1
        left = line.split("#", 1)[0]
        span, version = [x.strip() for x in left.split(";")[:2]]
        lo, hi = rng_points(span)
        for cp in range(lo, hi + 1):
            ages[cp] = version
    return ages, rows


def parse_unicode_data(path: Path) -> tuple[dict[int, dict], int]:
    chars, rows, pending = {}, 0, None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        rows += 1
        fields = line.split(";")
        cp, name, category = int(fields[0], 16), fields[1], fields[2]
        if name.endswith(", First>"):
            pending = (cp, name, category)
            continue
        if name.endswith(", Last>") and pending:
            first, first_name, first_category = pending
            base = first_name[1:].replace(", First>", "")
            for x in range(first, cp + 1):
                chars[x] = {"name": base, "category": first_category, "range": True, "source_line": line_no}
            pending = None
            continue
        chars[cp] = {"name": name, "category": category, "range": False, "source_line": line_no}
    return chars, rows


def parse_emoji_data(path: Path) -> tuple[dict[str, set[int]], int]:
    props, rows = defaultdict(set), 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        rows += 1
        left = line.split("#", 1)[0]
        span, prop = [x.strip() for x in left.split(";")[:2]]
        lo, hi = rng_points(span)
        props[prop].update(range(lo, hi + 1))
    return props, rows


def surface_key(cps: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(cp for cp in cps if cp not in SKIN)


def strict_atomic(cps: tuple[int, ...]) -> bool:
    return len(cps) == 1 or (len(cps) == 2 and cps[1] == 0xFE0F)


def parse_emoji_test(path: Path) -> tuple[list[dict], int, list[str]]:
    rows, data_lines, errors = [], 0, []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        data_lines += 1
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
    return rows, data_lines, errors


class FrequencyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = self.in_cell = False
        self.row, self.cell, self.rows = [], "", []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_row, self.row = True, []
        elif self.in_row and tag == "td":
            self.in_cell, self.cell = True, ""

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.row.append(" ".join(self.cell.split()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell += data


def parse_frequency(path: Path) -> dict[str, int]:
    parser = FrequencyParser()
    parser.feed(path.read_text(errors="replace"))
    out = {}
    for row in parser.rows:
        if len(row) >= 2 and row[0].isdigit():
            for emoji in row[1].split():
                out[emoji] = int(row[0])
    return out


def unicode_release_dates(path: Path) -> dict[str, str]:
    text = re.sub(r"\s+", " ", path.read_text(errors="replace"))
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    out = {}
    for version, year, month, day in re.findall(r"<td>(\d+\.\d+(?:\.\d+)?)</td> <td>(\d{4})</td> <td>([A-Za-z]+)(?: (\d+))?</td>", text):
        key = ".".join(version.split(".")[:2])
        out[key] = f"{year}-{months[month]:02d}-{int(day):02d}" if day else f"{year}-{months[month]:02d}"
    return out


def build_emoji_universe(rows: list[dict]) -> tuple[list[dict], dict]:
    groups, audit = defaultdict(list), defaultdict(lambda: {"raw": 0, "families": 0})
    fully = [r for r in rows if r["status"] == "fully-qualified"]
    for row in fully:
        cps = row["codepoints"]
        typ = "atomic" if strict_atomic(cps) else ("zwj" if 0x200D in cps else "other")
        if typ in ("atomic", "zwj"):
            audit[(row["emoji_version"], typ)]["raw"] += 1
        if typ != "other":
            groups[(row["emoji_version"], typ, surface_key(cps))].append(row)
    universe = []
    for (version, typ, key), members in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        first = sorted(members, key=lambda r: r["source_line"])[0]
        audit[(version, typ)]["families"] += 1
        universe.append({**first, "sequence_type": typ, "surface_family_key": " ".join(f"U+{x:04X}" for x in key), "variant_count": len(members)})
    counts = {f"{v}/{t}": dict(vals) for (v, t), vals in sorted(audit.items())}
    return universe, counts


def main():
    rows = download_all()
    ages, age_rows = parse_age(DATA / "unicode" / "DerivedAge.txt")
    chars, unicode_rows = parse_unicode_data(DATA / "unicode" / "UnicodeData.txt")
    emoji_props, emoji_data_rows = parse_emoji_data(DATA / "unicode" / "emoji-data.txt")
    emoji_rows, emoji_test_lines, parse_errors = parse_emoji_test(DATA / "unicode" / "emoji-test.txt")
    universe, audit = build_emoji_universe(emoji_rows)
    frequency = parse_frequency(DATA / "unicode" / "emoji-frequency.html")
    release_dates = unicode_release_dates(DATA / "unicode" / "publicationdates.html")
    cldr_text = (DATA / "unicode" / "cldr-en-annotations.xml").read_text(errors="replace")
    with (RESULTS / "emoji_universe.jsonl").open("w") as fh:
        for row in universe:
            out = dict(row)
            out["codepoints"] = [f"U+{x:04X}" for x in row["codepoints"]]
            out["text"] = "".join(chr(x) for x in row["codepoints"])
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    summary = {
        "input_files": len(rows), "all_hashes_verified": all(r["sha256"] for r in rows),
        "independent_downloads_verified": all(r["independent_download_sha256_match"] for r in rows),
        "derivedage_rows": age_rows, "unicodedata_rows": unicode_rows, "assigned_expanded_count": len(chars),
        "emoji_data_rows": emoji_data_rows, "emoji_test_data_lines": emoji_test_lines,
        "emoji_test_parse_errors": parse_errors, "fully_qualified_rows": sum(1 for r in emoji_rows if r["status"] == "fully-qualified"),
        "audit_counts": audit, "cldr_annotation_rows": len(re.findall(r"<annotation\s", cldr_text)),
        "frequency_rows_mapped": len(frequency), "unicode_release_dates": {k: release_dates.get(k) for k in ["13.0", "14.0", "15.0", "15.1", "16.0", "17.0", "18.0"]},
        "emoji_dates_source": "pinned tr51.html Emoji Versions table", "emoji_dates": EMOJI_DATES,
    }
    (RESULTS / "ucd_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"inputs": len(rows), "parse_errors": len(parse_errors), "audit_E15.1_zwj": audit.get("E15.1/zwj"), "audit_E17_zwj": audit.get("E17.0/zwj")}, indent=2))


if __name__ == "__main__":
    main()
