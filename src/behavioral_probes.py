from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
BEHAVIORAL = RESULTS / "behavioral"
MAX_TOKENS = 64
BATCH_SIZE = 32

COPY_TEMPLATE = "Copy the text between `<IN>` and `</IN>` exactly. Preserve every Unicode code point, including invisible joiners and variation selectors. Put only the copied text inside `<OUT>` and `</OUT>`. Input: `<IN>\n{target}\n</IN>`"
COUNT_TEMPLATE = "Count the extended grapheme clusters (user-perceived characters), not bytes or code points. Reply with one integer only. ZWJ joins emoji components into one cluster; combining marks and variation selectors attach to their base character. Input:\n{target}"
DECOMP_TEMPLATE = "List the Unicode scalar values in order as hexadecimal codepoints. Use exactly this format: `U+1F600 U+200D`. Input:\n{target}"
COMPARE_TEMPLATE = "Compare A and B byte-exactly. If identical output `SAME`; if same codepoints in a different order output `REORDERED`; otherwise output `DIFFERENT`. Reply with one label only.\nA:\n{a}\nB:\n{b}"

PROMPT_TEMPLATES = {
    "copy": COPY_TEMPLATE,
    "count": COUNT_TEMPLATE,
    "decomposition": DECOMP_TEMPLATE,
    "comparison": COMPARE_TEMPLATE,
}

MODELS = [
    {
        "model_id": "tinyllama-1.1b-chat-v1.0-bf16",
        "family": "Llama2-vintage",
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "upstream_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "upstream_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "scan_tokenizer_id": "tinyllama-1.1b-chat",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "chat_template_kwargs": {},
        "selection_note": "mandatory Llama-2-tokenizer vintage carrier; canonical unquantized checkpoint",
    },
    {
        "model_id": "qwen2.5-1.5b-instruct-bf16",
        "family": "Qwen2.5",
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "upstream_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "upstream_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "scan_tokenizer_id": "qwen2.5-0.5b-instruct",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "chat_template_kwargs": {},
        "selection_note": "1.5B chosen instead of 3B under the pilot compute/memory budget; canonical unquantized checkpoint; tokenizer files are byte-identical to the scanned 0.5B tokenizer artifact",
    },
    {
        "model_id": "qwen3-1.7b-bf16",
        "family": "Qwen3",
        "hf_repo": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "upstream_repo": "Qwen/Qwen3-1.7B",
        "upstream_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "scan_tokenizer_id": "qwen3-0.6b",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "chat_template_kwargs": {"enable_thinking": False},
        "selection_note": "smaller Qwen3 instruct/post-trained variant chosen instead of 4B under the pilot compute/memory budget; canonical unquantized checkpoint; tokenizer files are byte-identical to the scanned 0.6B tokenizer artifact; enable_thinking=false uses a native template switch",
    },
    {
        "model_id": "gemma3-1b-it-bf16-mirror",
        "family": "Gemma3",
        "hf_repo": "unsloth/gemma-3-1b-it",
        "revision": "5b11413a10db4e486ef16a20101fd028f8f2499c",
        "upstream_repo": "google/gemma-3-1b-it",
        "upstream_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "scan_tokenizer_id": "gemma3-1b-it-mirror",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "chat_template_kwargs": {},
        "extra_eos_tokens": ["<end_of_turn>"],
        "selection_note": "canonical Gemma repository is license-gated for this account; smallest accessible 1B IT mirror checkpoint used, consistent with the existing mirror-tokenizer scan",
    },
]

SKIPPED_MODELS = [
    {
        "requested_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "requested_revision": "9213176726f574b556790deb65791e0c5aa438b6",
        "alternative_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "alternative_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "status": "skipped",
        "reason": "canonical Meta repositories are manual-license-gated and config.json download returned HTTP 403 for this account; user instruction said to note and skip if gated",
    }
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def model_configs() -> list[dict]:
    return copy.deepcopy(MODELS)


def model_config(model_id: str) -> dict:
    matches = [x for x in MODELS if x["model_id"] == model_id]
    if len(matches) != 1:
        raise KeyError(f"unknown model_id {model_id!r}; choices: {[x['model_id'] for x in MODELS]}")
    return copy.deepcopy(matches[0])


def copy_cohort(row: dict) -> str:
    role = row["copy_role"]
    if role == "ascii_control":
        return "ASCII"
    if role == "old_emoji_control":
        return "legacy emoji"
    if role == "old_nonemoji_control":
        return "legacy non-emoji"
    if role == "new_nonemoji":
        return "new non-emoji"
    if role == "new_emoji":
        return row["emoji_version"].replace("E", "E") + " emoji"
    raise ValueError(role)


def prompt_text(row: dict) -> str:
    typ = row["probe_type"]
    if typ == "copy":
        return COPY_TEMPLATE.format(target=row["target"])
    if typ == "count":
        return COUNT_TEMPLATE.format(target=row["target"])
    if typ == "decomposition":
        return DECOMP_TEMPLATE.format(target=row["target"])
    if typ == "comparison":
        return COMPARE_TEMPLATE.format(a=row["a_text"], b=row["b_text"])
    raise ValueError(typ)


def probe_rows(scan_tokenizer_id: str) -> list[dict]:
    rows = []
    for item in read_jsonl(RESULTS / "copy_pilot.jsonl"):
        rows.append({
            "probe_key": "copy:" + item["item_id"], "probe_type": "copy", "item_id": item["item_id"],
            "target": item["text"], "expected_text": item["text"], "copy_role": item["copy_role"],
            "cohort": copy_cohort(item), "item_type": item["item_type"], "item": item,
        })
    structural = read_jsonl(RESULTS / "structural_probes.jsonl")
    for row in structural:
        if row["probe_type"] == "count" and row.get("tokenizer_id") != scan_tokenizer_id:
            continue
        key = row["probe_type"] + ":" + row["probe_id"]
        out = {"probe_key": key, "source_probe": row}
        out.update({k: row[k] for k in row if k not in ("candidate_pool_diagnostics",)})
        rows.append(out)
    expected = 182 + 48 + 36 + 24
    if len(rows) != expected:
        raise RuntimeError(f"{scan_tokenizer_id}: got {len(rows)} behavioral probes, expected {expected}")
    for row in rows:
        row["user_content"] = prompt_text(row)
        row["user_content_sha256"] = sha_text(row["user_content"])
    return rows


def main():
    BEHAVIORAL.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cfg in model_configs():
        rows = probe_rows(cfg["scan_tokenizer_id"])
        for row in rows:
            all_rows.append({"model_id": cfg["model_id"], **{k: v for k, v in row.items() if k not in ("item", "source_probe")}})
        print(cfg["model_id"], len(rows), cfg["scan_tokenizer_id"])
    write_jsonl(BEHAVIORAL / "probe_plan.jsonl", all_rows)
    print("wrote", BEHAVIORAL / "probe_plan.jsonl", len(all_rows))


if __name__ == "__main__":
    main()
