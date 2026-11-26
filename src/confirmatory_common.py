from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT = ucd.ROOT
DATA = ucd.DATA
RESULTS = ucd.RESULTS
CONFIRMATORY = RESULTS / "confirmatory"
SCAN_DIR = DATA / "confirmatory_scan"
TOKENIZER_DIR = DATA / "confirmatory_tokenizers"
SEED = 20261126
H1_BOOTSTRAP_SEED = 20261127
H2_BOOTSTRAP_SEED = 20261128
MAX_TOKENS = 64
BATCH_SIZE = 16

COPY_USER_TEMPLATE = "Copy the input exactly. Input: <IN>{target}</IN>"
COUNT_TEMPLATE = "Count the extended grapheme clusters (user-perceived characters), not bytes or code points. Reply with one integer only. ZWJ joins emoji components into one cluster; combining marks and variation selectors attach to their base character. Input:\n{target}"
SCALAR_TEMPLATE = "List the Unicode scalar values in order as hexadecimal codepoints. Use exactly this format: `U+1F600 U+200D`. Input:\n{target}"
FEWSHOT_EXEMPLARS = [
    "bO5*Fe%Yx^9pM=5g",
    ";@N@fd=K\\+w=kEJ3",
    "mX3 *vQ9!zB 2#7R",
    "p~4tD14z/4g;[zM!",
]
ITEM_TYPE_CYCLE = ["complex_nonemoji", "zwj_emoji", "atomic_nonemoji", "atomic_emoji"]
LEVELS = {
    "L1": (1.0, 2.5),
    "L2": (2.5, 4.5),
    "L3": (4.5, 6.5),
    "L4": (6.5, float("inf")),
}
NEWEST_UNICODE_VERSIONS = {"15.1", "16.0", "17.0", "18.0"}
NEWEST_EMOJI_VERSIONS = {"E15.1", "E16.0", "E17.0", "E18.0"}

CARRIERS = [
    {
        "carrier_id": "tinyllama-1.1b-chat-v1.0-bf16",
        "execution_order": 1,
        "primary_ladder": True,
        "family": "Llama2-vintage",
        "tokenizer_family": "llama2_sentencepiece",
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "upstream_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "upstream_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "quantization_class": "bf16",
        "tokenizer_source_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tokenizer_source_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "tokenizer_files": ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"],
        "default_batch_size": 16,
        "chat_template_kwargs": {},
    },
    {
        "carrier_id": "qwen2.5-3b-instruct-bf16",
        "execution_order": 2,
        "primary_ladder": True,
        "family": "Qwen2.5",
        "tokenizer_family": "qwen_byte_bpe",
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "upstream_repo": "Qwen/Qwen2.5-3B-Instruct",
        "upstream_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "quantization_class": "bf16",
        "tokenizer_source_repo": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_source_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "tokenizer_files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        "default_batch_size": 16,
        "chat_template_kwargs": {},
    },
    {
        "carrier_id": "llama32-3b-instruct-mlx8",
        "execution_order": 3,
        "primary_ladder": True,
        "family": "Llama3.2",
        "tokenizer_family": "llama3_byte_bpe",
        "hf_repo": "mlx-community/Llama-3.2-3B-Instruct-8bit",
        "revision": "ff054899609078569493def2823f9acd2780c0c9",
        "upstream_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "upstream_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "quantization": "MLX 8-bit affine quantization derived from canonical bf16",
        "quantization_class": "mlx8",
        "tokenizer_source_repo": "mlx-community/Llama-3.2-3B-Instruct-8bit",
        "tokenizer_source_revision": "ff054899609078569493def2823f9acd2780c0c9",
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "expected_tokenizer_sha256": {
            "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
            "tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
            "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
        },
        "default_batch_size": 16,
        "chat_template_kwargs": {},
    },
    {
        "carrier_id": "gemma3-4b-it-mlx8-mirror",
        "execution_order": 4,
        "primary_ladder": True,
        "family": "Gemma3",
        "tokenizer_family": "gemma_sentencepiece",
        "hf_repo": "mlx-community/gemma-3-4b-it-8bit",
        "revision": "61cfff4de4bc5b975437be840d504f07f0424df2",
        "upstream_repo": "google/gemma-3-4b-it",
        "upstream_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "quantization": "MLX 8-bit affine quantization derived from google/gemma-3-4b-it",
        "quantization_class": "mlx8",
        "tokenizer_source_repo": "mlx-community/gemma-3-4b-it-8bit",
        "tokenizer_source_revision": "61cfff4de4bc5b975437be840d504f07f0424df2",
        "tokenizer_files": ["added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
        "extra_eos_tokens": ["<end_of_turn>"],
        "default_batch_size": 8,
        "chat_template_kwargs": {},
    },
    {
        "carrier_id": "qwen3-4b-instruct-2507-mlx8",
        "execution_order": 5,
        "primary_ladder": True,
        "family": "Qwen3",
        "tokenizer_family": "qwen_byte_bpe",
        "hf_repo": "mlx-community/Qwen3-4B-Instruct-2507-8bit",
        "revision": "0e42af58449718de7931ee04f28191fbe6c43a56",
        "upstream_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "upstream_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "quantization": "MLX 8-bit affine quantization derived from canonical bf16",
        "quantization_class": "mlx8",
        "tokenizer_source_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_source_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "tokenizer_files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        "tokenizer_overlay": {
            "repo": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            "reason": "canonical tokenizer/chat-template overlay required by the round-2 frozen measurement instrument",
        },
        "default_batch_size": 16,
        "chat_template_kwargs": {},
    },
    {
        "carrier_id": "mistral7b-instruct-v03-mlx8",
        "execution_order": 6,
        "primary_ladder": False,
        "descriptive_robustness_only": True,
        "family": "Mistral0.3",
        "tokenizer_family": "mistral_sentencepiece",
        "hf_repo": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
        "revision": "c282bcd11a64be44ac3f130f0b819d48ecc1846d",
        "upstream_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "upstream_revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "quantization": "MLX 8-bit affine quantization derived from canonical bf16",
        "quantization_class": "mlx8",
        "tokenizer_source_repo": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
        "tokenizer_source_revision": "c282bcd11a64be44ac3f130f0b819d48ecc1846d",
        "tokenizer_files": ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"],
        "expected_tokenizer_sha256": {
            "tokenizer.json": "ad379250e3a923686183c1ee60398c6f278da0a40520dc4b22e69c53cfc9af9e",
            "tokenizer.model": "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
            "tokenizer_config.json": "489746d6758f2e62d01eb30b7b6486e60ebae4728a97cbce4905a2eb7264f5fd",
            "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
        },
        "default_batch_size": 4,
        "chat_template_kwargs": {},
    },
]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def carrier_configs(include_descriptive=True) -> list[dict]:
    rows = copy.deepcopy(CARRIERS)
    if not include_descriptive:
        rows = [x for x in rows if x["primary_ladder"]]
    return sorted(rows, key=lambda x: x["execution_order"])


def carrier_config(carrier_id: str) -> dict:
    matches = [x for x in CARRIERS if x["carrier_id"] == carrier_id]
    if len(matches) != 1:
        raise KeyError(f"unknown carrier {carrier_id!r}")
    return copy.deepcopy(matches[0])


def copy_user_content(target: str) -> str:
    return COPY_USER_TEMPLATE.format(target=target)


def copy_messages(target: str) -> list[dict]:
    messages = []
    for exemplar in FEWSHOT_EXEMPLARS:
        messages.append({"role": "user", "content": copy_user_content(exemplar)})
        messages.append({"role": "assistant", "content": f"<OUT>{exemplar}</OUT>"})
    messages.append({"role": "user", "content": copy_user_content(target)})
    return messages


def count_user_content(target: str) -> str:
    return COUNT_TEMPLATE.format(target=target)


def scalar_user_content(target: str) -> str:
    return SCALAR_TEMPLATE.format(target=target)


def grapheme_count(text: str) -> int:
    return len(regex.findall(r"\X", text))


def fragmentation_level(tokens_per_cluster: float) -> str | None:
    for level, (lo, hi) in LEVELS.items():
        if lo <= tokens_per_cluster < hi:
            return level
    return None


def scalar_text(text: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in text)


def ascii_count_controls() -> list[dict]:
    rng = random.Random(SEED)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'()*+,-./:;=?@[]^_`{|}~"
    rows, seen = [], set()
    for i in range(12):
        n = (i % 5) + 1
        while True:
            text = "".join(rng.sample(alphabet, n))
            if text not in seen and not any(x in text for x in ("<IN>", "</IN>", "<OUT>", "</OUT>")):
                seen.add(text)
                break
        rows.append({"target": text, "expected_count": n, "control_slot": i, "target_sha256": sha_text(text)})
    return rows


def ascii_scalar_controls() -> list[dict]:
    rng = random.Random(SEED + 1)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'()*+,-./:;=?@[]^_`{|}~"
    rows, seen = [], set()
    for i in range(12):
        n = (i % 3) + 1
        while True:
            text = "".join(rng.sample(alphabet, n))
            if text not in seen and not any(x in text for x in ("<IN>", "</IN>", "<OUT>", "</OUT>")):
                seen.add(text)
                break
        rows.append({"target": text, "expected_scalars": n, "expected_text": scalar_text(text), "control_slot": i, "target_sha256": sha_text(text)})
    return rows


def round2_ascii_copy_controls() -> list[dict]:
    rows = [x for x in read_jsonl(RESULTS / "copy_pilot.jsonl") if x.get("copy_role") == "ascii_control"]
    if len(rows) != 24:
        raise RuntimeError(f"round-2 ASCII copy control count is {len(rows)}, expected 24")
    return [{"target": x["text"], "control_slot": i, "round2_item_id": x["item_id"], "target_sha256": sha_text(x["text"])} for i, x in enumerate(rows)]


def prompt_messages(row: dict) -> list[dict]:
    if row["probe_type"] == "copy":
        return copy_messages(row["target"])
    if row["probe_type"] == "count":
        return [{"role": "user", "content": count_user_content(row["target"])}]
    if row["probe_type"] == "scalar_listing":
        return [{"role": "user", "content": scalar_user_content(row["target"])}]
    raise ValueError(row["probe_type"])


def user_content_for_probe(row: dict) -> str:
    if row["probe_type"] == "copy":
        return canon(copy_messages(row["target"]))
    if row["probe_type"] == "count":
        return count_user_content(row["target"])
    if row["probe_type"] == "scalar_listing":
        return scalar_user_content(row["target"])
    raise ValueError(row["probe_type"])


def level_sort_key(level: str) -> int:
    return int(level[1])
