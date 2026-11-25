from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
ROUND2 = RESULTS / "behavioral_round2"
MAX_TOKENS = 64
BATCH_SIZE = 16

COPY_INSTRUCTION_TEMPLATE = bp.COPY_TEMPLATE
FEWSHOT_EXEMPLARS = [
    "bO5*Fe%Yx^9pM=5g",
    ";@N@fd=K\\+w=kEJ3",
    "mX3 *vQ9!zB 2#7R",
    "p~4tD14z/4g;[zM!",
]
COPY_FEWSHOT_TEMPLATE = """Copy each input exactly.
Input: <IN>bO5*Fe%Yx^9pM=5g</IN>
Output: <OUT>bO5*Fe%Yx^9pM=5g</OUT>
Input: <IN>;@N@fd=K\\+w=kEJ3</IN>
Output: <OUT>;@N@fd=K\\+w=kEJ3</OUT>
Input: <IN>mX3 *vQ9!zB 2#7R</IN>
Output: <OUT>mX3 *vQ9!zB 2#7R</OUT>
Input: <IN>p~4tD14z/4g;[zM!</IN>
Output: <OUT>p~4tD14z/4g;[zM!</OUT>
Input: <IN>{target}</IN>
Output: """

COPY_FEWSHOT_FINALCUE_TEMPLATE = """Copy each input exactly.
Input: <IN>bO5*Fe%Yx^9pM=5g</IN>
Output: <OUT>bO5*Fe%Yx^9pM=5g</OUT>
Input: <IN>;@N@fd=K\\+w=kEJ3</IN>
Output: <OUT>;@N@fd=K\\+w=kEJ3</OUT>
Input: <IN>mX3 *vQ9!zB 2#7R</IN>
Output: <OUT>mX3 *vQ9!zB 2#7R</OUT>
Input: <IN>p~4tD14z/4g;[zM!</IN>
Output: <OUT>p~4tD14z/4g;[zM!</OUT>
Copy this input exactly.
Input: <IN>{target}</IN>
Output: """

COPY_MULTITURN_USER_TEMPLATE = "Copy the input exactly. Input: <IN>{target}</IN>\nOutput: "
COPY_MULTITURN_CUELESS_USER_TEMPLATE = "Copy the input exactly. Input: <IN>{target}</IN>"
COPY_MULTITURN_MINIMAL_USER_TEMPLATE = "<IN>{target}</IN>"

COPY_PROTOCOLS = {
    "instruction_v1": {
        "template": COPY_INSTRUCTION_TEMPLATE,
        "message_style": "single_user",
        "description": "round-1 instruction-only copy prompt, retained only for ASCII-control calibration",
    },
    "fewshot_copy_v1": {
        "template": COPY_FEWSHOT_TEMPLATE[:-1],
        "message_style": "single_user",
        "description": "four held-out ASCII filler copy exemplars with inline input/output wrappers, followed by the probe; superseded ASCII calibration candidate without trailing space",
    },
    "fewshot_copy_v2": {
        "template": COPY_FEWSHOT_TEMPLATE,
        "message_style": "single_user",
        "description": "four held-out ASCII filler copy exemplars with inline input/output wrappers, followed by the probe; final output cue includes the same trailing space as exemplars",
    },
    "fewshot_copy_v3_finalcue": {
        "template": COPY_FEWSHOT_FINALCUE_TEMPLATE,
        "message_style": "single_user",
        "description": "four held-out ASCII filler copy exemplars plus an explicit final-copy cue and trailing output space",
    },
    "fewshot_copy_v4_multiturn": {
        "template": COPY_MULTITURN_USER_TEMPLATE,
        "message_style": "assistant_exemplar_pairs",
        "description": "four held-out ASCII filler copy exemplars as alternating user/assistant chat pairs, followed by the probe user turn with an output cue",
    },
    "fewshot_copy_v5_multiturn_cueless": {
        "template": COPY_MULTITURN_CUELESS_USER_TEMPLATE,
        "message_style": "assistant_exemplar_pairs",
        "description": "four held-out ASCII filler copy exemplars as alternating user/assistant chat pairs; probe user turn ends at the input and the chat template supplies the assistant boundary",
    },
    "fewshot_copy_v6_multiturn_minimal": {
        "template": COPY_MULTITURN_MINIMAL_USER_TEMPLATE,
        "message_style": "assistant_exemplar_pairs",
        "description": "four held-out ASCII filler copy exemplars as minimal alternating input/output chat pairs with no prose instruction",
    },
}

MODELS = [
    {
        "model_id": "qwen2.5-3b-instruct-bf16",
        "family": "Qwen2.5",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "upstream_repo": "Qwen/Qwen2.5-3B-Instruct",
        "upstream_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "scan_tokenizer_id": "qwen2.5-0.5b-instruct",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "quantization_class": "bf16",
        "chat_template_kwargs": {},
        "default_batch_size": 16,
        "priority": 1,
        "selection_note": "canonical 3B instruct checkpoint; bf16 fits the 16GB host; tokenizer files byte-identical to scanned Qwen2.5 artifact",
    },
    {
        "model_id": "qwen3-4b-instruct-2507-mlx8",
        "family": "Qwen3",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "mlx-community/Qwen3-4B-Instruct-2507-8bit",
        "revision": "0e42af58449718de7931ee04f28191fbe6c43a56",
        "upstream_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "upstream_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "scan_tokenizer_id": "qwen3-0.6b",
        "quantization": "MLX 8-bit affine quantization (bits=8, group_size=64) derived from canonical bf16",
        "quantization_class": "mlx8",
        "chat_template_kwargs": {},
        "tokenizer_overlay": {
            "repo": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            "reason": "mlx-community 8-bit artifact omits the canonical chat template; tokenizer files are overlaid byte-for-byte from the pinned canonical upstream while quantized weights remain the pinned MLX artifact",
        },
        "default_batch_size": 16,
        "priority": 2,
        "selection_note": "4B 2507 instruct checkpoint; q8 chosen because bf16 is tight on a 16GB host; canonical tokenizer/chat-template overlay; tokenizer core verified against scanned Qwen3 artifact",
    },
    {
        "model_id": "qwen2.5-7b-instruct-mlx8",
        "family": "Qwen2.5",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        "revision": "df11cb390bd674c24a9ddedf19f05bc09f6e0148",
        "upstream_repo": "Qwen/Qwen2.5-7B-Instruct",
        "upstream_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "scan_tokenizer_id": "qwen2.5-0.5b-instruct",
        "quantization": "MLX 8-bit affine quantization (bits=8, group_size=64) derived from canonical bf16",
        "quantization_class": "mlx8",
        "chat_template_kwargs": {},
        "tokenizer_overlay": {
            "repo": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            "reason": "mlx-community Qwen2.5 8-bit artifacts package a non-byte-identical Qwen3-like tokenizer; canonical tokenizer files are overlaid byte-for-byte so quantization, not tokenizer substitution, is the measurement contrast",
        },
        "default_batch_size": 8,
        "priority": 3,
        "selection_note": "7B instruct checkpoint attempted in q8 under the 16GB memory constraint; canonical upstream recorded; canonical tokenizer overlay byte-identical to scanned Qwen2.5 artifact",
    },
    {
        "model_id": "gemma3-4b-it-mlx8-mirror",
        "family": "Gemma3",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "mlx-community/gemma-3-4b-it-8bit",
        "revision": "61cfff4de4bc5b975437be840d504f07f0424df2",
        "upstream_repo": "google/gemma-3-4b-it",
        "upstream_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "scan_tokenizer_id": "gemma3-1b-it-mirror",
        "quantization": "MLX 8-bit affine quantization (bits=8, group_size=64) derived from google/gemma-3-4b-it",
        "quantization_class": "mlx8",
        "chat_template_kwargs": {},
        "extra_eos_tokens": ["<end_of_turn>"],
        "default_batch_size": 8,
        "priority": 4,
        "selection_note": "canonical google repository is license-gated for this account; MLX community q8 inference artifact used with canonical upstream recorded; unsloth mirror revision bf46152c47f5dd20b896357cb51abc4c03b8ee8c documents the accessible bf16 mirror lineage; tokenizer core matches scanned Gemma artifact",
    },
    {
        "model_id": "tinyllama-1.1b-chat-v1.0-bf16",
        "family": "Llama2-vintage",
        "vintage_group": "vintage_llama2_tokenizer",
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "upstream_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "upstream_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "scan_tokenizer_id": "tinyllama-1.1b-chat",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "quantization_class": "bf16",
        "chat_template_kwargs": {},
        "default_batch_size": 16,
        "priority": 5,
        "selection_note": "mandatory Llama-2-tokenizer vintage carrier; canonical unquantized checkpoint",
    },
]

QUANT_CHECK_MODELS = [
    {
        "model_id": "quantcheck-qwen2.5-1.5b-instruct-bf16",
        "family": "Qwen2.5",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "upstream_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "upstream_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "scan_tokenizer_id": "qwen2.5-0.5b-instruct",
        "quantization": "unquantized bfloat16 safetensors (config torch_dtype=bfloat16)",
        "quantization_class": "bf16",
        "chat_template_kwargs": {},
        "default_batch_size": 24,
        "priority": 0,
        "selection_note": "quantization-sensitivity control only; not part of the round-2 model slate",
    },
    {
        "model_id": "quantcheck-qwen2.5-1.5b-instruct-mlx8",
        "family": "Qwen2.5",
        "vintage_group": "newer_tokenizer",
        "hf_repo": "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        "revision": "c7e1ff25efda4a2c8f7ee312380557fb42e145a2",
        "upstream_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "upstream_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "scan_tokenizer_id": "qwen2.5-0.5b-instruct",
        "quantization": "MLX 8-bit affine quantization (bits=8, group_size=64) derived from canonical bf16",
        "quantization_class": "mlx8",
        "chat_template_kwargs": {},
        "tokenizer_overlay": {
            "repo": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "files": ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            "reason": "mlx-community Qwen2.5 8-bit artifacts package a non-byte-identical Qwen3-like tokenizer; canonical tokenizer files are overlaid byte-for-byte so quantization, not tokenizer substitution, is the measurement contrast",
        },
        "default_batch_size": 24,
        "priority": 0,
        "selection_note": "quantization-sensitivity control only; not part of the round-2 model slate",
    },
]

SKIPPED_MODELS = [
    {
        "requested_repo": "meta-llama/Llama-2-7b-chat-hf",
        "requested_revision": "f5db02db724555f92da89c216ac04704f23d4590",
        "status": "skipped",
        "reason": "canonical config.json download returned HTTP 403 GatedRepoError for this account; NousResearch/Llama-2-7b-chat-hf is accessible at 351844e75ed0bcbbe3f10671b3c808d2b83894ee and has tokenizer files byte-identical to the scanned llama2 mirror, but neither its model card nor available repository metadata documents weight-hash equivalence to canonical Meta weights, so the round-2 instructions permit TinyLlama alone as the vintage carrier",
    }
]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def model_configs(include_quant_check: bool = False) -> list[dict]:
    rows = copy.deepcopy(MODELS)
    if include_quant_check:
        rows += copy.deepcopy(QUANT_CHECK_MODELS)
    return rows


def all_known_configs() -> list[dict]:
    return model_configs(include_quant_check=True)


def model_config(model_id: str) -> dict:
    matches = [x for x in all_known_configs() if x["model_id"] == model_id]
    if len(matches) != 1:
        raise KeyError(f"unknown model_id {model_id!r}; choices: {[x['model_id'] for x in all_known_configs()]}")
    return copy.deepcopy(matches[0])


def copy_prompt(target: str, protocol: str) -> str:
    if protocol not in COPY_PROTOCOLS:
        raise KeyError(f"unknown copy protocol {protocol!r}")
    return COPY_PROTOCOLS[protocol]["template"].format(target=target)


def copy_messages(target: str, protocol: str) -> list[dict]:
    meta = COPY_PROTOCOLS[protocol]
    if meta.get("message_style") == "assistant_exemplar_pairs":
        messages = []
        for exemplar in FEWSHOT_EXEMPLARS:
            messages.append({"role": "user", "content": copy_prompt(exemplar, protocol)})
            messages.append({"role": "assistant", "content": f"<OUT>{exemplar}</OUT>"})
        messages.append({"role": "user", "content": copy_prompt(target, protocol)})
        return messages
    return [{"role": "user", "content": copy_prompt(target, protocol)}]


def copy_user_content_repr(target: str, protocol: str) -> str:
    messages = copy_messages(target, protocol)
    if COPY_PROTOCOLS[protocol].get("message_style") == "assistant_exemplar_pairs":
        return canon(messages)
    return messages[0]["content"]


def probe_rows(scan_tokenizer_id: str, copy_protocol: str) -> list[dict]:
    rows = bp.probe_rows(scan_tokenizer_id)
    for row in rows:
        if row["probe_type"] == "copy":
            row["copy_protocol"] = copy_protocol
            row["messages"] = copy_messages(row["target"], copy_protocol)
            row["user_content"] = copy_user_content_repr(row["target"], copy_protocol)
            row["user_content_sha256"] = sha_text(row["user_content"])
        else:
            row["copy_protocol"] = None
            row["messages"] = [{"role": "user", "content": row["user_content"]}]
    return rows


def item_identity(row: dict) -> dict:
    keys = [
        "probe_key", "probe_type", "item_id", "target", "expected_text", "copy_role", "cohort", "item_type",
        "era", "fragmentation_group", "pair_id", "probe_id", "expected_count", "expected_codepoints",
        "a_text", "b_text", "expected_label", "requested_label", "constituent_ids", "tokenizer_id", "utf8_bytes",
    ]
    return {k: row[k] for k in keys if k in row}


def item_identity_sha256(rows: list[dict]) -> str:
    return sha_text(canon([item_identity(x) for x in rows]))


def exemplar_overlap_audit() -> dict:
    copy_rows = [json.loads(line) for line in (RESULTS / "copy_pilot.jsonl").read_text().splitlines() if line.strip()]
    targets = {x["text"] for x in copy_rows}
    exemplars = list(FEWSHOT_EXEMPLARS)
    return {
        "exemplars": exemplars,
        "count": len(exemplars),
        "unique": len(set(exemplars)) == len(exemplars),
        "exact_overlap_with_any_copy_probe": sorted(set(exemplars) & targets),
        "all_ascii": all(all(ord(c) < 128 for c in x) for x in exemplars),
        "no_wrapper_literals": all(not any(tok in x for tok in ("<IN>", "</IN>", "<OUT>", "</OUT>")) for x in exemplars),
    }


def protocol_format_fingerprint(protocol: str) -> str:
    meta = COPY_PROTOCOLS[protocol]
    obj = {
        "message_style": meta.get("message_style", "single_user"),
        "user_template": meta["template"],
        "exemplars": FEWSHOT_EXEMPLARS if meta.get("message_style") == "assistant_exemplar_pairs" or protocol.startswith("fewshot") else [],
        "assistant_output_template": "<OUT>{exemplar}</OUT>" if meta.get("message_style") == "assistant_exemplar_pairs" else None,
        "probe_position": "after_all_exemplars",
    }
    return sha_text(canon(obj))


def protocol_template_metadata() -> dict:
    return {
        protocol: {
            **meta,
            "sha256": sha_text(meta["template"]),
            "format_sha256": protocol_format_fingerprint(protocol),
        }
        for protocol, meta in COPY_PROTOCOLS.items()
    }
