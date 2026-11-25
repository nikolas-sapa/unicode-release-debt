from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm
import transformers
from huggingface_hub import snapshot_download
from mlx_lm.sample_utils import greedy_sampler

sys.path.insert(0, str(Path(__file__).parent))
import behavioral_probes as bp
import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
BEHAVIORAL = bp.BEHAVIORAL
RAW_PATH = BEHAVIORAL / "behavioral_raw.jsonl"
RUNTIME_PATH = BEHAVIORAL / "behavioral_runtime.jsonl"
PROMPT_DIR = BEHAVIORAL / "prompts"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()


def file_manifest(snapshot: Path) -> list[dict]:
    rows = []
    for path in sorted(x for x in snapshot.iterdir() if x.is_file()):
        row = {"file": path.name, "size": path.stat().st_size}
        if path.stat().st_size <= 50_000_000:
            row["sha256"] = sha_file(path)
        else:
            row["sha256"] = None
            row["hash_note"] = "large weight file not hashed; immutable pinned repository revision is the artifact identity"
        rows.append(row)
    return rows


def render_prompts(tokenizer, rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for row in rows:
        messages = [{"role": "user", "content": row["user_content"]}]
        kwargs = {"tokenize": False, "add_generation_prompt": True, **cfg.get("chat_template_kwargs", {})}
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        bos = tokenizer.bos_token
        add_special = bos is None or not rendered.startswith(bos)
        ids = tokenizer.encode(rendered, add_special_tokens=add_special)
        out.append({
            "model_id": cfg["model_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"],
            "user_content": row["user_content"], "user_content_sha256": row["user_content_sha256"],
            "messages": messages, "chat_template_kwargs": cfg.get("chat_template_kwargs", {}),
            "rendered_prompt": rendered, "rendered_prompt_sha256": sha_text(rendered),
            "encode_add_special_tokens": add_special, "prompt_token_count": len(ids), "prompt_token_ids": ids,
        })
    return out


def completion_key(model_id: str, probe_key: str, prompt_hash: str, extra_eos_tokens=None) -> str:
    eos_part = "\x1f".join(sorted(extra_eos_tokens or []))
    return sha_text("\0".join([model_id, probe_key, prompt_hash, "greedy", str(bp.MAX_TOKENS), eos_part]))


def chunks(rows: list[dict], n: int):
    for i in range(0, len(rows), n):
        yield i // n, rows[i:i + n]


def runtime_base(cfg: dict) -> dict:
    return {
        "model_id": cfg["model_id"], "host_python_version": platform.python_version(),
        "host_unicodedata_version": unicodedata.unidata_version,
        "host_platform": platform.platform(), "host_machine": platform.machine(),
        "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__, "transformers_version": transformers.__version__,
    }


def prepare_mlx_snapshot(snapshot_path: Path, model_id: str) -> tuple[Path, str | None]:
    if list(snapshot_path.glob("model*.safetensors")):
        return snapshot_path, None
    weights = sorted(snapshot_path.glob("weights*.safetensors"))
    if len(weights) != 1:
        return snapshot_path, None
    prepared = ROOT / "data" / "mlx_local" / model_id
    prepared.mkdir(parents=True, exist_ok=True)
    for source in snapshot_path.iterdir():
        if not source.is_file():
            continue
        dest = prepared / source.name
        if not dest.exists():
            os.symlink(source.resolve(), dest)
    alias = prepared / "model.safetensors"
    if not alias.exists():
        os.symlink((prepared / weights[0].name).resolve(), alias)
    return prepared, f"created local symlink alias model.safetensors for legacy MLX weight name {weights[0].name}; immutable HF snapshot unchanged"


def run_model(cfg: dict, batch_size: int, limit: int | None, probe_type: str | None, raw_path: Path = RAW_PATH, runtime_path: Path = RUNTIME_PATH):
    BEHAVIORAL.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot_download(cfg["hf_repo"], revision=cfg["revision"]))
    load_path, alias_note = prepare_mlx_snapshot(snapshot_path, cfg["model_id"])
    load_started = utcnow()
    tic = time.perf_counter()
    model, tokenizer = mlx_lm.load(str(load_path), lazy=False)
    for token in cfg.get("extra_eos_tokens", []):
        tokenizer.add_eos_token(token)
    load_seconds = time.perf_counter() - tic
    rows = bp.probe_rows(cfg["scan_tokenizer_id"])
    if probe_type:
        rows = [x for x in rows if x["probe_type"] == probe_type]
    if limit is not None:
        rows = rows[:limit]
    rendered = render_prompts(tokenizer, rows, cfg)
    prompt_path = PROMPT_DIR / f"{cfg['model_id']}.jsonl"
    bp.write_jsonl(prompt_path, [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in rendered])

    existing = read_jsonl(raw_path)
    completed = {x["completion_key"] for x in existing if x.get("status") == "ok"}
    metadata_rows = file_manifest(snapshot_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    invocation = {
        **runtime_base(cfg), "event": "model_loaded", "utc": utcnow(), "load_started_utc": load_started,
        "load_seconds": load_seconds, "hf_repo": cfg["hf_repo"], "revision": cfg["revision"],
        "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
        "snapshot_path": str(snapshot_path), "load_path": str(load_path), "weight_alias_note": alias_note, "file_manifest": metadata_rows,
        "chat_template_sha256": sha_text(chat_template) if isinstance(chat_template, str) else None,
        "eos_token_ids": sorted(tokenizer.eos_token_ids), "bos_token": tokenizer.bos_token,
        "decode_params": {"sampler": "greedy_sampler/argmax", "temperature": 0.0, "max_tokens": bp.MAX_TOKENS, "repetition_penalty": None, "top_p": None, "top_k": None, "extra_eos_tokens": cfg.get("extra_eos_tokens", [])},
        "quantization": cfg["quantization"], "batch_size": batch_size,
    }
    append_jsonl(runtime_path, [invocation])

    generated = 0
    for chunk_index, chunk_rows in chunks(list(zip(rows, rendered)), batch_size):
        keys = [completion_key(cfg["model_id"], row["probe_key"], prompt["rendered_prompt_sha256"], cfg.get("extra_eos_tokens", [])) for row, prompt in chunk_rows]
        if all(key in completed for key in keys):
            continue
        if any(key in completed for key in keys):
            raise RuntimeError(f"partially complete chunk {chunk_index} for {cfg['model_id']}; append-only resume requires manual inspection")
        prompt_ids = [prompt["prompt_token_ids"] for _, prompt in chunk_rows]
        batch_id = f"{cfg['model_id']}-chunk{chunk_index:03d}"
        started = utcnow()
        tic = time.perf_counter()
        response = mlx_lm.batch_generate(
            model, tokenizer, prompt_ids, max_tokens=bp.MAX_TOKENS, return_token_ids=True,
            sampler=greedy_sampler, completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size),
        )
        elapsed = time.perf_counter() - tic
        finished = utcnow()
        if len(response.texts) != len(chunk_rows) or len(response.token_ids) != len(chunk_rows):
            raise RuntimeError(f"batch response length mismatch: {len(response.texts)} texts, {len(response.token_ids or [])} token rows, {len(chunk_rows)} prompts")
        out_rows = []
        for (row, prompt), key, text, token_ids in zip(chunk_rows, keys, response.texts, response.token_ids):
            out_rows.append({
                "status": "ok", "completion_key": key, "batch_id": batch_id, "batch_member_keys": keys,
                "model_id": cfg["model_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"],
                "hf_repo": cfg["hf_repo"], "model_revision": cfg["revision"], "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
                "scan_tokenizer_id": cfg["scan_tokenizer_id"], "prompt_sha256": prompt["rendered_prompt_sha256"],
                "user_content_sha256": row["user_content_sha256"], "decode_params": invocation["decode_params"],
                "dtype": cfg["quantization"], "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
                "raw_text": text, "output_token_ids": token_ids, "output_token_count": len(token_ids),
                "finish_reason_inferred": "length" if len(token_ids) == bp.MAX_TOKENS else "stop",
                "batch_started_utc": started, "batch_finished_utc": finished, "batch_elapsed_seconds": elapsed,
                "prompt_token_count": prompt["prompt_token_count"],
            })
        append_jsonl(raw_path, out_rows)
        completed.update(keys)
        generated += len(out_rows)
        stats = response.stats
        append_jsonl(runtime_path, [{
            **runtime_base(cfg), "event": "batch_completed", "utc": finished, "batch_id": batch_id,
            "probe_count": len(out_rows), "elapsed_seconds": elapsed,
            "prompt_tokens": getattr(stats, "prompt_tokens", None), "generation_tokens": getattr(stats, "generation_tokens", None),
            "prompt_tps": getattr(stats, "prompt_tps", None), "generation_tps": getattr(stats, "generation_tps", None),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        }])
        print(json.dumps({"model": cfg["model_id"], "batch": batch_id, "rows": len(out_rows), "seconds": round(elapsed, 2), "generated_so_far": generated}), flush=True)
    mx.clear_cache()
    return {"model_id": cfg["model_id"], "generated": generated, "probes": len(rows), "load_seconds": load_seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[x["model_id"] for x in bp.MODELS])
    parser.add_argument("--batch-size", type=int, default=bp.BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-type", choices=["copy", "decomposition", "count", "comparison"])
    parser.add_argument("--smoke", action="store_true", help="write to smoke raw/runtime files instead of pilot append-only files")
    args = parser.parse_args()
    raw_path = BEHAVIORAL / "behavioral_smoke_raw.jsonl" if args.smoke else RAW_PATH
    runtime_path = BEHAVIORAL / "behavioral_smoke_runtime.jsonl" if args.smoke else RUNTIME_PATH
    result = run_model(bp.model_config(args.model), args.batch_size, args.limit, args.probe_type, raw_path, runtime_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
