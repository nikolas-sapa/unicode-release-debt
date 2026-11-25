from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm
import transformers
from huggingface_hub import hf_hub_download, snapshot_download
from mlx_lm.sample_utils import greedy_sampler

sys.path.insert(0, str(Path(__file__).parent))
import round2_common as r2

ROUND2 = r2.ROUND2
RAW_PATH = ROUND2 / "round2_raw.jsonl"
RUNTIME_PATH = ROUND2 / "round2_runtime.jsonl"
PROMPT_DIR = ROUND2 / "prompts"


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


def process_rss_gb() -> float:
    ps = shutil.which("ps") or "/bin/ps"
    out = subprocess.check_output([ps, "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
    return int(out) * 1024 / 1e9


def prepare_inference_snapshot(snapshot_path: Path, cfg: dict) -> tuple[Path, dict]:
    overlay = cfg.get("tokenizer_overlay")
    if not overlay:
        return snapshot_path, {"overlay_used": False}
    identity = sha_text("\0".join([str(snapshot_path), overlay["repo"], overlay["revision"], "\x1f".join(overlay["files"])]))[:16]
    prepared = r2.ROOT / "data" / "mlx_round2" / cfg["model_id"] / identity
    prepared.mkdir(parents=True, exist_ok=True)
    overlay_files = set(overlay["files"])
    for source in sorted(x for x in snapshot_path.iterdir() if x.is_file()):
        if source.name in overlay_files:
            continue
        dest = prepared / source.name
        if dest.exists():
            if not dest.is_symlink() or Path(os.readlink(dest)) != source.resolve():
                raise RuntimeError(f"prepared inference path conflict at {dest}")
        else:
            os.symlink(source.resolve(), dest)
    overlay_manifest = []
    for filename in overlay["files"]:
        source = Path(hf_hub_download(overlay["repo"], filename, revision=overlay["revision"]))
        dest = prepared / filename
        data = source.read_bytes()
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f"prepared tokenizer overlay conflict at {dest}")
        if not dest.exists():
            dest.write_bytes(data)
        overlay_manifest.append({"file": filename, "source": str(source), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    metadata = {
        "overlay_used": True,
        "overlay_repo": overlay["repo"],
        "overlay_revision": overlay["revision"],
        "overlay_reason": overlay["reason"],
        "overlay_files": overlay_manifest,
        "prepared_path": str(prepared),
    }
    (prepared / "round2_preparation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return prepared, metadata


def render_prompts(tokenizer, rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["user_content"]}]
        kwargs = {"tokenize": False, "add_generation_prompt": True, **cfg.get("chat_template_kwargs", {})}
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        bos = tokenizer.bos_token
        add_special = bos is None or not rendered.startswith(bos)
        ids = tokenizer.encode(rendered, add_special_tokens=add_special)
        out.append({
            "model_id": cfg["model_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"],
            "copy_protocol": row.get("copy_protocol"), "user_content": row["user_content"], "user_content_sha256": row["user_content_sha256"],
            "messages": messages, "chat_template_kwargs": cfg.get("chat_template_kwargs", {}),
            "rendered_prompt": rendered, "rendered_prompt_sha256": sha_text(rendered),
            "encode_add_special_tokens": add_special, "prompt_token_count": len(ids), "prompt_token_ids": ids,
        })
    return out


def completion_key(model_id: str, probe_key: str, prompt_hash: str, extra_eos_tokens=None, artifact_seed: str = "") -> str:
    eos_part = "\x1f".join(sorted(extra_eos_tokens or []))
    return sha_text("\0".join([model_id, probe_key, prompt_hash, "greedy", str(r2.MAX_TOKENS), eos_part, artifact_seed]))


def chunks(rows: list[dict], n: int):
    for i in range(0, len(rows), n):
        yield i // n, rows[i:i + n]


def runtime_base(cfg: dict, copy_protocol: str) -> dict:
    return {
        "model_id": cfg["model_id"], "copy_protocol": copy_protocol,
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "host_platform": platform.platform(), "host_machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", None), "mlx_lm_version": mlx_lm.__version__, "transformers_version": transformers.__version__,
    }


def run_model(cfg: dict, batch_size: int, limit: int | None, probe_type: str | None, ascii_only: bool, copy_protocol: str, raw_path: Path, runtime_path: Path, prompt_dir: Path):
    ROUND2.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot_download(cfg["hf_repo"], revision=cfg["revision"]))
    load_path, preparation = prepare_inference_snapshot(snapshot_path, cfg)
    rows = r2.probe_rows(cfg["scan_tokenizer_id"], copy_protocol)
    if probe_type:
        rows = [x for x in rows if x["probe_type"] == probe_type]
    if ascii_only:
        rows = [x for x in rows if x["probe_type"] == "copy" and x.get("cohort") == "ASCII"]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no probes selected")

    load_started = utcnow()
    tic = time.perf_counter()
    rss_before_load = process_rss_gb()
    model, tokenizer = mlx_lm.load(str(load_path), lazy=False)
    for token in cfg.get("extra_eos_tokens", []):
        if hasattr(tokenizer, "add_eos_token"):
            tokenizer.add_eos_token(token)
    load_seconds = time.perf_counter() - tic
    rss_after_load = process_rss_gb()
    rendered = render_prompts(tokenizer, rows, cfg)
    prompt_path = prompt_dir / f"{cfg['model_id']}-{copy_protocol}.jsonl"
    r2.write_jsonl(prompt_path, [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in rendered])

    existing = r2.read_jsonl(raw_path)
    completed = {x["completion_key"] for x in existing if x.get("status") == "ok"}
    metadata_rows = file_manifest(snapshot_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    invocation = {
        **runtime_base(cfg, copy_protocol), "event": "model_loaded", "utc": utcnow(), "load_started_utc": load_started,
        "load_seconds": load_seconds, "hf_repo": cfg["hf_repo"], "revision": cfg["revision"],
        "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
        "snapshot_path": str(snapshot_path), "load_path": str(load_path), "preparation": preparation, "file_manifest": metadata_rows,
        "chat_template_sha256": sha_text(chat_template) if isinstance(chat_template, str) else None,
        "eos_token_ids": sorted(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else [tokenizer.eos_token_id],
        "bos_token": tokenizer.bos_token,
        "decode_params": {"sampler": "greedy_sampler/argmax", "temperature": 0.0, "max_tokens": r2.MAX_TOKENS, "repetition_penalty": None, "top_p": None, "top_k": None, "extra_eos_tokens": cfg.get("extra_eos_tokens", [])},
        "quantization": cfg["quantization"], "quantization_class": cfg["quantization_class"], "batch_size": batch_size,
        "selected_probe_rows": len(rows), "ascii_only": ascii_only, "probe_type_filter": probe_type,
        "process_rss_gb_before_load": rss_before_load, "process_rss_gb_after_load": rss_after_load,
    }
    append_jsonl(runtime_path, [invocation])

    artifact_seed = sha_text(json.dumps({
        "hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "tokenizer_overlay": cfg.get("tokenizer_overlay"),
        "chat_template_sha256": invocation["chat_template_sha256"], "eos_token_ids": invocation["eos_token_ids"],
    }, sort_keys=True, separators=(",", ":")))
    generated = 0
    for chunk_index, chunk_rows in chunks(list(zip(rows, rendered)), batch_size):
        keys = [completion_key(cfg["model_id"], row["probe_key"], prompt["rendered_prompt_sha256"], cfg.get("extra_eos_tokens", []), artifact_seed) for row, prompt in chunk_rows]
        if all(key in completed for key in keys):
            continue
        if any(key in completed for key in keys):
            raise RuntimeError(f"partially complete chunk {chunk_index} for {cfg['model_id']}; append-only resume requires manual inspection")
        prompt_ids = [prompt["prompt_token_ids"] for _, prompt in chunk_rows]
        batch_id = f"{cfg['model_id']}-{copy_protocol}-chunk{chunk_index:03d}"
        started = utcnow()
        tic = time.perf_counter()
        rss_before_batch = process_rss_gb()
        response = mlx_lm.batch_generate(
            model, tokenizer, prompt_ids, max_tokens=r2.MAX_TOKENS, return_token_ids=True,
            sampler=greedy_sampler, completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size),
        )
        elapsed = time.perf_counter() - tic
        finished = utcnow()
        rss_after_batch = process_rss_gb()
        if len(response.texts) != len(chunk_rows) or len(response.token_ids) != len(chunk_rows):
            raise RuntimeError(f"batch response length mismatch: {len(response.texts)} texts, {len(response.token_ids or [])} token rows, {len(chunk_rows)} prompts")
        out_rows = []
        for (row, prompt), key, text, token_ids in zip(chunk_rows, keys, response.texts, response.token_ids):
            out_rows.append({
                "status": "ok", "completion_key": key, "batch_id": batch_id, "batch_member_keys": keys, "artifact_seed": artifact_seed,
                "model_id": cfg["model_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"], "copy_protocol": row.get("copy_protocol"),
                "hf_repo": cfg["hf_repo"], "model_revision": cfg["revision"], "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
                "scan_tokenizer_id": cfg["scan_tokenizer_id"], "prompt_sha256": prompt["rendered_prompt_sha256"],
                "user_content_sha256": row["user_content_sha256"], "decode_params": invocation["decode_params"],
                "dtype": cfg["quantization"], "quantization_class": cfg["quantization_class"], "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
                "raw_text": text, "output_token_ids": token_ids, "output_token_count": len(token_ids),
                "finish_reason_inferred": "length" if len(token_ids) == r2.MAX_TOKENS else "stop",
                "batch_started_utc": started, "batch_finished_utc": finished, "batch_elapsed_seconds": elapsed,
                "prompt_token_count": prompt["prompt_token_count"],
            })
        append_jsonl(raw_path, out_rows)
        completed.update(keys)
        generated += len(out_rows)
        stats = response.stats
        append_jsonl(runtime_path, [{
            **runtime_base(cfg, copy_protocol), "event": "batch_completed", "utc": finished, "batch_id": batch_id,
            "probe_count": len(out_rows), "elapsed_seconds": elapsed,
            "prompt_tokens": getattr(stats, "prompt_tokens", None), "generation_tokens": getattr(stats, "generation_tokens", None),
            "prompt_tps": getattr(stats, "prompt_tps", None), "generation_tps": getattr(stats, "generation_tps", None),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "process_rss_gb_before_batch": rss_before_batch, "process_rss_gb_after_batch": rss_after_batch,
        }])
        print(json.dumps({"model": cfg["model_id"], "batch": batch_id, "rows": len(out_rows), "seconds": round(elapsed, 2), "generated_so_far": generated, "rss_gb": round(rss_after_batch, 2)}), flush=True)
    mx.clear_cache()
    append_jsonl(runtime_path, [{**runtime_base(cfg, copy_protocol), "event": "model_run_finished", "utc": utcnow(), "process_rss_gb_before_exit": process_rss_gb(), "generated_rows_this_invocation": generated}])
    return {"model_id": cfg["model_id"], "generated": generated, "probes": len(rows), "load_seconds": load_seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[x["model_id"] for x in r2.all_known_configs()])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-type", choices=["copy", "decomposition", "count", "comparison"])
    parser.add_argument("--ascii-only", action="store_true")
    parser.add_argument("--copy-protocol", default="fewshot_copy_v1", choices=sorted(r2.COPY_PROTOCOLS))
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--runtime-path", type=Path, default=RUNTIME_PATH)
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    args = parser.parse_args()
    cfg = r2.model_config(args.model)
    result = run_model(cfg, args.batch_size or cfg.get("default_batch_size", r2.BATCH_SIZE), args.limit, args.probe_type, args.ascii_only, args.copy_protocol, args.raw_path, args.runtime_path, args.prompt_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
