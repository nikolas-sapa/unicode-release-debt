from __future__ import annotations

import argparse
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
import confirmatory_common as cf

RAW_PATH = cf.CONFIRMATORY / "confirmatory_raw.jsonl"
RUNTIME_PATH = cf.CONFIRMATORY / "confirmatory_runtime.jsonl"
SMOKE_PATH = cf.CONFIRMATORY / "memory_smoke.jsonl"
PROMPT_DIR = cf.CONFIRMATORY / "rendered_prompts"
RESUME_PATH = cf.CONFIRMATORY / "RESUME_STATE.md"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_rss_gb() -> float:
    ps = shutil.which("ps") or "/bin/ps"
    out = subprocess.check_output([ps, "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
    return int(out) * 1024 / 1e9


def sha_file(path: Path) -> str:
    return cf.sha_file(path)


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


def prepare_inference_snapshot(snapshot_path: Path, cfg: dict) -> tuple[Path, dict]:
    overlay = cfg.get("tokenizer_overlay")
    if not overlay:
        return snapshot_path, {"overlay_used": False}
    identity = cf.sha_text("\0".join([str(snapshot_path), overlay["repo"], overlay["revision"], "\x1f".join(overlay["files"])]))[:16]
    prepared = cf.ROOT / "data" / "mlx_confirmatory" / cfg["carrier_id"] / identity
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
        overlay_manifest.append({"file": filename, "source": str(source), "sha256": cf.sha_text(data.decode(errors="surrogateescape")) if False else __import__("hashlib").sha256(data).hexdigest(), "size": len(data)})
    metadata = {
        "overlay_used": True,
        "overlay_repo": overlay["repo"],
        "overlay_revision": overlay["revision"],
        "overlay_reason": overlay["reason"],
        "overlay_files": overlay_manifest,
        "prepared_path": str(prepared),
    }
    (prepared / "confirmatory_preparation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return prepared, metadata


def load_probe_plan(carrier_id: str) -> list[dict]:
    rows = [x for x in cf.read_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl") if x["carrier_id"] == carrier_id]
    return sorted(rows, key=lambda x: x["execution_index"])


def validate_probe_messages(row: dict, messages: list[dict]):
    if row["probe_type"] == "copy":
        user = cf.canon(messages)
    else:
        user = messages[0]["content"]
    if cf.sha_text(user) != row["user_content_sha256"]:
        raise RuntimeError(f"probe user-content hash mismatch for {row['probe_id']}")


def render_prompts(tokenizer, rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for row in rows:
        messages = cf.prompt_messages(row)
        validate_probe_messages(row, messages)
        kwargs = {"tokenize": False, "add_generation_prompt": True, **cfg.get("chat_template_kwargs", {})}
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        bos = tokenizer.bos_token
        add_special = bos is None or not rendered.startswith(bos)
        ids = tokenizer.encode(rendered, add_special_tokens=add_special)
        out.append({
            "carrier_id": cfg["carrier_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"], "probe_scope": row["probe_scope"],
            "user_content_sha256": row["user_content_sha256"], "rendered_prompt": rendered, "rendered_prompt_sha256": cf.sha_text(rendered),
            "encode_add_special_tokens": add_special, "prompt_token_count": len(ids), "prompt_token_ids": ids,
        })
    return out


def completion_key(carrier_id: str, probe_key: str, prompt_hash: str, extra_eos_tokens=None, artifact_seed: str = "") -> str:
    eos_part = "\x1f".join(sorted(extra_eos_tokens or []))
    return cf.sha_text("\0".join([carrier_id, probe_key, prompt_hash, "greedy", str(cf.MAX_TOKENS), eos_part, artifact_seed]))


def chunks(rows: list[dict], n: int):
    for i in range(0, len(rows), n):
        yield i // n, rows[i:i + n]


def runtime_base(cfg: dict) -> dict:
    return {
        "carrier_id": cfg["carrier_id"],
        "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
        "host_platform": platform.platform(), "host_machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", None), "mlx_lm_version": mlx_lm.__version__, "transformers_version": transformers.__version__,
    }


def write_resume_state(current_carrier: str | None = None, phase: str | None = None, extra: str | None = None):
    plan = cf.read_jsonl(cf.CONFIRMATORY / "probe_plan.jsonl")
    raw = [x for x in cf.read_jsonl(RAW_PATH) if x.get("status") == "ok"]
    done = {(x["carrier_id"], x["probe_key"]) for x in raw}
    lines = [
        "# Confirmatory execution resume state",
        "",
        f"Updated UTC: {utcnow()}",
        "",
        "Resume rule: rerun `src/run_confirmatory_behavioral.py` for the interrupted carrier/phase with the same or a lower batch size. Completed completion keys are skipped; never regenerate completed keys.",
        "",
        "| carrier | complete | total | remaining |",
        "|---|---:|---:|---:|",
    ]
    for cfg in cf.carrier_configs(include_descriptive=True):
        rows = [x for x in plan if x["carrier_id"] == cfg["carrier_id"]]
        finished = sum((x["carrier_id"], x["probe_key"]) in done for x in rows)
        lines.append(f"| `{cfg['carrier_id']}` | {finished} | {len(rows)} | {len(rows) - finished} |")
    lines.extend(["", f"Current carrier: `{current_carrier}`", f"Current phase: `{phase}`"])
    if extra:
        lines.extend(["", extra])
    RESUME_PATH.write_text("\n".join(lines) + "\n")


def append_runtime(path: Path, rows: list[dict]):
    cf.append_jsonl(path, rows)
    write_resume_state()


def template_check(tokenizer, cfg: dict, load_path: Path) -> dict:
    target = "test"
    messages = cf.prompt_messages({"probe_type": "copy", "target": target})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **cfg.get("chat_template_kwargs", {}))
    return {
        "carrier_id": cfg["carrier_id"], "load_path": str(load_path), "rendered_sha256": cf.sha_text(rendered),
        "assistant_boundary_suffix": rendered[-80:], "chat_template_sha256": cf.sha_text(getattr(tokenizer, "chat_template", "") or ""),
        "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token,
    }


def smoke(cfg: dict, batch_size: int):
    snapshot_path = Path(snapshot_download(cfg["hf_repo"], revision=cfg["revision"]))
    load_path, preparation = prepare_inference_snapshot(snapshot_path, cfg)
    started = utcnow()
    tic = time.perf_counter()
    rss_before = process_rss_gb()
    model, tokenizer = mlx_lm.load(str(load_path), lazy=False)
    load_seconds = time.perf_counter() - tic
    rss_after = process_rss_gb()
    check = template_check(tokenizer, cfg, load_path)
    messages = cf.prompt_messages({"probe_type": "copy", "target": "test"})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **cfg.get("chat_template_kwargs", {}))
    add_special = tokenizer.bos_token is None or not rendered.startswith(tokenizer.bos_token or "\0")
    ids = tokenizer.encode(rendered, add_special_tokens=add_special)
    gen_tic = time.perf_counter()
    response = mlx_lm.batch_generate(model, tokenizer, [ids], max_tokens=8, return_token_ids=True, sampler=greedy_sampler, completion_batch_size=1, prefill_batch_size=1)
    gen_seconds = time.perf_counter() - gen_tic
    row = {
        **runtime_base(cfg), "event": "memory_smoke", "utc": utcnow(), "started_utc": started,
        "hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "snapshot_path": str(snapshot_path), "load_path": str(load_path), "preparation": preparation,
        "file_manifest": file_manifest(snapshot_path), "load_seconds": load_seconds,
        "process_rss_gb_before_load": rss_before, "process_rss_gb_after_load": rss_after, "process_rss_gb_after_generation": process_rss_gb(),
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1e9, "generation_seconds": gen_seconds,
        "smoke_prompt": "Copy the input exactly. Input: <IN>test</IN>", "smoke_output": response.texts[0],
        "smoke_output_token_count": len(response.token_ids[0]), "batch_size_requested": batch_size,
        "template_rendering_check": check,
        "quantization_class": cfg["quantization_class"], "dtype": cfg["quantization"],
        "host_memory_bytes": 17179869184,
        "memory_pass_margin_estimate_gb": 16 - process_rss_gb(),
    }
    cf.append_jsonl(SMOKE_PATH, [row])
    write_resume_state(cfg["carrier_id"], "smoke")
    mx.clear_cache()
    print(json.dumps({"carrier_id": cfg["carrier_id"], "smoke_output": row["smoke_output"], "rss_after_load": rss_after, "peak_memory_gb": row["mlx_peak_memory_gb"]}, indent=2))
    return row


def run_model(cfg: dict, batch_size: int, phase: str, limit: int | None):
    cf.CONFIRMATORY.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot_download(cfg["hf_repo"], revision=cfg["revision"]))
    load_path, preparation = prepare_inference_snapshot(snapshot_path, cfg)
    rows = load_probe_plan(cfg["carrier_id"])
    if phase == "ascii":
        rows = [x for x in rows if x["probe_scope"] == "ascii"]
    elif phase == "unicode":
        rows = [x for x in rows if x["probe_scope"] == "unicode"]
    elif phase != "all":
        raise ValueError(phase)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no probes selected")
    write_resume_state(cfg["carrier_id"], phase, "Model loading started.")
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
    prompt_path = PROMPT_DIR / f"{cfg['carrier_id']}-{phase}.jsonl"
    cf.write_jsonl(prompt_path, [{k: v for k, v in row.items() if k != "prompt_token_ids"} for row in rendered])
    existing = cf.read_jsonl(RAW_PATH)
    completed = {x["completion_key"] for x in existing if x.get("status") == "ok"}
    metadata_rows = file_manifest(snapshot_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    invocation = {
        **runtime_base(cfg), "event": "model_loaded", "utc": utcnow(), "load_started_utc": load_started,
        "load_seconds": load_seconds, "hf_repo": cfg["hf_repo"], "revision": cfg["revision"],
        "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
        "snapshot_path": str(snapshot_path), "load_path": str(load_path), "preparation": preparation, "file_manifest": metadata_rows,
        "chat_template_sha256": cf.sha_text(chat_template) if isinstance(chat_template, str) else None,
        "eos_token_ids": sorted(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else [tokenizer.eos_token_id],
        "bos_token": tokenizer.bos_token,
        "decode_params": {"sampler": "greedy_sampler/argmax", "temperature": 0.0, "max_tokens": cf.MAX_TOKENS, "repetition_penalty": None, "top_p": None, "top_k": None, "extra_eos_tokens": cfg.get("extra_eos_tokens", [])},
        "quantization": cfg["quantization"], "quantization_class": cfg["quantization_class"], "batch_size": batch_size,
        "selected_probe_rows": len(rows), "phase": phase,
        "process_rss_gb_before_load": rss_before_load, "process_rss_gb_after_load": rss_after_load,
    }
    cf.append_jsonl(RUNTIME_PATH, [invocation])
    artifact_seed = cf.sha_text(json.dumps({
        "hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "tokenizer_overlay": cfg.get("tokenizer_overlay"),
        "chat_template_sha256": invocation["chat_template_sha256"], "eos_token_ids": invocation["eos_token_ids"],
    }, sort_keys=True, separators=(",", ":")))
    generated = 0
    paired = list(zip(rows, rendered))
    for chunk_index, chunk_rows in chunks(paired, batch_size):
        keys = [completion_key(cfg["carrier_id"], row["probe_key"], prompt["rendered_prompt_sha256"], cfg.get("extra_eos_tokens", []), artifact_seed) for row, prompt in chunk_rows]
        if all(key in completed for key in keys):
            continue
        if any(key in completed for key in keys):
            raise RuntimeError(f"partially complete chunk {chunk_index} for {cfg['carrier_id']}; append-only resume requires manual inspection")
        prompt_ids = [prompt["prompt_token_ids"] for _, prompt in chunk_rows]
        batch_id = f"{cfg['carrier_id']}-{phase}-chunk{chunk_index:03d}"
        started = utcnow()
        tic = time.perf_counter()
        rss_before_batch = process_rss_gb()
        try:
            response = mlx_lm.batch_generate(
                model, tokenizer, prompt_ids, max_tokens=cf.MAX_TOKENS, return_token_ids=True,
                sampler=greedy_sampler, completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size),
            )
        except Exception as exc:
            cf.append_jsonl(RUNTIME_PATH, [{**runtime_base(cfg), "event": "batch_error", "utc": utcnow(), "batch_id": batch_id, "error_type": type(exc).__name__, "error": str(exc), "batch_size": batch_size}])
            write_resume_state(cfg["carrier_id"], phase, f"Batch error at `{batch_id}` with batch size {batch_size}: `{type(exc).__name__}: {exc}`. Resume with the next lower predefined batch size after inspecting memory state.")
            raise
        elapsed = time.perf_counter() - tic
        finished = utcnow()
        rss_after_batch = process_rss_gb()
        if len(response.texts) != len(chunk_rows) or len(response.token_ids) != len(chunk_rows):
            raise RuntimeError(f"batch response length mismatch: {len(response.texts)} texts, {len(response.token_ids or [])} token rows, {len(chunk_rows)} prompts")
        out_rows = []
        for (row, prompt), key, text, token_ids in zip(chunk_rows, keys, response.texts, response.token_ids):
            inventory_identity = {k: v for k, v in row.items() if k not in ("user_content",)}
            out_rows.append({
                "status": "ok", "completion_key": key, "batch_id": batch_id, "batch_member_keys": keys, "artifact_seed": artifact_seed,
                "carrier_id": cfg["carrier_id"], "model_id": cfg["carrier_id"], "probe_key": row["probe_key"], "probe_type": row["probe_type"], "probe_scope": row["probe_scope"],
                "hf_repo": cfg["hf_repo"], "model_revision": cfg["revision"], "upstream_repo": cfg["upstream_repo"], "upstream_revision": cfg["upstream_revision"],
                "prompt_sha256": prompt["rendered_prompt_sha256"], "user_content_sha256": row["user_content_sha256"], "inventory_row_sha256": cf.sha_text(cf.canon(inventory_identity)),
                "decode_params": invocation["decode_params"], "dtype": cfg["quantization"], "quantization_class": cfg["quantization_class"],
                "host_python_version": platform.python_version(), "host_unicodedata_version": unicodedata.unidata_version,
                "raw_text": text, "output_token_ids": token_ids, "output_token_count": len(token_ids),
                "finish_reason_inferred": "length" if len(token_ids) == cf.MAX_TOKENS else "stop",
                "batch_started_utc": started, "batch_finished_utc": finished, "batch_elapsed_seconds": elapsed,
                "prompt_token_count": prompt["prompt_token_count"],
            })
        cf.append_jsonl(RAW_PATH, out_rows)
        completed.update(keys)
        generated += len(out_rows)
        stats = response.stats
        cf.append_jsonl(RUNTIME_PATH, [{
            **runtime_base(cfg), "event": "batch_completed", "utc": finished, "batch_id": batch_id,
            "probe_count": len(out_rows), "elapsed_seconds": elapsed,
            "prompt_tokens": getattr(stats, "prompt_tokens", None), "generation_tokens": getattr(stats, "generation_tokens", None),
            "prompt_tps": getattr(stats, "prompt_tps", None), "generation_tps": getattr(stats, "generation_tps", None),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "process_rss_gb_before_batch": rss_before_batch, "process_rss_gb_after_batch": rss_after_batch,
        }])
        write_resume_state(cfg["carrier_id"], phase, f"Last completed batch `{batch_id}` at `{finished}`.")
        print(json.dumps({"carrier_id": cfg["carrier_id"], "batch": batch_id, "rows": len(out_rows), "seconds": round(elapsed, 2), "generated_so_far": generated, "rss_gb": round(rss_after_batch, 2)}), flush=True)
    mx.clear_cache()
    cf.append_jsonl(RUNTIME_PATH, [{**runtime_base(cfg), "event": "model_phase_finished", "utc": utcnow(), "phase": phase, "process_rss_gb_before_exit": process_rss_gb(), "generated_rows_this_invocation": generated}])
    write_resume_state(cfg["carrier_id"], phase, f"Phase `{phase}` finished for `{cfg['carrier_id']}`.")
    return {"carrier_id": cfg["carrier_id"], "generated": generated, "probes": len(rows), "load_seconds": load_seconds, "phase": phase}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True, choices=[x["carrier_id"] for x in cf.carrier_configs(include_descriptive=True)])
    parser.add_argument("--phase", choices=["ascii", "unicode", "all"], default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = cf.carrier_config(args.carrier)
    batch_size = args.batch_size or cfg.get("default_batch_size", cf.BATCH_SIZE)
    if args.smoke:
        result = smoke(cfg, batch_size)
    else:
        result = run_model(cfg, batch_size, args.phase, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
