from __future__ import annotations

import base64
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
VYCE = RESULTS / "vyce_hosted"
LEDGER_PATH = VYCE / "cost_ledger.jsonl"
ENDPOINT = "https://vyceai.com/v1/chat/completions"
MODELS_ENDPOINT = "https://vyceai.com/v1/models"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36"
HARD_CAP_USD = 8.0
CAP_INPUT_PER_MILLION = 1.0
CAP_OUTPUT_PER_MILLION = 3.0
MAX_TOKENS = 64

CANDIDATE_MODELS = ["gpt-oss-120b", "deepseek-v4-flash", "deepseek-v4-flash-lr", "glm-5.3-flash"]
MODEL_PRIORITY = ["gpt-oss-120b", "deepseek-v4-flash", "glm-5.3-flash", "deepseek-v4-flash-lr"]

PUBLISHED_PRICE_ESTIMATES = {
    "gpt-oss-120b": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
        "source": "OpenCode/gpt-oss-120b catalog list-price proxy ($0.15/$0.60 per 1M, checked 2026-11-26, https://opencode.ai/data/compare/openai/gpt-oss-120b/zhipu/glm-5-3-flash), external proxy only; vyce relay pricing was not exposed",
    },
    "deepseek-v4-flash": {
        "input_per_million": 0.30,
        "output_per_million": 1.20,
        "source": "official DeepSeek API peak list price for deepseek-flash/DeepSeek-V4.1-Flash-1017 ($0.30/$1.20 per 1M, checked 2026-11-26, https://api-docs.deepseek.com/quick_start/pricing/), external proxy only; vyce relay pricing was not exposed",
    },
    "glm-5.3-flash": {
        "input_per_million": 0.15,
        "output_per_million": 0.50,
        "source": "official Z.AI GLM-5.3-Flash list price ($0.15/$0.50 per 1M, checked 2026-11-26, https://docs.z.ai/guides/overview/pricing), external proxy only; vyce relay pricing was not exposed",
    },
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"corrupt append-only jsonl at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def load_key() -> str:
    key_path = Path.home() / ".secrets" / "vyceai-api-key"
    key = key_path.read_text().strip()
    if not key:
        raise RuntimeError("vyce key file is empty")
    return key


def price_estimate(model: str) -> dict:
    return PUBLISHED_PRICE_ESTIMATES.get(model, {
        "input_per_million": CAP_INPUT_PER_MILLION,
        "output_per_million": CAP_OUTPUT_PER_MILLION,
        "source": "no model-specific published estimate; conservative fallback used",
    })


def cost_breakdown(model: str, input_tokens: int, output_tokens: int) -> dict:
    price = price_estimate(model)
    estimated = input_tokens * price["input_per_million"] / 1_000_000 + output_tokens * price["output_per_million"] / 1_000_000
    cap_cost = input_tokens * CAP_INPUT_PER_MILLION / 1_000_000 + output_tokens * CAP_OUTPUT_PER_MILLION / 1_000_000
    return {
        "price_input_per_million_usd": price["input_per_million"],
        "price_output_per_million_usd": price["output_per_million"],
        "pricing_source": price["source"],
        "estimated_cost_usd": estimated,
        "cap_input_per_million_usd": CAP_INPUT_PER_MILLION,
        "cap_output_per_million_usd": CAP_OUTPUT_PER_MILLION,
        "cap_cost_usd": cap_cost,
    }


def ledger_totals(path: Path = LEDGER_PATH) -> dict:
    estimated = 0.0
    cap = 0.0
    calls = 0
    for row in read_jsonl(path):
        if row.get("event") == "api_call":
            calls += 1
            estimated += float(row.get("estimated_cost_usd") or 0.0)
            cap += float(row.get("cap_cost_usd") or 0.0)
    return {"api_calls": calls, "estimated_cost_usd": estimated, "cap_cost_usd": cap}


def message_token_upper_bound(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages) + 16 * len(messages) + 64


def projected_cap_cost(model: str, messages: list[dict], max_tokens: int) -> dict:
    input_upper = message_token_upper_bound(messages)
    output_upper = max_tokens
    cost = cost_breakdown(model, input_upper, output_upper)
    return {
        "input_tokens_upper_bound": input_upper,
        "output_tokens_upper_bound": output_upper,
        "projected_cap_cost_usd": cost["cap_cost_usd"],
        "cost_bound_method": "input UTF-8 byte count plus chat-message overhead, with output at max_tokens; conservative because relay tokenizer and hidden prompt injection are unobservable",
    }


def append_ledger(row: dict, ledger_path: Path = LEDGER_PATH):
    row = dict(row)
    row["entry_sha256"] = sha_text(canon(row))
    append_jsonl(ledger_path, [row])


def budget_abort(reason: str, model: str | None, projection: dict, cap_usd: float, ledger_path: Path = LEDGER_PATH, purpose: str = "budget_preflight") -> RuntimeError:
    totals = ledger_totals(ledger_path)
    row = {
        "event": "budget_abort", "utc": utcnow(), "purpose": purpose, "model": model,
        "reason": reason, "cap_usd": cap_usd, **projection,
        "cumulative_estimated_cost_usd": totals["estimated_cost_usd"],
        "cumulative_cap_cost_usd": totals["cap_cost_usd"],
    }
    append_ledger(row, ledger_path)
    return RuntimeError(json.dumps({"budget_abort": reason, "cumulative_cap_cost_usd": totals["cap_cost_usd"], "projection": projection, "cap_usd": cap_usd}, sort_keys=True))


def ensure_budget(model: str, messages: list[dict], max_tokens: int, cap_usd: float, ledger_path: Path = LEDGER_PATH, purpose: str = "budget_preflight"):
    projection = projected_cap_cost(model, messages, max_tokens)
    totals = ledger_totals(ledger_path)
    if totals["cap_cost_usd"] + projection["projected_cap_cost_usd"] > cap_usd:
        raise budget_abort("projected conservative cost would exceed hard cap", model, projection, cap_usd, ledger_path, purpose)
    return projection


def usage_tokens(parsed: dict | None, messages: list[dict], max_tokens: int) -> dict:
    usage = parsed.get("usage") if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict) else {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    estimation = []
    if not isinstance(prompt, int) or prompt < 0:
        prompt = message_token_upper_bound(messages)
        estimation.append("prompt_tokens missing; conservative byte/overhead upper bound used")
    if not isinstance(completion, int) or completion < 0:
        completion = max_tokens
        estimation.append("completion_tokens missing; max_tokens upper bound used")
    if not isinstance(total, int) or total < 0:
        total = prompt + completion
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "usage": usage,
        "usage_source": "relay usage field where present; conservative upper bound otherwise",
        "usage_estimation_notes": estimation,
        "usage_missing": not bool(usage),
    }


def classify_relay_artifacts(parsed: dict | None, body_text: str, http_status: int | None, network_error: str | None, expected_model: str, finish_reason: str | None, content: str | None) -> list[str]:
    artifacts = []
    if network_error:
        artifacts.append("network_error")
    if http_status is not None and http_status != 200:
        artifacts.append("http_error")
    if parsed is None:
        artifacts.append("invalid_or_missing_json_response")
    if isinstance(parsed, dict):
        if parsed.get("error"):
            artifacts.append("api_error_object")
        choices = parsed.get("choices")
        if not choices:
            artifacts.append("no_choices")
        if parsed.get("model") and parsed.get("model") != expected_model:
            artifacts.append("returned_model_differs_from_requested")
        if "usage" not in parsed:
            artifacts.append("usage_missing")
    if content is None:
        artifacts.append("content_missing")
    elif content == "":
        artifacts.append("empty_content")
    else:
        low = content.lower()
        if "i'm sorry" in low or "i cannot" in low or "can't assist" in low:
            artifacts.append("possible_refusal_text")
        if "<think" in low or "</think" in low:
            artifacts.append("think_markup_in_content")
        if "<|" in content or "|>" in content:
            artifacts.append("special_token_literal_in_content")
    if finish_reason == "length":
        artifacts.append("truncated_at_max_tokens")
    elif finish_reason not in (None, "stop"):
        artifacts.append("unusual_finish_reason")
    return sorted(set(artifacts))


def redacted_body(data: bytes, key: str) -> tuple[bytes, bool]:
    secret = key.encode()
    if secret and secret in data:
        return data.replace(secret, b"[REDACTED_SECRET]"), True
    return data, False


def parse_response_body(data: bytes) -> tuple[str, dict | None, bool]:
    try:
        text = data.decode("utf-8")
        decode_error = False
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        decode_error = True
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return text, parsed, decode_error


def selected_headers(headers) -> dict:
    wanted = ["content-type", "x-request-id", "request-id", "cf-ray", "date"]
    out = {}
    for name in wanted:
        value = headers.get(name) if headers else None
        if value:
            out[name] = value
    return out


def post_chat_completion(model: str, messages: list[dict], max_tokens: int, temperature: float | None, purpose: str, cap_usd: float = HARD_CAP_USD, ledger_path: Path = LEDGER_PATH, timeout: int = 180, extra_body: dict | None = None) -> dict:
    projection = ensure_budget(model, messages, max_tokens, cap_usd, ledger_path, purpose)
    key = load_key()
    request_obj = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        request_obj["temperature"] = temperature
    if extra_body:
        request_obj.update(extra_body)
    request_text = json.dumps(request_obj, ensure_ascii=False, separators=(",", ":"))
    request_bytes = request_text.encode("utf-8")
    request_hash = sha_bytes(request_bytes)
    started = utcnow()
    tic = time.perf_counter()
    status = None
    headers = {}
    body = b""
    network_error = None
    req = urllib.request.Request(
        ENDPOINT,
        data=request_bytes,
        method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            headers = selected_headers(response.headers)
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = selected_headers(exc.headers)
        body = exc.read()
    except Exception as exc:
        network_error = type(exc).__name__ + ": " + str(exc)[:300]
    latency = time.perf_counter() - tic
    finished = utcnow()
    body, secret_redacted = redacted_body(body, key)
    body_text, parsed, decode_error = parse_response_body(body)
    content = None
    finish_reason = None
    refusal = None
    message_obj = None
    if isinstance(parsed, dict) and parsed.get("choices"):
        choice = parsed["choices"][0]
        finish_reason = choice.get("finish_reason")
        message_obj = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message_obj.get("content")
        refusal = message_obj.get("refusal")
    toks = usage_tokens(parsed, messages, max_tokens)
    costs = cost_breakdown(model, toks["input_tokens"], toks["output_tokens"])
    totals = ledger_totals(ledger_path)
    artifacts = classify_relay_artifacts(parsed, body_text, status, network_error, model, finish_reason, content)
    if decode_error:
        artifacts.append("response_body_utf8_decode_error")
    if secret_redacted:
        artifacts.append("secret_reflected_and_redacted")
    if refusal:
        artifacts.append("refusal_field_present")
    artifacts = sorted(set(artifacts))
    ledger_row = {
        "event": "api_call", "utc": finished, "purpose": purpose, "model": model, "endpoint": ENDPOINT,
        "http_status": status, "network_error": network_error, "request_sha256": request_hash,
        "response_body_sha256": sha_bytes(body), "request_id": (parsed or {}).get("request_id") or headers.get("x-request-id") or headers.get("request-id"),
        **{k: toks[k] for k in ("input_tokens", "output_tokens", "total_tokens", "usage_source", "usage_missing")},
        "usage_estimation_notes": toks["usage_estimation_notes"],
        **costs,
        "cumulative_estimated_cost_before_usd": totals["estimated_cost_usd"],
        "cumulative_cap_cost_before_usd": totals["cap_cost_usd"],
        "cumulative_estimated_cost_after_usd": totals["estimated_cost_usd"] + costs["estimated_cost_usd"],
        "cumulative_cap_cost_after_usd": totals["cap_cost_usd"] + costs["cap_cost_usd"],
        "cap_usd": cap_usd, "max_tokens": max_tokens, "temperature": temperature,
        "latency_seconds": latency, "relay_artifacts": artifacts,
    }
    append_ledger(ledger_row, ledger_path)
    return {
        "request_obj": request_obj,
        "request_sha256": request_hash,
        "http_status": status,
        "headers": headers,
        "network_error": network_error,
        "started_utc": started,
        "finished_utc": finished,
        "latency_seconds": latency,
        "body_bytes": body,
        "body_text": body_text,
        "body_base64": base64.b64encode(body).decode("ascii"),
        "body_sha256": sha_bytes(body),
        "parsed": parsed,
        "content": content,
        "finish_reason": finish_reason,
        "refusal": refusal,
        "message_obj": message_obj,
        "tokens": toks,
        "costs": costs,
        "ledger_projection": projection,
        "relay_artifacts": artifacts,
        "secret_redacted": secret_redacted,
    }


def get_models(cap_usd: float = HARD_CAP_USD, ledger_path: Path = LEDGER_PATH) -> dict:
    key = load_key()
    messages = []
    projection = ensure_budget("gpt-oss-120b", messages, 0, cap_usd, ledger_path, "models_list")
    started = utcnow()
    tic = time.perf_counter()
    status = None
    headers = {}
    body = b""
    network_error = None
    req = urllib.request.Request(MODELS_ENDPOINT, headers={"Authorization": "Bearer " + key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            status = response.status
            headers = selected_headers(response.headers)
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = selected_headers(exc.headers)
        body = exc.read()
    except Exception as exc:
        network_error = type(exc).__name__ + ": " + str(exc)[:300]
    latency = time.perf_counter() - tic
    finished = utcnow()
    body, secret_redacted = redacted_body(body, key)
    body_text, parsed, decode_error = parse_response_body(body)
    totals = ledger_totals(ledger_path)
    ledger_row = {
        "event": "api_call", "utc": finished, "purpose": "models_list", "model": None, "endpoint": MODELS_ENDPOINT,
        "http_status": status, "network_error": network_error, "request_sha256": None,
        "response_body_sha256": sha_bytes(body), "request_id": headers.get("x-request-id") or headers.get("request-id"),
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "usage_source": "non-completion metadata endpoint; zero token cost assumed", "usage_missing": True, "usage_estimation_notes": [],
        "price_input_per_million_usd": None, "price_output_per_million_usd": None,
        "pricing_source": "non-completion metadata endpoint", "estimated_cost_usd": 0.0,
        "cap_input_per_million_usd": CAP_INPUT_PER_MILLION, "cap_output_per_million_usd": CAP_OUTPUT_PER_MILLION, "cap_cost_usd": 0.0,
        "cumulative_estimated_cost_before_usd": totals["estimated_cost_usd"], "cumulative_cap_cost_before_usd": totals["cap_cost_usd"],
        "cumulative_estimated_cost_after_usd": totals["estimated_cost_usd"], "cumulative_cap_cost_after_usd": totals["cap_cost_usd"],
        "cap_usd": cap_usd, "max_tokens": 0, "temperature": None, "latency_seconds": latency, "relay_artifacts": [],
    }
    append_ledger(ledger_row, ledger_path)
    return {
        "http_status": status, "headers": headers, "network_error": network_error,
        "started_utc": started, "finished_utc": finished, "latency_seconds": latency,
        "body_text": body_text, "body_base64": base64.b64encode(body).decode("ascii"), "body_sha256": sha_bytes(body),
        "parsed": parsed, "decode_error": decode_error, "secret_redacted": secret_redacted,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
