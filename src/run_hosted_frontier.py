from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import confirmatory_common as cf
from hosted_frontier_adapters import DEFAULT_MODELS, adapter_for
from hosted_frontier_common import (
    DEFAULT_CAP_USD,
    DRY_RUN_ROOT,
    HOSTED,
    LOGICAL_BATCH_SIZE,
    MAX_TOKENS,
    BudgetAbort,
    HashChainLedger,
    HostedFrontierError,
    MissingApiKey,
    ProviderAPIError,
    ResponseIntegrityError,
    ServedModelMismatch,
    UrllibTransport,
    append_jsonl,
    b64e,
    canon,
    ensure_budget,
    input_token_upper_bound,
    load_api_key,
    model_slug,
    price_for,
    pricing_table_sha256,
    projection_for_messages,
    read_jsonl,
    redact_secret,
    sha_bytes,
    sha_text,
    sum_projections,
    token_cost_usd,
    transport_event_hash,
    utcnow,
    validate_ledger_rows,
    verify_archive_rows,
    write_json,
    write_jsonl,
)
from hosted_frontier_fake import CorrectFakeTransport
from hosted_frontier_plan import DEFAULT_REFERENCE_CARRIER, phase_for_probe, phase_ordered_rows, transform_plan
from score_confirmatory_behavioral import extractor_mutation_checks, score_row

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
PHASES = ("ascii_gate", "unicode_copy_gate", "unicode_scalar", "unicode_count")


@dataclass(frozen=True)
class HarnessConfig:
    provider: str
    model: str
    cap_usd: float = DEFAULT_CAP_USD
    dry_run: bool = False
    artifact_root: Path | None = None
    reference_carrier: str = DEFAULT_REFERENCE_CARRIER
    batch_size: int = LOGICAL_BATCH_SIZE
    retry_limit: int = 4
    timeout_seconds: int = 180


def artifact_paths(root: Path, provider: str, model: str, reference_carrier: str) -> dict[str, Path]:
    provider_dir = root / provider
    model_dir = provider_dir / model_slug(model) / model_slug(reference_carrier)
    return {
        "provider_dir": provider_dir,
        "model_dir": model_dir,
        "ledger": provider_dir / "cost_ledger.jsonl",
        "transport": model_dir / "transport_events.jsonl",
        "raw": model_dir / "hosted_raw.jsonl",
        "scored": model_dir / "hosted_scored.jsonl",
        "runtime": model_dir / "runtime.jsonl",
        "gates": model_dir / "gates.json",
        "preflight": model_dir / "preflight.json",
        "manifest": model_dir / "run_manifest.json",
        "plan": model_dir / "probe_plan_hosted.jsonl",
    }


def runtime_event(event: str, provider: str, model: str, extra: dict | None = None) -> dict:
    return {
        "event": event,
        "utc": utcnow(),
        "provider": provider,
        "model": model,
        "host_python_version": platform.python_version(),
        "host_unicodedata_version": unicodedata.unidata_version,
        **(extra or {}),
    }


def completion_key(adapter, row: dict, inventory_hash: str) -> str:
    obj = {
        "provider": adapter.provider,
        "endpoint": adapter.endpoint,
        "model": adapter.model,
        "probe_key": row["probe_key"],
        "reference_probe_key": row["reference_probe_key"],
        "user_content_sha256": row["user_content_sha256"],
        "inventory_identity_sha256": inventory_hash,
        "decode_params": adapter.prepare_request(cf.prompt_messages(row), MAX_TOKENS).decode_params,
    }
    return sha_text(canon(obj))


def usage_costs(price: dict, projection: dict, adapted) -> dict:
    projected = projection["projected_conservative_cost_usd"]
    priced = None
    if adapted is not None and adapted.usage_present:
        priced = token_cost_usd(adapted.input_tokens, adapted.output_tokens, price["input_per_million_usd"], price["output_per_million_usd"])
        debit = max(projected, priced)
        basis = "max(conservative byte/overhead plus max_tokens projection, provider-reported priced usage)"
    else:
        debit = projected
        basis = "conservative byte/overhead plus max_tokens projection because provider usage was unavailable or invalid"
    return {
        "priced_usage_cost_usd": priced,
        "conservative_cap_debit_usd": debit,
        "debit_basis": basis,
    }


def archive_attempt(adapter, prepared, transport_response, api_key: str, adapted, integrity_status: str, integrity_error: str | None, purpose: str, attempt: int, projection: dict, price: dict, paths: dict[str, Path], ledger: HashChainLedger, cap_usd: float) -> tuple[dict, dict]:
    stored_request, request_secret_redacted = redact_secret(prepared.body_bytes, api_key)
    stored_response, response_secret_redacted = redact_secret(transport_response.body, api_key)
    if response_secret_redacted:
        integrity_status = "secret_reflected_in_response"
        integrity_error = "provider response reflected the API key; response was redacted and refused"
    request_sha = sha_bytes(stored_request)
    response_sha = sha_bytes(stored_response)
    request_headers = {}
    for name, value in prepared.headers_with_auth(api_key).items():
        request_headers[name] = "[REDACTED]" if name.lower() in ("authorization", "x-api-key", "x-goog-api-key") else value
    costs = usage_costs(price, projection, adapted)
    totals_before = ledger.totals()
    event = {
        "event": "transport_attempt",
        "utc": utcnow(),
        "provider": adapter.provider,
        "model": adapter.model,
        "adapter_id": adapter.adapter_id,
        "purpose": purpose,
        "attempt": attempt,
        "url": prepared.url,
        "method": prepared.method,
        "request_headers": request_headers,
        "request_body_base64": b64e(stored_request),
        "request_body_sha256": request_sha,
        "request_body_unredacted_sha256": sha_bytes(prepared.body_bytes),
        "response_body_base64": b64e(stored_response),
        "response_body_sha256": response_sha,
        "response_body_unredacted_sha256": sha_bytes(transport_response.body),
        "http_status": transport_response.http_status,
        "network_error": transport_response.network_error,
        "response_headers": transport_response.headers,
        "latency_seconds": transport_response.latency_seconds,
        "integrity_status": integrity_status,
        "integrity_error": integrity_error,
        "secret_redacted_in_request": request_secret_redacted,
        "secret_redacted_in_response": response_secret_redacted,
    }
    event["event_sha256"] = transport_event_hash(event)
    append_jsonl(paths["transport"], [event])
    ledger_event = {
        "event": "api_call",
        "utc": utcnow(),
        "provider": adapter.provider,
        "model": adapter.model,
        "purpose": purpose,
        "attempt": attempt,
        "http_status": transport_response.http_status,
        "network_error": transport_response.network_error,
        "request_body_sha256": request_sha,
        "response_body_sha256": response_sha,
        "latency_seconds": transport_response.latency_seconds,
        "input_tokens_reported": adapted.input_tokens if adapted else None,
        "output_tokens_reported": adapted.output_tokens if adapted else None,
        "total_tokens_reported": adapted.total_tokens if adapted else None,
        "usage_present": bool(adapted and adapted.usage_present),
        **costs,
        "price_input_per_million_usd": price["input_per_million_usd"],
        "price_output_per_million_usd": price["output_per_million_usd"],
        "pricing_source_url": price["source_url"],
        "pricing_source_checked_date": price["source_checked_date"],
        "pricing_table_sha256": pricing_table_sha256(),
        "cumulative_conservative_cap_debit_before_usd": totals_before["conservative_cap_debit_usd"],
        "cumulative_conservative_cap_debit_after_usd": totals_before["conservative_cap_debit_usd"] + costs["conservative_cap_debit_usd"],
        "cap_usd": cap_usd,
        "max_tokens": prepared.max_tokens,
        "decode_params": prepared.decode_params,
    }
    ledger_row = ledger.append(ledger_event)
    return event, ledger_row


def perform_attempt(adapter, transport, key_loader, messages: list[dict], max_tokens: int, purpose: str, attempt: int, price: dict, paths: dict[str, Path], ledger: HashChainLedger, cap_usd: float) -> dict:
    prepared = adapter.prepare_request(messages, max_tokens)
    projection = projection_for_messages(price, messages, max_tokens)
    ensure_budget(ledger, projection, cap_usd, adapter.provider, adapter.model, purpose)
    api_key = key_loader()
    if api_key.encode("utf-8") in prepared.body_bytes:
        raise HostedFrontierError("API key would appear inside the request body; refusing transport")
    transport_response = transport.send(prepared, api_key)
    adapted = None
    integrity_error = None
    if api_key.encode("utf-8") in transport_response.body:
        integrity_status = "secret_reflected_in_response"
        integrity_error = "provider response reflected the API key; response was redacted and refused"
    elif transport_response.network_error:
        integrity_status = "network_error"
    elif transport_response.http_status != 200:
        integrity_status = "http_error"
    else:
        integrity_status = "ok"
        try:
            adapted = adapter.parse_response(transport_response.body)
        except ResponseIntegrityError as exc:
            integrity_status = "malformed_response"
            integrity_error = str(exc)
    event, ledger_row = archive_attempt(
        adapter, prepared, transport_response, api_key, adapted, integrity_status, integrity_error,
        purpose, attempt, projection, price, paths, ledger, cap_usd,
    )
    if integrity_error:
        raise ResponseIntegrityError(integrity_error)
    return {
        "prepared": prepared,
        "transport_response": transport_response,
        "adapted": adapted,
        "transport_event": event,
        "ledger_row": ledger_row,
        "projection": projection,
    }


def retry_delay_seconds(attempt: int) -> float:
    return min(30.0, 2.0 ** (attempt - 1))


def call_with_retries(adapter, transport, key_loader, messages: list[dict], max_tokens: int, purpose: str, price: dict, paths: dict[str, Path], ledger: HashChainLedger, cap_usd: float, retry_limit: int, sleep_func) -> dict:
    last = None
    for attempt in range(1, retry_limit + 2):
        result = perform_attempt(adapter, transport, key_loader, messages, max_tokens, purpose, attempt, price, paths, ledger, cap_usd)
        response = result["transport_response"]
        retryable = response.network_error is not None or response.http_status in RETRYABLE_STATUSES
        if not retryable:
            if response.http_status != 200:
                raise ProviderAPIError({
                    "provider_api_error": "non-200 response after retry policy",
                    "provider": adapter.provider,
                    "model": adapter.model,
                    "purpose": purpose,
                    "http_status": response.http_status,
                    "response_body_sha256": result["transport_event"]["response_body_sha256"],
                    "attempt": attempt,
                })
            return result
        last = result
        if attempt <= retry_limit:
            delay = retry_delay_seconds(attempt)
            append_jsonl(paths["runtime"], [runtime_event("retry_backoff", adapter.provider, adapter.model, {
                "purpose": purpose, "attempt": attempt, "delay_seconds": delay,
                "http_status": response.http_status, "network_error": response.network_error,
            })])
            sleep_func(delay)
    response = last["transport_response"]
    raise ProviderAPIError({
        "provider_api_error": "retry limit exhausted",
        "provider": adapter.provider,
        "model": adapter.model,
        "purpose": purpose,
        "http_status": response.http_status,
        "network_error": response.network_error,
        "response_body_sha256": last["transport_event"]["response_body_sha256"],
    })


def scientific_raw_row(adapter, probe: dict, result: dict, key: str, batch_id: str, batch_elapsed: float) -> dict:
    prepared = result["prepared"]
    adapted = result["adapted"]
    event = result["transport_event"]
    output_tokens = adapted.output_tokens if adapted.output_tokens is not None else prepared.max_tokens
    row = {
        "status": "ok",
        "completion_key": key,
        "carrier_id": probe["carrier_id"],
        "probe_key": probe["probe_key"],
        "probe_type": probe["probe_type"],
        "probe_scope": probe["probe_scope"],
        "batch_id": batch_id,
        "batch_elapsed_seconds": batch_elapsed,
        "raw_text": adapted.text,
        "output_token_count": output_tokens,
        "output_token_count_basis": "provider-reported completion/output tokens" if adapted.output_tokens is not None else "max_tokens conservative placeholder because usage was missing",
        "finish_reason_inferred": adapted.finish_reason or "missing_finish_reason",
        "hf_repo": f"hosted:{adapter.provider}",
        "model_revision": adapter.model,
        "upstream_repo": prepared.url,
        "upstream_revision": adapted.served_model,
        "served_model": adapted.served_model,
        "served_model_matches_requested": adapted.served_model == adapter.model,
        "dtype": "hosted_api_unspecified",
        "quantization_class": "hosted_api_unspecified",
        "decode_params": prepared.decode_params,
        "host_python_version": platform.python_version(),
        "host_unicodedata_version": unicodedata.unidata_version,
        "prompt_sha256": sha_text(canon(prepared.logical_messages)),
        "user_content_sha256": probe["user_content_sha256"],
        "request_body_sha256": event["request_body_sha256"],
        "response_body_sha256": event["response_body_sha256"],
        "request_body_base64": event["request_body_base64"],
        "response_body_base64": event["response_body_base64"],
        "transport_event_sha256": event["event_sha256"],
        "http_status": result["transport_response"].http_status,
        "latency_seconds": result["transport_response"].latency_seconds,
        "usage": {
            "input_tokens": adapted.input_tokens,
            "output_tokens": adapted.output_tokens,
            "total_tokens": adapted.total_tokens,
            "usage_present": adapted.usage_present,
            "integrity_notes": adapted.integrity_notes,
        },
        "hosted_reference_carrier_id": probe["hosted_reference_carrier_id"],
        "reference_probe_key": probe["reference_probe_key"],
    }
    row["raw_row_sha256"] = sha_text(canon(row))
    return row


def score_current(plan_rows: list[dict], raw_rows: list[dict], scored_path: Path) -> list[dict]:
    lookup = {(x["carrier_id"], x["probe_key"]): x for x in plan_rows}
    scored = []
    missing = []
    for raw in raw_rows:
        if raw.get("status") != "ok":
            continue
        probe = lookup.get((raw["carrier_id"], raw["probe_key"]))
        if probe is None:
            missing.append(raw["completion_key"])
        else:
            scored.append(score_row(raw, probe))
    if missing:
        raise HostedFrontierError(f"raw rows without hosted probe definitions: {missing[:5]}")
    write_jsonl(scored_path, scored)
    return scored


def compute_hosted_gates(scored: list[dict], plan_rows: list[dict]) -> dict:
    carrier = plan_rows[0]["carrier_id"]
    rows = [x for x in scored if x["carrier_id"] == carrier]
    ascii_copy = [x for x in rows if x["probe_type"] == "copy" and x["probe_scope"] == "ascii"]
    all_copy = [x for x in rows if x["probe_type"] == "copy"]
    ascii_count = [x for x in rows if x["probe_type"] == "count" and x["probe_scope"] == "ascii"]
    ascii_scalar = [x for x in rows if x["probe_type"] == "scalar_listing" and x["probe_scope"] == "ascii"]
    ascii_copy_exact = sum(bool(x["task_exact"]) for x in ascii_copy)
    copy_format_failures = sum(bool(x["format_failure"]) for x in all_copy)
    ascii_count_exact = sum(bool(x["task_exact"]) for x in ascii_count)
    ascii_scalar_exact = sum(bool(x["task_exact"]) for x in ascii_scalar)
    copy_complete = len(all_copy) == 124
    copy_valid = copy_complete and ascii_copy_exact >= 23 and copy_format_failures <= 6
    structural_ascii_valid = ascii_count_exact >= 11 and ascii_scalar_exact >= 11
    mutation_checks = {
        "ascii_copy_22_of_24_fails": (22 >= 23) is False,
        "ascii_copy_23_of_24_passes": (23 >= 23) is True,
        "seven_wrapper_failures_fail": (7 <= 6) is False,
        "six_wrapper_failures_pass": (6 <= 6) is True,
        "ascii_structural_10_of_12_fails": (10 >= 11) is False,
        "ascii_structural_11_of_12_passes": (11 >= 11) is True,
    }
    return {
        "created_utc": utcnow(),
        "carrier_id": carrier,
        "ascii_copy_rows": len(ascii_copy),
        "ascii_copy_exact": ascii_copy_exact,
        "ascii_copy_gate_pass": ascii_copy_exact >= 23,
        "ascii_count_rows": len(ascii_count),
        "ascii_count_exact": ascii_count_exact,
        "ascii_count_gate_pass": ascii_count_exact >= 11,
        "ascii_scalar_rows": len(ascii_scalar),
        "ascii_scalar_exact": ascii_scalar_exact,
        "ascii_scalar_gate_pass": ascii_scalar_exact >= 11,
        "ascii_structural_valid": structural_ascii_valid,
        "ascii_gate_battery_pass": ascii_copy_exact >= 23 and structural_ascii_valid,
        "copy_rows": len(all_copy),
        "copy_complete": copy_complete,
        "copy_wrapper_format_failures": copy_format_failures,
        "copy_wrapper_gate_pass_at_most_6": copy_format_failures <= 6,
        "copy_valid": copy_valid,
        "structural_valid": copy_valid and structural_ascii_valid,
        "gate_thresholds": {
            "ascii_copy_exact_minimum": "23/24",
            "copy_wrapper_format_failures_maximum": "6/124",
            "ascii_count_exact_minimum": "11/12",
            "ascii_scalar_exact_minimum": "11/12",
        },
        "gate_mutation_checks": mutation_checks,
        "gate_mutation_checks_pass": all(mutation_checks.values()),
    }


def phase_rows(rows: list[dict], phase: str) -> list[dict]:
    return [x for x in rows if phase_for_probe(x) == phase]


def chunks(rows: list[dict], n: int):
    for i in range(0, len(rows), n):
        yield i // n, rows[i:i + n]


def execute_phase(phase: str, rows: list[dict], adapter, transport, key_loader, price, paths, ledger, config: HarnessConfig, inventory_hash: str, existing_keys: set[str], sleep_func) -> dict:
    selected = []
    skipped = 0
    for row in rows:
        key = completion_key(adapter, row, inventory_hash)
        if key in existing_keys:
            skipped += 1
        else:
            selected.append((row, key))
    generated = 0
    for batch_index, batch in chunks(selected, config.batch_size):
        projections = [projection_for_messages(price, cf.prompt_messages(row), MAX_TOKENS) for row, _ in batch]
        projection = sum_projections(projections)
        ensure_budget(ledger, projection, config.cap_usd, adapter.provider, adapter.model, phase)
        batch_id = f"hosted-{adapter.provider}-{model_slug(adapter.model)}-{phase}-batch{batch_index:03d}"
        append_jsonl(paths["runtime"], [runtime_event("logical_batch_started", adapter.provider, adapter.model, {
            "phase": phase, "batch_id": batch_id, "rows": len(batch), "projection": projection,
        })])
        tic = time.perf_counter()
        batch_rows = []
        for row, key in batch:
            messages = cf.prompt_messages(row)
            result = call_with_retries(
                adapter, transport, key_loader, messages, MAX_TOKENS, phase, price, paths, ledger,
                config.cap_usd, config.retry_limit, sleep_func,
            )
            adapted = result["adapted"]
            if adapted.served_model != adapter.model:
                raise ServedModelMismatch(json.dumps({
                    "requested_model": adapter.model,
                    "served_model": adapted.served_model,
                    "probe_key": row["probe_key"],
                    "response_body_sha256": result["transport_event"]["response_body_sha256"],
                }, sort_keys=True))
            raw = scientific_raw_row(adapter, row, result, key, batch_id, time.perf_counter() - tic)
            append_jsonl(paths["raw"], [raw])
            batch_rows.append(raw)
            existing_keys.add(key)
            generated += 1
        append_jsonl(paths["runtime"], [runtime_event("logical_batch_completed", adapter.provider, adapter.model, {
            "phase": phase, "batch_id": batch_id, "rows": len(batch_rows), "batch_elapsed_seconds": time.perf_counter() - tic,
        })])
    return {"phase": phase, "already_complete": skipped, "generated": generated, "remaining_before_phase": len(selected)}


def initial_manifest(config: HarnessConfig, adapter, price: dict, inventory_manifest: dict, paths: dict[str, Path]) -> dict:
    return {
        "created_utc": utcnow(),
        "provider": config.provider,
        "model": config.model,
        "adapter_id": adapter.adapter_id,
        "endpoint": adapter.endpoint,
        "dry_run": config.dry_run,
        "reference_carrier_id": config.reference_carrier,
        "inventory": inventory_manifest,
        "cap_usd": config.cap_usd,
        "pricing": price,
        "pricing_table_sha256": pricing_table_sha256(),
        "artifact_paths": {k: str(v) for k, v in paths.items()},
        "key_handling": "key read from environment or ~/.secrets immediately before each call; never written to artifacts or logs",
        "ascii_first_gate_order": [
            "24 ASCII copy controls",
            "12 ASCII scalar-listing controls",
            "12 ASCII count controls",
            "only then 100 Unicode copy probes",
            "only after copy validity, Unicode scalar-listing and count probes",
        ],
        "fragmentation_covariate_caveat": inventory_manifest["fragmentation_covariate_caveat"],
    }


def run_harness(config: HarnessConfig, transport=None, key_loader=None, sleep_func=None) -> dict:
    if config.provider not in DEFAULT_MODELS:
        raise HostedFrontierError(f"unsupported provider {config.provider!r}")
    if config.cap_usd <= 0:
        raise HostedFrontierError("cap_usd must be positive")
    if config.batch_size <= 0:
        raise HostedFrontierError("batch_size must be positive")
    if config.artifact_root is not None:
        root = config.artifact_root
    elif config.dry_run:
        stamp = utcnow().replace("-", "").replace(":", "").replace("+", "_")
        root = DRY_RUN_ROOT / stamp
    else:
        root = HOSTED
    if config.dry_run:
        try:
            root.resolve().relative_to(HOSTED.resolve())
            inside_results = True
        except ValueError:
            inside_results = False
        if inside_results:
            raise HostedFrontierError("dry-run fake artifacts may not be written under results/hosted_frontier")
    paths = artifact_paths(root, config.provider, config.model, config.reference_carrier)
    adapter = adapter_for(config.provider, config.model)
    price = price_for(config.provider, config.model)
    rows, inventory_manifest = transform_plan(config.provider, config.model, config.reference_carrier)
    rows = phase_ordered_rows(rows)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["plan"], rows)
    manifest = initial_manifest(config, adapter, price, inventory_manifest, paths)
    write_json(paths["manifest"], manifest)
    ledger = HashChainLedger(paths["ledger"])
    if transport is None:
        transport = CorrectFakeTransport(config.provider) if config.dry_run else UrllibTransport(config.timeout_seconds)
    if key_loader is None:
        key_loader = (lambda: "dry-run-key-not-a-credential") if config.dry_run else (lambda: load_api_key(config.provider))
    if sleep_func is None:
        sleep_func = (lambda seconds: None) if config.dry_run else time.sleep
    if not config.dry_run:
        key_loader()
    append_jsonl(paths["runtime"], [runtime_event("run_started", config.provider, config.model, {
        "dry_run": config.dry_run, "cap_usd": config.cap_usd, "reference_carrier": config.reference_carrier,
        "probe_rows": len(rows), "inventory_identity_sha256": inventory_manifest["hosted_probe_identity_sha256"],
    })])
    raw_rows = read_jsonl(paths["raw"])
    keys = [x["completion_key"] for x in raw_rows if x.get("status") == "ok"]
    if len(keys) != len(set(keys)):
        raise HostedFrontierError("duplicate completion keys in append-only hosted raw file")
    probe_pairs = [(x.get("carrier_id"), x.get("probe_key")) for x in raw_rows if x.get("status") == "ok"]
    if len(probe_pairs) != len(set(probe_pairs)):
        raise HostedFrontierError("duplicate carrier/probe keys in append-only hosted raw file")
    existing_keys = set(keys)
    remaining = [x for x in rows if completion_key(adapter, x, inventory_manifest["hosted_probe_identity_sha256"]) not in existing_keys]
    projections = [projection_for_messages(price, cf.prompt_messages(x), MAX_TOKENS) for x in remaining]
    preflight_projection = projection_for_messages(price, adapter.preflight_messages(), MAX_TOKENS)
    full_projection = sum_projections([preflight_projection] + projections)
    write_json(paths["preflight"], {
        "created_utc": utcnow(),
        "status": "cost_projection_before_transport",
        "provider": config.provider,
        "model": config.model,
        "remaining_probe_rows": len(remaining),
        "full_worst_case_projection_including_availability_call": full_projection,
        "ledger_totals_before_transport": ledger.totals(),
        "cap_usd": config.cap_usd,
    })
    ensure_budget(ledger, full_projection, config.cap_usd, config.provider, config.model, "initial_full_battery_preflight")
    preflight_result = call_with_retries(
        adapter, transport, key_loader, adapter.preflight_messages(), MAX_TOKENS, "availability_preflight", price, paths, ledger,
        config.cap_usd, config.retry_limit, sleep_func,
    )
    preflight_adapted = preflight_result["adapted"]
    served_match = preflight_adapted.served_model == config.model
    preflight_record = {
        "created_utc": utcnow(),
        "status": "pass" if served_match else "fail",
        "provider": config.provider,
        "requested_model": config.model,
        "served_model": preflight_adapted.served_model,
        "served_model_matches_requested": served_match,
        "response_text": preflight_adapted.text,
        "latency_seconds": preflight_result["transport_response"].latency_seconds,
        "usage": {
            "input_tokens": preflight_adapted.input_tokens,
            "output_tokens": preflight_adapted.output_tokens,
            "total_tokens": preflight_adapted.total_tokens,
            "usage_present": preflight_adapted.usage_present,
        },
        "request_body_sha256": preflight_result["transport_event"]["request_body_sha256"],
        "response_body_sha256": preflight_result["transport_event"]["response_body_sha256"],
        "cap_usd": config.cap_usd,
        "ledger_totals_after_availability_call": ledger.totals(),
    }
    write_json(paths["preflight"], preflight_record)
    if not served_match:
        raise ServedModelMismatch(json.dumps({"requested_model": config.model, "served_model": preflight_adapted.served_model}, sort_keys=True))
    phase_summaries = []
    stopped_by_gate = None
    scored = []
    gates = None
    for phase in PHASES:
        rows_phase = phase_rows(rows, phase)
        summary = execute_phase(
            phase, rows_phase, adapter, transport, key_loader, price, paths, ledger, config,
            inventory_manifest["hosted_probe_identity_sha256"], existing_keys, sleep_func,
        )
        phase_summaries.append(summary)
        raw_rows = read_jsonl(paths["raw"])
        scored = score_current(rows, raw_rows, paths["scored"])
        gates = compute_hosted_gates(scored, rows)
        gates["phases_completed"] = [x["phase"] for x in phase_summaries]
        write_json(paths["gates"], gates)
        if phase == "ascii_gate" and not gates["ascii_gate_battery_pass"]:
            stopped_by_gate = "ascii_gate_failed; Unicode spend refused"
            break
        if phase == "unicode_copy_gate" and not gates["copy_valid"]:
            stopped_by_gate = "copy_validity_failed_after_unicode_copy; Unicode structural spend refused"
            break
    mutation = extractor_mutation_checks()
    raw_rows = read_jsonl(paths["raw"])
    scored = score_current(rows, raw_rows, paths["scored"])
    gates = compute_hosted_gates(scored, rows)
    gates["phases_completed"] = [x["phase"] for x in phase_summaries]
    write_json(paths["gates"], gates)
    ledger_validation = validate_ledger_rows(ledger.rows)
    archive_validation = verify_archive_rows(read_jsonl(paths["transport"]))
    if not ledger_validation["chain_valid"]:
        raise HostedFrontierError(json.dumps({"ledger_integrity_failure_after_run": ledger_validation["problems"][:5]}, sort_keys=True))
    if not archive_validation["archive_valid"]:
        raise HostedFrontierError(json.dumps({"transport_archive_integrity_failure_after_run": archive_validation["problems"][:5]}, sort_keys=True))
    status = "completed" if stopped_by_gate is None else "stopped_by_gate"
    outcome = {
        "status": status,
        "stopped_by_gate": stopped_by_gate,
        "provider": config.provider,
        "model": config.model,
        "dry_run": config.dry_run,
        "reference_carrier": config.reference_carrier,
        "planned_rows": len(rows),
        "raw_rows": len(raw_rows),
        "scored_rows": len(scored),
        "phase_summaries": phase_summaries,
        "gates": gates,
        "ledger_totals": ledger.totals(),
        "ledger_validation": ledger_validation,
        "transport_archive_validation": archive_validation,
        "scorer_extractor_mutation_checks_pass": mutation["extractor_mutation_checks_pass"],
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    manifest["final_outcome"] = {k: outcome[k] for k in ("status", "stopped_by_gate", "raw_rows", "scored_rows", "ledger_totals")}
    manifest["completed_utc"] = utcnow()
    write_json(paths["manifest"], manifest)
    append_jsonl(paths["runtime"], [runtime_event("run_finished", config.provider, config.model, {
        "status": status, "stopped_by_gate": stopped_by_gate, "raw_rows": len(raw_rows), "scored_rows": len(scored),
        "ledger_totals": ledger.totals(),
    })])
    return outcome


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the frozen confirmatory structural battery against a first-party hosted frontier endpoint.")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), required=True)
    parser.add_argument("--model", default=None, help="pinned provider model identifier; default is the committed flagship snapshot")
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    parser.add_argument("--dry-run", action="store_true", help="use the correct-response fake transport; no key or network required")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--reference-carrier", default=DEFAULT_REFERENCE_CARRIER)
    parser.add_argument("--batch-size", type=int, default=LOGICAL_BATCH_SIZE)
    parser.add_argument("--retry-limit", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model = args.model or DEFAULT_MODELS[args.provider]
    config = HarnessConfig(
        provider=args.provider,
        model=model,
        cap_usd=args.cap_usd,
        dry_run=args.dry_run,
        artifact_root=args.artifact_root,
        reference_carrier=args.reference_carrier,
        batch_size=args.batch_size,
        retry_limit=args.retry_limit,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        outcome = run_harness(config)
    except BudgetAbort as exc:
        print(json.dumps({"status": "budget_abort", **exc.detail}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (MissingApiKey, ProviderAPIError, ResponseIntegrityError, ServedModelMismatch, HostedFrontierError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(outcome, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
