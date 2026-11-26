from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hosted_frontier_adapters import DEFAULT_MODELS
from hosted_frontier_common import (
    HOSTED,
    ROOT,
    BudgetAbort,
    ResponseIntegrityError,
    ServedModelMismatch,
    b64d,
    read_jsonl,
    sha_bytes,
    transport_event_hash,
    utcnow,
    validate_ledger_file,
    validate_ledger_rows,
    verify_archive_rows,
    write_json,
)
from hosted_frontier_fake import CorrectFakeTransport
from hosted_frontier_plan import DEFAULT_REFERENCE_CARRIER
from run_hosted_frontier import HarnessConfig, artifact_paths, run_harness

NO_SLEEP = lambda seconds: None


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def validation_stamp() -> str:
    return utcnow().replace("-", "").replace(":", "").replace("+", "_")


def scan_for_secret(root: Path, secret: str) -> dict:
    needle = secret.encode("utf-8")
    matches = []
    files = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                files += 1
                if needle in path.read_bytes():
                    matches.append(rel(path))
    return {"files_scanned": files, "secret_matches": matches, "secret_absent": not matches}


def payload_shape_checks(provider: str, fake: CorrectFakeTransport) -> dict:
    if len(fake.captured) < 2:
        return {"pass": False, "problem": "fewer than two captured calls"}
    payload = fake.captured[1].payload_obj
    if provider == "openai":
        checks = {
            "chat_completions_shape": isinstance(payload.get("messages"), list) and payload.get("model") == DEFAULT_MODELS[provider],
            "temperature_zero": payload.get("temperature") == 0.0,
            "max_completion_tokens_64": payload.get("max_completion_tokens") == 64,
            "reasoning_effort_none": payload.get("reasoning_effort") == "none",
            "no_system_message": not any(m.get("role") == "system" for m in payload["messages"]),
        }
    elif provider == "anthropic":
        checks = {
            "messages_shape": isinstance(payload.get("messages"), list) and payload.get("model") == DEFAULT_MODELS[provider],
            "temperature_zero": payload.get("temperature") == 0.0,
            "max_tokens_64": payload.get("max_tokens") == 64,
            "text_block_shape": all(isinstance(m.get("content"), list) and all(b.get("type") == "text" for b in m["content"]) for m in payload["messages"]),
            "no_system_field": "system" not in payload,
        }
    elif provider == "gemini":
        generation = payload.get("generationConfig", {})
        checks = {
            "contents_shape": isinstance(payload.get("contents"), list),
            "temperature_zero": generation.get("temperature") == 0.0,
            "max_output_tokens_64": generation.get("maxOutputTokens") == 64,
            "candidate_count_1": generation.get("candidateCount") == 1,
            "thinking_low_no_thoughts": generation.get("thinkingConfig") == {"thinkingLevel": "LOW", "includeThoughts": False},
            "assistant_mapped_to_model_role": any(x.get("role") == "model" for x in payload["contents"]),
            "no_system_instruction": "systemInstruction" not in payload,
        }
    else:
        checks = {"unknown_provider": False}
    return {"pass": all(checks.values()), "checks": checks}


def exact_capture_checks(fake: CorrectFakeTransport, transport_path: Path) -> dict:
    events = read_jsonl(transport_path)
    by_request_hash = {x["request_body_sha256"]: x for x in events}
    missing = []
    mismatched = []
    for call in fake.captured:
        event = by_request_hash.get(sha_bytes(call.body_bytes))
        if event is None:
            missing.append(call.call_index)
        elif b64d(event["request_body_base64"]) != call.body_bytes:
            mismatched.append(call.call_index)
    archive = verify_archive_rows(events)
    return {
        "captured_calls": len(fake.captured),
        "transport_events": len(events),
        "missing_archived_requests": missing,
        "mismatched_archived_request_bytes": mismatched,
        "archive_integrity": archive,
        "pass": not missing and not mismatched and archive["archive_valid"] and len(events) == len(fake.captured),
    }


def run_correct_e2e(provider: str, root: Path) -> dict:
    secret = f"validation-secret-{provider}-do-not-store"
    fake = CorrectFakeTransport(provider)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    outcome = run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    artifacts = {k: Path(v) for k, v in outcome["artifacts"].items()}
    scored = read_jsonl(artifacts["scored"])
    plan = read_jsonl(artifacts["plan"])
    first48 = plan[:48]
    ledger_validation = validate_ledger_file(artifacts["ledger"])
    capture = exact_capture_checks(fake, artifacts["transport"])
    secret_scan = scan_for_secret(root / provider, secret)
    payload = payload_shape_checks(provider, fake)
    checks = {
        "status_completed": outcome["status"] == "completed",
        "planned_rows_complete": outcome["raw_rows"] == outcome["planned_rows"] == len(scored),
        "all_fake_correct_responses_scored_exact": all(x["task_exact"] for x in scored),
        "ascii_first_48_rows": len(first48) == 48 and all(x["probe_scope"] == "ascii" for x in first48),
        "ascii_copy_first_24": all(x["probe_type"] == "copy" for x in first48[:24]),
        "ascii_gates_pass_exact_thresholds": outcome["gates"]["ascii_copy_exact"] == 24 and outcome["gates"]["ascii_count_exact"] == 12 and outcome["gates"]["ascii_scalar_exact"] == 12,
        "copy_valid_and_structural_valid": outcome["gates"]["copy_valid"] and outcome["gates"]["structural_valid"],
        "preflight_served_model_matches": json.loads(artifacts["preflight"].read_text())["served_model_matches_requested"] is True,
        "ledger_hash_chain_valid": ledger_validation["chain_valid"],
        "ledger_distinguishes_priced_and_conservative": outcome["ledger_totals"]["priced_usage_cost_usd"] < outcome["ledger_totals"]["conservative_cap_debit_usd"],
        "cap_not_exceeded": outcome["ledger_totals"]["conservative_cap_debit_usd"] <= config.cap_usd,
        "exact_payload_capture_and_archive": capture["pass"],
        "provider_payload_shape": payload["pass"],
        "secret_absent_from_artifacts": secret_scan["secret_absent"],
        "scorer_extractor_mutations_pass": outcome["scorer_extractor_mutation_checks_pass"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "rows": outcome["raw_rows"],
        "ascii_gate_counts": {
            "copy": outcome["gates"]["ascii_copy_exact"],
            "count": outcome["gates"]["ascii_count_exact"],
            "scalar": outcome["gates"]["ascii_scalar_exact"],
        },
        "ledger_totals": outcome["ledger_totals"],
        "payload_shape": payload,
        "capture": {k: capture[k] for k in ("captured_calls", "transport_events", "missing_archived_requests", "mismatched_archived_request_bytes")},
        "secret_scan": secret_scan,
        "artifact_dir": rel(artifacts["model_dir"]),
        "ledger_path": rel(artifacts["ledger"]),
    }


def run_corrupted_response_mutation(provider: str, root: Path) -> dict:
    secret = f"mutation-secret-{provider}-do-not-store"
    fake = CorrectFakeTransport(provider, corrupt_call_index=1)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    outcome = run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    scored = read_jsonl(Path(outcome["artifacts"]["scored"]))
    nonexact = [x for x in scored if not x["task_exact"]]
    first = scored[0]
    checks = {
        "run_completed_despite_response_content_mutation": outcome["status"] == "completed",
        "exactly_one_scored_row_became_nonexact": len(nonexact) == 1,
        "mutated_row_is_first_ascii_copy": first["probe_type"] == "copy" and first["probe_scope"] == "ascii" and not first["task_exact"],
        "ascii_copy_gate_degraded_to_23_of_24": outcome["gates"]["ascii_copy_exact"] == 23,
    }
    return {"pass": all(checks.values()), "checks": checks, "nonexact_probe_key": first["probe_key"], "nonexact_rows": len(nonexact)}


def run_malformed_payload_mutation(provider: str, root: Path) -> dict:
    secret = f"malformed-secret-{provider}-do-not-store"
    fake = CorrectFakeTransport(provider, malformed_call_index=1)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    error = None
    try:
        run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    except ResponseIntegrityError as exc:
        error = str(exc)
    paths = artifact_paths(root, provider, DEFAULT_MODELS[provider], DEFAULT_REFERENCE_CARRIER)
    events = read_jsonl(paths["transport"])
    raw_rows = read_jsonl(paths["raw"])
    checks = {
        "adapter_integrity_error_raised": error is not None,
        "malformed_transport_event_recorded": len(events) >= 2 and events[1]["integrity_status"] == "malformed_response",
        "no_scientific_raw_row_accepted": len(raw_rows) == 0,
        "transport_stopped_after_malformed_response": len(fake.captured) == 2,
    }
    return {"pass": all(checks.values()), "checks": checks, "error": error, "transport_events": len(events)}


def run_tiny_cap_mutation(provider: str, root: Path) -> dict:
    secret = f"tiny-cap-secret-{provider}-do-not-store"
    fake = CorrectFakeTransport(provider)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root, cap_usd=1e-9)
    detail = None
    try:
        run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    except BudgetAbort as exc:
        detail = exc.detail
    paths = artifact_paths(root, provider, DEFAULT_MODELS[provider], DEFAULT_REFERENCE_CARRIER)
    ledger_rows = read_jsonl(paths["ledger"])
    transport_rows = read_jsonl(paths["transport"])
    checks = {
        "budget_abort_raised": detail is not None,
        "zero_transport_attempts": len(fake.captured) == 0 and len(transport_rows) == 0,
        "budget_abort_logged_to_hash_chained_ledger": len(ledger_rows) == 1 and ledger_rows[0]["event"] == "budget_abort" and validate_ledger_rows(ledger_rows)["chain_valid"],
    }
    return {"pass": all(checks.values()), "checks": checks, "abort_detail": detail}


def run_rate_limit_retry_check(root: Path) -> dict:
    provider = "openai"
    secret = "retry-secret-openai-do-not-store"
    fake = CorrectFakeTransport(provider, rate_limit_call_indexes={1})
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    outcome = run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    artifacts = {k: Path(v) for k, v in outcome["artifacts"].items()}
    events = read_jsonl(artifacts["transport"])
    runtime = read_jsonl(artifacts["runtime"])
    checks = {
        "run_completed_after_one_429": outcome["status"] == "completed",
        "one_http_429_event": sum(x["http_status"] == 429 for x in events) == 1,
        "one_retry_backoff_event": sum(x["event"] == "retry_backoff" for x in runtime) == 1,
        "retry_attempt_archived": any(x["http_status"] == 429 and x["attempt"] == 1 for x in events),
        "all_rows_completed": outcome["raw_rows"] == outcome["planned_rows"],
        "extra_transport_attempt_is_in_cost_ledger": outcome["ledger_totals"]["api_calls"] == outcome["planned_rows"] + 2,
    }
    return {"pass": all(checks.values()), "checks": checks, "transport_events": len(events), "api_calls": outcome["ledger_totals"]["api_calls"]}


def run_secret_reflection_check(root: Path) -> dict:
    provider = "openai"
    secret = "reflected-validation-secret-do-not-store"
    fake = CorrectFakeTransport(provider, reflect_key_call_index=1)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    error = None
    try:
        run_harness(config, transport=fake, key_loader=lambda: secret, sleep_func=NO_SLEEP)
    except ResponseIntegrityError as exc:
        error = str(exc)
    paths = artifact_paths(root, provider, DEFAULT_MODELS[provider], DEFAULT_REFERENCE_CARRIER)
    scan = scan_for_secret(root / provider, secret)
    events = read_jsonl(paths["transport"])
    checks = {
        "secret_reflection_refused": error is not None,
        "reflection_event_recorded": len(events) >= 2 and events[1]["integrity_status"] == "secret_reflected_in_response",
        "secret_absent_from_all_artifacts": scan["secret_absent"],
    }
    return {"pass": all(checks.values()), "checks": checks, "secret_scan": scan, "error": error}


def run_served_model_mismatch_check(root: Path) -> dict:
    class WrongServedModelTransport(CorrectFakeTransport):
        def response_body(self, model: str, text: str, input_tokens: int, output_tokens: int) -> bytes:
            return super().response_body(model + "-unexpected", text, input_tokens, output_tokens)

    provider = "anthropic"
    fake = WrongServedModelTransport(provider)
    config = HarnessConfig(provider=provider, model=DEFAULT_MODELS[provider], dry_run=True, artifact_root=root)
    error = None
    try:
        run_harness(config, transport=fake, key_loader=lambda: "mismatch-secret", sleep_func=NO_SLEEP)
    except ServedModelMismatch as exc:
        error = str(exc)
    checks = {
        "served_model_mismatch_aborted": error is not None,
        "only_availability_call_was_made": len(fake.captured) == 1,
    }
    return {"pass": all(checks.values()), "checks": checks, "error": error}


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def mutation_artifact_checks(e2e_result: dict) -> dict:
    transport_path = resolve_repo_path(e2e_result["artifact_dir"]) / "transport_events.jsonl"
    ledger_path = resolve_repo_path(e2e_result["ledger_path"])
    ledger_rows = read_jsonl(ledger_path)
    mutated_ledger = copy.deepcopy(ledger_rows)
    if mutated_ledger:
        mutated_ledger[0]["conservative_cap_debit_usd"] = float(mutated_ledger[0].get("conservative_cap_debit_usd") or 0.0) + 0.01
    ledger_validation = validate_ledger_rows(mutated_ledger)
    events = read_jsonl(transport_path)
    mutated_event = copy.deepcopy(events[1])
    encoded = mutated_event["response_body_base64"]
    mutated_event["response_body_base64"] = ("A" if encoded[0] != "A" else "B") + encoded[1:]
    mutated_event["event_sha256"] = transport_event_hash(mutated_event)
    archive_validation = verify_archive_rows([mutated_event])
    checks = {
        "ledger_cost_mutation_detected": not ledger_validation["chain_valid"] and any(x["problem"] == "entry hash mismatch" for x in ledger_validation["problems"]),
        "archived_response_payload_mutation_detected": not archive_validation["archive_valid"] and any(x["problem"] == "response body hash mismatch" for x in archive_validation["problems"]),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "ledger_mutation_problems": ledger_validation["problems"][:2],
        "archive_mutation_problems": archive_validation["problems"][:2],
    }


def validate_all(base_root: Path | None = None, output_path: Path | None = None) -> dict:
    root = base_root or (ROOT / "out" / "hosted_frontier_validation" / validation_stamp())
    root.mkdir(parents=True, exist_ok=True)
    providers = {}
    for provider in sorted(DEFAULT_MODELS):
        providers[provider] = run_correct_e2e(provider, root / "e2e")
    corrupted = {}
    malformed = {}
    tiny_caps = {}
    for provider in sorted(DEFAULT_MODELS):
        corrupted[provider] = run_corrupted_response_mutation(provider, root / "mutation_corrupt_response")
        malformed[provider] = run_malformed_payload_mutation(provider, root / "mutation_malformed_payload")
        tiny_caps[provider] = run_tiny_cap_mutation(provider, root / "mutation_tiny_cap")
    artifact_mutations = mutation_artifact_checks(providers["openai"])
    retry = run_rate_limit_retry_check(root / "retry")
    secret_reflection = run_secret_reflection_check(root / "secret_reflection")
    served_model_mismatch = run_served_model_mismatch_check(root / "served_model_mismatch")
    components = {
        "fake_transport_end_to_end": providers,
        "corrupted_canned_response_scorer_mutations": corrupted,
        "corrupted_payload_adapter_mutations": malformed,
        "tiny_cap_aborts_before_transport": tiny_caps,
        "hash_pin_mutation_checks": artifact_mutations,
        "rate_limit_retry_with_backoff": retry,
        "secret_reflection_fail_closed": secret_reflection,
        "served_model_pin_fail_closed": served_model_mismatch,
    }

    def flatten_pass(obj):
        if isinstance(obj, dict) and set(obj) >= {"pass", "checks"}:
            return obj["pass"] is True
        if isinstance(obj, dict):
            return all(flatten_pass(x) for x in obj.values())
        return bool(obj)

    overall = all(flatten_pass(x) for x in components.values())
    result = {
        "created_utc": utcnow(),
        "validation_root": rel(root),
        "components": components,
        "overall_pass": overall,
        "verify_real": [
            f"fake transport end-to-end pass for providers: {sorted(DEFAULT_MODELS)}",
            "corrupted canned response mutation changed byte-exact scorer outcome for every provider",
            "corrupted payload bytes triggered adapter integrity failure before scientific-row acceptance for every provider",
            "tiny cap aborted before any transport attempt for every provider",
            "hash-chained cost-ledger mutation and archived-payload mutation were detected",
        ],
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate the hosted-frontier harness entirely against fake transports.")
    parser.add_argument("--output", type=Path, default=HOSTED / "harness_validation.json")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = validate_all(args.artifact_root, args.output)
    print(json.dumps({
        "overall_pass": result["overall_pass"],
        "output": rel(args.output),
        "validation_root": result["validation_root"],
        "providers": {k: v["pass"] for k, v in result["components"]["fake_transport_end_to_end"].items()},
        "mutations": {
            "corrupted_response": {k: v["pass"] for k, v in result["components"]["corrupted_canned_response_scorer_mutations"].items()},
            "corrupted_payload": {k: v["pass"] for k, v in result["components"]["corrupted_payload_adapter_mutations"].items()},
            "tiny_cap": {k: v["pass"] for k, v in result["components"]["tiny_cap_aborts_before_transport"].items()},
            "hash_pins": result["components"]["hash_pin_mutation_checks"]["pass"],
        },
    }, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
