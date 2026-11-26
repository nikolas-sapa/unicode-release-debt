from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
HOSTED = RESULTS / "hosted_frontier"
DRY_RUN_ROOT = ROOT / "out" / "hosted_frontier_dry_run"
DEFAULT_CAP_USD = 4.0
MAX_TOKENS = 64
LOGICAL_BATCH_SIZE = 8
GENESIS_HASH = "0" * 64
USER_AGENT = "unicode-release-debt-hosted-frontier/1.0"

KEY_SOURCES = {
    "openai": {
        "env": ["OPENAI_API_KEY"],
        "files": ["openai-api-key", "openai"],
    },
    "anthropic": {
        "env": ["ANTHROPIC_API_KEY"],
        "files": ["anthropic-api-key", "anthropic", "claude-api-key"],
    },
    "gemini": {
        "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "files": ["gemini-api-key", "google-api-key", "gemini"],
    },
}


class HostedFrontierError(RuntimeError):
    pass


class MissingApiKey(HostedFrontierError):
    pass


class BudgetAbort(HostedFrontierError):
    def __init__(self, detail: dict):
        super().__init__(json.dumps(detail, sort_keys=True))
        self.detail = detail


class ProviderAPIError(HostedFrontierError):
    def __init__(self, detail: dict):
        super().__init__(json.dumps(detail, sort_keys=True, ensure_ascii=False))
        self.detail = detail


class ResponseIntegrityError(HostedFrontierError):
    pass


class ServedModelMismatch(HostedFrontierError):
    pass


@dataclass(frozen=True)
class PreparedRequest:
    provider: str
    model: str
    url: str
    method: str
    headers: dict[str, str]
    body_bytes: bytes
    body_obj: dict
    decode_params: dict
    max_tokens: int
    logical_messages: list[dict]
    auth_header_name: str
    auth_header_prefix: str = ""

    def headers_with_auth(self, api_key: str) -> dict[str, str]:
        headers = dict(self.headers)
        headers[self.auth_header_name] = self.auth_header_prefix + api_key
        return headers


@dataclass(frozen=True)
class TransportResponse:
    http_status: int | None
    headers: dict[str, str]
    body: bytes
    latency_seconds: float
    network_error: str | None = None


@dataclass(frozen=True)
class AdaptedResponse:
    text: str
    served_model: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_present: bool
    parsed: dict
    integrity_notes: list[str]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


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
                    raise HostedFrontierError(f"corrupt jsonl at {path}:{line_no}: {exc}") from exc
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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def model_slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in model)


def load_api_key(provider: str) -> str:
    if provider not in KEY_SOURCES:
        raise MissingApiKey(f"unknown provider {provider!r}")
    cfg = KEY_SOURCES[provider]
    for name in cfg["env"]:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    secret_dir = Path.home() / ".secrets"
    for filename in cfg["files"]:
        path = secret_dir / filename
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    names = ", ".join(cfg["env"])
    files = ", ".join("~/.secrets/" + x for x in cfg["files"])
    raise MissingApiKey(f"missing {provider} API key; set one of {names} or create one of {files}")


def redact_secret(data: bytes, api_key: str | None) -> tuple[bytes, bool]:
    if not api_key:
        return data, False
    secret = api_key.encode("utf-8")
    if secret and secret in data:
        return data.replace(secret, b"[REDACTED_SECRET]"), True
    return data, False


def selected_response_headers(headers) -> dict[str, str]:
    wanted = ("content-type", "x-request-id", "request-id", "date", "anthropic-request-id", "x-goog-request-id")
    out = {}
    if headers:
        for name in wanted:
            value = headers.get(name)
            if value:
                out[name] = str(value)
    return out


class UrllibTransport:
    def __init__(self, timeout_seconds: int = 180):
        self.timeout_seconds = timeout_seconds

    def send(self, prepared: PreparedRequest, api_key: str) -> TransportResponse:
        req = urllib.request.Request(
            prepared.url,
            data=prepared.body_bytes,
            method=prepared.method,
            headers=prepared.headers_with_auth(api_key),
        )
        tic = time.perf_counter()
        status = None
        headers = {}
        body = b""
        network_error = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                status = response.status
                headers = selected_response_headers(response.headers)
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            headers = selected_response_headers(exc.headers)
            body = exc.read()
        except Exception as exc:
            network_error = type(exc).__name__ + ": " + str(exc)[:240]
        return TransportResponse(status, headers, body, time.perf_counter() - tic, network_error)


def input_token_upper_bound(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages) + 16 * len(messages) + 64


def token_cost_usd(input_tokens: int, output_tokens: int, input_per_million: float, output_per_million: float) -> float:
    return input_tokens * input_per_million / 1_000_000 + output_tokens * output_per_million / 1_000_000


def projection_for_messages(price: dict, messages: list[dict], max_tokens: int) -> dict:
    input_upper = input_token_upper_bound(messages)
    output_upper = max_tokens
    cost = token_cost_usd(input_upper, output_upper, price["input_per_million_usd"], price["output_per_million_usd"])
    return {
        "input_tokens_upper_bound": input_upper,
        "output_tokens_upper_bound": output_upper,
        "projected_conservative_cost_usd": cost,
        "cost_bound_method": "UTF-8 byte count plus fixed chat-message overhead for input, max_tokens for output, priced at the committed list-price table; hidden provider instructions are unobservable and reported usage overrides this bound when larger",
    }


def sum_projections(parts: list[dict]) -> dict:
    if not parts:
        return {"input_tokens_upper_bound": 0, "output_tokens_upper_bound": 0, "projected_conservative_cost_usd": 0.0, "cost_bound_method": "empty projection"}
    return {
        "input_tokens_upper_bound": sum(x["input_tokens_upper_bound"] for x in parts),
        "output_tokens_upper_bound": sum(x["output_tokens_upper_bound"] for x in parts),
        "projected_conservative_cost_usd": sum(x["projected_conservative_cost_usd"] for x in parts),
        "cost_bound_method": "sum of per-call conservative projections",
    }


def pricing_table_path() -> Path:
    return Path(__file__).with_name("hosted_frontier_pricing.json")


def pricing_table() -> dict:
    return json.loads(pricing_table_path().read_text(encoding="utf-8"))


def pricing_table_sha256() -> str:
    return sha_bytes(pricing_table_path().read_bytes())


def price_for(provider: str, model: str) -> dict:
    key = f"{provider}/{model}"
    row = pricing_table()["models"].get(key)
    if row is None:
        raise HostedFrontierError(f"no committed pricing-table entry for {key}; refusing unpriced spend")
    return row


def ledger_hash(row: dict) -> str:
    stripped = {k: v for k, v in row.items() if k != "entry_sha256"}
    return sha_text(canon(stripped))


def validate_ledger_rows(rows: list[dict]) -> dict:
    problems = []
    previous = GENESIS_HASH
    for index, row in enumerate(rows, 1):
        if row.get("sequence") != index:
            problems.append({"sequence": index, "problem": "sequence mismatch", "actual": row.get("sequence")})
        if row.get("previous_entry_sha256") != previous:
            problems.append({"sequence": index, "problem": "previous hash mismatch"})
        actual_hash = row.get("entry_sha256")
        computed = ledger_hash(row)
        if actual_hash != computed:
            problems.append({"sequence": index, "problem": "entry hash mismatch", "actual": actual_hash, "computed": computed})
        previous = actual_hash or previous
    return {"rows": len(rows), "problems": problems, "chain_valid": not problems}


def ledger_totals_from_rows(rows: list[dict]) -> dict:
    api_rows = [x for x in rows if x.get("event") == "api_call"]
    return {
        "api_calls": len(api_rows),
        "priced_usage_cost_usd": sum(float(x.get("priced_usage_cost_usd") or 0.0) for x in api_rows if x.get("priced_usage_cost_usd") is not None),
        "conservative_cap_debit_usd": sum(float(x.get("conservative_cap_debit_usd") or 0.0) for x in api_rows),
    }


class HashChainLedger:
    def __init__(self, path: Path):
        self.path = path
        self.rows = read_jsonl(path)
        validation = validate_ledger_rows(self.rows)
        if not validation["chain_valid"]:
            raise HostedFrontierError(json.dumps({"ledger_integrity_failure": validation["problems"][:5]}, sort_keys=True))

    def totals(self) -> dict:
        return ledger_totals_from_rows(self.rows)

    def append(self, event: dict) -> dict:
        if "sequence" in event or "previous_entry_sha256" in event or "entry_sha256" in event:
            raise HostedFrontierError("ledger callers may not supply chain fields")
        row = dict(event)
        row["sequence"] = len(self.rows) + 1
        row["previous_entry_sha256"] = self.rows[-1]["entry_sha256"] if self.rows else GENESIS_HASH
        row["entry_sha256"] = ledger_hash(row)
        append_jsonl(self.path, [row])
        self.rows.append(row)
        return row


def validate_ledger_file(path: Path) -> dict:
    return validate_ledger_rows(read_jsonl(path))


def ensure_budget(ledger: HashChainLedger, projection: dict, cap_usd: float, provider: str, model: str | None, purpose: str) -> dict:
    totals = ledger.totals()
    if totals["conservative_cap_debit_usd"] + projection["projected_conservative_cost_usd"] > cap_usd + 1e-12:
        event = {
            "event": "budget_abort",
            "utc": utcnow(),
            "provider": provider,
            "model": model,
            "purpose": purpose,
            "cap_usd": cap_usd,
            "cumulative_conservative_cap_debit_usd": totals["conservative_cap_debit_usd"],
            **projection,
        }
        ledger.append(event)
        detail = {
            "budget_abort": "projected conservative cost would exceed hard cap before transport",
            "provider": provider,
            "model": model,
            "purpose": purpose,
            "cap_usd": cap_usd,
            "cumulative_conservative_cap_debit_usd": totals["conservative_cap_debit_usd"],
            "projection": projection,
        }
        raise BudgetAbort(detail)
    return projection


def transport_event_hash(row: dict) -> str:
    stripped = {k: v for k, v in row.items() if k != "event_sha256"}
    return sha_text(canon(stripped))


def verify_archive_rows(rows: list[dict]) -> dict:
    problems = []
    for index, row in enumerate(rows, 1):
        for prefix in ("request", "response"):
            encoded = row.get(f"{prefix}_body_base64")
            expected = row.get(f"{prefix}_body_sha256")
            if encoded is None or expected is None:
                problems.append({"row": index, "problem": f"missing {prefix} archive fields"})
                continue
            try:
                data = b64d(encoded)
            except Exception:
                problems.append({"row": index, "problem": f"{prefix} body base64 does not decode"})
                continue
            if sha_bytes(data) != expected:
                problems.append({"row": index, "problem": f"{prefix} body hash mismatch"})
        if row.get("event_sha256") != transport_event_hash(row):
            problems.append({"row": index, "problem": "transport event hash mismatch"})
    return {"rows": len(rows), "problems": problems, "archive_valid": not problems}
