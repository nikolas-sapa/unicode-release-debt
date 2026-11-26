from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vyce_common as vc

DIAGNOSTIC_PATH = vc.VYCE / "diagnostic_probe.jsonl"


def diagnostic_key(purpose: str, model: str, endpoint: str) -> str:
    return vc.sha_text("\0".join(["vyce-diagnostic-v1", purpose, model, endpoint]))


def append_once(row: dict):
    existing = vc.read_jsonl(DIAGNOSTIC_PATH)
    if row["diagnostic_key"] in {x.get("diagnostic_key") for x in existing}:
        return False
    vc.append_jsonl(DIAGNOSTIC_PATH, [row])
    return True


def record_reconstructed_www_404() -> dict:
    body_text = '{"error":{"message":"Not found","type":"not_found","param":null,"code":null}}'
    body = body_text.encode()
    ledger = [x for x in vc.read_jsonl(vc.LEDGER_PATH) if x.get("purpose") == "www_host_diagnostic"]
    if not ledger:
        return {"recorded": False, "reason": "www diagnostic ledger row not present"}
    row = ledger[-1]
    if row["response_body_sha256"] != vc.sha_bytes(body):
        raise RuntimeError("reconstructed www diagnostic body hash does not match ledger")
    out = {
        "diagnostic_key": diagnostic_key("www_host_diagnostic", "gpt-oss-120b", "https://www.vyceai.com/v1/chat/completions"),
        "utc": row["utc"], "purpose": "www_host_diagnostic", "model": "gpt-oss-120b",
        "endpoint": "https://www.vyceai.com/v1/chat/completions", "http_status": row["http_status"],
        "content": None, "response_body_text": body_text, "response_body_base64": base64.b64encode(body).decode("ascii"),
        "response_body_sha256": row["response_body_sha256"], "request_sha256": row["request_sha256"],
        "relay_artifacts": row["relay_artifacts"], "note": "body reconstructed and hash-verified against the cost-ledger row because this initial infrastructure diagnostic was logged before the diagnostic jsonl existed",
    }
    return {"recorded": append_once(out), "row": out}


def wave_control() -> dict:
    model = "wave"
    key = diagnostic_key("non_target_infrastructure_control", model, vc.ENDPOINT)
    existing = vc.read_jsonl(DIAGNOSTIC_PATH)
    if key in {x.get("diagnostic_key") for x in existing}:
        row = next(x for x in existing if x.get("diagnostic_key") == key)
        return {"recorded": False, "row": row}
    result = vc.post_chat_completion(model, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=4, temperature=0.0, purpose="non_target_infrastructure_control", timeout=90)
    row = {
        "diagnostic_key": key, "utc": result["finished_utc"], "purpose": "non_target_infrastructure_control", "model": model,
        "endpoint": vc.ENDPOINT, "http_status": result["http_status"], "content": result["content"], "finish_reason": result["finish_reason"],
        "response_body_text": result["body_text"], "response_body_base64": result["body_base64"], "response_body_sha256": result["body_sha256"],
        "request_sha256": result["request_sha256"], "usage": result["tokens"]["usage"], "input_tokens": result["tokens"]["input_tokens"], "output_tokens": result["tokens"]["output_tokens"],
        "relay_artifacts": result["relay_artifacts"], "returned_model": result["parsed"].get("model") if isinstance(result["parsed"], dict) else None,
        "note": "non-target control only: establishes whether the key/relay can complete any chat request; not part of the scientific battery",
    }
    return {"recorded": append_once(row), "row": row}


def alias_probe(model: str) -> dict:
    key = diagnostic_key("alias_model_diagnostic", model, vc.ENDPOINT)
    existing = vc.read_jsonl(DIAGNOSTIC_PATH)
    if key in {x.get("diagnostic_key") for x in existing}:
        row = next(x for x in existing if x.get("diagnostic_key") == key)
        return {"recorded": False, "row": row}
    result = vc.post_chat_completion(model, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=4, temperature=0.0, purpose="alias_model_diagnostic", timeout=90)
    row = {
        "diagnostic_key": key, "utc": result["finished_utc"], "purpose": "alias_model_diagnostic", "model": model,
        "endpoint": vc.ENDPOINT, "http_status": result["http_status"], "content": result["content"], "finish_reason": result["finish_reason"],
        "response_body_text": result["body_text"], "response_body_base64": result["body_base64"], "response_body_sha256": result["body_sha256"],
        "request_sha256": result["request_sha256"], "usage": result["tokens"]["usage"], "input_tokens": result["tokens"]["input_tokens"], "output_tokens": result["tokens"]["output_tokens"],
        "relay_artifacts": result["relay_artifacts"], "returned_model": result["parsed"].get("model") if isinstance(result["parsed"], dict) else None,
        "note": "alias diagnostic only: checks whether an upstream-style model identifier is routed; not part of the scientific battery unless separately justified",
    }
    return {"recorded": append_once(row), "row": row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aliases", action="store_true")
    args = parser.parse_args()
    www = record_reconstructed_www_404()
    wave = wave_control()
    out = {"www_diagnostic_recorded": www.get("recorded"), "wave_control_recorded": wave.get("recorded"), "wave_status": wave["row"].get("http_status"), "wave_content": wave["row"].get("content"), "wave_artifacts": wave["row"].get("relay_artifacts")}
    if args.aliases:
        aliases = {}
        for model in ["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/deepseek-v4-flash", "zai-org/GLM-5.3-Flash", "zai-org/glm-5.3-flash"]:
            row = alias_probe(model)["row"]
            aliases[model] = {"http_status": row["http_status"], "content": row["content"], "returned_model": row.get("returned_model"), "artifacts": row["relay_artifacts"]}
        out["aliases"] = aliases
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
