from __future__ import annotations

import json
from dataclasses import dataclass

import regex

from hosted_frontier_common import PreparedRequest, TransportResponse, json_bytes

COPY_PREFIX = "Copy the input exactly. Input: <IN>"
COPY_SUFFIX = "</IN>"
COUNT_PREFIX = "Count the extended grapheme clusters (user-perceived characters), not bytes or code points. Reply with one integer only. ZWJ joins emoji components into one cluster; combining marks and variation selectors attach to their base character. Input:\n"
SCALAR_PREFIX = "List the Unicode scalar values in order as hexadecimal codepoints. Use exactly this format: `U+1F600 U+200D`. Input:\n"


@dataclass(frozen=True)
class CapturedCall:
    call_index: int
    url: str
    method: str
    body_bytes: bytes
    payload_obj: dict
    sanitized_headers: dict[str, str]
    auth_header_present: bool


def scalar_text(text: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in text)


def extract_provider_messages(provider: str, payload: dict) -> list[dict]:
    if provider == "openai":
        return payload["messages"]
    if provider == "anthropic":
        out = []
        for message in payload["messages"]:
            text = "".join(block.get("text", "") for block in message.get("content", []) if isinstance(block, dict))
            out.append({"role": message["role"], "content": text})
        return out
    if provider == "gemini":
        out = []
        for message in payload["contents"]:
            role = "assistant" if message["role"] == "model" else message["role"]
            text = "".join(part.get("text", "") for part in message.get("parts", []) if isinstance(part, dict))
            out.append({"role": role, "content": text})
        return out
    raise ValueError(provider)


def solve_prompt(prompt: str) -> str:
    if prompt == "Reply with exactly: 0":
        return "0"
    if prompt.startswith(COPY_PREFIX) and prompt.endswith(COPY_SUFFIX):
        target = prompt[len(COPY_PREFIX):-len(COPY_SUFFIX)]
        return f"<OUT>{target}</OUT>"
    if prompt.startswith(COUNT_PREFIX):
        target = prompt[len(COUNT_PREFIX):]
        return str(len(regex.findall(r"\X", target)))
    if prompt.startswith(SCALAR_PREFIX):
        target = prompt[len(SCALAR_PREFIX):]
        return scalar_text(target)
    raise ValueError("fake transport received an unrecognized prompt")


def corrupt_visible_text(text: str) -> str:
    if text.startswith("<OUT>") and text.endswith("</OUT>") and len(text) > 12:
        inner = text[5:-6]
        replacement = "X" if not inner.startswith("X") else "Y"
        return "<OUT>" + replacement + inner[1:] + "</OUT>"
    return text + " corrupt"


class CorrectFakeTransport:
    def __init__(self, provider: str, corrupt_call_index: int | None = None, malformed_call_index: int | None = None, rate_limit_call_indexes: set[int] | None = None, reflect_key_call_index: int | None = None):
        self.provider = provider
        self.corrupt_call_index = corrupt_call_index
        self.malformed_call_index = malformed_call_index
        self.rate_limit_call_indexes = rate_limit_call_indexes or set()
        self.reflect_key_call_index = reflect_key_call_index
        self.captured: list[CapturedCall] = []
        self._rate_limited_seen: set[int] = set()

    def send(self, prepared: PreparedRequest, api_key: str) -> TransportResponse:
        call_index = len(self.captured)
        full_headers = prepared.headers_with_auth(api_key)
        sanitized = {}
        auth_present = False
        for name, value in full_headers.items():
            if name.lower() in ("authorization", "x-api-key", "x-goog-api-key"):
                auth_present = bool(value)
                sanitized[name] = "[REDACTED]"
            else:
                sanitized[name] = value
        payload = json.loads(prepared.body_bytes.decode("utf-8"))
        self.captured.append(CapturedCall(call_index, prepared.url, prepared.method, prepared.body_bytes, payload, sanitized, auth_present))
        tic = 0.000001 * (call_index + 1)
        if call_index in self.rate_limit_call_indexes and call_index not in self._rate_limited_seen:
            self._rate_limited_seen.add(call_index)
            error = {"error": {"type": "rate_limit_error", "message": "fake rate limit"}}
            return TransportResponse(429, {"content-type": "application/json", "x-request-id": f"fake-{call_index}"}, json_bytes(error), tic)
        messages = extract_provider_messages(self.provider, payload)
        user_messages = [x["content"] for x in messages if x["role"] == "user"]
        if not user_messages:
            raise ValueError("fake transport found no user message")
        text = solve_prompt(user_messages[-1])
        if call_index == self.corrupt_call_index:
            text = corrupt_visible_text(text)
        input_tokens = max(1, len(prepared.body_bytes) // 4)
        output_tokens = max(1, len(text.encode("utf-8")))
        if call_index == self.reflect_key_call_index:
            text = api_key
        body = self.response_body(prepared.model, text, input_tokens, output_tokens)
        if call_index == self.malformed_call_index:
            body = body[:-1]
        return TransportResponse(200, {"content-type": "application/json", "x-request-id": f"fake-{call_index}"}, body, tic)

    def response_body(self, model: str, text: str, input_tokens: int, output_tokens: int) -> bytes:
        if self.provider == "openai":
            return json_bytes({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
            })
        if self.provider == "anthropic":
            return json_bytes({
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            })
        if self.provider == "gemini":
            return json_bytes({
                "candidates": [{
                    "content": {"parts": [{"text": text}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {"promptTokenCount": input_tokens, "candidatesTokenCount": output_tokens, "totalTokenCount": input_tokens + output_tokens},
                "modelVersion": model,
                "responseId": "fake-response",
            })
        raise ValueError(self.provider)
