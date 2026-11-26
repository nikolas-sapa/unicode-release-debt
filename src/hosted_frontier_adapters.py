from __future__ import annotations

import json

from hosted_frontier_common import (
    AdaptedResponse,
    PreparedRequest,
    ResponseIntegrityError,
    USER_AGENT,
    json_bytes,
)

DEFAULT_MODELS = {
    "openai": "gpt-5.5-2026-04-23",
    "anthropic": "claude-opus-4-5-20251101",
    "gemini": "gemini-3.1-pro-preview",
}


def parse_json_response(body: bytes, provider: str) -> dict:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResponseIntegrityError(f"{provider} response body is not valid UTF-8") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseIntegrityError(f"{provider} response body is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ResponseIntegrityError(f"{provider} response top level is not a JSON object")
    return parsed


def nonnegative_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class ProviderAdapter:
    provider: str
    endpoint: str
    adapter_id: str

    def __init__(self, model: str):
        self.model = model

    def prepare_request(self, messages: list[dict], max_tokens: int) -> PreparedRequest:
        raise NotImplementedError

    def parse_response(self, body: bytes) -> AdaptedResponse:
        raise NotImplementedError

    def preflight_messages(self) -> list[dict]:
        return [{"role": "user", "content": "Reply with exactly: 0"}]


class OpenAIAdapter(ProviderAdapter):
    provider = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"
    adapter_id = "openai_chat_completions_2026-11-26"

    def prepare_request(self, messages: list[dict], max_tokens: int) -> PreparedRequest:
        decode = {"temperature": 0.0, "max_completion_tokens": max_tokens, "reasoning_effort": "none"}
        body = {"model": self.model, "messages": messages, **decode}
        return PreparedRequest(
            provider=self.provider,
            model=self.model,
            url=self.endpoint,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            body_bytes=json_bytes(body),
            body_obj=body,
            decode_params=decode,
            max_tokens=max_tokens,
            logical_messages=messages,
            auth_header_name="Authorization",
            auth_header_prefix="Bearer ",
        )

    def parse_response(self, body: bytes) -> AdaptedResponse:
        parsed = parse_json_response(body, self.provider)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ResponseIntegrityError("openai response has no choices[0] object")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ResponseIntegrityError("openai response choices[0].message is not an object")
        text = message.get("content")
        if not isinstance(text, str):
            raise ResponseIntegrityError("openai response message.content is missing or not a string")
        served = parsed.get("model")
        if not isinstance(served, str) or not served:
            raise ResponseIntegrityError("openai response missing served model string")
        usage = parsed.get("usage")
        usage_present = isinstance(usage, dict)
        input_tokens = output_tokens = total_tokens = None
        notes = []
        if usage_present:
            input_tokens = nonnegative_int(usage.get("prompt_tokens"))
            output_tokens = nonnegative_int(usage.get("completion_tokens"))
            total_tokens = nonnegative_int(usage.get("total_tokens"))
            if input_tokens is None or output_tokens is None:
                notes.append("usage object present but prompt_tokens/completion_tokens missing or invalid")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
                notes.append("total_tokens missing; computed as prompt+completion")
        else:
            notes.append("usage object missing; conservative cost debit required")
        return AdaptedResponse(
            text=text,
            served_model=served,
            finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_present=usage_present and input_tokens is not None and output_tokens is not None,
            parsed=parsed,
            integrity_notes=notes,
        )


class AnthropicAdapter(ProviderAdapter):
    provider = "anthropic"
    endpoint = "https://api.anthropic.com/v1/messages"
    adapter_id = "anthropic_messages_2023-06-01"
    api_version = "2023-06-01"

    def prepare_request(self, messages: list[dict], max_tokens: int) -> PreparedRequest:
        anthropic_messages = []
        for message in messages:
            if message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
                raise ResponseIntegrityError("logical message inventory contains a non-text or unsupported-role message")
            anthropic_messages.append({
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            })
        decode = {"temperature": 0.0, "max_tokens": max_tokens}
        body = {"model": self.model, "messages": anthropic_messages, **decode}
        return PreparedRequest(
            provider=self.provider,
            model=self.model,
            url=self.endpoint,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                "anthropic-version": self.api_version,
            },
            body_bytes=json_bytes(body),
            body_obj=body,
            decode_params=decode,
            max_tokens=max_tokens,
            logical_messages=messages,
            auth_header_name="x-api-key",
        )

    def parse_response(self, body: bytes) -> AdaptedResponse:
        parsed = parse_json_response(body, self.provider)
        content = parsed.get("content")
        if not isinstance(content, list) or not content:
            raise ResponseIntegrityError("anthropic response content is missing or empty")
        parts = []
        for index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise ResponseIntegrityError(f"anthropic response content block {index} is not a text block")
            parts.append(block["text"])
        served = parsed.get("model")
        if not isinstance(served, str) or not served:
            raise ResponseIntegrityError("anthropic response missing served model string")
        usage = parsed.get("usage")
        usage_present = isinstance(usage, dict)
        input_tokens = output_tokens = total_tokens = None
        notes = []
        if usage_present:
            input_tokens = nonnegative_int(usage.get("input_tokens"))
            output_tokens = nonnegative_int(usage.get("output_tokens"))
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            if input_tokens is None or output_tokens is None:
                notes.append("usage object present but input_tokens/output_tokens missing or invalid")
        else:
            notes.append("usage object missing; conservative cost debit required")
        return AdaptedResponse(
            text="".join(parts),
            served_model=served,
            finish_reason=parsed.get("stop_reason") if isinstance(parsed.get("stop_reason"), str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_present=usage_present and input_tokens is not None and output_tokens is not None,
            parsed=parsed,
            integrity_notes=notes,
        )


class GeminiAdapter(ProviderAdapter):
    provider = "gemini"
    adapter_id = "gemini_generatecontent_v1beta_2026-11-26"

    @property
    def endpoint(self) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def prepare_request(self, messages: list[dict], max_tokens: int) -> PreparedRequest:
        contents = []
        for message in messages:
            if message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
                raise ResponseIntegrityError("logical message inventory contains a non-text or unsupported-role message")
            contents.append({
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            })
        decode = {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": False},
        }
        body = {"contents": contents, "generationConfig": decode}
        return PreparedRequest(
            provider=self.provider,
            model=self.model,
            url=self.endpoint,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            body_bytes=json_bytes(body),
            body_obj=body,
            decode_params=decode,
            max_tokens=max_tokens,
            logical_messages=messages,
            auth_header_name="x-goog-api-key",
        )

    def parse_response(self, body: bytes) -> AdaptedResponse:
        parsed = parse_json_response(body, self.provider)
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            raise ResponseIntegrityError("gemini response has no candidates[0] object")
        candidate = candidates[0]
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list) or not content["parts"]:
            raise ResponseIntegrityError("gemini response candidates[0].content.parts is missing or empty")
        parts = []
        for index, part in enumerate(content["parts"]):
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise ResponseIntegrityError(f"gemini response part {index} is not a text part")
            if part.get("thought") is True:
                raise ResponseIntegrityError("gemini response unexpectedly exposed a thought part")
            parts.append(part["text"])
        served = parsed.get("modelVersion")
        if not isinstance(served, str) or not served:
            raise ResponseIntegrityError("gemini response missing modelVersion string")
        usage = parsed.get("usageMetadata")
        usage_present = isinstance(usage, dict)
        input_tokens = output_tokens = total_tokens = None
        notes = []
        if usage_present:
            input_tokens = nonnegative_int(usage.get("promptTokenCount"))
            output_tokens = nonnegative_int(usage.get("candidatesTokenCount"))
            total_tokens = nonnegative_int(usage.get("totalTokenCount"))
            if input_tokens is None or output_tokens is None:
                notes.append("usageMetadata present but promptTokenCount/candidatesTokenCount missing or invalid")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
                notes.append("totalTokenCount missing; computed as prompt+candidates")
        else:
            notes.append("usageMetadata missing; conservative cost debit required")
        return AdaptedResponse(
            text="".join(parts),
            served_model=served,
            finish_reason=candidate.get("finishReason") if isinstance(candidate.get("finishReason"), str) else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_present=usage_present and input_tokens is not None and output_tokens is not None,
            parsed=parsed,
            integrity_notes=notes,
        )


def adapter_for(provider: str, model: str) -> ProviderAdapter:
    adapters = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "gemini": GeminiAdapter}
    if provider not in adapters:
        raise ResponseIntegrityError(f"unsupported provider {provider!r}")
    return adapters[provider](model)
