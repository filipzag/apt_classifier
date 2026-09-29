import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from apt_classifier.config import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_PRICING = {
    "openai/gpt-5.6-sol": (2.0, 10.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
}


class OpenRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    parsed: dict[str, Any] | list[Any] | None
    usage: LLMUsage
    model: str
    raw: dict[str, Any] = field(repr=False)


class OpenRouterClient:
    def __init__(self, settings: Settings, timeout: float = 180.0) -> None:
        self.settings = settings
        self.api_key = settings.require_api_key()
        self._client = httpx.Client(timeout=timeout)
        self.total_usage = LLMUsage()

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        effort: str,
        schema: dict[str, Any] | None = None,
        schema_name: str = "structured_response",
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "reasoning_effort": effort,
            "max_completion_tokens": max_tokens,
        }
        if schema is not None and model.startswith("anthropic/") and effort in {"max", "xhigh"}:
            body.pop("reasoning_effort")
            body["reasoning"] = {"effort": "medium", "exclude": True}
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        if seed is not None:
            body["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/apt-classifier",
            "X-Title": "APT Attribution Assistant",
        }
        attempts = 2 if schema is not None else 1
        last_finish_reason = "unknown"
        last_content = ""
        for attempt in range(attempts):
            request_body = body if attempt == 0 else _structured_retry_body(body, max_tokens)
            response = self._send_with_retry(request_body, headers)
            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise OpenRouterError("OpenRouter returned a non-JSON API response") from exc
            message, finish_reason = _response_message(payload)
            content = _message_content(message)
            usage = self._parse_usage(payload, model)
            self._record_usage(usage)
            if schema is None:
                return LLMResponse(content, None, usage, model, payload)
            try:
                parsed = _parse_structured_message(message, content, schema)
            except (json.JSONDecodeError, TypeError, ValueError):
                last_finish_reason = finish_reason
                last_content = content
                if attempt + 1 < attempts:
                    continue
                raise OpenRouterError(
                    "OpenRouter returned invalid structured JSON after "
                    f"{attempts} attempts (finish_reason={last_finish_reason}, "
                    f"content_chars={len(last_content)})"
                ) from None
            return LLMResponse(content, parsed, usage, model, payload)
        raise OpenRouterError("OpenRouter structured response retry failed")

    def _record_usage(self, usage: LLMUsage) -> None:
        self.total_usage = LLMUsage(
            self.total_usage.prompt_tokens + usage.prompt_tokens,
            self.total_usage.completion_tokens + usage.completion_tokens,
            self.total_usage.cost_usd + usage.cost_usd,
        )

    def _send_with_retry(self, body: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(OPENROUTER_URL, headers=headers, json=body)
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = OpenRouterError(
                    f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}"
                )
            except (httpx.HTTPError, OpenRouterError) as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
        raise OpenRouterError(str(last_error))

    @staticmethod
    def _parse_usage(payload: dict[str, Any], model: str) -> LLMUsage:
        raw = payload.get("usage") or {}
        prompt_tokens = int(raw.get("prompt_tokens") or 0)
        completion_tokens = int(raw.get("completion_tokens") or 0)
        reported_cost = raw.get("cost")
        if reported_cost is not None:
            cost = float(reported_cost)
        else:
            input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
            cost = prompt_tokens * input_price / 1_000_000
            cost += completion_tokens * output_price / 1_000_000
        return LLMUsage(prompt_tokens, completion_tokens, cost)


def _structured_retry_body(body: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    retry = dict(body)
    retry["max_completion_tokens"] = max(max_tokens + 2048, round(max_tokens * 1.5))
    messages = [dict(message) for message in body.get("messages", [])]
    instruction = (
        "Return exactly one complete JSON value matching the response schema. Do not use Markdown "
        "fences or explanatory text. Ensure the JSON is closed before ending the response."
    )
    system = next((message for message in messages if message.get("role") == "system"), None)
    if system is None:
        messages.insert(0, {"role": "system", "content": instruction})
    else:
        system["content"] = f"{system.get('content', '')}\n\n{instruction}"
    retry["messages"] = messages
    return retry


def _response_message(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("OpenRouter returned a malformed completion response") from exc
    if not isinstance(message, dict):
        raise OpenRouterError("OpenRouter returned a malformed completion message")
    return message, str(choice.get("finish_reason") or "unknown")


def _message_content(message: dict[str, Any]) -> str:
    raw = message.get("content")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str):
            return raw["text"]
        return json.dumps(raw.get("json", raw), ensure_ascii=False)
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
                elif "json" in part:
                    parts.append(json.dumps(part["json"], ensure_ascii=False))
        return "".join(parts)
    return "" if raw is None else str(raw)


def _parse_structured_message(
    message: dict[str, Any], content: str, schema: dict[str, Any]
) -> dict[str, Any] | list[Any]:
    provider_parsed = message.get("parsed")
    if provider_parsed is not None:
        value = (
            _parse_json_text(provider_parsed)
            if isinstance(provider_parsed, str)
            else provider_parsed
        )
    else:
        value = _parse_tool_arguments(message)
        if value is None:
            value = _parse_json_text(content)
    _validate_structured_value(value, schema)
    return value


def _parse_tool_arguments(message: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            return _parse_json_text(arguments)
        if isinstance(arguments, dict | list):
            return arguments
    return None


def _parse_json_text(content: str) -> dict[str, Any] | list[Any]:
    text = content.lstrip("\ufeff").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as direct_error:
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n")
            if first_newline >= 0:
                fenced = text[first_newline + 1 : -3].strip()
                try:
                    value = json.loads(fenced)
                except json.JSONDecodeError:
                    value = None
            else:
                value = None
        else:
            value = None
        if value is None:
            decoder = json.JSONDecoder()
            for index, character in enumerate(text):
                if character not in "[{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(text[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict | list):
                    return candidate
            raise direct_error
    if not isinstance(value, dict | list):
        raise ValueError("Structured response must be a JSON object or array")
    return value


def _validate_structured_value(value: Any, schema: dict[str, Any]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(value, dict):
        raise ValueError("Structured response must be an object")
    if expected_type == "array" and not isinstance(value, list):
        raise ValueError("Structured response must be an array")
    if isinstance(value, dict):
        missing = set(schema.get("required", ())) - set(value)
        if missing:
            raise ValueError("Structured response omitted required fields")
