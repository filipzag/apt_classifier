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
# Model ids with this prefix run on the local Ollama server (``local/qwen3.5:4b``); every
# other id goes to OpenRouter. The per-task model settings are the only switch.
LOCAL_PREFIX = "local/"
# Qwen3.5's recommended sampling (model card), non-thinking for the structured tasks and
# thinking for the high-effort ones, but without its chat presence_penalty (1.5): extraction
# copies report text verbatim and JSON repeats its keys, both of which a repeat penalty
# pushes against. A fixed seed keeps benchmark runs repeatable.
_LOCAL_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
_LOCAL_THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


class OpenRouterError(RuntimeError):
    pass


# Raised for both providers; the old name stays because callers and tests catch it.
LLMError = OpenRouterError


def is_local_model(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)


def supports_seed(model: str) -> bool:
    """Providers that honour ``seed``: OpenAI models via OpenRouter and the local server."""
    return model.startswith("openai/") or "gpt" in model.casefold() or is_local_model(model)


def _setting(settings: Any, name: str, default: Any) -> Any:
    value = getattr(settings, name, None)
    return default if value is None else value


def _optional_api_key(settings: Any) -> str | None:
    try:
        return settings.require_api_key()
    except RuntimeError:
        return None


def local_llm_enabled(settings: Any) -> bool:
    return bool(_setting(settings, "local_llm_enabled", False))


_TAGS_CACHE: dict[str, tuple[float, list[str] | None]] = {}
_TAGS_TTL_SECONDS = 30.0


def local_llm_models(settings: Any, timeout: float = 2.0) -> list[str] | None:
    """Models pulled on the local Ollama server, or None when it is disabled or unreachable.

    Cached for 30 s per URL: the Streamlit UI asks on every rerun."""
    if not local_llm_enabled(settings):
        return None
    url = str(_setting(settings, "local_llm_url", "http://localhost:11434")).rstrip("/")
    cached = _TAGS_CACHE.get(url)
    if cached is not None and time.monotonic() - cached[0] < _TAGS_TTL_SECONDS:
        return cached[1]
    try:
        response = httpx.get(f"{url}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = [str(row.get("name", "")) for row in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError, AttributeError):
        models = None
    _TAGS_CACHE[url] = (time.monotonic(), models)
    return models


def local_llm_ready(settings: Any) -> bool:
    """The local server answers and has the fallback model pulled (``ollama pull``)."""
    pulled = local_llm_models(settings)
    if pulled is None:
        return False
    name = str(_setting(settings, "local_llm_fallback_model", "")).removeprefix(LOCAL_PREFIX)
    return any(tag in {name, f"{name}:latest"} for tag in pulled)


def llm_available(settings: Any) -> bool:
    """True when some LLM can answer: an OpenRouter key, or a local server with the model."""
    return _optional_api_key(settings) is not None or local_llm_ready(settings)


def prose_normalisation(client: Any, settings: Any) -> tuple[str, str | None]:
    """How report prose becomes observations: ``("extract", model)`` when OpenRouter answers
    the extract model (whole-report extraction), ``("sentences", model)`` when a local model
    answers it (split, then each sentence rewritten 1:1, ``evidence/standardise.py``) and
    ``("split", None)`` without an LLM.

    A local model never rewrites a whole report: the 4B dropped actions (CTI-HAL technique F1
    0.431 -> 0.370); one sentence at a time it gained over the plain split on all 55 CTI-HAL
    reports (0.459 -> 0.472; docs 16.4a).
    """
    if client is None:
        return "split", None
    model = str(getattr(settings, "extract_model", ""))
    resolve = getattr(client, "resolve_model", None)
    if resolve is None:  # test fakes: a plain client extracts
        return "extract", model
    try:
        resolved = resolve(model)
    except RuntimeError:
        return "split", None
    return ("sentences", resolved) if is_local_model(resolved) else ("extract", resolved)


def build_llm_client(settings: Any) -> "OpenRouterClient | None":
    return OpenRouterClient(settings) if llm_available(settings) else None


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
    """Chat client for OpenRouter and, for ``local/`` model ids, a local Ollama server.

    Without an OpenRouter key every remote model id is answered by
    ``local_llm_fallback_model`` (hybrid mode: the review stays on OpenRouter when a key
    exists and runs locally otherwise).
    """

    def __init__(self, settings: Settings, timeout: float = 180.0) -> None:
        self.settings = settings
        self.local_enabled = local_llm_enabled(settings)
        self.api_key = _optional_api_key(settings)
        if self.api_key is None and not self.local_enabled:
            self.api_key = settings.require_api_key()  # raises the usual configuration error
        self.local_url = str(_setting(settings, "local_llm_url", "http://localhost:11434")).rstrip(
            "/"
        )
        self.local_timeout = float(_setting(settings, "local_llm_timeout", 300.0))
        self._client = httpx.Client(timeout=timeout)
        self.total_usage = LLMUsage()

    def answers_directly(self, model: str) -> bool:
        """True when ``model`` itself answers (OpenRouter with a key, or a ``local/`` id)."""
        return is_local_model(model) or self.api_key is not None

    def resolve_model(self, model: str) -> str:
        """The model that will actually answer a request for ``model``."""
        if is_local_model(model) or self.api_key is not None:
            return model
        fallback = str(_setting(self.settings, "local_llm_fallback_model", ""))
        if self.local_enabled and is_local_model(fallback):
            return fallback
        raise OpenRouterError(
            f"{model} needs an OpenRouter key; set OPENROUTER_API_KEY or a local/ model"
        )

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
        model = self.resolve_model(model)
        if is_local_model(model):
            if not self.local_enabled:
                raise OpenRouterError(f"{model} needs LOCAL_LLM_ENABLED=true and a running Ollama")
            return self._complete_local(
                messages,
                model=model,
                effort=effort,
                schema=schema,
                max_tokens=max_tokens,
                seed=seed,
            )
        if self.api_key is None:
            raise OpenRouterError(f"{model} needs an OpenRouter key")
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
            "X-Title": "Vestige",
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

    def _complete_local(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        effort: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
        seed: int | None,
    ) -> LLMResponse:
        # Ollama's native API, not its OpenAI-compatible /v1: only the native one takes
        # num_ctx per request, and the default context silently truncates whole reports.
        think_efforts = {
            part.strip()
            for part in str(_setting(self.settings, "local_llm_think_efforts", "")).split(",")
            if part.strip()
        }
        think = effort in think_efforts
        num_predict = max_tokens
        if think:
            num_predict += int(_setting(self.settings, "local_llm_think_tokens", 8192))
        options: dict[str, Any] = dict(_LOCAL_THINKING_SAMPLING if think else _LOCAL_SAMPLING)
        options.update(
            num_ctx=int(_setting(self.settings, "local_llm_num_ctx", 32768)),
            num_predict=num_predict,
            seed=0 if seed is None else seed,
        )
        if _setting(self.settings, "local_llm_gpu_only", True):
            # every layer on the GPU: Ollama then fails to load (CUDA out of memory) rather
            # than silently running part of the model on the CPU
            options["num_gpu"] = 999
            options["num_thread"] = int(_setting(self.settings, "local_llm_num_thread", 2))
        body: dict[str, Any] = {
            "model": model.removeprefix(LOCAL_PREFIX),
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": str(_setting(self.settings, "local_llm_keep_alive", "5m")),
            "options": options,
        }
        if schema is not None:
            body["format"] = schema
        attempts = 2 if schema is not None else 1
        last_finish_reason = "unknown"
        last_content = ""
        for attempt in range(attempts):
            request_body = body
            if attempt:
                request_body = dict(body, messages=_retry_messages(messages))
                request_body["options"] = dict(options, num_predict=round(num_predict * 1.5))
            try:
                response = self._send_with_retry(
                    request_body,
                    {"Content-Type": "application/json"},
                    url=f"{self.local_url}/api/chat",
                    timeout=self.local_timeout,
                    provider="Local LLM",
                )
            except OpenRouterError as exc:
                if _out_of_gpu_memory(str(exc)):
                    raise OpenRouterError(
                        f"Local LLM {model} does not fit in the free GPU memory and GPU-only mode "
                        "keeps it off the CPU. Free GPU memory, lower LOCAL_LLM_NUM_CTX or use a "
                        "smaller model (local/qwen3.5:4b)."
                    ) from exc
                raise
            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise OpenRouterError("Local LLM returned a non-JSON API response") from exc
            message = payload.get("message")
            if not isinstance(message, dict):
                raise OpenRouterError("Local LLM returned a malformed chat response")
            content = _message_content(message)
            usage = LLMUsage(
                int(payload.get("prompt_eval_count") or 0), int(payload.get("eval_count") or 0)
            )
            self._record_usage(usage)
            if schema is None:
                return LLMResponse(content, None, usage, model, payload)
            try:
                parsed = _parse_json_text(content)
                _validate_structured_value(parsed, schema)
            except (json.JSONDecodeError, TypeError, ValueError):
                last_finish_reason = str(payload.get("done_reason") or "unknown")
                last_content = content
                continue
            return LLMResponse(content, parsed, usage, model, payload)
        raise OpenRouterError(
            f"Local LLM {model} returned invalid structured JSON after {attempts} attempts "
            f"(done_reason={last_finish_reason}, content_chars={len(last_content)})"
        )

    def _record_usage(self, usage: LLMUsage) -> None:
        self.total_usage = LLMUsage(
            self.total_usage.prompt_tokens + usage.prompt_tokens,
            self.total_usage.completion_tokens + usage.completion_tokens,
            self.total_usage.cost_usd + usage.cost_usd,
        )

    def _send_with_retry(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        *,
        url: str = OPENROUTER_URL,
        timeout: float | None = None,
        provider: str = "OpenRouter",
    ) -> httpx.Response:
        last_error: Exception | None = None
        extra = {} if timeout is None else {"timeout": timeout}
        for attempt in range(3):
            try:
                response = self._client.post(url, headers=headers, json=body, **extra)
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = OpenRouterError(
                    f"{provider} returned HTTP {response.status_code}: {response.text[:500]}"
                )
                if _out_of_gpu_memory(response.text):
                    break  # a model that does not fit will not fit a second later either
            except OpenRouterError as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = OpenRouterError(f"{provider} request failed ({url}): {exc}")
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


def _out_of_gpu_memory(text: str) -> bool:
    lowered = text.casefold()
    return "out of memory" in lowered or "cudamalloc failed" in lowered


def _structured_retry_body(body: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    retry = dict(body)
    retry["max_completion_tokens"] = max(max_tokens + 2048, round(max_tokens * 1.5))
    retry["messages"] = _retry_messages(body.get("messages", []))
    return retry


def _retry_messages(original: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = [dict(message) for message in original]
    instruction = (
        "Return exactly one complete JSON value matching the response schema. Do not use Markdown "
        "fences or explanatory text. Ensure the JSON is closed before ending the response."
    )
    system = next((message for message in messages if message.get("role") == "system"), None)
    if system is None:
        messages.insert(0, {"role": "system", "content": instruction})
    else:
        system["content"] = f"{system.get('content', '')}\n\n{instruction}"
    return messages


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
