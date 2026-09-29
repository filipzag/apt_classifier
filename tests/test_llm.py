import json

import httpx
import pytest

from apt_classifier.llm import OpenRouterClient, OpenRouterError


class FakeSettings:
    def require_api_key(self):
        return "test-key"


def response(content, *, finish_reason="stop", parsed=None):
    message = {"content": content}
    if parsed is not None:
        message["parsed"] = parsed
    return httpx.Response(
        200,
        json={
            "choices": [{"finish_reason": finish_reason, "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def client_for(responses):
    pending = list(responses)
    calls = []

    def handler(request):
        calls.append(request)
        return pending.pop(0)

    client = OpenRouterClient(FakeSettings())
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def test_structured_response_accepts_fenced_content_blocks():
    client, calls = client_for(
        [response([{"type": "text", "text": '```json\n{"result":"ok"}\n```'}])]
    )
    result = client.complete(
        [{"role": "user", "content": "test"}],
        model="test/model",
        effort="low",
        schema={"type": "object"},
    )
    assert result.parsed == {"result": "ok"}
    assert len(calls) == 1


def test_structured_response_uses_provider_parsed_field():
    client, calls = client_for([response("", parsed={"result": "ok"})])
    result = client.complete(
        [{"role": "user", "content": "test"}],
        model="test/model",
        effort="low",
        schema={"type": "object"},
    )
    assert result.parsed == {"result": "ok"}
    assert len(calls) == 1


def test_invalid_structured_response_retries_once():
    client, calls = client_for(
        [
            response('{"result":', finish_reason="length"),
            response('{"result":"ok"}'),
        ]
    )
    result = client.complete(
        [{"role": "user", "content": "test"}],
        model="test/model",
        effort="max",
        schema={"type": "object"},
    )
    assert result.parsed == {"result": "ok"}
    assert len(calls) == 2
    assert client.total_usage.prompt_tokens == 20
    assert client.total_usage.completion_tokens == 10


def test_invalid_structured_response_reports_safe_diagnostics():
    client, _ = client_for(
        [response("not json"), response("still not json", finish_reason="length")]
    )
    with pytest.raises(OpenRouterError, match=r"finish_reason=length.*content_chars=14"):
        client.complete(
            [{"role": "user", "content": "test"}],
            model="test/model",
            effort="max",
            schema={"type": "object"},
        )


def test_anthropic_max_reasoning_is_capped_for_structured_output():
    client, calls = client_for([response('{"result":"ok"}')])
    client.complete(
        [{"role": "user", "content": "test"}],
        model="anthropic/test-model",
        effort="max",
        schema={"type": "object"},
        max_tokens=9000,
    )
    request = json.loads(calls[0].content)
    assert "reasoning_effort" not in request
    assert request["reasoning"] == {"effort": "medium", "exclude": True}
