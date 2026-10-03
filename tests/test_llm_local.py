import json

import httpx
import pytest

import apt_classifier.llm as llm
from apt_classifier.llm import OpenRouterClient, OpenRouterError, supports_seed

SCHEMA = {"type": "object", "required": ["result"]}


class FakeSettings:
    def __init__(self, *, key=None, local=True, **overrides):
        self.key = key
        self.local_llm_enabled = local
        self.local_llm_url = "http://ollama.test:11434"
        self.local_llm_fallback_model = "local/qwen3.5:4b"
        self.local_llm_num_ctx = 32768
        self.local_llm_keep_alive = "30m"
        self.local_llm_timeout = 300.0
        self.local_llm_think_efforts = "high,xhigh,max"
        self.local_llm_think_tokens = 8192
        for name, value in overrides.items():
            setattr(self, name, value)

    def require_api_key(self):
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")
        return self.key


def ollama_response(content, *, done_reason="stop", thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return httpx.Response(
        200,
        json={
            "model": "qwen3.5:9b",
            "message": message,
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": 120,
            "eval_count": 30,
        },
    )


def openrouter_response(content):
    return httpx.Response(
        200,
        json={
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
        },
    )


@pytest.fixture(autouse=True)
def _fresh_tags_cache():
    llm._TAGS_CACHE.clear()
    yield
    llm._TAGS_CACHE.clear()


def client_for(settings, responses):
    pending = list(responses)
    calls = []

    def handler(request):
        calls.append(request)
        return pending.pop(0)

    client = OpenRouterClient(settings)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def test_local_model_uses_ollama_native_chat_with_schema_and_context():
    client, calls = client_for(FakeSettings(), [ollama_response('{"result":"ok"}')])
    result = client.complete(
        [{"role": "user", "content": "test"}],
        model="local/qwen3.5:9b",
        effort="low",
        schema=SCHEMA,
        max_tokens=500,
        seed=41,
    )
    assert result.parsed == {"result": "ok"}
    assert result.model == "local/qwen3.5:9b"
    assert result.usage == llm.LLMUsage(120, 30, 0.0)
    assert str(calls[0].url) == "http://ollama.test:11434/api/chat"
    body = json.loads(calls[0].content)
    assert body["model"] == "qwen3.5:9b"
    assert body["format"] == SCHEMA
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["num_ctx"] == 32768
    assert body["options"]["num_predict"] == 500
    assert body["options"]["seed"] == 41
    assert "authorization" not in calls[0].headers


def test_high_effort_turns_thinking_on_with_extra_budget():
    client, calls = client_for(
        FakeSettings(), [ollama_response('{"result":"ok"}', thinking="weighing hypotheses")]
    )
    result = client.complete(
        [{"role": "user", "content": "review"}],
        model="local/qwen3.5:9b",
        effort="max",
        schema=SCHEMA,
        max_tokens=1000,
    )
    assert result.parsed == {"result": "ok"}
    body = json.loads(calls[0].content)
    assert body["think"] is True
    assert body["options"]["num_predict"] == 1000 + 8192
    assert body["options"]["seed"] == 0


def test_local_invalid_json_retries_once_with_instruction_then_fails():
    client, calls = client_for(
        FakeSettings(),
        [ollama_response('{"result": "trunc', done_reason="length"), ollama_response("still bad")],
    )
    with pytest.raises(
        OpenRouterError, match="Local LLM .*invalid structured JSON after 2 attempts"
    ):
        client.complete(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}],
            model="local/qwen3.5:9b",
            effort="low",
            schema=SCHEMA,
            max_tokens=400,
        )
    assert len(calls) == 2
    retry = json.loads(calls[1].content)
    assert "Return exactly one complete JSON value" in retry["messages"][0]["content"]
    assert retry["options"]["num_predict"] == 600


def test_local_fenced_json_is_accepted():
    client, _ = client_for(FakeSettings(), [ollama_response('```json\n{"result":"ok"}\n```')])
    result = client.complete(
        [{"role": "user", "content": "x"}], model="local/qwen3.5:9b", effort="low", schema=SCHEMA
    )
    assert result.parsed == {"result": "ok"}


def test_remote_model_falls_back_to_local_without_key():
    client, calls = client_for(FakeSettings(key=None), [ollama_response('{"result":"ok"}')])
    assert client.resolve_model("anthropic/claude-sonnet-5") == "local/qwen3.5:4b"
    result = client.complete(
        [{"role": "user", "content": "x"}],
        model="anthropic/claude-sonnet-5",
        effort="max",
        schema=SCHEMA,
    )
    assert result.model == "local/qwen3.5:4b"
    assert str(calls[0].url).endswith("/api/chat")


def test_hybrid_routes_remote_ids_to_openrouter_and_local_ids_to_ollama():
    client, calls = client_for(
        FakeSettings(key="test-key"),
        [openrouter_response('{"result":"remote"}'), ollama_response('{"result":"local"}')],
    )
    remote = client.complete(
        [{"role": "user", "content": "x"}],
        model="anthropic/claude-sonnet-5",
        effort="low",
        schema=SCHEMA,
    )
    local = client.complete(
        [{"role": "user", "content": "x"}], model="local/qwen3.5:9b", effort="low", schema=SCHEMA
    )
    assert remote.parsed == {"result": "remote"}
    assert local.parsed == {"result": "local"}
    assert str(calls[0].url) == llm.OPENROUTER_URL
    assert str(calls[1].url).endswith("/api/chat")
    assert client.total_usage.cost_usd == pytest.approx(0.01)


def test_no_key_and_local_disabled_raises_configuration_error():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(FakeSettings(key=None, local=False))


def test_local_model_needs_local_enabled():
    client, _ = client_for(FakeSettings(key="test-key", local=False), [])
    with pytest.raises(OpenRouterError, match="LOCAL_LLM_ENABLED"):
        client.complete([{"role": "user", "content": "x"}], model="local/qwen3.5:9b", effort="low")


def test_unreachable_server_names_local_llm(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    client = OpenRouterClient(FakeSettings())
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenRouterError, match="Local LLM request failed"):
        client.complete([{"role": "user", "content": "x"}], model="local/qwen3.5:9b", effort="low")


def test_llm_available_with_reachable_local_server(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://ollama.test:11434/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "qwen3.5:4b"}]}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    assert llm.local_llm_models(FakeSettings()) == ["qwen3.5:4b"]
    assert llm.llm_available(FakeSettings())
    assert not llm.llm_available(FakeSettings(local=False))
    assert llm.llm_available(FakeSettings(key="k", local=False))


def test_llm_unavailable_when_local_server_down(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    assert llm.local_llm_models(FakeSettings()) is None
    assert not llm.llm_available(FakeSettings())


def test_supports_seed():
    assert supports_seed("openai/gpt-5.6-sol")
    assert supports_seed("local/qwen3.5:9b")
    assert not supports_seed("anthropic/claude-sonnet-5")


def test_gpu_only_forces_all_layers_on_the_gpu_and_can_be_switched_off():
    client, calls = client_for(FakeSettings(), [ollama_response('{"result":"ok"}')])
    client.complete([{"role": "user", "content": "x"}], model="local/qwen3.5:4b", effort="low")
    options = json.loads(calls[0].content)["options"]
    assert options["num_gpu"] == 999
    assert options["num_thread"] == 2
    client, calls = client_for(
        FakeSettings(local_llm_gpu_only=False), [ollama_response('{"result":"ok"}')]
    )
    client.complete([{"role": "user", "content": "x"}], model="local/qwen3.5:4b", effort="low")
    assert "num_gpu" not in json.loads(calls[0].content)["options"]


def test_model_that_does_not_fit_fails_once_with_a_gpu_message():
    oom = httpx.Response(
        500,
        json={"error": "llama-server process has terminated: cudaMalloc failed: out of memory"},
    )
    client, calls = client_for(FakeSettings(), [oom])
    with pytest.raises(OpenRouterError, match="does not fit in the free GPU memory"):
        client.complete([{"role": "user", "content": "x"}], model="local/qwen3.5:9b", effort="low")
    assert len(calls) == 1  # no pointless reload attempts


def test_prose_goes_to_openrouter_extraction_or_local_sentence_rewriting():
    from apt_classifier.llm import prose_normalisation

    no_key = OpenRouterClient(FakeSettings(key=None))
    remote = FakeSettings(extract_model="openai/gpt-5.6-sol")
    assert prose_normalisation(no_key, remote) == ("sentences", "local/qwen3.5:4b")
    local = FakeSettings(extract_model="local/qwen3.5:9b")
    assert prose_normalisation(no_key, local) == ("sentences", "local/qwen3.5:9b")
    with_key = OpenRouterClient(FakeSettings(key="k"))
    assert prose_normalisation(with_key, remote) == ("extract", "openai/gpt-5.6-sol")
    assert prose_normalisation(None, remote) == ("split", None)

    class DuckClient:  # test fakes without resolve_model keep whole-report extraction
        def complete(self, *args, **kwargs):
            raise AssertionError

    assert prose_normalisation(DuckClient(), remote)[0] == "extract"


def test_local_llm_counts_as_available_only_with_the_model_pulled(monkeypatch):
    tags = {"models": [{"name": "qwen3:4b"}]}

    def fake_get(url, timeout):
        return httpx.Response(200, json=tags, request=httpx.Request("GET", url))

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    assert not llm.llm_available(FakeSettings())  # server up, fallback model missing
    llm._TAGS_CACHE.clear()
    tags["models"].append({"name": "qwen3.5:4b"})
    assert llm.llm_available(FakeSettings())
