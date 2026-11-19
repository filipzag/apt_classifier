"""The model's second opinion through the case service and the API (settings, service, route).

Fakes only: a duck client stands in for OpenRouter, so no model is called and no key is needed.
The service tests run on a fake engine that fails the test if an analysis is started. The API
round trip uses the small real engine of the software-question tests, because a full case
payload needs a real analysis state; that case is created without any model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from apt_classifier import model_opinion as mo
from apt_classifier.attribution import MaturityThresholds
from apt_classifier.config import Settings
from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.llm import LLMResponse, LLMUsage, OpenRouterError
from apt_classifier.report import EXPORT_KINDS
from apt_classifier.service import (
    CaseMeta,
    CaseNotFound,
    CaseService,
    IntakePayload,
    ServiceUnavailable,
)
from apt_classifier.users import UserStore
from tests.test_top3_view import _kb
from tests.test_top3_view import _state as _view_state

ROOT = Path(__file__).resolve().parents[1]
MODEL = mo.REGISTERED_MODEL
EFFORT = mo.REGISTERED_EFFORT
LOCAL_MODEL = "local/qwen3.5:4b-text"
# a text only the model's answer holds, to look for where it must not appear
RATIONALE = "RATIONALE-WRITTEN-BY-THE-MODEL"
COST_NOTE = "about $0.05-0.06 a call (measured on 119 reports)"
SETTING_NAMES = ("MODEL_OPINION_ENABLED", "MODEL_OPINION_MODEL", "MODEL_OPINION_EFFORT")
PAYLOAD_KEYS = {
    "asked_at",
    "model",
    "effort",
    "prompt_version",
    "decision",
    "first",
    "candidates",
    "confident_name",
    "confident_rule",
    "rationale",
    "agreement",
    "statements_sent",
    "statements_total",
    "names_masked",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "stale",
    "note",
}
READY = {
    "enabled": True,
    "available": True,
    "model": MODEL,
    "effort": EFFORT,
    "status": "ready",
    "confident_probability": 0.99,
    "registered_model": MODEL,
    "registered_effort": EFFORT,
    "cost_note": COST_NOTE,
}


# ----- fakes


def _answer(name: str, probability: float = 0.99, decision: str = "name") -> dict[str, Any]:
    return {
        "candidates": [{"group_id": "G0999", "group_name": name, "probability": probability}],
        "decision": decision,
        "rationale": RATIONALE,
    }


class Client:
    """A duck client: an OpenRouter key and ``complete``. ``during`` runs while the model
    "answers"; ``error`` is raised in place of an answer."""

    def __init__(
        self,
        parsed: Any = None,
        *,
        api_key: str | None = "key-for-tests",
        answered_by: str | None = None,
        during: Callable[[], None] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.api_key = api_key
        self.parsed = _answer("FIN7") if parsed is None else parsed
        self.answered_by = answered_by
        self.during = during
        self.error = error
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append((messages, kwargs))
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        return LLMResponse(
            "{}",
            self.parsed,
            LLMUsage(3625, 1808, 0.0543),
            self.answered_by or kwargs["model"],
            {},
        )


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a second opinion must never start an analysis, a rescore or a review")


def _observation(text: str, status: EvidenceStatus = EvidenceStatus.POSITIVE) -> SimpleNamespace:
    return SimpleNamespace(text=text, status=status)


def _case_state(kb: Any, leader: str = "G0046") -> SimpleNamespace:
    """A state whose product answer is FIN7 first, then Carbanak, then APT29."""
    state = _view_state(kb, ["G0046", "G0008", "G0016"])
    state.attribution_decision.leading_group_id = leader
    state.attribution_decision.named_cluster = ()
    state.ledger = SimpleNamespace(
        observations=[
            _observation("FIN7 sent spearphishing e-mails with a malicious attachment."),
            _observation("The actor dumped credentials from LSASS memory."),
            _observation("No lateral movement was seen.", EvidenceStatus.NEGATIVE),
        ]
    )
    return state


def _service(
    tmp_path: Path,
    *,
    client: Any = None,
    enabled: bool = True,
    model: str = MODEL,
    effort: str = EFFORT,
) -> tuple[CaseService, SimpleNamespace]:
    """A case service on a fake engine with one case, "c1", put straight into memory."""
    kb = _kb()
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        model_cache_dir=tmp_path / "hf",
        confidence_labels_enabled=False,
        model_opinion_enabled=enabled,
        model_opinion_model=model,
        model_opinion_effort=effort,
    )
    engine = SimpleNamespace(
        settings=settings,
        kb=kb,
        combiner=None,
        attribution_thresholds=MaturityThresholds(),
        llm=client,
        analyze=_never,
        rescore=_never,
        run_review=_never,
        default_weights=dict,
    )
    service = CaseService(settings, engine=engine, cases_dir=tmp_path / "cases")  # type: ignore[arg-type]
    state = _case_state(kb)
    service._states["c1"] = state  # noqa: SLF001
    service._meta["c1"] = CaseMeta(case_id="c1", name="case")  # noqa: SLF001
    return service, state


def _api(service: Any, **options: Any) -> TestClient:
    """The API over a service, sign-in off, its user table beside the service's cases."""
    settings = Settings(_env_file=None, auth_mode="off")  # type: ignore[call-arg]
    users = UserStore(service.store.directory.parent / "users.db")
    app = create_app(service, serve_web=False, settings=settings, users=users)
    return TestClient(app, **options)


def _free_from_another_thread(lock: Any) -> bool:
    """Whether a thread other than this one can take the lock right now."""
    taken: list[bool] = []

    def attempt() -> None:
        acquired = lock.acquire(blocking=False)
        taken.append(acquired)
        if acquired:
            lock.release()

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(10)
    return taken == [True]


def _snapshot(state: Any) -> tuple[Any, ...]:
    """What the product decided, to compare before and after a second opinion."""
    return (
        sorted(vars(state)),
        [(entry.group_id, entry.rank, entry.score) for entry in state.fusion.entries],
        state.fusion is state.deterministic_fusion,
        dict(vars(state.attribution_decision)),
        json.dumps(state.context.combiner, sort_keys=True),
        [(item.text, item.status) for item in state.ledger.observations],
    )


# ----- 1. settings


def test_the_second_opinion_is_off_by_default_with_the_registered_model(monkeypatch):
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.model_opinion_enabled is False
    assert settings.model_opinion_model == mo.REGISTERED_MODEL == "openai/gpt-5.6-sol"
    assert settings.model_opinion_effort == mo.REGISTERED_EFFORT == "medium"
    monkeypatch.setenv("MODEL_OPINION_ENABLED", "true")
    monkeypatch.setenv("MODEL_OPINION_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("MODEL_OPINION_EFFORT", "high")
    changed = Settings(_env_file=None)  # type: ignore[call-arg]
    assert changed.model_opinion_enabled is True
    assert (changed.model_opinion_model, changed.model_opinion_effort) == (
        "anthropic/claude-sonnet-5",
        "high",
    )


def test_the_example_env_file_lists_the_settings_with_their_defaults():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "MODEL_OPINION_ENABLED=false" in lines
    assert f"MODEL_OPINION_MODEL={mo.REGISTERED_MODEL}" in lines
    assert f"MODEL_OPINION_EFFORT={mo.REGISTERED_EFFORT}" in lines


# ----- 2. when it can be asked


def test_status_is_ready_with_a_key_and_a_cloud_model(tmp_path: Path):
    service, _ = _service(tmp_path, client=Client())
    assert service.model_opinion_status() == READY
    assert service.system_info()["model_opinion"] == READY
    assert READY["confident_probability"] == mo.CONFIDENT_PROBABILITY


@pytest.mark.parametrize(
    ("options", "status", "detail"),
    [
        ({"client": Client(), "enabled": False}, "disabled", "MODEL_OPINION_ENABLED"),
        # the default: nothing was switched on
        ({"client": None, "enabled": False}, "disabled", "MODEL_OPINION_ENABLED"),
        # local mode: no LLM at all
        ({"client": None}, "needs an OpenRouter key", "OPENROUTER_API_KEY"),
        # a client that only has the local model would answer in the cloud model's name
        ({"client": Client(api_key=None)}, "needs an OpenRouter key", "OPENROUTER_API_KEY"),
        ({"client": Client(api_key="")}, "needs an OpenRouter key", "OPENROUTER_API_KEY"),
        ({"client": Client(), "model": LOCAL_MODEL}, "needs a cloud model", LOCAL_MODEL),
    ],
)
def test_the_model_is_not_asked_when_it_cannot_be(
    tmp_path: Path, options: dict[str, Any], status: str, detail: str
):
    service, state = _service(tmp_path, **options)
    before = _snapshot(state)
    reported = service.model_opinion_status()
    assert set(reported) == set(READY)
    assert (reported["status"], reported["available"]) == (status, False)
    assert reported["enabled"] is options.get("enabled", True)
    assert reported["model"] == options.get("model", MODEL)
    assert service.system_info()["model_opinion"] == reported
    with pytest.raises(ServiceUnavailable, match="second opinion cannot be asked") as refused:
        service.ask_model_opinion("c1")
    assert detail in str(refused.value)
    response = _api(service).post("/api/cases/c1/model-opinion")
    assert response.status_code == 409
    assert detail in response.json()["detail"]
    # nothing was sent, stored or changed
    client = options.get("client")
    assert client is None or client.calls == []
    assert service._model_opinions == {}  # noqa: SLF001
    assert service.model_opinion("c1", state) is None
    assert _snapshot(state) == before


def test_status_never_raises(tmp_path: Path):
    class Broken:
        @property
        def model_opinion_enabled(self) -> bool:
            raise RuntimeError("settings cannot be read")

    service, _ = _service(tmp_path, client=Client())
    service.settings = Broken()  # type: ignore[assignment]
    assert service.model_opinion_status()["status"] == "disabled"
    # settings from before the feature existed: off, with the registered model as the default
    service.settings = SimpleNamespace()  # type: ignore[assignment]
    assert service.model_opinion_status() == {
        **READY,
        "enabled": False,
        "available": False,
        "status": "disabled",
    }
    # an engine without a client
    service.settings = SimpleNamespace(model_opinion_enabled=True)  # type: ignore[assignment]
    service.engine = SimpleNamespace()  # type: ignore[assignment]
    assert service.model_opinion_status()["status"] == "needs an OpenRouter key"


# ----- 3. asking


def test_asking_stores_the_answer_beside_the_case_and_changes_nothing_else(tmp_path: Path):
    client = Client()
    service, state = _service(tmp_path, client=client)
    before = _snapshot(state)
    assert service.model_opinion("c1", state) is None

    assert service.ask_model_opinion("c1") is state

    # one call, with the tested question and the settings' model and effort
    [(messages, options)] = client.calls
    assert (options["model"], options["effort"]) == (MODEL, EFFORT)
    assert (options["schema_name"], options["max_tokens"]) == (mo.SCHEMA_NAME, mo.MAX_TOKENS)
    assert options["schema"] == mo.SCHEMA and options.get("seed") is None
    assert messages[0] == {"role": "system", "content": mo.SYSTEM}
    # the positive statements only, with the ATT&CK group name hidden by the service's own kb
    assert messages[1]["content"] == mo.prompt_for(
        [
            "the adversary sent spearphishing e-mails with a malicious attachment.",
            "The actor dumped credentials from LSASS memory.",
        ]
    )
    assert "FIN7" not in messages[1]["content"]

    stored = service._model_opinions["c1"]  # noqa: SLF001
    assert isinstance(stored, mo.ModelOpinion)
    assert (stored.model, stored.requested_model, stored.effort) == (MODEL, MODEL, EFFORT)

    block = service.model_opinion("c1", state)
    assert block is not None and set(block) == PAYLOAD_KEYS
    assert block["first"] == {
        "name": "FIN7",
        "group_id": "G0046",  # by name: the model's own id (G0999) is ignored
        "attack_name": "FIN7",
        "probability": 0.99,
    }
    assert block["candidates"] == [block["first"]]
    assert (block["decision"], block["confident_name"]) == ("name", True)
    assert block["confident_rule"] == {"probability": 0.99, "applies": True}
    assert block["agreement"]["kind"] == "same"
    assert (block["statements_sent"], block["statements_total"], block["names_masked"]) == (2, 2, 1)
    assert (block["prompt_tokens"], block["completion_tokens"], block["cost_usd"]) == (
        3625,
        1808,
        0.0543,
    )
    assert (block["model"], block["effort"], block["prompt_version"]) == (MODEL, EFFORT, 1)
    assert (block["rationale"], block["stale"], block["note"]) == (RATIONALE, False, mo.NOTE)
    json.dumps(block)  # plain JSON

    # display only: the state is as the product left it and nothing was saved with the case
    assert _snapshot(state) == before
    assert service.store.list() == []
    assert not list((tmp_path / "cases").glob("*.json"))


def test_the_block_compares_the_answer_with_the_products_own(tmp_path: Path):
    """The service hands the block its knowledge base and the state it is shown with."""
    for name, kind in (("Carbanak", "same_activity"), ("APT29", "shortlist"), ("Nobody", None)):
        service, state = _service(tmp_path / name, client=Client(_answer(name)))
        service.ask_model_opinion("c1")
        block = service.model_opinion("c1", state)
        assert block["agreement"]["kind"] == (kind or "outside_attack")
        assert block["agreement"]["text"]
        if kind is None:
            assert block["first"]["group_id"] is None and block["confident_name"] is False
    # the comparison is made with the state of the read, not the one the model was asked on
    service, state = _service(tmp_path / "moved", client=Client(_answer("APT29")))
    other = _case_state(service.engine.kb, leader="G0016")
    service.ask_model_opinion("c1")
    assert service.model_opinion("c1", state)["agreement"]["kind"] == "shortlist"
    assert service.model_opinion("c1", other)["agreement"]["kind"] == "same"


def test_another_model_is_asked_as_configured_but_never_gives_a_confident_name(tmp_path: Path):
    client = Client()
    service, state = _service(
        tmp_path, client=client, model="anthropic/claude-sonnet-5", effort="high"
    )
    status = service.model_opinion_status()
    assert (status["status"], status["model"], status["effort"]) == (
        "ready",
        "anthropic/claude-sonnet-5",
        "high",
    )
    assert status["registered_model"] == MODEL
    service.ask_model_opinion("c1")
    assert (client.calls[0][1]["model"], client.calls[0][1]["effort"]) == (
        "anthropic/claude-sonnet-5",
        "high",
    )
    block = service.model_opinion("c1", state)
    assert block["first"]["probability"] == 0.99 and block["decision"] == "name"
    assert block["confident_name"] is False
    assert block["confident_rule"] == {"probability": 0.99, "applies": False}


def test_no_lock_is_held_while_the_model_answers(tmp_path: Path):
    seen: dict[str, Any] = {}

    def while_the_model_answers() -> None:
        seen["engine lock free"] = _free_from_another_thread(service._engine_lock)  # noqa: SLF001
        seen["case lock free"] = _free_from_another_thread(service._lock("c1"))  # noqa: SLF001

        def read_the_case() -> None:
            seen["case read"] = service.get("c1")[0]

        # a daemon: a reader left waiting for a lock fails the test, it does not hang the run
        reader = threading.Thread(target=read_the_case, daemon=True)
        reader.start()
        reader.join(10)

    service, state = _service(tmp_path, client=Client(during=while_the_model_answers))
    # the probe does see a held lock
    with service._engine_lock:  # noqa: SLF001
        assert not _free_from_another_thread(service._engine_lock)  # noqa: SLF001
    service.ask_model_opinion("c1")
    assert seen == {"engine lock free": True, "case lock free": True, "case read": state}


def test_a_second_request_for_the_same_case_is_refused_while_the_first_is_pending(tmp_path: Path):
    """Two clicks are two paid calls: while the model answers, the case takes no second one."""
    attempts: list[int] = []
    refused: list[str] = []

    def ask_again() -> None:
        if attempts:  # only the first call asks again while it is pending
            return
        attempts.append(1)
        with pytest.raises(ServiceUnavailable) as second:
            service.ask_model_opinion("c1")
        refused.append(str(second.value))
        service.ask_model_opinion("c2")  # another case is not held up

    client = Client(during=ask_again)
    service, state = _service(tmp_path, client=client)
    service._states["c2"] = _case_state(service.engine.kb)  # noqa: SLF001
    service._meta["c2"] = CaseMeta(case_id="c2", name="another case")  # noqa: SLF001
    service.ask_model_opinion("c1")
    assert len(refused) == 1 and "already answering" in refused[0]
    assert len(client.calls) == 2  # one for each case
    assert sorted(service._model_opinions) == ["c1", "c2"]  # noqa: SLF001
    assert service.model_opinion("c1", state) is not None
    # once the answer is there the case can be asked again, also after a failed call
    service.ask_model_opinion("c1")
    assert len(client.calls) == 3
    client.error = OpenRouterError("OpenRouter is unreachable")
    with pytest.raises(OpenRouterError):
        service.ask_model_opinion("c1")
    client.error = None
    service.ask_model_opinion("c1")
    assert len(client.calls) == 5
    assert service._model_opinions_pending == set()  # noqa: SLF001


# ----- 4. keeping the answer


def test_the_answer_survives_reads_and_is_stale_once_the_case_text_changes(tmp_path: Path):
    client = Client()
    service, state = _service(tmp_path, client=client)
    service.ask_model_opinion("c1")
    first = service.model_opinion("c1", state)
    assert first == service.model_opinion("c1", state) and first["stale"] is False

    # a re-analysis puts a new state in place: the answer stays, and with the same text it is
    # not stale
    again = _case_state(service.engine.kb)
    service._states["c1"] = again  # noqa: SLF001
    assert service.model_opinion("c1", again) == first

    # evidence that is not sent to the model does not make the answer stale
    again.ledger.observations.append(_observation("No e-mail was found.", EvidenceStatus.NEGATIVE))
    assert service.model_opinion("c1", again)["stale"] is False
    again.ledger.observations.append(_observation("A scheduled task ran the payload hourly."))
    stale = service.model_opinion("c1", again)
    assert stale["stale"] is True
    assert {key: value for key, value in stale.items() if key != "stale"} == {
        key: value for key, value in first.items() if key != "stale"
    }

    # asking again replaces the answer
    client.parsed = _answer("APT29", 0.6, "open")
    assert service.ask_model_opinion("c1") is again
    fresh = service.model_opinion("c1", again)
    assert (fresh["stale"], fresh["statements_sent"]) == (False, 3)
    assert (fresh["first"]["group_id"], fresh["decision"]) == ("G0016", "open")
    assert fresh["confident_name"] is False
    assert len(service._model_opinions) == 1  # noqa: SLF001


def test_text_added_while_the_model_answers_makes_the_answer_stale(tmp_path: Path):
    """The answer is for the text the case had when it was asked."""

    def add_a_statement() -> None:
        state.ledger.observations.append(_observation("A scheduled task ran the payload hourly."))

    service, state = _service(tmp_path, client=Client(during=add_a_statement))
    service.ask_model_opinion("c1")
    block = service.model_opinion("c1", state)
    assert (block["statements_sent"], block["statements_total"], block["stale"]) == (2, 2, True)


def test_deleting_the_case_drops_the_answer(tmp_path: Path):
    service, state = _service(tmp_path, client=Client())
    service.ask_model_opinion("c1")
    assert "c1" in service._model_opinions  # noqa: SLF001
    service.delete("c1")
    assert service._model_opinions == {}  # noqa: SLF001
    assert service.model_opinion("c1", state) is None


def test_a_case_deleted_while_the_model_answers_keeps_no_answer(tmp_path: Path):
    client = Client(during=lambda: service.delete("c1"))
    service, _ = _service(tmp_path, client=client)
    with pytest.raises(CaseNotFound):
        service.ask_model_opinion("c1")
    assert len(client.calls) == 1
    assert service._model_opinions == {}  # noqa: SLF001
    assert service._model_opinions_pending == set()  # noqa: SLF001


def test_a_failing_block_never_breaks_a_case_view(tmp_path: Path, monkeypatch):
    service, state = _service(tmp_path, client=Client())
    service.ask_model_opinion("c1")
    assert service.model_opinion("c1", state) is not None

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the block cannot be built")

    with monkeypatch.context() as patch:
        patch.setattr(mo, "payload", broken)
        assert service.model_opinion("c1", state) is None
        assert service.model_opinion("another case", state) is None
    # a state the block cannot read is no reason to fail either
    assert service.model_opinion("c1", None)["stale"] is True
    assert service.model_opinion("c1", state)["stale"] is False


# ----- 5. failures


def test_a_failed_call_keeps_the_earlier_answer(tmp_path: Path):
    client = Client()
    service, state = _service(tmp_path, client=client)
    service.ask_model_opinion("c1")
    earlier = service.model_opinion("c1", state)
    api = _api(service, raise_server_exceptions=False)

    client.error = OpenRouterError("OpenRouter returned 502")
    with pytest.raises(OpenRouterError, match="502"):
        service.ask_model_opinion("c1")
    failed = api.post("/api/cases/c1/model-opinion")
    assert failed.status_code == 500 and "502" in failed.json()["detail"]

    # an answer that cannot be read, and an answer a local model gave, are not shown
    client.error = None
    client.parsed = {"candidates": [], "decision": "maybe", "rationale": ""}
    with pytest.raises(OpenRouterError, match="unknown decision"):
        service.ask_model_opinion("c1")
    client.parsed = _answer("APT29")
    client.answered_by = LOCAL_MODEL
    with pytest.raises(OpenRouterError, match="local model"):
        service.ask_model_opinion("c1")

    assert service.model_opinion("c1", state) == earlier
    assert service._model_opinions_pending == set()  # noqa: SLF001


def test_a_refusal_of_the_call_itself_is_reported_as_unavailable(tmp_path: Path, monkeypatch):
    """The module checks the key again when it is asked; its refusal is a 409, not a 500."""
    service, state = _service(tmp_path, client=Client())

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise mo.ModelOpinionUnavailable("the model's second opinion needs an OpenRouter key")

    monkeypatch.setattr(mo, "ask", refuse)
    with pytest.raises(ServiceUnavailable, match="needs an OpenRouter key"):
        service.ask_model_opinion("c1")
    response = _api(service).post("/api/cases/c1/model-opinion")
    assert response.status_code == 409
    assert "needs an OpenRouter key" in response.json()["detail"]
    assert service.model_opinion("c1", state) is None
    assert service._model_opinions_pending == set()  # noqa: SLF001


def test_a_case_without_statements_is_a_bad_request(tmp_path: Path):
    client = Client()
    service, state = _service(tmp_path, client=client)
    state.ledger.observations[:] = [_observation("Nothing was seen.", EvidenceStatus.NEGATIVE)]
    with pytest.raises(ValueError, match="no statements to send"):
        service.ask_model_opinion("c1")
    response = _api(service).post("/api/cases/c1/model-opinion")
    assert response.status_code == 400
    assert response.json()["detail"] == "the case has no statements to send"
    assert client.calls == [] and service._model_opinions == {}  # noqa: SLF001


def test_an_unknown_case_is_not_found_and_costs_nothing(tmp_path: Path):
    client = Client()
    service, _ = _service(tmp_path, client=client)
    with pytest.raises(CaseNotFound):
        service.ask_model_opinion("missing")
    assert _api(service).post("/api/cases/missing/model-opinion").status_code == 404
    assert client.calls == []
    assert service._model_opinions_pending == set()  # noqa: SLF001


def test_asking_needs_a_signed_in_user(tmp_path: Path):
    from tests.test_api import StubService
    from tests.test_api_auth import google_settings

    class Recording(StubService):
        def ask_model_opinion(self, case_id: str) -> Any:
            self.calls.append(("model_opinion", case_id))
            return super().ask_model_opinion(case_id)

    service = Recording()
    app = create_app(
        service,  # type: ignore[arg-type]
        serve_web=False,
        settings=google_settings(),
        users=UserStore(tmp_path / "users.db"),
    )
    response = TestClient(app).post(
        "/api/cases/c1/model-opinion", headers={"X-Requested-With": "Vestige"}
    )
    assert response.status_code == 401
    assert service.calls == []


# ----- 6. the API round trip on a real analysis state


def test_the_api_shows_the_answer_and_nothing_else_changes(tmp_path: Path, monkeypatch):
    from tests.test_software_questions import LSASS, POWERSHELL, software_engine

    engine = software_engine(
        model_opinion_enabled=True, model_opinion_model=MODEL, model_opinion_effort=EFFORT
    )
    service = CaseService(engine.settings, engine=engine, cases_dir=tmp_path / "cases")
    case_id = service.create_case(
        IntakePayload(narrative=f"{POWERSHELL} {LSASS} Charlie staged the tools on a share.")
    )
    api = _api(service)
    path = f"/api/cases/{case_id}"
    weights = {"weights": {"software_tools": 2.0}}
    # the re-analysis the answer must not change, made once before the model is asked
    assert api.post(f"{path}/weights", json=weights).status_code == 200
    # the case was built without any model; from here on only the second opinion has a client.
    # The model names a group the product ranks but does not put first, so a rule that took
    # the model's first choice would show in the decision.
    client = Client(_answer("Alpha"))
    engine.llm = client  # type: ignore[assignment]
    analyses: list[str] = []
    real_analyze = engine.analyze

    def counted_analyze(*args: Any, **kwargs: Any) -> Any:
        analyses.append("analyze")
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(engine, "analyze", counted_analyze)
    monkeypatch.setattr(engine, "rescore", _never)
    monkeypatch.setattr(engine, "run_review", _never)

    def exports() -> dict[str, tuple[str, str, str]]:
        return {kind: service.export(case_id, kind) for kind in EXPORT_KINDS}

    def saved_files() -> dict[str, str]:
        return {
            file.name: file.read_text(encoding="utf-8")
            for file in sorted((tmp_path / "cases").glob("*.json"))
        }

    assert api.get("/api/system").json()["model_opinion"] == READY
    before = api.get(path).json()
    assert "model_opinion" in before and before["model_opinion"] is None
    assert before["decision"]["leading_group_id"] == "G3"
    exports_before, saved_before = exports(), saved_files()
    assert saved_before

    asked = api.post(f"{path}/model-opinion")
    assert asked.status_code == 200, asked.text
    case = asked.json()
    block = case["model_opinion"]
    assert set(block) == PAYLOAD_KEYS
    assert block["first"] == {
        "name": "Alpha",
        "group_id": "G1",
        "attack_name": "Alpha",
        "probability": 0.99,
    }
    assert block["confident_name"] is True and block["agreement"]["kind"] == "shortlist"
    assert (block["statements_sent"], block["names_masked"], block["stale"]) == (3, 1, False)
    assert block["rationale"] == RATIONALE and block["note"] == mo.NOTE
    assert len(client.calls) == 1
    assert "Charlie" not in client.calls[0][0][1]["content"]

    # the answer is the only difference in the case, and a second read shows it again
    assert {k: v for k, v in case.items() if k != "model_opinion"} == {
        k: v for k, v in before.items() if k != "model_opinion"
    }
    assert api.get(path).json() == case
    # no analysis ran, no export and no saved file carries anything of the answer
    assert analyses == []
    exports_after, saved_after = exports(), saved_files()
    assert exports_after == exports_before and saved_after == saved_before
    for content in [*(export[0] for export in exports_after.values()), *saved_after.values()]:
        assert RATIONALE not in content and "model_opinion" not in content

    # a block that cannot be built leaves the case view whole
    with monkeypatch.context() as patch:
        patch.setattr(mo, "payload", _never)
        broken = api.get(path)
        assert broken.status_code == 200 and broken.json()["model_opinion"] is None

    # a real re-analysis (run without a model, like the first one) keeps the answer
    engine.llm = None  # type: ignore[assignment]
    monkeypatch.undo()
    rescored = api.post(f"{path}/weights", json=weights)
    assert rescored.status_code == 200, rescored.text
    assert rescored.json()["weights"]["software_tools"] == 2.0
    assert rescored.json()["model_opinion"] == block
    # the same re-analysis gives the same case with a confident answer stored: decision,
    # ranking, Top-three and summary are the product's own
    apart = {"model_opinion", "updated_at"}
    assert {k: v for k, v in rescored.json().items() if k not in apart} == {
        k: v for k, v in before.items() if k not in apart
    }
    # ... and neither the file this re-analysis saved nor an export carries anything of it
    assert exports() == exports_before
    for content in [*(export[0] for export in exports().values()), *saved_files().values()]:
        assert RATIONALE not in content and "model_opinion" not in content
    # without a key the answer is still shown; only asking again is refused
    assert api.get("/api/system").json()["model_opinion"]["status"] == "needs an OpenRouter key"
    assert api.post(f"{path}/model-opinion").status_code == 409

    # new case text marks it stale
    state, _ = service.get(case_id)
    extra = state.ledger.observations[0].model_copy(
        update={"observation_id": "obs-added-later", "text": "A scheduled task ran hourly."}
    )
    state.ledger.observations.append(extra)
    assert api.get(path).json()["model_opinion"]["stale"] is True

    assert api.delete(path).status_code == 204
    assert service._model_opinions == {}  # noqa: SLF001
    assert api.get(path).status_code == 404
