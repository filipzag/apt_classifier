"""HTTP API tests.

The first group runs against a stub service and needs nothing installed beyond FastAPI: it
pins the error mapping and the request models. The second group drives the real local pipeline
(no LLM) through the API and is skipped when the ingested knowledge base or Chroma is missing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from apt_classifier.service import CaseNotFound, ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb" / "attack_kb.json"


class StubService:
    local_mode = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def system_info(self) -> dict[str, Any]:
        return {"mode": "local", "cases": 0}

    def sample_text(self) -> str:
        return "sample"

    def search_techniques(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        self.calls.append(("search", (query, limit)))
        return [{"id": "T1059", "name": "Command and Scripting Interpreter"}][:limit]

    def list_cases(self) -> list[Any]:
        return []

    def get(self, case_id: str) -> tuple[Any, Any]:
        raise CaseNotFound(case_id)

    def delete(self, case_id: str) -> bool:
        return False

    def run_review(self, case_id: str) -> Any:
        raise ServiceUnavailable("needs a key")

    def apply_decisions(self, case_id: str, decisions: list[dict[str, Any]]) -> Any:
        raise ValueError("bad replacement")

    def set_target_platforms(self, case_id: str, platforms: Any) -> Any:
        self.calls.append(("platforms", (case_id, platforms)))
        raise ValueError("unknown operating system(s): Solaris")

    def export(self, case_id: str, kind: str) -> tuple[str, str, str]:
        raise CaseNotFound(case_id)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(StubService(), serve_web=False))  # type: ignore[arg-type]


def test_health_and_system(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/system").json()["mode"] == "local"
    assert client.get("/api/sample").json() == {"text": "sample"}


def test_technique_search_clamps_limit(client: TestClient) -> None:
    response = client.get("/api/techniques", params={"q": "t1059", "limit": 500})
    assert response.status_code == 200
    assert response.json()[0]["id"] == "T1059"


def test_error_mapping(client: TestClient) -> None:
    assert client.get("/api/cases/missing").status_code == 404
    assert client.delete("/api/cases/missing").status_code == 404
    assert client.post("/api/cases/x/review").status_code == 409
    response = client.post(
        "/api/cases/x/decisions",
        json={"decisions": [{"observation_id": "o", "technique_id": "T1", "action": "replace"}]},
    )
    assert response.status_code == 400
    assert "bad replacement" in response.json()["detail"]
    assert client.get("/api/cases/x/export/pdf").status_code == 404


def test_service_not_started_is_503() -> None:
    app = create_app(None, serve_web=False)
    response = TestClient(app).get("/api/system")  # no lifespan: service never built
    assert response.status_code == 503


def test_create_case_rejects_bad_phases(client: TestClient) -> None:
    response = TestClient(client.app).post("/api/cases", data={"narrative": "x", "phases": "nope"})
    assert response.status_code == 400


def test_target_os_errors_are_bad_requests(client: TestClient) -> None:
    response = client.post("/api/cases/x/platforms", json={"target_platforms": ["Solaris"]})
    assert response.status_code == 400
    assert "Solaris" in response.json()["detail"]
    assert (
        client.post("/api/cases/x/platforms", json={"target_platforms": "Linux"}).status_code == 422
    )


# ----- real local pipeline -----

pytestmark_local = pytest.mark.skipif(
    not KB.exists() or os.environ.get("APT_SKIP_APP_FLOW") == "1",
    reason="requires the ingested knowledge base and Chroma index",
)


@pytest.fixture(scope="module")
def local_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    from apt_classifier.config import get_settings
    from apt_classifier.service import CaseService

    # blank key and local LLM off override .env: local mode
    overrides = {"OPENROUTER_API_KEY": "", "LOCAL_LLM_ENABLED": "false"}
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()
    try:
        service = CaseService(cases_dir=tmp_path_factory.mktemp("cases"))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()
    assert service.local_mode
    return TestClient(create_app(service, serve_web=False), raise_server_exceptions=False)


@pytestmark_local
def test_local_flow_end_to_end(local_client: TestClient) -> None:
    sample = local_client.get("/api/sample").json()["text"]
    created = local_client.post(
        "/api/cases", data={"name": "API flow", "narrative": sample, "iocs": "Mimikatz"}
    )
    assert created.status_code == 201, created.text
    case = created.json()
    case_id = case["case_id"]
    assert case["observations"] and case["candidates"] and case["ranking"]
    assert case["decision"]["stage"] in {
        "insufficient_evidence",
        "unresolved_activity_cluster",
        "known_group_shortlist",
        "known_group_hypothesis",
    }
    assert {"maturity", "margin", "separation", "gate_mode"} <= set(case["decision"]["thresholds"])
    assert local_client.get("/api/cases").json()[0]["case_id"] == case_id

    # interview: a question is selected deterministically and an answer re-ranks
    interview = local_client.get(f"/api/cases/{case_id}/interview").json()
    assert interview["question"] is not None and interview["leads"]
    answered = local_client.post(
        f"/api/cases/{case_id}/interview/answer", json={"status": "yes", "text": "seen"}
    ).json()
    assert len(answered["answered"]) == 1
    assert local_client.post(f"/api/cases/{case_id}/interview/restart").json()["answered"] == []

    # technique decision on a needs-review candidate
    pending = next((c for c in case["candidates"] if c["state"] == "needs_review"), None)
    if pending is not None:
        updated = local_client.post(
            f"/api/cases/{case_id}/decisions",
            json={
                "decisions": [
                    {
                        "observation_id": pending["observation_id"],
                        "technique_id": pending["technique_id"],
                        "action": "confirm",
                    }
                ]
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["decisions"][0]["action"] == "confirm"

    distinguish = local_client.get(f"/api/cases/{case_id}/distinguish?top_k=10").json()
    assert distinguish["top_k"] == 10 and len(distinguish["shortlist"]) <= 10
    assert {"separation", "techniques", "pending", "leads", "leader"} <= set(distinguish)
    assert all("delta_separation" in item for item in distinguish["pending"])
    assert all({"delta_yes", "delta_no"} <= set(item) for item in distinguish["leads"])
    # by default the view replays the naming gate's own window: same separation as the decision
    gate_view = local_client.get(f"/api/cases/{case_id}/distinguish").json()
    current = local_client.get(f"/api/cases/{case_id}").json()
    assert gate_view["top_k"] == max(2, gate_view["gate_top_k"])
    assert gate_view["leader"]["group_id"] == current["decision"]["leading_group_id"]
    assert gate_view["separation"] == pytest.approx(current["decision"]["separation"], abs=1e-3)

    top = case["ranking"][0]["group_id"]
    dossier = local_client.get(f"/api/cases/{case_id}/dossier/{top}").json()
    assert dossier["group_id"] == top and "matched_techniques" in dossier
    canvas = local_client.get(f"/api/cases/{case_id}/canvas").json()
    assert canvas["nodes"] and canvas["edges"]
    diamond = local_client.get(f"/api/cases/{case_id}/diamond").json()
    assert {n["vertex"] for n in diamond["nodes"]} <= {
        "adversary",
        "capability",
        "infrastructure",
        "victim",
    }
    for kind in ("markdown", "json", "navigator", "attack-flow"):
        response = local_client.get(f"/api/cases/{case_id}/export/{kind}")
        assert response.status_code == 200, kind
        assert "attachment" in response.headers["content-disposition"]
        assert "vestige-" in response.headers["content-disposition"]
    assert local_client.post(f"/api/cases/{case_id}/review").status_code == 409

    # verifier verdicts saved with the case are replayed when it is reopened, even in local mode
    service = local_client.app.state.service
    current = local_client.get(f"/api/cases/{case_id}").json()
    pending_again = next(c for c in current["candidates"] if c["state"] == "needs_review")
    saved = service.store.load(case_id)
    service.store.save(
        case_id=case_id,
        name=saved["name"],
        ledger=saved["ledger"],
        weights=saved["weights"],
        verifications=[
            {
                "observation_id": pending_again["observation_id"],
                "technique_id": pending_again["technique_id"],
                "verdict": "supported",
                "reason": "test verdict",
                "support_span": "",
                "model": "test",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }
        ],
    )
    service._states.pop(case_id, None)  # noqa: SLF001 - force a reload from disk
    service._meta.pop(case_id, None)  # noqa: SLF001
    reopened = local_client.get(f"/api/cases/{case_id}").json()
    replayed = next(c for c in reopened["candidates"] if c["key"] == pending_again["key"])
    assert replayed["verification"] == {"verdict": "supported", "reason": "test verdict"}
    assert replayed["state"] == "auto_accepted"
    assert reopened["retrieval_diagnostics"]["verifier_reused"] == 1
    assert (
        service.store.load(case_id)["verifications"][0]["technique_id"]
        == pending_again["technique_id"]
    )
    renamed = local_client.patch(f"/api/cases/{case_id}", json={"name": "Renamed"}).json()
    assert renamed["name"] == "Renamed"
    rescored = local_client.post(
        f"/api/cases/{case_id}/weights", json={"weights": {"software_tools": 2.0}}
    )
    assert rescored.status_code == 200 and rescored.json()["weights"]["software_tools"] == 2.0
    assert local_client.delete(f"/api/cases/{case_id}").status_code == 204
    assert local_client.get(f"/api/cases/{case_id}").status_code == 404


@pytestmark_local
def test_local_flow_target_os(local_client: TestClient) -> None:
    from apt_classifier.platforms import PlatformScope

    techniques = local_client.app.state.service.engine.kb.techniques
    linux_scope = PlatformScope(("Linux",), "case")

    def windows_only(case: dict[str, Any]) -> set[str]:
        return {
            c["technique_id"]
            for c in case["candidates"]
            if not linux_scope.allows(techniques[c["technique_id"]].platforms)
        }

    narrative = (
        "The attacker created a scheduled task that runs the payload every hour for persistence."
    )
    plain = local_client.post("/api/cases", data={"narrative": narrative}).json()
    assert plain["target_platforms"] == [] and windows_only(plain)  # e.g. T1053.005
    linux = local_client.post(
        "/api/cases", data={"narrative": narrative, "target_platforms": "linux"}
    ).json()
    assert linux["target_platforms"] == ["Linux"] and linux["platform_options"][0] == "Windows"
    assert linux["candidates"] and not windows_only(linux)
    scopes = [o["platform"] for o in linux["observations"] if o["platform"]]
    assert scopes and all(s == {"platforms": ["Linux"], "source": "case"} for s in scopes)

    # changing the Target OS maps the case again and is saved with it
    case_id = linux["case_id"]
    cleared = local_client.post(f"/api/cases/{case_id}/platforms", json={"target_platforms": []})
    assert cleared.status_code == 200 and cleared.json()["target_platforms"] == []
    assert windows_only(cleared.json())
    local_client.post(f"/api/cases/{case_id}/platforms", json={"target_platforms": ["Linux"]})
    saved = local_client.app.state.service.store.load(case_id)["ledger"]
    assert saved.metadata["target_platforms"] == ["Linux"]

    # the interview and the Distinguish leads never ask about a Windows-only technique
    def linux_compatible(technique_id: str) -> bool:
        return linux_scope.allows(techniques[technique_id].platforms)

    interview = local_client.get(f"/api/cases/{case_id}/interview").json()
    asked = [interview["question"]["technique_id"]] if interview["question"] else []
    asked += [lead["technique_id"] for lead in interview["leads"]]
    assert asked and all(linux_compatible(t) for t in asked)
    # and the investigation steps cite no Windows analytic ("[AN0983] Sources: WinEventLog...")
    analytics = local_client.app.state.service.engine.kb.analytics
    steps = list((interview["question"] or {}).get("investigation_steps", []))
    steps += [lead["first_step"] for lead in interview["leads"]]
    cited = {aid for step in steps for aid in re.findall(r"\[(AN\d+)\]", step)}
    assert all(linux_scope.allows(analytics[aid].platforms) for aid in cited if aid in analytics)
    assert not any("PRE systems" in step or "(None)" in step for step in steps)
    distinguish = local_client.get(f"/api/cases/{case_id}/distinguish?top_k=10").json()
    assert all(linux_compatible(lead["technique_id"]) for lead in distinguish["leads"])
    assert all(linux_compatible(row["technique_id"]) for row in distinguish["pending"])
    bad = local_client.post(
        f"/api/cases/{case_id}/platforms", json={"target_platforms": ["Solaris"]}
    )
    assert bad.status_code == 400
    refused = local_client.post(
        "/api/cases", data={"narrative": narrative, "target_platforms": "Solaris"}
    )
    assert refused.status_code == 400
    for case in (plain, linux):
        assert local_client.delete(f"/api/cases/{case['case_id']}").status_code == 204


# ----- software questions through the API (fake vector store, no index needed) -----

PILOT_KEYS = {
    "schema",
    "case_id",
    "question_id",
    "software_id",
    "software_name",
    "target_rank",
    "window_size",
    "asked_at",
    "answered_at",
    "answer",
    "note_present",
    "before",
    "after",
    "predicted_delta_found",
    "attack_version",
    "pipeline_fingerprint",
    "settings",
}
SNAPSHOT_KEYS = {"stage", "leader", "named_cluster", "separation", "combiner_probability"}


@pytest.fixture
def software_client(tmp_path: Path) -> TestClient:
    from apt_classifier.service import CaseService
    from tests.test_software_questions import software_engine

    engine = software_engine()
    service = CaseService(engine.settings, engine=engine, cases_dir=tmp_path / "cases")
    return TestClient(create_app(service, serve_web=False), raise_server_exceptions=False)


def test_software_questions_take_three_answers_and_log_a_pilot_record(
    software_client: TestClient,
) -> None:
    from tests.test_software_questions import LSASS, POWERSHELL

    created = software_client.post(
        "/api/cases",
        data={"narrative": f"{POWERSHELL} {LSASS}"},
        headers={"X-Requested-With": "Vestige"},
    )
    assert created.status_code == 201, created.text
    case_id = created.json()["case_id"]
    interview = software_client.get(f"/api/cases/{case_id}/interview").json()
    question = interview["question"]
    assert question["kind"] == "software" and question["software_id"].startswith("S")
    assert question["target_rank"] in {1, 2, 3}
    # "not visible" is not an answer to a software question
    refused = software_client.post(
        f"/api/cases/{case_id}/interview/answer", json={"status": "not_visible"}
    )
    assert refused.status_code == 400
    note = "seen on host WS-17 by analyst jane.doe@example.com"
    answers = []
    for status in ("not_checked", "not_found", "found"):
        state = software_client.post(
            f"/api/cases/{case_id}/interview/answer", json={"status": status, "text": note}
        ).json()
        answers.append(state["answered"][-1])
        if not state["question"] or state["question"].get("kind") != "software":
            break
    assert [item["status"] for item in answers] == ["unknown", "negative", "positive"]
    assert all(item["kind"] == "software" and item["software_id"] for item in answers)
    log = software_client.get(f"/api/cases/{case_id}/interview/pilot-log").json()
    assert [row["answer"] for row in log] == ["not_checked", "not_found", "found"]
    for row in log:
        assert set(row) == PILOT_KEYS
        assert set(row["before"]) == SNAPSHOT_KEYS and set(row["after"]) == SNAPSHOT_KEYS
        assert row["note_present"] is True and row["case_id"] == case_id
        text = repr(row)
        assert "jane.doe" not in text and "WS-17" not in text and "owner" not in text
    exported = software_client.get(f"/api/cases/{case_id}/export/pilot-log")
    assert exported.status_code == 200
    lines = [line for line in exported.text.splitlines() if line]
    assert len(lines) == len(log) and "pilot-log.jsonl" in exported.headers["content-disposition"]
    # the aggregate across cases is for administrators (sign-in off: the local admin)
    aggregate = software_client.get("/api/pilot/software-questions.jsonl")
    assert aggregate.status_code == 200
    assert aggregate.headers["content-type"].startswith("application/x-ndjson")
    assert len([line for line in aggregate.text.splitlines() if line]) == len(log)


def test_pilot_aggregate_is_admin_only(software_client: TestClient) -> None:
    from api.auth import current_user
    from apt_classifier.users import User

    analyst = User(
        id="u1",
        email="analyst@example.com",
        name="Analyst",
        picture="",
        role="analyst",
        active=True,
        created_at="",
    )
    software_client.app.dependency_overrides[current_user] = lambda: analyst
    try:
        assert software_client.get("/api/pilot/software-questions.jsonl").status_code == 403
        assert software_client.get("/api/cases").status_code == 200
    finally:
        software_client.app.dependency_overrides.clear()
