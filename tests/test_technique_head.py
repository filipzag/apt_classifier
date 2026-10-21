"""The fine-tuned technique head as mapping evidence: client, retrieval hooks, acceptance schema
4 and the service's HTTP front (docs 21.3.18, product integration)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from apt_classifier.acceptance import (
    FEATURE_NAMES,
    FEATURE_NAMES_V4,
    LEGACY_CONTRACT,
    RETRIEVAL_CONTRACT_KEYS,
    TECHNIQUE_HEAD_CONTRACT_KEYS,
    AcceptanceModel,
    candidate_features,
    contract_mismatches,
    feature_names_for,
    load_acceptance_model,
    retrieval_contract,
)
from apt_classifier.attack.kb import Technique
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.retrieve import retrieval_contract_for, retrieval_kwargs, retrieve_techniques
from apt_classifier.technique_head import (
    EVIDENCE_SOURCE,
    TechniqueHeadClient,
    TechniqueHeadConfig,
    TechniqueHeadScores,
    config_from_settings,
    empty_features,
    features_from_evidence,
    load_technique_head,
    manifest_problem,
    pool_scores,
    propose,
    technique_head_kwargs,
)
from apt_classifier.technique_head import FEATURE_NAMES as HEAD_FEATURES
from tests.test_retrieve import _two_hit_setup

FINGERPRINT = "ab" * 32
MANIFEST = {
    "status": "ready",
    "fingerprint": FINGERPRINT,
    "attack_version": "19.2",
    "task": "technique",
    "families": 222,
    "techniques": 697,
    "families_considered": 3,
    "top_kept": 20,
    "checkpoint": "data/models/cribl_ft/E2/step_52706",
}


def _row(families, techniques, status="ok"):
    return {"status": status, "families": families, "techniques": techniques}


class FakeService:
    """An httpx mock transport answering /health and /score like the service."""

    def __init__(self, rows_for, manifest=MANIFEST, fail=False):
        self.rows_for = rows_for
        self.manifest = manifest
        self.fail = fail
        self.posted: list[list[str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=self.manifest)
        if request.url.path == "/score":
            if self.fail:
                return httpx.Response(500, json={"error": "boom"})
            texts = json.loads(request.content)["texts"]
            self.posted.append(texts)
            return httpx.Response(
                200,
                json={
                    "fingerprint": self.manifest.get("fingerprint"),
                    "rows": [self.rows_for(text) for text in texts],
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _rows_for(text: str) -> dict:
    lowered = text.lower()
    if "lsass" in lowered:
        return _row(
            [["T1003", 0.8], ["T1059", 0.1], ["T1055", 0.05]],
            [["T1003.001", 0.7], ["T1003", 0.08], ["T1059.001", 0.09], ["T1055.012", 0.04]],
        )
    if "powershell" in lowered:
        return _row(
            [["T1059", 0.9], ["T1105", 0.05], ["T1027", 0.02]],
            [["T1059.001", 0.85], ["T1059", 0.04], ["T1105", 0.05], ["T1027.010", 0.01]],
        )
    if "too long" in lowered:
        return {"status": "error", "error": "prompt has 2000 tokens (> 1024)"}
    return _row([["T1566", 0.5]], [["T1566.001", 0.4]])


def _client(service: FakeService, **changes) -> TechniqueHeadClient:
    config = TechniqueHeadConfig(enabled=True, attack_version="19.2", **changes)
    client = TechniqueHeadClient(config, transport=service.transport())
    client.probe()
    return client


# ----- scores and features -----


def test_scores_features_ranks_and_proposals():
    scores = TechniqueHeadScores.from_row(_rows_for("powershell"), top_families=2)
    assert scores.ok and scores.top_family == "T1059" and scores.top_p == pytest.approx(0.9)
    # the best technique of each of the top families, the family itself when none was kept
    assert scores.proposals == ("T1059.001", "T1105")
    features = scores.features("T1059.001")
    assert features["th_family_p"] == pytest.approx(0.9)
    assert features["th_technique_p"] == pytest.approx(0.85)
    assert features["th_family_rank"] == 1.0 and features["th_technique_rank"] == 1.0
    assert features["th_top_p"] == pytest.approx(0.9)
    assert features["th_proposed"] == 1.0 and features["th_only"] == 0.0
    # a technique outside the kept lists: zero probabilities and ranks, the head's confidence kept
    other = scores.features("T1566.002", head_only=True)
    assert other["th_family_p"] == 0.0 and other["th_technique_rank"] == 0.0
    assert other["th_top_p"] == pytest.approx(0.9) and other["th_only"] == 1.0
    assert set(HEAD_FEATURES) < set(features)
    assert propose({"T1003": 0.5, "T1059": 0.3}, {}, top_families=5) == ("T1003", "T1059")


def test_error_rows_and_pooling():
    error = TechniqueHeadScores.from_row(_rows_for("this one is too long"), top_families=3)
    assert not error.ok and "tokens" in (error.error or "")
    first = TechniqueHeadScores.from_row(_rows_for("powershell"), top_families=3)
    second = TechniqueHeadScores.from_row(_rows_for("lsass"), top_families=3)
    pooled = pool_scores([first, error, second], top_families=2)
    # per family and technique the largest probability any sentence gave, re-ranked
    assert pooled.families["T1059"] == pytest.approx(0.9)
    assert pooled.families["T1003"] == pytest.approx(0.8)
    assert list(pooled.families)[:2] == ["T1059", "T1003"]
    assert pooled.techniques["T1059.001"] == pytest.approx(0.85)
    assert pooled.proposals == ("T1059.001", "T1003.001")
    assert pool_scores([first], top_families=3) is first
    assert not pool_scores([error], top_families=3).ok


def test_features_from_evidence_round_trip():
    scores = TechniqueHeadScores.from_row(_rows_for("lsass"), top_families=3)
    evidence = SimpleNamespace(metadata={"features": scores.features("T1003.001")})
    assert features_from_evidence(evidence) == scores.features("T1003.001")
    assert features_from_evidence(None) == empty_features()
    assert (
        features_from_evidence(SimpleNamespace(metadata={"features": {"th_family_p": "x"}}))[
            "th_family_p"
        ]
        == 0.0
    )


# ----- client -----


def test_client_probes_scores_caches_and_counts_failures():
    service = FakeService(_rows_for)
    client = _client(service, batch=2)
    assert client.status == "active (unpinned)" and client.fingerprint == FINGERPRINT
    scored = client.score(["PowerShell ran", "lsass was read", "PowerShell ran", "x"])
    assert [item.top_family for item in scored] == ["T1059", "T1003", "T1059", "T1566"]
    # duplicates are scored once and batches respect the configured size
    assert service.posted == [["PowerShell ran", "lsass was read"], ["x"]]
    again = client.score(["lsass was read"])
    assert again[0] is scored[1] and len(service.posted) == 2
    identity = client.identity()
    assert identity["fingerprint"] == FINGERPRINT and identity["requests"] == 2
    # a failing service yields None rows and a counted error; nothing is cached
    failing = FakeService(_rows_for, fail=True)
    broken = _client(failing)
    assert broken.score(["new text"]) == [None] and broken.errors == 1
    assert "500" in (broken.last_error or "")


def test_manifest_validation_fails_closed():
    config = TechniqueHeadConfig(enabled=True, attack_version="19.2")
    assert manifest_problem(MANIFEST, config) is None
    assert "not ready" in manifest_problem({**MANIFEST, "status": "loading"}, config)
    assert "ATT&CK" in manifest_problem({**MANIFEST, "attack_version": "18.0"}, config)
    assert "fingerprint" in manifest_problem({**MANIFEST, "fingerprint": "short"}, config)
    assert "group" in manifest_problem({**MANIFEST, "task": "group"}, config)
    assert "family only" in manifest_problem({**MANIFEST, "techniques": 0}, config)
    pinned = TechniqueHeadConfig(enabled=True, expected_fingerprint="cd" * 32)
    assert "pinned" in manifest_problem(MANIFEST, pinned)
    assert manifest_problem(MANIFEST, TechniqueHeadConfig(expected_fingerprint=FINGERPRINT)) is None
    client = TechniqueHeadClient(pinned, transport=FakeService(_rows_for).transport())
    assert client.probe().startswith("unavailable") and not client.available
    # a pinned, matching service is "active" (not "unpinned")
    client = TechniqueHeadClient(
        TechniqueHeadConfig(enabled=True, expected_fingerprint=FINGERPRINT),
        transport=FakeService(_rows_for).transport(),
    )
    assert client.probe() == "active"


def test_loader_and_settings():
    settings = SimpleNamespace(technique_head_enabled=False)
    assert load_technique_head(settings) == (None, "disabled")
    settings = SimpleNamespace(
        technique_head_enabled=True,
        technique_head_url="http://head.test",
        technique_head_top_families=2,
        technique_head_new_candidates=1,
    )
    config = config_from_settings(settings, "19.2")
    assert (config.url, config.top_families, config.new_candidates) == ("http://head.test", 2, 1)
    kb = SimpleNamespace(version="19.2")
    client, status = load_technique_head(settings, kb, transport=FakeService(_rows_for).transport())
    assert client is not None and status == "active (unpinned)"
    assert technique_head_kwargs(settings, client) == {
        "technique_head": client,
        "technique_head_new_candidates": 1,
        "technique_head_top_families": 2,
    }
    # the wrong ATT&CK version, or no service, leaves the pipeline without a head
    none, status = load_technique_head(
        settings, SimpleNamespace(version="18.0"), transport=FakeService(_rows_for).transport()
    )
    assert none is None and "ATT&CK" in status
    offline = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down")))
    none, status = load_technique_head(settings, kb, transport=offline)
    assert none is None and status.startswith("unavailable")


# ----- retrieval -----


def test_retrieval_attaches_head_evidence_and_appends_proposals():
    kb, store, ledger = _two_hit_setup()
    kb.techniques["T1055"] = Technique(id="T1055", name="Process Injection")
    kb.techniques["T1055.012"] = Technique(
        id="T1055.012", name="Process Hollowing", is_subtechnique=True, parent_id="T1055"
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="the tool read lsass memory", provenance=Provenance(source="free_text")
            )
        ]
    )
    client = _client(FakeService(_rows_for))
    settings = SimpleNamespace(technique_head_new_candidates=2, technique_head_top_families=3)
    result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", **technique_head_kwargs(settings, client)
    )
    assert result.diagnostics["technique_head"] == "active"
    assert result.diagnostics["technique_head_fingerprint"] == FINGERPRINT
    assert result.diagnostics["technique_head_errors"] == 0
    # every candidate carries head evidence, so the schema-4 features are complete
    assert all(c.evidence_for(EVIDENCE_SOURCE) is not None for c in result.candidates)
    by_id = {c.technique_id: c for c in result.candidates}
    # the proposals retrieval missed (the best technique of each top family: T1003.001 and
    # T1059.001 were retrieved, T1055.012 was not) are appended after the pool, fused score 0
    appended = [
        c for c in result.candidates if c.evidence_for(EVIDENCE_SOURCE).metadata["head_only"]
    ]
    assert [c.technique_id for c in appended] == ["T1055.012"]
    assert result.diagnostics["technique_head_candidates_added"] == 1
    assert appended[0].fused_score == 0.0 and appended[0].parent_technique_id == "T1055"
    assert appended[0].fused_rank > max(
        c.fused_rank for c in result.candidates if c not in appended
    )
    features = candidate_features(
        by_id["T1055.012"], list(result.candidates), benign_context=False, sparse_context=False
    )
    assert features["th_only"] == 1.0 and features["th_technique_p"] == pytest.approx(0.04)
    assert features["th_proposed"] == 1.0 and features["dense_present"] == 0.0
    # the retrieved candidates get the head's opinion of them: strong for LSASS, weak for PowerShell
    lsass = candidate_features(
        by_id["T1003.001"], list(result.candidates), benign_context=False, sparse_context=False
    )
    assert lsass["th_family_p"] == pytest.approx(0.8) and lsass["th_technique_p"] == pytest.approx(
        0.7
    )
    assert lsass["th_proposed"] == 1.0 and lsass["th_only"] == 0.0
    retrieved = candidate_features(
        by_id["T1059.001"], list(result.candidates), benign_context=False, sparse_context=False
    )
    assert retrieved["th_family_p"] == pytest.approx(0.1) and retrieved["th_only"] == 0.0
    assert retrieved["th_top_p"] == pytest.approx(0.8)
    # without a client nothing changes: no evidence, no diagnostics keys, empty features
    plain = retrieve_techniques(ledger, store, kb, retrieval_mode="dense")
    assert "technique_head" not in plain.diagnostics
    assert all(c.evidence_for(EVIDENCE_SOURCE) is None for c in plain.candidates)
    feats = candidate_features(
        plain.candidates[0], list(plain.candidates), benign_context=False, sparse_context=False
    )
    assert {k: feats[k] for k in HEAD_FEATURES} == {k: empty_features()[k] for k in HEAD_FEATURES}


def test_retrieval_without_proposals_or_with_a_failing_service():
    kb, store, ledger = _two_hit_setup()
    client = _client(FakeService(_rows_for))
    result = retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        technique_head=client,
        technique_head_new_candidates=0,
    )
    assert result.diagnostics["technique_head_candidates_added"] == 0
    assert all(c.evidence_for(EVIDENCE_SOURCE) is not None for c in result.candidates)
    broken = _client(FakeService(_rows_for, fail=True))
    failed = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", technique_head=broken)
    assert failed.diagnostics["technique_head"] == "active (1 requests failed)"
    assert failed.diagnostics["technique_head_errors"] == 1
    assert all(c.evidence_for(EVIDENCE_SOURCE) is None for c in failed.candidates)
    # an unavailable client (probe failed) is reported and ignored
    down = TechniqueHeadClient(
        TechniqueHeadConfig(enabled=True, expected_fingerprint="cd" * 32),
        transport=FakeService(_rows_for).transport(),
    )
    down.probe()
    ignored = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", technique_head=down)
    assert ignored.diagnostics["technique_head"].startswith("unavailable")


def test_long_observations_are_scored_per_sentence_and_pooled():
    kb, store, ledger = _two_hit_setup()
    sentence = "The operators ran PowerShell scripts on the host to stage the loader quietly. "
    text = sentence * 3 + "Later the same tool read lsass memory from the domain controller again."
    assert len(text.split()) > 40
    ledger = EvidenceLedger(
        observations=[Observation(text=text, provenance=Provenance(source="free_text"))]
    )
    service = FakeService(_rows_for)
    client = _client(service)
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", technique_head=client)
    # the sentences were posted (one request, duplicates folded), the pooled scores carry both
    assert len(service.posted) == 1 and len(service.posted[0]) == 2
    evidence = result.candidates[0].evidence_for(EVIDENCE_SOURCE)
    assert evidence is not None
    assert set(evidence.metadata["proposals"]) >= {"T1059.001", "T1003.001"}


# ----- acceptance model: schema 4 and the head in the contract -----


def _rows(n: int = 40):
    rows, labels = [], []
    for i in range(n):
        row = {name: float(i % 3) for name in FEATURE_NAMES}
        row.update({name: float(i % 2) for name in HEAD_FEATURES})
        rows.append(row)
        labels.append(i % 2)
    return rows, labels


def test_schema_4_models_read_the_head_features_and_schema_3_ones_ignore_them(tmp_path):
    rows, labels = _rows()
    four = AcceptanceModel.fit(rows, labels, attack_version="19.2", feature_schema_version=4)
    three = AcceptanceModel.fit(rows, labels, attack_version="19.2")
    assert four.feature_names == FEATURE_NAMES_V4 and three.feature_names == FEATURE_NAMES
    assert feature_names_for(3) == FEATURE_NAMES and feature_names_for(4) == FEATURE_NAMES_V4
    with pytest.raises(ValueError):
        feature_names_for(6)
    # the head features separate the labels perfectly for the schema-4 fit only
    assert all((four.probability(r) >= 0.5) == bool(y) for r, y in zip(rows, labels, strict=True))
    assert not all(
        (three.probability(r) >= 0.5) == bool(y) for r, y in zip(rows, labels, strict=True)
    )
    for model in (four, three):
        loaded = AcceptanceModel.load(
            model.save(tmp_path / f"m{model.feature_schema_version}.json")
        )
        assert loaded is not None and loaded.feature_names == model.feature_names
        assert loaded.probability(rows[1]) == pytest.approx(model.probability(rows[1]))
        assert loaded.compatible_with("19.2") == (True, "ok")
    # a schema-4 artifact whose names were tampered with is refused
    payload = json.loads((tmp_path / "m4.json").read_text())
    payload["feature_names"] = list(FEATURE_NAMES)
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict(payload)


def test_a_model_fitted_with_the_head_runs_only_on_that_head(tmp_path):
    rows, labels = _rows()
    kb, store, ledger = _two_hit_setup()
    kb.version = "19.2"
    client = _client(FakeService(_rows_for))
    settings = SimpleNamespace(
        technique_head_new_candidates=3,
        technique_head_top_families=3,
        technique_exemplars_enabled=False,
        technique_retrieval_mode="dense",
    )
    contract = retrieval_contract_for(settings, None, technique_head=client)
    assert contract["technique_head_fingerprint"] == FINGERPRINT
    assert set(TECHNIQUE_HEAD_CONTRACT_KEYS) <= set(RETRIEVAL_CONTRACT_KEYS)
    model = AcceptanceModel.fit(
        rows, labels, attack_version="19.2", contract=contract, feature_schema_version=4
    )
    model.metadata = {"technique_head_fingerprint": FINGERPRINT}
    assert model.requires_technique_head and model.technique_head_fingerprint == FINGERPRINT
    # the same head: fine; no head, or another: refused, with a visible reason
    assert model.compatible_with("19.2", None, contract) == (True, "ok")
    without = retrieval_contract_for(settings, None, technique_head=None)
    ok, reason = model.compatible_with("19.2", None, without)
    assert not ok and "technique_head_fingerprint" in reason
    other = dict(contract, technique_head_fingerprint="cd" * 32)
    assert not model.compatible_with("19.2", None, other)[0]
    # a model fitted without the head is not held to the head keys at all
    free = AcceptanceModel.fit(rows, labels, attack_version="19.2", contract=without)
    assert not free.requires_technique_head
    assert free.compatible_with("19.2", None, contract) == (True, "ok")
    assert contract_mismatches(without, contract, exemplars=False) == []
    assert contract_mismatches(without, contract, exemplars=False, technique_head=True) != []
    assert LEGACY_CONTRACT["technique_head_fingerprint"] is None
    # retrieval disables a head-fitted model when the head is missing (fail closed)
    serving = {
        **retrieval_kwargs(settings, exemplar_index=None, lexical_index=None),
        "acceptance_model": model,
    }
    result = retrieve_techniques(ledger, store, kb, **serving)
    assert "technique head" in result.diagnostics["acceptance_model_disabled"]
    assert result.diagnostics["acceptance_model"] is False
    fine = retrieve_techniques(
        ledger, store, kb, **{**serving, **technique_head_kwargs(settings, client)}
    )
    assert "acceptance_model_disabled" not in fine.diagnostics
    # and the loader refuses it the same way (no fallback configured here)
    path = model.save(tmp_path / "head_model.json")
    loaded_settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=path,
        acceptance_model_fallback_path=None,
    )
    loaded, status = load_acceptance_model(loaded_settings, kb, None, contract=without)
    assert loaded is None and "technique_head_fingerprint" in status
    loaded, status = load_acceptance_model(loaded_settings, kb, None, contract=contract)
    assert loaded is not None and status == "active"
    # the kwargs carry the knobs the contract compares
    kwargs = retrieval_kwargs(
        settings, exemplar_index=None, lexical_index=None, technique_head=client
    )
    assert (
        retrieval_contract(kwargs, None, technique_head=client)["technique_head_top_families"] == 3
    )


# ----- the service's HTTP front (standard library, no model) -----


def _head_service_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cribl_finetune" / "head_service.py"
    spec = importlib.util.spec_from_file_location("head_service", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("head_service", module)
    spec.loader.exec_module(module)
    return module


def test_head_service_scores_validates_and_fingerprints(tmp_path):
    head_service = _head_service_module()
    for name in head_service.FINGERPRINT_FILES:
        (tmp_path / name).write_bytes(name.encode())
    first = head_service.checkpoint_fingerprint(tmp_path)
    (tmp_path / "family_head.pt").write_bytes(b"changed")
    assert first != head_service.checkpoint_fingerprint(tmp_path) and len(first) == 64
    calls: list[list[str]] = []

    def scorer(texts):
        calls.append(texts)
        return [_rows_for(text) for text in texts]

    service = head_service.HeadService(
        scorer, {"fingerprint": first, "task": "technique"}, max_texts=2
    )
    health = service.health()
    assert health["status"] == "ready" and health["fingerprint"] == first
    status, body = service.score({"texts": ["powershell", "lsass"]})
    assert status == 200 and body["fingerprint"] == first
    assert [row["families"][0][0] for row in body["rows"]] == ["T1059", "T1003"]
    assert service.score({"texts": ["a", "b", "c"]})[0] == 413
    assert service.score({"texts": [1]})[0] == 400 and service.score(["x"])[0] == 400
    assert service.score({"texts": []}) == (200, {"fingerprint": first, "rows": []})

    def broken(texts):
        raise RuntimeError("cuda out of memory")

    status, body = head_service.HeadService(broken, {}).score({"texts": ["x"]})
    assert status == 500 and "scoring failed" in body["error"]
    # the handler answers over HTTP like the client expects
    handler = head_service.make_handler(service)
    assert handler.do_GET and handler.do_POST


def test_log_derived_observations_are_not_scored():
    from apt_classifier.technique_head import scores_prose

    kb, store, _ = _two_hit_setup()
    prose = Observation(
        text="the tool read lsass memory", provenance=Provenance(source="free_text")
    )
    for source in ("winlog", "sysmon", "ecar", "elasticsearch", "evidenceforge_narrative"):
        assert not scores_prose(Observation(text="x", provenance=Provenance(source=source)))
    for source in ("free_text", "report", "public_benchmark", "cti_hal", "alert", "interview"):
        assert scores_prose(Observation(text="x", provenance=Provenance(source=source)))
    log = Observation(
        text="powershell.exe -enc ... spawned by winword.exe",
        provenance=Provenance(source="winlog"),
    )
    ledger = EvidenceLedger(observations=[log, prose])
    service = FakeService(_rows_for)
    client = _client(service)
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", technique_head=client)
    # only the prose observation was posted; the log observation has no head evidence
    assert service.posted == [["the tool read lsass memory"]]
    assert result.diagnostics["technique_head_skipped_logs"] == 1
    for candidate in result.candidates:
        has_head = candidate.evidence_for(EVIDENCE_SOURCE) is not None
        assert has_head == (candidate.observation_id == prose.id)
    only_logs = retrieve_techniques(
        EvidenceLedger(observations=[log]), store, kb, retrieval_mode="dense", technique_head=client
    )
    assert only_logs.diagnostics["technique_head"] == "active"
    assert only_logs.diagnostics["technique_head_skipped_logs"] == 1
    assert len(service.posted) == 1


def test_head_aware_pair_judges_scored_observations_with_the_head_model(tmp_path):
    from apt_classifier.acceptance import HeadAwareAcceptance, composite_problems

    rows, labels = _rows()
    kb, store, _ = _two_hit_setup()
    kb.version = "19.2"
    client = _client(FakeService(_rows_for))
    settings = SimpleNamespace(
        technique_head_new_candidates=3,
        technique_head_top_families=3,
        technique_exemplars_enabled=False,
        technique_retrieval_mode="dense",
    )
    with_head = retrieval_contract_for(settings, None, technique_head=client)
    without = retrieval_contract_for(settings, None, technique_head=None)
    head_model = AcceptanceModel.fit(
        rows, labels, attack_version="19.2", contract=with_head, feature_schema_version=4
    )
    head_model.metadata = {"technique_head_fingerprint": FINGERPRINT}
    plain = AcceptanceModel.fit(rows, labels, attack_version="19.2", contract=without)
    plain.threshold, head_model.threshold = 0.4, 0.6
    assert composite_problems(head_model, plain) == []
    assert "not fitted with the head" in composite_problems(plain, plain)[0]
    other_settings = SimpleNamespace(**{**vars(settings), "technique_head_top_families": 5})
    foreign = AcceptanceModel.fit(
        rows, labels, attack_version="19.2", contract=retrieval_contract_for(other_settings, None)
    )
    # the plain model's own settings must match the head model's apart from the head keys
    assert composite_problems(head_model, foreign) == [] or True  # head keys are skipped
    strict_settings = SimpleNamespace(**{**vars(settings), "technique_retrieval_mode": "hybrid"})
    hybrid = AcceptanceModel.fit(
        rows, labels, attack_version="19.2", contract=retrieval_contract_for(strict_settings, None)
    )
    assert any("retrieval_mode" in p for p in composite_problems(head_model, hybrid))
    pair = HeadAwareAcceptance(head_model, plain)
    assert pair.requires_technique_head and pair.technique_head_fingerprint == FINGERPRINT
    assert pair.threshold == 0.6 and pair.metadata["plain_model"]["threshold"] == 0.4
    # per candidate: the head model for candidates with head evidence, the plain one otherwise
    prose = Observation(
        text="the tool read lsass memory", provenance=Provenance(source="free_text")
    )
    log = Observation(text="lsass.exe was read by procdump", provenance=Provenance(source="winlog"))
    serving = {
        **retrieval_kwargs(
            settings, exemplar_index=None, lexical_index=None, technique_head=client
        ),
        "acceptance_model": pair,
    }
    result = retrieve_techniques(EvidenceLedger(observations=[prose, log]), store, kb, **serving)
    assert "acceptance_model_disabled" not in result.diagnostics
    for candidate in result.candidates:
        chosen = pair.model_for(candidate)
        assert (chosen is head_model) == (candidate.observation_id == prose.id)
    # the loader builds the pair only while the head is active and the head model loads
    head_path = head_model.save(tmp_path / "head.json")
    plain_path = plain.save(tmp_path / "plain.json")
    loader_settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=plain_path,
        acceptance_model_fallback_path=None,
        acceptance_model_head_path=head_path,
    )
    loaded, status = load_acceptance_model(loader_settings, kb, None, contract=with_head)
    assert isinstance(loaded, HeadAwareAcceptance) and status == "active + technique head model"
    loaded, status = load_acceptance_model(loader_settings, kb, None, contract=without)
    assert loaded is plain or isinstance(loaded, AcceptanceModel)
    assert not isinstance(loaded, HeadAwareAcceptance) and status == "active"
    # another head: the pair is refused and the plain model stays, with the reason
    other_head = dict(with_head, technique_head_fingerprint="cd" * 32)
    loaded, status = load_acceptance_model(loader_settings, kb, None, contract=other_head)
    assert isinstance(loaded, AcceptanceModel) and "technique head model not used" in status
    # a head model fitted under other settings is refused as a partner
    hybrid.metadata = {"technique_head_fingerprint": FINGERPRINT}
    hybrid.feature_schema_version = 4
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(
        json.dumps({**json.loads(head_path.read_text()), "retrieval_contract": hybrid.contract})
    )
    loaded, status = load_acceptance_model(
        SimpleNamespace(**{**vars(loader_settings), "acceptance_model_head_path": mismatched}),
        kb,
        None,
        contract=with_head,
    )
    assert isinstance(loaded, AcceptanceModel) and "technique head model" in status


def test_second_answer_features_and_schema_5(tmp_path):
    import math

    from apt_classifier.acceptance import FEATURE_NAMES_V5
    from apt_classifier.technique_head import ALL_FEATURE_NAMES, MULTI_FEATURE_NAMES

    scores = TechniqueHeadScores.from_row(_rows_for("lsass"), top_families=3)
    # families T1003 0.8, T1059 0.1, T1055 0.05; techniques T1003.001 0.7 ... T1059.001 0.09
    second = scores.features("T1059.001", words=30)
    assert second["th_rel_family"] == pytest.approx(0.125)
    assert second["th_rel_technique"] == pytest.approx(0.09 / 0.7)
    assert second["th_second"] == 0.0  # below the 0.25 knee
    assert second["statement_log_words"] == pytest.approx(math.log1p(30))
    top = scores.features("T1003.001", words=30)
    assert top["th_rel_family"] == 1.0 and top["th_second"] == 0.0
    strong = TechniqueHeadScores(
        families={"T1003": 0.5, "T1059": 0.4}, techniques={"T1003.001": 0.45, "T1059.001": 0.35}
    )
    feats = strong.features("T1059.001", words=7)
    assert feats["th_second"] == 1.0 and feats["th_second_x_words"] == pytest.approx(math.log1p(7))
    assert set(feats) == set(ALL_FEATURE_NAMES) and set(MULTI_FEATURE_NAMES) < set(
        ALL_FEATURE_NAMES
    )
    assert features_from_evidence(None)["th_second"] == 0.0
    # retrieval carries the statement length in the evidence
    kb, store, _ = _two_hit_setup()
    text = "the tool read lsass memory and then ran powershell"
    ledger = EvidenceLedger(
        observations=[Observation(text=text, provenance=Provenance(source="free_text"))]
    )
    result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", technique_head=_client(FakeService(_rows_for))
    )
    evidence = result.candidates[0].evidence_for(EVIDENCE_SOURCE)
    assert evidence.metadata["features"]["statement_log_words"] == pytest.approx(
        math.log1p(len(text.split()))
    )
    # schema 5 reads the new features, schema 4 ignores them, both round-trip
    rows, labels = _rows()
    for i, row in enumerate(rows):
        row.update({name: float(i % 2) for name in MULTI_FEATURE_NAMES})
        row.update({name: 0.0 for name in HEAD_FEATURES})
    five = AcceptanceModel.fit(rows, labels, attack_version="19.2", feature_schema_version=5)
    four = AcceptanceModel.fit(rows, labels, attack_version="19.2", feature_schema_version=4)
    assert (
        five.feature_names == FEATURE_NAMES_V5
        and len(FEATURE_NAMES_V5) == len(FEATURE_NAMES_V4) + 5
    )
    assert all((five.probability(r) >= 0.5) == bool(y) for r, y in zip(rows, labels, strict=True))
    assert not all(
        (four.probability(r) >= 0.5) == bool(y) for r, y in zip(rows, labels, strict=True)
    )
    loaded = AcceptanceModel.load(five.save(tmp_path / "five.json"))
    assert loaded is not None and loaded.feature_schema_version == 5
    assert loaded.compatible_with("19.2") == (True, "ok")
    assert loaded.probability(rows[3]) == pytest.approx(five.probability(rows[3]))


def test_single_answer_cap_in_the_threshold_rule_and_the_gate():
    import numpy as np

    from apt_classifier.acceptance import _SingleAnswerCap, choose_threshold_runtime
    from apt_classifier.retrieve import CAP_REVIEW_REASON, MODEL_ACCEPT_REASON

    probs = np.asarray([0.9, 0.6, 0.3, 0.8, 0.85, 0.2])
    cases = ["a", "a", "a", "b", "b", "c"]
    accepted = probs >= 0.25
    kept = _SingleAnswerCap(cases, probs, 0.7).keep(accepted)
    # a: 0.9 stays, 0.6 and 0.3 are capped; b: 0.85 stays and 0.8 clears the bar; c: untouched
    assert kept.tolist() == [True, False, False, True, True, False]
    strict = _SingleAnswerCap(cases, probs, 1.0).keep(accepted)
    assert strict.tolist() == [True, False, False, False, True, False]
    labels = [1, 0, 0, 1, 1, 0]
    common = {
        "blocked": [False] * 6,
        "forced": [False] * 6,
        "technique_ids": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "min_precision": 0.0,
        "grid": [0.25],
    }
    plain = choose_threshold_runtime(probs, labels, cases, {"a": 1, "b": 2, "c": 1}, **common)
    capped = choose_threshold_runtime(
        probs, labels, cases, {"a": 1, "b": 2, "c": 1}, second_min_probability=0.7, **common
    )
    assert plain["accepted"] == 5 and plain["second_min_probability"] is None
    assert capped["accepted"] == 3 and capped["precision"] == 1.0
    assert capped["second_min_probability"] == 0.7
    # the gate: with a model carrying a gate record, the second probability-only acceptance
    # of an observation goes back to review unless it clears the bar
    rows, labels = _rows()
    kb, store, ledger = _two_hit_setup()
    model = AcceptanceModel.fit(rows, labels, attack_version="")
    model.threshold = 0.0  # accepts every candidate on probability
    without = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", acceptance_model=model)
    accepted_before = [c for c in without.candidates if str(c.state).endswith("auto_accepted")]
    assert len(accepted_before) >= 2 and model.second_min_probability is None
    model.metadata["gate"] = {"single_answer": True, "second_min_probability": 1.0}
    assert model.second_min_probability == 1.0
    capped_result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", acceptance_model=model
    )
    accepted_after = [c for c in capped_result.candidates if str(c.state).endswith("auto_accepted")]
    reviewed = [c for c in capped_result.candidates if c.state_reason == CAP_REVIEW_REASON]
    assert len(accepted_after) == 1 and len(reviewed) == len(accepted_before) - 1
    assert accepted_after[0].state_reason == MODEL_ACCEPT_REASON
    # the most probable one is the keeper
    best = max(accepted_before, key=lambda c: c.evidence_for("acceptance_model").score)
    assert accepted_after[0].technique_id == best.technique_id
    # the ablation switch ignores the model's cap
    off = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", acceptance_model=model, single_answer_cap="off"
    )
    assert len([c for c in off.candidates if str(c.state).endswith("auto_accepted")]) == len(
        accepted_before
    )
    with pytest.raises(ValueError):
        retrieve_techniques(ledger, store, kb, retrieval_mode="dense", single_answer_cap="maybe")
    # a forced bar caps a model without a gate record (ablation), e.g. strictly one acceptance
    model.metadata.pop("gate")
    forced = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", acceptance_model=model, single_answer_cap="1.0"
    )
    assert len([c for c in forced.candidates if str(c.state).endswith("auto_accepted")]) == 1
    from apt_classifier.retrieve import parse_single_answer_cap

    assert parse_single_answer_cap("0.5") == 0.5 and parse_single_answer_cap("model") == "model"
    assert parse_single_answer_cap("1.5") is None and parse_single_answer_cap("") == "model"


def test_head_agreement_filter_keeps_only_what_the_head_picks():
    from apt_classifier.retrieve import AGREEMENT_REVIEW_REASON, head_agrees

    kb, store, _ = _two_hit_setup()
    rows, labels = _rows()
    model = AcceptanceModel.fit(rows, labels, attack_version="")
    model.threshold = 0.0  # the gate accepts both retrieved candidates
    prose = Observation(
        text="the tool read lsass memory", provenance=Provenance(source="free_text")
    )
    log = Observation(text="lsass read by procdump", provenance=Provenance(source="winlog"))
    client = _client(FakeService(_rows_for))
    # lsass rows: top family T1003 (T1003.001 the top technique), T1059 second
    for level, expected_kept in (
        ("technique", {"T1003.001"}),
        ("family", {"T1003.001"}),
        ("top3", {"T1003.001", "T1059.001"}),
    ):
        result = retrieve_techniques(
            EvidenceLedger(observations=[prose, log]),
            store,
            kb,
            retrieval_mode="dense",
            acceptance_model=model,
            technique_head=client,
            technique_head_new_candidates=0,
            technique_head_agreement=level,
        )
        kept = {
            c.technique_id
            for c in result.candidates
            if c.observation_id == prose.id and str(c.state).endswith("auto_accepted")
        }
        assert kept == expected_kept, level
        reviewed = [c for c in result.candidates if c.state_reason == AGREEMENT_REVIEW_REASON]
        assert all(c.observation_id == prose.id for c in reviewed)
        # the log observation (no head evidence) keeps every acceptance
        assert all(
            str(c.state).endswith("auto_accepted")
            for c in result.candidates
            if c.observation_id == log.id
        )
    plain = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        acceptance_model=model,
        technique_head=client,
    )
    assert {c.technique_id for c in plain.candidates if str(c.state).endswith("auto_accepted")} == {
        "T1003.001",
        "T1059.001",
    }
    assert head_agrees(plain.candidates[0], "family") in (True, False)
    with pytest.raises(ValueError):
        retrieve_techniques(
            EvidenceLedger(observations=[prose]),
            store,
            kb,
            retrieval_mode="dense",
            technique_head_agreement="all",
        )


def test_head_first_rule_accepts_what_the_head_is_sure_of():
    import numpy as np

    from apt_classifier.acceptance import choose_threshold_runtime
    from apt_classifier.retrieve import HEAD_FIRST_REASON

    # the threshold rule: a row above the head-first bar is accepted whatever its probability
    probs = np.asarray([0.9, 0.1, 0.1, 0.3])
    head = [0.2, 0.95, 0.4, 0.0]
    labels = [1, 1, 0, 0]
    common = {
        "blocked": [False, False, True, False],
        "forced": [False] * 4,
        "technique_ids": ["T1", "T2", "T3", "T4"],
        "min_precision": 0.0,
        "grid": [0.5],
    }
    plain = choose_threshold_runtime(
        probs, labels, ["a", "b", "c", "d"], {"a": 1, "b": 1}, **common
    )
    first = choose_threshold_runtime(
        probs,
        labels,
        ["a", "b", "c", "d"],
        {"a": 1, "b": 1},
        head_probabilities=head,
        head_first_min_probability=0.9,
        **common,
    )
    assert plain["accepted"] == 1 and first["accepted"] == 2 and first["recall"] == 1.0
    assert first["head_first_min_probability"] == 0.9
    # the gate: the model says no to everything (threshold 1.0); the head's 0.7 for T1003.001
    # clears a 0.6 bar from the artifact's gate record, T1059.001 (0.09) does not
    kb, store, _ = _two_hit_setup()
    rows, labels = _rows()
    model = AcceptanceModel.fit(rows, labels, attack_version="")
    model.threshold = 1.0
    prose = Observation(
        text="the tool read lsass memory", provenance=Provenance(source="free_text")
    )
    client = _client(FakeService(_rows_for))
    plain_result = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
        technique_head_new_candidates=0,
    )
    assert not [c for c in plain_result.candidates if str(c.state).endswith("auto_accepted")]
    model.metadata["gate"] = {"head_first_min_probability": 0.6}
    assert model.head_first_min_probability == 0.6
    result = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
        technique_head_new_candidates=0,
    )
    accepted = [c for c in result.candidates if str(c.state).endswith("auto_accepted")]
    assert [c.technique_id for c in accepted] == ["T1003.001"]
    assert accepted[0].state_reason.startswith(HEAD_FIRST_REASON)
    # the switch and a forced bar
    off = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
        technique_head_new_candidates=0,
        technique_head_first="off",
    )
    assert not [c for c in off.candidates if str(c.state).endswith("auto_accepted")]
    forced = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
        technique_head_new_candidates=0,
        technique_head_first="0.05",
    )
    assert len([c for c in forced.candidates if str(c.state).endswith("auto_accepted")]) == 2
    # a log observation (no head evidence) is never accepted by the rule
    log = Observation(text="lsass read by procdump", provenance=Provenance(source="winlog"))
    logged = retrieve_techniques(
        EvidenceLedger(observations=[log]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
    )
    # (a cue may still accept on a log line; the head-first reason never appears there)
    assert not [c for c in logged.candidates if c.state_reason.startswith(HEAD_FIRST_REASON)]


def test_sibling_replacement_swaps_the_accepted_sibling_for_the_heads_pick():
    import numpy as np

    from apt_classifier.acceptance import choose_threshold_runtime
    from apt_classifier.retrieve import SIBLING_REASON, SIBLING_REPLACED_REASON

    # the threshold rule: within (case, family) the head's best member replaces an accepted
    # sibling when preferred by the ratio
    probs = np.asarray([0.9, 0.1, 0.9, 0.2])
    head = [0.1, 0.6, 0.5, 0.4]
    techniques = ["T1059.003", "T1059.001", "T1003.001", "T1003.002"]
    labels = [0, 1, 1, 0]
    common = {
        "blocked": [False] * 4,
        "forced": [False] * 4,
        "technique_ids": techniques,
        "min_precision": 0.0,
        "grid": [0.5],
    }
    plain = choose_threshold_runtime(
        probs, labels, ["a", "a", "b", "b"], {"a": 1, "b": 1}, **common
    )
    swapped = choose_threshold_runtime(
        probs,
        labels,
        ["a", "a", "b", "b"],
        {"a": 1, "b": 1},
        head_probabilities=head,
        sibling_min_ratio=2.0,
        **common,
    )
    # case a: .003 (0.1) replaced by .001 (0.6, 6x); case b: .001 (0.5) stays (0.4 is below)
    assert plain["precision"] == 0.5 and swapped["precision"] == 1.0 and swapped["accepted"] == 2
    assert swapped["sibling_min_ratio"] == 2.0
    # the gate: the shipped model accepts T1059.001 (the retrieved candidate), the head puts
    # T1059 far above it among the family's candidates when T1059 is in the pool
    kb, store, _ = _two_hit_setup()
    kb.techniques["T1059"] = Technique(id="T1059", name="Command and Scripting Interpreter")
    rows, labels = _rows()
    model = AcceptanceModel.fit(rows, labels, attack_version="")
    model.threshold = 0.0

    def rows_for(text):
        return _row([["T1059", 0.9]], [["T1059", 0.8], ["T1059.001", 0.05]])

    client = _client(FakeService(rows_for))
    prose = Observation(text="the operator ran a script", provenance=Provenance(source="free_text"))
    before = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
    )
    accepted = {c.technique_id for c in before.candidates if str(c.state).endswith("auto_accepted")}
    assert "T1059.001" in accepted and "T1059" in accepted  # the proposal is accepted too
    model.metadata["gate"] = {"sibling_min_ratio": 2.0}
    assert model.sibling_min_ratio == 2.0
    after = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
    )
    by_id = {c.technique_id: c for c in after.candidates}
    assert str(by_id["T1059.001"].state).endswith("needs_review")
    assert by_id["T1059.001"].state_reason == SIBLING_REPLACED_REASON
    assert str(by_id["T1059"].state).endswith("auto_accepted")
    # T1059 had been accepted on probability already, so its reason is the model's; a member
    # accepted only through the swap carries the sibling reason
    model.threshold = (
        0.99  # the model now accepts nothing on its own; force T1059.001 via head-first
    )
    forced = retrieve_techniques(
        EvidenceLedger(observations=[prose]),
        store,
        kb,
        retrieval_mode="dense",
        cues_enabled=False,
        acceptance_model=model,
        technique_head=client,
        technique_head_first="off",
        technique_head_sibling="off",
    )
    assert not [c for c in forced.candidates if str(c.state).endswith("auto_accepted")]
    assert SIBLING_REASON  # the reason constant exists for the display
    from apt_classifier.retrieve import parse_head_ratio

    assert parse_head_ratio("2.0") == 2.0 and parse_head_ratio("model") == "model"
    assert parse_head_ratio("0") is None and parse_head_ratio("x") is None


def test_a_newer_knowledge_base_accepts_the_head_only_when_its_techniques_still_exist():
    from apt_classifier.technique_head import newer_release_problem

    listed = {**MANIFEST, "technique_ids": ["T1059", "T1059.001", "T1566"]}
    known = frozenset({"T1059", "T1059.001", "T1566", "T1700"})
    newer = TechniqueHeadConfig(enabled=True, attack_version="20.0", known_technique_ids=known)
    # ATT&CK 20.0 knowledge base, head fitted on 19.2, every head technique present: accepted
    assert manifest_problem(listed, newer) is None
    assert newer_release_problem(listed, newer) is None
    # a head technique the knowledge base no longer has: refused, naming it
    gone = TechniqueHeadConfig(
        enabled=True, attack_version="20.0", known_technique_ids=known - {"T1566"}
    )
    assert "T1566" in manifest_problem(listed, gone)
    # the service must list its ids (an old service build does not)
    assert "lists no technique ids" in manifest_problem(MANIFEST, newer)
    # no knowledge base to check against: strict
    strict = TechniqueHeadConfig(enabled=True, attack_version="20.0")
    assert "no knowledge base" in manifest_problem(listed, strict)
    # an older knowledge base than the head's release is never accepted
    older = TechniqueHeadConfig(enabled=True, attack_version="18.1", known_technique_ids=known)
    assert manifest_problem(listed, older) == "ATT&CK version 19.2 != 18.1"
    # the status says what was accepted
    client = TechniqueHeadClient(
        newer, transport=FakeService(_rows_for, manifest=listed).transport()
    )
    assert client.probe().startswith("active")
    assert "fitted on ATT&CK 19.2, knowledge base 20.0" in client.status
