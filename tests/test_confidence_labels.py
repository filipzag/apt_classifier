"""Calibrated confidence labels (apt_classifier/confidence_labels.py): display only."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from apt_classifier.attribution import AttributionStage, MaturityThresholds
from apt_classifier.confidence_labels import (
    FEATURE_NAMES,
    NOTE,
    SCHEMA_VERSION,
    ConfidenceCalibrator,
    ConfidenceInputs,
    combiner_sha256,
    confidence_label,
    decide,
    features,
    features_from_row,
    load_confidence_labels,
)
from apt_classifier.config import Settings
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.separation import discriminate
from apt_classifier.top3_view import build_top3_view, top3_summary
from tests.test_top3_view import _kb
from tests.test_top3_view import _state as _view_state

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb" / "attack_kb.json"
DESIGN = [ROOT / "data" / "audits" / "top3" / f"design_seed{seed}.json" for seed in (41, 53)]
ARTIFACT = ROOT / "data" / "models" / "calibration" / "confidence_labels.json"

# the frozen coefficients of the spec (known+removed, seeds 41 and 53)
SPEC_MODELS = {
    "leader": (0.958838, (0.960787, 2.788935, 0.189776, -0.049452, -0.255389)),
    "top2": (1.201793, (1.031001, 2.383805, 0.307257, -0.144812, -0.465618)),
    "top3": (0.657502, (0.918505, 1.968876, 0.209900, -0.071492, -0.228189)),
}
EVALUATION = {
    "design_oof": {
        "description": "ATT&CK test folds",
        "rows": 356,
        "labels": {
            "strong": {"n": 182, "right": 178},
            "set": {"n": 23, "right": 22},
            "open": {"n": 151, "truth_in_top3": 83},
        },
    },
    "cti_hal_anonymized": {
        "description": "CTI-HAL",
        "rows": 55,
        "labels": {
            "strong": {"n": 24, "right": 21},
            "set": {"n": 0, "right": 0},
            "open": {"n": 31, "truth_in_top3": 7},
        },
    },
}


def _calibrator(*, identity: bool = False, **overrides: Any) -> ConfidenceCalibrator:
    """The spec's model, or one whose every probability is its own summed combiner p."""
    if identity:
        intercepts = {model: 0.0 for model in SPEC_MODELS}
        weights = {model: (1.0, 0.0, 0.0, 0.0, 0.0) for model in SPEC_MODELS}
    else:
        intercepts = {model: b0 for model, (b0, _) in SPEC_MODELS.items()}
        weights = {model: w for model, (_, w) in SPEC_MODELS.items()}
    values: dict[str, Any] = {
        "intercepts": intercepts,
        "weights": weights,
        "thresholds": {"strong": 0.88, "set": 0.865},
        "attack_version": "t",
        "combiner_fingerprint": "fingerprint",
        "combiner_sha256": "digest",
        "evaluation": EVALUATION,
    }
    values.update(overrides)
    return ConfidenceCalibrator(**values)


def _fusion(ids: list[str], software_leader: bool = True) -> FusionResult:
    entries = []
    for rank, gid in enumerate(ids, 1):
        ranks = {"bernoulli_naive_bayes": rank}
        if software_leader and rank == 1:
            ranks["software_tools"] = 1
        score = 1.0 / (20 + rank)
        entries.append(FusionEntry(gid, score, 100.0 * score * 21, rank, ranks, 1, 1, 0.0, False))
    return FusionResult(tuple(entries), ("c",), {}, 20)


def _state(
    probabilities: tuple[float, ...] = (0.7, 0.2, 0.05),
    *,
    stage: Any = "known_group_shortlist",
    null: float | None = 0.03,
    separation: float = 0.4,
    accepted: int = 8,
    status: str = "active",
    named_cluster: tuple[str, ...] | None = None,
    ids: tuple[str, ...] = ("G0046", "G0008", "G0016"),
) -> SimpleNamespace:
    fusion = _fusion(list(ids[: len(probabilities)]))
    combiner = {
        "status": status,
        "top_n": 20,
        "probabilities": dict(zip(ids, probabilities, strict=False)),
        "leader_probability": probabilities[0] if probabilities else None,
        "null_probability": null,
        "rrf_ranks": {gid: rank for rank, gid in enumerate(ids, 1)},
    }
    context = SimpleNamespace(
        combiner=combiner, observed_techniques={f"T{i}": 0.9 for i in range(accepted)}
    )
    decision = SimpleNamespace(
        stage=AttributionStage(stage) if isinstance(stage, str) else stage,
        separation=separation,
    )
    if named_cluster is not None:
        decision.named_cluster = named_cluster
    return SimpleNamespace(
        fusion=fusion, deterministic_fusion=fusion, context=context, attribution_decision=decision
    )


# ----- 1. formula


def test_probability_matches_a_hand_computation():
    calibrator = _calibrator()
    inputs = ConfidenceInputs((0.7, 0.2, 0.15), 0.4, 0.03, 8, 1)

    def by_hand(model: str, p: float) -> float:
        b0, w = SPEC_MODELS[model]
        p = min(max(p, 1e-4), 1 - 1e-4)
        x = [math.log(p / (1 - p)), 0.4, math.log(0.03 / 0.97), math.log(8), 1.0]
        return 1.0 / (1.0 + math.exp(-(b0 + sum(a * b for a, b in zip(w, x, strict=True)))))

    predicted = calibrator.predict(inputs)
    assert predicted["leader"] == pytest.approx(by_hand("leader", 0.7), abs=1e-9)
    assert predicted["top2"] == pytest.approx(by_hand("top2", 0.9), abs=1e-9)
    # the top-three sum (1.05) is capped at 1, then clipped to 1 - 1e-4 before the logit
    assert predicted["top3"] == pytest.approx(by_hand("top3", 1.0), abs=1e-9)
    # no accepted technique counts as one (ln 1 = 0)
    empty = ConfidenceInputs((0.7, 0.2, 0.15), 0.4, 0.03, 0, 1)
    assert empty.vector("leader")[3] == 0.0


# ----- 2. label boundaries


def test_label_boundaries_are_inclusive():
    identity = _calibrator(identity=True)
    inputs = ConfidenceInputs((0.6, 0.25, 0.1), 0.0, 0.05, 5, 0)
    at = identity.predict(inputs)
    ids = ["G0046", "G0008", "G0016"]

    def label(strong: float, shown: float) -> dict[str, Any]:
        model = _calibrator(identity=True, thresholds={"strong": strong, "set": shown})
        return decide(model, inputs, stage="known_group_shortlist", group_ids=ids, probabilities={})

    assert label(at["leader"], 0.99)["label"] == "strong"
    assert label(at["leader"], 0.99)["group_ids"] == ["G0046"]
    above = math.nextafter(at["leader"], 1.0)
    assert label(above, 0.99)["label"] == "open"
    # set size 2 exactly at the top-two value, 3 just above it (top three is larger)
    two = label(0.99, at["top2"])
    assert (two["label"], two["set_size"], two["group_ids"]) == ("set", 2, ids[:2])
    assert two["probability"] == round(at["top2"], 4)
    three = label(0.99, math.nextafter(at["top2"], 1.0))
    assert (three["label"], three["set_size"], three["group_ids"]) == ("set", 3, ids)
    assert label(0.99, math.nextafter(at["top3"], 1.0))["label"] == "open"
    # a set needs that many candidates
    short = decide(
        _calibrator(identity=True, thresholds={"strong": 0.99, "set": 0.5}),
        inputs,
        stage="known_group_shortlist",
        group_ids=ids[:1],
        probabilities={},
    )
    assert short["label"] == "open"


def test_insufficient_evidence_is_open_even_when_confident():
    calibrator = _calibrator()
    state = _state((0.99, 0.005, 0.001), stage="insufficient_evidence", separation=1.0)
    assert calibrator.predict(features(state))["leader"] > 0.88
    block = confidence_label(calibrator, state)
    assert (block["label"], block["text_key"], block["group_ids"]) == ("open", "open", [])
    assert block["p_leader"] > 0.88  # still reported, never shown as an answer


def test_strong_and_named_wording_keys():
    calibrator = _calibrator()
    strong = confidence_label(calibrator, _state((0.95, 0.03, 0.01), separation=0.9), kb=_kb())
    assert strong["label"] == "strong" and strong["text_key"] == "strong"
    assert strong["members"] == [{"group_id": "G0046", "name": "FIN7"}]
    assert strong["note"] == NOTE and "real reports run overconfident" in NOTE
    assert strong["measured"]["folds"]["strong"] == {"n": 182, "right": 178}
    assert strong["measured"]["real_reports"]["open"] == {"n": 31, "truth_in_top3": 7}
    assert strong["thresholds"] == {"strong": 0.88, "set": 0.865}
    named_low = confidence_label(
        calibrator, _state((0.3, 0.2, 0.1), stage="known_group_hypothesis", separation=0.0)
    )
    assert named_low["label"] == "open" and named_low["text_key"] == "named_lower"


# ----- 3. no label


def test_no_label_without_an_active_combiner_or_null_probability():
    calibrator = _calibrator()
    assert confidence_label(None, _state()) is None
    assert confidence_label(calibrator, _state(status="incompatible artifact: x")) is None
    assert confidence_label(calibrator, _state(null=None)) is None
    state = _state()
    state.context.combiner = None
    assert confidence_label(calibrator, state) is None
    assert features(state) is None


# ----- 4. fail-closed loader


def _artifact(tmp_path: Path, **changes: Any) -> tuple[Path, dict[str, Any]]:
    combiner_file = tmp_path / "combiner.json"
    combiner_file.write_text('{"kind": "x"}', encoding="utf-8")
    payload = _calibrator(combiner_sha256=combiner_sha256(combiner_file)).to_dict()
    for key, value in changes.items():
        payload[key] = value
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _load(path: Path | None, *, enabled: bool = True, version: str = "t", combiner: Any = "ok"):
    settings = SimpleNamespace(confidence_labels_enabled=enabled, confidence_labels_path=path)
    model = SimpleNamespace(pipeline_fingerprint="fingerprint") if combiner == "ok" else combiner
    combiner_path = path.parent / "combiner.json" if path is not None else None
    return load_confidence_labels(settings, SimpleNamespace(version=version), model, combiner_path)


def test_loader_is_fail_closed(tmp_path: Path):
    path, payload = _artifact(tmp_path)
    model, status = _load(path)
    assert status == "active" and model is not None
    assert model.thresholds == {"strong": 0.88, "set": 0.865}

    assert _load(path, enabled=False) == (None, "disabled")
    assert _load(None) == (None, "no path configured")
    missing = tmp_path / "missing.json"
    assert _load(missing) == (None, f"artifact not found: {missing}")
    path.write_text("{not json", encoding="utf-8")
    assert _load(path)[1].startswith("invalid artifact:")

    bad_weights = json.loads(json.dumps(payload["models"]))
    bad_weights["top2"]["weights"] = [1.0, 2.0]
    infinite = json.loads(json.dumps(payload["models"]))
    infinite["leader"]["intercept"] = float("inf")
    invalid = {
        "kind": ({"kind": "something_else"}, "not a confidence-label artifact"),
        "schema": ({"schema_version": SCHEMA_VERSION + 1}, "schema version"),
        "features": ({"feature_names": list(reversed(FEATURE_NAMES))}, "feature names"),
        "shape": ({"models": bad_weights}, "wrong shape"),
        "finite": ({"models": infinite}, "not finite"),
        "thresholds": ({"thresholds": {"strong": 1.0, "set": 0.865}}, "thresholds"),
    }
    for name, (change, reason) in invalid.items():
        path, _ = _artifact(tmp_path, **change)
        model, status = _load(path)
        assert model is None and status.startswith("invalid artifact:"), name
        assert reason in status, name

    path, _ = _artifact(tmp_path)
    assert _load(path, version="19.3") == (None, "incompatible artifact: ATT&CK t != 19.3")
    model, status = _load(path, combiner=None)
    assert model is None and "no active learned combiner" in status
    model, status = _load(path, combiner=SimpleNamespace(pipeline_fingerprint="other"))
    assert model is None and status.startswith("incompatible artifact: combiner pipeline")
    assert "refit the calibration" in status
    (tmp_path / "combiner.json").write_text('{"kind": "changed"}', encoding="utf-8")
    model, status = _load(path)
    assert model is None and status == (
        "incompatible artifact: combiner artifact changed (refit the calibration)"
    )


# ----- 5. live state == benchmark row


def test_features_of_a_state_equal_the_features_of_its_benchmark_row():
    from scripts.attribution_benchmark import _combiner_fields, _separation_fields, _top3_fields

    kb = _kb()
    ids = ["G0046", "G0008", "G0016"]
    fusion = _fusion(ids)
    context = SimpleNamespace(
        kb=kb,
        observed_techniques={"T1": 1.0, "T3": 0.9, "T4": 0.8},
        negative_techniques=set(),
        software_ids={"S0030"},
        specific_behavior_matches=(),
        parent_policy="ancestor_compatible",
        decisive_software={},
    )
    context.discrimination = discriminate(kb, fusion, context, top_k=3)
    context.combiner = {
        "status": "active",
        "top_n": 20,
        "probabilities": {"G0046": 0.6123, "G0008": 0.2511, "G0016": 0.0832},
        "leader_probability": 0.6123,
        "null_probability": 0.0291,
        "rrf_ranks": {"G0046": 2, "G0008": 1, "G0016": 3},
    }
    decision = SimpleNamespace(
        stage=AttributionStage.KNOWN_GROUP_SHORTLIST,
        separation=round(float(context.discrimination.separation), 3),
        separation_factors={"leader_specificity": 0.5},
        runner_up_group_id=context.discrimination.runner_up_id,
        runner_up_related=False,
        maturity_factors={},
    )
    state = SimpleNamespace(
        fusion=fusion, deterministic_fusion=fusion, context=context, attribution_decision=decision
    )
    # exactly what scripts/attribution_benchmark.py writes per row
    row = {
        "accepted_techniques": sorted(context.observed_techniques),
        **_combiner_fields(state, "G0046"),
        **_separation_fields(state, "G0046", decision),
        **_top3_fields(state, ids),
    }
    row = json.loads(json.dumps(row))  # through JSON, as a replay reads it
    live = features(state)
    assert live is not None and live == features_from_row(row)
    assert live.leader_software == 1 and live.n_accepted == 3
    assert live.probabilities == (0.6123, 0.2511, 0.0832)
    # a row without an active combiner has no label inputs
    assert features_from_row({**row, "combiner_status": None}) is None


# ----- 6. display only


ENGINE_MODULES = (
    "session.py",
    "attribution.py",
    "fusion.py",
    "combiner.py",
    "separation.py",
    "report.py",
    "navigator.py",
    "attack_flow.py",
)


def test_engine_modules_never_import_the_labels():
    package = ROOT / "apt_classifier"
    for name in ENGINE_MODULES:
        assert "confidence_labels" not in (package / name).read_text(encoding="utf-8"), name
    probe = (
        "import sys, apt_classifier.session, apt_classifier.attribution, apt_classifier.fusion, "
        "apt_classifier.report; print('apt_classifier.confidence_labels' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "OPENROUTER_API_KEY": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_label_never_changes_the_state_it_reads():
    state = _state((0.95, 0.03, 0.01), separation=0.9)
    before = (
        [e.group_id for e in state.fusion.entries],
        json.dumps(state.context.combiner, sort_keys=True),
        vars(state.attribution_decision).copy(),
    )
    confidence_label(_calibrator(), state)
    after = (
        [e.group_id for e in state.fusion.entries],
        json.dumps(state.context.combiner, sort_keys=True),
        vars(state.attribution_decision).copy(),
    )
    assert before == after


# ----- cluster rule (works before the cluster stage exists: fake decision object)


def test_named_cluster_is_the_pair_and_never_strong():
    calibrator = _calibrator()
    cluster_stage = SimpleNamespace(value="known_cluster_hypothesis")
    # confident enough for "strong" under the ordinary rule
    probabilities = (0.93, 0.02, 0.03)
    ordinary = confidence_label(calibrator, _state(probabilities, separation=0.9))
    assert ordinary["label"] == "strong"
    state = _state(
        probabilities, stage=cluster_stage, separation=0.9, named_cluster=("G0046", "G0016")
    )
    block = confidence_label(calibrator, state, kb=_kb())
    assert (block["label"], block["text_key"], block["set_size"]) == ("set", "named_cluster", 2)
    assert block["group_ids"] == ["G0046", "G0016"]  # the cluster, not the top two
    assert [m["name"] for m in block["members"]] == ["FIN7", "APT29"]
    inputs = features(state)
    expected = calibrator.probability(
        "top2", inputs.vector("top2", p=0.93 + 0.03, clip=calibrator.logit_clip)
    )
    assert block["probability"] == round(expected, 4)
    assert block["probability"] != block["p_top2"]  # the pair's sum, not P1 + P2
    # a cluster stage without a two-member cluster never says "strong" either
    other = confidence_label(
        calibrator, _state(probabilities, stage=cluster_stage, separation=0.9, named_cluster=())
    )
    assert other["label"] != "strong"


# ----- 7. summary keys


def test_top3_summary_carries_the_confidence_block():
    kb = _kb()
    view = build_top3_view(kb, _view_state(kb, ["G0046", "G0008", "G0016"]))
    assert "confidence" not in view  # the service adds it; the view builder never reads it
    view["confidence"] = {"label": "open"}
    summary = top3_summary(view)
    assert set(summary) == {"state", "label", "summary", "leader", "other", "confidence"}
    assert summary["confidence"] == {"label": "open"}


# ----- 8. service and API


def _service(tmp_path: Path, *, enabled: bool = True):
    from apt_classifier.service import CaseMeta, CaseService

    kb = _kb()
    tmp_path.mkdir(parents=True, exist_ok=True)
    combiner_file = tmp_path / "combiner.json"
    combiner_file.write_text('{"kind": "x"}', encoding="utf-8")
    artifact = tmp_path / "labels.json"
    _calibrator(combiner_sha256=combiner_sha256(combiner_file)).save(artifact)
    settings = Settings(
        data_dir=tmp_path,
        model_cache_dir=tmp_path / "hf",
        attribution_combiner_path=combiner_file,
        confidence_labels_enabled=enabled,
        confidence_labels_path=artifact,
    )
    engine = SimpleNamespace(
        settings=settings,
        kb=kb,
        combiner=SimpleNamespace(pipeline_fingerprint="fingerprint"),
        attribution_thresholds=MaturityThresholds(),
    )
    service = CaseService(settings, engine=engine, cases_dir=tmp_path / "cases")  # type: ignore[arg-type]
    state = _view_state(kb, ["G0046", "G0008", "G0016"])
    state.context.combiner.update(
        {"status": "active", "leader_probability": 0.6, "null_probability": 0.05}
    )
    service._states["c1"] = state  # noqa: SLF001
    service._meta["c1"] = CaseMeta(case_id="c1", name="case")  # noqa: SLF001
    return service, state


def test_service_and_api_carry_the_block(tmp_path: Path):
    service, state = _service(tmp_path)
    assert service.confidence_labels_status == "active"
    view = service.top3_view(state)
    assert view["confidence"]["note"] == NOTE
    assert "calibrated on att&ck test folds" in view["confidence"]["note"].casefold()
    assert service.top3_summary(state)["confidence"] == view["confidence"]
    client = TestClient(
        create_app(service, serve_web=False, settings=Settings(auth_mode="off")),
        raise_server_exceptions=True,
    )
    top3 = client.get("/api/cases/c1/top3").json()
    assert top3["confidence"]["label"] in {"strong", "set", "open"}
    assert top3["confidence"]["note"].startswith("Calibrated on ATT&CK test folds")

    off, off_state = _service(tmp_path / "off", enabled=False)
    assert off.confidence_labels_status == "disabled"
    off_view = off.top3_view(off_state)
    assert off_view["confidence"] is None
    # the label is the only difference
    assert {k: v for k, v in view.items() if k != "confidence"} == {
        k: v for k, v in off_view.items() if k != "confidence"
    }


pytestmark_local = pytest.mark.skipif(
    not KB.exists() or os.environ.get("APT_SKIP_APP_FLOW") == "1",
    reason="requires the ingested knowledge base and Chroma index",
)


@pytestmark_local
def test_labels_on_or_off_never_change_decisions_rankings_or_exports(
    tmp_path_factory: pytest.TempPathFactory,
):
    from api import serializers
    from apt_classifier.config import get_settings
    from apt_classifier.report import EXPORT_KINDS
    from apt_classifier.service import CaseService, IntakePayload

    previous = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = ""  # blank overrides .env: local mode
    get_settings.cache_clear()
    cases = tmp_path_factory.mktemp("cases")
    try:
        off = CaseService(cases_dir=cases)
    finally:
        if previous is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = previous
        get_settings.cache_clear()
    engine = off.engine
    if engine.combiner is None:
        pytest.skip(f"learned combiner not active: {engine.combiner_status}")
    off = CaseService(  # the off arm is explicit: the shipped default is on
        off.settings.model_copy(update={"confidence_labels_enabled": False}),
        engine=engine,
        cases_dir=cases,
    )
    artifact = tmp_path_factory.mktemp("labels") / "labels.json"
    _calibrator(
        attack_version=str(engine.kb.version),
        combiner_fingerprint=engine.combiner.pipeline_fingerprint,
        combiner_sha256=combiner_sha256(engine.settings.attribution_combiner_path),
    ).save(artifact)
    on = CaseService(
        off.settings.model_copy(
            update={"confidence_labels_enabled": True, "confidence_labels_path": artifact}
        ),
        engine=engine,
        cases_dir=cases,
    )
    assert off.confidence_labels_status == "disabled" and on.confidence_labels_status == "active"
    case_id = off.create_case(IntakePayload(narrative=off.sample_text(), iocs="Mimikatz"))
    off._states.pop(case_id)  # noqa: SLF001 - both services rebuild the case from disk
    off._meta.pop(case_id)  # noqa: SLF001

    def payload(service: CaseService) -> dict[str, Any]:
        state, meta = service.get(case_id)
        return serializers.case_payload(
            engine.kb,
            state,
            meta,
            engine.attribution_thresholds,
            service.summary_for(state),
            top3=service.top3_summary(state),
        )

    with_labels, without = payload(on), payload(off)
    assert without["top3"]["confidence"] is None
    assert with_labels["top3"]["confidence"]["note"] == NOTE
    with_labels["top3"].pop("confidence")
    without["top3"].pop("confidence")
    assert json.dumps(with_labels, sort_keys=True, default=str) == json.dumps(
        without, sort_keys=True, default=str
    )
    for kind in EXPORT_KINDS:
        assert on.export(case_id, kind) == off.export(case_id, kind), kind
    view_on, view_off = on.top3(case_id), off.top3(case_id)
    assert view_on.pop("confidence") is not None and view_off.pop("confidence") is None
    assert view_on == view_off

    client = TestClient(create_app(on, serve_web=False, settings=Settings(auth_mode="off")))
    assert client.get("/api/system").json()["confidence_labels"] == "active"
    assert client.get(f"/api/cases/{case_id}/top3").json()["confidence"]["note"] == NOTE
    assert client.get(f"/api/cases/{case_id}").json()["top3"]["confidence"]["note"] == NOTE
    off.delete(case_id)


# ----- 9. refit reproduces the coefficients


@pytest.mark.skipif(not all(path.is_file() for path in DESIGN), reason="design artifacts absent")
def test_refit_reproduces_the_frozen_coefficients():
    pytest.importorskip("sklearn")
    from scripts.fit_confidence_labels import benchmark_rows, fit_models, load_benchmark

    rows = [row for path in DESIGN for row in benchmark_rows(load_benchmark(path))]
    assert len(rows) == 356
    intercepts, weights = fit_models(rows)
    for model, (b0, w) in SPEC_MODELS.items():
        assert intercepts[model] == pytest.approx(b0, abs=1e-6), model
        assert weights[model] == pytest.approx(w, abs=1e-6), model
    if ARTIFACT.is_file():
        shipped = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        for model in SPEC_MODELS:
            assert shipped["models"][model]["intercept"] == pytest.approx(
                intercepts[model], abs=1e-9
            )
            assert shipped["models"][model]["weights"] == pytest.approx(weights[model], abs=1e-9)


def _benchmark_file(tmp_path: Path, name: str, **fields: Any) -> Path:
    payload = {
        "benchmark": "attribution_text_heldout",
        "artifact_hash": name,
        "attack_version": "t",
        "parameters": {"seed": 41, "actor_names": "anonymized"},
        "fingerprint_inputs": {},
        "known": [],
        "unknown": [],
        **fields,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fit_script_refuses_cti_hal_and_evaluation_seeds_as_design(tmp_path: Path):
    from scripts.fit_confidence_labels import main

    out = ["--artifact", str(tmp_path / "a.json"), "--report", str(tmp_path / "r.json")]
    cti = _benchmark_file(tmp_path, "cti", benchmark="attribution_cti_hal_reports")
    with pytest.raises(SystemExit, match="CTI-HAL"):
        main(["--design", str(cti), *out])
    seven = _benchmark_file(tmp_path, "seven", parameters={"seed": 7})
    with pytest.raises(SystemExit, match="evaluation seed"):
        main(["--design", str(seven), *out])
    design = _benchmark_file(tmp_path, "design")
    with pytest.raises(SystemExit, match="also passed to --evaluate"):
        main(["--design", str(design), "--evaluate", str(design), *out])
    assert not (tmp_path / "a.json").exists()


def test_evaluate_only_scores_with_the_frozen_artifact(tmp_path: Path):
    from scripts.fit_confidence_labels import main

    row = {
        "group_id": "G0046",
        "stage": "known_group_shortlist",
        "combiner_status": "active",
        "combiner_leader_probability": 0.95,
        "combiner_null_probability": 0.01,
        "separation": 0.9,
        "accepted_confidences": {"T1": 0.9, "T2": 0.8},
        "top3_detail": [
            {"group_id": "G0046", "combiner_probability": 0.95, "classifier_ranks": {}},
            {"group_id": "G0008", "combiner_probability": 0.02, "classifier_ranks": {}},
            {"group_id": "G0016", "combiner_probability": 0.01, "classifier_ranks": {}},
        ],
    }
    fresh = _benchmark_file(
        tmp_path,
        "fresh",
        parameters={"seed": 59, "actor_names": "anonymized"},
        known=[row, {**row, "group_id": "G0016", "combiner_leader_probability": 0.2}],
        unknown=[{**row, "group_id": "G9999"}],
    )
    artifact = tmp_path / "frozen.json"
    _calibrator(provenance={"design_artifacts": [{"artifact_hash": "design"}]}).save(artifact)
    before = artifact.read_bytes()
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "--evaluate-only",
                "--artifact",
                str(artifact),
                "--evaluate",
                str(fresh),
                "--acceptance-bar",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert artifact.read_bytes() == before  # never refitted or rewritten
    result = json.loads(report.read_text(encoding="utf-8"))
    seed = result["evaluation"]["seed59"]
    assert seed["rows"] == 3 and seed["labels"]["strong"]["n"] == 2
    assert seed["labels"]["strong"]["right"] == 1  # the removed actor's leader is wrong
    assert result["evaluation"]["seed59_known_only"]["rows"] == 2
    assert result["acceptance_bar"]["A1_strong_error"]["passed"] is False
    design = _benchmark_file(tmp_path, "design")
    with pytest.raises(SystemExit, match="design artifact"):
        main(["--evaluate-only", "--artifact", str(artifact), "--evaluate", str(design)])
