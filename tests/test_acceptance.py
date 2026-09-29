"""Tests for the calibrated technique-acceptance model."""

from __future__ import annotations

import numpy as np
import pytest

from apt_classifier.acceptance import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    PRIOR_FEATURE_NAMES,
    AcceptanceModel,
    candidate_features,
    choose_threshold,
    compute_technique_priors,
    feature_vector,
    prior_features,
)
from apt_classifier.evidence.models import TechniqueCandidate, TechniqueRetrievalEvidence


def _candidate(
    tid: str,
    rank: int,
    fused: float,
    dense: float | None,
    field: str = "description",
    lexical_rank: int | None = None,
    cue: bool = False,
) -> TechniqueCandidate:
    evidence = []
    if dense is not None:
        evidence.append(
            TechniqueRetrievalEvidence(source="dense", field=field, score=dense, rank=rank)
        )
    if lexical_rank is not None:
        evidence.append(
            TechniqueRetrievalEvidence(source="lexical", field="name", score=2.0, rank=lexical_rank)
        )
    if cue:
        evidence.append(TechniqueRetrievalEvidence(source="explicit_cue", score=0.85, rank=1))
    return TechniqueCandidate(
        observation_id="obs-1",
        technique_id=tid,
        technique_name=tid,
        fused_score=fused,
        fused_rank=rank,
        evidence=evidence,
    )


def test_candidate_features_cover_schema_and_margins():
    top = _candidate("T1059.001", 1, 0.040, 0.86, lexical_rank=1, cue=True)
    second = _candidate("T1003.001", 2, 0.017, 0.71, field="procedure")
    third = _candidate("T1021.002", 3, 0.016, None, lexical_rank=7)
    siblings = [second, third, top]
    features = candidate_features(second, siblings, benign_context=False, sparse_context=True)
    # candidate features plus the per-technique priors (looked up by id) cover the schema
    assert set(features) | set(PRIOR_FEATURE_NAMES) == set(FEATURE_NAMES)
    assert features["is_rank_one"] == 0.0
    assert features["fused_margin_to_top"] == pytest.approx(0.023)
    assert features["fused_margin_to_next"] == pytest.approx(0.001)
    assert features["dense_field_procedure"] == 1.0
    assert features["dense_margin"] == pytest.approx(0.15)
    assert features["sparse_context"] == 1.0
    assert features["candidate_count"] == 3.0
    top_features = candidate_features(top, siblings, benign_context=True, sparse_context=False)
    assert top_features["explicit_cue"] == 1.0
    assert top_features["lexical_rank_le3"] == 1.0
    assert top_features["benign_context"] == 1.0
    assert feature_vector(top_features).shape == (len(FEATURE_NAMES),)


def _synthetic_rows(seed: int = 3, n: int = 400):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n):
        dense = float(rng.uniform(0.3, 0.95))
        rank_one = float(rng.integers(0, 2))
        positive = int(dense + 0.2 * rank_one + rng.normal(0, 0.08) > 0.75)
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        row.update({"dense_score": dense, "is_rank_one": rank_one, "dense_present": 1.0})
        rows.append(row)
        labels.append(positive)
    return rows, labels


def test_fit_learns_separable_signal_and_roundtrips(tmp_path):
    rows, labels = _synthetic_rows()
    model = AcceptanceModel.fit(rows, labels, attack_version="19.2")
    weights = dict(zip(FEATURE_NAMES, model.weights.tolist(), strict=True))
    assert weights["dense_score"] > 0.5
    assert weights["is_rank_one"] > 0.0
    probs = [model.probability(row) for row in rows]
    accuracy = np.mean([(p >= 0.5) == bool(y) for p, y in zip(probs, labels, strict=True)])
    assert accuracy > 0.85
    path = model.save(tmp_path / "model.json")
    loaded = AcceptanceModel.load(path)
    assert loaded is not None
    assert loaded.probability(rows[0]) == pytest.approx(model.probability(rows[0]))
    assert loaded.compatible_with("19.2") == (True, "ok")
    ok, reason = loaded.compatible_with("18.0")
    assert not ok and "ATT&CK" in reason
    loaded.feature_schema_version = FEATURE_SCHEMA_VERSION + 1
    assert loaded.compatible_with("19.2")[0] is False
    assert AcceptanceModel.load(tmp_path / "missing.json") is None


def test_choose_threshold_respects_precision_floor_and_unrecoverable_labels():
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 0]
    case_ids = ["a", "a", "b", "b", "c", "c"]
    # case "a" has a third expected label that never became a candidate
    expected_counts = {"a": 3, "b": 1, "c": 0}
    choice = choose_threshold(probabilities, labels, case_ids, expected_counts, min_precision=0.9)
    assert choice["precision_floor_met"] is True
    assert choice["threshold"] > 0.7  # excludes the 0.7 false positive
    assert choice["recall"] == pytest.approx(2 / 4)
    relaxed = choose_threshold(probabilities, labels, case_ids, expected_counts, min_precision=0.5)
    assert relaxed["threshold"] <= 0.6
    assert relaxed["recall"] == pytest.approx(3 / 4)
    impossible = choose_threshold(
        probabilities, labels, case_ids, expected_counts, min_precision=1.1
    )
    assert impossible["precision_floor_met"] is False


def test_from_dict_rejects_foreign_artifacts():
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict({"kind": "something_else"})
    payload = AcceptanceModel.fit(*_synthetic_rows(), attack_version="x").to_dict()
    payload["feature_names"] = payload["feature_names"][:-1]
    with pytest.raises(ValueError):
        AcceptanceModel.from_dict(payload)


def test_technique_priors_from_kb_and_train_counts_roundtrip(tmp_path):
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique

    kb = AttackKnowledgeBase(
        techniques={
            "T1105": Technique(id="T1105", name="Ingress Tool Transfer"),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={
            "G0001": Group(
                id="G0001",
                name="A",
                technique_ids=["T1105", "T1003.001"],
                technique_behaviors={"T1105": ["A downloaded tools.", "A fetched a payload."]},
            ),
            "G0002": Group(id="G0002", name="B", technique_ids=["T1105"]),
        },
    )
    priors = compute_technique_priors(kb, {"T1105": 300, "T1003.001": 3})
    assert set(priors) == {"T1105", "T1003.001"}
    assert priors["T1105"][0] == pytest.approx(np.log1p(300))
    assert priors["T1105"][2] == pytest.approx(np.log1p(2))  # two groups document it
    assert priors["T1003.001"][2] == pytest.approx(np.log1p(1))
    assert priors["T1105"][1] > priors["T1003.001"][1]  # more procedure sentences
    assert prior_features(priors, "T9999") == dict.fromkeys(PRIOR_FEATURE_NAMES, 0.0)

    rows, labels = _synthetic_rows()
    for row in rows:
        row.update(prior_features(priors, "T1105"))
    model = AcceptanceModel.fit(rows, labels, attack_version="19.2", technique_priors=priors)
    loaded = AcceptanceModel.load(model.save(tmp_path / "priors.json"))
    assert loaded is not None
    assert loaded.prior_features("T1105") == prior_features(priors, "T1105")
    assert loaded.prior_features("T9999")["technique_prior_log_freq"] == 0.0
    assert loaded.probability(rows[0]) == pytest.approx(model.probability(rows[0]))
