"""Tests for manifest-gated reranking and leak-resistant calibration."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
from apt_classifier.evaluation import TechniqueEvaluationCase
from apt_classifier.evidence.models import (
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueRetrievalEvidence,
    TechniqueVerification,
    TechniqueVerificationVerdict,
)
from apt_classifier.reranker import (
    MANIFEST_FILENAME,
    RERANKER_FEATURE_SCHEMA,
    LocalReranker,
    RerankerConfig,
    artifact_fingerprint,
)
from scripts.train_technique_reranker import (
    PairExample,
    _pairs_hash,
    build_pairs,
    calibrate_thresholds,
    filter_cross_split_duplicates,
)


def _candidate(technique_id: str, rank: int) -> TechniqueCandidate:
    return TechniqueCandidate(
        observation_id="obs-1",
        technique_id=technique_id,
        technique_name=f"Technique {technique_id}",
        parent_technique_id=None,
        is_subtechnique=False,
        fused_score=1.0 / rank,
        fused_rank=rank,
        evidence=[
            TechniqueRetrievalEvidence(
                source="dense",
                field="description",
                text="test",
                score=0.9,
                rank=1,
            )
        ],
        verification=TechniqueVerification(verdict=TechniqueVerificationVerdict.NOT_RUN),
        state=TechniqueCandidateState.NEEDS_REVIEW,
    )


def _manifest(**updates):
    manifest = {
        "schema_version": 1,
        "artifact_type": "apt_classifier_technique_reranker",
        "attack_version": "15.1",
        "base_model": "cross-encoder/test",
        "base_model_revision": "a" * 40,
        "feature_schema": RERANKER_FEATURE_SCHEMA,
        "training_data_hash": "b" * 64,
        "training": {"seed": 7, "pair_count": 4},
        "calibration": {
            "method": "balanced_f1_dev",
            "promote_threshold": 0.8,
            "demote_threshold": 0.2,
            "dev_pair_count": 4,
            "held_out_metrics": {"balanced_f1": 1.0},
        },
    }
    manifest.update(updates)
    manifest["artifact_fingerprint"] = artifact_fingerprint(manifest)
    return manifest


def _artifact(tmp_path, manifest=None):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if manifest is not None:
        (model_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return model_dir


def _install_fake_cross_encoder(monkeypatch, scores=(0.2, 0.9)):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            assert len(pairs) == len(scores)
            return list(scores)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeModel)
    )


def test_disabled_reranker_returns_original_order():
    reranker = LocalReranker(RerankerConfig(enabled=False))
    assert not reranker.available
    candidates = [_candidate("T1", 1), _candidate("T2", 2)]
    result = reranker.rerank("test", candidates, SimpleNamespace(techniques={}))
    assert [candidate.technique_id for candidate in result] == ["T1", "T2"]


def test_enabled_without_model_is_unavailable_with_clear_error():
    reranker = LocalReranker(RerankerConfig(enabled=True, model_name=""))
    assert not reranker.available
    assert "RERANKER_MODEL is empty" in str(reranker.load_error)


def test_missing_manifest_fails_closed(tmp_path):
    reranker = LocalReranker(RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path))))
    assert not reranker.available
    assert not reranker.validated
    assert MANIFEST_FILENAME in str(reranker.load_error)
    candidate = _candidate("T1", 1)
    assert reranker.acceptance_score(candidate) is None


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"schema_version": 99}, "schema_version mismatch"),
        ({"attack_version": "latest"}, "pinned ATT&CK version"),
        ({"base_model_revision": None}, "must be pinned"),
        ({"training_data_hash": "not-a-hash"}, "training_data_hash"),
        ({"feature_schema": {"version": 99}}, "feature_schema"),
        ({"calibration": {}}, "calibration method"),
    ],
)
def test_invalid_manifest_fails_closed(tmp_path, change, error):
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, _manifest(**change))))
    )
    assert not reranker.available
    assert error in str(reranker.load_error)


def test_tampered_manifest_fingerprint_is_rejected(tmp_path):
    manifest = _manifest()
    manifest["calibration"]["promote_threshold"] = 0.7
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, manifest)))
    )
    assert not reranker.available
    assert "fingerprint does not match" in str(reranker.load_error)


def test_valid_manifest_loads_and_exposes_calibrated_thresholds(tmp_path, monkeypatch):
    _install_fake_cross_encoder(monkeypatch)
    model_dir = _artifact(tmp_path, _manifest())
    reranker = LocalReranker(
        RerankerConfig(
            enabled=True,
            model_name=str(model_dir),
            attack_version="15.1",
            expected_base_model="cross-encoder/test",
            expected_base_model_revision="a" * 40,
        )
    )
    assert reranker.available
    assert reranker.validated
    assert reranker.calibrated_thresholds == (0.8, 0.2)
    assert reranker.promote_threshold == 0.8
    assert reranker.demote_threshold == 0.2


def test_attack_kb_version_mismatch_disables_loaded_artifact(tmp_path, monkeypatch):
    _install_fake_cross_encoder(monkeypatch)
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, _manifest())))
    )
    candidates = [_candidate("T1", 1), _candidate("T2", 2)]
    kb = SimpleNamespace(attack_version="16.0", techniques={})
    assert reranker.rerank("text", candidates, kb) == tuple(candidates)
    assert not reranker.available
    assert "ATT&CK version mismatch" in str(reranker.load_error)
    assert reranker.acceptance_score(candidates[0]) is None


def test_available_reranker_records_scores_and_ranks(tmp_path, monkeypatch):
    _install_fake_cross_encoder(monkeypatch)
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, _manifest())))
    )
    candidates = [_candidate("T1", 1), _candidate("T2", 2)]
    kb = SimpleNamespace(
        attack_version="15.1",
        techniques={
            "T1": SimpleNamespace(description="First technique"),
            "T2": SimpleNamespace(description="Second technique"),
        },
    )
    result = reranker.rerank("observation", candidates, kb)
    # T2 scored 0.9 >= promote 0.8 -> confident, moves to the front; T1 keeps its place.
    assert [candidate.technique_id for candidate in result] == ["T2", "T1"]
    evidence = result[0].evidence_for("reranker")
    assert evidence is not None
    assert evidence.score == 0.9
    assert evidence.rank == 1
    assert reranker.acceptance_score(result[0]) == 1.0


def test_unconfident_scores_do_not_reorder_candidates(tmp_path, monkeypatch):
    # Both scores fall inside the abstention band (0.2, 0.8): fused order is preserved.
    _install_fake_cross_encoder(monkeypatch, scores=(0.3, 0.6))
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, _manifest())))
    )
    candidates = [_candidate("T1", 1), _candidate("T2", 2)]
    kb = SimpleNamespace(attack_version="15.1", techniques={})
    result = reranker.rerank("observation", candidates, kb)
    assert [candidate.technique_id for candidate in result] == ["T1", "T2"]
    assert [candidate.fused_rank for candidate in result] == [1, 2]
    assert result[1].evidence_for("reranker").score == 0.6
    assert result[1].evidence_for("reranker").rank == 1  # score rank is still recorded


def test_training_pairs_include_hierarchy_and_zero_label_negatives():
    kb = AttackKnowledgeBase(
        version="15.1",
        techniques={
            "T1": Technique("T1", "Parent", tactics=["execution"]),
            "T1.001": Technique(
                "T1.001",
                "Child One",
                tactics=["execution"],
                is_subtechnique=True,
                parent_id="T1",
            ),
            "T1.002": Technique(
                "T1.002",
                "Child Two",
                tactics=["execution"],
                is_subtechnique=True,
                parent_id="T1",
            ),
            "T2": Technique("T2", "Other", tactics=["collection"]),
        },
    )
    cases = [
        TechniqueEvaluationCase("positive", "doc-one", "action", ("T1.001",)),
        TechniqueEvaluationCase("negative", "doc-two", "benign text", ()),
    ]
    pairs = build_pairs(cases, kb, negatives_per_positive=2)
    positive = [pair for pair in pairs if pair.case_id == "positive"]
    negative = [pair for pair in pairs if pair.case_id == "negative"]
    assert sum(pair.label == 1.0 for pair in positive) == 1
    assert any(pair.negative_type == "sibling" for pair in positive)
    assert all(pair.label == 0.0 for pair in negative)
    assert all(str(pair.negative_type).startswith("zero_label") for pair in negative)


def test_cross_split_exact_and_near_duplicates_are_removed_from_train():
    train = [
        TechniqueEvaluationCase("t1", "d1", "PowerShell downloaded payload", ("T1",)),
        TechniqueEvaluationCase("t2", "d2", "powershell downloaded the payload", ("T1",)),
        TechniqueEvaluationCase("t3", "d3", "Unrelated archive command", ("T2",)),
    ]
    dev = [
        TechniqueEvaluationCase("d1", "h1", "PowerShell downloaded payload", ("T1",)),
        TechniqueEvaluationCase("d2", "h2", "PowerShell downloaded payload", ("T1",)),
    ]
    filtered_train, filtered_dev, stats = filter_cross_split_duplicates(
        train, dev, near_duplicate_threshold=0.7
    )
    assert [case.case_id for case in filtered_train] == ["t3"]
    assert [case.case_id for case in filtered_dev] == ["d1"]
    assert stats == {
        "train_internal_removed": 1,
        "dev_internal_removed": 1,
        "cross_split_train_removed": 1,
    }


def test_calibration_selects_a_precision_floored_band():
    calibration = calibrate_thresholds([0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0])
    assert calibration["method"] == "precision_floor_band_dev"
    assert calibration["demote_threshold"] < calibration["promote_threshold"]
    # promote: lowest threshold whose promoted set is >= 85% positive -> midpoint 0.5
    assert calibration["promote_threshold"] == pytest.approx(0.5)
    # demote: highest threshold whose demoted set is >= 90% negative -> same midpoint region
    assert 0.2 <= calibration["demote_threshold"] < 0.5
    assert calibration["held_out_metrics"]["promote_precision"] == 1.0
    assert calibration["held_out_metrics"]["demote_precision"] == 1.0
    assert calibration["held_out_metrics"]["balanced_f1"] == 1.0


def test_calibration_never_collapses_to_a_single_threshold():
    import random

    rng = random.Random(3)
    labels = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(3000)]
    scores = [min(1.0, max(0.0, rng.gauss(0.65 if y else 0.35, 0.18))) for y in labels]
    calibration = calibrate_thresholds(scores, labels)
    assert calibration["demote_threshold"] < calibration["promote_threshold"]
    assert calibration["held_out_metrics"]["promote_precision"] >= 0.85
    assert calibration["held_out_metrics"]["demote_precision"] >= 0.90
    assert calibration["held_out_metrics"]["decision_coverage"] < 1.0


def test_calibration_disables_a_side_when_its_floor_is_unreachable():
    # Scores carry no signal: no threshold reaches the promote floor, so promotion is off.
    calibration = calibrate_thresholds([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0])
    assert calibration["promote_threshold"] == 1.0
    assert calibration["held_out_metrics"]["promote_true_positive"] == 0


def test_calibration_requires_both_dev_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        calibrate_thresholds([0.1, 0.2], [0.0, 0.0])


def test_training_data_hash_is_order_independent():
    first = PairExample("a", "one", "T1", "technique", 1.0)
    second = PairExample("b", "two", "T2", "technique", 0.0, "retrieval_mined")
    assert _pairs_hash([first, second]) == _pairs_hash([second, first])


def test_inference_pairs_use_the_shared_training_text(tmp_path, monkeypatch):
    from apt_classifier.reranker import technique_pair_text

    captured: list[list[tuple[str, str]]] = []

    class CapturingModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            captured.append(list(pairs))
            return [0.3, 0.6][: len(pairs)]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CapturingModel)
    )
    reranker = LocalReranker(
        RerankerConfig(enabled=True, model_name=str(_artifact(tmp_path, _manifest())))
    )
    kb = SimpleNamespace(
        attack_version="15.1",
        techniques={
            "T1": SimpleNamespace(id="T1", name="Parent", tactics=["execution"], description="d1"),
            "T2": SimpleNamespace(id="T2", name="Child", tactics=[], description="x" * 5000),
        },
    )
    reranker.rerank("observation text", [_candidate("T1", 1), _candidate("T2", 2)], kb)
    assert captured and captured[0][0] == ("observation text", technique_pair_text(kb, "T1"))
    assert captured[0][0][1] == "T1 Parent. Tactics: execution. d1"
    assert captured[0][1][1] == technique_pair_text(kb, "T2")
    assert len(captured[0][1][1]) <= len("T2 Child. ") + 3000
    assert technique_pair_text(kb, "T9") == "T9"


def test_pairs_cache_round_trips_slots_dataclass(tmp_path):
    from scripts.train_technique_reranker import (
        PairExample,
        _load_pairs_cache,
        _save_pairs_cache,
    )

    pairs = [
        PairExample("c1", "obs", "T1", "T1 text", 1.0),
        PairExample("c1", "obs", "T2", "T2 text", 0.0, "sibling"),
    ]
    dev = [PairExample("d1", "obs2", "T3", "T3 text", 1.0)]
    path = tmp_path / "cache.json"
    assert _load_pairs_cache(path) is None
    _save_pairs_cache(path, pairs, dev)
    loaded = _load_pairs_cache(path)
    assert loaded is not None
    assert loaded[0] == pairs and loaded[1] == dev


def test_precision_floor_band_manifest_loads(tmp_path, monkeypatch):
    _install_fake_cross_encoder(monkeypatch)
    calibration = {
        "method": "precision_floor_band_dev",
        "promote_threshold": 0.9,
        "demote_threshold": 0.1,
        "dev_pair_count": 4,
        "held_out_metrics": {"balanced_f1": 1.0, "promote_precision": 1.0},
    }
    reranker = LocalReranker(
        RerankerConfig(
            enabled=True, model_name=str(_artifact(tmp_path, _manifest(calibration=calibration)))
        )
    )
    assert reranker.available and reranker.validated
    assert reranker.calibrated_thresholds == (0.9, 0.1)
    (tmp_path / "b").mkdir()
    rejected = LocalReranker(
        RerankerConfig(
            enabled=True,
            model_name=str(
                _artifact(tmp_path / "b", _manifest(calibration={**calibration, "method": "x"}))
            ),
        )
    )
    assert not rejected.available
    assert "calibration method" in str(rejected.load_error)
