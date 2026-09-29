"""Candidate-relative discriminative separation (apt_classifier/separation.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.separation import (
    DiscriminationWeights,
    apply_discriminative_rerank,
    discriminate,
    technique_document_counts,
)


def _kb():
    groups = {gid: SimpleNamespace(id=gid, name=gid, prominence=0) for gid in ("G1", "G2", "G3")}
    return SimpleNamespace(
        version="t",
        groups=groups,
        techniques={
            "T1": SimpleNamespace(parent_id=None),
            "T2": SimpleNamespace(parent_id=None),
            "T3": SimpleNamespace(parent_id=None),
            "T3.001": SimpleNamespace(parent_id="T3"),
            **{f"T{i}": SimpleNamespace(parent_id=None) for i in range(10, 40)},
        },
        group_techniques={
            "G1": ["T1", "T2", "T3"],  # T1 exclusive to G1, T2 shared by all, T3 parent
            "G2": ["T2", "T3.001"],
            "G3": ["T2"] + [f"T{i}" for i in range(10, 40)],
        },
        group_software={"G1": ["S_mal", "S_tool"], "G2": ["S_tool"], "G3": ["S_tool"]},
        software_groups={"S_mal": ["G1"], "S_tool": ["G1", "G2", "G3"]},
        software={
            "S_mal": SimpleNamespace(software_type="malware"),
            "S_tool": SimpleNamespace(software_type="tool"),
        },
    )


def _fusion(*ids, scores=None):
    scores = scores or [1.0 / (20 + i) for i in range(1, len(ids) + 1)]
    entries = tuple(
        FusionEntry(
            group_id=gid,
            score=score,
            relative_score=100.0 * score / scores[0],
            rank=rank,
            classifier_ranks={},
            rank_min=1,
            rank_max=1,
            rank_std=0.0,
            contested=False,
        )
        for rank, (gid, score) in enumerate(zip(ids, scores, strict=True), 1)
    )
    return FusionResult(entries, ("x",), {}, 20)


def _context(**overrides):
    base = dict(
        observed_techniques={},
        negative_techniques=set(),
        software_ids=set(),
        specific_behavior_matches=(),
        decisive_software={},
        parent_policy="ancestor_compatible",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_shared_evidence_counts_for_nobody_and_exclusive_evidence_counts_fully():
    kb = _kb()
    result = discriminate(
        kb, _fusion("G1", "G2", "G3"), _context(observed_techniques={"T2": 1.0}), floor=0.0
    )
    assert all(item.technique == 0.0 for item in result.scores)
    assert result.separation == 0.0
    result = discriminate(kb, _fusion("G1", "G2", "G3"), _context(observed_techniques={"T1": 1.0}))
    by = result.by_group()
    # specificity of T1 = 1 - 1/3; exclusivity 1
    assert by["G1"].technique == pytest.approx(1 - 1 / 3)
    assert by["G2"].technique == 0.0 and by["G3"].technique == 0.0
    assert result.leader_id == "G1" and result.runner_up_id in {"G2", "G3"}
    assert result.leader_specificity == pytest.approx(1 - 1 / 3)


def test_hierarchy_factor_and_exclusivity_scale_between_candidates():
    kb = _kb()
    # T3.001 observed: G2 exact (1.0), G1 documents the parent T3 (0.95), G3 nothing
    result = discriminate(
        kb, _fusion("G1", "G2", "G3"), _context(observed_techniques={"T3.001": 1.0})
    )
    by = result.by_group()
    x = 1 - (2 - 1) / (3 - 1)  # two of three candidates document it
    s = 1 - 1 / 3  # T3.001 documented for one group
    assert by["G2"].technique == pytest.approx(1.0 * x * s)
    assert by["G1"].technique == pytest.approx(0.95 * x * s)
    assert by["G3"].technique == 0.0


def test_soft_techniques_are_ignored_and_negatives_subtract_only_where_discriminative():
    kb = _kb()
    context = _context(
        observed_techniques={},
        soft_techniques={"T1": 0.9},  # never reaches the separation score
        negative_techniques={"T1"},
    )
    result = discriminate(kb, _fusion("G1", "G2"), context, floor=0.0)
    by = result.by_group()
    assert by["G1"].technique == 0.0
    assert by["G1"].negative == pytest.approx(1 - 1 / 3)  # exclusive to G1, specificity 2/3
    assert by["G2"].negative == 0.0
    assert by["G1"].total == 0.0  # floored
    context = _context(negative_techniques={"T2"})  # shared by both -> exclusivity 0
    result = discriminate(kb, _fusion("G1", "G2"), context)
    assert all(item.negative == 0.0 for item in result.scores)


def test_behaviour_excess_over_alternative_and_threshold():
    kb = _kb()
    matches = (
        SimpleNamespace(group_id="G1", observation_id="o1", score=0.90),
        SimpleNamespace(group_id="G2", observation_id="o1", score=0.70),
        SimpleNamespace(group_id="G1", observation_id="o2", score=0.60),  # below threshold
        SimpleNamespace(group_id="G9", observation_id="o1", score=0.99),  # not a candidate
    )
    result = discriminate(
        kb,
        _fusion("G1", "G2"),
        _context(specific_behavior_matches=matches),
        behavior_threshold=0.65,
    )
    by = result.by_group()
    assert by["G1"].behavior == pytest.approx(0.90 - 0.70)
    assert by["G2"].behavior == 0.0


def test_software_rarity_exclusivity_and_decisive_factor():
    kb = _kb()
    weights = DiscriminationWeights(technique=1.0, behavior=2.0, software=1.0, negative=0.3)
    result = discriminate(
        kb,
        _fusion("G1", "G2", "G3"),
        _context(software_ids={"S_mal", "S_tool"}),
        weights,
        floor=0.0,
    )
    by = result.by_group()
    # the tool is shared by every candidate -> 0; the one-group malware scores 1.0 for G1
    assert by["G2"].software == 0.0 and by["G3"].software == 0.0
    assert by["G1"].software == pytest.approx(1.0)
    assert result.separation == pytest.approx(1.0)
    boosted = discriminate(
        kb,
        _fusion("G1", "G2", "G3"),
        _context(
            software_ids={"S_mal"}, decisive_software={"factor": 3.0, "software_ids": ["S_mal"]}
        ),
        weights,
    )
    assert boosted.by_group()["G1"].software == pytest.approx(3.0)


def test_separation_floor_and_negative_separation():
    kb = _kb()
    # tiny exclusive evidence with no competitor: the floor keeps it from looking decisive
    result = discriminate(
        kb, _fusion("G1", "G2"), _context(observed_techniques={"T1": 0.3}), floor=1.0
    )
    assert 0 < result.separation < 0.3
    # runner-up better supported than the RRF leader -> negative separation
    result = discriminate(
        kb, _fusion("G2", "G1"), _context(observed_techniques={"T1": 1.0}), floor=0.0
    )
    assert result.leader_id == "G2" and result.runner_up_id == "G1"
    assert result.separation == -1.0
    assert discriminate(kb, _fusion(), _context()).scores == ()


def test_unanimous_leader_with_close_rrf_margin_clears_a_separation_bar():
    kb = _kb()
    fusion = _fusion("G1", "G2", scores=[1 / 21, 1 / 22])  # rank 1 vs 2 everywhere: margin 0.045
    margin = (fusion.entries[0].relative_score - fusion.entries[1].relative_score) / 100
    assert margin < 0.10
    result = discriminate(
        kb,
        fusion,
        _context(observed_techniques={"T1": 1.0}, software_ids={"S_mal"}),
        floor=1.0,
    )
    assert result.separation >= 0.5


def test_rerank_blends_discriminative_mass_and_keeps_top_k_set():
    kb = _kb()
    fusion = _fusion("G2", "G1", "G3", scores=[1 / 21, 1 / 21.5, 1 / 25])
    result = discriminate(kb, fusion, _context(observed_techniques={"T1": 1.0}), top_k=2)
    assert apply_discriminative_rerank(fusion, result, 0.0) is fusion
    reranked = apply_discriminative_rerank(fusion, result, 1.0, floor=0.1)
    order = [entry.group_id for entry in reranked.entries]
    assert order == ["G1", "G2", "G3"]  # G1 has all the discriminative evidence
    assert [entry.rank for entry in reranked.entries] == [1, 2, 3]
    assert reranked.entries[0].relative_score == pytest.approx(100.0)
    assert reranked.entries[2].score == pytest.approx(1 / 25)  # beyond top-K untouched
    assert reranked.k == fusion.k and reranked.active_classifiers == fusion.active_classifiers


def test_document_counts_are_memoized_per_kb():
    kb = _kb()
    first = technique_document_counts(kb)
    assert first["T2"] == 3 and first["T1"] == 1
    assert technique_document_counts(kb) is first
