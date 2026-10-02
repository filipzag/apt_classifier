"""Documented-overlap clusters (apt_classifier/overlap_cluster.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.overlap_cluster import (
    cluster_candidate,
    cluster_label,
    containment,
    select_twin,
    unit_totals,
)
from apt_classifier.separation import DiscriminationWeights, discriminate


def _kb() -> AttackKnowledgeBase:
    kb = AttackKnowledgeBase(
        version="t",
        techniques={tid: Technique(id=tid, name=tid) for tid in ("T1", "T2", "T3", "T4", "T5")},
        groups={
            "G1": Group(id="G1", name="Alpha", technique_ids=["T1", "T2", "T3"]),
            # Beta's description names Alpha, and its profile lies inside Alpha's
            "G2": Group(
                id="G2",
                name="Beta",
                description="Beta shares tooling with Alpha.",
                technique_ids=["T2", "T3"],
            ),
            "G3": Group(id="G3", name="Gamma", technique_ids=["T4", "T2"]),
            "G4": Group(id="G4", name="Delta", technique_ids=["T5"]),
        },
    )
    kb.recompute_statistics()
    return kb


def _fusion(*ids: str) -> FusionResult:
    entries = tuple(
        FusionEntry(gid, 1.0 / (20 + rank), 100.0 - rank, rank, {}, 1, 1, 0.0, False)
        for rank, gid in enumerate(ids, 1)
    )
    return FusionResult(entries, ("c",), {}, 20)


def _context(techniques):
    return SimpleNamespace(
        observed_techniques=techniques,
        negative_techniques=set(),
        software_ids=set(),
        specific_behavior_matches=(),
        parent_policy="ancestor_compatible",
        decisive_software={},
    )


def test_singleton_units_equal_the_gate_discrimination():
    kb = _kb()
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    fusion = _fusion("G1", "G2", "G3", "G4")
    reference = discriminate(kb, fusion, context, top_k=4)
    totals = unit_totals(kb, [(gid,) for gid in ("G1", "G2", "G3", "G4")], context)
    assert totals == pytest.approx([item.total for item in reference.scores])


def test_evidence_the_pair_shares_is_exclusive_to_the_merged_unit():
    kb = _kb()
    context = _context({"T3": 1.0})
    single = unit_totals(kb, [("G1",), ("G2",), ("G4",)], context)
    merged = unit_totals(kb, [("G1", "G2"), ("G4",)], context)
    # T3 is documented by both Alpha and Beta: shared between them, exclusive to the pair
    assert merged[0] > single[0] and merged[0] > single[1]
    assert merged[1] == 0.0


def test_twin_is_a_related_group_that_nests_or_does_not_separate():
    kb = _kb()
    assert containment(kb, "G1", "G2") == 1.0
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    discrimination = discriminate(kb, _fusion("G1", "G2", "G3"), context, top_k=3)
    twin = select_twin(kb, discrimination)
    assert twin is not None and twin["group_id"] == "G2" and twin["nested"]
    assert "references" in twin["reason"]
    # an unrelated leader has no twin
    unrelated = discriminate(kb, _fusion("G3", "G4"), context, top_k=2)
    assert select_twin(kb, unrelated) is None
    # the twin must be in the window
    assert select_twin(kb, discriminate(kb, _fusion("G1", "G3"), context, top_k=2)) is None


def test_cluster_candidate_merges_the_pair_and_sums_the_probabilities():
    kb = _kb()
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    discrimination = discriminate(kb, _fusion("G1", "G2", "G3"), context, top_k=3)
    candidate = cluster_candidate(
        kb, discrimination, context, probabilities={"G1": 0.6, "G2": 0.25, "G3": 0.1}
    )
    assert candidate is not None and candidate.members == ("G1", "G2")
    assert candidate.probability == pytest.approx(0.85)
    assert candidate.runner_up_id == "G3"
    expected = unit_totals(kb, [("G1", "G2"), ("G3",)], context, DiscriminationWeights())
    assert candidate.discriminative == pytest.approx(expected[0])
    assert -1.0 <= candidate.separation <= 1.0
    record = candidate.to_dict()
    assert record["members"] == ["G1", "G2"] and record["nested"] is True
    # without a combiner there is no probability
    assert cluster_candidate(kb, discrimination, context, probabilities=None).probability is None
    assert cluster_label(kb, ("G1", "G2")) == "Alpha / Beta (documented overlap)"
