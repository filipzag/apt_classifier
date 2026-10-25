"""Naming probability floor (prereg attribution_probability_floor_2026-10-25): the gate condition,
its fall-through to the documented-cluster step, and the offline replay and design rule."""

from __future__ import annotations

from apt_classifier.attribution import AttributionStage, assess_attribution
from tests.test_attribution import (
    _candidate,
    _cluster_thresholds,
    _context,
    _separation_kb,
    _two_candidates,
)

EXCLUSIVE = {"T1": 1.0, "T2": 1.0}  # T1 is Alpha's alone: the leader clears the separation gate


def _decide(probability, floor, *, cluster=None, **overrides):
    combiner = None if probability is None else {"leader_probability": probability}
    if combiner is not None and cluster is not None:
        combiner["cluster"] = cluster
    return assess_attribution(
        _separation_kb(),
        _two_candidates(),
        _context(techniques=EXCLUSIVE),
        thresholds=_cluster_thresholds(hypothesis_min_probability=floor, **overrides),
        combiner=combiner,
    )


def test_floor_holds_a_single_name_below_it_and_only_then():
    assert _decide(0.30, 0.0).stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    held = _decide(0.30, 0.40)
    assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert held.leading_group_id == "G1"
    assert "combiner probability 0.30 below the naming floor 0.40" in held.rationale
    assert _decide(0.40, 0.40).stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    named = _decide(0.85, 0.40)
    assert named.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    assert named.calibrated_probability == 0.85
    # without an active combiner there is no probability and no floor
    assert _decide(None, 0.40).stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def test_a_held_leader_can_still_be_named_through_its_documented_cluster():
    cluster = _candidate()  # same-activity twin, combined probability 0.9, clears its bars
    # the leader clears alone and its twin does not nest: the single name stands
    assert _decide(0.30, 0.0, cluster=cluster).stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    # the floor fails the single gate: the cluster step decides as for any other failure
    fallback = _decide(0.30, 0.40, cluster=cluster)
    assert fallback.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    assert fallback.named_cluster == ("G1", "G2")
    # a twin below the cluster's own bar leaves the shortlist
    weak = _decide(0.30, 0.40, cluster=_candidate(probability=0.79))
    assert weak.stage is AttributionStage.KNOWN_GROUP_SHORTLIST


def _row(stage, leading, probability, *, truth="G1", cluster=None):
    return {
        "group_id": truth,
        "stage": stage,
        "leading": leading,
        "rank": 1 if leading == truth else 2,
        "combiner_leader_probability": probability,
        "cluster_candidate": cluster,
        "accepted_techniques": ["T1", "T2"],
        "decisive_software": {"factor": 1.0},
        "leader_guard_fired": False,
        "named_cluster": [],
    }


def _artifact(known, unknown=(), open_set=()):
    thresholds = _cluster_thresholds().__dict__
    return {
        "fingerprint_inputs": {"thresholds": thresholds},
        "known": list(known),
        "unknown": list(unknown),
        "open_set": list(open_set),
    }


def test_replay_matches_the_gate_and_counts_what_the_floor_held():
    from scripts.probability_floor_replay import replay

    named = "known_group_hypothesis"
    artifact = _artifact(
        known=[
            _row(named, "G1", 0.90),
            _row(named, "G2", 0.20),  # a wrong name the floor holds
            _row(named, "G1", 0.30, cluster=_candidate()),  # held, named through its cluster
        ],
        unknown=[_row(named, "G3", 0.45, truth="G9")],  # a removed actor named
        open_set=[_row(named, "G4", 0.95, truth="")],
    )
    off = replay(artifact, 0.0)
    assert off["correct"] == 2 and len(off["wrong"]) == 1 and len(off["removed"]) == 1
    assert len(off["open_set"]) == 1
    on = replay(artifact, 0.50)
    assert on["correct"] == 2  # the clustered row stays correct (the truth is a member)
    assert on["wrong"] == [] and on["removed"] == [] and len(on["open_set"]) == 1
    assert len(on["held_bad"]) == 2 and len(on["held_correct"]) == 1
    assert on["cluster_fallback"] == ["known:G1 as G1 / G2 (correct)"]


def test_design_rule_takes_the_smallest_grid_floor_above_the_catchable_bad_names():
    from scripts.probability_floor_replay import choose

    named = "known_group_hypothesis"
    design = [
        _artifact(
            known=[_row(named, "G1", 0.66), _row(named, "G2", 0.19)],
            unknown=[_row(named, "G3", 0.488, truth="G9"), _row(named, "G4", 0.95, truth="G8")],
        )
    ]
    chosen = choose(design)
    assert chosen["floor"] == 0.5 and chosen["catchable"] == [0.19, 0.488]
    assert chosen["lowest_correct"] == 0.66
    # nothing bad below the lowest correct name: no floor
    clean = [
        _artifact(known=[_row(named, "G1", 0.66)], unknown=[_row(named, "G3", 0.9, truth="G9")])
    ]
    assert choose(clean)["floor"] == 0.0
    # no grid point between them: none chosen
    tight = [_artifact(known=[_row(named, "G1", 0.52), _row(named, "G2", 0.51)])]
    assert choose(tight)["floor"] is None
