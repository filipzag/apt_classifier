"""Top-three exclusive-evidence rule (apt_classifier/top3.py; measured, not adopted)."""

from __future__ import annotations

from apt_classifier.separation import (
    CandidateDiscrimination,
    DiscriminationResult,
    DiscriminationWeights,
)
from apt_classifier.top3 import (
    PREREGISTERED_MARGIN,
    PREREGISTERED_TOLERANCE,
    decide_promotion,
    leader_advantage,
)

WEIGHTS = DiscriminationWeights(technique=1.0, behavior=2.0, software=1.0, negative=0.3, soft=1.0)


def _candidate(gid, technique=0.0, behavior=0.0, software=0.0, negative=0.0, soft=0.0):
    total = max(
        0.0,
        WEIGHTS.technique * technique
        + WEIGHTS.behavior * behavior
        + WEIGHTS.software * software
        + WEIGHTS.soft * soft
        - WEIGHTS.negative * negative,
    )
    return CandidateDiscrimination(
        group_id=gid,
        rrf_score=0.0,
        technique=technique,
        behavior=behavior,
        software=software,
        negative=negative,
        total=total,
        soft=soft,
    )


def _discrimination(*candidates):
    return DiscriminationResult(scores=tuple(candidates), top_k=len(candidates), weights=WEIGHTS)


def test_off_by_default_and_inert_without_a_clear_gap():
    disc = _discrimination(
        _candidate("A", technique=1.0), _candidate("B", technique=1.8), _candidate("C")
    )
    assert decide_promotion(["A", "B", "C"], disc, margin=0.0, tolerance=0.75) is None
    # B leads on D but by less than the margin
    assert decide_promotion(["A", "B", "C"], disc, margin=1.25, tolerance=0.75) is None
    # the leader already has the most exclusive evidence
    disc = _discrimination(_candidate("A", technique=4.0), _candidate("B", technique=1.0))
    assert decide_promotion(["A", "B"], disc, margin=1.25, tolerance=0.75) is None


def test_promotes_the_top_three_member_with_the_largest_mass():
    disc = _discrimination(
        _candidate("A", technique=0.3),
        _candidate("B", technique=1.2),
        _candidate("C", technique=3.1),
        _candidate("D", technique=9.0),  # in the gate window but outside the top three
    )
    promotion = decide_promotion(["A", "B", "C", "D"], disc, margin=1.25, tolerance=0.75)
    assert promotion is not None and not promotion.blocked
    assert (promotion.promoted, promotion.previous_leader, promotion.from_rank) == ("C", "A", 3)
    assert abs(promotion.gap - 2.8) < 1e-9


def test_ties_on_mass_keep_the_better_placed_group():
    disc = _discrimination(
        _candidate("A"), _candidate("B", technique=2.0), _candidate("C", technique=2.0)
    )
    promotion = decide_promotion(["A", "B", "C"], disc, margin=1.25, tolerance=0.75)
    assert promotion is not None and promotion.promoted == "B"


def test_guard_blocks_when_one_channel_clearly_favours_the_leader():
    # Kimsuky-over-FIN7-like case: the rival has more technique mass, the leader wins the
    # actor-behaviour channel by a weighted 1.48
    disc = _discrimination(
        _candidate("LEAD", technique=0.3, behavior=0.84),
        _candidate("RIVAL", technique=3.4, behavior=0.10),
    )
    promotion = decide_promotion(["LEAD", "RIVAL"], disc, margin=1.25, tolerance=0.75)
    assert promotion is not None and promotion.blocked
    assert promotion.advantage_channel == "behavior"
    assert abs(promotion.leader_advantage - 1.48) < 1e-9


def test_leader_advantage_uses_the_engine_weights_and_counts_negative_evidence():
    leader = _candidate("A", software=0.8)
    rival = _candidate("B", technique=3.0, negative=3.0)
    value, channel = leader_advantage(leader, rival, WEIGHTS)
    assert channel == "negative" and abs(value - 0.9) < 1e-9
    heavier = DiscriminationWeights(software=2.0)
    value, channel = leader_advantage(leader, _candidate("B", technique=3.0), heavier)
    assert channel == "software" and abs(value - 1.6) < 1e-9


def test_defaults_are_the_preregistered_rule():
    assert (PREREGISTERED_MARGIN, PREREGISTERED_TOLERANCE) == (1.25, 0.75)
    disc = _discrimination(_candidate("A"), _candidate("B", technique=1.3))
    promotion = decide_promotion(["A", "B"], disc)
    assert promotion is not None and promotion.margin == 1.25 and promotion.tolerance == 0.75
    assert (
        decide_promotion(
            ["A", "B"], _discrimination(_candidate("A"), _candidate("B", technique=1.2))
        )
        is None
    )


def test_promotion_record_round_trips_to_a_plain_dict():
    disc = _discrimination(_candidate("A"), _candidate("B", technique=2.0))
    promotion = decide_promotion(["A", "B"], disc, margin=1.25, tolerance=0.75)
    record = promotion.to_dict()
    assert record["promoted"] == "B" and record["from_rank"] == 2 and record["blocked"] is False
    assert set(record) >= {"gap", "leader_advantage", "advantage_channel", "margin", "tolerance"}
