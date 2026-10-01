"""Top-three promotion rule (apt_classifier/top3.py)."""

from __future__ import annotations

from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.separation import (
    CandidateDiscrimination,
    DiscriminationResult,
    DiscriminationWeights,
)
from apt_classifier.top3 import (
    apply_promotion,
    decide_promotion,
    leader_advantage,
    order_before_promotion,
)

WEIGHTS = DiscriminationWeights(technique=1.0, behavior=2.0, software=1.0, negative=0.3, soft=1.0)


def _fusion(*ids):
    scores = [1.0 / (20 + i) for i in range(1, len(ids) + 1)]
    entries = tuple(
        FusionEntry(
            group_id=gid,
            score=score,
            relative_score=100.0 * score / scores[0],
            rank=rank,
            classifier_ranks={"c": rank},
            rank_min=rank,
            rank_max=rank,
            rank_std=0.0,
            contested=False,
        )
        for rank, (gid, score) in enumerate(zip(ids, scores, strict=True), 1)
    )
    return FusionResult(entries, ("c",), {}, 20)


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
    fusion = _fusion("LEAD", "RIVAL")
    assert apply_promotion(fusion, promotion) is fusion


def test_leader_advantage_uses_the_engine_weights_and_counts_negative_evidence():
    leader = _candidate("A", software=0.8)
    rival = _candidate("B", technique=3.0, negative=3.0)
    value, channel = leader_advantage(leader, rival, WEIGHTS)
    assert channel == "negative" and abs(value - 0.9) < 1e-9
    heavier = DiscriminationWeights(software=2.0)
    value, channel = leader_advantage(leader, _candidate("B", technique=3.0), heavier)
    assert channel == "software" and abs(value - 1.6) < 1e-9


def test_apply_promotion_moves_the_group_up_and_keeps_scores_monotone():
    fusion = _fusion("A", "B", "C", "D", "E")
    disc = _discrimination(*(_candidate(g) for g in "ABDE"), _candidate("C", technique=3.0))
    promotion = decide_promotion(["A", "B", "C"], disc, margin=1.25, tolerance=0.75)
    moved = apply_promotion(fusion, promotion)
    assert [e.group_id for e in moved.entries] == ["C", "A", "B", "D", "E"]
    assert [e.rank for e in moved.entries] == [1, 2, 3, 4, 5]
    # each place keeps its score, so scores and relative scores stay monotone
    assert [e.score for e in moved.entries] == [e.score for e in fusion.entries]
    assert [e.relative_score for e in moved.entries] == [e.relative_score for e in fusion.entries]
    # the entry's own evidence (classifier ranks) travels with it
    assert moved.entries[0].classifier_ranks == {"c": 3}
    assert {e.group_id for e in moved.entries[:3]} == {e.group_id for e in fusion.entries[:3]}
    assert apply_promotion(fusion, None) is fusion


def test_order_before_promotion_restores_the_order_the_rule_saw():
    record = {"promoted": "C", "from_rank": 3, "blocked": False}
    assert order_before_promotion(["C", "A", "B", "D"], record) == ["A", "B", "C", "D"]
    record = {"promoted": "B", "from_rank": 2, "blocked": False}
    assert order_before_promotion(["B", "A", "C"], record) == ["A", "B", "C"]
    # blocked, absent, or a different leader on display (e.g. the LLM-vote ranking): unchanged
    assert order_before_promotion(["A", "B"], {"promoted": "B", "blocked": True}) == ["A", "B"]
    assert order_before_promotion(["A", "B"], None) == ["A", "B"]
    assert order_before_promotion(["X", "A"], {"promoted": "C", "from_rank": 3}) == ["X", "A"]


def test_promotion_record_round_trips_to_a_plain_dict():
    disc = _discrimination(_candidate("A"), _candidate("B", technique=2.0))
    promotion = decide_promotion(["A", "B"], disc, margin=1.25, tolerance=0.75)
    record = promotion.to_dict()
    assert record["promoted"] == "B" and record["from_rank"] == 2 and record["blocked"] is False
    assert set(record) >= {"gap", "leader_advantage", "advantage_channel", "margin", "tolerance"}
