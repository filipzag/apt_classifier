"""Distinguishing the top three: a guarded promotion within the shortlist's first three places.

The learned combiner (combiner.py) orders the shortlist by how the classifiers agree; the
candidate-relative discrimination (separation.py) measures, for the naming gate's window, the
observed evidence each candidate explains that the others do not (its discriminative mass D).
When the two disagree by a wide margin inside the top three, the combiner's leader is usually
wrong (docs 10.10): this module moves the top-three member with the largest D to first place
when

- its D exceeds the leader's by at least ``margin`` (``ATTRIBUTION_TOP3_PROMOTION_MARGIN``,
  0 = off), and
- no single evidence channel clearly favours the leader: the leader's largest weighted
  per-channel advantage over it is at most ``tolerance``
  (``ATTRIBUTION_TOP3_CONTRADICTION_TOLERANCE``).

The guard can only remove promotions. A promoted leader is never named in the same analysis
(``attribution.assess_attribution``), so the naming gate's promise does not depend on the rule.

Everything here is a pure function of recorded values, shared by the engine
(``AttributionEngine._fuse``), the offline replay (``scripts/top3_replay.py``) and the analyst's
Top-three view. The module lies outside the combiner fingerprint (fusion.py, separation.py,
classifiers/), so changing it never invalidates the combiner artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from apt_classifier.fusion import FusionResult

PROMOTION_POOL = 3
# channels that add to the discriminative mass; negative evidence subtracts (see separation.py)
POSITIVE_CHANNELS = ("technique", "behavior", "software", "soft")


@dataclass(frozen=True)
class Promotion:
    """A promotion decision inside the top three (``blocked`` ones are recorded, not applied)."""

    promoted: str
    previous_leader: str
    from_rank: int
    gap: float
    leader_advantage: float
    advantage_channel: str | None
    blocked: bool
    margin: float
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "previous_leader": self.previous_leader,
            "from_rank": self.from_rank,
            "gap": round(self.gap, 4),
            "leader_advantage": round(self.leader_advantage, 4),
            "advantage_channel": self.advantage_channel,
            "blocked": self.blocked,
            "margin": self.margin,
            "tolerance": self.tolerance,
        }


def leader_advantage(leader: Any, rival: Any, weights: Any) -> tuple[float, str | None]:
    """Largest weighted single-channel lead of ``leader`` over ``rival`` (0, None when none).

    Channels are the discrimination's own (``CandidateDiscrimination``) with the weights the
    engine scored them with; negative evidence counts for the leader when the rival carries more.
    """
    best_value, best_channel = 0.0, None
    for channel in POSITIVE_CHANNELS:
        value = float(getattr(weights, channel)) * (
            float(getattr(leader, channel)) - float(getattr(rival, channel))
        )
        if value > best_value:
            best_value, best_channel = value, channel
    negative = float(weights.negative) * (float(rival.negative) - float(leader.negative))
    if negative > best_value:
        best_value, best_channel = negative, "negative"
    return best_value, best_channel


def decide_promotion(
    order: Sequence[str],
    discrimination: Any,
    *,
    margin: float,
    tolerance: float,
    pool: int = PROMOTION_POOL,
) -> Promotion | None:
    """The promotion the rule makes for the shortlist ``order`` (group ids, leader first), or
    None. ``discrimination`` is the gate window's ``DiscriminationResult``; members of the pool
    it did not score are ignored. Ties on D keep the better-placed group."""
    if margin <= 0 or discrimination is None or not getattr(discrimination, "scores", ()):
        return None
    by_group = discrimination.by_group()
    top = list(order[: max(int(pool), 1)])
    if len(top) < 2 or top[0] not in by_group:
        return None
    present = [group_id for group_id in top if group_id in by_group]
    best = max(present, key=lambda group_id: (by_group[group_id].total, -top.index(group_id)))
    if best == top[0]:
        return None
    gap = float(by_group[best].total) - float(by_group[top[0]].total)
    if gap < margin:
        return None
    advantage, channel = leader_advantage(by_group[top[0]], by_group[best], discrimination.weights)
    return Promotion(
        promoted=best,
        previous_leader=top[0],
        from_rank=top.index(best) + 1,
        gap=gap,
        leader_advantage=advantage,
        advantage_channel=channel,
        blocked=advantage > tolerance,
        margin=float(margin),
        tolerance=float(tolerance),
    )


def apply_promotion(fusion: FusionResult, promotion: Promotion | None) -> FusionResult:
    """Move the promoted group to first place; the groups it passed shift down one place.

    Each position keeps its original score and relative score, so the scores stay monotone, the
    margin and every set of the first N places are unchanged except for the order inside the
    first ``from_rank`` places. A blocked or absent promotion returns the input unchanged.
    """
    if promotion is None or promotion.blocked:
        return fusion
    entries = list(fusion.entries)
    ids = [entry.group_id for entry in entries]
    if promotion.promoted not in ids:
        return fusion
    index = ids.index(promotion.promoted)
    if index == 0:
        return fusion
    moved = [entries[index], *entries[:index], *entries[index + 1 :]]
    return replace(
        fusion,
        entries=tuple(
            replace(entry, score=slot.score, relative_score=slot.relative_score, rank=rank)
            for rank, (entry, slot) in enumerate(zip(moved, entries, strict=True), 1)
        ),
    )


def order_before_promotion(order: Sequence[str], record: dict[str, Any] | None) -> list[str]:
    """The shortlist order the rule saw, from the displayed order and its promotion record."""
    order = list(order)
    if not record or record.get("blocked") or not order or order[0] != record.get("promoted"):
        return order
    from_rank = int(record.get("from_rank") or 1)
    head = order[1:from_rank]
    return [*head, order[0], *order[from_rank:]]
