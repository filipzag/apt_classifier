"""Distinguishing the top three: the exclusive-evidence promotion rule (measured, not adopted).

The learned combiner (combiner.py) orders the shortlist by how the classifiers agree; the
candidate-relative discrimination (separation.py) measures, for the naming gate's window, the
observed evidence each candidate explains that the others do not (its discriminative mass D).
The rule asks whether a top-three member holds clearly more of that evidence than the leader:

- its D exceeds the leader's by at least ``margin``, and
- no single evidence channel clearly favours the leader: the leader's largest weighted
  per-channel advantage over it is at most ``tolerance`` (negative evidence counts for the
  leader when the rival carries more).

Pre-registered on 2026-10-01 as a reorder (move that member to first place, never name it),
it lifted the seed-7 folds +5/-0 but only +1/-1 and +1/-0 on the fresh seeds 41 and 53, so it
does not reorder anything (docs 10.10, 21.3.12). It remains as a pure function for the
offline replay (``scripts/top3_replay.py``) and as the "evidence pulls both ways" signal of the
analyst's Top-three view (``top3_view.py``). Neither the engine nor the combiner imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

PROMOTION_POOL = 3
# the pre-registered values (docs 10.10); found on the seed-7 folds and CTI-HAL
PREREGISTERED_MARGIN = 1.25
PREREGISTERED_TOLERANCE = 0.75
# channels that add to the discriminative mass; negative evidence subtracts (see separation.py)
POSITIVE_CHANNELS = ("technique", "behavior", "software", "soft")


@dataclass(frozen=True)
class Promotion:
    """What the rule says about a shortlist: ``promoted`` would move up unless ``blocked``."""

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
    margin: float = PREREGISTERED_MARGIN,
    tolerance: float = PREREGISTERED_TOLERANCE,
    pool: int = PROMOTION_POOL,
) -> Promotion | None:
    """The rule's verdict for the shortlist ``order`` (group ids, leader first), or None when
    no top-three member leads the leader's D by ``margin``. ``discrimination`` is the gate
    window's ``DiscriminationResult``; pool members it did not score are ignored. Ties on D
    keep the better-placed group."""
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
