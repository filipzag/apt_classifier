"""The one naming rule.

Name actor A only when the ranker's first choice is A and leads its second by at least
``margin``; when a proposer (a frontier model) answered, it must also have decided to name and
put A first. Otherwise nothing is named and the ranking is shown as it is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    first: str | None  # the proposer's first candidate, matched to an actor id (None: none matched)
    decision: str  # "name", "open" or "not_in_attack"


def decide(
    first: str | None,
    lead: float | None,
    margin: float,
    proposal: Proposal | None = None,
) -> str | None:
    """The named actor, or None."""
    if first is None or lead is None or lead < margin:
        return None
    if proposal is not None and (proposal.decision != "name" or proposal.first != first):
        return None
    return first
