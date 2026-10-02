"""Documented-overlap clusters: two groups ATT&CK documents as overlapping, named together.

Some groups cannot be told apart from the evidence because ATT&CK itself links them and their
profiles nest or barely differ (Carbanak's 9 documented techniques all lie inside FIN7's). When the
leader of the naming gate's window has such a *twin* in the window, the pair can be named as one
cluster ("FIN7 / Carbanak (documented overlap)") if, merged into one unit, it clears the same
evidence bars against the rest of the window (docs 10.11, 21.3.15).

The link must document the same activity (``ATTRIBUTION_CLUSTER_SCOPE=documented``, the default
since 2026-10-02). Either ATT&CK's descriptions document the pair as the same or overlapping
activity (the reviewed table in ``attack/same_activity.py``), or the profiles nest and the review
did not find ATT&CK calling the two distinct. A mere mention is not enough: many descriptions
name another group to say it is a different actor. Two groups that only share one rarely used
malware family (Sandworm Team and APT38 both used KillDisk) are different actors too. The
Top-three view still shows those links, but they never make a twin. Scope ``full`` is the first,
pre-registered rule (any documented relation); ``nested`` allows nesting pairs only.

The merged unit is scored like ``separation.discriminate`` scores a group, with the pair counted
once: an item is the unit's when any member documents it (hierarchy and negative factors: the
best member; behaviour: the best member's similarity per observation), and exclusivity is counted
over units. Evidence the pair shares is therefore exclusive to the pair, which is exactly what the
two members cannot be separated by. With singleton units the totals equal ``discriminate``'s.

This module is outside the combiner fingerprint (fusion.py, separation.py, classifiers/).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from apt_classifier.attack.related import description_links, related_to
from apt_classifier.attack.same_activity import (
    is_false_match,
    review_status,
    review_verdict,
    same_activity_statement,
)
from apt_classifier.classifiers.base import (
    hierarchy_matches_for,
    kb_mapping,
    negative_hierarchy_factor,
)
from apt_classifier.classifiers.software import (
    COMMODITY_TOOL_WEIGHT_CAP,
    software_rarity_weights,
)
from apt_classifier.separation import DiscriminationWeights, technique_document_counts

# one definition shared by the naming gate and the analyst's Top-three view
OVERLAP_CONTAINMENT = 0.8
OVERLAP_SEPARATION = 0.5
MAX_MEMBERS = 2

# how ATT&CK links two groups, strongest first
LINK_SAME_ACTIVITY = "same_activity"  # descriptions document the same activity (reviewed table)
LINK_NESTING = "nesting"  # related, profiles nest, and not reviewed as distinct
LINK_DESCRIPTION = "description"  # a description names the other, but not as the same activity
LINK_SHARED_SOFTWARE = "shared_software"  # related only through rarely used malware
SAME_ACTIVITY_LINKS = (LINK_SAME_ACTIVITY, LINK_NESTING)
CLUSTER_SCOPES = ("documented", "full", "nested")
DEFAULT_CLUSTER_SCOPE = "documented"


def pair_separation(own: float, other: float, floor: float) -> float:
    """Candidate-relative separation of one group over another, as the gate computes it."""
    denominator = max(own, other, floor)
    return max(-1.0, min(1.0, (own - other) / denominator)) if denominator > 0 else 0.0


def containment(kb: Any, first: str, second: str) -> float:
    """Share of the smaller documented profile that the larger one also documents."""
    profiles = kb_mapping(kb, "group_techniques")
    a, b = set(profiles.get(first, ()) or ()), set(profiles.get(second, ()) or ())
    smaller = min(len(a), len(b))
    return len(a & b) / smaller if smaller else 0.0


def link_kind(kb: Any, first: str, second: str) -> str | None:
    """How ATT&CK links two groups (``LINK_*``), or None when it does not relate them.

    Fail closed: the nesting route needs no reviewed verdict against the pair (distinct, unclear
    or a false match), and a description link the review has not seen blocks it as well, since
    the description may call the two distinct. A reviewed false match is no link at all."""
    related = related_to(kb, first).get(second)
    if not related:
        return None
    described = bool(description_links(kb).get(first, {}).get(second))
    status = review_status(kb, first, second)
    if status == "same_activity":
        return LINK_SAME_ACTIVITY
    false_match = is_false_match(first, second)
    if described and false_match and not shares_rare_software(kb, first, second):
        return None  # the only relation was the false description match
    if status is None and not described and containment(kb, first, second) >= OVERLAP_CONTAINMENT:
        return LINK_NESTING
    if described and not false_match:
        return LINK_DESCRIPTION
    return LINK_SHARED_SOFTWARE


def shares_rare_software(kb: Any, first: str, second: str) -> bool:
    """Both groups document a non-tool software used by at most three groups (the software
    half of attack/related.py, which keeps only one reason per pair)."""
    from apt_classifier.attack.related import RARE_SOFTWARE_MAX_USERS

    groups = getattr(kb, "groups", {}) or {}
    software = getattr(kb, "software", {}) or {}
    a = set(getattr(groups.get(first), "software_ids", ()) or ())
    b = set(getattr(groups.get(second), "software_ids", ()) or ())
    for software_id in a & b:
        if str(getattr(software.get(software_id), "software_type", "") or "") == "tool":
            continue
        users = sum(1 for group in groups.values() if software_id in (group.software_ids or ()))
        if 2 <= users <= RARE_SOFTWARE_MAX_USERS:
            return True
    return False


def link_reason(kb: Any, first: str, second: str, kind: str | None = None) -> str | None:
    """The ATT&CK link a pair's kind rests on, in words; None for an unrelated pair."""
    kind = kind if kind is not None else link_kind(kb, first, second)
    if kind is None:
        return None
    related = str(related_to(kb, first).get(second) or "")
    if kind == LINK_SAME_ACTIVITY:
        return str(same_activity_statement(kb, first, second))
    if kind == LINK_NESTING:
        nest = containment(kb, first, second)
        return f"{related}; their documented technique profiles nest ({nest:.0%})"
    if kind == LINK_DESCRIPTION:
        text = str(description_links(kb).get(first, {}).get(second) or related)
        verdict = review_verdict(first, second)
        return f"{text} (reviewed: {verdict})" if verdict else f"{text} (not reviewed)"
    return related


def recorded_cluster_scope(state: Any, default: str = DEFAULT_CLUSTER_SCOPE) -> str:
    """The cluster scope an analysis selected its twin under (recorded with the combiner state),
    else ``default``."""
    context = getattr(state, "context", state)
    combiner = getattr(context, "combiner", None)
    scope = combiner.get("cluster_scope") if isinstance(combiner, dict) else None
    return str(scope) if scope in CLUSTER_SCOPES else default


def scope_counts_as_one(kind: str | None, nested: bool, scope: str) -> bool:
    """Whether a link of this kind may be read as one cluster under ``scope`` (the link part of
    ``twin_link``; the window and separation parts are the gate's)."""
    if kind is None:
        return False
    if scope == "nested":
        return nested
    if scope == "documented":
        return kind in SAME_ACTIVITY_LINKS
    return True


def split_related(
    kb: Any,
    group_id: str,
    related: dict[str, str],
    *,
    scope: str = DEFAULT_CLUSTER_SCOPE,
    named_cluster: tuple[str, ...] | list[str] = (),
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """A leader's ATT&CK-related groups split into links that may be read as one cluster under
    ``scope`` ``(id, reason)`` and the others ``(id, reason, kind)``. A named cluster's twin is
    always in the first list; a reviewed false match is in neither."""
    same: list[tuple[str, str]] = []
    other: list[tuple[str, str, str]] = []
    members = {str(gid) for gid in named_cluster}
    for gid, reason in related.items():
        try:
            kind = link_kind(kb, group_id, gid)
            text = link_reason(kb, group_id, gid, kind) if kind else None
            nested = containment(kb, group_id, gid) >= OVERLAP_CONTAINMENT
        except Exception:  # noqa: BLE001 - overlap hints never break a page or an export
            kind, text, nested = LINK_SHARED_SOFTWARE, None, False
        if gid in members or scope_counts_as_one(kind, nested, scope):
            same.append((gid, text or reason))
        elif kind is not None:
            other.append((gid, text or reason, kind))
    return same, other


def twin_link(
    kb: Any, first: str, second: str, *, separation: float, scope: str = DEFAULT_CLUSTER_SCOPE
) -> dict[str, Any] | None:
    """Why ``second`` is ``first``'s documented twin under ``scope``, or None.

    Every scope needs an ATT&CK relation and nesting profiles or a pair separation below
    ``OVERLAP_SEPARATION``. ``documented`` also needs the link to document the same activity
    (``SAME_ACTIVITY_LINKS``); ``nested`` needs nesting profiles. The reason names the link the
    kind rests on and does not depend on the separation."""
    if scope not in CLUSTER_SCOPES:
        raise ValueError(f"unknown cluster scope {scope!r}; expected one of {CLUSTER_SCOPES}")
    kind = link_kind(kb, first, second)
    if kind is None:
        return None
    nest = containment(kb, first, second)
    nested = nest >= OVERLAP_CONTAINMENT
    if not nested and abs(separation) >= OVERLAP_SEPARATION:
        return None
    if not scope_counts_as_one(kind, nested, scope):
        return None
    return {
        "reason": str(link_reason(kb, first, second, kind)),
        "link": kind,
        "nested": nested,
        "containment": nest,
    }


def _exclusivity(documenting: int, unit_count: int) -> float:
    if unit_count <= 1:
        return 1.0
    return max(0.0, 1.0 - (documenting - 1) / (unit_count - 1))


@dataclass(frozen=True)
class ClusterCandidate:
    """The leader's documented twin in the gate window and the merged pair's evidence."""

    members: tuple[str, ...]
    reason: str
    link: str
    nested: bool
    containment: float
    pair_separation: float
    separation: float
    discriminative: float
    runner_up_id: str | None
    probability: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": list(self.members),
            "reason": self.reason,
            "link": self.link,
            "nested": self.nested,
            "containment": round(self.containment, 4),
            "pair_separation": round(self.pair_separation, 4),
            "separation": round(self.separation, 4),
            "discriminative": round(self.discriminative, 4),
            "runner_up_id": self.runner_up_id,
            "probability": None if self.probability is None else round(self.probability, 4),
        }


def unit_totals(
    kb: Any,
    units: list[tuple[str, ...]],
    context: Any,
    weights: DiscriminationWeights | None = None,
    *,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
    soft_techniques: dict[str, float] | None = None,
) -> list[float]:
    """Discriminative mass of each unit (a tuple of group ids) against the other units."""
    weights = weights or DiscriminationWeights()
    count = len(units)
    if not count:
        return []
    groups = [gid for unit in units for gid in unit]
    unit_of = {gid: index for index, unit in enumerate(units) for gid in unit}
    total_groups = len(kb_mapping(kb, "groups")) or 1
    doc_counts = technique_document_counts(kb)

    def specificity(technique_id: str) -> float:
        return max(0.0, 1.0 - doc_counts.get(technique_id, 0) / total_groups)

    observed: dict[str, float] = dict(getattr(context, "observed_techniques", {}) or {})
    view = SimpleNamespace(
        kb=kb,
        parent_policy=str(getattr(context, "parent_policy", "ancestor_compatible")),
        observed_techniques=observed,
    )

    def unit_factors(items: dict[str, float], view_: Any) -> list[dict[str, float]]:
        factors: list[dict[str, float]] = [{} for _ in units]
        for gid in groups:
            matches = hierarchy_matches_for(view_, gid, observed_techniques=set(items))
            unit = factors[unit_of[gid]]
            for tid, match in matches.items():
                unit[tid] = max(unit.get(tid, 0.0), float(match.factor))
        return factors

    technique = [0.0] * count
    factors = unit_factors(observed, view)
    for technique_id, confidence in observed.items():
        documenting = sum(1 for unit in factors if unit.get(technique_id, 0.0) > 0)
        if not documenting:
            continue
        x = _exclusivity(documenting, count)
        s = specificity(technique_id)
        for index, unit in enumerate(factors):
            factor = unit.get(technique_id, 0.0)
            if factor > 0:
                technique[index] += float(confidence) * factor * x * s

    soft = [0.0] * count
    soft_items = {
        tid: float(value)
        for tid, value in (soft_techniques or {}).items()
        if tid not in observed and float(value) > 0
    }
    if soft_items:
        soft_view = SimpleNamespace(
            kb=kb, parent_policy=view.parent_policy, observed_techniques=soft_items
        )
        soft_factors = unit_factors(soft_items, soft_view)
        for technique_id, value in soft_items.items():
            documenting = sum(1 for unit in soft_factors if unit.get(technique_id, 0.0) > 0)
            if not documenting:
                continue
            x = _exclusivity(documenting, count)
            s = specificity(technique_id)
            for index, unit in enumerate(soft_factors):
                factor = unit.get(technique_id, 0.0)
                if factor > 0:
                    soft[index] += value * factor * x * s

    negative = [0.0] * count
    for technique_id in sorted(getattr(context, "negative_techniques", set()) or set()):
        per_unit = [
            max(negative_hierarchy_factor(view, gid, technique_id) for gid in unit)
            for unit in units
        ]
        documenting = sum(1 for value in per_unit if value > 0)
        if not documenting:
            continue
        x = _exclusivity(documenting, count)
        s = specificity(technique_id)
        for index, factor in enumerate(per_unit):
            if factor > 0:
                negative[index] += factor * x * s

    behavior = [0.0] * count
    best: dict[str, dict[int, float]] = {}
    for match in getattr(context, "specific_behavior_matches", ()) or ():
        gid = str(getattr(match, "group_id", ""))
        if gid not in unit_of:
            continue
        per_observation = best.setdefault(str(getattr(match, "observation_id", "")), {})
        index = unit_of[gid]
        score = float(getattr(match, "score", 0.0) or 0.0)
        if score > per_observation.get(index, 0.0):
            per_observation[index] = score
    for per_observation in best.values():
        for index, score in per_observation.items():
            alternative = max(
                (value for other, value in per_observation.items() if other != index),
                default=0.0,
            )
            behavior[index] += max(0.0, score - max(alternative, behavior_threshold))

    software = [0.0] * count
    software_ids = set(getattr(context, "software_ids", set()) or set())
    if software_ids:
        rarity = software_rarity_weights(kb, software_ids, tool_weight_cap=tool_weight_cap)
        max_weight = math.log((total_groups + 1) / 2.0) + 1.0
        decisive = getattr(context, "decisive_software", None) or {}
        decisive_ids = set(decisive.get("software_ids", ()) or ())
        decisive_factor = float(decisive.get("factor", 1.0) or 1.0)
        group_software = kb_mapping(kb, "group_software")
        for software_id in sorted(software_ids):
            documented = [
                index
                for index, unit in enumerate(units)
                if any(software_id in set(group_software.get(gid, ()) or ()) for gid in unit)
            ]
            if not documented:
                continue
            value = min(1.0, rarity.get(software_id, 1.0) / max_weight)
            if software_id in decisive_ids:
                value *= decisive_factor
            x = _exclusivity(len(documented), count)
            for index in documented:
                software[index] += value * x

    return [
        max(
            0.0,
            weights.technique * technique[i]
            + weights.behavior * behavior[i]
            + weights.software * software[i]
            + weights.soft * soft[i]
            - weights.negative * negative[i],
        )
        for i in range(count)
    ]


def select_twin(
    kb: Any,
    discrimination: Any,
    *,
    floor: float = 1.0,
    scope: str = DEFAULT_CLUSTER_SCOPE,
) -> dict[str, Any] | None:
    """The leader's documented twin in the gate window, or None.

    A twin is a window member that qualifies under ``scope`` (``twin_link``). With several, the
    one with the largest window mass wins (ties: window order); a member the scope excludes is
    skipped, so a narrower scope can pick a lighter twin than ``full`` would."""
    if scope not in CLUSTER_SCOPES:
        raise ValueError(f"unknown cluster scope {scope!r}; expected one of {CLUSTER_SCOPES}")
    scores = list(getattr(discrimination, "scores", ()) or ())
    if len(scores) < 2:
        return None
    leader = scores[0]
    best: dict[str, Any] | None = None
    for item in scores[1:]:
        separation = pair_separation(float(leader.total), float(item.total), floor)
        try:
            link = twin_link(kb, leader.group_id, item.group_id, separation=separation, scope=scope)
        except Exception:  # noqa: BLE001 - overlap hints never break the decision
            return None
        if link is None:
            continue
        if best is None or float(item.total) > best["total"]:
            best = {
                "group_id": item.group_id,
                **link,
                "pair_separation": separation,
                "total": float(item.total),
            }
    return best


def cluster_candidate(
    kb: Any,
    discrimination: Any,
    context: Any,
    *,
    probabilities: dict[str, float] | None,
    floor: float = 1.0,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
    soft_techniques: dict[str, float] | None = None,
    scope: str = DEFAULT_CLUSTER_SCOPE,
) -> ClusterCandidate | None:
    """The leader + twin cluster scored against the rest of the gate window, or None."""
    twin = select_twin(kb, discrimination, floor=floor, scope=scope)
    if twin is None:
        return None
    ids = [item.group_id for item in discrimination.scores]
    leader = ids[0]
    members = (leader, twin["group_id"])
    others = [gid for gid in ids if gid not in members]
    units: list[tuple[str, ...]] = [members, *((gid,) for gid in others)]
    totals = unit_totals(
        kb,
        units,
        context,
        getattr(discrimination, "weights", None),
        tool_weight_cap=tool_weight_cap,
        behavior_threshold=behavior_threshold,
        soft_techniques=soft_techniques,
    )
    own = totals[0]
    rest = totals[1:]
    runner_index = max(range(len(rest)), key=lambda i: (rest[i], -i)) if rest else None
    runner = rest[runner_index] if runner_index is not None else 0.0
    denominator = max(own, runner, float(floor))
    separation = (own - runner) / denominator if denominator > 0 else 0.0
    probability = (
        float(probabilities.get(leader, 0.0)) + float(probabilities.get(twin["group_id"], 0.0))
        if probabilities is not None
        else None
    )
    return ClusterCandidate(
        members=members,
        reason=twin["reason"],
        link=twin["link"],
        nested=twin["nested"],
        containment=twin["containment"],
        pair_separation=twin["pair_separation"],
        separation=max(-1.0, min(1.0, separation)),
        discriminative=own,
        runner_up_id=others[runner_index] if runner_index is not None else None,
        probability=probability,
    )


def cluster_label(kb: Any, members: tuple[str, ...] | list[str]) -> str:
    """'FIN7 / Carbanak (documented overlap)'."""
    groups = getattr(kb, "groups", {}) or {}
    names = [str(getattr(groups.get(gid), "name", gid) or gid) for gid in members]
    return " / ".join(names) + " (documented overlap)"
