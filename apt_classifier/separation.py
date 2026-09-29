"""Candidate-relative discriminative support between the top fused candidates.

Rank fusion measures agreement between classifiers, not evidence: with ``RRF_K=20`` a leader
that every classifier ranks first against a runner-up ranked second everywhere is only 4.5
points ahead, and shared evidence (commodity tooling, generic techniques, the Carbanak/FIN7
overlap) counts for both sides. This module scores each of the top-K candidates by the
observed evidence that the *other* top-K candidates do not explain:

- accepted techniques, weighted by acceptance confidence, the hierarchy factor, how few of the
  top-K document them (``x``) and their documented specificity (``s``);
- actor-behaviour matches, by how much the candidate's best similarity for an observation
  exceeds the best alternative's (and the retrieval threshold);
- named software, by normalised rarity (ATT&CK tools capped) and exclusivity, with the
  decisive-malware factor applied;
- telemetry-validated negative interview answers, subtracted where they discriminate.

Shared evidence contributes nothing to any side, so documented overlap needs no special case.
The result gives a *separation* between the leader and its best-supported alternative, the
leader's evidence specificity, and an optional re-ordering of the top-K that blends the
discriminative mass into the RRF score. Everything is deterministic and settings-gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from apt_classifier.attack.related import related_to
from apt_classifier.classifiers.base import (
    hierarchy_matches_for,
    kb_mapping,
    negative_hierarchy_factor,
)
from apt_classifier.classifiers.software import (
    COMMODITY_TOOL_WEIGHT_CAP,
    software_rarity_weights,
)
from apt_classifier.fusion import FusionResult

_DOC_COUNT_CACHE: dict[int, tuple[tuple[str, int], dict[str, int]]] = {}


@dataclass(frozen=True)
class DiscriminationWeights:
    technique: float = 1.0
    behavior: float = 2.0
    software: float = 1.5
    negative: float = 0.3
    # contrastive near-miss techniques (unaccepted candidates that split the shortlist)
    soft: float = 1.0


@dataclass(frozen=True)
class CandidateDiscrimination:
    group_id: str
    rrf_score: float
    technique: float
    behavior: float
    software: float
    negative: float
    total: float
    soft: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "group_id": self.group_id,
            "rrf": round(self.rrf_score, 6),
            "technique": round(self.technique, 4),
            "behavior": round(self.behavior, 4),
            "software": round(self.software, 4),
            "negative": round(self.negative, 4),
            "soft": round(self.soft, 4),
            "total": round(self.total, 4),
        }


@dataclass(frozen=True)
class DiscriminationResult:
    scores: tuple[CandidateDiscrimination, ...] = ()
    top_k: int = 0
    leader_id: str | None = None
    runner_up_id: str | None = None
    separation: float = 0.0
    leader_specificity: float = 0.0
    runner_up_related: bool = False
    floor: float = 1.0
    weights: DiscriminationWeights = field(default_factory=DiscriminationWeights)

    def by_group(self) -> dict[str, CandidateDiscrimination]:
        return {item.group_id: item for item in self.scores}

    @property
    def leader_total(self) -> float:
        return self.scores[0].total if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "leader_id": self.leader_id,
            "runner_up_id": self.runner_up_id,
            "separation": round(self.separation, 4),
            "leader_specificity": round(self.leader_specificity, 4),
            "runner_up_related": self.runner_up_related,
            "floor": self.floor,
            "candidates": [item.to_dict() for item in self.scores],
        }


def technique_document_counts(kb: Any) -> dict[str, int]:
    """Number of groups documenting each technique, memoized per knowledge-base object."""
    groups = kb_mapping(kb, "groups")
    stamp = (str(getattr(kb, "version", "")), len(groups))
    cached = _DOC_COUNT_CACHE.get(id(kb))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    counts: dict[str, int] = {}
    for technique_ids in kb_mapping(kb, "group_techniques").values():
        for technique_id in set(technique_ids):
            counts[technique_id] = counts.get(technique_id, 0) + 1
    _DOC_COUNT_CACHE[id(kb)] = (stamp, counts)
    return counts


def _exclusivity(documenting: int, candidate_count: int) -> float:
    """1 when one candidate documents the item, 0 when every candidate does."""
    if candidate_count <= 1:
        return 1.0
    return max(0.0, 1.0 - (documenting - 1) / (candidate_count - 1))


def discriminate(
    kb: Any,
    fusion: FusionResult,
    context: Any,
    weights: DiscriminationWeights | None = None,
    *,
    top_k: int = 5,
    floor: float = 1.0,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
    soft_techniques: dict[str, float] | None = None,
) -> DiscriminationResult:
    """Score the top-K fused candidates by the evidence the other candidates do not explain.

    ``soft_techniques`` (technique id -> weight in [0, 1]) are near-miss candidates that were not
    accepted; they enter a separate ``soft`` channel with the same exclusivity and specificity
    weighting, so a generic near miss shared by the shortlist still counts for nobody while a
    distinctive one sharpens the separation. Accepted techniques are never double counted.
    """
    weights = weights or DiscriminationWeights()
    candidates = [entry for entry in fusion.entries[: max(int(top_k), 1)]]
    if not candidates:
        return DiscriminationResult(top_k=0, floor=floor, weights=weights)
    ids = [entry.group_id for entry in candidates]
    count = len(ids)
    total_groups = len(kb_mapping(kb, "groups")) or 1
    doc_counts = technique_document_counts(kb)

    def specificity(technique_id: str) -> float:
        return max(0.0, 1.0 - doc_counts.get(technique_id, 0) / total_groups)

    # --- techniques (accepted only) ---
    observed: dict[str, float] = dict(getattr(context, "observed_techniques", {}) or {})
    # The hierarchy helpers read ``kb`` and ``parent_policy`` from a context; build a minimal
    # view so any object with observed techniques (tests, benchmarks) can be scored.
    view = SimpleNamespace(
        kb=kb,
        parent_policy=str(getattr(context, "parent_policy", "ancestor_compatible")),
        observed_techniques=observed,
    )
    factors: dict[str, dict[str, float]] = {}
    for group_id in ids:
        matches = hierarchy_matches_for(view, group_id, observed_techniques=set(observed))
        factors[group_id] = {tid: float(match.factor) for tid, match in matches.items()}
    technique_score: dict[str, float] = dict.fromkeys(ids, 0.0)
    specificity_sum: dict[str, float] = dict.fromkeys(ids, 0.0)
    confidence_sum: dict[str, float] = dict.fromkeys(ids, 0.0)
    for technique_id, confidence in observed.items():
        documenting = sum(1 for group_id in ids if factors[group_id].get(technique_id, 0.0) > 0)
        if documenting == 0:
            continue
        x = _exclusivity(documenting, count)
        s = specificity(technique_id)
        for group_id in ids:
            factor = factors[group_id].get(technique_id, 0.0)
            if factor <= 0:
                continue
            technique_score[group_id] += float(confidence) * factor * x * s
            specificity_sum[group_id] += float(confidence) * s
            confidence_sum[group_id] += float(confidence)

    # --- contrastive near-miss techniques ---
    soft_score: dict[str, float] = dict.fromkeys(ids, 0.0)
    soft_items = {
        tid: float(value)
        for tid, value in (soft_techniques or {}).items()
        if tid not in observed and float(value) > 0
    }
    if soft_items:
        soft_view = SimpleNamespace(
            kb=kb, parent_policy=view.parent_policy, observed_techniques=soft_items
        )
        soft_factors: dict[str, dict[str, float]] = {}
        for group_id in ids:
            matches = hierarchy_matches_for(
                soft_view, group_id, observed_techniques=set(soft_items)
            )
            soft_factors[group_id] = {tid: float(m.factor) for tid, m in matches.items()}
        for technique_id, value in soft_items.items():
            documenting = sum(
                1 for group_id in ids if soft_factors[group_id].get(technique_id, 0.0) > 0
            )
            if documenting == 0:
                continue
            x = _exclusivity(documenting, count)
            s = specificity(technique_id)
            for group_id in ids:
                factor = soft_factors[group_id].get(technique_id, 0.0)
                if factor > 0:
                    soft_score[group_id] += value * factor * x * s

    # --- negative evidence ---
    negative_score: dict[str, float] = dict.fromkeys(ids, 0.0)
    for technique_id in sorted(getattr(context, "negative_techniques", set()) or set()):
        neg_factors = {gid: negative_hierarchy_factor(view, gid, technique_id) for gid in ids}
        documenting = sum(1 for value in neg_factors.values() if value > 0)
        if documenting == 0:
            continue
        x = _exclusivity(documenting, count)
        s = specificity(technique_id)
        for group_id, factor in neg_factors.items():
            if factor > 0:
                negative_score[group_id] += factor * x * s

    # --- actor behaviour ---
    behavior_score: dict[str, float] = dict.fromkeys(ids, 0.0)
    best: dict[str, dict[str, float]] = {}
    for match in getattr(context, "specific_behavior_matches", ()) or ():
        group_id = str(getattr(match, "group_id", ""))
        if group_id not in technique_score:
            continue
        observation_id = str(getattr(match, "observation_id", ""))
        score = float(getattr(match, "score", 0.0) or 0.0)
        per_observation = best.setdefault(observation_id, {})
        if score > per_observation.get(group_id, 0.0):
            per_observation[group_id] = score
    for per_observation in best.values():
        for group_id, score in per_observation.items():
            alternative = max(
                (value for other, value in per_observation.items() if other != group_id),
                default=0.0,
            )
            behavior_score[group_id] += max(0.0, score - max(alternative, behavior_threshold))

    # --- software ---
    software_score: dict[str, float] = dict.fromkeys(ids, 0.0)
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
                gid for gid in ids if software_id in set(group_software.get(gid, ()) or ())
            ]
            if not documented:
                continue
            value = min(1.0, rarity.get(software_id, 1.0) / max_weight)
            if software_id in decisive_ids:
                value *= decisive_factor
            x = _exclusivity(len(documented), count)
            for gid in documented:
                software_score[gid] += value * x

    scores = []
    for entry in candidates:
        gid = entry.group_id
        total = (
            weights.technique * technique_score[gid]
            + weights.behavior * behavior_score[gid]
            + weights.software * software_score[gid]
            + weights.soft * soft_score[gid]
            - weights.negative * negative_score[gid]
        )
        scores.append(
            CandidateDiscrimination(
                group_id=gid,
                rrf_score=float(entry.score),
                technique=technique_score[gid],
                behavior=behavior_score[gid],
                software=software_score[gid],
                negative=negative_score[gid],
                total=max(0.0, total),
                soft=soft_score[gid],
            )
        )
    leader = scores[0]
    others = scores[1:]
    runner = max(others, key=lambda item: item.total) if others else None
    runner_total = runner.total if runner is not None else 0.0
    denominator = max(leader.total, runner_total, float(floor))
    separation = (leader.total - runner_total) / denominator if denominator > 0 else 0.0
    separation = max(-1.0, min(1.0, separation))
    leader_specificity = (
        specificity_sum[leader.group_id] / confidence_sum[leader.group_id]
        if confidence_sum[leader.group_id] > 0
        else 0.0
    )
    related = False
    if runner is not None:
        try:
            related = runner.group_id in related_to(kb, leader.group_id)
        except Exception:  # noqa: BLE001 - overlap hints never break the decision
            related = False
    return DiscriminationResult(
        scores=tuple(scores),
        top_k=count,
        leader_id=leader.group_id,
        runner_up_id=runner.group_id if runner is not None else None,
        separation=separation,
        leader_specificity=leader_specificity,
        runner_up_related=related,
        floor=float(floor),
        weights=weights,
    )


def apply_discriminative_rerank(
    fusion: FusionResult, result: DiscriminationResult, beta: float, floor: float = 1.0
) -> FusionResult:
    """Blend the discriminative mass into the RRF score of the top-K and re-rank.

    ``score'(g) = rrf(g) * (1 + beta * D(g) / max(D_max, floor))`` for the top-K; other entries
    keep their score. The factor is >= 1, so the top-K set never changes; only its order and the
    relative scores do. ``beta <= 0`` returns the input unchanged.
    """
    if beta <= 0 or not result.scores or not fusion.entries:
        return fusion
    totals = result.by_group()
    d_max = max(item.total for item in result.scores)
    denominator = max(d_max, float(floor))
    adjusted: list[tuple[float, int, Any]] = []
    for index, entry in enumerate(fusion.entries):
        item = totals.get(entry.group_id)
        score = float(entry.score)
        if item is not None and index < result.top_k:
            score *= 1.0 + float(beta) * item.total / denominator
        adjusted.append((score, index, entry))
    adjusted.sort(key=lambda row: (-row[0], row[1]))
    maximum = adjusted[0][0] if adjusted else 1.0
    entries = tuple(
        replace(
            entry,
            score=score,
            relative_score=100.0 * score / maximum if maximum else 0.0,
            rank=rank,
        )
        for rank, (score, _, entry) in enumerate(adjusted, 1)
    )
    return replace(fusion, entries=entries)


@dataclass(frozen=True)
class TechniqueContrast:
    """One technique's contribution to separating the shortlist."""

    technique_id: str
    confidence: float
    pending: bool
    documented_by: tuple[str, ...]
    exclusivity: float
    specificity: float
    weight: float
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_id": self.technique_id,
            "confidence": round(self.confidence, 4),
            "pending": self.pending,
            "documented_by": list(self.documented_by),
            "exclusivity": round(self.exclusivity, 4),
            "specificity": round(self.specificity, 4),
            "weight": round(self.weight, 4),
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }


def distinguishing_techniques(
    kb: Any,
    fusion: FusionResult,
    context: Any,
    *,
    top_k: int = 10,
    soft_techniques: dict[str, float] | None = None,
) -> list[TechniqueContrast]:
    """Per-technique view of what separates the top-K: which shortlist groups document it, its
    exclusivity among them, its documented specificity and the resulting weight. Accepted
    techniques and pending near misses are both listed (``pending`` marks the latter). Sorted by
    weight, so the first rows are the techniques the ranking is decided by."""
    candidates = [entry for entry in fusion.entries[: max(int(top_k), 1)]]
    ids = [entry.group_id for entry in candidates]
    count = len(ids)
    if not ids:
        return []
    total_groups = len(kb_mapping(kb, "groups")) or 1
    doc_counts = technique_document_counts(kb)
    observed: dict[str, float] = dict(getattr(context, "observed_techniques", {}) or {})
    items: list[tuple[str, float, bool]] = [(tid, float(c), False) for tid, c in observed.items()]
    items += [
        (tid, float(value), True)
        for tid, value in (soft_techniques or {}).items()
        if tid not in observed and float(value) > 0
    ]
    if not items:
        return []
    view = SimpleNamespace(
        kb=kb,
        parent_policy=str(getattr(context, "parent_policy", "ancestor_compatible")),
        observed_techniques={tid: value for tid, value, _ in items},
    )
    factors: dict[str, dict[str, float]] = {}
    for group_id in ids:
        matches = hierarchy_matches_for(
            view, group_id, observed_techniques=set(view.observed_techniques)
        )
        factors[group_id] = {tid: float(match.factor) for tid, match in matches.items()}
    rows: list[TechniqueContrast] = []
    for technique_id, confidence, pending in items:
        documented = tuple(gid for gid in ids if factors[gid].get(technique_id, 0.0) > 0)
        x = _exclusivity(len(documented), count) if documented else 0.0
        s = max(0.0, 1.0 - doc_counts.get(technique_id, 0) / total_groups)
        weight = confidence * x * s if documented else 0.0
        rows.append(
            TechniqueContrast(
                technique_id=technique_id,
                confidence=confidence,
                pending=pending,
                documented_by=documented,
                exclusivity=x,
                specificity=s,
                weight=weight,
                contributions={
                    gid: confidence * factors[gid][technique_id] * x * s for gid in documented
                },
            )
        )
    rows.sort(key=lambda row: (-row.weight, row.pending, row.technique_id))
    return rows
