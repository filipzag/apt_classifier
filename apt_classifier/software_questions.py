"""Interview questions about malware that only one top-three group uses (docs 21.3.15 C).

A malware entry ATT&CK documents for exactly one group is decisive evidence for that group when
the analyst finds it. For each of the decision's first three candidates this module lists the
malware only it documents (and whose users all sit in the naming gate's window), ranks it by how
much a "found" would move the gate's separation, and phrases one fixed question per item:
"Did you find <malware> (<S-id>) or its known artefacts?". The wording never names a group, is
never rephrased by the LLM (pilot answers stay comparable) and malware whose name is a group's
name or alias is not asked about.

How the answers count:

- **found** adds the software id to ``context.software_ids`` exactly like a name matched in the
  case text (retrieve.apply_software_answers), so it reaches the software classifier, the decisive
  boost, the combiner and the separation;
- **not found** never reaches a classifier, the fusion or the combiner. The naming gate subtracts
  ``weight x value x exclusivity`` from the mass of the unit holding the leader (the leader, or the
  named cluster when any member documents the malware) and recomputes its separation against the
  unchanged runner-up (:func:`leader_only_negative`, :func:`cluster_negative`). The penalty is
  never negative and the separation never decreases as the mass grows, so a "not found" can only
  move a name to the shortlist, never create or change one;
- **not checked** records nothing.

Nothing here is in the combiner fingerprint (fusion.py, separation.py, classifiers/): it imports
``discriminate`` and the overlap-cluster unit scoring, it does not change them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from apt_classifier.classifiers.base import kb_mapping
from apt_classifier.classifiers.software import (
    COMMODITY_TOOL_WEIGHT_CAP,
    decisive_software_factor,
    software_rarity_weights,
)
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.memo import ObjectMemo
from apt_classifier.overlap_cluster import unit_totals
from apt_classifier.questions import SelectedQuestion
from apt_classifier.separation import discriminate, technique_document_counts

SOFTWARE_HYPOTHESIS = "ATT&CK documents this malware for exactly one current top-three candidate."
MAX_ALIASES = 3
MAX_TECHNIQUES = 3
NAMED_STAGES = frozenset({"known_group_hypothesis", "known_cluster_hypothesis"})
# pilot-log answer vocabulary (the UI sends these; parse_interview_answer maps them)
ANSWER_LABELS = {"positive": "found", "negative": "not_found", "unknown": "not_checked"}

_NAME_PATTERN_CACHE: ObjectMemo[tuple[re.Pattern[str] | None]] = ObjectMemo()


def group_name_pattern(kb: Any) -> re.Pattern[str] | None:
    """Every ATT&CK group name and alias as one case-insensitive whole-word pattern."""
    groups = kb_mapping(kb, "groups")
    stamp = (str(getattr(kb, "version", "")), len(groups))
    cached = _NAME_PATTERN_CACHE.get(kb, stamp)
    if cached is not None:
        return cached[0]
    names = {
        str(name).strip()
        for group in groups.values()
        for name in (getattr(group, "name", ""), *(getattr(group, "aliases", ()) or ()))
        if name and str(name).strip()
    }
    ordered = sorted(names, key=lambda item: (-len(item), item.casefold()))
    pattern = (
        re.compile(
            r"(?<![\w-])(?:" + "|".join(re.escape(name) for name in ordered) + r")(?![\w-])",
            re.IGNORECASE,
        )
        if ordered
        else None
    )
    return _NAME_PATTERN_CACHE.put(kb, stamp, (pattern,))[0]


def _names_a_group(text: str, pattern: re.Pattern[str] | None) -> bool:
    return bool(pattern is not None and text and pattern.search(text))


@dataclass(frozen=True)
class PoolItem:
    software_id: str
    group_id: str
    target_rank: int
    users: tuple[str, ...]


def software_question_pool(
    kb: Any,
    window_ids: Iterable[str],
    top3_ids: Iterable[str],
    observed_sw: Iterable[str],
    answered_sw: Iterable[str],
    max_users: int = 1,
) -> list[PoolItem]:
    """Malware a software question may ask about, one item per (software, top-three group).

    An item is ATT&CK *malware* (never a tool) that the case has not named or found and that was
    not asked before, documented for at most ``max_users`` groups, all of them inside the gate
    window, and for exactly one of the decision's first three candidates. With the default of one
    user the candidate is the malware's only ATT&CK user. Malware whose name is a group's name or
    alias is left out: asking about it would name the group."""
    software = kb_mapping(kb, "software")
    software_groups = kb_mapping(kb, "software_groups")
    window = set(window_ids)
    top3 = list(dict.fromkeys(top3_ids))[:3]
    top = set(top3)
    skip = set(observed_sw or ()) | set(answered_sw or ())
    limit = max(1, int(max_users))
    pattern = group_name_pattern(kb)
    pool: list[PoolItem] = []
    for software_id in sorted(software):
        if software_id in skip:
            continue
        entry = software[software_id]
        if str(getattr(entry, "software_type", "") or "") != "malware":
            continue
        users = set(software_groups.get(software_id, ()) or ())
        if not users or len(users) > limit or not users <= window:
            continue
        owners = users & top
        if len(owners) != 1:
            continue
        if _names_a_group(str(getattr(entry, "name", "") or ""), pattern):
            continue
        group_id = next(iter(owners))
        pool.append(
            PoolItem(
                software_id=software_id,
                group_id=group_id,
                target_rank=top3.index(group_id) + 1,
                users=tuple(sorted(users)),
            )
        )
    return pool


def window_fusion(discrimination: Any) -> FusionResult:
    """The naming gate's window in its recorded order, as a fusion ``discriminate`` can replay
    (the totals depend on the members and the evidence only, never on the fused scores)."""
    entries = tuple(
        FusionEntry(
            group_id=item.group_id,
            score=float(item.rrf_score),
            relative_score=0.0,
            rank=index,
            classifier_ranks={},
            rank_min=index,
            rank_max=index,
            rank_std=0.0,
            contested=False,
        )
        for index, item in enumerate(getattr(discrimination, "scores", ()) or (), 1)
    )
    return FusionResult(entries=entries, active_classifiers=(), correlations={}, k=0)


def _software_view(
    kb: Any, context: Any, software_ids: Iterable[str] | None, decisive_boost: float
) -> SimpleNamespace:
    if software_ids is None:
        software = set(getattr(context, "software_ids", set()) or set())
        decisive = dict(getattr(context, "decisive_software", {}) or {})
    else:
        software = set(software_ids)
        factor, triggers = decisive_software_factor(kb, software, boost=float(decisive_boost))
        decisive = {"factor": factor, "software_ids": list(triggers)}
    return SimpleNamespace(
        observed_techniques=dict(getattr(context, "observed_techniques", {}) or {}),
        negative_techniques=set(getattr(context, "negative_techniques", set()) or set()),
        parent_policy=str(getattr(context, "parent_policy", "ancestor_compatible")),
        specific_behavior_matches=tuple(getattr(context, "specific_behavior_matches", ()) or ()),
        software_ids=software,
        decisive_software=decisive,
    )


def replay_window(
    kb: Any,
    discrimination: Any,
    context: Any,
    *,
    software_ids: Iterable[str] | None = None,
    decisive_boost: float = 0.0,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
) -> Any:
    """``discriminate`` over the recorded gate window, as the engine scored it. ``software_ids``
    replaces the case's software (the decisive factor is then recomputed with ``decisive_boost``,
    as the engine does); ``None`` keeps it, which reproduces the recorded discrimination."""
    view = _software_view(kb, context, software_ids, decisive_boost)
    return discriminate(
        kb,
        window_fusion(discrimination),
        view,
        getattr(discrimination, "weights", None),
        top_k=len(getattr(discrimination, "scores", ()) or ()),
        floor=float(getattr(discrimination, "floor", 1.0)),
        tool_weight_cap=tool_weight_cap,
        behavior_threshold=behavior_threshold,
        soft_techniques=getattr(context, "contrast_techniques", None) or None,
    )


def cluster_window_separation(
    kb: Any,
    discrimination: Any,
    context: Any,
    members: Iterable[str],
    *,
    software_ids: Iterable[str] | None = None,
    decisive_boost: float = 0.0,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
) -> float:
    """Separation of the merged pair against the rest of the gate window
    (``overlap_cluster.cluster_candidate``'s formula) with the case's software replaced."""
    pair = tuple(str(gid) for gid in members)
    ids = [item.group_id for item in getattr(discrimination, "scores", ()) or ()]
    others = [gid for gid in ids if gid not in pair]
    units: list[tuple[str, ...]] = [pair, *((gid,) for gid in others)]
    view = _software_view(kb, context, software_ids, decisive_boost)
    totals = unit_totals(
        kb,
        units,
        view,
        getattr(discrimination, "weights", None),
        tool_weight_cap=tool_weight_cap,
        behavior_threshold=behavior_threshold,
        soft_techniques=getattr(context, "contrast_techniques", None) or None,
    )
    own, rest = totals[0], totals[1:]
    runner = max(rest) if rest else 0.0
    denominator = max(own, runner, float(getattr(discrimination, "floor", 1.0)))
    separation = (own - runner) / denominator if denominator > 0 else 0.0
    return max(-1.0, min(1.0, separation))


def software_value(
    kb: Any, software_id: str, *, tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP
) -> float:
    """The separation's per-item software value (normalised rarity, before the decisive factor)."""
    total_groups = len(kb_mapping(kb, "groups")) or 1
    rarity = software_rarity_weights(kb, {software_id}, tool_weight_cap=tool_weight_cap)
    max_weight = math.log((total_groups + 1) / 2.0) + 1.0
    return min(1.0, rarity.get(software_id, 1.0) / max_weight)


def _exclusivity(documenting: int, unit_count: int) -> float:
    if unit_count <= 1:
        return 1.0
    return max(0.0, 1.0 - (documenting - 1) / (unit_count - 1))


@dataclass(frozen=True)
class NegativeSoftwareEffect:
    """What the "not found" answers did to the unit holding the leader."""

    software_ids: tuple[str, ...]  # every not-found answer
    applied: tuple[str, ...]  # the ones the unit documents (only these count)
    penalty: float
    total_before: float
    total: float
    separation_before: float
    separation: float
    members: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "software_ids": list(self.software_ids),
            "applied": list(self.applied),
            "penalty": round(self.penalty, 4),
            "total_before": round(self.total_before, 4),
            "total": round(self.total, 4),
            "separation_before": round(self.separation_before, 4),
            "separation": round(self.separation, 4),
            "members": list(self.members),
        }


def negative_software_penalty(
    kb: Any,
    units: list[tuple[str, ...]],
    software_ids: Iterable[str],
    *,
    weight: float,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
) -> tuple[float, tuple[str, ...]]:
    """``weight x value x exclusivity`` summed over the not-found items ``units[0]`` documents
    (an item counts for a unit when any member documents it; exclusivity over the units)."""
    group_software = kb_mapping(kb, "group_software")
    count = len(units)
    penalty = 0.0
    applied: list[str] = []
    for software_id in sorted(set(software_ids or ())):
        documented = [
            index
            for index, unit in enumerate(units)
            if any(software_id in set(group_software.get(gid, ()) or ()) for gid in unit)
        ]
        if not documented or documented[0] != 0:
            continue
        amount = (
            max(0.0, float(weight))
            * software_value(kb, software_id, tool_weight_cap=tool_weight_cap)
            * _exclusivity(len(documented), count)
        )
        if amount > 0:
            penalty += amount
            applied.append(software_id)
    return penalty, tuple(applied)


def penalized_separation(
    total: float, runner_total: float, penalty: float, floor: float
) -> tuple[float, float]:
    """(mass, separation) after subtracting ``penalty`` from the unit's mass; the separation
    ``(a - b) / max(a, b, floor)`` never decreases as ``a`` grows, so it can only go down."""
    reduced = max(0.0, float(total) - max(0.0, float(penalty)))
    denominator = max(reduced, float(runner_total), float(floor))
    separation = (reduced - float(runner_total)) / denominator if denominator > 0 else 0.0
    return reduced, max(-1.0, min(1.0, separation))


def leader_only_negative(
    discrimination: Any,
    kb: Any,
    ids: Iterable[str],
    weight: float,
    floor: float | None = None,
) -> NegativeSoftwareEffect:
    """The gate leader's mass and separation after the "not found" answers it documents. Every
    other candidate's mass, and so the runner-up, is left alone."""
    scores = list(getattr(discrimination, "scores", ()) or ())
    software_ids = tuple(sorted(set(ids or ())))
    separation_before = float(getattr(discrimination, "separation", 0.0))
    if not scores:
        return NegativeSoftwareEffect(software_ids, (), 0.0, 0.0, 0.0, separation_before, 0.0)
    units = [(item.group_id,) for item in scores]
    penalty, applied = negative_software_penalty(kb, units, software_ids, weight=weight)
    leader_total = float(scores[0].total)
    if penalty <= 0:
        return NegativeSoftwareEffect(
            software_ids,
            (),
            0.0,
            leader_total,
            leader_total,
            separation_before,
            separation_before,
            (scores[0].group_id,),
        )
    runner_id = getattr(discrimination, "runner_up_id", None)
    runner = discrimination.by_group().get(runner_id) if runner_id else None
    floor = float(getattr(discrimination, "floor", 1.0) if floor is None else floor)
    total, separation = penalized_separation(
        leader_total, float(runner.total) if runner is not None else 0.0, penalty, floor
    )
    return NegativeSoftwareEffect(
        software_ids,
        applied,
        penalty,
        leader_total,
        total,
        separation_before,
        separation,
        (scores[0].group_id,),
    )


def cluster_negative(
    candidate: dict[str, Any],
    discrimination: Any,
    kb: Any,
    ids: Iterable[str],
    weight: float,
) -> NegativeSoftwareEffect:
    """The named cluster's merged mass and separation after the "not found" answers any member
    documents. The runner-up's mass is recovered from the recorded cluster separation (the
    cluster bars read the same recorded values)."""
    members = tuple(str(gid) for gid in candidate.get("members") or ())
    software_ids = tuple(sorted(set(ids or ())))
    total = float(candidate.get("discriminative", 0.0) or 0.0)
    separation_before = float(candidate.get("separation", 0.0) or 0.0)
    others = [
        item.group_id
        for item in getattr(discrimination, "scores", ()) or ()
        if item.group_id not in members
    ]
    units: list[tuple[str, ...]] = [members, *((gid,) for gid in others)]
    penalty, applied = negative_software_penalty(kb, units, software_ids, weight=weight)
    if penalty <= 0:
        return NegativeSoftwareEffect(
            software_ids, (), 0.0, total, total, separation_before, separation_before, members
        )
    floor = float(getattr(discrimination, "floor", 1.0))
    runner_total = max(0.0, total - max(0.0, separation_before) * max(total, floor))
    reduced, separation = penalized_separation(total, runner_total, penalty, floor)
    return NegativeSoftwareEffect(
        software_ids, applied, penalty, total, reduced, separation_before, separation, members
    )


@dataclass(frozen=True)
class SoftwareLead:
    software_id: str
    name: str
    group_id: str
    target_rank: int
    users: tuple[str, ...]
    delta_found: float
    delta_not_found: float
    technique_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "software_id": self.software_id,
            "name": self.name,
            "group_id": self.group_id,
            "target_rank": self.target_rank,
            "users": list(self.users),
            "delta_found": self.delta_found,
            "delta_not_found": self.delta_not_found,
            "technique_count": self.technique_count,
        }


def rank_software_leads(
    kb: Any,
    discrimination: Any,
    context: Any,
    pool: list[PoolItem],
    *,
    per_group: int = 1,
    decisive_boost: float = 0.0,
    negative_weight: float = 0.3,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    behavior_threshold: float = 0.55,
    asked: dict[str, int] | None = None,
) -> list[SoftwareLead]:
    """Pool items ranked by how far a "found" would move the gate's separation (|delta|), then by
    how many ATT&CK techniques the malware documents (more to inspect), then by id; at most
    ``per_group`` per top-three candidate over the whole interview (``asked``: software questions
    already asked per target group). ``delta_not_found`` is the leader-only rule above."""
    if discrimination is None or not getattr(discrimination, "scores", ()) or not pool:
        return []
    options = {"tool_weight_cap": tool_weight_cap, "behavior_threshold": behavior_threshold}
    base = replay_window(kb, discrimination, context, **options)
    current = set(getattr(context, "software_ids", set()) or set())
    software = kb_mapping(kb, "software")
    software_techniques = kb_mapping(kb, "software_techniques")
    leads: list[SoftwareLead] = []
    for item in pool:
        found = replay_window(
            kb,
            discrimination,
            context,
            software_ids=current | {item.software_id},
            decisive_boost=decisive_boost,
            **options,
        )
        missing = leader_only_negative(base, kb, {item.software_id}, negative_weight)
        leads.append(
            SoftwareLead(
                software_id=item.software_id,
                name=str(getattr(software.get(item.software_id), "name", item.software_id)),
                group_id=item.group_id,
                target_rank=item.target_rank,
                users=item.users,
                delta_found=round(found.separation - base.separation, 4),
                delta_not_found=round(missing.separation - missing.separation_before, 4),
                technique_count=len(set(software_techniques.get(item.software_id, ()) or ())),
            )
        )
    leads.sort(key=lambda lead: (-abs(lead.delta_found), -lead.technique_count, lead.software_id))
    picked: list[SoftwareLead] = []
    per: Counter[str] = Counter(asked or {})
    for lead in leads:
        if per[lead.group_id] >= max(1, int(per_group)):
            continue
        per[lead.group_id] += 1
        picked.append(lead)
    return picked


def software_question_text(kb: Any, software_id: str) -> str:
    entry = kb_mapping(kb, "software").get(software_id)
    name = str(getattr(entry, "name", "") or software_id)
    return f"Did you find {name} ({software_id}) or its known artefacts?"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def software_investigation_steps(
    kb: Any, software_id: str, platforms: Iterable[str] = ()
) -> tuple[str, ...]:
    """What to inspect: the malware's names (up to three aliases), up to three techniques ATT&CK
    documents for it (the most specific first) and where to look. Nothing that names a group."""
    pattern = group_name_pattern(kb)
    entry = kb_mapping(kb, "software").get(software_id)
    name = str(getattr(entry, "name", "") or software_id)
    aliases = [
        str(alias).strip()
        for alias in (getattr(entry, "aliases", ()) or ())
        if str(alias).strip()
        and str(alias).strip().casefold() != name.casefold()
        and not _names_a_group(str(alias), pattern)
    ]
    names = [name, *list(dict.fromkeys(aliases))[:MAX_ALIASES]]
    steps = [
        f"Names to search for: {', '.join(names)}: detections, file and service names, hashes, "
        "YARA or antivirus signatures and any recovered samples."
    ]
    techniques = kb_mapping(kb, "techniques")
    counts = technique_document_counts(kb)
    documented = sorted(
        set(kb_mapping(kb, "software_techniques").get(software_id, ()) or ()),
        key=lambda tid: (counts.get(tid, 0), tid),
    )
    shown = []
    for technique_id in documented:
        technique_name = str(getattr(techniques.get(technique_id), "name", "") or "")
        label = f"{technique_name} ({technique_id})" if technique_name else technique_id
        if _names_a_group(label, pattern):
            continue
        shown.append(label)
        if len(shown) >= MAX_TECHNIQUES:
            break
    if shown:
        steps.append(f"Behaviour ATT&CK documents for this malware: {'; '.join(shown)}.")
    target = [str(item) for item in platforms if str(item).strip()]
    software_platforms = [
        str(item) for item in (getattr(entry, "platforms", ()) or ()) if str(item).strip()
    ]
    if target:
        steps.append(f"Scope the search to {_join(target)} hosts (the case's target OS).")
    elif software_platforms:
        steps.append(f"ATT&CK lists this malware for {_join(software_platforms)}.")
    return tuple(steps)


def software_selected_question(
    kb: Any, lead: SoftwareLead, *, platforms: Iterable[str] = (), window_size: int = 0
) -> SelectedQuestion:
    """The fixed-wording question for one lead (``technique_id`` is empty; the target group is
    kept for the per-group cap and the audit, never shown in the wording)."""
    return SelectedQuestion(
        technique_id="",
        question=software_question_text(kb, lead.software_id),
        information_gain=0.0,
        probability_yes=0.0,
        profile_support=0.0,
        supporting_group_count=len(lead.users),
        candidate_group_count=int(window_size),
        investigation_priority=abs(float(lead.delta_found)),
        hypothesis=SOFTWARE_HYPOTHESIS,
        investigation_steps=software_investigation_steps(kb, lead.software_id, platforms),
        kind="software",
        software_id=lead.software_id,
        target_rank=lead.target_rank,
        target_group_id=lead.group_id,
        delta_found=lead.delta_found,
        asked_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def is_named(decision: Any) -> bool:
    stage = getattr(getattr(decision, "stage", None), "value", None)
    return stage in NAMED_STAGES
