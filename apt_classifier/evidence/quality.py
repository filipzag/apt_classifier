"""Evidence quality, corroboration, and documented specificity.

This module provides Admiralty-style source reliability (A–F) and information
credibility (1–6) ratings, claim-level corroboration summaries that prevent
duplicated/syndicated reporting from counting as independent support, and
documented-specificity scoring based on frequency in loaded sources.

Key design decisions:
- Reliability and credibility are separate, fully auditable values. They are
  never silently converted into probabilities.
- Corroboration requires distinct independence keys. Repeated copies of the
  same report do not count as independent support.
- Documented specificity measures how distinctive a signal is in the loaded
  sources, not how unique it claims to be. A rare signal may only appear rare
  because reporting is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceReliability(StrEnum):
    """Admiralty-style source reliability rating (A–F)."""

    A = "A"  # Completely reliable
    B = "B"  # Usually reliable
    C = "C"  # Fairly reliable
    D = "D"  # Not usually reliable
    E = "E"  # Unreliable
    F = "F"  # Reliability cannot be judged
    UNRATED = "unrated"


class InformationCredibility(StrEnum):
    """Admiralty-style information credibility rating (1–6)."""

    ONE = "1"  # Confirmed by other sources
    TWO = "2"  # Probably true
    THREE = "3"  # Possibly true
    FOUR = "4"  # Doubtful
    FIVE = "5"  # Improbable
    SIX = "6"  # Truth cannot be judged
    UNRATED = "unrated"


# Default reliability/credibility suggestions by source type.
# These are transparent defaults that analysts can override.
_DEFAULT_RELIABILITY: dict[str, SourceReliability] = {
    "edr": SourceReliability.A,
    "siem": SourceReliability.A,
    "syslog": SourceReliability.B,
    "free_text": SourceReliability.C,
    "question": SourceReliability.B,
    "threat_intel": SourceReliability.C,
    "log": SourceReliability.B,
    "ioc": SourceReliability.C,
    "software": SourceReliability.B,
    "narrative": SourceReliability.C,
    "strategic_context": SourceReliability.C,
    "unknown": SourceReliability.UNRATED,
}

_DEFAULT_CREDIBILITY: dict[str, InformationCredibility] = {
    "edr": InformationCredibility.ONE,
    "siem": InformationCredibility.ONE,
    "syslog": InformationCredibility.TWO,
    "free_text": InformationCredibility.THREE,
    "question": InformationCredibility.TWO,
    "threat_intel": InformationCredibility.THREE,
    "log": InformationCredibility.TWO,
    "ioc": InformationCredibility.THREE,
    "software": InformationCredibility.TWO,
    "narrative": InformationCredibility.THREE,
    "strategic_context": InformationCredibility.THREE,
    "unknown": InformationCredibility.UNRATED,
}


def default_reliability(source_type: str) -> SourceReliability:
    return _DEFAULT_RELIABILITY.get(
        str(source_type or "unknown").lower().replace(" ", "_"),
        SourceReliability.UNRATED,
    )


def default_credibility(source_type: str) -> InformationCredibility:
    return _DEFAULT_CREDIBILITY.get(
        str(source_type or "unknown").lower().replace(" ", "_"),
        InformationCredibility.UNRATED,
    )


@dataclass(slots=True)
class EvidenceQuality:
    """Quality ratings for a single evidence provenance entry."""

    reliability: SourceReliability = SourceReliability.UNRATED
    credibility: InformationCredibility = InformationCredibility.UNRATED
    rating_rationale: str = ""
    independence_key: str = ""
    observed_time: str | None = None
    collection_window: tuple[str, str] | None = None

    @property
    def is_rated(self) -> bool:
        return (
            self.reliability is not SourceReliability.UNRATED
            and self.credibility is not InformationCredibility.UNRATED
        )

    @property
    def numeric_reliability(self) -> int:
        """Convert reliability to a 1–6 scale (A=6, F=1, unrated=0)."""
        mapping = {
            SourceReliability.A: 6,
            SourceReliability.B: 5,
            SourceReliability.C: 4,
            SourceReliability.D: 3,
            SourceReliability.E: 2,
            SourceReliability.F: 1,
        }
        return mapping.get(self.reliability, 0)

    @property
    def numeric_credibility(self) -> int:
        """Convert credibility to a 1–6 scale (1=6, 6=1, unrated=0)."""
        mapping = {
            InformationCredibility.ONE: 6,
            InformationCredibility.TWO: 5,
            InformationCredibility.THREE: 4,
            InformationCredibility.FOUR: 3,
            InformationCredibility.FIVE: 2,
            InformationCredibility.SIX: 1,
        }
        return mapping.get(self.credibility, 0)


@dataclass(slots=True)
class CorroborationSummary:
    """Claim-level corroboration summary for a technique or signal."""

    technique_id: str
    independent_source_count: int = 0
    total_source_count: int = 0
    independence_keys: list[str] = field(default_factory=list)
    quality_scores: list[EvidenceQuality] = field(default_factory=list)
    is_corroborated: bool = False
    corroboration_note: str = ""

    @property
    def effective_quality(self) -> float:
        """Weighted average quality of independent sources only.

        Returns 0.0 if no independent sources are rated.
        """
        independent_qualities = [q for q in self.quality_scores if q.independence_key]
        if not independent_qualities:
            return 0.0
        # Deduplicate by independence key, keeping the highest-quality entry
        by_key: dict[str, EvidenceQuality] = {}
        for q in independent_qualities:
            existing = by_key.get(q.independence_key)
            if existing is None or (
                q.numeric_reliability + q.numeric_credibility
                > existing.numeric_reliability + existing.numeric_credibility
            ):
                by_key[q.independence_key] = q
        total = 0.0
        count = 0
        for q in by_key.values():
            if q.is_rated:
                total += (q.numeric_reliability + q.numeric_credibility) / 12.0
                count += 1
        return total / count if count > 0 else 0.0


def build_corroboration_summary(
    technique_id: str,
    provenance_entries: list[tuple[Any, EvidenceQuality]],
) -> CorroborationSummary:
    """Build a corroboration summary from provenance entries.

    ``provenance_entries`` is a list of (provenance, quality) tuples.
    Corroboration requires distinct independence keys. Entries without an
    independence key are counted but do not contribute to independent
    corroboration.
    """
    by_key: dict[str, list[tuple[Any, EvidenceQuality]]] = {}
    no_key: list[tuple[Any, EvidenceQuality]] = []
    for provenance, quality in provenance_entries:
        key = quality.independence_key or ""
        if key:
            by_key.setdefault(key, []).append((provenance, quality))
        else:
            no_key.append((provenance, quality))
    independent_keys = sorted(by_key.keys())
    independent_count = len(independent_keys)
    total_count = independent_count + len(no_key)
    quality_scores: list[EvidenceQuality] = []
    for key in independent_keys:
        # Keep the highest-quality entry per independence key
        best = max(
            by_key[key],
            key=lambda item: item[1].numeric_reliability + item[1].numeric_credibility,
        )
        quality_scores.append(best[1])
    for _, quality in no_key:
        quality_scores.append(quality)
    is_corroborated = independent_count >= 2
    note = ""
    if is_corroborated:
        note = f"Corroborated by {independent_count} independent sources."
    elif independent_count == 1:
        note = "Single independent source; not independently corroborated."
    else:
        note = "No independent sources; claims are uncorroborated."
    return CorroborationSummary(
        technique_id=technique_id,
        independent_source_count=independent_count,
        total_source_count=total_count,
        independence_keys=independent_keys,
        quality_scores=quality_scores,
        is_corroborated=is_corroborated,
        corroboration_note=note,
    )


@dataclass(slots=True)
class SpecificityScore:
    """Documented specificity score for a signal.

    Specificity measures how distinctive a signal is in the loaded sources.
    A high specificity score means the signal appears in few groups/techniques,
    making it more discriminative. A low score means it is widely shared.

    Note: specificity is "documented" — a rare signal may only appear rare
    because reporting is incomplete. The score does not claim actor-exclusivity.
    """

    signal_type: str  # "technique", "software", "procedure", "infrastructure"
    signal_id: str
    group_count: int = 0
    total_groups: int = 0
    specificity: float = 0.0
    profile_completeness: float = 0.0
    note: str = ""

    @property
    def is_group_specific(self) -> bool:
        """True if the signal appears in only one group."""
        return self.group_count == 1 and self.total_groups > 1


def compute_technique_specificity(
    technique_id: str,
    kb: Any,
) -> SpecificityScore:
    """Compute documented specificity for a technique.

    Specificity = 1 - (group_count / total_groups).
    A technique used by all groups has specificity 0.0.
    A technique used by one group has specificity near 1.0.
    """
    groups = getattr(kb, "groups", {})
    total_groups = len(groups)
    if total_groups == 0:
        return SpecificityScore(
            signal_type="technique",
            signal_id=technique_id,
            note="No groups loaded; specificity undefined.",
        )
    group_count = sum(
        1 for group in groups.values() if technique_id in getattr(group, "technique_ids", [])
    )
    specificity = 1.0 - (group_count / total_groups)
    # Profile completeness: fraction of groups that have any technique data
    groups_with_techniques = sum(1 for g in groups.values() if getattr(g, "technique_ids", []))
    completeness = groups_with_techniques / total_groups if total_groups else 0.0
    note = ""
    if group_count <= 1:
        note = (
            f"Documented in {group_count}/{total_groups} groups. "
            "This signal is rare in the loaded sources, but rarity may reflect "
            "incomplete reporting rather than actor-exclusivity."
        )
    elif group_count > 5 or (total_groups > 0 and group_count / total_groups >= 0.5):
        note = (
            f"Documented in {group_count}/{total_groups} groups. "
            "This is a widely shared technique with low discriminative value."
        )
    else:
        note = f"Documented in {group_count}/{total_groups} groups."
    return SpecificityScore(
        signal_type="technique",
        signal_id=technique_id,
        group_count=group_count,
        total_groups=total_groups,
        specificity=specificity,
        profile_completeness=completeness,
        note=note,
    )


def compute_software_specificity(
    software_id: str,
    kb: Any,
) -> SpecificityScore:
    """Compute documented specificity for a software/tool."""
    groups = getattr(kb, "groups", {})
    total_groups = len(groups)
    if total_groups == 0:
        return SpecificityScore(
            signal_type="software",
            signal_id=software_id,
            note="No groups loaded; specificity undefined.",
        )
    group_count = sum(
        1 for group in groups.values() if software_id in getattr(group, "software_ids", [])
    )
    specificity = 1.0 - (group_count / total_groups)
    return SpecificityScore(
        signal_type="software",
        signal_id=software_id,
        group_count=group_count,
        total_groups=total_groups,
        specificity=specificity,
        note=f"Documented in {group_count}/{total_groups} groups.",
    )
