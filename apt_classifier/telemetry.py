"""ATT&CK Analytic/log-source matching and observability decisions.

This module bridges the parsed ATT&CK Detection Strategies, Analytics, and
Data Components to produce telemetry-aware collection guidance for the
interview flow. It distinguishes three evidence-visibility states:

- ``available``: the telemetry source is confirmed present for the case.
- ``unavailable``: the telemetry source is confirmed absent.
- ``unknown``: the analyst has not yet confirmed whether the source exists.

A negative interview answer is treated as evidence only when the relevant
telemetry is ``available``. Missing or unknown telemetry becomes
``NOT_VISIBLE``/``UNKNOWN`` and cannot penalize a group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from apt_classifier.attack.kb import Analytic, AttackKnowledgeBase, LogSource


class TelemetryCoverage(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    PARTIAL = "partial"


@dataclass(slots=True)
class TelemetrySource:
    """A single log source channel with its coverage state."""

    name: str
    channel: str = ""
    coverage: TelemetryCoverage = TelemetryCoverage.UNKNOWN
    data_component_id: str = ""

    @property
    def label(self) -> str:
        # ATT&CK writes the channel "None" for a log source without a specific channel
        if self.channel and self.channel.casefold() != "none":
            return f"{self.name} ({self.channel})"
        return self.name


@dataclass(slots=True)
class CollectionStep:
    """A recommended collection step derived from an ATT&CK Analytic."""

    technique_id: str
    analytic_id: str
    description: str
    log_sources: list[TelemetrySource] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    mutable_elements: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class TelemetryProfile:
    """Per-case telemetry coverage state for a set of log sources."""

    sources: dict[str, TelemetrySource] = field(default_factory=dict)
    validity_window: tuple[str, str] | None = None

    def set_coverage(self, name: str, channel: str, coverage: TelemetryCoverage) -> None:
        key = self._key(name, channel)
        existing = self.sources.get(key)
        if existing:
            existing.coverage = coverage
        else:
            self.sources[key] = TelemetrySource(name=name, channel=channel, coverage=coverage)

    def get_coverage(self, name: str, channel: str) -> TelemetryCoverage:
        source = self.sources.get(self._key(name, channel))
        return source.coverage if source else TelemetryCoverage.UNKNOWN

    def is_available(self, name: str, channel: str = "") -> bool:
        return self.get_coverage(name, channel) is TelemetryCoverage.AVAILABLE

    @staticmethod
    def _key(name: str, channel: str) -> str:
        return f"{name}|{channel}"


def technique_collection_steps(
    kb: AttackKnowledgeBase,
    technique_id: str,
    profile: TelemetryProfile | None = None,
) -> list[CollectionStep]:
    """Generate collection steps for a technique from its ATT&CK Analytics.

    Falls back to an empty list if the technique has no detection strategies
    or analytics in the loaded KB. The caller should fall back to hand-authored
    guidance when this returns empty.
    """
    ds_ids = kb.technique_detection_strategies.get(technique_id, [])
    if not ds_ids:
        return []
    steps: list[CollectionStep] = []
    seen_analytics: set[str] = set()
    for ds_id in ds_ids:
        ds = kb.detection_strategies.get(ds_id)
        if ds is None:
            continue
        for an_id in ds.analytic_ids:
            if an_id in seen_analytics:
                continue
            seen_analytics.add(an_id)
            analytic = kb.analytics.get(an_id)
            if analytic is None:
                continue
            steps.append(_build_collection_step(technique_id, analytic, kb, profile))
    return steps


def _build_collection_step(
    technique_id: str,
    analytic: Analytic,
    kb: AttackKnowledgeBase,
    profile: TelemetryProfile | None,
) -> CollectionStep:
    log_sources: list[TelemetrySource] = []
    for ref in analytic.log_source_refs:
        coverage = TelemetryCoverage.UNKNOWN
        if profile is not None:
            coverage = profile.get_coverage(ref.name, ref.channel)
        dc_id = ""
        # Try to find the data component ID for this log source ref
        for dc in analytic.data_component_ids:
            dc_obj = kb.data_components.get(dc)
            if dc_obj is None:
                continue
            for ls in dc_obj.log_sources:
                if ls.name == ref.name and ls.channel == ref.channel:
                    dc_id = dc
                    break
            if dc_id:
                break
        log_sources.append(
            TelemetrySource(
                name=ref.name,
                channel=ref.channel,
                coverage=coverage,
                data_component_id=dc_id,
            )
        )
    return CollectionStep(
        technique_id=technique_id,
        analytic_id=analytic.id,
        description=analytic.description or analytic.name,
        log_sources=log_sources,
        platforms=list(analytic.platforms),
        mutable_elements=list(analytic.mutable_elements),
    )


def negative_answer_is_valid(
    technique_id: str,
    kb: AttackKnowledgeBase,
    profile: TelemetryProfile | None,
) -> bool:
    """Check whether a negative answer has compatible, available telemetry.

    A negative answer (checked and absent) is valid only if at least one
    log source for the technique's analytics is marked ``available`` in the
    telemetry profile. If no telemetry is available, the negative answer
    becomes ``NOT_VISIBLE`` and cannot penalize a group.
    """
    if profile is None:
        return False
    steps = technique_collection_steps(kb, technique_id, profile)
    for step in steps:
        for source in step.log_sources:
            if source.coverage is TelemetryCoverage.AVAILABLE:
                return True
    return False


def all_log_sources_for_technique(kb: AttackKnowledgeBase, technique_id: str) -> list[LogSource]:
    """Return all distinct log sources referenced by a technique's analytics."""
    steps = technique_collection_steps(kb, technique_id)
    seen: set[tuple[str, str]] = set()
    result: list[LogSource] = []
    for step in steps:
        for source in step.log_sources:
            key = (source.name, source.channel)
            if key in seen:
                continue
            seen.add(key)
            result.append(LogSource(name=source.name, channel=source.channel))
    return result
