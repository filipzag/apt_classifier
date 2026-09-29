"""ATT&CK Navigator layer export for observed, checked-absent, and gap techniques.

Exports three distinct layers so analysts can visualize:
- Observed techniques (techniques retrieved or confirmed in the case).
- Checked-absent techniques (negative interview answers with valid telemetry).
- Telemetry gaps (techniques where collection guidance exists but coverage
  is unknown or unavailable).

Navigator layer format reference:
https://github.com/mitre-attack/attack-navigator/blob/master/layers/LAYERFORMATv4_5.md
"""

from __future__ import annotations

from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.telemetry import TelemetryCoverage, TelemetryProfile


def _layer_template(name: str, version: str) -> dict[str, Any]:
    return {
        "versions": {"attack": version, "navigator": "4.9.0", "layer": "4.5"},
        "name": name,
        "domain": "enterprise-attack",
        "description": "",
        "filters": {
            "platforms": [
                "Linux",
                "macOS",
                "Windows",
                "Azure AD",
                "Office 365",
                "SaaS",
                "IaaS",
                "Google Workspace",
                "Network",
                "Containers",
                "PRE",
            ]
        },
        "sorting": 0,
        "layout": {"layout": "side", "aggregateFunction": "average"},
        "hideDisabled": False,
        "techniques": [],
        "gradient": {
            "colors": ["#ff6666", "#ffe766", "#8ec843"],
            "minValue": 0,
            "maxValue": 100,
        },
        "legendItems": [],
        "metadata": [],
        "showTacticRowBackground": False,
        "tacticRowBackground": "#dddddd",
        "selectTechniquesAcrossTactics": True,
        "selectSubtechniquesWithParent": False,
    }


def _technique_entry(
    technique_id: str,
    kb: AttackKnowledgeBase,
    *,
    score: int | None = None,
    comment: str = "",
    color: str = "",
) -> dict[str, Any]:
    technique = kb.techniques.get(technique_id)
    tactic_short_names: list[str] = []
    if technique is not None:
        for tactic in technique.tactics:
            # ATT&CK Navigator uses short names (e.g., "initial-access")
            tactic_short_names.append(tactic.replace(" ", "-").lower())
    entry: dict[str, Any] = {
        "techniqueID": technique_id,
        "tactic": tactic_short_names[0] if tactic_short_names else "",
        "comment": comment,
    }
    if score is not None:
        entry["score"] = score
    if color:
        entry["color"] = color
    return entry


def build_observed_layer(
    kb: AttackKnowledgeBase,
    observed_techniques: dict[str, float],
) -> dict[str, Any]:
    """Build a Navigator layer for observed techniques."""
    layer = _layer_template("Observed Techniques", kb.version)
    layer["description"] = "Techniques retrieved or confirmed in the current case."
    for technique_id, score in sorted(observed_techniques.items()):
        layer["techniques"].append(
            _technique_entry(
                technique_id,
                kb,
                score=int(round(score * 100)),
                comment=f"Observed (confidence: {score:.2f})",
            )
        )
    layer["legendItems"] = [
        {"label": "Observed", "color": "#8ec843"},
    ]
    return layer


def build_checked_absent_layer(
    kb: AttackKnowledgeBase,
    negative_techniques: set[str],
) -> dict[str, Any]:
    """Build a Navigator layer for checked-absent techniques.

    These are techniques where the analyst confirmed the relevant telemetry
    was available and the behavior was absent.
    """
    layer = _layer_template("Checked Absent Techniques", kb.version)
    layer["description"] = "Techniques confirmed absent with compatible, available telemetry."
    for technique_id in sorted(negative_techniques):
        layer["techniques"].append(
            _technique_entry(
                technique_id,
                kb,
                color="#ff6666",
                comment="Checked and absent with available telemetry",
            )
        )
    layer["legendItems"] = [
        {"label": "Checked absent", "color": "#ff6666"},
    ]
    return layer


def build_telemetry_gap_layer(
    kb: AttackKnowledgeBase,
    candidate_technique_ids: list[str],
    profile: TelemetryProfile | None,
) -> dict[str, Any]:
    """Build a Navigator layer for telemetry gaps.

    These are candidate techniques where collection guidance exists but
    telemetry coverage is unknown or unavailable.
    """
    layer = _layer_template("Telemetry Gaps", kb.version)
    layer["description"] = (
        "Techniques with ATT&CK collection guidance but unknown or unavailable telemetry."
    )
    from apt_classifier.telemetry import technique_collection_steps

    for technique_id in sorted(set(candidate_technique_ids)):
        steps = technique_collection_steps(kb, technique_id, profile)
        if not steps:
            continue
        has_gap = False
        gap_sources: list[str] = []
        for step in steps:
            for source in step.log_sources:
                if source.coverage in (TelemetryCoverage.UNKNOWN, TelemetryCoverage.UNAVAILABLE):
                    has_gap = True
                    gap_sources.append(source.label)
        if not has_gap:
            continue
        layer["techniques"].append(
            _technique_entry(
                technique_id,
                kb,
                color="#ffe766",
                comment=f"Telemetry gap: {', '.join(gap_sources[:3])}",
            )
        )
    layer["legendItems"] = [
        {"label": "Telemetry gap", "color": "#ffe766"},
    ]
    return layer


def export_layers(
    kb: AttackKnowledgeBase,
    observed_techniques: dict[str, float],
    negative_techniques: set[str],
    candidate_technique_ids: list[str],
    profile: TelemetryProfile | None = None,
) -> dict[str, dict[str, Any]]:
    """Export all three Navigator layers as a dict keyed by layer name."""
    return {
        "observed": build_observed_layer(kb, observed_techniques),
        "checked_absent": build_checked_absent_layer(kb, negative_techniques),
        "telemetry_gaps": build_telemetry_gap_layer(kb, candidate_technique_ids, profile),
    }
