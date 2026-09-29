import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CanvasNode:
    id: str
    label: str
    kind: str
    score: float = 0.0
    details: str = ""


@dataclass(frozen=True)
class CanvasEdge:
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    details: str = ""


@dataclass(frozen=True)
class AnalystCanvas:
    nodes: tuple[CanvasNode, ...]
    edges: tuple[CanvasEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }


def build_analyst_canvas(
    kb: Any,
    state: Any,
    *,
    top_groups: int = 8,
    max_evidence: int = 12,
    max_techniques: int = 15,
) -> AnalystCanvas:
    nodes: dict[str, CanvasNode] = {}
    edges: dict[tuple[str, str, str], CanvasEdge] = {}

    def add_node(node: CanvasNode) -> None:
        existing = nodes.get(node.id)
        if existing is None or node.score > existing.score:
            nodes[node.id] = node

    def add_edge(edge: CanvasEdge) -> None:
        key = (edge.source, edge.target, edge.relationship)
        existing = edges.get(key)
        if existing is None or edge.weight > existing.weight:
            edges[key] = edge

    groups = getattr(kb, "groups", {})
    techniques = getattr(kb, "techniques", {})
    software = getattr(kb, "software", {})
    campaigns = getattr(kb, "campaigns", {})
    group_techniques = getattr(kb, "group_techniques", {})
    group_software = getattr(kb, "group_software", {})
    top_entries = tuple(state.fusion.entries[:top_groups])
    top_group_ids = {entry.group_id for entry in top_entries}

    observation_scores = {observation.id: 0.0 for observation in state.ledger.observations}
    for match in getattr(state.retrieval, "matches", ()):
        for observation_id in match.observation_ids:
            observation_scores[observation_id] = max(
                observation_scores.get(observation_id, 0.0), float(match.score)
            )
    for match in getattr(state.retrieval, "specific_behavior_matches", ()):
        observation_scores[match.observation_id] = max(
            observation_scores.get(match.observation_id, 0.0), float(match.score)
        )
    selected_observations = sorted(
        state.ledger.observations,
        key=lambda item: (-observation_scores.get(item.id, 0.0), item.id),
    )[:max_evidence]
    selected_observation_ids = {observation.id for observation in selected_observations}
    for observation in selected_observations:
        add_node(
            CanvasNode(
                id=f"evidence:{observation.id}",
                label=_shorten(observation.text, 72),
                kind="evidence",
                score=observation_scores.get(observation.id, 0.0),
                details=(
                    f"Source: {observation.provenance.source}; kind: {observation.kind}; "
                    f"occurrences: {observation.occurrence_count}"
                ),
            )
        )

    observed = dict(getattr(state.retrieval, "observed_techniques", {}))
    connected_techniques = {
        technique_id
        for group_id in top_group_ids
        for technique_id in group_techniques.get(group_id, ())
        if technique_id in observed
    }
    technique_pool = connected_techniques or set(observed)
    selected_techniques = {
        technique_id
        for technique_id, _ in sorted(
            ((technique_id, observed[technique_id]) for technique_id in technique_pool),
            key=lambda item: (-item[1], item[0]),
        )[:max_techniques]
    }
    for technique_id in selected_techniques:
        technique = techniques.get(technique_id)
        name = str(_record_value(technique, "name", technique_id))
        specificity_note = ""
        try:
            from apt_classifier.evidence.quality import compute_technique_specificity

            spec = compute_technique_specificity(technique_id, kb)
            if spec.is_group_specific:
                specificity_note = " [group-specific in loaded sources]"
            elif spec.specificity < 0.3:
                specificity_note = " [widely shared — low discriminative value]"
        except Exception:
            pass
        add_node(
            CanvasNode(
                id=f"technique:{technique_id}",
                label=f"{name} ({technique_id})",
                kind="technique",
                score=float(observed.get(technique_id, 0.0)),
                details=f"Observed ATT&CK technique candidate{specificity_note}",
            )
        )
    for match in getattr(state.retrieval, "matches", ()):
        if match.technique_id not in selected_techniques:
            continue
        for observation_id in match.observation_ids:
            if observation_id not in selected_observation_ids:
                continue
            add_edge(
                CanvasEdge(
                    source=f"evidence:{observation_id}",
                    target=f"technique:{match.technique_id}",
                    relationship="maps_to_technique",
                    weight=float(match.score),
                    details=str(match.matched_text or ""),
                )
            )

    for entry in top_entries:
        group = groups.get(entry.group_id)
        add_node(
            CanvasNode(
                id=f"group:{entry.group_id}",
                label=f"#{entry.rank} {_record_value(group, 'name', entry.group_id)}",
                kind="group",
                score=float(entry.relative_score) / 100.0,
                details=(
                    f"Relative ensemble index: {entry.relative_score:.2f}; "
                    f"classifiers: {len(entry.classifier_ranks)}; contested: {entry.contested}"
                ),
            )
        )
        for technique_id in selected_techniques & set(group_techniques.get(entry.group_id, ())):
            add_edge(
                CanvasEdge(
                    source=f"technique:{technique_id}",
                    target=f"group:{entry.group_id}",
                    relationship="documented_technique",
                    weight=float(observed.get(technique_id, 0.0)),
                    details="Technique is documented for this group in ATT&CK",
                )
            )

    for match in getattr(state.retrieval, "specific_behavior_matches", ()):
        if (
            match.group_id not in top_group_ids
            or match.observation_id not in selected_observation_ids
        ):
            continue
        add_edge(
            CanvasEdge(
                source=f"evidence:{match.observation_id}",
                target=f"group:{match.group_id}",
                relationship="specific_behavior",
                weight=float(match.score),
                details=match.matched_text,
            )
        )

    software_ids = set(getattr(state.context, "software_ids", ()))
    software_ids.update(
        str(observation.metadata["software_id"])
        for observation in selected_observations
        if observation.metadata.get("software_id")
    )
    for software_id in sorted(software_ids):
        item = software.get(software_id)
        add_node(
            CanvasNode(
                id=f"tool:{software_id}",
                label=f"{_record_value(item, 'name', software_id)} ({software_id})",
                kind="tool",
                score=1.0,
                details=str(_record_value(item, "description", "Named software/tool evidence")),
            )
        )
        for observation in selected_observations:
            if str(observation.metadata.get("software_id") or "") == software_id:
                add_edge(
                    CanvasEdge(
                        source=f"evidence:{observation.id}",
                        target=f"tool:{software_id}",
                        relationship="identified_tool",
                        details="Named tool extracted from evidence",
                    )
                )
        for group_id in sorted(top_group_ids):
            if software_id in set(group_software.get(group_id, ())):
                add_edge(
                    CanvasEdge(
                        source=f"tool:{software_id}",
                        target=f"group:{group_id}",
                        relationship="documented_tool",
                        details="ATT&CK software-to-group relationship",
                    )
                )

    campaign_result = _classifier_result(state, "campaign_profile")
    if campaign_result is not None:
        for entry in campaign_result.entries:
            if entry.group_id not in top_group_ids:
                continue
            for campaign_id in entry.signal_ids:
                campaign = campaigns.get(campaign_id)
                if campaign is None:
                    continue
                add_node(
                    CanvasNode(
                        id=f"campaign:{campaign_id}",
                        label=f"{_record_value(campaign, 'name', campaign_id)} ({campaign_id})",
                        kind="campaign",
                        score=float(entry.score),
                        details=str(_record_value(campaign, "description", "")),
                    )
                )
                add_edge(
                    CanvasEdge(
                        source=f"campaign:{campaign_id}",
                        target=f"group:{entry.group_id}",
                        relationship="campaign_link",
                        weight=float(entry.score),
                        details="Positive campaign technique overlap",
                    )
                )

    strategic_result = _classifier_result(state, "strategic_context")
    matched_context_ids: set[str] = set()
    matched_context_fields: set[str] = set()
    if strategic_result is not None:
        for entry in strategic_result.entries:
            if entry.group_id not in top_group_ids:
                continue
            for signal in entry.signal_ids:
                field_name, separator, value = signal.partition(":")
                if not separator:
                    continue
                node_id = f"context:{field_name}:{_slug(value)}"
                matched_context_ids.add(node_id)
                matched_context_fields.add(field_name)
                add_node(
                    CanvasNode(
                        id=node_id,
                        label=f"{_factor_label(field_name)}: {value}",
                        kind="context",
                        score=float(entry.score),
                        details="Strategic context matched to public group reporting",
                    )
                )
                add_edge(
                    CanvasEdge(
                        source=node_id,
                        target=f"group:{entry.group_id}",
                        relationship="context_match",
                        weight=float(entry.score),
                        details=signal,
                    )
                )
    strategic_context = getattr(state.ledger, "strategic_context", None)
    if strategic_context is not None:
        for field_name, values in strategic_context.field_values().items():
            if field_name in matched_context_fields:
                continue
            for value in values:
                node_id = f"context:{field_name}:{_slug(value)}"
                if node_id not in matched_context_ids:
                    add_node(
                        CanvasNode(
                            id=node_id,
                            label=f"{_factor_label(field_name)}: {value}",
                            kind="context",
                            details="Supplied context with no documented match among shown groups",
                        )
                    )

    for hypothesis in state.origin_assessment.origin.hypotheses:
        country_id = f"country:{_slug(hypothesis.country)}"
        add_node(
            CanvasNode(
                id=country_id,
                label=hypothesis.country,
                kind="country",
                score=float(hypothesis.relative_support) / 100.0,
                details=f"Relative documented origin support: {hypothesis.relative_support:.2f}",
            )
        )
        for item in hypothesis.evidence:
            if item.group_id in top_group_ids:
                add_edge(
                    CanvasEdge(
                        source=f"group:{item.group_id}",
                        target=country_id,
                        relationship="documented_origin",
                        weight=float(item.contribution),
                        details=item.excerpt,
                    )
                )

    threat_type = state.origin_assessment.threat_type
    assessment_id = "assessment:threat-type"
    add_node(
        CanvasNode(
            id=assessment_id,
            label=threat_type.label,
            kind="assessment",
            score=max(threat_type.apt_like_support, threat_type.criminal_support) / 100.0,
            details=(
                f"APT-like: {threat_type.apt_like_support:.1f}; criminal: "
                f"{threat_type.criminal_support:.1f}; unresolved: "
                f"{threat_type.unresolved_support:.1f}"
            ),
        )
    )
    for group_id in threat_type.apt_supporting_group_ids:
        if group_id in top_group_ids:
            add_edge(
                CanvasEdge(
                    source=f"group:{group_id}",
                    target=assessment_id,
                    relationship="supports_apt_like",
                    details="Public profile contains state/espionage indicators",
                )
            )
    for group_id in threat_type.criminal_supporting_group_ids:
        if group_id in top_group_ids:
            add_edge(
                CanvasEdge(
                    source=f"group:{group_id}",
                    target=assessment_id,
                    relationship="supports_criminal",
                    details="Public profile contains financially motivated criminal indicators",
                )
            )

    snapshot = getattr(state, "threat_intel_snapshot", None)
    intel_items = snapshot.item_by_id() if snapshot is not None else {}
    for match in getattr(state, "threat_intel_matches", ()):
        item = intel_items.get(match.item_id)
        if item is None:
            continue
        node_id = f"live-intel:{item.id}"
        add_node(
            CanvasNode(
                id=node_id,
                label=f"{item.source}: {item.title}",
                kind="live_intel",
                score=min(float(match.score) / 8.0, 1.0),
                details=(
                    f"Published: {item.published_at}; verification: {item.verification}; "
                    f"match reasons: {', '.join(match.reasons)}; summary: {item.summary[:500]}"
                ),
            )
        )
        for group_id in match.group_ids:
            if group_id in top_group_ids:
                add_edge(
                    CanvasEdge(
                        source=node_id,
                        target=f"group:{group_id}",
                        relationship="live_actor_reference",
                        weight=min(float(match.score) / 8.0, 1.0),
                        details="; ".join(match.reasons),
                    )
                )
        for observation_id in match.observation_ids:
            if observation_id in selected_observation_ids:
                add_edge(
                    CanvasEdge(
                        source=f"evidence:{observation_id}",
                        target=node_id,
                        relationship="live_indicator_overlap",
                        weight=min(float(match.score) / 8.0, 1.0),
                        details="; ".join(match.reasons),
                    )
                )

    return AnalystCanvas(
        nodes=tuple(sorted(nodes.values(), key=lambda item: (item.kind, item.label, item.id))),
        edges=tuple(
            sorted(edges.values(), key=lambda item: (item.relationship, item.source, item.target))
        ),
    )


def build_candidate_dossier(kb: Any, state: Any, group_id: str) -> dict[str, Any]:
    fusion_entry = state.fusion.by_group().get(group_id)
    if fusion_entry is None:
        raise ValueError(f"Unknown fused candidate: {group_id}")
    group = getattr(kb, "groups", {}).get(group_id)
    known = set(getattr(kb, "group_techniques", {}).get(group_id, ()))
    observed = dict(getattr(state.retrieval, "observed_techniques", {}))
    techniques = getattr(kb, "techniques", {})

    def technique_row(technique_id: str) -> dict[str, Any]:
        technique = techniques.get(technique_id)
        return {
            "technique_id": technique_id,
            "name": str(_record_value(technique, "name", technique_id)),
            "confidence": float(observed.get(technique_id, 0.0)),
            "observation_ids": tuple(
                getattr(state.retrieval, "technique_observations", {}).get(technique_id, ())
            ),
        }

    matched = sorted(set(observed) & known, key=lambda item: (-observed[item], item))
    unmatched = sorted(set(observed) - known, key=lambda item: (-observed[item], item))
    software_ids = set(getattr(state.context, "software_ids", ())) & set(
        getattr(kb, "group_software", {}).get(group_id, ())
    )
    software = getattr(kb, "software", {})
    classifier_signals = {
        result.classifier: tuple(entry.signal_ids)
        for result in state.classifier_results
        if (entry := result.by_group().get(group_id)) is not None and entry.signal_ids
    }
    country_evidence = [
        {
            "country": hypothesis.country,
            "excerpt": item.excerpt,
            "relative_support": hypothesis.relative_support,
        }
        for hypothesis in state.origin_assessment.origin.hypotheses
        for item in hypothesis.evidence
        if item.group_id == group_id
    ]
    behavior_matches = [
        {
            "observation_id": match.observation_id,
            "technique_id": match.technique_id,
            "score": match.score,
            "matched_text": match.matched_text,
        }
        for match in getattr(state.retrieval, "specific_behavior_matches", ())
        if match.group_id == group_id
    ]
    return {
        "group_id": group_id,
        "name": str(_record_value(group, "name", group_id)),
        "description": str(_record_value(group, "description", "")),
        "rank": fusion_entry.rank,
        "relative_score": fusion_entry.relative_score,
        "contested": fusion_entry.contested,
        "classifier_ranks": dict(fusion_entry.classifier_ranks),
        "matched_techniques": [technique_row(item) for item in matched],
        "unmatched_techniques": [technique_row(item) for item in unmatched],
        "contradicting_techniques": sorted(
            set(getattr(state.context, "negative_techniques", ())) & known
        ),
        "specific_behaviors": behavior_matches,
        "tools": [
            {
                "software_id": software_id,
                "name": str(_record_value(software.get(software_id), "name", software_id)),
            }
            for software_id in sorted(software_ids)
        ],
        "classifier_signals": classifier_signals,
        "country_evidence": country_evidence,
    }


def _classifier_result(state: Any, name: str) -> Any:
    return next(
        (
            result
            for result in state.classifier_results
            if result.classifier == name and result.applicable
        ),
        None,
    )


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _shorten(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _factor_label(field_name: str) -> str:
    return {
        "target_geographies": "Target geography",
        "target_industries": "Industry",
        "motivations": "Motivation",
        "objectives": "Objective",
        "suspected_origins": "Analyst origin prior",
    }.get(field_name, field_name.replace("_", " ").title())
