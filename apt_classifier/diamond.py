from dataclasses import asdict, dataclass
from typing import Any

DIAMOND_VERTICES = ("adversary", "capability", "infrastructure", "victim")


@dataclass(frozen=True)
class DiamondNode:
    id: str
    label: str
    vertex: str
    score: float = 0.0
    status: str = "unknown"
    details: str = ""


@dataclass(frozen=True)
class DiamondEdge:
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    details: str = ""


@dataclass(frozen=True)
class DiamondModel:
    nodes: tuple[DiamondNode, ...]
    edges: tuple[DiamondEdge, ...]
    caveats: tuple[str, ...] = ()

    def nodes_for(self, vertex: str) -> tuple[DiamondNode, ...]:
        return tuple(node for node in self.nodes if node.vertex == vertex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "caveats": list(self.caveats),
        }


def build_diamond_model(
    kb: Any,
    state: Any,
    *,
    top_groups: int = 8,
    max_capabilities: int = 20,
    max_infrastructure: int = 20,
) -> DiamondModel:
    nodes: dict[str, DiamondNode] = {}
    edges: dict[tuple[str, str, str], DiamondEdge] = {}

    def add_node(node: DiamondNode) -> None:
        if node.vertex not in DIAMOND_VERTICES:
            raise ValueError(f"Unknown Diamond Model vertex: {node.vertex}")
        existing = nodes.get(node.id)
        if existing is None or node.score > existing.score:
            nodes[node.id] = node

    def add_edge(edge: DiamondEdge) -> None:
        if edge.source not in nodes or edge.target not in nodes:
            return
        key = (edge.source, edge.target, edge.relationship)
        existing = edges.get(key)
        if existing is None or edge.weight > existing.weight:
            edges[key] = edge

    groups = getattr(kb, "groups", {})
    techniques = getattr(kb, "techniques", {})
    group_techniques = getattr(kb, "group_techniques", {})
    group_software = getattr(kb, "group_software", {})
    software = getattr(kb, "software", {})
    top_entries = tuple(state.fusion.entries[:top_groups])
    top_group_ids = {entry.group_id for entry in top_entries}

    for entry in top_entries:
        group = groups.get(entry.group_id)
        name = _record_value(group, "name", entry.group_id)
        add_node(
            DiamondNode(
                id=f"adversary:{entry.group_id}",
                label=f"#{entry.rank} {name}",
                vertex="adversary",
                score=max(min(float(entry.relative_score) / 100.0, 1.0), 0.0),
                status="hypothesis",
                details=(
                    f"Ranked group hypothesis; relative ensemble index {entry.relative_score:.2f}; "
                    f"contested={entry.contested}. This is not proof of attribution."
                ),
            )
        )

    observed = dict(getattr(state.retrieval, "observed_techniques", {}))
    selected_techniques = sorted(
        observed,
        key=lambda technique_id: (-float(observed[technique_id]), technique_id),
    )[:max_capabilities]
    technique_observations = getattr(state.retrieval, "technique_observations", {})
    for technique_id in selected_techniques:
        technique = techniques.get(technique_id)
        observation_ids = tuple(technique_observations.get(technique_id, ()))
        add_node(
            DiamondNode(
                id=f"capability:technique:{technique_id}",
                label=f"{_record_value(technique, 'name', technique_id)} ({technique_id})",
                vertex="capability",
                score=max(min(float(observed[technique_id]), 1.0), 0.0),
                status="observed",
                details=(
                    f"ATT&CK technique retrieved from intake; observation IDs: "
                    f"{', '.join(observation_ids) or 'not recorded'}"
                ),
            )
        )
        for group_id in sorted(top_group_ids):
            if technique_id in set(group_techniques.get(group_id, ())):
                add_edge(
                    DiamondEdge(
                        source=f"capability:technique:{technique_id}",
                        target=f"adversary:{group_id}",
                        relationship="documented_capability",
                        weight=max(min(float(observed[technique_id]), 1.0), 0.0),
                        details="ATT&CK documents this technique for the candidate group.",
                    )
                )

    software_ids = sorted(getattr(state.context, "software_ids", ()))
    for software_id in software_ids[:max_capabilities]:
        item = software.get(software_id)
        add_node(
            DiamondNode(
                id=f"capability:tool:{software_id}",
                label=f"{_record_value(item, 'name', software_id)} ({software_id})",
                vertex="capability",
                score=1.0,
                status="observed",
                details="Named software/tool extracted from analyst evidence.",
            )
        )
        for group_id in sorted(top_group_ids):
            if software_id in set(group_software.get(group_id, ())):
                add_edge(
                    DiamondEdge(
                        source=f"capability:tool:{software_id}",
                        target=f"adversary:{group_id}",
                        relationship="documented_tool_association",
                        details="ATT&CK documents this software relationship.",
                    )
                )

    infrastructure_observations = [
        observation
        for observation in getattr(state.ledger, "observations", ())
        if observation.kind == "ioc" or observation.metadata.get("indicator_type")
    ]
    for observation in infrastructure_observations[:max_infrastructure]:
        metadata = observation.metadata
        value = str(metadata.get("value") or observation.text)
        indicator_type = str(
            metadata.get("indicator_type") or metadata.get("ioc_type") or "indicator"
        )
        add_node(
            DiamondNode(
                id=f"infrastructure:{observation.id}",
                label=f"{indicator_type}: {_shorten(value, 70)}",
                vertex="infrastructure",
                score=1.0,
                status="observed",
                details=(
                    f"Source: {observation.provenance.source}; source name: "
                    f"{observation.provenance.source_name or 'unspecified'}; observation ID: {observation.id}"
                ),
            )
        )

    snapshot = getattr(state, "threat_intel_snapshot", None)
    intel_items = snapshot.item_by_id() if snapshot is not None else {}
    for match in getattr(state, "threat_intel_matches", ())[:max_infrastructure]:
        item = intel_items.get(match.item_id)
        if item is None:
            continue
        verification = str(item.verification or "unverified")
        add_node(
            DiamondNode(
                id=f"infrastructure:intel:{item.id}",
                label=f"{item.source}: {_shorten(item.title, 65)}",
                vertex="infrastructure",
                score=min(max(float(match.score) / 8.0, 0.0), 1.0),
                status=verification,
                details=(
                    f"Public intelligence record; verification={verification}; "
                    f"reasons: {', '.join(match.reasons)}"
                ),
            )
        )
        for group_id in match.group_ids:
            if group_id in top_group_ids:
                add_edge(
                    DiamondEdge(
                        source=f"infrastructure:intel:{item.id}",
                        target=f"adversary:{group_id}",
                        relationship="intelligence_reference",
                        weight=min(max(float(match.score) / 8.0, 0.0), 1.0),
                        details=(
                            f"{verification} public record matched the candidate; "
                            "not voting attribution evidence."
                        ),
                    )
                )

    strategic_context = getattr(state.ledger, "strategic_context", None)
    victim_values = strategic_context.field_values() if strategic_context is not None else {}
    for field_name in ("target_geographies", "target_industries"):
        for value in victim_values.get(field_name, ()):
            node_id = f"victim:{field_name}:{_slug(value)}"
            add_node(
                DiamondNode(
                    id=node_id,
                    label=f"{_factor_label(field_name)}: {value}",
                    vertex="victim",
                    score=1.0,
                    status="reported",
                    details=(
                        "Analyst-supplied victimology. This is not attacker origin and does not "
                        "prove group identity."
                    ),
                )
            )
            strategic_result = _classifier_result(state, "strategic_context")
            if strategic_result is not None:
                for entry in strategic_result.entries:
                    if entry.group_id not in top_group_ids:
                        continue
                    if f"{field_name}:{value.casefold()}" in {
                        signal.casefold() for signal in entry.signal_ids
                    }:
                        add_edge(
                            DiamondEdge(
                                source=f"adversary:{entry.group_id}",
                                target=node_id,
                                relationship="targeting_context_match",
                                weight=max(min(float(entry.score), 1.0), 0.0),
                                details="Strategic context matched public group reporting.",
                            )
                        )

    if not any(node.vertex == "victim" for node in nodes.values()):
        add_node(
            DiamondNode(
                id="victim:unspecified",
                label="Victim / affected environment not specified",
                vertex="victim",
                status="unknown",
                details="Add supported victim geography or industry context to populate this vertex.",
            )
        )

    caveats = (
        "Diamond vertices are evidence-linked hypotheses, not proof of attribution.",
        "Adversary nodes come from the fused candidate ranking and remain uncertain.",
        "Infrastructure is shown only when explicitly observed or matched to public intelligence.",
        "Victim geography and industry are kept separate from attacker origin.",
    )
    return DiamondModel(
        nodes=tuple(sorted(nodes.values(), key=lambda node: (node.vertex, node.label, node.id))),
        edges=tuple(
            sorted(edges.values(), key=lambda edge: (edge.relationship, edge.source, edge.target))
        ),
        caveats=caveats,
    )


def _classifier_result(state: Any, name: str) -> Any:
    return next(
        (
            result
            for result in getattr(state, "classifier_results", ())
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


def _shorten(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def _slug(value: str) -> str:
    return "-".join(
        "".join(character if character.isalnum() else " " for character in value.casefold()).split()
    )


def _factor_label(field_name: str) -> str:
    return {
        "target_geographies": "Target geography",
        "target_industries": "Target industry",
    }.get(field_name, field_name.replace("_", " ").title())
