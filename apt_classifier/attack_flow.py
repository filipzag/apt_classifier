"""Attack Flow model, construction, and STIX 2.1 export.

This module builds a local Attack Flow representation from the case's
observations and technique matches. It preserves action dependencies and
sequence structure while distinguishing chronological order (from timestamps)
from dependency (from explicit source language or analyst confirmation).

Key design decisions:
- Timestamps can establish chronology, but dependency requires explicit source
  language, deterministic entity/state linkage, or analyst confirmation.
- ``precedes`` edges are labeled chronological, never causal/dependent.
- ``depends_on`` edges require explicit evidence.
- Flow similarity scoring is gated behind benchmarks and not enabled by
  default.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class FlowAction:
    """A single action in an Attack Flow."""

    id: str
    technique_id: str
    technique_name: str
    description: str
    observation_id: str
    timestamp: str | None = None
    tactic: str = ""
    confidence: float = 0.0
    provenance: str = ""


@dataclass(frozen=True)
class FlowAsset:
    """An asset/entity involved in the attack flow."""

    id: str
    name: str
    asset_type: str = "system"
    description: str = ""


@dataclass(frozen=True)
class FlowEdge:
    """An edge between flow actions or assets.

    ``edge_type`` is one of:
    - ``precedes``: chronological ordering (from timestamps)
    - ``depends_on``: dependency (from explicit source/analyst)
    - ``acts_on``: action affects an asset
    """

    source: str
    target: str
    edge_type: str
    confidence: float = 0.0
    details: str = ""
    is_chronological: bool = False


@dataclass(frozen=True)
class AttackFlow:
    """A complete attack flow with actions, assets, and edges."""

    id: str
    name: str
    actions: tuple[FlowAction, ...]
    assets: tuple[FlowAsset, ...]
    edges: tuple[FlowEdge, ...]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "actions": [
                {
                    "id": a.id,
                    "technique_id": a.technique_id,
                    "technique_name": a.technique_name,
                    "description": a.description,
                    "observation_id": a.observation_id,
                    "timestamp": a.timestamp,
                    "tactic": a.tactic,
                    "confidence": a.confidence,
                    "provenance": a.provenance,
                }
                for a in self.actions
            ],
            "assets": [
                {
                    "id": a.id,
                    "name": a.name,
                    "asset_type": a.asset_type,
                    "description": a.description,
                }
                for a in self.assets
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "confidence": e.confidence,
                    "details": e.details,
                    "is_chronological": e.is_chronological,
                }
                for e in self.edges
            ],
        }

    def to_stix(self) -> dict[str, Any]:
        """Export as Attack Flow STIX 2.1 JSON.

        References:
        https://center-for-threat-informed-defense.github.io/attack-flow/
        """
        actions_stix = []
        for action in self.actions:
            actions_stix.append(
                {
                    "type": "attack-flow",
                    "spec_version": "2.1",
                    "id": action.id,
                    "name": action.technique_name,
                    "description": action.description,
                    "attack_id": action.technique_id,
                    "tactic": action.tactic,
                    "timestamp": action.timestamp,
                    "confidence": action.confidence,
                    "provenance": action.provenance,
                }
            )
        edges_stix = []
        for edge in self.edges:
            if edge.edge_type in ("precedes", "depends_on"):
                edges_stix.append(
                    {
                        "type": "relationship",
                        "spec_version": "2.1",
                        "source_ref": edge.source,
                        "target_ref": edge.target,
                        "relationship_type": edge.edge_type,
                        "description": edge.details,
                        "confidence": edge.confidence,
                    }
                )
            elif edge.edge_type == "acts_on":
                edges_stix.append(
                    {
                        "type": "relationship",
                        "spec_version": "2.1",
                        "source_ref": edge.source,
                        "target_ref": edge.target,
                        "relationship_type": "relates-to",
                        "description": f"acts on: {edge.details}",
                        "confidence": edge.confidence,
                    }
                )
        return {
            "type": "bundle",
            "id": self.id,
            "objects": [
                {
                    "type": "attack-flow",
                    "spec_version": "2.1",
                    "id": self.id,
                    "name": self.name,
                    "description": self.description,
                    "actions": [a.id for a in self.actions],
                    "assets": [a.id for a in self.assets],
                },
                *actions_stix,
                *[
                    {
                        "type": "x-attack-flow-asset",
                        "spec_version": "2.1",
                        "id": a.id,
                        "name": a.name,
                        "asset_type": a.asset_type,
                        "description": a.description,
                    }
                    for a in self.assets
                ],
                *edges_stix,
            ],
        }


def build_attack_flow(
    kb: Any,
    state: Any,
    *,
    case_id: str = "case",
) -> AttackFlow:
    """Build an Attack Flow from the analysis state.

    Actions are derived from technique matches and their supporting
    observations. Chronological edges are added when timestamps are
    available. Dependency edges are NOT inferred from co-occurrence.
    """
    techniques = getattr(kb, "techniques", {})
    matches = getattr(state.retrieval, "matches", ())
    observations = {obs.id: obs for obs in getattr(state.ledger, "observations", ())}
    actions: list[FlowAction] = []
    action_by_observation: dict[str, FlowAction] = {}
    for match in matches:
        technique = techniques.get(match.technique_id)
        technique_name = (
            getattr(technique, "name", match.technique_id) if technique else match.technique_name
        )
        tactic = ""
        if technique and getattr(technique, "tactics", ()):
            tactic = technique.tactics[0]
        for observation_id in match.observation_ids:
            obs = observations.get(observation_id)
            if obs is None:
                continue
            timestamp = obs.metadata.get("timestamp") if hasattr(obs, "metadata") else None
            action_id = f"action:{observation_id}:{match.technique_id}"
            action = FlowAction(
                id=action_id,
                technique_id=match.technique_id,
                technique_name=technique_name,
                description=str(match.matched_text or obs.text)[:500],
                observation_id=observation_id,
                timestamp=str(timestamp) if timestamp else None,
                tactic=tactic,
                confidence=float(match.score),
                provenance=str(
                    getattr(obs, "provenance", SimpleNamespace(source="unknown")).source
                ),
            )
            actions.append(action)
            action_by_observation[observation_id] = action

    # Sort by timestamp if available
    def _sort_key(action: FlowAction) -> tuple[int, str]:
        if action.timestamp:
            return (0, action.timestamp)
        return (1, action.id)

    actions.sort(key=_sort_key)
    # Build chronological edges from timestamps
    edges: list[FlowEdge] = []
    timestamped = [a for a in actions if a.timestamp]
    for i in range(len(timestamped) - 1):
        current = timestamped[i]
        next_action = timestamped[i + 1]
        if (
            current.timestamp
            and next_action.timestamp
            and current.timestamp < next_action.timestamp
        ):
            edges.append(
                FlowEdge(
                    source=current.id,
                    target=next_action.id,
                    edge_type="precedes",
                    confidence=0.7,
                    details=f"Chronological: {current.timestamp} → {next_action.timestamp}",
                    is_chronological=True,
                )
            )
    # Build assets from observation metadata
    assets: list[FlowAsset] = []
    asset_ids: set[str] = set()
    for obs in observations.values():
        metadata = getattr(obs, "metadata", {}) if hasattr(obs, "metadata") else {}
        asset_name = metadata.get("host") or metadata.get("asset") or metadata.get("system")
        if asset_name and asset_name not in asset_ids:
            asset_id = f"asset:{asset_name}"
            asset_ids.add(asset_name)
            assets.append(
                FlowAsset(
                    id=asset_id,
                    name=str(asset_name),
                    asset_type=str(metadata.get("asset_type", "system")),
                )
            )
            # Link actions to assets
            action = action_by_observation.get(obs.id)
            if action:
                edges.append(
                    FlowEdge(
                        source=action.id,
                        target=asset_id,
                        edge_type="acts_on",
                        confidence=0.5,
                        details=f"Action on {asset_name}",
                    )
                )
    return AttackFlow(
        id=f"attack-flow:{case_id}",
        name=f"Attack Flow for {case_id}",
        actions=tuple(actions),
        assets=tuple(assets),
        edges=tuple(edges),
        description=f"Attack flow constructed from {len(actions)} technique matches.",
    )


# --- Flow similarity (experimental, gated behind benchmarks) ---


def flow_similarity(flow_a: AttackFlow, flow_b: AttackFlow) -> float:
    """Compute a simple set-overlap similarity between two attack flows.

    This is a baseline method. More sophisticated methods (tactic-aware
    weighted subsequence, graph-motif) should be compared against this
    baseline and only enabled if held-out attribution metrics improve.

    Returns a 0–1 similarity score.
    """
    techniques_a = {a.technique_id for a in flow_a.actions}
    techniques_b = {a.technique_id for a in flow_b.actions}
    if not techniques_a and not techniques_b:
        return 0.0
    intersection = techniques_a & techniques_b
    union = techniques_a | techniques_b
    return len(intersection) / len(union) if union else 0.0
