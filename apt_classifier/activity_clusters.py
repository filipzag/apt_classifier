"""Local longitudinal activity-cluster tracking.

This module provides opt-in local case/cluster storage using standard-library
SQLite for structured metadata. It does not store raw logs in vectors. Case
signatures are built from reviewed techniques, procedure details, tools,
infrastructure, victimology, temporal features, and evidence-quality metadata.

Key design decisions:
- Opt-in only; never enabled by default.
- Never auto-merge clusters or promote a cluster to a known actor.
- Cluster suggestions expose component scores and require explicit analyst
  approval for merge/link.
- External live intelligence remains non-voting.
- Raw logs and secrets never enter case vectors or artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseSignature:
    """A redacted signature for a case, used for similarity matching.

    Raw logs are never stored. Only structured, reviewed metadata is retained.
    """

    case_id: str
    techniques: tuple[str, ...]
    software: tuple[str, ...]
    infrastructure: tuple[str, ...]
    victimology: tuple[str, ...]
    temporal_features: dict[str, Any]
    evidence_quality_summary: dict[str, Any]
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "techniques": list(self.techniques),
            "software": list(self.software),
            "infrastructure": list(self.infrastructure),
            "victimology": list(self.victimology),
            "temporal_features": dict(self.temporal_features),
            "evidence_quality_summary": dict(self.evidence_quality_summary),
            "created_at": self.created_at,
        }

    def fingerprint(self) -> str:
        """Stable hash of the signature for deduplication."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ClusterSuggestion:
    """A suggestion that a case may be related to a prior case/cluster.

    Component scores are exposed so the analyst can understand why the
    suggestion was made. The analyst must explicitly approve any merge/link.
    """

    case_id: str
    suggested_case_id: str
    overall_similarity: float
    technique_similarity: float
    software_similarity: float
    infrastructure_similarity: float
    temporal_similarity: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suggested_case_id": self.suggested_case_id,
            "overall_similarity": round(self.overall_similarity, 3),
            "technique_similarity": round(self.technique_similarity, 3),
            "software_similarity": round(self.software_similarity, 3),
            "infrastructure_similarity": round(self.infrastructure_similarity, 3),
            "temporal_similarity": round(self.temporal_similarity, 3),
            "rationale": self.rationale,
        }


def build_case_signature(
    case_id: str,
    state: Any,
    *,
    evidence_quality: dict[str, Any] | None = None,
) -> CaseSignature:
    """Build a redacted case signature from the analysis state.

    Only structured, reviewed metadata is retained. Raw logs are excluded.
    """
    observed = dict(getattr(state.retrieval, "observed_techniques", {}))
    techniques = tuple(sorted(observed.keys()))
    software_ids = tuple(sorted(getattr(state.context, "software_ids", set())))
    # Infrastructure from observation metadata (IOCs, domains, IPs)
    infrastructure: set[str] = set()
    victimology: set[str] = set()
    for obs in getattr(state.ledger, "observations", ()):
        metadata = getattr(obs, "metadata", {}) if hasattr(obs, "metadata") else {}
        if metadata.get("ioc"):
            infrastructure.add(str(metadata["ioc"]))
        if metadata.get("domain"):
            infrastructure.add(str(metadata["domain"]))
        if metadata.get("victim"):
            victimology.add(str(metadata["victim"]))
        if metadata.get("sector"):
            victimology.add(str(metadata["sector"]))
    # Temporal features
    timestamps: list[str] = []
    for obs in getattr(state.ledger, "observations", ()):
        metadata = getattr(obs, "metadata", {}) if hasattr(obs, "metadata") else {}
        ts = metadata.get("timestamp")
        if ts:
            timestamps.append(str(ts))
    temporal_features: dict[str, Any] = {}
    if timestamps:
        temporal_features["first_seen"] = min(timestamps)
        temporal_features["last_seen"] = max(timestamps)
        temporal_features["count"] = len(timestamps)
    return CaseSignature(
        case_id=case_id,
        techniques=techniques,
        software=software_ids,
        infrastructure=tuple(sorted(infrastructure)),
        victimology=tuple(sorted(victimology)),
        temporal_features=temporal_features,
        evidence_quality_summary=evidence_quality or {},
        created_at=datetime.now(UTC).isoformat(),
    )


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _temporal_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute temporal similarity based on time window overlap."""
    if not a or not b:
        return 0.0
    a_start = a.get("first_seen", "")
    a_end = a.get("last_seen", "")
    b_start = b.get("first_seen", "")
    b_end = b.get("last_seen", "")
    if not all([a_start, a_end, b_start, b_end]):
        return 0.0
    # Simple: if windows overlap, similarity is high
    if a_start <= b_end and b_start <= a_end:
        return 0.8
    return 0.0


def compute_similarity(a: CaseSignature, b: CaseSignature) -> ClusterSuggestion:
    """Compute decomposed similarity between two case signatures."""
    tech_sim = _jaccard(a.techniques, b.techniques)
    sw_sim = _jaccard(a.software, b.software)
    infra_sim = _jaccard(a.infrastructure, b.infrastructure)
    temp_sim = _temporal_similarity(a.temporal_features, b.temporal_features)
    # Weighted combination
    overall = (
        0.35 * tech_sim
        + 0.20 * sw_sim
        + 0.20 * infra_sim
        + 0.15 * temp_sim
        + 0.10 * (1.0 if a.victimology and a.victimology == b.victimology else 0.0)
    )
    components = []
    if tech_sim > 0:
        components.append(f"techniques: {tech_sim:.2f}")
    if sw_sim > 0:
        components.append(f"software: {sw_sim:.2f}")
    if infra_sim > 0:
        components.append(f"infrastructure: {infra_sim:.2f}")
    if temp_sim > 0:
        components.append(f"temporal: {temp_sim:.2f}")
    rationale = (
        f"Similarity based on {', '.join(components)}"
        if components
        else "No significant similarity"
    )
    return ClusterSuggestion(
        case_id=a.case_id,
        suggested_case_id=b.case_id,
        overall_similarity=overall,
        technique_similarity=tech_sim,
        software_similarity=sw_sim,
        infrastructure_similarity=infra_sim,
        temporal_similarity=temp_sim,
        rationale=rationale,
    )


class ActivityClusterStore:
    """SQLite-backed local store for case signatures and cluster links.

    All data is stored locally. No raw logs are retained. The store is
    opt-in and must be explicitly enabled by the analyst.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS case_signatures (
                    case_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    signature_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_links (
                    case_id_a TEXT NOT NULL,
                    case_id_b TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    rationale TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (case_id_a, case_id_b)
                )
                """
            )

    def store_signature(self, signature: CaseSignature) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO case_signatures (case_id, fingerprint, signature_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    signature.case_id,
                    signature.fingerprint(),
                    json.dumps(signature.to_dict(), ensure_ascii=False),
                    signature.created_at or datetime.now(UTC).isoformat(),
                ),
            )

    def get_signature(self, case_id: str) -> CaseSignature | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT signature_json FROM case_signatures WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return CaseSignature(
            case_id=data["case_id"],
            techniques=tuple(data["techniques"]),
            software=tuple(data["software"]),
            infrastructure=tuple(data["infrastructure"]),
            victimology=tuple(data["victimology"]),
            temporal_features=data["temporal_features"],
            evidence_quality_summary=data["evidence_quality_summary"],
            created_at=data["created_at"],
        )

    def find_related(
        self,
        signature: CaseSignature,
        *,
        min_similarity: float = 0.2,
        max_results: int = 10,
    ) -> list[ClusterSuggestion]:
        """Find related prior cases by similarity."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT signature_json FROM case_signatures WHERE case_id != ?",
                (signature.case_id,),
            ).fetchall()
        suggestions: list[ClusterSuggestion] = []
        for (sig_json,) in rows:
            data = json.loads(sig_json)
            other = CaseSignature(
                case_id=data["case_id"],
                techniques=tuple(data["techniques"]),
                software=tuple(data["software"]),
                infrastructure=tuple(data["infrastructure"]),
                victimology=tuple(data["victimology"]),
                temporal_features=data["temporal_features"],
                evidence_quality_summary=data["evidence_quality_summary"],
                created_at=data["created_at"],
            )
            suggestion = compute_similarity(signature, other)
            if suggestion.overall_similarity >= min_similarity:
                suggestions.append(suggestion)
        suggestions.sort(key=lambda s: -s.overall_similarity)
        return suggestions[:max_results]

    def add_link(
        self,
        case_id_a: str,
        case_id_b: str,
        link_type: str,
        rationale: str = "",
    ) -> None:
        """Record an analyst-approved link between cases."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cluster_links (case_id_a, case_id_b, link_type, rationale, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    case_id_a,
                    case_id_b,
                    link_type,
                    rationale,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_links(self, case_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT case_id_a, case_id_b, link_type, rationale, created_at
                FROM cluster_links WHERE case_id_a = ? OR case_id_b = ?
                """,
                (case_id, case_id),
            ).fetchall()
        return [
            {
                "case_id_a": row[0],
                "case_id_b": row[1],
                "link_type": row[2],
                "rationale": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def delete_signature(self, case_id: str) -> None:
        """Delete a case signature and its links."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM case_signatures WHERE case_id = ?", (case_id,))
            conn.execute(
                "DELETE FROM cluster_links WHERE case_id_a = ? OR case_id_b = ?",
                (case_id, case_id),
            )

    def export_all(self) -> dict[str, Any]:
        """Export all stored data for audit/deletion purposes."""
        with sqlite3.connect(self.db_path) as conn:
            sigs = conn.execute(
                "SELECT case_id, fingerprint, signature_json, created_at FROM case_signatures"
            ).fetchall()
            links = conn.execute(
                "SELECT case_id_a, case_id_b, link_type, rationale, created_at FROM cluster_links"
            ).fetchall()
        return {
            "signatures": [
                {
                    "case_id": row[0],
                    "fingerprint": row[1],
                    "signature": json.loads(row[2]),
                    "created_at": row[3],
                }
                for row in sigs
            ],
            "links": [
                {
                    "case_id_a": row[0],
                    "case_id_b": row[1],
                    "link_type": row[2],
                    "rationale": row[3],
                    "created_at": row[4],
                }
                for row in links
            ],
        }
