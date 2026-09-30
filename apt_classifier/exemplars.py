"""Exemplars: labelled real-report sentences as technique-mapping evidence.

The dense index holds ATT&CK's own prose (technique descriptions, procedure examples). Real
incident and report text is written differently, and on it the acceptance model left 40% of
the correct techniques in review because retrieval similarity to ATT&CK prose could not tell
them apart from wrong neighbours. An exemplar index adds a second kind of evidence: *how other
real report sentences that read like this one were labelled*. It holds the public training
sentences (TRAM, expert and derived-procedure splits of tumeteor/mitre-ttp-mapping, CC-BY-4.0;
never the leaky ATT&CK-procedure split, never dev or test, never CTI-HAL) embedded with the same
model as the technique index.

For an observation the index returns the K most similar exemplars (any exemplar at or above
``duplicate_similarity`` is skipped, so a near-copy of the query can never vote its own label),
weighted by a softmax over similarity. Each technique gets a vote share, a hierarchy-relaxed
share (its own votes plus those of its sub-techniques), the similarity of its closest exemplar,
its rank and count, and the nearest exemplar text for display. These become acceptance features
(``apt_classifier.acceptance``) and can add a few exemplar-only candidates the ATT&CK-prose
retrieval missed. The artifact is ``.npz`` with a JSON manifest, pinned to the embedding model
and revision, integrity-checked by a fingerprint, and loaded fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EXEMPLAR_SCHEMA_VERSION = 1
KIND = "technique_exemplar_index"
DEFAULT_K = 64
DEFAULT_TAU = 0.05
DEFAULT_DUPLICATE_SIMILARITY = 0.95
FEATURE_NAMES: tuple[str, ...] = (
    "ex_vote",
    "ex_family",
    "ex_maxsim_rel",
    "ex_rank_inv",
    "ex_count",
    "ex_top_sim",
    "ex_vote_x_top",
    "exemplar_only",
)
EVIDENCE_SOURCE = "exemplar"
NO_EXEMPLAR_SIMILARITY = -0.5  # ex_maxsim_rel when no exemplar carries the technique


def _normalise_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


# Structured telemetry reaches the pipeline as field/value records ("command_line: whoami /all.
# pid: 5512. process_name: C:\\Windows\\System32\\whoami.exe"). The exemplars are report prose, where
# words such as "pid" and "process_name" belong to sentences about enumerating processes, so the
# record syntax alone pulls the neighbours towards Process Discovery. The exemplar query therefore
# keeps a record's values and drops its schema: lowercase identifier keys and values that are bare
# identifiers (numbers, hex, GUIDs, hashes). Prose, and records with fewer than two fields, are
# queried unchanged; retrieval over ATT&CK text always sees the original observation.
_RECORD_FIELD_RE = re.compile(r"(?:^|(?<=[.;,\n]))\s*([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\s*[:=]\s+")
_IDENTIFIER_VALUE_RE = re.compile(
    r"^(?:\d+|0x[0-9a-fA-F]+|\{?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\}?|[0-9a-fA-F]{32,})$"
)


def exemplar_query_text(text: str) -> str:
    """The text used to query the exemplar index: a record's values, or the prose unchanged."""
    source = str(text or "")
    fields = list(_RECORD_FIELD_RE.finditer(source))
    if len(fields) < 2 or source[: fields[0].start()].strip():
        return source
    values: list[str] = []
    for position, match in enumerate(fields):
        end = fields[position + 1].start() if position + 1 < len(fields) else len(source)
        value = source[match.end() : end].strip().rstrip(".;,").strip()
        if value and not _IDENTIFIER_VALUE_RE.match(value):
            values.append(value)
    return ". ".join(values) if values else source


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass(frozen=True)
class ExemplarVotes:
    """What the nearest labelled exemplars say about one observation."""

    top_similarity: float = 0.0
    votes: dict[str, float] = field(default_factory=dict)
    family: dict[str, float] = field(default_factory=dict)
    max_similarity: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    best_text: dict[str, str] = field(default_factory=dict)
    best_source: dict[str, str] = field(default_factory=dict)
    neighbours: int = 0
    excluded_duplicates: int = 0
    k: int = DEFAULT_K

    @property
    def ranked(self) -> list[str]:
        return sorted(self.ranks, key=lambda technique_id: self.ranks[technique_id])

    def features(self, technique_id: str, *, exemplar_only: bool = False) -> dict[str, float]:
        vote = float(self.votes.get(technique_id, 0.0))
        maxsim = self.max_similarity.get(technique_id)
        rank = self.ranks.get(technique_id)
        return {
            "ex_vote": vote,
            "ex_family": float(self.family.get(technique_id, 0.0)),
            "ex_maxsim_rel": (float(maxsim) - self.top_similarity)
            if maxsim is not None
            else NO_EXEMPLAR_SIMILARITY,
            "ex_rank_inv": 1.0 / rank if rank else 0.0,
            "ex_count": float(self.counts.get(technique_id, 0)) / max(1, self.k),
            "ex_top_sim": float(self.top_similarity),
            "ex_vote_x_top": vote * float(self.top_similarity),
            "exemplar_only": 1.0 if exemplar_only else 0.0,
        }


def empty_features() -> dict[str, float]:
    """Exemplar features of a candidate that carries no exemplar evidence."""
    return {
        "ex_vote": 0.0,
        "ex_family": 0.0,
        "ex_maxsim_rel": NO_EXEMPLAR_SIMILARITY,
        "ex_rank_inv": 0.0,
        "ex_count": 0.0,
        "ex_top_sim": 0.0,
        "ex_vote_x_top": 0.0,
        "exemplar_only": 0.0,
    }


def features_from_evidence(evidence: Any) -> dict[str, float]:
    """Rebuild the exemplar features from a candidate's ``exemplar`` evidence item."""
    if evidence is None:
        return empty_features()
    metadata = dict(getattr(evidence, "metadata", {}) or {})
    values = metadata.get("features")
    if isinstance(values, Mapping):
        base = empty_features()
        base.update({name: float(values.get(name, base[name])) for name in FEATURE_NAMES})
        return base
    return empty_features()


@dataclass
class ExemplarIndex:
    embeddings: np.ndarray
    labels: tuple[tuple[str, ...], ...]
    texts: tuple[str, ...]
    sources: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.labels):
            raise ValueError("exemplar embeddings and labels are misaligned")
        if not (len(self.labels) == len(self.texts) == len(self.sources)):
            raise ValueError("exemplar texts, labels and sources are misaligned")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def fingerprint(self) -> str:
        return str(self.metadata.get("fingerprint") or "")

    def compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.embeddings, dtype=np.float32).tobytes())
        digest.update(json.dumps([list(label) for label in self.labels]).encode())
        digest.update(
            json.dumps(
                {
                    "embedding_model": self.metadata.get("embedding_model"),
                    "embedding_revision": self.metadata.get("embedding_revision"),
                    "schema": self.metadata.get("schema_version"),
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()[:24]

    # ----- build and persistence -----
    @classmethod
    def build(
        cls,
        rows: Iterable[tuple[str, Sequence[str], str]],
        embedder: Any,
        *,
        embedding_model: str,
        embedding_revision: str | None,
        attack_version: str,
        extra_metadata: Mapping[str, Any] | None = None,
        batch_size: int = 128,
    ) -> ExemplarIndex:
        """Rows are (text, technique ids, source dataset). Identical texts are merged (labels
        united); rows without labels are dropped."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for text, labels, source in rows:
            clean = " ".join(str(text).split())
            ids = [str(label).strip().upper() for label in labels if str(label).strip()]
            if not clean or not ids:
                continue
            key = _normalise_text(clean)
            if key not in merged:
                merged[key] = {"text": clean, "labels": set(), "source": source}
                order.append(key)
            merged[key]["labels"].update(ids)
        if not order:
            raise ValueError("no labelled exemplar rows")
        texts = [merged[key]["text"] for key in order]
        vectors = np.asarray(embedder.embed(texts, batch_size=batch_size), dtype=np.float32)
        index = cls(
            embeddings=_unit(vectors),
            labels=tuple(tuple(sorted(merged[key]["labels"])) for key in order),
            texts=tuple(texts),
            sources=tuple(str(merged[key]["source"]) for key in order),
            metadata={
                "kind": KIND,
                "schema_version": EXEMPLAR_SCHEMA_VERSION,
                "embedding_model": embedding_model,
                "embedding_revision": embedding_revision,
                "attack_version": attack_version,
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "count": len(order),
                "labels": len({label for key in order for label in merged[key]["labels"]}),
                **dict(extra_metadata or {}),
            },
        )
        index.metadata["fingerprint"] = index.compute_fingerprint()
        return index

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            embeddings=np.ascontiguousarray(self.embeddings, dtype=np.float32),
            labels=np.asarray(json.dumps([list(label) for label in self.labels])),
            texts=np.asarray(json.dumps(list(self.texts))),
            sources=np.asarray(json.dumps(list(self.sources))),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ExemplarIndex | None:
        source = Path(path)
        if not source.is_file():
            return None
        with np.load(source, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("kind") != KIND:
                raise ValueError("not an exemplar index artifact")
            if int(metadata.get("schema_version", 0)) != EXEMPLAR_SCHEMA_VERSION:
                raise ValueError("exemplar index schema version does not match")
            index = cls(
                embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
                labels=tuple(tuple(label) for label in json.loads(str(payload["labels"]))),
                texts=tuple(json.loads(str(payload["texts"]))),
                sources=tuple(json.loads(str(payload["sources"]))),
                metadata=metadata,
            )
        if index.compute_fingerprint() != index.fingerprint:
            raise ValueError("exemplar index fingerprint does not match its contents")
        return index

    def compatible_with(
        self, embedding_model: str, embedding_revision: str | None
    ) -> tuple[bool, str]:
        model = self.metadata.get("embedding_model")
        revision = self.metadata.get("embedding_revision")
        if model != embedding_model:
            return False, f"embedded with {model}, the technique index uses {embedding_model}"
        if (revision or None) != (embedding_revision or None):
            return False, f"embedding revision {revision} != {embedding_revision}"
        return True, "ok"

    # ----- query -----
    def query_vectors(
        self,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        *,
        parent_of: Mapping[str, str | None] | None = None,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        duplicate_similarity: float = DEFAULT_DUPLICATE_SIMILARITY,
    ) -> list[ExemplarVotes]:
        queries = np.asarray(vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if not len(queries):
            return []
        queries = _unit(queries)
        similarities = queries @ self.embeddings.T
        parents = parent_of or {}
        k = max(1, int(k))
        tau = max(1e-4, float(tau))
        window = min(len(self), k * 2 + 16)
        results: list[ExemplarVotes] = []
        for row in similarities:
            if window < len(row):
                candidates = np.argpartition(-row, window - 1)[:window]
            else:
                candidates = np.arange(len(row))
            ordered = candidates[np.argsort(-row[candidates], kind="stable")]
            excluded = int(np.sum(row[ordered] >= duplicate_similarity))
            chosen = [int(j) for j in ordered if row[j] < duplicate_similarity][:k]
            if not chosen:
                results.append(ExemplarVotes(excluded_duplicates=excluded, k=k))
                continue
            sims = np.asarray([row[j] for j in chosen], dtype=np.float64)
            weights = np.exp((sims - sims[0]) / tau)
            weights = weights / weights.sum()
            votes: dict[str, float] = {}
            family: dict[str, float] = {}
            maxsim: dict[str, float] = {}
            counts: dict[str, int] = {}
            best_text: dict[str, str] = {}
            best_source: dict[str, str] = {}
            for weight, sim, j in zip(weights, sims, chosen, strict=True):
                for label in self.labels[j]:
                    votes[label] = votes.get(label, 0.0) + float(weight)
                    counts[label] = counts.get(label, 0) + 1
                    family[label] = family.get(label, 0.0) + float(weight)
                    parent = parents.get(label)
                    if parent:
                        family[parent] = family.get(parent, 0.0) + float(weight)
                    if label not in maxsim or sim > maxsim[label]:
                        maxsim[label] = float(sim)
                        best_text[label] = self.texts[j]
                        best_source[label] = self.sources[j]
            ranked = sorted(votes, key=lambda technique_id: (-votes[technique_id], technique_id))
            results.append(
                ExemplarVotes(
                    top_similarity=float(sims[0]),
                    votes=votes,
                    family=family,
                    max_similarity=maxsim,
                    counts=counts,
                    ranks={technique_id: rank for rank, technique_id in enumerate(ranked, 1)},
                    best_text=best_text,
                    best_source=best_source,
                    neighbours=len(chosen),
                    excluded_duplicates=excluded,
                    k=k,
                )
            )
        return results


def parent_map(kb: Any) -> dict[str, str | None]:
    techniques = getattr(kb, "techniques", {}) or {}
    return {tid: getattr(t, "parent_id", None) for tid, t in techniques.items()}


def load_exemplar_index(settings: Any) -> tuple[ExemplarIndex | None, str]:
    """Load the configured exemplar index fail-closed: ``(index, status)``."""
    if not bool(getattr(settings, "technique_exemplars_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "technique_exemplar_index_path", None)
    if not path:
        return None, "no path configured"
    try:
        index = ExemplarIndex.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        return None, f"invalid artifact: {error}"
    if index is None:
        return None, f"artifact not found: {path} (run make exemplars)"
    ok, reason = index.compatible_with(
        str(getattr(settings, "embedding_model", "") or ""),
        getattr(settings, "embedding_model_revision", None),
    )
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return index, "active"


def exemplar_kwargs(settings: Any, index: ExemplarIndex | None) -> dict[str, Any]:
    """Keyword arguments for ``retrieve_techniques`` from the settings and a loaded index."""
    return {
        "exemplar_index": index,
        "exemplar_k": int(getattr(settings, "technique_exemplar_k", DEFAULT_K)),
        "exemplar_tau": float(getattr(settings, "technique_exemplar_tau", DEFAULT_TAU)),
        "exemplar_duplicate_similarity": float(
            getattr(
                settings, "technique_exemplar_duplicate_similarity", DEFAULT_DUPLICATE_SIMILARITY
            )
        ),
        "exemplar_new_candidates": int(getattr(settings, "technique_exemplar_new_candidates", 0)),
    }
