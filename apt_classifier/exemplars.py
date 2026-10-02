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

The library can also be served from a ChromaDB collection the npz was copied into
(``apt_classifier.exemplar_store``, ``TECHNIQUE_EXEMPLAR_BACKEND=chroma|auto``); the default
``npz`` backend is the artifact above. Chroma re-normalises the vectors on insert, so the library
it serves has its own fingerprint (an acceptance model runs only on the library it was fitted
on). Either way the votes come from the same exact search over a verified in-memory copy.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

EXEMPLAR_SCHEMA_VERSION = 1
KIND = "technique_exemplar_index"
DEFAULT_K = 64
DEFAULT_TAU = 0.05
DEFAULT_DUPLICATE_SIMILARITY = 0.95
# How a neighbour's weight is shared among its labels: "none" gives every label the full weight
# (the shipped behaviour); "label_split" divides it by the neighbour's label count.
VOTE_NORMS = ("none", "label_split")
DEFAULT_VOTE_NORM = "none"
# Long observations are retrieved sentence by sentence but query the exemplars once with the
# whole text ("off", the shipped behaviour). "max" / "mean" also query each sentence and pool the
# votes per technique over the whole text and its sentences (:func:`pool_votes`).
SEGMENT_VOTE_MODES = ("off", "max", "mean")
DEFAULT_SEGMENT_VOTES = "off"
EXEMPLAR_BACKENDS = ("npz", "chroma", "auto")
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


def window_size(count: int, k: int) -> int:
    """Neighbours fetched per query: twice ``k`` plus a margin, so that skipping near-copies of
    the query (``duplicate_similarity``) still leaves ``k`` voters."""
    return min(int(count), max(1, int(k)) * 2 + 16)


def exact_window(row: np.ndarray, window: int) -> np.ndarray:
    """The ``window`` most similar exemplars of one similarity row, most similar first.

    ``argpartition`` then a stable ``argsort``: ties keep the partition's order, so this is the
    one ordering every backend reuses (a second implementation could order ties differently)."""
    if window < len(row):
        candidates = np.argpartition(-row, window - 1)[:window]
    else:
        candidates = np.arange(len(row))
    return candidates[np.argsort(-row[candidates], kind="stable")]


def embedding_compatibility(
    metadata: Mapping[str, Any], embedding_model: str, embedding_revision: str | None
) -> tuple[bool, str]:
    """Whether a library's recorded embedding model and revision match the runtime's."""
    model = metadata.get("embedding_model")
    revision = metadata.get("embedding_revision")
    if model != embedding_model:
        return False, f"embedded with {model}, the technique index uses {embedding_model}"
    if (revision or None) != (embedding_revision or None):
        return False, f"embedding revision {revision} != {embedding_revision}"
    return True, "ok"


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


def pool_votes(queries: Sequence[ExemplarVotes], mode: str) -> ExemplarVotes:
    """One observation's votes pooled over several queries (its whole text first, then each of
    its sentences), with ranks recomputed from the pooled votes.

    ``max`` takes, per technique, the largest vote, hierarchy-relaxed vote and count any query
    gave; ``mean`` averages them over all queries (a query without the technique adds 0). Either
    way a technique's closest exemplar (similarity, text, source) is the closest over all
    queries and the top similarity the largest. A single query is returned unchanged."""
    if mode not in ("max", "mean"):
        raise ValueError(f"unknown exemplar segment-vote pooling {mode!r}")
    if not queries:
        return ExemplarVotes()
    if len(queries) == 1:
        return queries[0]
    count = len(queries)

    def combine(values: list[float]) -> float:
        return max(values) if mode == "max" else sum(values) / count

    techniques = sorted({tid for query in queries for tid in query.votes})
    family_ids = sorted({tid for query in queries for tid in query.family})
    votes = {tid: combine([query.votes.get(tid, 0.0) for query in queries]) for tid in techniques}
    family = {tid: combine([query.family.get(tid, 0.0) for query in queries]) for tid in family_ids}
    counts: dict[str, Any] = {}
    for tid in techniques:
        values = [query.counts.get(tid, 0) for query in queries]
        counts[tid] = max(values) if mode == "max" else sum(values) / count
    maxsim: dict[str, float] = {}
    best_text: dict[str, str] = {}
    best_source: dict[str, str] = {}
    for query in queries:
        for tid, similarity in query.max_similarity.items():
            if tid not in maxsim or similarity > maxsim[tid]:
                maxsim[tid] = similarity
                best_text[tid] = query.best_text.get(tid, "")
                best_source[tid] = query.best_source.get(tid, "")
    ranked = sorted(votes, key=lambda technique_id: (-votes[technique_id], technique_id))
    return ExemplarVotes(
        top_similarity=max(query.top_similarity for query in queries),
        votes=votes,
        family=family,
        max_similarity=maxsim,
        counts=counts,
        ranks={technique_id: rank for rank, technique_id in enumerate(ranked, 1)},
        best_text=best_text,
        best_source=best_source,
        neighbours=max(query.neighbours for query in queries),
        excluded_duplicates=max(query.excluded_duplicates for query in queries),
        k=queries[0].k,
    )


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


def npz_path(path: str | Path) -> Path:
    """The file ``numpy.savez`` writes for ``path``: the path itself when it ends in ``.npz``,
    else the path with ``.npz`` appended (so an overwrite guard checks the real target)."""
    target = Path(path)
    return target if str(target).endswith(".npz") else target.with_name(target.name + ".npz")


@dataclass
class ExemplarIndex:
    embeddings: np.ndarray
    labels: tuple[tuple[str, ...], ...]
    texts: tuple[str, ...]
    sources: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    # where the library was loaded from and how it is searched (the Chroma store says otherwise)
    backend: ClassVar[str] = "npz"
    search: ClassVar[str] = "exact"
    search_class: ClassVar[str] = "exact"

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

    def signature(self) -> dict[str, Any]:
        """What a run's results depend on besides the settings (recorded by the benchmarks)."""
        return {
            "backend": self.backend,
            "search": self.search,
            "search_class": self.search_class,
            "fingerprint": self.fingerprint,
            "count": len(self),
        }

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
        """Write the library and return the file written (``numpy`` adds ``.npz`` to a path
        without it, :func:`npz_path`)."""
        target = npz_path(path)
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
        return embedding_compatibility(self.metadata, embedding_model, embedding_revision)

    # ----- query -----
    def _similarities(self, vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """One full-batch ``queries @ E.T`` (float32): the reference similarities of every
        backend. The result depends on the batch (one gemm per call), so callers that compare
        backends must pass the same batches."""
        queries = np.asarray(vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if not len(queries):
            return np.zeros((0, len(self)), dtype=np.float32)
        queries = _unit(queries)
        return queries @ self.embeddings.T

    def exact_windows(
        self, vectors: Sequence[Sequence[float]] | np.ndarray, *, k: int = DEFAULT_K
    ) -> list[np.ndarray]:
        """The ordered exact neighbour window of each query (what the votes are drawn from)."""
        window = window_size(len(self), k)
        return [exact_window(row, window) for row in self._similarities(vectors)]

    def query_vectors(
        self,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        *,
        parent_of: Mapping[str, str | None] | None = None,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        duplicate_similarity: float = DEFAULT_DUPLICATE_SIMILARITY,
        vote_norm: str = DEFAULT_VOTE_NORM,
        allowed_rows: Sequence[Sequence[int] | np.ndarray] | None = None,
        stats: MutableMapping[str, int] | None = None,
    ) -> list[ExemplarVotes]:
        """Exemplar votes per query vector.

        ``allowed_rows`` (one row set per query) are the candidates an approximate search
        returned. The votes still come from the exact window of the full similarity row: when
        the window is inside the allowed rows it is the same ordered array, and when it is not
        the query is counted as a ``recall_miss`` in ``stats`` and the exact window is used, so
        the result never depends on the approximate search."""
        if vote_norm not in VOTE_NORMS:
            raise ValueError(f"unknown exemplar vote normalisation {vote_norm!r}")
        similarities = self._similarities(vectors)
        if allowed_rows is not None and len(allowed_rows) != len(similarities):
            raise ValueError("allowed_rows needs one row set per query")
        if not len(similarities):
            return []
        parents = parent_of or {}
        k = max(1, int(k))
        tau = max(1e-4, float(tau))
        window = window_size(len(self), k)
        results: list[ExemplarVotes] = []
        for position, row in enumerate(similarities):
            ordered = exact_window(row, window)
            if (
                allowed_rows is not None
                and stats is not None
                and not bool(
                    np.isin(ordered, np.asarray(allowed_rows[position], dtype=np.int64)).all()
                )
            ):
                # the candidate search missed part of the window: counted, the window stays exact
                stats["recall_misses"] = int(stats.get("recall_misses", 0)) + 1
            results.append(
                self._row_votes(
                    row,
                    ordered,
                    parents=parents,
                    k=k,
                    tau=tau,
                    duplicate_similarity=duplicate_similarity,
                    vote_norm=vote_norm,
                )
            )
        return results

    def _row_votes(
        self,
        row: np.ndarray,
        ordered: np.ndarray,
        *,
        parents: Mapping[str, str | None],
        k: int,
        tau: float,
        duplicate_similarity: float,
        vote_norm: str,
    ) -> ExemplarVotes:
        """The votes of one query from its ordered window. The float operations and their order
        are those of the original single-loop implementation (``tests/test_exemplars.py`` keeps
        a frozen copy and checks bit equality)."""
        excluded = int(np.sum(row[ordered] >= duplicate_similarity))
        chosen = [int(j) for j in ordered if row[j] < duplicate_similarity][:k]
        if not chosen:
            return ExemplarVotes(excluded_duplicates=excluded, k=k)
        sims = np.asarray([row[j] for j in chosen], dtype=np.float64)
        weights = np.exp((sims - sims[0]) / tau)
        weights = weights / weights.sum()
        votes: dict[str, float] = {}
        family: dict[str, float] = {}
        maxsim: dict[str, float] = {}
        counts: dict[str, int] = {}
        best_text: dict[str, str] = {}
        best_source: dict[str, str] = {}
        split = vote_norm == "label_split"
        for weight, sim, j in zip(weights, sims, chosen, strict=True):
            labels = self.labels[j]
            share = float(weight) / max(1, len(labels)) if split else float(weight)
            for label in labels:
                votes[label] = votes.get(label, 0.0) + share
                counts[label] = counts.get(label, 0) + 1
                family[label] = family.get(label, 0.0) + share
                parent = parents.get(label)
                if parent:
                    family[parent] = family.get(parent, 0.0) + share
                if label not in maxsim or sim > maxsim[label]:
                    maxsim[label] = float(sim)
                    best_text[label] = self.texts[j]
                    best_source[label] = self.sources[j]
        ranked = sorted(votes, key=lambda technique_id: (-votes[technique_id], technique_id))
        return ExemplarVotes(
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


_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{24}$")


def configured_library_fingerprint(settings: Any) -> str:
    """``technique_exemplar_library_fingerprint``: the fingerprint the served library must have
    (``""`` when none is pinned). Raises ``ValueError`` unless it is empty or 24 lower-case hex
    digits (:meth:`ExemplarIndex.compute_fingerprint`)."""
    value = str(getattr(settings, "technique_exemplar_library_fingerprint", "") or "").strip()
    if value and not _FINGERPRINT_RE.match(value):
        raise ValueError(
            f"TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT {value!r} is not a library fingerprint "
            "(24 lower-case hex digits)"
        )
    return value


def pinned_fingerprint_problem(fingerprint: str, pinned: str) -> str | None:
    """Why a library of ``fingerprint`` is refused under the pin ``pinned`` (``None`` when it
    is not pinned or matches)."""
    if not pinned or fingerprint == pinned:
        return None
    return (
        f"library {fingerprint} is not the pinned TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT "
        f"{pinned} (its vectors, labels or manifest changed since the fingerprint was registered)"
    )


def parent_map(kb: Any) -> dict[str, str | None]:
    techniques = getattr(kb, "techniques", {}) or {}
    return {tid: getattr(t, "parent_id", None) for tid, t in techniques.items()}


def load_exemplar_index(settings: Any, *, client: Any = None) -> tuple[Any, str]:
    """Load the configured exemplar library fail-closed: ``(index, status)``.

    ``technique_exemplar_backend`` picks the source: ``npz`` (default) the artifact,
    ``chroma`` the verified Chroma collection only (``None`` when it is unusable), ``auto`` the
    collection with the artifact as a visible fallback (``active (npz fallback: <reason>)``).
    A Chroma library checked against neither the npz nor a pinned
    ``technique_exemplar_library_fingerprint`` reads ``active (unverified: <why>)``; a pinned
    fingerprint refuses any other library on both backends.
    ``client`` is the vector store's Chroma client; the loader never builds one of its own, so a
    store without a client (the test fakes) never reaches the network."""
    if not bool(getattr(settings, "technique_exemplars_enabled", False)):
        return None, "disabled"
    backend = str(getattr(settings, "technique_exemplar_backend", "npz") or "npz").strip().lower()
    if backend == "npz":
        return _load_npz_index(settings)
    if backend not in EXEMPLAR_BACKENDS:
        return None, f"unknown exemplar backend {backend!r} (npz, chroma or auto)"
    # imported here: the default npz path needs neither chromadb nor the store module
    from apt_classifier.exemplar_store import load_chroma_exemplar_store

    store, reason = load_chroma_exemplar_store(settings, client)
    if store is not None:
        return store, store.status
    if backend == "chroma":
        return None, f"chroma exemplar store unavailable: {reason}"
    index, status = _load_npz_index(settings)
    if index is None:
        return None, f"{status}; chroma exemplar store unavailable: {reason}"
    return index, f"active (npz fallback: {reason})"


def _load_npz_index(settings: Any) -> tuple[ExemplarIndex | None, str]:
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
    try:
        pinned = configured_library_fingerprint(settings)
    except ValueError as error:
        return None, str(error)
    problem = pinned_fingerprint_problem(index.fingerprint, pinned)
    if problem is not None:
        return None, f"incompatible artifact: {problem}"
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
        "exemplar_vote_norm": str(
            getattr(settings, "technique_exemplar_vote_norm", DEFAULT_VOTE_NORM)
            or DEFAULT_VOTE_NORM
        ),
        "exemplar_segment_votes": str(
            getattr(settings, "technique_exemplar_segment_votes", DEFAULT_SEGMENT_VOTES)
            or DEFAULT_SEGMENT_VOTES
        ),
    }
