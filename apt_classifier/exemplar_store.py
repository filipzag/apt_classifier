"""The exemplar library in ChromaDB: a verified, read-only copy of the npz exemplar index.

``scripts/sync_exemplars_to_chroma.py`` writes the npz library once into a new collection
(``technique_exemplars-<fingerprint>-s1-m<M>-efc<ef_construction>``): one record per exemplar
row with the exact text as document, the npz embedding row as vector (copied, never re-embedded)
and the row's labels and source as metadata. The collection metadata carries the npz manifest
verbatim (``source_metadata_json``), the fingerprint and a hash over every record.

At runtime :func:`load_chroma_exemplar_store` opens the collection with ``get_collection`` only
(it never creates, modifies, adds, upserts or deletes), downloads every record and refuses the
copy unless it is complete and identical: kind and schema, embedding model and revision, count
and rows ``0..N-1``, the fingerprint recomputed from the downloaded vectors and labels (against
the collection's own and, when the npz exists, the npz's), the record hash, and the HNSW
configuration (cosine, ``ef_search`` at least ``technique_exemplar_hnsw_ef_search``). The
verified copy is memoised per client and collection, so the per-fold engines of a benchmark
share one download.

Votes never use Chroma's distances (the server re-normalises its own float32 copy). With
``search="exact"`` the store answers from the verified copy exactly as the npz would. With
``search="hnsw"`` it also asks Chroma for each query's nearest records; Chroma only proposes
candidates, the votes still come from the exact window of the local similarity row, and a query
whose exact window is not inside Chroma's results is counted as a ``recall_miss``. Results are
therefore bit-identical on every backend and search; the counters measure the ANN index. A
failed Chroma query falls back to the local search (``search_fallbacks``) unless the backend is
strict (``chroma``), where it raises.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from apt_classifier.exemplars import (
    DEFAULT_DUPLICATE_SIMILARITY,
    DEFAULT_K,
    DEFAULT_TAU,
    DEFAULT_VOTE_NORM,
    EXEMPLAR_SCHEMA_VERSION,
    KIND,
    ExemplarIndex,
    ExemplarVotes,
    _unit,
    embedding_compatibility,
    window_size,
)
from apt_classifier.memo import ObjectMemo
from apt_classifier.vectorstore import (
    batched_query,
    canonical_json,
    hnsw_configuration,
)
from apt_classifier.vectorstore import records_hash as compute_records_hash

STORE_KIND = "technique_exemplars"
STORE_SCHEMA_VERSION = 1
COLLECTION_PREFIX = "technique_exemplars-"
SEARCH_MODES = ("exact", "hnsw")
GET_PAGE = 1000  # records per get() while downloading the library
# Collection metadata written once at creation (scripts/sync_exemplars_to_chroma.py).
REQUIRED_METADATA = (
    "kind",
    "store_schema_version",
    "exemplar_schema_version",
    "fingerprint",
    "count",
    "records_hash",
    "source_metadata_json",
)


class ExemplarStoreError(ValueError):
    """The collection is not a complete, identical copy of an exemplar library."""


@runtime_checkable
class ExemplarSource(Protocol):
    """What retrieval needs from an exemplar library (npz index or Chroma store)."""

    backend: str
    search: str
    search_class: str

    @property
    def fingerprint(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    def __len__(self) -> int: ...

    def signature(self) -> dict[str, Any]: ...

    def query_vectors(
        self,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        *,
        parent_of: Mapping[str, str | None] | None = None,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        duplicate_similarity: float = DEFAULT_DUPLICATE_SIMILARITY,
        vote_norm: str = DEFAULT_VOTE_NORM,
    ) -> list[ExemplarVotes]: ...


# ----- naming and record schema -----
def exemplar_collection_name(fingerprint: str, m: int, ef_construction: int) -> str:
    """Every setting that cannot change after creation is in the name (58 characters for a
    24-character fingerprint; Chroma allows 63)."""
    return (
        f"{COLLECTION_PREFIX}{fingerprint[:24]}-s{STORE_SCHEMA_VERSION}"
        f"-m{int(m)}-efc{int(ef_construction)}"
    )


def record_id(row: int) -> str:
    return f"ex-{int(row):05d}"


def record_metadata(index: ExemplarIndex, row: int) -> dict[str, Any]:
    labels = list(index.labels[row])
    return {
        "row": int(row),
        "labels": json.dumps(labels),
        "label_count": len(labels),
        "source": str(index.sources[row]),
    }


def exemplar_records(index: ExemplarIndex) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """``(ids, documents, metadatas)`` of every exemplar row, in row order."""
    rows = range(len(index))
    return (
        [record_id(row) for row in rows],
        [index.texts[row] for row in rows],
        [record_metadata(index, row) for row in rows],
    )


def collection_metadata(
    index: ExemplarIndex, *, source_npz_sha256: str | None, records: str
) -> dict[str, Any]:
    """The collection metadata written once at creation (Chroma values are scalars; the npz
    manifest is kept verbatim as JSON so ``None`` revisions survive the round trip)."""
    metadata = index.metadata
    return {
        "kind": STORE_KIND,
        "store_schema_version": STORE_SCHEMA_VERSION,
        "exemplar_schema_version": int(metadata.get("schema_version", 0) or 0),
        "fingerprint": index.fingerprint,
        "count": len(index),
        "labels": int(metadata.get("labels", 0) or 0),
        "embedding_model": str(metadata.get("embedding_model") or ""),
        "embedding_revision": str(metadata.get("embedding_revision") or ""),
        "attack_version": str(metadata.get("attack_version") or ""),
        "split": str(metadata.get("split") or ""),
        "source": str(metadata.get("source") or ""),
        "license": str(metadata.get("license") or ""),
        "source_npz_sha256": str(source_npz_sha256 or ""),
        "records_hash": records,
        "source_metadata_json": canonical_json(metadata),
    }


def npz_fingerprint(path: Any) -> str | None:
    """The fingerprint recorded in an npz exemplar artifact, ``None`` when there is none
    (reads the manifest only, not the vectors)."""
    if not path or not Path(path).is_file():
        return None
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            value = json.loads(str(payload["metadata"])).get("fingerprint")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return str(value) if value else None


# ----- download and verification -----
def _metadata_problems(
    metadata: Mapping[str, Any], *, count: int, embedding_model: str, revision: str | None
) -> str | None:
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        return f"collection metadata lacks {', '.join(missing)}"
    if metadata.get("kind") != STORE_KIND:
        return f"not an exemplar store (kind {metadata.get('kind')!r})"
    if int(metadata.get("store_schema_version", 0) or 0) != STORE_SCHEMA_VERSION:
        return f"store schema {metadata.get('store_schema_version')!r} != {STORE_SCHEMA_VERSION}"
    if int(metadata.get("exemplar_schema_version", 0) or 0) != EXEMPLAR_SCHEMA_VERSION:
        return (
            f"exemplar schema {metadata.get('exemplar_schema_version')!r} != "
            f"{EXEMPLAR_SCHEMA_VERSION}"
        )
    try:
        source = json.loads(str(metadata["source_metadata_json"]))
    except json.JSONDecodeError:
        return "source_metadata_json is not JSON"
    if not isinstance(source, dict) or source.get("kind") != KIND:
        return "source_metadata_json is not an exemplar index manifest"
    if int(source.get("schema_version", 0) or 0) != EXEMPLAR_SCHEMA_VERSION:
        return "source manifest schema does not match"
    if source.get("fingerprint") != metadata.get("fingerprint"):
        return "collection fingerprint differs from its source manifest"
    ok, reason = embedding_compatibility(source, embedding_model, revision)
    if not ok:
        return reason
    if int(metadata.get("count", -1)) != count or int(source.get("count", -1)) != count:
        return (
            f"collection holds {count} records, its metadata says {metadata.get('count')} "
            f"(source manifest {source.get('count')}): incomplete or changed"
        )
    return None


def read_exemplar_collection(collection: Any, metadata: Mapping[str, Any]) -> ExemplarIndex:
    """Download every record and rebuild the library in npz row order; raises
    :class:`ExemplarStoreError` unless rows ``0..N-1`` are present once each, the labels parse,
    and the recomputed record hash and fingerprint match the collection metadata."""
    count = int(metadata["count"])
    ids = [record_id(row) for row in range(count)]
    vectors: list[Any] = [None] * count
    documents: list[str | None] = [None] * count
    metadatas: list[dict[str, Any] | None] = [None] * count
    for start in range(0, count, GET_PAGE):
        page = ids[start : start + GET_PAGE]
        result = collection.get(ids=page, include=["embeddings", "documents", "metadatas"])
        returned = [str(item) for item in (result.get("ids") or [])]
        if sorted(returned) != sorted(page):
            raise ExemplarStoreError(f"records {page[0]}..{page[-1]} are missing or duplicated")
        page_vectors = result.get("embeddings")
        page_documents = result.get("documents") or []
        page_metadatas = result.get("metadatas") or []
        if page_vectors is None or len(page_vectors) != len(returned):
            raise ExemplarStoreError("records came back without their embeddings")
        for position, item_id in enumerate(returned):
            row = int(item_id[3:])
            record = dict(page_metadatas[position] or {})
            if record.get("row") != row:
                raise ExemplarStoreError(f"record {item_id} carries row {record.get('row')!r}")
            vectors[row] = page_vectors[position]
            documents[row] = page_documents[position]
            metadatas[row] = record
    if compute_records_hash(ids, documents, metadatas) != metadata.get("records_hash"):
        raise ExemplarStoreError("record hash does not match: contents changed since the sync")
    try:
        labels = tuple(tuple(json.loads(str(record["labels"]))) for record in metadatas)
        sources = tuple(str(record["source"]) for record in metadatas)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ExemplarStoreError(f"record metadata is malformed: {error}") from error
    if any(
        len(label) != int(record["label_count"])
        for label, record in zip(labels, metadatas, strict=True)
    ):
        raise ExemplarStoreError("record label counts do not match their labels")
    index = ExemplarIndex(
        embeddings=np.ascontiguousarray(np.asarray(vectors, dtype=np.float32)),
        labels=labels,
        texts=tuple(str(document) for document in documents),
        sources=sources,
        metadata=json.loads(str(metadata["source_metadata_json"])),
    )
    if index.compute_fingerprint() != index.fingerprint:
        raise ExemplarStoreError(
            "fingerprint recomputed from the downloaded vectors and labels does not match"
        )
    return index


# ----- query accounting -----
_TOTALS_LOCK = threading.Lock()
_TOTALS = {"queries": 0, "recall_misses": 0, "search_fallbacks": 0}


def query_totals() -> dict[str, int]:
    """Queries, recall misses and search fallbacks of every Chroma exemplar store in this
    process (the benchmarks' ``--require-exemplar-backend`` reads them at the end of a run)."""
    with _TOTALS_LOCK:
        return dict(_TOTALS)


def reset_query_totals() -> None:
    with _TOTALS_LOCK:
        for key in _TOTALS:
            _TOTALS[key] = 0


def _add_totals(stats: Mapping[str, int]) -> None:
    with _TOTALS_LOCK:
        for key in _TOTALS:
            _TOTALS[key] += int(stats.get(key, 0))


# ----- runtime store -----
@dataclass(eq=False)  # identity semantics: comparing stores would compare vector arrays
class ChromaExemplarStore:
    """The exemplar library served from a verified Chroma collection (see the module notes)."""

    index: ExemplarIndex
    collection: Any
    search: str = "exact"
    overfetch: float = 2.0
    strict: bool = False
    collection_name: str = ""
    collection_id: str = ""
    server_version: str | None = None
    hnsw: dict[str, Any] = field(default_factory=dict)
    records_hash: str = ""
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _rows: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    backend: ClassVar[str] = "chroma"

    def __post_init__(self) -> None:
        if self.search not in SEARCH_MODES:
            raise ValueError(f"unknown exemplar search {self.search!r}")
        self._rows = {record_id(row): row for row in range(len(self.index))}

    # the library itself (retrieval and the acceptance loader read these)
    def __len__(self) -> int:
        return len(self.index)

    @property
    def fingerprint(self) -> str:
        return self.index.fingerprint

    @property
    def metadata(self) -> dict[str, Any]:
        return self.index.metadata

    @property
    def embeddings(self) -> np.ndarray:
        return self.index.embeddings

    @property
    def labels(self) -> tuple[tuple[str, ...], ...]:
        return self.index.labels

    @property
    def texts(self) -> tuple[str, ...]:
        return self.index.texts

    @property
    def sources(self) -> tuple[str, ...]:
        return self.index.sources

    def compute_fingerprint(self) -> str:
        return self.index.compute_fingerprint()

    def compatible_with(
        self, embedding_model: str, embedding_revision: str | None
    ) -> tuple[bool, str]:
        return self.index.compatible_with(embedding_model, embedding_revision)

    @property
    def search_class(self) -> str:
        """``exact`` unless the candidate search can miss part of the window: HNSW with an
        ``ef_search`` below the library size or less than twice the window requested."""
        if self.search == "exact":
            return "exact"
        ef_search = int(self.hnsw.get("ef_search") or 0)
        return "exact" if ef_search >= len(self.index) and self.overfetch >= 2.0 else "approximate"

    def signature(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "collection": self.collection_name,
            "collection_id": self.collection_id,
            "server_version": self.server_version,
            "hnsw": {
                key: self.hnsw.get(key)
                for key in ("space", "max_neighbors", "ef_construction", "ef_search")
            },
            "search": self.search,
            "overfetch": float(self.overfetch),
            "search_class": self.search_class,
            "fingerprint": self.fingerprint,
            "records_hash": self.records_hash,
            "count": len(self),
        }

    @property
    def last_query_stats(self) -> dict[str, int]:
        """Counters of this thread's last ``query_vectors`` call."""
        return dict(
            getattr(self._local, "stats", {"queries": 0, "recall_misses": 0, "search_fallbacks": 0})
        )

    def candidate_rows(self, vectors: np.ndarray, *, k: int) -> list[np.ndarray]:
        """Chroma's nearest records per query (``ceil(window * overfetch)`` of them), as rows."""
        queries = _unit(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        window = window_size(len(self.index), k)
        n_results = min(len(self.index), max(window, math.ceil(window * float(self.overfetch))))
        rows: list[np.ndarray] = []
        for result in batched_query(
            self.collection, queries, n_results=n_results, include=["distances"]
        ):
            for ids in result.get("ids") or []:
                try:
                    rows.append(np.asarray([self._rows[str(item)] for item in ids], dtype=np.int64))
                except KeyError as error:
                    raise RuntimeError(
                        f"Chroma returned {error} which is not in the verified library"
                    ) from error
        if len(rows) != len(queries):
            raise RuntimeError(f"Chroma answered {len(rows)} of {len(queries)} queries")
        return rows

    def query_vectors(
        self,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        *,
        parent_of: Mapping[str, str | None] | None = None,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        duplicate_similarity: float = DEFAULT_DUPLICATE_SIMILARITY,
        vote_norm: str = DEFAULT_VOTE_NORM,
    ) -> list[ExemplarVotes]:
        arguments = {
            "parent_of": parent_of,
            "k": k,
            "tau": tau,
            "duplicate_similarity": duplicate_similarity,
            "vote_norm": vote_norm,
        }
        queries = np.asarray(vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        stats = {"queries": len(queries), "recall_misses": 0, "search_fallbacks": 0}
        allowed: list[np.ndarray] | None = None
        if self.search == "hnsw" and len(queries):
            try:
                allowed = self.candidate_rows(queries, k=k)
            except Exception:
                if self.strict:
                    raise
                stats["search_fallbacks"] = len(queries)
        votes = self.index.query_vectors(queries, allowed_rows=allowed, stats=stats, **arguments)
        self._local.stats = stats
        _add_totals(stats)
        return votes


# ----- loader -----
_VERIFIED: ObjectMemo[ExemplarIndex] = ObjectMemo(capacity=4)


def configured_collection_name(settings: Any) -> tuple[str | None, str]:
    """The configured collection, or the derived name for the npz library; ``(None, reason)``
    when neither is available."""
    name = str(getattr(settings, "technique_exemplar_collection", "") or "").strip()
    if name:
        return name, "configured"
    fingerprint = npz_fingerprint(getattr(settings, "technique_exemplar_index_path", None))
    if fingerprint is None:
        return None, (
            "no TECHNIQUE_EXEMPLAR_COLLECTION configured and no npz artifact to derive it from"
        )
    return (
        exemplar_collection_name(
            fingerprint,
            int(getattr(settings, "technique_exemplar_hnsw_m", 32)),
            int(getattr(settings, "technique_exemplar_hnsw_ef_construction", 400)),
        ),
        "derived from the npz fingerprint",
    )


def load_chroma_exemplar_store(
    settings: Any, client: Any
) -> tuple[ChromaExemplarStore | None, str]:
    """Open and verify the configured exemplar collection: ``(store, "ok")`` or
    ``(None, reason)``. Read-only (``get_collection``, ``count``, ``get``, ``query`` later);
    never raises for an unusable collection."""
    if client is None:
        return None, "no Chroma client (the vector store has none)"
    search = str(getattr(settings, "technique_exemplar_search", "exact") or "exact").lower()
    if search not in SEARCH_MODES:
        return None, f"unknown exemplar search {search!r} (exact or hnsw)"
    overfetch = float(getattr(settings, "technique_exemplar_overfetch", 2.0))
    if not overfetch >= 1.0:
        return None, f"exemplar overfetch {overfetch} is below 1"
    name, how = configured_collection_name(settings)
    if name is None:
        return None, how
    try:
        collection = client.get_collection(name=name, embedding_function=None)
    except Exception as error:  # NotFoundError, connection errors
        return None, f"collection {name} unavailable: {type(error).__name__}: {error}"
    try:
        metadata = dict(collection.metadata or {})
        count = int(collection.count())
        problem = _metadata_problems(
            metadata,
            count=count,
            embedding_model=str(getattr(settings, "embedding_model", "") or ""),
            revision=getattr(settings, "embedding_model_revision", None),
        )
        if problem is not None:
            return None, f"collection {name}: {problem}"
        hnsw = hnsw_configuration(collection)
        if hnsw.get("space") != "cosine":
            return None, f"collection {name} uses space {hnsw.get('space')!r}, not cosine"
        required_ef = int(getattr(settings, "technique_exemplar_hnsw_ef_search", 0) or 0)
        if int(hnsw.get("ef_search") or 0) < required_ef:
            return None, (
                f"collection {name} searches with ef_search {hnsw.get('ef_search')!r}, "
                f"below the configured {required_ef}"
            )
        npz = npz_fingerprint(getattr(settings, "technique_exemplar_index_path", None))
        if npz is not None and npz != metadata["fingerprint"]:
            return None, (
                f"collection {name} holds library {metadata['fingerprint']}, the npz artifact "
                f"is {npz}: resync or point TECHNIQUE_EXEMPLAR_COLLECTION elsewhere"
            )
        stamp = (str(collection.id), name, metadata["fingerprint"], metadata["records_hash"], count)
        index = _VERIFIED.get(client, stamp)
        if index is None:
            index = _VERIFIED.put(client, stamp, read_exemplar_collection(collection, metadata))
    except ExemplarStoreError as error:
        return None, f"collection {name}: {error}"
    except Exception as error:  # the server went away mid-download
        return None, f"collection {name} unreadable: {type(error).__name__}: {error}"
    try:
        server_version = str(client.get_version())
    except Exception:
        server_version = None
    backend = str(getattr(settings, "technique_exemplar_backend", "auto") or "auto").lower()
    store = ChromaExemplarStore(
        index=index,
        collection=collection,
        search=search,
        overfetch=overfetch,
        strict=backend == "chroma",
        collection_name=name,
        collection_id=str(collection.id),
        server_version=server_version,
        hnsw=hnsw,
        records_hash=str(metadata["records_hash"]),
    )
    return store, "ok"


def clear_memo() -> None:
    """Forget verified downloads (tests; a long-lived process after a resync)."""
    _VERIFIED.clear()
