"""Dev-only retrieval tuning harness: the cache, the retrieval stand-ins and the candidate rows.

``scripts/tune_retrieval.py`` (docs 21.3.16, plan section 4) runs three stages on top of this:

Stage A, :func:`build_cache`, the only stage that embeds or reads ChromaDB. Every dev case text,
its retrieval segments and its exemplar query texts are embedded once (``CachedEmbedder``).
``attack_techniques`` is read with ``get_collection``, ``count``, ``get`` and ``query`` only:
Chroma's ANN hits with the runtime field filter at each configured depth, the exact top-N by brute
force over the collection's own embeddings (``get(include=[embeddings])``), and optionally the
ANN hits of a configured collection copy (Phase 5b). The BM25 top-N per segment is stored for
each lexical field set. Everything is keyed by :func:`cache_key`.

Stage B, :func:`build_rowset`: the real ``retrieve_techniques(**retrieval_kwargs(settings))``
with stand-ins: :class:`CachedTechniqueStore` (``.embedder`` is the memo, ``.query_techniques``
replays the cached hits), :class:`CachedLexicalIndex`, and the exemplar library searched exactly.
Rows are extracted as ``scripts/fit_acceptance_model.extract_rows`` does (``acceptance_views``,
``candidate_features``, ``candidate_gate_flags``, ``suppressed_parent_flags``), so harness rows
can only differ from serving where a stand-in differs from what it stands in for (parity P1).

Stage C helpers, :func:`score_rows`: the runtime decision (``runtime_gate_masks``) and the
acceptance ranking (``rank_candidates_by_acceptance``) emulated on stored rows, and the metrics.

Dev only: cases come from the ``dev`` split of tram, expert and derived_procedures; a path naming
CTI-HAL, EvidenceForge, the Elastic posts, the clinic bundle or a test/train/held-out split is
refused (:func:`require_dev_path`), and every loaded case must carry ``split == "dev"``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.acceptance import FEATURE_NAMES, candidate_features
from apt_classifier.evaluation import source_fingerprint
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.exemplars import exemplar_query_text
from apt_classifier.retrieve import (
    LexicalTechniqueIndex,
    _observation_segments,
    acceptance_views,
    candidate_gate_flags,
    context_gate_flags,
    retrieval_kwargs,
    retrieve_techniques,
    suppressed_parent_flags,
)
from apt_classifier.vectorstore import (
    DEFAULT_SYNTHETIC_PATH,
    TECHNIQUE_COLLECTION,
    VectorHit,
    batched_query,
    canonical_json,
    collection_copy_status,
    procedure_sentence_texts,
    records_hash,
)
from scripts.benchmark_mapping import FINGERPRINTED_SOURCES, dataset_of, load_cases
from scripts.exemplar_backend_equivalence import CachedEmbedder
from scripts.run_tooling import REPO_ROOT, file_sha256, git_state, guard_output_paths
from scripts.sync_attack_collection_copy import embeddings_sha256, read_records

CACHE_FORMAT_VERSION = 1
ROWS_FORMAT_VERSION = 1
DEV_SPLIT = "dev"
DEV_DATASETS = ("tram", "expert", "derived_procedures")
REFUSED_MARKERS = ("cti_hal", "cti-hal", "ctihal", "evidenceforge", "elastic", "clinic")
_SPLIT_NAME_RE = re.compile(r"(?:^|[^a-z0-9])(test|train|heldout|holdout)(?:[^a-z0-9]|$)")
DEFAULT_ANN_DEPTHS = (90, 150, 300)
DEFAULT_SYNTHETIC_ANN_DEPTHS = (90,)
DEFAULT_EXACT_DEPTH = 400
DEFAULT_LEXICAL_DEPTH = 400
# the dense field filters retrieve_techniques passes (include_synthetic off / on)
SERVING_FIELDS = ("name", "description", "procedure")
SYNTHETIC_FIELDS = (*SERVING_FIELDS, "synthetic")
FLAG_NAMES = ("benign", "sparse", "hierarchy_parent", "cue_supported", "parent_suppressed")
DENSE_SOURCES = ("ann", "exact", "copy")
TIE_TOLERANCE = 1e-6
SHARED_MODELS = REPO_ROOT / "data" / "models"
# what the cached hits and rows depend on: the benchmark's fingerprinted sources plus the
# sentence splitter (retrieval segments) and the evidence models (observation normalisation)
CACHE_SOURCES = (
    *FINGERPRINTED_SOURCES,
    REPO_ROOT / "apt_classifier" / "attack" / "parse.py",
    REPO_ROOT / "apt_classifier" / "evidence" / "models.py",
)
CACHE_FILES = (
    "vectors.npz",
    "dense_table.npz",
    "lexical_table.npz",
    "collection_embeddings.npy",
    "lists.npz",
    "cases.json",
)


class DevOnlyError(ValueError):
    """An input outside the public dev split was offered to the harness."""


class CacheError(RuntimeError):
    """The cache is missing, damaged or does not match the current inputs."""


class CacheMiss(LookupError):
    """A retrieval call asked for something the cache does not hold (never a silent fallback)."""


# ----- dev-only guards -----
def dev_path_problem(path: str | Path) -> str | None:
    """Why the harness must not read ``path`` (``None`` when it may)."""
    raw = Path(path)
    for text in {str(raw), str(raw.resolve())}:
        lowered = text.casefold()
        for marker in REFUSED_MARKERS:
            if marker in lowered:
                return f"{path}: the tuning harness never reads {marker} data (dev only)"
    if _SPLIT_NAME_RE.search(raw.name.casefold()):
        return (
            f"{path}: names a test, train or held-out split; the tuning harness reads the public "
            "dev split only"
        )
    return None


def require_dev_path(path: str | Path | None, what: str = "path") -> None:
    if path is None:
        return
    problem = dev_path_problem(path)
    if problem:
        raise DevOnlyError(f"{what}: {problem}")


def check_dev_datasets(datasets: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(sorted(set(datasets)))
    unknown = sorted(set(chosen) - set(DEV_DATASETS))
    if unknown or not chosen:
        raise DevOnlyError(
            f"datasets {', '.join(unknown) or '(none)'} refused: the harness tunes on the dev "
            f"split of {', '.join(DEV_DATASETS)} only"
        )
    return chosen


def check_dev_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    """A benchmark artifact the harness compares against must be a dev run of dev datasets."""
    require_dev_path(path, "artifact")
    if payload.get("split") != DEV_SPLIT:
        raise DevOnlyError(f"{path}: split {payload.get('split')!r}, the harness reads dev only")
    check_dev_datasets(payload.get("datasets") or ())


def output_problems(paths: Iterable[Path | str | None], *, overwrite: bool) -> list[str]:
    """Output guard: inside this repository, never the shared models directory (in any checkout),
    an existing file only with ``overwrite``."""
    candidates = [Path(path) for path in paths if path is not None]
    problems = guard_output_paths(candidates, repo_root=REPO_ROOT, overwrite=overwrite)
    shared = SHARED_MODELS.resolve()
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_relative_to(shared) or path.absolute().is_relative_to(SHARED_MODELS):
            problems.append(f"{path} is under data/models: the tuning harness never writes there")
    return problems


# ----- dev cases -----
@dataclass(frozen=True)
class DevCase:
    case_id: str
    dataset: str
    text: str
    source: str
    expected: tuple[str, ...]

    def ledger(self) -> EvidenceLedger:
        """The ledger the fit and the mapping benchmark build for a case."""
        return EvidenceLedger(
            observations=[
                Observation(
                    text=self.text,
                    provenance=Provenance(source="public_benchmark", source_name=self.source),
                )
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "text": self.text,
            "source": self.source,
            "expected": list(self.expected),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DevCase:
        return cls(
            case_id=str(value["case_id"]),
            dataset=str(value["dataset"]),
            text=str(value["text"]),
            source=str(value["source"]),
            expected=tuple(str(item) for item in value["expected"]),
        )


def load_dev_cases(
    root: Path, datasets: Iterable[str], kb: Any
) -> tuple[list[DevCase], dict[str, Any]]:
    """The dev cases exactly as ``fit_acceptance_model.py --split dev`` loads them (indexed
    procedure sentences dropped), refused unless every case is a dev case of a dev dataset."""
    require_dev_path(root, "--root")
    chosen = check_dev_datasets(datasets)
    cases, diagnostics = load_cases(
        Path(root),
        datasets=set(chosen),
        split=DEV_SPLIT,
        indexed_texts=procedure_sentence_texts(kb),
    )
    output: list[DevCase] = []
    for case in cases:
        dataset = dataset_of(case)
        if case.split != DEV_SPLIT or dataset not in chosen:
            raise DevOnlyError(
                f"case {case.case_id}: split {case.split!r}, dataset {dataset!r}; dev only"
            )
        output.append(
            DevCase(
                case_id=case.case_id,
                dataset=dataset,
                text=case.text,
                source=case.source,
                expected=tuple(case.expected_technique_ids),
            )
        )
    if len({case.case_id for case in output}) != len(output):
        raise DevOnlyError("case ids are not unique")
    return output, diagnostics


@dataclass(frozen=True)
class CaseTexts:
    """Every text retrieval embeds or queries for one case."""

    observation: str
    segments: tuple[str, ...]
    exemplar_queries: tuple[str, ...]  # the whole text first, then each segment when segmented


def case_texts(case: DevCase) -> CaseTexts:
    text = str(case.ledger().observations[0].text)
    segments = tuple(_observation_segments(text))
    exemplar = [exemplar_query_text(text)]
    if len(segments) > 1:
        exemplar.extend(exemplar_query_text(str(segment)) for segment in segments)
    return CaseTexts(text, segments, tuple(exemplar))


def text_key(text: str) -> str:
    return CachedEmbedder.key(str(text))


def cases_sha256(cases: Sequence[DevCase]) -> str:
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.case_id):
        digest.update(canonical_json(case.to_dict()).encode())
        digest.update(b"\n")
    return digest.hexdigest()


# ----- hit tables and lists -----
@dataclass
class HitTable:
    """Documents and metadata of every hit a list refers to, by position."""

    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    _position: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._position = {item: index for index, item in enumerate(self.ids)}

    def add(self, item_id: str, document: str, metadata: Mapping[str, Any]) -> int:
        position = self._position.get(item_id)
        if position is not None:
            if self.documents[position] != document or self.metadatas[position] != dict(metadata):
                raise CacheError(
                    f"hit {item_id} came back with another document or metadata than before"
                )
            return position
        self._position[item_id] = len(self.ids)
        self.ids.append(item_id)
        self.documents.append(document)
        self.metadatas.append(dict(metadata))
        return len(self.ids) - 1

    def position(self, item_id: str) -> int | None:
        return self._position.get(item_id)

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            np.savez(
                handle,
                ids=np.asarray(self.ids, dtype=str),
                documents=np.asarray(json.dumps(self.documents, ensure_ascii=False)),
                metadatas=np.asarray(json.dumps(self.metadatas, ensure_ascii=False)),
            )

    @classmethod
    def load(cls, path: Path) -> HitTable:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                ids=[str(item) for item in payload["ids"].tolist()],
                documents=list(json.loads(str(payload["documents"]))),
                metadatas=[dict(item) for item in json.loads(str(payload["metadatas"]))],
            )


@dataclass
class HitList:
    """Cached hits of one (kind, source, fields, depth): per query text key, table positions
    and a value (the Chroma distance for dense hits, the BM25 score for lexical ones)."""

    kind: str
    source: str
    fields: tuple[str, ...]
    depth: int
    entries: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.kind}|{self.source}|{','.join(self.fields)}|{self.depth}"

    def spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "fields": list(self.fields),
            "depth": self.depth,
        }


def _save_lists(path: Path, lists: Sequence[HitList]) -> None:
    payload: dict[str, np.ndarray] = {}
    for number, hit_list in enumerate(lists):
        keys = sorted(hit_list.entries)
        lengths = [len(hit_list.entries[key][0]) for key in keys]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        index = (
            np.concatenate([hit_list.entries[key][0] for key in keys]).astype(np.int64)
            if keys
            else np.zeros(0, dtype=np.int64)
        )
        value = (
            np.concatenate([hit_list.entries[key][1] for key in keys]).astype(np.float64)
            if keys
            else np.zeros(0, dtype=np.float64)
        )
        payload[f"l{number}_keys"] = np.asarray(keys, dtype=str)
        payload[f"l{number}_offsets"] = offsets
        payload[f"l{number}_index"] = index
        payload[f"l{number}_value"] = value
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def _load_lists(path: Path, specs: Sequence[Mapping[str, Any]]) -> list[HitList]:
    output: list[HitList] = []
    with np.load(path, allow_pickle=False) as payload:
        for number, spec in enumerate(specs):
            keys = [str(item) for item in payload[f"l{number}_keys"].tolist()]
            offsets = payload[f"l{number}_offsets"]
            index = payload[f"l{number}_index"]
            value = payload[f"l{number}_value"]
            hit_list = HitList(
                str(spec["kind"]), str(spec["source"]), tuple(spec["fields"]), int(spec["depth"])
            )
            for position, key in enumerate(keys):
                start, end = int(offsets[position]), int(offsets[position + 1])
                hit_list.entries[key] = (index[start:end].copy(), value[start:end].copy())
            output.append(hit_list)
    return output


def _where(fields: Sequence[str]) -> dict[str, Any]:
    """The field filter ``ChromaVectorStore.query_techniques`` sends."""
    return {"field": fields[0]} if len(fields) == 1 else {"field": {"$in": list(fields)}}


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _ann_list(
    collection: Any,
    vectors: Sequence[Sequence[float]],
    keys: Sequence[str],
    table: HitTable,
    *,
    source: str,
    fields: tuple[str, ...],
    depth: int,
) -> HitList:
    """Chroma's hits as ``ChromaVectorStore._query`` reads them (``limit = min(depth, count)``,
    distance as returned, ``similarity = 1 - distance`` later), checked against the records."""
    count = int(collection.count())
    limit = min(max(int(depth), 1), count)
    hit_list = HitList("dense", source, fields, int(depth))
    results = batched_query(collection, list(vectors), n_results=limit, where=_where(fields))
    position = 0
    for result in results:
        documents_all = result.get("documents") or []
        metadatas_all = result.get("metadatas") or []
        distances_all = result.get("distances") or []
        for query_index, ids in enumerate(result.get("ids") or []):
            documents = (documents_all[query_index] if documents_all else None) or []
            metadatas = (metadatas_all[query_index] if metadatas_all else None) or []
            distances = (distances_all[query_index] if distances_all else None) or []
            index: list[int] = []
            values: list[float] = []
            for hit, item_id in enumerate(ids):
                at = table.position(str(item_id))
                if at is None:
                    raise CacheError(f"{source} returned {item_id}, which get() did not return")
                if table.documents[at] != str(documents[hit]) or table.metadatas[at] != dict(
                    metadatas[hit] or {}
                ):
                    raise CacheError(
                        f"{source} returned {item_id} with another document or metadata than get()"
                    )
                index.append(at)
                values.append(float(distances[hit]))
            hit_list.entries[keys[position]] = (
                np.asarray(index, dtype=np.int64),
                np.asarray(values, dtype=np.float64),
            )
            position += 1
    if position != len(keys):
        raise CacheError(f"{source} answered {position} of {len(keys)} queries")
    return hit_list


def _exact_list(
    records: Mapping[str, Any],
    vectors: np.ndarray,
    keys: Sequence[str],
    *,
    fields: tuple[str, ...],
    depth: int,
    block: int = 256,
) -> HitList:
    """Brute-force top-``depth`` over the records whose field is in ``fields`` (float64 dot of
    float32 unit vectors; ties by record order). The stored value is the distance Chroma would
    report, ``float32(1 - similarity)``."""
    mask = np.asarray(
        [(metadata or {}).get("field") in set(fields) for metadata in records["metadatas"]],
        dtype=bool,
    )
    rows = np.flatnonzero(mask)
    hit_list = HitList("dense", "exact", fields, int(depth))
    matrix = _unit(np.asarray(records["embeddings"], dtype=np.float32)[rows]).astype(np.float64)
    queries = _unit(np.asarray(vectors, dtype=np.float32)).astype(np.float64)
    keep = min(int(depth), len(rows))
    for start in range(0, len(queries), block):
        similarities = queries[start : start + block] @ matrix.T
        for offset, row in enumerate(similarities):
            if keep < len(row):
                top = np.sort(np.argpartition(-row, keep - 1)[:keep])
            else:
                top = np.arange(len(row))
            order = top[np.argsort(-row[top], kind="stable")]
            distances = (1.0 - row[order]).astype(np.float32).astype(np.float64)
            hit_list.entries[keys[start + offset]] = (rows[order].astype(np.int64), distances)
    return hit_list


def _lexical_list(
    index: LexicalTechniqueIndex,
    texts: Sequence[str],
    keys: Sequence[str],
    table: HitTable,
    *,
    depth: int,
) -> HitList:
    hit_list = HitList("lexical", "bm25", tuple(index.fields), int(depth))
    for text, key in zip(texts, keys, strict=True):
        hits = index.query(text, n_results=int(depth))
        hit_list.entries[key] = (
            np.asarray(
                [table.add(hit.id, hit.document, hit.metadata) for hit in hits], dtype=np.int64
            ),
            np.asarray([float(hit.similarity) for hit in hits], dtype=np.float64),
        )
    return hit_list


# ----- cache key -----
def cache_key(material: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(material)).encode()).hexdigest()


def collection_identity(collection: Any, records: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(collection.name),
        "id": str(collection.id),
        "metadata": canonical_json(dict(collection.metadata or {})),
        "count": int(collection.count()),
        "embeddings_sha256": embeddings_sha256(records["embeddings"]),
        "records_hash": records_hash(records["ids"], records["documents"], records["metadatas"]),
    }


def exemplar_identity(source: Any) -> dict[str, Any]:
    """The exemplar library the votes are read from: its fingerprint (for a Chroma library the
    one recomputed from the vectors Chroma returns), backend and the npz it was copied from."""
    return {
        "fingerprint": str(getattr(source, "fingerprint", "") or ""),
        "count": len(source) if source is not None else 0,
        "records_hash": getattr(source, "records_hash", None) or None,
        "backend": str(getattr(source, "backend", "npz") or "npz") if source is not None else None,
        "source_fingerprint": getattr(source, "source_fingerprint", None) or None,
    }


def input_identity(settings: Any) -> dict[str, Any]:
    """The inputs besides Chroma and the library that the cached hits depend on."""
    return {
        "embedding_model": str(getattr(settings, "embedding_model", "")),
        "embedding_revision": getattr(settings, "embedding_model_revision", None),
        "kb_sha256": file_sha256(getattr(settings, "kb_path", None)),
        "synthetic_sha256": file_sha256(DEFAULT_SYNTHETIC_PATH),
        "source_fingerprint": source_fingerprint(CACHE_SOURCES),
    }


def key_material(
    *,
    cases: Sequence[DevCase],
    text_keys: Iterable[str],
    inputs: Mapping[str, Any],
    collection: Mapping[str, Any],
    copy: Mapping[str, Any] | None,
    exemplars: Mapping[str, Any],
    lists: Sequence[Mapping[str, Any]],
    git_commit: str | None,
) -> dict[str, Any]:
    """What the cache key covers (plan section 4, Stage A item 5)."""
    digest = hashlib.sha256()
    for key in sorted(set(text_keys)):
        digest.update(key.encode())
    return {
        "format": CACHE_FORMAT_VERSION,
        "cases_sha256": cases_sha256(cases),
        "case_count": len(cases),
        "texts_sha256": digest.hexdigest(),
        **dict(inputs),
        "technique_collection": dict(collection),
        "collection_copy": dict(copy) if copy else None,
        "exemplars": dict(exemplars),
        "lists": [dict(item) for item in lists],
        "git_commit": git_commit,
    }


# ----- Stage A -----
def build_cache(
    output_dir: Path,
    *,
    settings: Any,
    kb: Any,
    cases: Sequence[DevCase],
    client: Any,
    embedder_factory: Callable[[], Any] | None,
    exemplar_source: Any,
    ann_depths: Sequence[int] = DEFAULT_ANN_DEPTHS,
    synthetic_ann_depths: Sequence[int] = DEFAULT_SYNTHETIC_ANN_DEPTHS,
    exact_depth: int = DEFAULT_EXACT_DEPTH,
    lexical_depth: int = DEFAULT_LEXICAL_DEPTH,
    copy_collection: str | None = None,
    notes: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> DevCache:
    """Build the dev cache in ``output_dir`` (which must not hold one yet). Read-only towards
    Chroma; the technique collection's metadata is checked as ``query_techniques`` checks it."""
    from apt_classifier.vectorstore import ChromaVectorStore

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CacheError(
            f"{output_dir} is not empty (a cache, or an unfinished build without manifest.json); "
            "choose a new directory"
        )
    texts = {case.case_id: case_texts(case) for case in cases}
    segments = sorted({segment for item in texts.values() for segment in item.segments})
    to_embed = sorted(
        {
            *(case.text for case in cases),
            *segments,
            *(query for item in texts.values() for query in item.exemplar_queries),
        }
    )
    embedder = CachedEmbedder(embedder_factory, None)
    log(f"embedding {len(to_embed)} texts ({len(segments)} retrieval segments)")
    embedder.embed(to_embed)
    segment_keys = [text_key(segment) for segment in segments]
    segment_vectors = embedder.embed(segments)

    collection = client.get_collection(name=TECHNIQUE_COLLECTION, embedding_function=None)
    # the same compatibility check query_techniques runs (reads count and metadata only)
    ChromaVectorStore(settings, embedder, client=client)._validate_technique_collection(collection)
    log(f"reading {TECHNIQUE_COLLECTION} ({collection.count()} records)")
    records = read_records(collection)
    dense_table = HitTable(
        ids=list(records["ids"]),
        documents=[str(document) for document in records["documents"]],
        metadatas=[dict(metadata or {}) for metadata in records["metadatas"]],
    )
    lists: list[HitList] = []
    plan = [(SERVING_FIELDS, tuple(ann_depths)), (SYNTHETIC_FIELDS, tuple(synthetic_ann_depths))]
    for fields, depths in plan:
        for depth in depths:
            log(f"ANN {','.join(fields)} at {depth}")
            lists.append(
                _ann_list(
                    collection,
                    segment_vectors,
                    segment_keys,
                    dense_table,
                    source="ann",
                    fields=fields,
                    depth=depth,
                )
            )
        log(f"exact {','.join(fields)} top {exact_depth}")
        lists.append(
            _exact_list(
                records,
                np.asarray(segment_vectors, dtype=np.float32),
                segment_keys,
                fields=fields,
                depth=exact_depth,
            )
        )
    copy_identity: dict[str, Any] | None = None
    if copy_collection:
        copy = client.get_collection(name=copy_collection, embedding_function=None)
        ok, reason = collection_copy_status(copy, collection)
        if not ok:
            raise CacheError(f"collection copy {copy_collection}: {reason}")
        copy_identity = {
            "name": str(copy.name),
            "id": str(copy.id),
            "metadata": canonical_json(dict(copy.metadata or {})),
            "count": int(copy.count()),
        }
        for fields, depths in plan:
            for depth in depths:
                log(f"copy ANN {','.join(fields)} at {depth}")
                lists.append(
                    _ann_list(
                        copy,
                        segment_vectors,
                        segment_keys,
                        dense_table,
                        source="copy",
                        fields=fields,
                        depth=depth,
                    )
                )
    lexical_table = HitTable()
    for fields in (SERVING_FIELDS, SYNTHETIC_FIELDS):
        log(f"BM25 {','.join(fields)} top {lexical_depth}")
        index = LexicalTechniqueIndex.from_kb(kb, fields=fields)
        lists.append(
            _lexical_list(index, segments, segment_keys, lexical_table, depth=lexical_depth)
        )
    material = key_material(
        cases=cases,
        text_keys=[text_key(text) for text in to_embed],
        inputs=input_identity(settings),
        collection=collection_identity(collection, records),
        copy=copy_identity,
        exemplars=exemplar_identity(exemplar_source),
        lists=[hit_list.spec() for hit_list in lists],
        git_commit=git_state(REPO_ROOT).get("commit"),
    )
    # the manifest is written last: a directory without one is an unfinished build
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder.path = output_dir / "vectors.npz"
    embedder.save()
    dense_table.save(output_dir / "dense_table.npz")
    lexical_table.save(output_dir / "lexical_table.npz")
    with (output_dir / "collection_embeddings.npy").open("wb") as handle:
        np.save(handle, np.ascontiguousarray(records["embeddings"], dtype=np.float32))
    _save_lists(output_dir / "lists.npz", lists)
    (output_dir / "cases.json").write_text(
        json.dumps([case.to_dict() for case in cases], ensure_ascii=False), encoding="utf-8"
    )
    files = {name: file_sha256(output_dir / name) for name in CACHE_FILES}
    manifest = {
        "kind": "tune_retrieval_cache",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "key": cache_key(material),
        "key_material": material,
        "files": files,
        "lists": [hit_list.spec() for hit_list in lists],
        "collection_fields": [str(item.get("field") or "") for item in dense_table.metadatas],
        "dense_table_size": len(dense_table.ids),
        "segments": len(segments),
        "texts": len(to_embed),
        "dev_only": {"split": DEV_SPLIT, "datasets": sorted({case.dataset for case in cases})},
        "notes": dict(notes or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return DevCache.load(output_dir)


# ----- the cache at use time -----
class DevCache:
    """A loaded, integrity-checked dev cache (:func:`build_cache`)."""

    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.cases = [
            DevCase.from_dict(item)
            for item in json.loads((self.path / "cases.json").read_text(encoding="utf-8"))
        ]
        for case in self.cases:
            if case.dataset not in DEV_DATASETS:
                raise DevOnlyError(f"cached case {case.case_id} is from {case.dataset}")
        self.embedder = CachedEmbedder(None, self.path / "vectors.npz")
        self.dense_table = HitTable.load(self.path / "dense_table.npz")
        self.lexical_table = HitTable.load(self.path / "lexical_table.npz")
        self.lists = {
            hit_list.name: hit_list
            for hit_list in _load_lists(self.path / "lists.npz", manifest["lists"])
        }

    @classmethod
    def load(cls, path: str | Path) -> DevCache:
        directory = Path(path)
        require_dev_path(directory, "--cache")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise CacheError(f"{directory} holds no cache (run: tune_retrieval.py cache)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "tune_retrieval_cache":
            raise CacheError(f"{directory}: not a tuning cache")
        material = manifest.get("key_material") or {}
        if int(material.get("format", 0)) != CACHE_FORMAT_VERSION:
            raise CacheError(
                f"{directory}: cache format {material.get('format')!r} is not supported"
            )
        if cache_key(material) != manifest.get("key"):
            raise CacheError(f"{directory}: the manifest key does not match its key material")
        for name, digest in (manifest.get("files") or {}).items():
            if file_sha256(directory / name) != digest:
                raise CacheError(f"{directory}/{name} changed since the cache was built")
        return cls(directory, manifest)

    @property
    def key(self) -> str:
        return str(self.manifest["key"])

    @property
    def material(self) -> dict[str, Any]:
        return dict(self.manifest["key_material"])

    @property
    def collection_name(self) -> str:
        return str(self.material["technique_collection"]["name"])

    @property
    def copy_name(self) -> str | None:
        copy = self.material.get("collection_copy")
        return str(copy["name"]) if copy else None

    def dense_sources(self) -> set[str]:
        return {hit_list.source for hit_list in self.lists.values() if hit_list.kind == "dense"}

    def ann_depths(self, fields: Sequence[str] = SERVING_FIELDS, source: str = "ann") -> list[int]:
        return sorted(
            hit_list.depth
            for hit_list in self.lists.values()
            if hit_list.kind == "dense"
            and hit_list.source == source
            and hit_list.fields == tuple(fields)
        )

    def hit_list(self, kind: str, source: str, fields: Sequence[str], depth: int) -> HitList:
        name = f"{kind}|{source}|{','.join(fields)}|{int(depth)}"
        hit_list = self.lists.get(name)
        if hit_list is None:
            raise CacheMiss(f"the cache holds no {name} list")
        return hit_list

    def _exact_or_lexical(self, kind: str, source: str, fields: tuple[str, ...]) -> HitList:
        found = [
            hit_list
            for hit_list in self.lists.values()
            if hit_list.kind == kind and hit_list.source == source and hit_list.fields == fields
        ]
        if not found:
            raise CacheMiss(f"the cache holds no {kind} {source} list for {','.join(fields)}")
        return max(found, key=lambda item: item.depth)

    def dense_hits(
        self, text: str, *, source: str, fields: tuple[str, ...], depth: int
    ) -> list[VectorHit]:
        key = text_key(text)
        if source == "exact":
            hit_list = self._exact_or_lexical("dense", "exact", tuple(fields))
            entry = hit_list.entries.get(key)
            if entry is None:
                raise CacheMiss(f"no exact hits cached for {text[:60]!r}")
            limit = min(max(int(depth), 1), len(self.dense_table.ids))
            if limit > hit_list.depth and len(entry[0]) >= hit_list.depth:
                raise CacheMiss(f"exact depth {depth} exceeds the cached {hit_list.depth}")
            index, values = entry[0][:limit], entry[1][:limit]
        else:
            hit_list = self.hit_list("dense", source, fields, depth)
            entry = hit_list.entries.get(key)
            if entry is None:
                raise CacheMiss(f"no {source} hits cached for {text[:60]!r}")
            index, values = entry
        table = self.dense_table
        return [
            VectorHit(
                id=table.ids[position],
                document=table.documents[position],
                metadata=dict(table.metadatas[position]),
                distance=float(distance),
                similarity=1.0 - float(distance),
            )
            for position, distance in zip(index.tolist(), values.tolist(), strict=True)
        ]

    def lexical_hits(
        self, text: str, *, fields: tuple[str, ...], n_results: int
    ) -> list[VectorHit]:
        hit_list = self._exact_or_lexical("lexical", "bm25", tuple(fields))
        entry = hit_list.entries.get(text_key(text))
        if entry is None:
            raise CacheMiss(f"no BM25 hits cached for {text[:60]!r}")
        limit = max(1, int(n_results))
        if limit > hit_list.depth and len(entry[0]) >= hit_list.depth:
            raise CacheMiss(f"BM25 depth {n_results} exceeds the cached {hit_list.depth}")
        table = self.lexical_table
        return [
            VectorHit(
                id=table.ids[position],
                document=table.documents[position],
                metadata=dict(table.metadatas[position]),
                distance=0.0,
                similarity=float(score),
            )
            for position, score in zip(
                entry[0][:limit].tolist(), entry[1][:limit].tolist(), strict=True
            )
        ]

    def technique_store(self, source: str = "ann") -> CachedTechniqueStore:
        return CachedTechniqueStore(self, source)

    def lexical_index(self, fields: Sequence[str]) -> CachedLexicalIndex:
        return CachedLexicalIndex(self, tuple(fields))

    def case_vectors(self) -> np.ndarray:
        """The vectors of the case texts (what the fit's near-duplicate grouping embeds)."""
        return np.asarray(self.embedder.embed([case.text for case in self.cases]), dtype=np.float32)

    def collection_embeddings(self) -> np.ndarray:
        return np.load(self.path / "collection_embeddings.npy", mmap_mode="r")

    def staleness_problems(self, *, settings: Any, exemplar_source: Any) -> list[str]:
        """Inputs that changed since the cache was built (the git commit may differ: only the
        sources the hits depend on are compared)."""
        material = self.material
        current = input_identity(settings)
        problems = [
            f"{key} is {current[key]!r}, the cache was built with {material.get(key)!r}"
            for key in current
            if material.get(key) != current[key]
        ]
        cached = material.get("exemplars") or {}
        now = exemplar_identity(exemplar_source)
        if cached.get("fingerprint") != now["fingerprint"]:
            problems.append(
                f"exemplar library {now['fingerprint']!r}, the cache was built with "
                f"{cached.get('fingerprint')!r}"
            )
        if (
            cached.get("records_hash")
            and now["records_hash"]
            and (cached["records_hash"] != now["records_hash"])
        ):
            problems.append("exemplar collection records_hash differs from the cache")
        return problems


class CachedTechniqueStore:
    """Stand-in for ``ChromaVectorStore`` in ``retrieve_techniques``: ``.embedder`` is the memo
    (the exemplar query reads it), ``.query_techniques`` replays the cached dense hits. Group
    queries feed attribution signals, never the technique candidates: group behaviours answer
    with no hits (``retrieve_techniques`` zips one list per observation) and there is no
    ``query_group_profile`` (skipped when absent)."""

    def __init__(self, cache: DevCache, source: str = "ann") -> None:
        if source not in DENSE_SOURCES:
            raise ValueError(f"unknown dense source {source!r}")
        if source not in cache.dense_sources():
            raise CacheMiss(f"the cache holds no {source} dense hits")
        self.cache = cache
        self.source = source
        self.embedder = cache.embedder
        self.technique_collection_name = (
            cache.copy_name if source == "copy" else cache.collection_name
        ) or TECHNIQUE_COLLECTION

    def query_techniques(
        self, texts: list[str], *, n_results: int = 10, fields: tuple[str, ...] | None = None
    ) -> list[list[VectorHit]]:
        if not texts:
            return []
        if fields is None:
            raise CacheMiss("unfiltered (legacy_dense) dense queries are not cached")
        return [
            self.cache.dense_hits(
                str(text), source=self.source, fields=tuple(fields), depth=int(n_results)
            )
            for text in texts
        ]

    def query_group_behaviors(self, texts: list[str], *, n_results: int = 20) -> list[list]:
        return [[] for _ in texts]


@dataclass(frozen=True, eq=False)
class CachedLexicalIndex:
    """Stand-in for ``LexicalTechniqueIndex``: the cached BM25 top-N, sliced to ``n_results``
    (a prefix of the real ranking, which sorts by score then document id)."""

    cache: DevCache
    fields: tuple[str, ...]

    def query(self, text: str, *, n_results: int) -> list[VectorHit]:
        return self.cache.lexical_hits(str(text), fields=self.fields, n_results=n_results)


# ----- Stage B -----
def candidate_rows(
    observation: Any, candidates: Sequence[Any], *, segment_acceptance: bool, threshold: float
) -> list[tuple[Any, dict[str, float], dict[str, bool]]]:
    """``(original candidate, features, gate flags)`` per judged candidate, exactly as
    ``fit_acceptance_model.extract_rows`` extracts them (without priors)."""
    candidates = list(candidates)
    suppressed = {
        id(candidate): flag
        for candidate, flag in zip(candidates, suppressed_parent_flags(candidates), strict=True)
    }
    output = []
    for view, views, group in acceptance_views(
        observation, candidates, segment_acceptance=segment_acceptance
    ):
        context = context_gate_flags(str(view.text))
        for candidate, original in zip(views, group, strict=True):
            features = candidate_features(
                candidate,
                views,
                benign_context=context["benign"],
                sparse_context=context["sparse"],
            )
            flags = candidate_gate_flags(
                original, str(view.text), threshold=threshold, context=context
            )
            flags["parent_suppressed"] = suppressed[id(original)]
            output.append((original, features, flags))
    return output


FEATURE_INDEX = {name: position for position, name in enumerate(FEATURE_NAMES)}


class RowView(Mapping[str, float]):
    """One feature row as a mapping (``feature_vector`` reads it with ``.get``)."""

    __slots__ = ("_matrix", "_row")

    def __init__(self, matrix: np.ndarray, row: int) -> None:
        self._matrix = matrix
        self._row = int(row)

    def __getitem__(self, name: str) -> float:
        return float(self._matrix[self._row, FEATURE_INDEX[name]])

    def __iter__(self):
        return iter(FEATURE_NAMES)

    def __len__(self) -> int:
        return len(FEATURE_NAMES)


@dataclass
class RowSet:
    """Candidate rows of every case for one retrieval configuration and cue mode."""

    case_ids: tuple[str, ...]
    datasets: tuple[str, ...]
    expected: tuple[tuple[str, ...], ...]
    case_index: np.ndarray
    technique_ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    flags: dict[str, np.ndarray]
    fused_score: np.ndarray
    fused_rank: np.ndarray
    dense_present: np.ndarray
    generic: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @cached_property
    def expected_counts(self) -> dict[str, int]:
        return {
            case_id: len(set(expected))
            for case_id, expected in zip(self.case_ids, self.expected, strict=True)
        }

    @cached_property
    def expected_array(self) -> np.ndarray:
        return np.asarray([len(set(item)) for item in self.expected], dtype=np.int64)

    @cached_property
    def row_case_ids(self) -> list[str]:
        return [self.case_ids[index] for index in self.case_index.tolist()]

    @cached_property
    def feature_dicts(self) -> list[RowView]:
        """One read-only mapping per row (what ``AcceptanceModel.fit`` / ``.probability`` read
        through ``feature_vector``), backed by the feature matrix instead of 22k dicts."""
        matrix = np.ascontiguousarray(self.features, dtype=np.float64)
        return [RowView(matrix, row) for row in range(len(matrix))]

    def flag_dicts(self) -> list[dict[str, bool]]:
        return [
            {name: bool(self.flags[name][row]) for name in FLAG_NAMES} for row in range(len(self))
        ]

    def gate_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """``runtime_gate_masks`` vectorised: ``(blocked, forced)``."""
        never = self.flags["hierarchy_parent"] | self.flags["parent_suppressed"]
        blocked = never | self.flags["benign"] | self.flags["sparse"]
        forced = ~never & self.flags["cue_supported"]
        return blocked, forced

    @cached_property
    def tid_rank(self) -> np.ndarray:
        """Lexicographic rank of each row's technique id (the ranking's last tie-break)."""
        _, inverse = np.unique(self.technique_ids, return_inverse=True)
        return inverse.astype(np.int64)

    @cached_property
    def pairs(self) -> dict[str, np.ndarray]:
        """Unique (case, technique) pairs: the accepted set is a set per case."""
        technique_count = int(self.tid_rank.max()) + 1 if len(self) else 1
        codes = self.case_index.astype(np.int64) * technique_count + self.tid_rank
        unique, inverse = np.unique(codes, return_inverse=True)
        truth = np.zeros(len(unique), dtype=bool)
        np.logical_or.at(truth, inverse, self.labels == 1)
        generic = np.zeros(len(unique), dtype=bool)
        np.logical_or.at(generic, inverse, self.generic)
        return {
            "inverse": inverse,
            "case": (unique // technique_count).astype(np.int64),
            "truth": truth,
            "generic": generic,
        }

    def case_rows(self, case: int) -> np.ndarray:
        return np.flatnonzero(self.case_index == case)

    def subset(self, cases: Sequence[int] | np.ndarray) -> tuple[RowSet, np.ndarray]:
        """The rows of the given cases (in that order, cases renumbered from 0) and the
        positions of those rows in this set."""
        chosen = [int(case) for case in cases]
        renumber = np.full(self.case_count, -1, dtype=np.int64)
        renumber[chosen] = np.arange(len(chosen))
        keep = renumber[self.case_index] >= 0
        rows = np.flatnonzero(keep)
        rows = rows[np.argsort(renumber[self.case_index[rows]], kind="stable")]
        return RowSet(
            case_ids=tuple(self.case_ids[case] for case in chosen),
            datasets=tuple(self.datasets[case] for case in chosen),
            expected=tuple(self.expected[case] for case in chosen),
            case_index=renumber[self.case_index[rows]],
            technique_ids=self.technique_ids[rows],
            features=self.features[rows],
            labels=self.labels[rows],
            flags={name: values[rows] for name, values in self.flags.items()},
            fused_score=self.fused_score[rows],
            fused_rank=self.fused_rank[rows],
            dense_present=self.dense_present[rows],
            generic=self.generic[rows],
            meta=dict(self.meta),
        ), rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                case_ids=np.asarray(self.case_ids, dtype=str),
                datasets=np.asarray(self.datasets, dtype=str),
                expected=np.asarray(json.dumps([list(item) for item in self.expected])),
                case_index=self.case_index,
                technique_ids=np.asarray(self.technique_ids, dtype=str),
                features=self.features,
                labels=self.labels,
                fused_score=self.fused_score,
                fused_rank=self.fused_rank,
                dense_present=self.dense_present,
                generic=self.generic,
                meta=np.asarray(json.dumps(self.meta, sort_keys=True, default=str)),
                **{f"flag_{name}": self.flags[name] for name in FLAG_NAMES},
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> RowSet:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                case_ids=tuple(str(item) for item in payload["case_ids"].tolist()),
                datasets=tuple(str(item) for item in payload["datasets"].tolist()),
                expected=tuple(tuple(item) for item in json.loads(str(payload["expected"]))),
                case_index=payload["case_index"].astype(np.int64),
                technique_ids=payload["technique_ids"].astype(str),
                features=payload["features"].astype(np.float64),
                labels=payload["labels"].astype(np.int8),
                flags={name: payload[f"flag_{name}"].astype(bool) for name in FLAG_NAMES},
                fused_score=payload["fused_score"].astype(np.float64),
                fused_rank=payload["fused_rank"].astype(np.int64),
                dense_present=payload["dense_present"].astype(bool),
                generic=payload["generic"].astype(bool),
                meta=json.loads(str(payload["meta"])),
            )


class _RowBuilder:
    def __init__(self, cases: Sequence[DevCase], generic: frozenset[str]) -> None:
        self.cases = list(cases)
        self.generic = generic
        self.case_index: list[int] = []
        self.technique_ids: list[str] = []
        self.features: list[np.ndarray] = []
        self.labels: list[int] = []
        self.flags: dict[str, list[bool]] = {name: [] for name in FLAG_NAMES}
        self.fused_score: list[float] = []
        self.fused_rank: list[int] = []
        self.dense_present: list[bool] = []

    def add_case(self, position: int, rows: Sequence[tuple[Any, dict, dict]]) -> None:
        expected = set(self.cases[position].expected)
        for original, features, flags in rows:
            self.case_index.append(position)
            self.technique_ids.append(str(original.technique_id))
            self.features.append(
                np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES])
            )
            self.labels.append(int(original.technique_id in expected))
            for name in FLAG_NAMES:
                self.flags[name].append(bool(flags[name]))
            self.fused_score.append(float(original.fused_score))
            self.fused_rank.append(int(original.fused_rank))
            self.dense_present.append(original.evidence_for("dense") is not None)

    def build(self, meta: dict[str, Any]) -> RowSet:
        return RowSet(
            case_ids=tuple(case.case_id for case in self.cases),
            datasets=tuple(case.dataset for case in self.cases),
            expected=tuple(tuple(case.expected) for case in self.cases),
            case_index=np.asarray(self.case_index, dtype=np.int64),
            technique_ids=np.asarray(self.technique_ids, dtype=str),
            features=np.vstack(self.features)
            if self.features
            else np.zeros((0, len(FEATURE_NAMES))),
            labels=np.asarray(self.labels, dtype=np.int8),
            flags={name: np.asarray(values, dtype=bool) for name, values in self.flags.items()},
            fused_score=np.asarray(self.fused_score, dtype=np.float64),
            fused_rank=np.asarray(self.fused_rank, dtype=np.int64),
            dense_present=np.asarray(self.dense_present, dtype=bool),
            generic=np.asarray(
                [technique_id in self.generic for technique_id in self.technique_ids], dtype=bool
            ),
            meta=meta,
        )


def retrieval_arguments(
    settings: Any, *, exemplar_index: Any, lexical_index: Any, cues_enabled: bool
) -> dict[str, Any]:
    """``retrieval_kwargs`` plus what the fit adds (no model; cues per run)."""
    return {
        **retrieval_kwargs(settings, exemplar_index=exemplar_index, lexical_index=lexical_index),
        "acceptance_model": None,
        "cues_enabled": bool(cues_enabled),
    }


def build_rowset(
    cases: Sequence[DevCase],
    *,
    kb: Any,
    settings: Any,
    store: Any,
    lexical_index: Any,
    exemplar_index: Any,
    cues_enabled: bool,
    generic: frozenset[str] = frozenset(),
    meta: dict[str, Any] | None = None,
) -> RowSet:
    """Stage B: the real ``retrieve_techniques`` per case (``store``/``lexical_index`` may be
    the cached stand-ins or the real ones) and the fit's row extraction."""
    kwargs = retrieval_arguments(
        settings,
        exemplar_index=exemplar_index,
        lexical_index=lexical_index,
        cues_enabled=cues_enabled,
    )
    builder = _RowBuilder(cases, generic)
    for position, case in enumerate(cases):
        ledger = case.ledger()
        result = retrieve_techniques(ledger, store, kb, **kwargs)
        builder.add_case(
            position,
            candidate_rows(
                ledger.observations[0],
                result.candidates,
                segment_acceptance=bool(kwargs["segment_acceptance"]),
                threshold=float(kwargs["threshold"]),
            ),
        )
    return builder.build(dict(meta or {}, cues_enabled=bool(cues_enabled)))


def rowset_differences(left: RowSet, right: RowSet, *, limit: int = 20) -> dict[str, Any]:
    """Per case, whether two row sets are bit-identical (technique order, features as IEEE
    bits, labels, gate flags, fused ranks and scores)."""
    if left.case_ids != right.case_ids:
        return {"cases": left.case_count, "differing_cases": -1, "reason": "different cases"}
    differing: list[dict[str, Any]] = []
    for case in range(left.case_count):
        a, b = left.case_rows(case), right.case_rows(case)
        fields: list[str] = []
        if len(a) != len(b) or not np.array_equal(left.technique_ids[a], right.technique_ids[b]):
            fields.append("technique_ids")
        else:
            if not np.array_equal(
                left.features[a].view(np.uint64), right.features[b].view(np.uint64)
            ):
                fields.append("features")
            if not np.array_equal(left.labels[a], right.labels[b]):
                fields.append("labels")
            for name in FLAG_NAMES:
                if not np.array_equal(left.flags[name][a], right.flags[name][b]):
                    fields.append(f"flag:{name}")
            if not np.array_equal(left.fused_rank[a], right.fused_rank[b]) or not np.array_equal(
                left.fused_score[a].view(np.uint64), right.fused_score[b].view(np.uint64)
            ):
                fields.append("fusion")
        if fields:
            differing.append({"case_id": left.case_ids[case], "fields": fields})
    return {
        "cases": left.case_count,
        "rows": [len(left), len(right)],
        "differing_cases": len(differing),
        "identical_share": 1.0 - len(differing) / max(left.case_count, 1),
        "examples": differing[:limit],
    }


# ----- Stage C helpers: the runtime decision, the ranking and the metrics -----
def generic_techniques(kb: Any, min_groups: int = 30) -> frozenset[str]:
    """Techniques documented for at least ``min_groups`` ATT&CK groups (no labels involved)."""
    counts: dict[str, int] = {}
    for group in (getattr(kb, "groups", {}) or {}).values():
        for technique_id in set(getattr(group, "technique_ids", ()) or ()):
            counts[technique_id] = counts.get(technique_id, 0) + 1
    return frozenset(tid for tid, count in counts.items() if count >= int(min_groups))


def accepted_rows(
    rows: RowSet, probabilities: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """The runtime decision per row: forced, or not blocked and ``p >= threshold`` (one
    threshold, or one per row as in nested cross-validation)."""
    blocked, forced = rows.gate_masks()
    limit = np.asarray(threshold, dtype=np.float64)
    return forced | (~blocked & (np.asarray(probabilities, dtype=np.float64) >= limit))


def ranking_order(
    rows: RowSet, probabilities: np.ndarray, *, rank_by_acceptance: bool = True
) -> np.ndarray:
    """Row order of every case's ranking: ``rank_candidates_by_acceptance`` (scored rows by
    ``-p``, ``-fused_score``, technique id; hierarchy parents, which the model never scores,
    after them by ``-fused_score``, id), or the fused order when ranking by acceptance is off."""
    if rank_by_acceptance:
        unscored = rows.flags["hierarchy_parent"]
        primary = np.where(unscored, 0.0, -np.asarray(probabilities, dtype=np.float64))
        return np.lexsort(
            (rows.tid_rank, -rows.fused_score, primary, unscored.astype(np.int8), rows.case_index)
        )
    return np.lexsort((rows.tid_rank, rows.fused_rank, rows.case_index))


def case_rankings(
    rows: RowSet, probabilities: np.ndarray, *, rank_by_acceptance: bool = True
) -> list[list[str]]:
    order = ranking_order(rows, probabilities, rank_by_acceptance=rank_by_acceptance)
    output: list[list[str]] = [[] for _ in range(rows.case_count)]
    for row in order.tolist():
        output[int(rows.case_index[row])].append(str(rows.technique_ids[row]))
    return output


def _first_relevant(rows: RowSet, order: np.ndarray) -> np.ndarray:
    """0-based rank of each case's first correct technique (``inf`` when none is ranked)."""
    case_sorted = rows.case_index[order]
    starts = np.searchsorted(case_sorted, np.arange(rows.case_count), side="left")
    position = np.arange(len(order)) - starts[case_sorted] if len(order) else np.zeros(0)
    relevant = rows.labels[order] == 1
    first = np.full(rows.case_count, np.inf)
    if relevant.any():
        np.minimum.at(first, case_sorted[relevant], position[relevant].astype(np.float64))
    return first


def _prf(tp: float, fp: float, expected: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass
class Score:
    metrics: dict[str, Any]
    tp: np.ndarray
    fp: np.ndarray


def score_rows(
    rows: RowSet,
    probabilities: np.ndarray,
    threshold: float | np.ndarray,
    *,
    rank_by_acceptance: bool = True,
) -> Score:
    """Accepted-set micro P/R/F1 (recall counts labels never retrieved), per dataset, hit@1/3/10
    and MRR of the acceptance ranking, the pool ceiling, accepted techniques per case and the
    generic share (accepted techniques documented for many groups)."""
    pairs = rows.pairs
    accepted = np.zeros(len(pairs["truth"]), dtype=bool)
    np.logical_or.at(accepted, pairs["inverse"], accepted_rows(rows, probabilities, threshold))
    tp = np.bincount(pairs["case"][accepted & pairs["truth"]], minlength=rows.case_count)
    fp = np.bincount(pairs["case"][accepted & ~pairs["truth"]], minlength=rows.case_count)
    expected = rows.expected_array
    limit = np.asarray(threshold, dtype=np.float64)
    metrics: dict[str, Any] = {
        "threshold": float(limit) if limit.ndim == 0 else None,
        **_prf(float(tp.sum()), float(fp.sum()), float(expected.sum())),
        "true_positives": int(tp.sum()),
        "accepted": int(tp.sum() + fp.sum()),
        "expected": int(expected.sum()),
        "accepted_per_case": float((tp.sum() + fp.sum()) / max(rows.case_count, 1)),
        "generic_share": float(
            (accepted & pairs["generic"]).sum() / accepted.sum() if accepted.sum() else 0.0
        ),
    }
    datasets = np.asarray(rows.datasets)
    metrics["datasets"] = {
        dataset: {
            **_prf(
                float(tp[datasets == dataset].sum()),
                float(fp[datasets == dataset].sum()),
                float(expected[datasets == dataset].sum()),
            ),
            "cases": int((datasets == dataset).sum()),
        }
        for dataset in sorted(set(rows.datasets))
    }
    first = _first_relevant(
        rows, ranking_order(rows, probabilities, rank_by_acceptance=rank_by_acceptance)
    )
    positive = expected > 0
    count = max(int(positive.sum()), 1)
    for cutoff in (1, 3, 10):
        metrics[f"hit_at_{cutoff}"] = float((first[positive] < cutoff).sum() / count)
    reciprocal = np.where(np.isfinite(first), 1.0 / (first + 1.0), 0.0)
    metrics["mrr"] = float(reciprocal[positive].sum() / count)
    pooled_truth = int(pairs["truth"].sum())
    metrics["pool_label_recall"] = float(pooled_truth / expected.sum()) if expected.sum() else 0.0
    metrics["pool_case_share"] = float(np.isfinite(first)[positive].sum() / count)
    return Score(metrics=metrics, tp=tp.astype(np.int64), fp=fp.astype(np.int64))


def threshold_curve(
    rows: RowSet, probabilities: np.ndarray, *, step: float = 0.05
) -> list[dict[str, float]]:
    """P/R/F1 and accepted count at every ``step`` of the threshold (descriptive only)."""
    output = []
    for threshold in np.round(np.arange(step, 1.0, step), 2).tolist():
        pairs = rows.pairs
        accepted = np.zeros(len(pairs["truth"]), dtype=bool)
        np.logical_or.at(accepted, pairs["inverse"], accepted_rows(rows, probabilities, threshold))
        tp = float((accepted & pairs["truth"]).sum())
        fp = float((accepted & ~pairs["truth"]).sum())
        output.append(
            {
                "threshold": float(threshold),
                **_prf(tp, fp, float(rows.expected_array.sum())),
                "accepted": int(tp + fp),
            }
        )
    return output


# ----- label-free diagnostics -----
def fold_leak_check(
    exemplar_embeddings: np.ndarray,
    exemplar_sources: Sequence[str],
    collection_embeddings: np.ndarray,
    collection_fields: Sequence[str],
    *,
    band: tuple[float, float] = (0.85, 0.95),
    block: int = 512,
) -> dict[str, Any]:
    """How many exemplars sit within cosine ``band`` (or above it) of an ATT&CK procedure
    sentence of the technique collection (risk R9; no labels involved)."""
    procedures = np.flatnonzero(np.asarray(collection_fields) == "procedure")
    if not len(procedures) or not len(exemplar_embeddings):
        return {"exemplars": int(len(exemplar_embeddings)), "procedures": int(len(procedures))}
    matrix = _unit(np.asarray(collection_embeddings, dtype=np.float32)[procedures])
    library = _unit(np.asarray(exemplar_embeddings, dtype=np.float32))
    best = np.empty(len(library), dtype=np.float32)
    for start in range(0, len(library), block):
        best[start : start + block] = (library[start : start + block] @ matrix.T).max(axis=1)
    low, high = band
    in_band = (best >= low) & (best < high)
    above = best >= high
    sources = np.asarray(exemplar_sources)
    return {
        "exemplars": int(len(library)),
        "procedures": int(len(procedures)),
        "band": [low, high],
        "in_band": int(in_band.sum()),
        "at_or_above_high": int(above.sum()),
        "by_source": {
            str(source): {
                "in_band": int((in_band & (sources == source)).sum()),
                "at_or_above_high": int((above & (sources == source)).sum()),
                "exemplars": int((sources == source).sum()),
            }
            for source in sorted(set(sources.tolist()))
        },
        "max_similarity_quantiles": {
            str(q): float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
    }


def ann_segment_recall(cache: DevCache, depth: int, *, source: str = "ann") -> dict[str, Any]:
    """Per retrieval segment, the share of Chroma's top-``depth`` that is in the exact top-
    ``depth`` (a returned hit counts when its similarity reaches the exact ``depth``-th one less
    a float tolerance, so boundary ties are not misses)."""
    ann = cache.hit_list("dense", source, SERVING_FIELDS, depth)
    exact = cache._exact_or_lexical("dense", "exact", SERVING_FIELDS)
    recalls: list[float] = []
    differs = 0
    top1 = 0
    for key, (ann_index, ann_distance) in ann.entries.items():
        exact_index, exact_distance = exact.entries[key]
        wanted = min(int(depth), len(exact_index))
        if not wanted:
            continue
        kth = 1.0 - float(exact_distance[wanted - 1])
        exact_set = set(exact_index[:wanted].tolist())
        hits = sum(
            1
            for position, distance in zip(ann_index.tolist(), ann_distance.tolist(), strict=True)
            if position in exact_set or (1.0 - distance) >= kth - TIE_TOLERANCE
        )
        recalls.append(min(hits, wanted) / wanted)
        differs += int(set(ann_index[:wanted].tolist()) != exact_set)
        top1 += int(len(ann_index) > 0 and ann_index[0] != exact_index[0])
    values = np.asarray(recalls) if recalls else np.zeros(0)
    return {
        "depth": int(depth),
        "source": source,
        "segments": len(recalls),
        "recall_mean": float(values.mean()) if len(values) else None,
        "recall_min": float(values.min()) if len(values) else None,
        "full_recall_share": float(np.mean(values >= 1.0)) if len(values) else None,
        "id_set_differs_share": differs / max(len(recalls), 1),
        "top1_differs_share": top1 / max(len(recalls), 1),
    }


def depth_cap_share(
    cache: DevCache, depths: Sequence[int], *, threshold: float = 0.45, margin: float = 0.12
) -> dict[str, Any]:
    """Share of segments whose exact hits within ``margin`` of the best (and above the floor)
    outnumber each depth: there the fetch depth, not the margin, decides the dense set."""
    exact = cache._exact_or_lexical("dense", "exact", SERVING_FIELDS)
    within: list[int] = []
    for _, distance in exact.entries.values():
        similarity = 1.0 - distance
        if not len(similarity):
            within.append(0)
            continue
        cutoff = max(threshold, float(similarity[0]) - margin)
        within.append(int((similarity >= cutoff).sum()))
    counts = np.asarray(within)
    return {
        "segments": int(len(counts)),
        "exact_depth": int(exact.depth),
        "within_margin_median": float(np.median(counts)) if len(counts) else None,
        "within_margin_p90": float(np.quantile(counts, 0.9)) if len(counts) else None,
        "capped_share": {str(depth): float(np.mean(counts > int(depth))) for depth in depths},
        "saturated_at_exact_depth_share": float(np.mean(counts >= exact.depth))
        if len(counts)
        else None,
    }
