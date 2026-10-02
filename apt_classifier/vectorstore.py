import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.parse import split_sentences, strip_attack_citations
from apt_classifier.config import Settings
from apt_classifier.embeddings import Embedder
from apt_classifier.memo import ObjectMemo

TECHNIQUE_COLLECTION = "attack_techniques"
GROUP_COLLECTION = "attack_groups"
TECHNIQUE_SCHEMA_VERSION = 4
TECHNIQUE_FIELDS = ("name", "description", "procedure", "synthetic", "detection")
DEFAULT_SYNTHETIC_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "kb" / "synthetic_sentences.json"
)

Metadata = dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class VectorDocument:
    id: str
    document: str
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class VectorStoreCounts:
    technique_sentences: int
    group_documents: int
    actor_behavior_documents: int = 0


@dataclass(frozen=True, slots=True)
class VectorHit:
    id: str
    document: str
    metadata: Metadata
    distance: float
    similarity: float


# result rows per ChromaDB query call; the server's limit is about 32k (measured 2026-10-01)
_MAX_QUERY_RESULT_ROWS = 16_000
_ARTICLE_BEFORE_ADVERSARY_RE = re.compile(r"\b(?:an?|the)\s+(the adversary)", re.IGNORECASE)
# a stand-in for an ATT&CK collection (scripts/sync_attack_collection_copy.py); schema 2 also
# records a hash of a fixed sample of the source's records, re-read at load
COLLECTION_COPY_SCHEMA_VERSION = 2
# records of the source hashed at copy time and compared at load (cheap: one get by ids)
COPY_CONTENT_SAMPLE = 64
# a copy's embeddings against its source's: Chroma re-normalises vectors in float32 on insert
# (about 1e-8 per component), so they are compared within this, never bit for bit
COPY_EMBEDDING_TOLERANCE = 1e-6


def batched_query(
    collection: Any,
    embeddings: Sequence[Any],
    *,
    n_results: int,
    where: dict[str, Any] | None = None,
    include: Sequence[str] = ("documents", "metadatas", "distances"),
) -> list[dict[str, Any]]:
    """The raw results of ``collection.query`` over ``embeddings``, one call per batch.

    ChromaDB's SQLite backend refuses one query whose result rows (queries x n_results) pass
    about 32k ("too many SQL variables"): a long pasted report crashed retrieval. Each query is
    independent, so the batches return exactly the single call's hits, in query order."""
    batch = max(1, _MAX_QUERY_RESULT_ROWS // max(1, int(n_results)))
    return [
        collection.query(
            query_embeddings=embeddings[start : start + batch],
            n_results=n_results,
            where=where,
            include=list(include),
        )
        for start in range(0, len(embeddings), batch)
    ]


def canonical_json(value: Any) -> str:
    """Key-sorted compact JSON: the form record and metadata hashes are taken over."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def records_hash(
    ids: Sequence[str],
    documents: Sequence[str | None],
    metadatas: Sequence[Mapping[str, Any] | None],
) -> str:
    """SHA-256 over (id, document, canonical metadata) per record, in the given order."""
    digest = hashlib.sha256()
    for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        digest.update(canonical_json([str(item_id), document, dict(metadata or {})]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def spread_sample_ids(ids: Iterable[str], size: int) -> list[str]:
    """A fixed sample of ``size`` ids spread evenly over the sorted ids (the same ids for the
    same id set, so two readings hash the same records)."""
    ordered = sorted(str(item) for item in ids)
    if not ordered or size <= 0:
        return []
    count = min(int(size), len(ordered))
    return sorted({ordered[(position * len(ordered)) // count] for position in range(count)})


def _records_by_id(collection: Any, ids: Sequence[str]) -> dict[str, tuple[Any, Any, Any]]:
    """``id -> (embedding, document, metadata)`` of ``ids`` as the collection holds them now
    (one ``get``; missing records are absent)."""
    wanted = sorted(str(item) for item in ids)
    page = (
        collection.get(ids=wanted, include=["embeddings", "documents", "metadatas"])
        if wanted
        else {}
    )
    returned = [str(item) for item in (page.get("ids") or [])]
    embeddings = page.get("embeddings")
    documents = page.get("documents") or [None] * len(returned)
    metadatas = page.get("metadatas") or [None] * len(returned)
    return {
        item_id: (
            embeddings[position] if embeddings is not None else None,
            documents[position],
            metadatas[position],
        )
        for position, item_id in enumerate(returned)
    }


def _records_sha256(wanted: Sequence[str], records: Mapping[str, tuple[Any, Any, Any]]) -> str:
    digest = hashlib.sha256()
    for item_id in wanted:
        record = records.get(item_id)
        if record is None:
            digest.update(canonical_json([item_id, "<missing>"]).encode())
        else:
            vector, document, metadata = record
            digest.update(canonical_json([item_id, document, dict(metadata or {})]).encode())
            if vector is not None:
                digest.update(np.ascontiguousarray(np.asarray(vector, dtype=np.float32)).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def sample_records_sha256(collection: Any, ids: Sequence[str]) -> str:
    """SHA-256 over (id, document, canonical metadata, float32 embedding) of ``ids`` as the
    collection holds them now, in sorted id order (a missing record hashes as missing)."""
    wanted = sorted(str(item) for item in ids)
    return _records_sha256(wanted, _records_by_id(collection, wanted))


def embedding_deviation(values: Any, reference: Any = None) -> dict[str, Any]:
    """How far float32 vectors are from unit length and, given ``reference`` vectors of the same
    records, from those: rows bit-identical and the largest component difference (float64 over
    the float32 values). Chroma re-normalises every vector in float32 on insert, so a copy of a
    collection is equal to its source only within a tolerance, never bit for bit."""
    vectors = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if vectors.ndim == 1:
        vectors = vectors[None, :] if vectors.size else vectors.reshape(0, 0)
    rows = int(len(vectors))
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1) if rows else np.zeros(0)
    report: dict[str, Any] = {
        "rows": rows,
        "finite": bool(np.isfinite(vectors).all()),
        "norm_min": float(norms.min()) if rows else None,
        "norm_max": float(norms.max()) if rows else None,
        "max_norm_deviation": float(np.max(np.abs(norms - 1.0))) if rows else 0.0,
        "compared_with_source": reference is not None,
    }
    if reference is None:
        return report
    other = np.ascontiguousarray(np.asarray(reference, dtype=np.float32))
    if other.ndim == 1:
        other = other[None, :] if other.size else other.reshape(0, 0)
    if other.shape != vectors.shape:
        report.update(
            {
                "shape_mismatch": [list(vectors.shape), list(other.shape)],
                "identical_rows": 0,
                "differing_rows": rows,
                "max_abs_diff": float("inf"),
            }
        )
        return report
    identical = (
        np.all(vectors.view(np.uint32) == other.view(np.uint32), axis=1)
        if vectors.size
        else np.ones(rows, dtype=bool)
    )
    difference = np.abs(vectors.astype(np.float64) - other.astype(np.float64))
    report.update(
        {
            "identical_rows": int(identical.sum()),
            "differing_rows": int(rows - identical.sum()),
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
        }
    )
    return report


def within_tolerance(deviation: Mapping[str, Any], tolerance: float) -> bool:
    """Whether :func:`embedding_deviation` against a reference is inside ``tolerance`` (finite
    vectors of the same shape, every component within it; ``NaN`` never is)."""
    return (
        bool(deviation.get("finite"))
        and bool(deviation.get("compared_with_source"))
        and not deviation.get("shape_mismatch")
        and float(deviation.get("max_abs_diff", float("inf"))) <= float(tolerance)
    )


def copy_sample_problem(
    copy_records: Mapping[str, tuple[Any, Any, Any]],
    source_records: Mapping[str, tuple[Any, Any, Any]],
    *,
    tolerance: float,
) -> str | None:
    """Why the copy's sampled records do not mirror the source's (``None`` when they do): every
    id present, documents and metadata equal, embeddings within ``tolerance``."""
    missing = sorted(set(source_records) - set(copy_records))
    if missing:
        return f"{len(missing)} sampled records are missing from the copy (e.g. {missing[0]})"
    ids = sorted(source_records)
    for item_id in ids:
        _, document, metadata = source_records[item_id]
        _, copied_document, copied_metadata = copy_records[item_id]
        if copied_document != document or canonical_json(dict(copied_metadata or {})) != (
            canonical_json(dict(metadata or {}))
        ):
            return f"sampled record {item_id} has another document or metadata than its source"
    if not ids:
        return None
    if any(copy_records[item][0] is None or source_records[item][0] is None for item in ids):
        return "sampled records came back without their embeddings"
    deviation = embedding_deviation(
        [copy_records[item][0] for item in ids], [source_records[item][0] for item in ids]
    )
    if not within_tolerance(deviation, tolerance):
        return (
            f"sampled embeddings differ from the source by up to {deviation.get('max_abs_diff')} "
            f"(tolerance {tolerance:g})"
        )
    return None


def hnsw_configuration(collection: Any) -> dict[str, Any]:
    """The HNSW settings a collection reports (``configuration_json["hnsw"]``), ``{}`` if none."""
    configuration = getattr(collection, "configuration_json", None) or {}
    hnsw = configuration.get("hnsw") if isinstance(configuration, Mapping) else None
    return dict(hnsw) if isinstance(hnsw, Mapping) else {}


def collection_copy_status(
    copy: Any, source: Any, *, tolerance: float = COPY_EMBEDDING_TOLERANCE
) -> tuple[bool, str]:
    """Whether ``copy`` still mirrors ``source``: same source name, id, metadata, count and
    sampled content.

    The copy records its source when it is made; a re-ingest (new id or count), a metadata
    change, or a compatible in-place re-ingest that rewrote the sampled records (new text, or
    the same text re-embedded) makes it stale: the source's sampled records must hash exactly as
    they did at copy time. The copy's own sampled records must equal the source's (ids,
    documents, metadata) with embeddings within ``tolerance`` (Chroma re-normalised them when
    the copy was written). A rewrite that spares every sampled record is not seen (the full
    hashes, ``copy_records_hash`` and ``copy_embeddings_sha256``, need the whole source;
    ``scripts/sync_attack_collection_copy.py --verify-only`` compares every record)."""
    metadata = dict(copy.metadata or {})
    if "copy_of" not in metadata:
        return False, "not a collection copy (no copy_of metadata)"
    if int(metadata.get("copy_schema_version", 0) or 0) != COLLECTION_COPY_SCHEMA_VERSION:
        return False, f"copy schema {metadata.get('copy_schema_version')!r} is not supported"
    if metadata.get("copy_of") != str(source.name):
        return False, f"copied from {metadata.get('copy_of')!r}, not {source.name!r}"
    if metadata.get("copy_of_id") != str(source.id):
        return False, f"source {source.name} was recreated since the copy was made"
    if metadata.get("copy_source_metadata_json") != canonical_json(dict(source.metadata or {})):
        return False, f"source {source.name} metadata changed since the copy was made"
    source_count = int(source.count())
    if int(metadata.get("copy_source_count", -1)) != source_count:
        return False, (
            f"source {source.name} has {source_count} records, the copy was made from "
            f"{metadata.get('copy_source_count')}"
        )
    if int(copy.count()) != source_count:
        return False, f"copy has {copy.count()} records, its source {source_count}"
    try:
        sample = json.loads(str(metadata.get("copy_sample_ids_json") or ""))
    except json.JSONDecodeError:
        sample = None
    if not isinstance(sample, list) or not metadata.get("copy_sample_sha256"):
        return False, "copy records no content sample of its source"
    wanted = sorted(str(item) for item in sample)
    source_records = _records_by_id(source, wanted)
    if _records_sha256(wanted, source_records) != metadata.get("copy_sample_sha256"):
        return False, (
            f"source {source.name} content changed since the copy was made (records rewritten "
            "or re-embedded in place)"
        )
    problem = copy_sample_problem(
        _records_by_id(copy, wanted), source_records, tolerance=float(tolerance)
    )
    if problem is not None:
        return False, f"copy content does not mirror {source.name}: {problem}"
    return True, "ok"


def _actor_name_pattern(actor_names: tuple[str, ...]) -> re.Pattern[str] | None:
    """One alternation over every actor name/alias, longest first, so one pass replaces all.

    The previous implementation ran one ``re.sub`` per name per sentence (about 600 x 5,000
    substitutions on the real KB, several minutes per call); the combined pattern yields
    identical output in well under a second.
    """
    names = [name for name in actor_names if name]
    if not names:
        return None
    names.sort(key=lambda item: (-len(item), item.casefold()))
    return re.compile(
        r"(?<![\w.-])(?:" + "|".join(re.escape(name) for name in names) + r")"
        r"(?P<possessive>['’]s)?(?![\w.-])",
        re.IGNORECASE,
    )


def _sanitize_procedure(
    value: str,
    actor_names: tuple[str, ...],
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    text = strip_attack_citations(value).strip()
    pattern = pattern if pattern is not None else _actor_name_pattern(actor_names)
    if pattern is not None:
        text = pattern.sub(lambda match: "the adversary" + (match.group("possessive") or ""), text)
    text = _ARTICLE_BEFORE_ADVERSARY_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


_PROCEDURE_CACHE: ObjectMemo[dict[str, list[str]]] = ObjectMemo()


def _procedure_sentences(knowledge_base: AttackKnowledgeBase) -> dict[str, list[str]]:
    """Anonymized, de-duplicated procedure sentences per technique (memoized per KB object).

    The result is cached by object identity with a (version, group count, technique count)
    stamp, because the same knowledge base is asked for these sentences by the vector
    store, the lexical index and the benchmark leak filter within one process.
    """
    stamp = (
        str(getattr(knowledge_base, "version", "")),
        len(knowledge_base.groups),
        len(knowledge_base.techniques),
    )
    cached = _PROCEDURE_CACHE.get(knowledge_base, stamp)
    if cached is not None:
        return cached
    names = tuple(
        sorted(
            {
                name.strip()
                for group in knowledge_base.groups.values()
                for name in (group.name, *group.aliases)
                if name and name.strip()
            },
            key=lambda item: (-len(item), item.casefold()),
        )
    )
    pattern = _actor_name_pattern(names)
    procedures: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for group in knowledge_base.groups.values():
        for technique_id, values in group.technique_behaviors.items():
            if technique_id not in knowledge_base.techniques:
                continue
            for value in values:
                text = _sanitize_procedure(value, names, pattern=pattern)
                normalized = text.casefold()
                if not text or normalized in seen.setdefault(technique_id, set()):
                    continue
                seen[technique_id].add(normalized)
                procedures.setdefault(technique_id, []).append(text)
    return _PROCEDURE_CACHE.put(knowledge_base, stamp, procedures)


def procedure_sentences_by_technique(knowledge_base: Any) -> dict[str, list[str]]:
    """Public accessor for the anonymized procedure sentences embedded per technique."""
    try:
        return _procedure_sentences(knowledge_base)
    except AttributeError:
        return {}


def normalize_procedure_text(value: str) -> str:
    """Casefolded, punctuation-free form used to compare text against indexed sentences."""
    return re.sub(r"\W+", " ", value or "").casefold().strip()


def procedure_sentence_texts(knowledge_base: AttackKnowledgeBase) -> frozenset[str]:
    """Normalized ATT&CK procedure sentences that end up in the technique index.

    Both the raw citation-stripped actor/software sentences and their sanitized indexed
    forms are returned so evaluators can drop benchmark cases whose text is identical to
    indexed content (see ``scripts/benchmark_mapping.py``).
    """
    texts: set[str] = set()
    records = list(knowledge_base.groups.values()) + list(
        getattr(knowledge_base, "software", {}).values()
    )
    for record in records:
        for values in getattr(record, "technique_behaviors", {}).values():
            for value in values:
                raw = strip_attack_citations(value)
                if raw:
                    texts.add(normalize_procedure_text(raw))
    for values in _procedure_sentences(knowledge_base).values():
        for value in values:
            texts.add(normalize_procedure_text(value))
    texts.discard("")
    return frozenset(texts)


def load_synthetic_sentences(path: str | Path | None = None) -> dict[str, list[str]]:
    """LLM-generated observation sentences per technique, or {} when the file is absent.

    Written by ``scripts/generate_synthetic_sentences.py`` from the ATT&CK knowledge base
    only (technique descriptions and anonymized procedure examples), never from benchmark
    text, so indexing them cannot leak evaluation data.
    """
    source = Path(path) if path is not None else DEFAULT_SYNTHETIC_PATH
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    sentences = payload.get("sentences", payload) if isinstance(payload, dict) else {}
    return {
        str(technique_id): [str(item) for item in items if str(item).strip()]
        for technique_id, items in sentences.items()
        if isinstance(items, list)
    }


def technique_documents(
    knowledge_base: AttackKnowledgeBase,
    synthetic: dict[str, list[str]] | None = None,
) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    procedures = _procedure_sentences(knowledge_base)
    synthetic = synthetic if synthetic is not None else {}
    for technique_id, technique in sorted(knowledge_base.techniques.items()):
        sentence_fields = (
            dict(technique.sentences)
            if technique.sentences
            else {
                "name": [strip_attack_citations(technique.name)] if technique.name else [],
                "description": split_sentences(technique.description),
                "detection": split_sentences(technique.detection),
            }
        )
        sentence_fields["procedure"] = procedures.get(technique_id, [])
        sentence_fields["synthetic"] = synthetic.get(technique_id, [])
        for field_name in TECHNIQUE_FIELDS:
            for index, sentence in enumerate(sentence_fields.get(field_name, [])):
                text = sentence.strip()
                if not text:
                    continue
                documents.append(
                    VectorDocument(
                        id=f"{technique_id}:{field_name}:{index:04d}",
                        document=text,
                        metadata={
                            "technique_id": technique_id,
                            "stix_id": technique.stix_id,
                            "name": technique.name,
                            "field": field_name,
                            "sentence_index": index,
                            "is_subtechnique": technique.is_subtechnique,
                            "parent_id": technique.parent_id or "",
                            "tactics": ",".join(technique.tactics),
                            "platforms": ",".join(technique.platforms),
                        },
                    )
                )
    return documents


def group_documents(knowledge_base: AttackKnowledgeBase) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    for group_id, group in sorted(knowledge_base.groups.items()):
        techniques = [
            knowledge_base.techniques[technique_id]
            for technique_id in group.technique_ids
            if technique_id in knowledge_base.techniques
        ]
        profile = ". ".join(f"{item.id} {strip_attack_citations(item.name)}" for item in techniques)
        if not profile:
            profile = f"{group.name} has no documented ATT&CK technique profile."
        description = strip_attack_citations(group.description or group.name)
        base_metadata: Metadata = {
            "group_id": group_id,
            "stix_id": group.stix_id,
            "name": group.name,
        }
        documents.extend(
            (
                VectorDocument(
                    id=f"{group_id}:technique-profile",
                    document=profile,
                    metadata=base_metadata
                    | {"document_type": "technique_profile", "technique_count": len(techniques)},
                ),
                VectorDocument(
                    id=f"{group_id}:description",
                    document=description,
                    metadata=base_metadata | {"document_type": "description"},
                ),
            )
        )
        for technique_id, sentences in sorted(group.technique_behaviors.items()):
            for index, sentence in enumerate(sentences):
                text = strip_attack_citations(sentence).strip()
                if not text:
                    continue
                documents.append(
                    VectorDocument(
                        id=f"{group_id}:actor-behavior:{technique_id}:{index:04d}",
                        document=text,
                        metadata=base_metadata
                        | {
                            "document_type": "actor_behavior",
                            "technique_id": technique_id,
                            "sentence_index": index,
                        },
                    )
                )
    return documents


def _batches(values: list[VectorDocument], size: int) -> Iterable[list[VectorDocument]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@dataclass(slots=True)
class ChromaVectorStore:
    settings: Settings
    embedder: Embedder
    client: Any = None
    batch_size: int = 128
    # verified read-only copies (TECHNIQUE_COLLECTION_NAME / GROUP_COLLECTION_NAME), by name
    _copies: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.client is None:
            self.client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
                ssl=self.settings.chroma_ssl,
            )

    def _collection(
        self,
        name: str,
        kind: str,
        *,
        metadata: Metadata | None = None,
    ) -> Any:
        return self.client.get_or_create_collection(
            name=name,
            metadata={"kind": kind} | dict(metadata or {}),
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def _technique_collection_metadata(self, attack_version: str | None = None) -> Metadata:
        return {
            "schema_version": TECHNIQUE_SCHEMA_VERSION,
            "attack_version": attack_version or self.settings.attack_version,
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.settings.embedding_model_revision,
            "fields": ",".join(TECHNIQUE_FIELDS),
        }

    def _validate_technique_collection(self, collection: Any) -> None:
        if collection.count() == 0:
            return
        metadata = collection.metadata or {}
        expected = self._technique_collection_metadata()
        mismatches = {
            key: (metadata.get(key), expected_value)
            for key, expected_value in expected.items()
            if metadata.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} expected {wanted!r}"
                for key, (actual, wanted) in sorted(mismatches.items())
            )
            raise RuntimeError(
                f"incompatible ATT&CK technique vector index ({details}); run make ingest"
            )

    def _technique_index_is_compatible(self, collection: Any) -> bool:
        if collection.count() == 0:
            return True
        expected = self._technique_collection_metadata()
        metadata = collection.metadata or {}
        return all(metadata.get(key) == value for key, value in expected.items())

    @property
    def technique_collection_name(self) -> str:
        return str(
            getattr(self.settings, "technique_collection_name", TECHNIQUE_COLLECTION)
            or TECHNIQUE_COLLECTION
        )

    @property
    def group_collection_name(self) -> str:
        return str(
            getattr(self.settings, "group_collection_name", GROUP_COLLECTION) or GROUP_COLLECTION
        )

    def _collection_copy(self, name: str, source_name: str) -> Any:
        """A configured copy, opened with ``get_collection`` only (never created or changed)
        and refused unless it still mirrors its source; verified once per store."""
        cached = self._copies.get(name)
        if cached is not None:
            return cached
        try:
            collection = self.client.get_collection(name=name, embedding_function=None)
            source = self.client.get_collection(name=source_name, embedding_function=None)
        except Exception as error:
            raise RuntimeError(
                f"collection copy {name} or its source is unavailable: {error}"
            ) from error
        ok, reason = collection_copy_status(collection, source)
        if not ok:
            raise RuntimeError(
                f"collection {name} cannot stand in for {source_name}: {reason}; rebuild it with "
                "scripts/sync_attack_collection_copy.py or unset the collection name setting"
            )
        self._copies[name] = collection
        return collection

    @property
    def technique_collection(self) -> Any:
        name = self.technique_collection_name
        if name != TECHNIQUE_COLLECTION:
            return self._collection_copy(name, TECHNIQUE_COLLECTION)
        return self._collection(
            TECHNIQUE_COLLECTION,
            "technique_sentences",
            metadata=self._technique_collection_metadata(),
        )

    @property
    def group_collection(self) -> Any:
        name = self.group_collection_name
        if name != GROUP_COLLECTION:
            return self._collection_copy(name, GROUP_COLLECTION)
        return self._collection(GROUP_COLLECTION, "group_profiles")

    def reset(self) -> None:
        for name in (TECHNIQUE_COLLECTION, GROUP_COLLECTION):
            try:
                self.client.delete_collection(name=name)
            except Exception as error:
                message = str(error).lower()
                if "not found" not in message and "does not exist" not in message:
                    raise

    def _upsert(
        self,
        collection: Any,
        documents: list[VectorDocument],
        explicit_embeddings: dict[str, list[float]] | None = None,
    ) -> int:
        explicit_embeddings = explicit_embeddings or {}
        for batch in _batches(documents, self.batch_size):
            texts = [item.document for item in batch]
            embeddings: list[list[float] | None] = [
                explicit_embeddings.get(item.id) for item in batch
            ]
            missing = [index for index, value in enumerate(embeddings) if value is None]
            generated = self.embedder.embed([texts[index] for index in missing])
            if len(generated) != len(missing):
                raise RuntimeError("Embedder returned an unexpected number of embeddings")
            for index, embedding in zip(missing, generated, strict=True):
                embeddings[index] = embedding
            collection.upsert(
                ids=[item.id for item in batch],
                documents=texts,
                metadatas=[item.metadata for item in batch],
                embeddings=[value for value in embeddings if value is not None],
            )
        return len(documents)

    def _profile_embeddings(self, knowledge_base: AttackKnowledgeBase) -> dict[str, list[float]]:
        technique_ids = sorted(
            {
                technique_id
                for group in knowledge_base.groups.values()
                for technique_id in group.technique_ids
                if technique_id in knowledge_base.techniques
            }
        )
        vectors = self.embedder.embed(
            [
                f"{technique_id} {knowledge_base.techniques[technique_id].name}"
                for technique_id in technique_ids
            ]
        )
        by_technique = dict(zip(technique_ids, vectors, strict=True))
        output: dict[str, list[float]] = {}
        for group_id, group in knowledge_base.groups.items():
            members = [
                by_technique[technique_id]
                for technique_id in group.technique_ids
                if technique_id in by_technique
            ]
            if not members:
                continue
            centroid = np.asarray(members, dtype=np.float32).mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm:
                centroid /= norm
            output[f"{group_id}:technique-profile"] = centroid.tolist()
        return output

    def upsert(
        self, knowledge_base: AttackKnowledgeBase, *, reset: bool = False
    ) -> VectorStoreCounts:
        if (self.technique_collection_name, self.group_collection_name) != (
            TECHNIQUE_COLLECTION,
            GROUP_COLLECTION,
        ):
            # a configured copy is read-only; ingestion writes only the default collections
            raise RuntimeError(
                "ingestion writes only attack_techniques and attack_groups; unset "
                "TECHNIQUE_COLLECTION_NAME / GROUP_COLLECTION_NAME (copies are read-only)"
            )
        # Procedure documents alter the technique corpus.  Never leave a
        # mixture of old and new schema documents in place: an incompatible
        # collection is rebuilt as a unit during ingestion.
        existing = self.technique_collection
        if not reset and not self._technique_index_is_compatible(existing):
            reset = True
        if reset:
            self.reset()
        techniques = technique_documents(
            knowledge_base,
            load_synthetic_sentences(getattr(self.settings, "synthetic_sentences_path", None)),
        )
        groups = group_documents(knowledge_base)
        behavior_count = sum(
            document.metadata.get("document_type") == "actor_behavior" for document in groups
        )
        technique_collection = self.technique_collection
        technique_count = self._upsert(technique_collection, techniques)
        technique_collection.modify(
            metadata={
                "kind": "technique_sentences",
                **self._technique_collection_metadata(),
                "attack_kb_version": str(getattr(knowledge_base, "version", "") or ""),
                "procedure_documents": sum(
                    document.metadata.get("field") == "procedure" for document in techniques
                ),
            }
        )
        group_collection = self.group_collection
        group_count = self._upsert(
            group_collection, groups, self._profile_embeddings(knowledge_base)
        )
        group_collection.modify(
            metadata={
                "kind": "group_profiles",
                "actor_behavior_count": behavior_count,
                "schema_version": 2,
            }
        )
        return VectorStoreCounts(technique_count, group_count, behavior_count)

    def health(self) -> dict[str, Any]:
        return {"heartbeat": self.client.heartbeat(), "counts": self.counts()}

    def counts(self) -> VectorStoreCounts:
        group_collection = self.group_collection
        metadata = group_collection.metadata or {}
        return VectorStoreCounts(
            self.technique_collection.count(),
            group_collection.count(),
            int(metadata.get("actor_behavior_count", 0)),
        )

    def query_techniques(
        self,
        texts: list[str],
        *,
        n_results: int = 10,
        fields: tuple[str, ...] | None = None,
    ) -> list[list[VectorHit]]:
        if not texts:
            return []
        self._validate_technique_collection(self.technique_collection)
        embeddings = self.embedder.embed(texts)
        where: dict[str, Any] | None = None
        if fields:
            where = {"field": fields[0]} if len(fields) == 1 else {"field": {"$in": list(fields)}}
        return self._query(
            self.technique_collection,
            embeddings,
            n_results=n_results,
            where=where,
        )

    def query_group_behaviors(
        self, texts: list[str], *, n_results: int = 20
    ) -> list[list[VectorHit]]:
        if not texts:
            return []
        embeddings = self.embedder.embed(texts)
        return self._query(
            self.group_collection,
            embeddings,
            n_results=n_results,
            where={"document_type": "actor_behavior"},
        )

    def query_group_profile(
        self, texts: list[str], *, document_type: str, n_results: int = 20
    ) -> list[VectorHit]:
        if not texts:
            return []
        values = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        centroid = values.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm:
            centroid /= norm
        results = self._query(
            self.group_collection,
            [centroid.tolist()],
            n_results=n_results,
            where={"document_type": document_type},
        )
        return results[0] if results else []

    def _query(
        self,
        collection: Any,
        embeddings: list[list[float]],
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[list[VectorHit]]:
        count = collection.count()
        if count == 0:
            return [[] for _ in embeddings]
        limit = min(max(n_results, 1), count)
        output: list[list[VectorHit]] = []
        for result in batched_query(collection, embeddings, n_results=limit, where=where):
            for query_index, ids in enumerate(result.get("ids") or []):
                documents = (result.get("documents") or [[]])[query_index] or []
                metadatas = (result.get("metadatas") or [[]])[query_index] or []
                distances = (result.get("distances") or [[]])[query_index] or []
                hits = []
                for index, item_id in enumerate(ids):
                    distance = float(distances[index])
                    hits.append(
                        VectorHit(
                            id=str(item_id),
                            document=str(documents[index]),
                            metadata=dict(metadatas[index] or {}),
                            distance=distance,
                            similarity=1.0 - distance,
                        )
                    )
                output.append(hits)
        return output

    ingest = upsert


AttackVectorStore = ChromaVectorStore
