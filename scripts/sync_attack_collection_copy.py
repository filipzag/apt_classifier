"""Copy an ATT&CK collection read-only into a new collection with chosen HNSW settings.

``attack_techniques`` and ``attack_groups`` are built by ``make ingest`` with Chroma's HNSW
defaults (M 16, ef_construction 100, ef_search 100), which the shared server cannot change after
creation. This script reads a source collection (ids, documents, metadata and embeddings; nothing
is re-embedded), creates a *new* collection with ``create_collection(get_or_create=False)`` and
the requested settings, adds the records in batches and verifies the copy. The source is only
read; the script never calls ``delete``, ``modify``, ``upsert`` or ``update``, and an existing
target is only verified.

    PYTHONPATH=. .venv/bin/python scripts/sync_attack_collection_copy.py --list
    PYTHONPATH=. .venv/bin/python scripts/sync_attack_collection_copy.py --source attack_techniques \\
        --max-neighbors 32 --ef-construction 400 --ef-search 1024 --dry-run
    PYTHONPATH=. .venv/bin/python scripts/sync_attack_collection_copy.py --source attack_techniques \\
        --max-neighbors 32 --ef-construction 400 --ef-search 1024
    PYTHONPATH=. .venv/bin/python scripts/sync_attack_collection_copy.py --target NAME --verify-only \\
        --query-vectors dev_queries.npy --fields name description procedure --recall-k 90 300

The default target name is ``<source>-s<schema>-<srcmeta12>-m<M>-efc<efc>`` (``srcmeta12``: the
first 12 hex digits of the SHA-256 of the source's canonical metadata). The copy keeps the
source's metadata keys (so the technique-index compatibility check applies to it unchanged) and
records its source: ``copy_of``, ``copy_of_id``, ``copy_source_count``,
``copy_source_metadata_json``, ``copy_records_hash``, ``copy_embeddings_sha256`` and a hash of a
fixed sample of source records (``copy_sample_ids_json``, ``copy_sample_sha256``).
``ChromaVectorStore`` uses a copy only when ``TECHNIQUE_COLLECTION_NAME`` /
``GROUP_COLLECTION_NAME`` names it, and refuses it once the source changed (re-ingest, new id,
other metadata or count, or sampled records rewritten or re-embedded in place); ``--verify-only``
compares every record.

Verification (after a copy, or alone with ``--verify-only`` on an existing copy, which is never
recreated): count, ids, documents and metadata equal, the record hash, embeddings within
``--tolerance`` (default 1e-6) of the source's (Chroma re-normalises every vector in float32 on
insert, so a copy is never bit-identical; rows bit-identical, the largest component difference and
the norm range are reported), the source's own embeddings bit for bit as they were copied
(``copy_embeddings_sha256``), the HNSW configuration echo against the requested settings (pass
the ones the copy was made with when verifying it later; a server that ignored ``max_neighbors``
or ``ef_construction`` fails it: copy under a new name such as ``-r2``, never delete), and ANN
recall against brute force at each ``--recall-k``: recall counts how many of the exact top-k
(ties at the k-th similarity included) Chroma returns, over ``--query-vectors`` (an ``.npy`` of
query embeddings, e.g. dev and train query texts embedded by the tuning harness) or
``--self-queries`` stored vectors, restricted to ``--fields`` like the runtime filter. Exit 0 when
every check passes and mean recall reaches ``--min-recall``, 1 otherwise, 2 when refused.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.evaluation import write_artifact  # noqa: E402
from apt_classifier.vectorstore import (  # noqa: E402
    COLLECTION_COPY_SCHEMA_VERSION,
    COPY_CONTENT_SAMPLE,
    COPY_EMBEDDING_TOLERANCE,
    GROUP_COLLECTION,
    TECHNIQUE_COLLECTION,
    batched_query,
    canonical_json,
    collection_copy_status,
    embedding_deviation,
    hnsw_configuration,
    records_hash,
    sample_records_sha256,
    spread_sample_ids,
    within_tolerance,
)
from scripts.run_tooling import add_output_guard_arguments, enforce_output_guard  # noqa: E402

SOURCES = (TECHNIQUE_COLLECTION, GROUP_COLLECTION)
PAGE = 1000
TIE_TOLERANCE = 1e-6
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")


class CopyError(RuntimeError):
    pass


def copy_collection_name(source: str, metadata: dict[str, Any], m: int, efc: int) -> str:
    schema = metadata.get("schema_version", 0)
    digest = hashlib.sha256(canonical_json(dict(metadata)).encode()).hexdigest()[:12]
    return f"{source}-s{schema}-{digest}-m{int(m)}-efc{int(efc)}"


def name_problem(source: str, target: str) -> str | None:
    if target == source:
        return "the target must be a new collection, not the source"
    if not target.startswith(f"{source}-"):
        return f"target {target!r} must start with {source + '-'!r}"
    if not _NAME_RE.match(target):
        return f"target {target!r} is not a valid Chroma name (3-63 of [A-Za-z0-9._-])"
    return None


def read_records(collection: Any) -> dict[str, Any]:
    """Every record of a collection, sorted by id: ids, documents, metadatas, float32 vectors."""
    count = int(collection.count())
    rows: dict[str, tuple[Any, Any, Any]] = {}
    for offset in range(0, count, PAGE):
        page = collection.get(
            limit=PAGE, offset=offset, include=["embeddings", "documents", "metadatas"]
        )
        ids = [str(item) for item in (page.get("ids") or [])]
        vectors = page.get("embeddings")
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        if vectors is None or len(vectors) != len(ids):
            raise CopyError(f"{collection.name}: records came back without embeddings")
        for position, item_id in enumerate(ids):
            if item_id in rows:
                raise CopyError(f"{collection.name}: record {item_id} returned twice")
            rows[item_id] = (vectors[position], documents[position], metadatas[position])
    if len(rows) != count:
        raise CopyError(f"{collection.name}: read {len(rows)} records, the count is {count}")
    ids = sorted(rows)
    return {
        "ids": ids,
        "documents": [rows[item][1] for item in ids],
        "metadatas": [dict(rows[item][2]) if rows[item][2] is not None else None for item in ids],
        "embeddings": np.ascontiguousarray(
            np.asarray([rows[item][0] for item in ids], dtype=np.float32)
        ),
    }


def embeddings_sha256(embeddings: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes()).hexdigest()


class _RecordsView:
    """``get(ids=..., include=...)`` over records already read (for the copy's sample hash)."""

    def __init__(self, records: dict[str, Any]) -> None:
        self._position = {item: index for index, item in enumerate(records["ids"])}
        self._records = records

    def get(self, ids: list[str], include: list[str]) -> dict[str, Any]:
        found = [item for item in ids if item in self._position]
        rows = [self._position[item] for item in found]
        return {
            "ids": found,
            "embeddings": self._records["embeddings"][rows] if rows else None,
            "documents": [self._records["documents"][row] for row in rows],
            "metadatas": [self._records["metadatas"][row] for row in rows],
        }


def copy_metadata(source: Any, records: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(source.metadata or {})
    sample = spread_sample_ids(records["ids"], COPY_CONTENT_SAMPLE)
    return {
        **metadata,
        "copy_schema_version": COLLECTION_COPY_SCHEMA_VERSION,
        "copy_of": str(source.name),
        "copy_of_id": str(source.id),
        "copy_source_count": len(records["ids"]),
        "copy_source_metadata_json": canonical_json(metadata),
        "copy_records_hash": records_hash(
            records["ids"], records["documents"], records["metadatas"]
        ),
        "copy_embeddings_sha256": embeddings_sha256(records["embeddings"]),
        # re-read from the source at every load (collection_copy_status)
        "copy_sample_ids_json": canonical_json(sample),
        "copy_sample_sha256": sample_records_sha256(_RecordsView(records), sample),
    }


def create_copy(
    client: Any,
    source: Any,
    target: str,
    records: dict[str, Any],
    *,
    configuration: dict[str, Any],
    batch_size: int = 1000,
) -> Any:
    collection = client.create_collection(
        name=target,
        configuration=configuration,
        metadata=copy_metadata(source, records),
        embedding_function=None,
        get_or_create=False,
    )
    size = max(1, int(batch_size))
    for start in range(0, len(records["ids"]), size):
        stop = start + size
        collection.add(
            ids=records["ids"][start:stop],
            embeddings=records["embeddings"][start:stop],
            documents=records["documents"][start:stop],
            metadatas=records["metadatas"][start:stop],
        )
    return collection


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def ann_recall(
    collection: Any,
    records: dict[str, Any],
    queries: np.ndarray,
    *,
    k: int,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Recall of Chroma's top-k against brute-force cosine over the same (filtered) records.

    A returned id counts as a hit when its exact similarity reaches the exact k-th similarity
    (less a float tolerance), so ties at the boundary are not counted as misses."""
    mask = np.ones(len(records["ids"]), dtype=bool)
    where = None
    if fields:
        mask = np.asarray(
            [(metadata or {}).get("field") in set(fields) for metadata in records["metadatas"]]
        )
        where = {"field": fields[0]} if len(fields) == 1 else {"field": {"$in": list(fields)}}
    ids = [item for item, keep in zip(records["ids"], mask, strict=True) if keep]
    if not ids or not len(queries):
        return {"k": k, "queries": int(len(queries)), "candidates": len(ids), "recall_mean": None}
    position = {item: index for index, item in enumerate(ids)}
    matrix = _unit(records["embeddings"][mask])
    unit = _unit(np.asarray(queries, dtype=np.float32))
    similarity = unit @ matrix.T
    depth = min(int(k), len(ids))
    recalls: list[float] = []
    for row, result in zip(
        similarity,
        (
            hits
            for chunk in batched_query(
                collection, unit, n_results=depth, where=where, include=["distances"]
            )
            for hits in (chunk.get("ids") or [])
        ),
        strict=True,
    ):
        kth = float(np.partition(-row, depth - 1)[depth - 1]) * -1.0
        hits = sum(
            1
            for item in result[:depth]
            if str(item) in position and row[position[str(item)]] >= kth - TIE_TOLERANCE
        )
        recalls.append(min(hits, depth) / depth)
    values = np.asarray(recalls)
    return {
        "k": depth,
        "queries": len(recalls),
        "candidates": len(ids),
        "recall_mean": float(values.mean()),
        "recall_min": float(values.min()),
        "full_recall_share": float(np.mean(values >= 1.0)),
    }


def verify_copy(
    client: Any,
    source: Any,
    target: Any,
    records: dict[str, Any],
    *,
    requested_hnsw: dict[str, Any] | None,
    queries: np.ndarray,
    recall_k: list[int],
    fields: list[str] | None,
    min_recall: float,
    tolerance: float = COPY_EMBEDDING_TOLERANCE,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(label: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": label, "passed": bool(passed), "detail": detail})

    ok, reason = collection_copy_status(target, source, tolerance=tolerance)
    check("copy still mirrors its source (name, id, metadata, count, sampled content)", ok, reason)
    copied = read_records(target)
    check("ids", copied["ids"] == records["ids"], len(copied["ids"]))
    check("documents", copied["documents"] == records["documents"])
    check(
        "record metadata",
        [canonical_json(item or {}) for item in copied["metadatas"]]
        == [canonical_json(item or {}) for item in records["metadatas"]],
    )
    # Chroma re-normalised every vector in float32 when the copy was written: within tolerance
    deviation = embedding_deviation(copied["embeddings"], records["embeddings"])
    check(
        f"embeddings within {tolerance:g} of the source",
        copied["ids"] == records["ids"] and within_tolerance(deviation, tolerance),
        {
            key: deviation.get(key)
            for key in (
                "rows",
                "identical_rows",
                "differing_rows",
                "max_abs_diff",
                "shape_mismatch",
            )
            if key in deviation
        },
    )
    # the source itself bit for bit as it was copied (Chroma returns its stored floats unchanged)
    check(
        "source embeddings unchanged since the copy (copy_embeddings_sha256)",
        embeddings_sha256(records["embeddings"])
        == dict(target.metadata or {}).get("copy_embeddings_sha256"),
    )
    expected_hash = dict(target.metadata or {}).get("copy_records_hash")
    check(
        "records hash",
        records_hash(copied["ids"], copied["documents"], copied["metadatas"]) == expected_hash,
        expected_hash,
    )
    echoed = hnsw_configuration(target)
    report: dict[str, Any] = {
        "hnsw": echoed,
        "tolerance": float(tolerance),
        "embeddings": {
            **deviation,
            "source_norm_range": [
                embedding_deviation(records["embeddings"]).get(key)
                for key in ("norm_min", "norm_max")
            ],
        },
    }
    if requested_hnsw:
        mismatched = {
            key: {"requested": value, "reported": echoed.get(key)}
            for key, value in requested_hnsw.items()
            if echoed.get(key) != value
        }
        check("hnsw configuration echo", not mismatched, mismatched or echoed)
        if mismatched:
            report["advice"] = (
                "the server did not keep the requested HNSW settings: leave this copy alone "
                "(never delete it) and copy under a new name, e.g. with a -r2 suffix"
            )
    recall = {str(k): ann_recall(target, copied, queries, k=k, fields=fields) for k in recall_k}
    for k, entry in recall.items():
        mean = entry.get("recall_mean")
        check(f"ANN recall@{k} >= {min_recall}", mean is not None and mean >= min_recall, mean)
    report.update({"checks": checks, "recall": recall})
    report["ok"] = all(item["passed"] for item in checks)
    return report


def _queries(args: argparse.Namespace, records: dict[str, Any]) -> tuple[np.ndarray, str]:
    if args.query_vectors:
        loaded = [np.load(path, allow_pickle=False) for path in args.query_vectors]
        return np.concatenate([np.atleast_2d(item) for item in loaded]).astype(np.float32), (
            "query vectors " + ", ".join(str(path) for path in args.query_vectors)
        )
    count = len(records["ids"])
    sample = min(int(args.self_queries), count)
    rows = np.random.default_rng(args.seed).choice(count, size=sample, replace=False)
    return records["embeddings"][np.sort(rows)], f"{sample} stored vectors (seed {args.seed})"


def _client(settings: Any) -> Any:
    import chromadb

    return chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port, ssl=settings.chroma_ssl
    )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default=TECHNIQUE_COLLECTION)
    parser.add_argument("--target", help="default: <source>-s<schema>-<srcmeta12>-m<M>-efc<efc>")
    parser.add_argument("--space", default="cosine", choices=["cosine"])
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--ef-search", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--query-vectors", type=Path, nargs="+", help=".npy query embeddings")
    parser.add_argument("--self-queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fields", nargs="+", help="restrict recall to these technique fields")
    parser.add_argument("--recall-k", type=int, nargs="+", default=[90, 300])
    parser.add_argument("--min-recall", type=float, default=0.999)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=COPY_EMBEDDING_TOLERANCE,
        help="largest |diff| of an embedding component against the source (default 1e-6)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--report", type=Path, help="default: data/audits/collection_copy_<target>.json"
    )
    add_output_guard_arguments(parser)
    args = parser.parse_args(argv)

    if not (np.isfinite(args.tolerance) and args.tolerance >= 0.0):
        parser.error(f"--tolerance {args.tolerance} is not a finite number >= 0")
    settings = Settings()
    client = client if client is not None else _client(settings)
    if args.list:
        for collection in client.list_collections():
            metadata = dict(collection.metadata or {})
            print(
                {
                    "name": str(collection.name),
                    "id": str(collection.id),
                    "count": int(collection.count()),
                    "copy_of": metadata.get("copy_of"),
                    "hnsw": hnsw_configuration(collection),
                }
            )
        return 0
    try:
        source = client.get_collection(name=args.source, embedding_function=None)
    except Exception as error:
        parser.error(f"source collection {args.source} unavailable: {error}")
    source_metadata = dict(source.metadata or {})
    if "copy_of" in source_metadata:
        parser.error(f"{args.source} is itself a copy; copy the original collection")
    target_name = args.target or copy_collection_name(
        args.source, source_metadata, args.max_neighbors, args.ef_construction
    )
    problem = name_problem(args.source, target_name)
    if problem:
        parser.error(problem)
    report_path = args.report or (
        REPO_ROOT / "data" / "audits" / f"collection_copy_{target_name}.json"
    )
    if not args.dry_run:
        enforce_output_guard(parser, args, [report_path], repo_root=REPO_ROOT)
    configuration = {
        "hnsw": {
            "space": args.space,
            "max_neighbors": int(args.max_neighbors),
            "ef_construction": int(args.ef_construction),
            "ef_search": int(args.ef_search),
        }
    }
    exists = target_name in {str(collection.name) for collection in client.list_collections()}
    before = (str(source.id), int(source.count()), canonical_json(source_metadata))
    records = read_records(source)
    after = (str(source.id), int(source.count()), canonical_json(dict(source.metadata or {})))
    if before != after:
        print(f"{args.source} changed while it was read; nothing written", file=sys.stderr)
        return 1
    if args.dry_run:
        print(
            f"{'exists, would verify only' if exists else 'would create'}: {target_name}\n"
            f"  from {args.source} ({len(records['ids'])} records, id {source.id})\n"
            f"  configuration {configuration}"
        )
        return 0
    action = "verified"
    if not exists:
        if args.verify_only:
            print(f"collection {target_name} does not exist; nothing to verify", file=sys.stderr)
            return 1
        create_copy(
            client,
            source,
            target_name,
            records,
            configuration=configuration,
            batch_size=args.batch_size,
        )
        action = "created and verified"
    elif not args.verify_only:
        print(f"collection {target_name} exists: verifying it, nothing is written")
    target = client.get_collection(name=target_name, embedding_function=None)
    queries, query_source = _queries(args, records)
    report = verify_copy(
        client,
        source,
        target,
        records,
        requested_hnsw=configuration["hnsw"],
        queries=queries,
        recall_k=list(args.recall_k),
        fields=args.fields,
        min_recall=float(args.min_recall),
        tolerance=float(args.tolerance),
    )
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "action": action,
            "source": {"name": args.source, "id": str(source.id), "count": len(records["ids"])},
            "target": {"name": target_name, "id": str(target.id)},
            "requested_configuration": configuration,
            "queries": query_source,
            "fields": args.fields,
        }
    )
    destination = write_artifact(report_path, report)
    for item in report["checks"]:
        print(f"  {'ok  ' if item['passed'] else 'FAIL'} {item['check']}: {item['detail']}")
    print(f"{action}: {target_name} -> {'ok' if report['ok'] else 'FAILED'}; report {destination}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
