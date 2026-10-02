"""Copy the npz exemplar library into a new ChromaDB collection once, and verify the copy.

This is the only write the exemplar migration makes to the shared Chroma server: one new
collection, created with ``create_collection(get_or_create=False)`` (it fails if the name
exists) and filled with ``add`` in batches. Records carry the npz bytes (the embedding rows are
copied, never re-embedded); see ``apt_classifier/exemplar_store.py`` for the schema. The script
never calls ``delete``, ``modify``, ``upsert`` or ``update``, and an existing name is only
verified.

    PYTHONPATH=. .venv/bin/python scripts/sync_exemplars_to_chroma.py --list
    PYTHONPATH=. .venv/bin/python scripts/sync_exemplars_to_chroma.py --collection NAME --dry-run
    PYTHONPATH=. .venv/bin/python scripts/sync_exemplars_to_chroma.py --collection NAME
    PYTHONPATH=. .venv/bin/python scripts/sync_exemplars_to_chroma.py --collection NAME --verify-only

Refused: a library not built from the public train splits of tumeteor/mitre-ttp-mapping (or one
that includes the leaky ``procedures`` split), and names without the ``technique_exemplars-``
prefix. Verification (after a sync, or alone) checks the count and row set, the embeddings bit
for bit (``view(np.uint32)``), labels, texts and sources against the npz, the recomputed
fingerprint and record hash, the HNSW configuration echo, and self-recall: for every exemplar the
exact window (``2k+16``) must be inside Chroma's top ``ceil(window * overfetch)``. A server that
ignored ``max_neighbors`` or ``ef_construction`` fails the echo check: leave that collection
alone and sync under a new name (``-r2`` suffix). Exit 0 when every check passes, 1 otherwise,
2 when refused.
"""

from __future__ import annotations

import argparse
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
from apt_classifier.exemplar_store import (  # noqa: E402
    COLLECTION_PREFIX,
    clear_memo,
    collection_metadata,
    exemplar_collection_name,
    exemplar_records,
    load_chroma_exemplar_store,
)
from apt_classifier.exemplars import ExemplarIndex, window_size  # noqa: E402
from apt_classifier.vectorstore import hnsw_configuration, records_hash  # noqa: E402
from scripts.run_tooling import (  # noqa: E402
    add_output_guard_arguments,
    enforce_output_guard,
    file_sha256,
)

LIBRARY_SOURCE = "tumeteor/mitre-ttp-mapping"
DEFAULT_REPORT = REPO_ROOT / "data" / "audits" / "exemplar_chroma_sync.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")


def library_problems(index: ExemplarIndex) -> list[str]:
    """Why a library must not be copied into Chroma (empty when it may)."""
    metadata = index.metadata
    problems = []
    if metadata.get("split") != "train":
        problems.append(f"built from split {metadata.get('split')!r}, only train is allowed")
    if metadata.get("source") != LIBRARY_SOURCE:
        problems.append(f"built from {metadata.get('source')!r}, only {LIBRARY_SOURCE}")
    if "procedures" in (metadata.get("datasets") or []):
        problems.append("includes the leaky procedures split")
    return problems


def name_problem(name: str) -> str | None:
    if not name.startswith(COLLECTION_PREFIX):
        return f"collection name {name!r} must start with {COLLECTION_PREFIX!r}"
    if not _NAME_RE.match(name):
        return f"collection name {name!r} is not a valid Chroma name (3-63 of [A-Za-z0-9._-])"
    return None


def creation_configuration(m: int, ef_construction: int, ef_search: int) -> dict[str, Any]:
    return {
        "hnsw": {
            "space": "cosine",
            "max_neighbors": int(m),
            "ef_construction": int(ef_construction),
            "ef_search": int(ef_search),
        }
    }


def create_and_fill(
    client: Any,
    index: ExemplarIndex,
    name: str,
    *,
    configuration: dict[str, Any],
    source_npz_sha256: str | None,
    batch_size: int = 1000,
) -> Any:
    """Create the collection (fails if it exists) and add every exemplar row in batches."""
    ids, documents, metadatas = exemplar_records(index)
    collection = client.create_collection(
        name=name,
        configuration=configuration,
        metadata=collection_metadata(
            index,
            source_npz_sha256=source_npz_sha256,
            records=records_hash(ids, documents, metadatas),
        ),
        embedding_function=None,
        get_or_create=False,
    )
    embeddings = np.ascontiguousarray(index.embeddings, dtype=np.float32)
    for start in range(0, len(ids), max(1, int(batch_size))):
        stop = start + max(1, int(batch_size))
        collection.add(
            ids=ids[start:stop],
            embeddings=embeddings[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )
    return collection


def verify(
    client: Any,
    name: str,
    index: ExemplarIndex,
    *,
    index_path: Path | None,
    requested_hnsw: dict[str, Any],
    recall_k: list[int],
    overfetch: float,
) -> dict[str, Any]:
    """Every check of the copy against the npz library; ``report["ok"]`` when all pass."""
    checks: list[dict[str, Any]] = []

    def check(label: str, passed: bool, detail: Any = None) -> bool:
        checks.append({"check": label, "passed": bool(passed), "detail": detail})
        return bool(passed)

    clear_memo()  # verify a fresh download, never a memoised copy
    probe = argparse.Namespace(
        technique_exemplar_collection=name,
        technique_exemplar_index_path=index_path,
        technique_exemplar_backend="chroma",
        technique_exemplar_search="hnsw",
        technique_exemplar_overfetch=float(overfetch),
        # the echo check below reports every HNSW setting, ef_search included
        technique_exemplar_hnsw_ef_search=0,
        embedding_model=index.metadata.get("embedding_model"),
        embedding_model_revision=index.metadata.get("embedding_revision"),
    )
    store, reason = load_chroma_exemplar_store(probe, client)
    report: dict[str, Any] = {"collection": name, "checks": checks}
    if not check(
        "loads fail-closed (kind, schema, count, rows, hashes, fingerprint)",
        store is not None,
        reason,
    ):
        report["ok"] = False
        return report
    copy = store.index
    report["collection_id"] = store.collection_id
    report["server_version"] = store.server_version
    check("count", len(copy) == len(index), {"chroma": len(copy), "npz": len(index)})
    check(
        "embeddings bitwise",
        copy.embeddings.shape == index.embeddings.shape
        and np.array_equal(
            np.ascontiguousarray(copy.embeddings, dtype=np.float32).view(np.uint32),
            np.ascontiguousarray(index.embeddings, dtype=np.float32).view(np.uint32),
        ),
    )
    check("labels", copy.labels == index.labels)
    check("texts", copy.texts == index.texts)
    check("sources", copy.sources == index.sources)
    check(
        "fingerprint",
        copy.compute_fingerprint() == index.fingerprint == store.fingerprint,
        {"chroma": copy.compute_fingerprint(), "npz": index.fingerprint},
    )
    ids, documents, metadatas = exemplar_records(index)
    check(
        "records hash",
        records_hash(ids, documents, metadatas) == store.records_hash,
        store.records_hash,
    )
    echoed = hnsw_configuration(store.collection)
    mismatched = {
        key: {"requested": value, "reported": echoed.get(key)}
        for key, value in requested_hnsw.items()
        if echoed.get(key) != value
    }
    if not check("hnsw configuration echo", not mismatched, mismatched or echoed):
        report["advice"] = (
            "the server did not keep the requested HNSW settings: leave this collection alone "
            "(never delete it) and sync under a new name, e.g. with a -r2 suffix"
        )
    recall: dict[str, Any] = {}
    for k in recall_k:
        window = window_size(len(index), k)
        exact = index.exact_windows(index.embeddings, k=k)
        found = store.candidate_rows(index.embeddings, k=k)
        contained = [bool(np.isin(e, f).all()) for e, f in zip(exact, found, strict=True)]
        raw = [
            len(np.intersect1d(e, f[:window])) / max(1, window)
            for e, f in zip(exact, found, strict=True)
        ]
        misses = len(contained) - sum(contained)
        recall[str(k)] = {
            "window": window,
            "n_results": int(len(found[0])) if found else 0,
            "queries": len(contained),
            "recall_misses": misses,
            "raw_recall_at_window_mean": float(np.mean(raw)) if raw else None,
            "raw_recall_at_window_min": float(np.min(raw)) if raw else None,
        }
        check(f"self-recall k={k} (exact window inside Chroma's results)", misses == 0, misses)
    report["self_recall"] = recall
    report["ok"] = all(item["passed"] for item in checks)
    return report


def list_rows(client: Any) -> list[dict[str, Any]]:
    rows = []
    for collection in client.list_collections():
        metadata = dict(collection.metadata or {})
        rows.append(
            {
                "name": str(collection.name),
                "id": str(collection.id),
                "count": int(collection.count()),
                "kind": metadata.get("kind"),
                "fingerprint": metadata.get("fingerprint"),
                "hnsw": hnsw_configuration(collection),
            }
        )
    return rows


def _client(settings: Any) -> Any:
    import chromadb

    return chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port, ssl=settings.chroma_ssl
    )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", help="default: technique_exemplars-<fingerprint>-s1-m<M>-efc<efc>"
    )
    parser.add_argument("--index", type=Path, help="default: TECHNIQUE_EXEMPLAR_INDEX_PATH")
    parser.add_argument("--m", type=int, help="HNSW max_neighbors (TECHNIQUE_EXEMPLAR_HNSW_M)")
    parser.add_argument("--ef-construction", type=int)
    parser.add_argument("--ef-search", type=int)
    parser.add_argument("--overfetch", type=float, help="default: TECHNIQUE_EXEMPLAR_OVERFETCH")
    parser.add_argument(
        "--recall-k", type=int, nargs="+", help="k values for self-recall (default: the k)"
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="show what would be created")
    parser.add_argument("--verify-only", action="store_true", help="never create; verify only")
    parser.add_argument("--list", action="store_true", help="list the server's collections")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    add_output_guard_arguments(parser)
    args = parser.parse_args(argv)

    settings = Settings()
    client = client if client is not None else _client(settings)
    if args.list:
        for row in list_rows(client):
            print(row)
        return 0
    index_path = Path(args.index or settings.technique_exemplar_index_path)
    try:
        index = ExemplarIndex.load(index_path)
    except (ValueError, KeyError, OSError) as error:
        parser.error(f"invalid exemplar index {index_path}: {error}")
    if index is None:
        parser.error(f"exemplar index not found: {index_path}")
    problems = library_problems(index)
    ok, reason = index.compatible_with(settings.embedding_model, settings.embedding_model_revision)
    if not ok:
        problems.append(f"incompatible with the configured embedder: {reason}")
    if problems:
        parser.error("refusing to sync: " + "; ".join(problems))
    m = int(args.m or settings.technique_exemplar_hnsw_m)
    ef_construction = int(args.ef_construction or settings.technique_exemplar_hnsw_ef_construction)
    ef_search = int(args.ef_search or settings.technique_exemplar_hnsw_ef_search)
    name = args.collection or exemplar_collection_name(index.fingerprint, m, ef_construction)
    problem = name_problem(name)
    if problem:
        parser.error(problem)
    configuration = creation_configuration(m, ef_construction, ef_search)
    if not args.dry_run:
        enforce_output_guard(parser, args, [args.report], repo_root=REPO_ROOT)
    exists = name in {str(collection.name) for collection in client.list_collections()}
    if args.dry_run:
        ids, documents, metadatas = exemplar_records(index)
        batches = -(-len(ids) // max(1, args.batch_size))
        print(
            f"{'exists, would verify only' if exists else 'would create'}: {name}\n"
            f"  configuration {configuration}\n"
            f"  {len(ids)} records in {batches} add() batches, fingerprint {index.fingerprint}, "
            f"records hash {records_hash(ids, documents, metadatas)}"
        )
        return 0
    action = "verified"
    if not exists:
        if args.verify_only:
            print(f"collection {name} does not exist; nothing to verify", file=sys.stderr)
            return 1
        create_and_fill(
            client,
            index,
            name,
            configuration=configuration,
            source_npz_sha256=file_sha256(index_path),
            batch_size=args.batch_size,
        )
        action = "created and verified"
    elif not args.verify_only:
        print(f"collection {name} exists: verifying it, nothing is written")
    report = verify(
        client,
        name,
        index,
        index_path=index_path,
        requested_hnsw=configuration["hnsw"],
        recall_k=list(args.recall_k or [int(settings.technique_exemplar_k)]),
        overfetch=float(args.overfetch or settings.technique_exemplar_overfetch),
    )
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "action": action,
            "npz": str(index_path),
            "npz_sha256": file_sha256(index_path),
            "fingerprint": index.fingerprint,
            "requested_configuration": configuration,
        }
    )
    destination = write_artifact(args.report, report)
    for item in report["checks"]:
        print(f"  {'ok  ' if item['passed'] else 'FAIL'} {item['check']}")
    print(f"{action}: {name} -> {'ok' if report['ok'] else 'FAILED'}; report {destination}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
