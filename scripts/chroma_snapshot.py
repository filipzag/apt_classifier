"""Read-only snapshot of the shared ChromaDB server, and a diff between two snapshots.

The Chroma server on localhost:8000 is shared by every checkout and session, and
``make ingest``, ``ChromaVectorStore.upsert`` or ``reset`` replace its collections. Wrap every
heavy run in a snapshot and a diff so a collection that changed under a run is caught:

    PYTHONPATH=. .venv/bin/python scripts/chroma_snapshot.py --output before.json
    ... run ...
    PYTHONPATH=. .venv/bin/python scripts/chroma_snapshot.py --diff before.json --output after.json

A snapshot records, for every collection, its name, id, count, metadata and
``configuration_json``. By default (``--no-hash-content`` turns it off) it also records the
content: a SHA-256 of the sorted record ids, a SHA-256 over every record's id, document and
canonical metadata (``records_sha256``), and a SHA-256 of the float32 embeddings of a fixed
sample of ``--embedding-sample`` records spread over the sorted ids. A compatible ``make ingest``
re-upserts every record in place (same collection id, ids, count and metadata), so only the
content hashes see a re-embedded or rewritten library; an HNSW graph rebuilt over identical
vectors stays invisible. ``--hash-ids`` alone records just the ids hash. It only reads:
``list_collections``, ``count``, ``get`` and the server version; it never creates, modifies or
deletes anything.

``--diff BEFORE`` compares BEFORE with a fresh snapshot (or with ``--after FILE``, offline) and
exits 1 when a collection that existed before changed or disappeared, when a new collection
appeared whose name lacks an allowed prefix (``technique_exemplars-`` by default), or when the
other snapshot lacks a hash BEFORE recorded; 0 otherwise. The baseline is read before anything is
written, the fresh snapshot records at least what BEFORE recorded, and ``--output`` /
``--diff-output`` may neither be BEFORE (or ``--after``) nor replace an existing file without
``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALLOWED_NEW_PREFIXES = ("technique_exemplars-",)
# content hashes, compared only when the BEFORE snapshot recorded them
CONTENT_FIELDS = ("ids_sha256", "records_sha256", "embeddings_sample_sha256")
COMPARED_FIELDS = ("id", "count", "metadata", "configuration_json", *CONTENT_FIELDS)
ID_PAGE = 5000
CONTENT_PAGE = 1000
EMBEDDING_SAMPLE = 256


def _jsonable(value: Any) -> Any:
    """Collection metadata and configuration as plain JSON (sorted, stringified leaves)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


def _sorted_ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(ids):
        digest.update(item.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _ids_sha256(collection: Any, count: int) -> str:
    """SHA-256 of the collection's sorted record ids, read in pages (ids only, no vectors)."""
    ids: list[str] = []
    for offset in range(0, max(count, 0), ID_PAGE):
        page = collection.get(include=[], limit=ID_PAGE, offset=offset)
        ids.extend(str(item) for item in (page.get("ids") or []))
    return _sorted_ids_sha256(ids)


def _content_hashes(collection: Any, count: int, *, sample: int) -> dict[str, Any]:
    """Hashes of what a collection holds: its ids, every record's id, document and canonical
    metadata, and the float32 embeddings of a fixed sample of ids spread over the sorted ids
    (documents and metadata are read in pages; embeddings only for the sample)."""
    from apt_classifier.vectorstore import records_hash, spread_sample_ids

    rows: dict[str, tuple[Any, Any]] = {}
    duplicates = 0
    for offset in range(0, max(count, 0), CONTENT_PAGE):
        page = collection.get(include=["documents", "metadatas"], limit=CONTENT_PAGE, offset=offset)
        ids = [str(item) for item in (page.get("ids") or [])]
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        for item_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            duplicates += int(item_id in rows)
            rows[item_id] = (document, metadata)
    ids = sorted(rows)
    sampled = spread_sample_ids(ids, sample)
    vectors: dict[str, Any] = {}
    if sampled:
        page = collection.get(ids=sampled, include=["embeddings"])
        returned = page.get("embeddings")
        for position, item_id in enumerate(str(item) for item in (page.get("ids") or [])):
            if returned is not None and position < len(returned):
                vectors[item_id] = returned[position]
    digest = hashlib.sha256()
    for item_id in sampled:
        digest.update(item_id.encode())
        vector = vectors.get(item_id)
        if vector is None:
            digest.update(b"<missing>")
        else:
            digest.update(np.ascontiguousarray(np.asarray(vector, dtype=np.float32)).tobytes())
        digest.update(b"\n")
    return {
        "ids_sha256": _sorted_ids_sha256(ids),
        "records_sha256": records_hash(
            ids, [rows[item][0] for item in ids], [rows[item][1] for item in ids]
        ),
        "records_read": len(ids),
        "records_read_twice": duplicates,
        "embeddings_sample_sha256": digest.hexdigest(),
        "embeddings_sample_size": len(sampled),
    }


def snapshot_hash(collections: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(collections, sort_keys=True).encode()).hexdigest()


def take_snapshot(
    client: Any,
    *,
    hash_ids: bool = False,
    hash_content: bool = False,
    embedding_sample: int = EMBEDDING_SAMPLE,
) -> dict[str, Any]:
    """Describe every collection on the server; reads only."""
    collections: dict[str, Any] = {}
    for collection in client.list_collections():
        name = str(collection.name)
        count = int(collection.count())
        entry: dict[str, Any] = {
            "id": str(collection.id),
            "count": count,
            "metadata": _jsonable(dict(collection.metadata or {})),
            "configuration_json": _jsonable(getattr(collection, "configuration_json", None)),
        }
        if hash_content:
            entry.update(_content_hashes(collection, count, sample=embedding_sample))
        elif hash_ids:
            entry["ids_sha256"] = _ids_sha256(collection, count)
        collections[name] = entry
    try:
        server_version = str(client.get_version())
    except Exception:  # a version is provenance, never a reason to fail
        server_version = None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "server_version": server_version,
        "options": {
            "hash_ids": bool(hash_ids or hash_content),
            "hash_content": bool(hash_content),
            "embedding_sample": int(embedding_sample) if hash_content else None,
        },
        "collections": collections,
        "snapshot_hash": snapshot_hash(collections),
    }


def recorded_options(snapshot: dict[str, Any]) -> dict[str, Any]:
    """What a snapshot hashed (its ``options``, or for an older file the fields it carries)."""
    options = dict(snapshot.get("options") or {})
    entries = list((snapshot.get("collections") or {}).values())
    options.setdefault("hash_ids", any("ids_sha256" in entry for entry in entries))
    options.setdefault("hash_content", any("records_sha256" in entry for entry in entries))
    if options.get("embedding_sample") is None:
        sizes = [
            entry["embeddings_sample_size"]
            for entry in entries
            if "embeddings_sample_size" in entry
        ]
        options["embedding_sample"] = max(sizes) if sizes else EMBEDDING_SAMPLE
    return options


def diff_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    allowed_new_prefixes: Sequence[str] = ALLOWED_NEW_PREFIXES,
) -> dict[str, Any]:
    """What changed between two snapshots; ``clean`` is False on any change that matters."""
    old = before.get("collections") or {}
    new = after.get("collections") or {}
    changed: dict[str, dict[str, Any]] = {}
    not_compared: dict[str, list[str]] = {}
    for name in sorted(set(old) & set(new)):
        fields = {
            field: {"before": old[name].get(field), "after": new[name].get(field)}
            for field in COMPARED_FIELDS
            # a content hash is compared only when BEFORE recorded one (an older baseline)
            if (field not in CONTENT_FIELDS or (field in old[name] and field in new[name]))
            and old[name].get(field) != new[name].get(field)
        }
        if fields:
            changed[name] = fields
        # a hash BEFORE recorded but the other snapshot lacks: the check cannot be made
        missing = [
            field for field in CONTENT_FIELDS if field in old[name] and field not in new[name]
        ]
        if missing:
            not_compared[name] = missing
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    unexpected = [name for name in added if not name.startswith(tuple(allowed_new_prefixes))]
    return {
        "clean": not changed and not removed and not unexpected and not not_compared,
        "changed": changed,
        "not_compared": not_compared,
        "removed": removed,
        "added": added,
        "unexpected_added": unexpected,
        "before_hash": before.get("snapshot_hash"),
        "after_hash": after.get("snapshot_hash"),
    }


def _client(settings: Any) -> Any:
    import chromadb

    return chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port, ssl=settings.chroma_ssl
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _same_file(left: Path | None, right: Path | None) -> bool:
    return left is not None and right is not None and left.resolve() == right.resolve()


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the (fresh) snapshot here")
    parser.add_argument("--diff", type=Path, metavar="BEFORE", help="snapshot to compare against")
    parser.add_argument(
        "--after", type=Path, help="with --diff: compare with this snapshot file (no server)"
    )
    parser.add_argument(
        "--hash-ids", action="store_true", help="also hash every collection's record ids"
    )
    parser.add_argument(
        "--hash-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hash ids, documents, metadata and a sample of embeddings (default: on)",
    )
    parser.add_argument(
        "--embedding-sample",
        type=int,
        default=EMBEDDING_SAMPLE,
        help=f"records whose embeddings are hashed (default {EMBEDDING_SAMPLE})",
    )
    parser.add_argument(
        "--allow-new-prefix",
        action="append",
        default=None,
        help=f"name prefix a new collection may have (default: {', '.join(ALLOWED_NEW_PREFIXES)})",
    )
    parser.add_argument("--diff-output", type=Path, help="with --diff: write the diff here")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing --output / --diff-output"
    )
    args = parser.parse_args(argv)
    if args.after is not None and args.diff is None:
        parser.error("--after needs --diff")
    if args.after is not None and args.output is not None:
        parser.error("--after compares two files; there is no fresh snapshot to --output")
    if args.diff_output is not None and args.diff is None:
        parser.error("--diff-output needs --diff")
    # never write over a snapshot this run compares (a copy-pasted "before" command would
    # otherwise replace the baseline before it is read and report clean)
    for label, path in (("--output", args.output), ("--diff-output", args.diff_output)):
        for other_label, other in (("--diff", args.diff), ("--after", args.after)):
            if _same_file(path, other):
                parser.error(f"{label} {path} is the {other_label} snapshot; write elsewhere")
        if path is not None and path.exists() and not args.overwrite:
            parser.error(f"{label} {path} already exists; pass --overwrite to replace it")
    if _same_file(args.output, args.diff_output):
        parser.error("--output and --diff-output name the same file")
    before = json.loads(args.diff.read_text(encoding="utf-8")) if args.diff is not None else None
    if args.after is not None:
        after = json.loads(args.after.read_text(encoding="utf-8"))
    else:
        if client is None:
            from apt_classifier.config import Settings

            client = _client(Settings())
        # a diff hashes at least what its baseline hashed
        baseline = recorded_options(before) if before is not None else {}
        # the baseline's sample size, so both snapshots hash the same records
        sample = baseline["embedding_sample"] if baseline.get("hash_content") else None
        after = take_snapshot(
            client,
            hash_ids=bool(args.hash_ids or baseline.get("hash_ids")),
            hash_content=bool(args.hash_content or baseline.get("hash_content")),
            embedding_sample=int(sample if sample is not None else args.embedding_sample),
        )
        if args.output is not None:
            _write(args.output, after)
    if before is None:
        summary = {name: entry["count"] for name, entry in after["collections"].items()}
        print(json.dumps({"snapshot_hash": after["snapshot_hash"], "counts": summary}, indent=2))
        return 0
    report = diff_snapshots(
        before, after, allowed_new_prefixes=tuple(args.allow_new_prefix or ALLOWED_NEW_PREFIXES)
    )
    if args.diff_output is not None:
        _write(args.diff_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
