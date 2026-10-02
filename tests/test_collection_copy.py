"""Read-only copies of the ATT&CK collections with other HNSW settings (fake Chroma client)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier.vectorstore import (
    GROUP_COLLECTION,
    TECHNIQUE_COLLECTION,
    TECHNIQUE_FIELDS,
    TECHNIQUE_SCHEMA_VERSION,
    ChromaVectorStore,
    collection_copy_status,
)
from scripts.sync_attack_collection_copy import (
    ann_recall,
    copy_collection_name,
    create_copy,
    name_problem,
    read_records,
)
from scripts.sync_attack_collection_copy import main as copy_main
from tests.chroma_fakes import FakeClient

DIM = 16
FIELDS = ("name", "description", "procedure", "synthetic")


class HashEmbedder:
    def embed(self, texts, batch_size=None):
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.normal(size=DIM).astype(np.float32)
            rows.append((vector / np.linalg.norm(vector)).tolist())
        return rows


def _settings(**overrides):
    return SimpleNamespace(
        attack_version="latest",
        embedding_model="basel/ATTACK-BERT",
        embedding_model_revision="81e30f983a822c606825507b63ae318a6830a8a2",
        **overrides,
    )


def _technique_metadata():
    settings = _settings()
    return {
        "kind": "technique_sentences",
        "schema_version": TECHNIQUE_SCHEMA_VERSION,
        "attack_version": settings.attack_version,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "fields": ",".join(TECHNIQUE_FIELDS),
        "attack_kb_version": "19.2",
        "procedure_documents": 75,
    }


def _seeded(count=300, **client_options):
    client = FakeClient(**client_options)
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(count, DIM)).astype(np.float32)
    vectors[7] = vectors[8]  # duplicate sentences of two techniques: an exact tie
    techniques = client.seed(TECHNIQUE_COLLECTION, metadata=_technique_metadata())
    techniques.add(
        ids=[f"T{1000 + row // 4}:{FIELDS[row % 4]}:{row:04d}" for row in range(count)],
        embeddings=vectors,
        documents=[f"sentence {row}" for row in range(count)],
        metadatas=[
            {"technique_id": f"T{1000 + row // 4}", "field": FIELDS[row % 4], "sentence_index": row}
            for row in range(count)
        ],
    )
    groups = client.seed(
        GROUP_COLLECTION,
        metadata={"kind": "group_profiles", "actor_behavior_count": 20, "schema_version": 2},
    )
    groups.add(
        ids=[f"G{row:04d}:actor-behavior:T1059:0000" for row in range(40)],
        embeddings=rng.normal(size=(40, DIM)).astype(np.float32),
        documents=[f"group behaviour {row}" for row in range(40)],
        metadatas=[
            {"group_id": f"G{row:04d}", "document_type": "actor_behavior"} for row in range(40)
        ],
    )
    client.calls.clear()
    return client


def _copy_args(tmp_path, *extra):
    return [
        "--max-neighbors",
        "32",
        "--ef-construction",
        "400",
        "--ef-search",
        "512",
        "--self-queries",
        "60",
        "--recall-k",
        "10",
        "50",
        "--report",
        str(tmp_path / "copy.json"),
        "--allow-shared-output",  # tmp_path lies outside the repository
        *extra,
    ]


def _target_name(client):
    source = client.collections[TECHNIQUE_COLLECTION]
    return copy_collection_name(TECHNIQUE_COLLECTION, dict(source.metadata), 32, 400)


def test_copy_name_and_refusals():
    metadata = _technique_metadata()
    name = copy_collection_name(TECHNIQUE_COLLECTION, metadata, 32, 400)
    assert name.startswith("attack_techniques-s4-") and name.endswith("-m32-efc400")
    assert len(name) == len("attack_techniques-s4-") + 12 + len("-m32-efc400")
    assert name_problem(TECHNIQUE_COLLECTION, name) is None
    assert name_problem(TECHNIQUE_COLLECTION, TECHNIQUE_COLLECTION)
    assert name_problem(TECHNIQUE_COLLECTION, "technique_copy")
    assert name_problem(TECHNIQUE_COLLECTION, "attack_techniques-" + "x" * 60)


def test_copy_is_bitwise_identical_and_the_source_is_only_read(tmp_path):
    client = _seeded()
    assert (
        copy_main(
            _copy_args(tmp_path, "--fields", "name", "description", "procedure"), client=client
        )
        == 0
    )
    target = _target_name(client)
    writes = client.writes()
    assert writes[0] == ("create_collection", target, {"get_or_create": False})
    assert {method for method, _, _ in writes} == {"create_collection", "add"}
    assert all(name == target for _, name, _ in writes)  # nothing written to the source
    source = client.collections[TECHNIQUE_COLLECTION]
    copy = client.collections[target]
    original, copied = read_records(source), read_records(copy)
    assert copied["ids"] == original["ids"] and copied["documents"] == original["documents"]
    assert copied["metadatas"] == original["metadatas"]
    assert np.array_equal(
        copied["embeddings"].view(np.uint32), original["embeddings"].view(np.uint32)
    )
    assert copy.configuration_json["hnsw"] == {
        "space": "cosine",
        "ef_construction": 400,
        "ef_search": 512,
        "max_neighbors": 32,
    }
    # the copy keeps the source metadata keys and records where it came from
    assert {key: copy.metadata[key] for key in source.metadata} == source.metadata
    assert copy.metadata["copy_of"] == TECHNIQUE_COLLECTION
    assert copy.metadata["copy_of_id"] == source.id
    assert collection_copy_status(copy, source) == (True, "ok")
    report = json.loads((tmp_path / "copy.json").read_text())
    assert report["ok"] and report["recall"]["10"]["recall_mean"] == 1.0
    assert report["recall"]["50"]["candidates"] == 225  # name, description, procedure only
    # a second run finds the copy and only verifies it
    client.calls.clear()
    assert copy_main(_copy_args(tmp_path, "--overwrite"), client=client) == 0
    assert client.writes() == []


def test_dry_run_and_verify_only_never_write(tmp_path):
    client = _seeded()
    assert copy_main(_copy_args(tmp_path, "--dry-run"), client=client) == 0
    assert copy_main(_copy_args(tmp_path, "--verify-only"), client=client) == 1  # nothing to verify
    assert client.writes() == [] and not (tmp_path / "copy.json").exists()


def test_refuses_bad_targets_and_copies_of_copies(tmp_path):
    client = _seeded()
    for target in (TECHNIQUE_COLLECTION, "techniques_fast", "attack_groups-copy"):
        with pytest.raises(SystemExit):
            copy_main(_copy_args(tmp_path, "--target", target), client=client)
    assert copy_main(_copy_args(tmp_path), client=client) == 0
    copy = client.collections[_target_name(client)]
    client.collections["attack_techniques"] = copy  # pretend the source is itself a copy
    with pytest.raises(SystemExit):
        copy_main(_copy_args(tmp_path, "--overwrite"), client=client)
    assert all(method != "delete_collection" for method in client.methods())


def test_ann_recall_counts_misses_and_tolerates_ties(tmp_path):
    client = _seeded()
    source = client.collections[TECHNIQUE_COLLECTION]
    records = read_records(source)
    target = create_copy(
        client,
        source,
        _target_name(client),
        records,
        configuration={"hnsw": {"space": "cosine"}},
    )
    queries = records["embeddings"][[7, 8, 20, 40]]
    full = ann_recall(target, records, queries, k=5)
    assert full["recall_mean"] == 1.0 and full["full_recall_share"] == 1.0
    # an ANN index that lost the nearest record of every query
    target.hidden_ids = {records["ids"][row] for row in (7, 8, 20, 40)}
    lossy = ann_recall(target, records, queries, k=5)
    assert lossy["recall_mean"] < 1.0 and lossy["recall_min"] <= 0.8
    # verification fails on recall, never deleting anything
    assert (
        copy_main(
            _copy_args(tmp_path, "--verify-only", "--min-recall", "1.0", "--self-queries", "300"),
            client=client,
        )
        == 1
    )


def test_a_server_that_ignored_the_build_settings_fails_verification(tmp_path):
    client = _seeded(ignore_build_settings=True)
    assert copy_main(_copy_args(tmp_path), client=client) == 1
    report = json.loads((tmp_path / "copy.json").read_text())
    failed = [item["check"] for item in report["checks"] if not item["passed"]]
    assert failed == ["hnsw configuration echo"] and "-r2" in report["advice"]
    assert all(
        method not in {"delete_collection", "delete", "modify"} for method in client.methods()
    )


def _store(client, **settings):
    return ChromaVectorStore(_settings(**settings), HashEmbedder(), client=client)


def test_vector_store_reads_a_configured_copy_with_get_collection_only(tmp_path):
    client = _seeded()
    assert copy_main(_copy_args(tmp_path), client=client) == 0
    target = _target_name(client)
    default = _store(client)
    expected = default.query_techniques(
        ["credential dumping"], n_results=20, fields=("name", "procedure")
    )
    client.calls.clear()
    store = _store(client, technique_collection_name=target)
    hits = store.query_techniques(
        ["credential dumping"], n_results=20, fields=("name", "procedure")
    )
    assert [hit.id for hit in hits[0]] == [hit.id for hit in expected[0]]
    assert client.writes() == []  # no get_or_create for the copy
    assert store.technique_collection is client.collections[target]
    # the group collection is unchanged unless configured too
    assert store.group_collection is client.collections[GROUP_COLLECTION]


def test_group_collection_copy(tmp_path):
    client = _seeded()
    args = _copy_args(tmp_path, "--source", "attack_groups", "--recall-k", "5")
    assert copy_main(args, client=client) == 0
    source = client.collections[GROUP_COLLECTION]
    target = copy_collection_name(GROUP_COLLECTION, dict(source.metadata), 32, 400)
    client.calls.clear()
    store = _store(client, group_collection_name=target)
    hits = store.query_group_behaviors(["group behaviour 3"], n_results=5)
    assert len(hits[0]) == 5 and client.writes() == []


@pytest.mark.parametrize(
    "change", ["metadata", "count", "recreated", "reembedded", "rewritten", "unsampled"]
)
def test_a_stale_copy_is_refused(tmp_path, change):
    client = _seeded()
    assert copy_main(_copy_args(tmp_path), client=client) == 0
    target = _target_name(client)
    source = client.collections[TECHNIQUE_COLLECTION]
    copy = client.collections[target]
    assert collection_copy_status(copy, source) == (True, "ok")
    sampled = json.loads(copy.metadata["copy_sample_ids_json"])
    assert 0 < len(sampled) <= 64
    if change == "metadata":
        source.metadata = {**source.metadata, "attack_kb_version": "19.3"}
    elif change == "count":
        source.add(
            ids=["T9999:name:0000"],
            embeddings=np.ones((1, DIM), dtype=np.float32),
            documents=["new"],
            metadatas=[{"technique_id": "T9999", "field": "name"}],
        )
    elif change == "reembedded":
        # a compatible make ingest upserting in place: same id, ids, count and metadata, the
        # same text re-embedded with tiny float differences (another device)
        for item_id in list(source._records):
            vector = source._records[item_id][0]
            source.tamper(item_id, embedding=vector + np.float32(1e-6))
    elif change == "rewritten":
        # a regenerated sentence with the same per-technique count
        source.tamper(sampled[3], document="a regenerated synthetic sentence")
    elif change == "unsampled":
        # a copy from before the content sample (schema 1) is refused, not trusted
        copy.metadata = {
            key: value
            for key, value in copy.metadata.items()
            if key not in ("copy_sample_ids_json", "copy_sample_sha256")
        }
    else:
        source.id = "a-new-id"
    if change in ("reembedded", "rewritten"):
        ok, reason = collection_copy_status(copy, source)
        assert not ok and "content changed" in reason
    if change == "unsampled":
        assert collection_copy_status(copy, source) == (
            False,
            "copy records no content sample of its source",
        )
    store = _store(client, technique_collection_name=target)
    with pytest.raises(RuntimeError, match="cannot stand in"):
        store.query_techniques(["credential dumping"], n_results=5)
    missing = _store(client, technique_collection_name="attack_techniques-missing")
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = missing.technique_collection


def test_ingest_refuses_to_write_while_a_copy_is_configured(tmp_path):
    client = _seeded()
    store = _store(client, technique_collection_name="attack_techniques-anything")
    with pytest.raises(RuntimeError, match="ingestion writes only"):
        store.upsert(SimpleNamespace(techniques={}, groups={}))
    assert client.writes() == []
