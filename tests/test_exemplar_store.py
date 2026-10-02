"""The exemplar library served from ChromaDB (fake client; no server, no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier import exemplar_store, vectorstore
from apt_classifier.exemplar_store import (
    ChromaExemplarStore,
    ExemplarSource,
    collection_metadata,
    configured_collection_name,
    exemplar_collection_name,
    exemplar_records,
    load_chroma_exemplar_store,
    query_totals,
    reset_query_totals,
)
from apt_classifier.exemplars import ExemplarIndex, load_exemplar_index
from apt_classifier.vectorstore import records_hash
from scripts.sync_exemplars_to_chroma import (
    create_and_fill,
    creation_configuration,
    library_problems,
    name_problem,
)
from scripts.sync_exemplars_to_chroma import main as sync_main
from scripts.sync_exemplars_to_chroma import verify as sync_verify
from tests.chroma_fakes import FakeClient
from tests.test_exemplars import _random_library, _vote_bits

MODEL = "test-model"
REVISION = "r1"


@pytest.fixture(autouse=True)
def _fresh_memo():
    exemplar_store.clear_memo()
    reset_query_totals()
    yield
    exemplar_store.clear_memo()


def _library(count=400):
    index, queries = _random_library(count=count)
    index.metadata.update(
        {
            "embedding_model": MODEL,
            "embedding_revision": REVISION,
            "attack_version": "19.2",
            "count": len(index),
            "labels": len({label for labels in index.labels for label in labels}),
            "split": "train",
            "source": "tumeteor/mitre-ttp-mapping",
            "datasets": ["derived_procedures", "expert", "tram"],
            "license": "CC-BY-4.0",
        }
    )
    index.metadata["fingerprint"] = index.compute_fingerprint()
    return index, queries


def _synced(*, client=None, index=None, ef_search=8192, name=None, **client_options):
    client = client or FakeClient(**client_options)
    index = index or _library()[0]
    name = name or exemplar_collection_name(index.fingerprint, 32, 400)
    create_and_fill(
        client,
        index,
        name,
        configuration=creation_configuration(32, 400, ef_search),
        source_npz_sha256="abc",
        batch_size=128,
    )
    client.calls.clear()
    return client, index, name


def _settings(name, *, backend="chroma", search="exact", index_path=None, **extra):
    return SimpleNamespace(
        technique_exemplars_enabled=True,
        technique_exemplar_backend=backend,
        technique_exemplar_collection=name,
        technique_exemplar_search=search,
        technique_exemplar_index_path=index_path,
        technique_exemplar_hnsw_m=32,
        technique_exemplar_hnsw_ef_construction=400,
        technique_exemplar_hnsw_ef_search=8192,
        technique_exemplar_overfetch=2.0,
        embedding_model=MODEL,
        embedding_model_revision=REVISION,
        **extra,
    )


def test_collection_name_and_record_schema():
    index, _ = _library(count=50)
    name = exemplar_collection_name("0f8d2e19a59967c63b03ea71", 32, 400)
    assert name == "technique_exemplars-0f8d2e19a59967c63b03ea71-s1-m32-efc400"
    assert len(name) == 58 and name_problem(name) is None
    assert name_problem("attack_exemplars-x") and name_problem("technique_exemplars-" + "x" * 60)
    ids, documents, metadatas = exemplar_records(index)
    assert ids[0] == "ex-00000" and ids[-1] == "ex-00049" and documents == list(index.texts)
    assert metadatas[3] == {
        "row": 3,
        "labels": json.dumps(list(index.labels[3])),
        "label_count": len(index.labels[3]),
        "source": index.sources[3],
    }
    metadata = collection_metadata(index, source_npz_sha256=None, records="h")
    assert metadata["kind"] == "technique_exemplars" and metadata["count"] == 50
    assert json.loads(metadata["source_metadata_json"]) == index.metadata
    assert not any(value is None for value in metadata.values())  # Chroma scalars only
    assert library_problems(index) == []
    leaky = ExemplarIndex(
        index.embeddings,
        index.labels,
        index.texts,
        index.sources,
        {**index.metadata, "split": "dev", "datasets": ["procedures"]},
    )
    assert len(library_problems(leaky)) == 2


def test_sync_round_trip_is_bitwise_identical_and_votes_match_the_npz():
    client, index, name = _synced()
    store, reason = load_chroma_exemplar_store(_settings(name), client)
    assert reason == "ok" and isinstance(store, ChromaExemplarStore)
    assert isinstance(store, ExemplarSource)
    copy = store.index
    assert np.array_equal(copy.embeddings.view(np.uint32), index.embeddings.view(np.uint32))
    assert copy.labels == index.labels and copy.texts == index.texts
    assert copy.sources == index.sources and copy.metadata == index.metadata
    assert store.fingerprint == index.fingerprint and len(store) == len(index)
    _, queries = _random_library()
    parents = {"T1059.001": "T1059", "T1003.001": "T1003"}
    expected = _vote_bits(index.query_vectors(queries, parent_of=parents))
    assert _vote_bits(store.query_vectors(queries, parent_of=parents)) == expected
    hnsw, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    assert _vote_bits(hnsw.query_vectors(queries, parent_of=parents)) == expected
    assert hnsw.last_query_stats == {
        "queries": len(queries),
        "recall_misses": 0,
        "search_fallbacks": 0,
    }
    signature = hnsw.signature()
    assert signature["backend"] == "chroma" and signature["search"] == "hnsw"
    assert signature["hnsw"] == {
        "space": "cosine",
        "max_neighbors": 32,
        "ef_construction": 400,
        "ef_search": 8192,
    }
    assert signature["search_class"] == "exact"  # ef_search >= library size, 2x overfetch
    assert signature["server_version"] == "1.0.0" and signature["records_hash"]
    # the sync script's own verification passes on the same copy
    report = sync_verify(
        client,
        name,
        index,
        index_path=None,
        requested_hnsw=creation_configuration(32, 400, 8192)["hnsw"],
        recall_k=[64, 256],
        overfetch=2.0,
    )
    assert report["ok"], [item for item in report["checks"] if not item["passed"]]
    assert report["self_recall"]["64"]["recall_misses"] == 0


def test_loader_and_queries_never_write():
    client, _, name = _synced()
    for backend in ("chroma", "auto"):
        for search in ("exact", "hnsw"):
            exemplar_store.clear_memo()
            store, _ = load_chroma_exemplar_store(
                _settings(name, backend=backend, search=search), client
            )
            store.query_vectors(np.ones((3, store.embeddings.shape[1]), dtype=np.float32))
    assert client.writes() == []
    assert {method for method in client.methods()} <= {"get_collection", "count", "get", "query"}


def _mutate_metadata(client, name, **changes):
    collection = client.collections[name]
    collection.metadata = {**collection.metadata, **changes}


@pytest.mark.parametrize(
    "mutation,needle",
    [
        (lambda c, n: c.collections.pop(n), "unavailable"),
        (lambda c, n: _mutate_metadata(c, n, kind="other"), "not an exemplar store"),
        (lambda c, n: _mutate_metadata(c, n, store_schema_version=2), "store schema"),
        (lambda c, n: _mutate_metadata(c, n, exemplar_schema_version=9), "exemplar schema"),
        (lambda c, n: _mutate_metadata(c, n, fingerprint="0" * 24), "fingerprint"),
        (lambda c, n: c.collections[n].drop("ex-00007"), "incomplete or changed"),
        (lambda c, n: _mutate_metadata(c, n, count=401), "incomplete or changed"),
        (
            lambda c, n: c.collections[n].tamper("ex-00003", document="another text"),
            "record hash",
        ),
        (
            lambda c, n: c.collections[n].tamper(
                "ex-00003", embedding=np.zeros(24, dtype=np.float32) + 0.2
            ),
            "fingerprint recomputed",
        ),
        (
            lambda c, n: c.collections[n].tamper(
                "ex-00003",
                metadata={"row": 4, "labels": "[]", "label_count": 0, "source": "tram"},
            ),
            "carries row",
        ),
        (
            lambda c, n: c.collections[n].configuration_json["hnsw"].update(space="l2"),
            "not cosine",
        ),
        (
            lambda c, n: c.collections[n].configuration_json["hnsw"].update(ef_search=100),
            "ef_search",
        ),
    ],
)
def test_every_fail_closed_reason(mutation, needle):
    client, _, name = _synced()
    mutation(client, name)
    store, reason = load_chroma_exemplar_store(_settings(name), client)
    assert store is None and needle in reason, reason


def test_model_and_revision_mismatches_and_bad_settings_are_refused():
    client, _, name = _synced()
    assert (
        "embedded with"
        in load_chroma_exemplar_store(_replace(_settings(name), embedding_model="x"), client)[1]
    )
    assert (
        "revision"
        in load_chroma_exemplar_store(
            _replace(_settings(name), embedding_model_revision="r2"), client
        )[1]
    )
    assert (
        "unknown exemplar search"
        in load_chroma_exemplar_store(_settings(name, search="ivf"), client)[1]
    )
    assert (
        "overfetch"
        in load_chroma_exemplar_store(
            _replace(_settings(name), technique_exemplar_overfetch=0.5), client
        )[1]
    )
    assert "no Chroma client" in load_chroma_exemplar_store(_settings(name), None)[1]


def _replace(settings, **changes):
    return SimpleNamespace(**{**settings.__dict__, **changes})


def test_a_server_that_normalises_vectors_fails_closed():
    # rows off unit length: a server-side re-normalisation changes their bytes
    scaled, _ = _library()
    scaled = ExemplarIndex(
        scaled.embeddings * np.float32(0.5),
        scaled.labels,
        scaled.texts,
        scaled.sources,
        dict(scaled.metadata),
    )
    scaled.metadata["fingerprint"] = scaled.compute_fingerprint()
    client, _, name = _synced(index=scaled, normalise_vectors=True)
    store, reason = load_chroma_exemplar_store(_settings(name), client)
    assert store is None and "fingerprint recomputed" in reason


def test_npz_fingerprint_must_match_and_names_derive_from_it(tmp_path):
    client, index, name = _synced()
    path = index.save(tmp_path / "ex.npz")
    derived, how = configured_collection_name(_settings("", index_path=path))
    assert derived == name and how.startswith("derived")
    assert configured_collection_name(_settings(""))[0] is None
    store, reason = load_chroma_exemplar_store(_settings("", index_path=path), client)
    assert store is not None, reason
    other, _ = _library(count=60)
    other_path = other.save(tmp_path / "other.npz")
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=other_path), client)
    assert store is None and "the npz artifact is" in reason


def test_auto_falls_back_to_the_npz_with_a_reason_and_strict_returns_none(tmp_path):
    client, index, name = _synced()
    path = index.save(tmp_path / "ex.npz")
    missing = _settings("technique_exemplars-missing-s1-m32-efc400", index_path=path)
    strict, reason = load_exemplar_index(missing, client=client)
    assert strict is None and reason.startswith("chroma exemplar store unavailable")
    auto, status = load_exemplar_index(
        _replace(missing, technique_exemplar_backend="auto"), client=client
    )
    assert auto is not None and auto.backend == "npz"
    assert status.startswith("active (npz fallback: collection technique_exemplars-missing")
    chroma, status = load_exemplar_index(_settings(name, index_path=path), client=client)
    assert status == "active" and chroma.backend == "chroma"
    # nothing usable at all: no library, and both reasons are kept
    gone = _replace(
        missing, technique_exemplar_backend="auto", technique_exemplar_index_path=tmp_path / "x.npz"
    )
    none, status = load_exemplar_index(gone, client=client)
    assert none is None and "artifact not found" in status and "chroma exemplar" in status


def test_query_errors_fall_back_in_auto_and_raise_when_strict():
    client, index, name = _synced()
    _, queries = _random_library()
    expected = _vote_bits(index.query_vectors(queries))
    auto, _ = load_chroma_exemplar_store(_settings(name, backend="auto", search="hnsw"), client)
    client.collections[name].fail_queries = True
    assert _vote_bits(auto.query_vectors(queries)) == expected
    assert auto.last_query_stats["search_fallbacks"] == len(queries)
    assert query_totals()["search_fallbacks"] == len(queries)
    strict, _ = load_chroma_exemplar_store(_settings(name, backend="chroma", search="hnsw"), client)
    with pytest.raises(ConnectionError):
        strict.query_vectors(queries)
    # exact search never asks Chroma, so it cannot fail there
    exact, _ = load_chroma_exemplar_store(_settings(name, backend="chroma"), client)
    assert _vote_bits(exact.query_vectors(queries)) == expected


def test_hnsw_recall_misses_are_counted_and_the_votes_stay_exact():
    client, index, name = _synced()
    _, queries = _random_library()
    store, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    window = index.exact_windows(queries[:1], k=64)[0]
    client.collections[name].hidden_ids = {f"ex-{int(row):05d}" for row in window[:2]}
    votes = store.query_vectors(queries)
    assert _vote_bits(votes) == _vote_bits(index.query_vectors(queries))
    assert store.last_query_stats["recall_misses"] >= 1
    assert query_totals()["recall_misses"] == store.last_query_stats["recall_misses"]


def test_chroma_queries_are_batched_under_the_result_row_limit(monkeypatch):
    client, _, name = _synced()
    monkeypatch.setattr(vectorstore, "_MAX_QUERY_RESULT_ROWS", 1000)
    store, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    queries = np.random.default_rng(5).normal(size=(30, 24)).astype(np.float32)
    store.query_vectors(queries, k=64)  # window 144, 288 results per query
    rows = client.collections[name].query_rows
    assert len(rows) == 10 and max(rows) <= 1000  # 1000 // 288 = 3 queries per call


def test_the_verified_download_is_memoised_per_client_and_collection():
    client, _, name = _synced()
    first, _ = load_chroma_exemplar_store(_settings(name), client)
    gets = client.methods().count("get")
    second, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    assert client.methods().count("get") == gets  # no second download
    assert second.index is first.index and second is not first
    other = FakeClient()
    _synced(client=other, index=first.index, name=name)
    third, _ = load_chroma_exemplar_store(_settings(name), other)
    assert third.index is not first.index  # another client downloads its own copy


def test_retrieval_reports_the_backend_and_counters():
    from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
    from apt_classifier.retrieve import retrieve_techniques
    from tests.test_exemplars import WordEmbedder, _index
    from tests.test_retrieve import _two_hit_setup

    kb, store, _ = _two_hit_setup()
    store.embedder = WordEmbedder()
    ledger = EvidenceLedger(
        observations=[
            Observation(text="powershell read lsass", provenance=Provenance(source="free_text"))
        ]
    )
    index = _index()
    index.metadata.update({"embedding_model": MODEL, "embedding_revision": REVISION})
    index.metadata["fingerprint"] = index.compute_fingerprint()
    client, _, name = _synced(index=index)
    chroma, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    npz_result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", exemplar_index=index
    )
    chroma_result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", exemplar_index=chroma
    )
    assert npz_result.diagnostics["exemplar_backend"] == "npz"
    assert npz_result.diagnostics["exemplar_search"] == "exact"
    assert chroma_result.diagnostics["exemplar_backend"] == "chroma"
    assert chroma_result.diagnostics["exemplar_search"] == "hnsw"
    assert chroma_result.diagnostics["exemplar_recall_misses"] == 0
    assert chroma_result.diagnostics["exemplar_search_fallbacks"] == 0
    assert [
        (c.technique_id, c.fused_rank, str(c.state), c.fused_score)
        for c in chroma_result.candidates
    ] == [
        (c.technique_id, c.fused_rank, str(c.state), c.fused_score) for c in npz_result.candidates
    ]
    plain = retrieve_techniques(ledger, store, kb, retrieval_mode="dense")
    assert plain.diagnostics["exemplar_backend"] is None
    assert plain.diagnostics["exemplar_recall_misses"] == 0


def test_sync_script_creates_once_verifies_existing_and_refuses_bad_names(tmp_path, monkeypatch):
    index, _ = _library(count=300)
    path = index.save(tmp_path / "ex.npz")
    monkeypatch.setenv("TECHNIQUE_EXEMPLAR_INDEX_PATH", str(path))
    monkeypatch.setenv("EMBEDDING_MODEL", MODEL)
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", REVISION)
    client = FakeClient()
    name = exemplar_collection_name(index.fingerprint, 32, 400)
    report = tmp_path / "sync.json"
    shared = "--allow-shared-output"  # tmp_path lies outside the repository
    # dry run: reads only
    assert sync_main(["--dry-run", "--report", str(report)], client=client) == 0
    assert client.writes() == [] and not report.exists()
    assert sync_main(["--report", str(report), shared], client=client) == 0
    creates = [call for call in client.writes() if call[0] == "create_collection"]
    assert creates == [("create_collection", name, {"get_or_create": False})]
    assert all(call[0] in {"create_collection", "add"} for call in client.writes())
    assert json.loads(report.read_text())["ok"] is True
    # a second run finds the collection and only verifies it
    client.calls.clear()
    assert sync_main(["--report", str(report), "--overwrite", shared], client=client) == 0
    assert client.writes() == []
    assert (
        sync_main(["--verify-only", "--report", str(report), "--overwrite", shared], client=client)
        == 0
    )
    # a missing collection with --verify-only is an error, not a create
    other = FakeClient()
    verify_only = ["--verify-only", "--report", str(tmp_path / "v.json"), shared]
    assert sync_main(verify_only, client=other) == 1
    assert other.writes() == []
    with pytest.raises(SystemExit):
        sync_main(["--collection", "attack_exemplars", "--report", str(report)], client=other)
    # an existing report is never replaced silently, nor written outside the repository
    with pytest.raises(SystemExit):
        sync_main(["--verify-only", "--report", str(report), shared], client=client)
    with pytest.raises(SystemExit):
        sync_main(["--verify-only", "--report", str(tmp_path / "new.json")], client=client)


def test_sync_verification_flags_a_server_that_ignored_the_build_settings():
    client, index, name = _synced(ignore_build_settings=True)
    report = sync_verify(
        client,
        name,
        index,
        index_path=None,
        requested_hnsw=creation_configuration(32, 400, 8192)["hnsw"],
        recall_k=[64],
        overfetch=2.0,
    )
    assert not report["ok"] and "-r2" in report["advice"]
    failed = [item["check"] for item in report["checks"] if not item["passed"]]
    assert failed == ["hnsw configuration echo"]
    assert (
        records_hash(*exemplar_records(index)) == client.collections[name].metadata["records_hash"]
    )
