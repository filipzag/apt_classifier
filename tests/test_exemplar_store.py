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


def _library_metadata(library):
    """A Chroma library's metadata without what the loader adds to the source manifest."""
    return {
        key: value
        for key, value in library.metadata.items()
        if key not in ("fingerprint", "source_fingerprint", "library_backend")
    }


def test_sync_round_trip_is_bitwise_identical_and_votes_match_the_npz():
    # a server that keeps the bytes: the Chroma library is the npz, fingerprint included (the
    # pin verifies it without the npz file)
    client, index, name = _synced()
    store, reason = load_chroma_exemplar_store(
        _settings(name, technique_exemplar_library_fingerprint=index.fingerprint), client
    )
    assert reason == "ok" and isinstance(store, ChromaExemplarStore)
    assert store.verification["verified_against"] == ["pinned library fingerprint"]
    assert isinstance(store, ExemplarSource)
    copy = store.index
    assert np.array_equal(copy.embeddings.view(np.uint32), index.embeddings.view(np.uint32))
    assert copy.labels == index.labels and copy.texts == index.texts
    assert copy.sources == index.sources
    assert _library_metadata(copy) == {
        key: value for key, value in index.metadata.items() if key != "fingerprint"
    }
    assert copy.metadata["source_fingerprint"] == index.fingerprint
    assert store.fingerprint == index.fingerprint and len(store) == len(index)
    assert store.source_fingerprint == index.fingerprint
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
    library = report["library"]
    assert library["bit_identical_to_npz"] and library["identical_rows"] == len(index)
    assert library["max_abs_diff"] == 0.0 and library["max_cosine_change"] == 0.0
    assert library["neighbour_lists"]["64"]["share_identical_order"] == 1.0
    assert report["library_fingerprint"] == index.fingerprint


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
            "vector norm",
        ),
        (
            lambda c, n: c.collections[n].tamper(
                "ex-00003", embedding=np.full(24, np.nan, dtype=np.float32)
            ),
            "not all finite",
        ),
        (
            lambda c, n: c.collections[n].tamper(
                "ex-00005",
                metadata={"row": 5, "labels": '["T9999"]', "label_count": 1, "source": "tram"},
            ),
            "record hash",
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


def _scaled(index, factor):
    scaled = ExemplarIndex(
        index.embeddings * np.float32(factor),
        index.labels,
        index.texts,
        index.sources,
        dict(index.metadata),
    )
    scaled.metadata["fingerprint"] = scaled.compute_fingerprint()
    return scaled


def test_a_renormalising_server_is_accepted_within_tolerance_as_its_own_library(tmp_path):
    """Chroma re-normalises in float32 on insert: some rows come back a few ulps away. That
    library is accepted (Chroma is the source) under its own fingerprint."""
    client, index, name = _synced(normalise_vectors=True)
    path = index.save(tmp_path / "ex.npz")
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert reason == "ok", reason
    verification = store.verification
    assert verification["compared_with_source"] and verification["identical_rows"] < len(index)
    assert 0.0 < verification["max_abs_diff"] <= 1e-6
    assert verification["max_norm_deviation"] <= 1e-6
    # its identity: the fingerprint of the vectors Chroma returned, not the npz's
    assert store.fingerprint == store.index.compute_fingerprint() != index.fingerprint
    assert store.source_fingerprint == index.fingerprint
    assert client.collections[name].metadata["fingerprint"] == index.fingerprint  # untouched
    signature = store.signature()
    assert signature["fingerprint"] == store.fingerprint
    assert signature["source_fingerprint"] == index.fingerprint
    assert signature["verification"]["identical_rows"] == verification["identical_rows"]
    assert store.verified and store.status == "active"
    assert store.verification["verified_against"] == ["source npz"]
    # without the npz only the norms are checked: the same library, served as unverified
    exemplar_store.clear_memo()
    alone, reason = load_chroma_exemplar_store(_settings(name), client)
    assert reason.startswith("ok (unverified: ") and alone.fingerprint == store.fingerprint
    assert not alone.verification["compared_with_source"] and not alone.verified
    assert client.writes() == []


def test_the_tolerance_is_a_setting_and_fails_closed_beyond_it(tmp_path):
    client, index, name = _synced()
    path = index.save(tmp_path / "ex.npz")
    collection = client.collections[name]
    vector = index.embeddings[9].astype(np.float64)
    vector[0] += 5e-7
    collection.tamper("ex-00009", embedding=(vector / np.linalg.norm(vector)).astype(np.float32))
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert reason == "ok" and store.verification["max_abs_diff"] < 1e-6
    exemplar_store.clear_memo()
    strict, reason = load_chroma_exemplar_store(
        _settings(name, index_path=path, technique_exemplar_tolerance=1e-8), client
    )
    assert strict is None and "differ from the source npz" in reason
    exemplar_store.clear_memo()
    vector[0] += 5e-5
    collection.tamper("ex-00009", embedding=(vector / np.linalg.norm(vector)).astype(np.float32))
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert store is None and "differ from the source npz" in reason
    # the same rotated row without the npz: norms are fine, so it is a library of its own, but
    # one nothing verified
    exemplar_store.clear_memo()
    alone, reason = load_chroma_exemplar_store(_settings(name), client)
    assert reason.startswith("ok (unverified: ") and alone.fingerprint != index.fingerprint
    for bad in (-1.0, float("nan"), float("inf")):
        exemplar_store.clear_memo()
        refused, reason = load_chroma_exemplar_store(
            _settings(name, technique_exemplar_tolerance=bad), client
        )
        assert refused is None and "tolerance" in reason


def test_off_unit_vectors_fail_closed_with_or_without_renormalisation(tmp_path):
    scaled = _scaled(_library()[0], 0.5)
    # a server that keeps the bytes: norms of 0.5 are refused
    client, _, name = _synced(index=scaled)
    store, reason = load_chroma_exemplar_store(_settings(name), client)
    assert store is None and "vector norm" in reason
    # a server that re-normalises: every component moves, so the npz comparison refuses it
    exemplar_store.clear_memo()
    client, _, name = _synced(index=scaled, normalise_vectors=True)
    path = scaled.save(tmp_path / "scaled.npz")
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert store is None and "differ from the source npz" in reason


def test_a_present_npz_of_another_library_is_refused(tmp_path):
    client, index, name = _synced(normalise_vectors=True)
    other, _ = _library(count=60)
    path = other.save(tmp_path / "other.npz")
    store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert store is None and "copied from library" in reason


def _replace_every_vector(client, name, count, *, seed=11):
    """Every stored vector replaced by a random unit vector; ids, documents and metadata (so
    the record hash) unchanged."""
    collection = client.collections[name]
    rng = np.random.default_rng(seed)
    for row in range(count):
        vector = rng.normal(size=collection._records[f"ex-{row:05d}"][0].shape)  # noqa: SLF001
        collection.tamper(f"ex-{row:05d}", embedding=(vector / np.linalg.norm(vector)))


def test_without_the_npz_changed_vectors_are_unverified_and_a_pin_refuses_them(tmp_path):
    """Review fix: the record hash does not cover the vectors, so without the npz a collection
    whose every vector was replaced passed as plain "active". It is now served as
    "active (unverified: ...)", --require-exemplar-backend chroma refuses it, and the pinned
    TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT refuses it outright, npz or not."""
    from scripts.run_tooling import exemplar_backend_problem

    client, index, name = _synced(normalise_vectors=True)
    path = index.save(tmp_path / "ex.npz")
    original, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert reason == "ok" and original.verified
    registered = original.fingerprint
    _replace_every_vector(client, name, len(index))
    exemplar_store.clear_memo()
    with_npz, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert with_npz is None and "differ from the source npz" in reason
    for missing in (None, tmp_path / "moved.npz"):
        exemplar_store.clear_memo()
        store, reason = load_chroma_exemplar_store(_settings(name, index_path=missing), client)
        assert store is not None and reason.startswith("ok (unverified: no source npz")
        assert not store.verified and store.fingerprint != registered
        assert store.status.startswith("active (unverified: no source npz")
        loaded, status = load_exemplar_index(_settings(name, index_path=missing), client=client)
        assert loaded is not None and status == store.status != "active"
        problem = exemplar_backend_problem(loaded, "chroma")
        assert problem and "verified against neither" in problem
        assert exemplar_backend_problem(loaded, None) is None  # only a required backend refuses
        # pinned to the registered library: refused bit for bit, on chroma and on auto
        for backend in ("chroma", "auto"):
            exemplar_store.clear_memo()
            pinned = _settings(
                name,
                backend=backend,
                index_path=missing,
                technique_exemplar_library_fingerprint=registered,
            )
            refused, reason = load_chroma_exemplar_store(pinned, client)
            assert refused is None and "not the pinned TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT" in (
                reason
            )
            none, status = load_exemplar_index(pinned, client=client)
            assert none is None and "pinned" in status
    # a malformed pin is refused, never ignored
    exemplar_store.clear_memo()
    bad, reason = load_chroma_exemplar_store(
        _settings(name, technique_exemplar_library_fingerprint="not-hex"), client
    )
    assert bad is None and "not a library fingerprint" in reason
    assert client.writes() == []  # the loader only reads


def test_a_pinned_fingerprint_verifies_the_chroma_library_without_the_npz(tmp_path):
    client, index, name = _synced(normalise_vectors=True)
    path = index.save(tmp_path / "ex.npz")
    registered = load_chroma_exemplar_store(_settings(name, index_path=path), client)[0].fingerprint
    exemplar_store.clear_memo()
    store, reason = load_chroma_exemplar_store(
        _settings(name, technique_exemplar_library_fingerprint=registered), client
    )
    assert reason == "ok" and store.verified and store.status == "active"
    assert store.verification["verified_against"] == ["pinned library fingerprint"]
    assert store.verification["pinned_fingerprint"] == registered
    both, _ = load_chroma_exemplar_store(
        _settings(name, index_path=path, technique_exemplar_library_fingerprint=registered),
        client,
    )
    assert both.verification["verified_against"] == ["source npz", "pinned library fingerprint"]
    # the pin is outside the signature: a pinned and an unpinned run of one library compare equal
    assert (
        both.signature()
        == load_chroma_exemplar_store(_settings(name, index_path=path), client)[0].signature()
    )
    # one changed vector, well within the tolerance and with the npz present: the pin refuses it
    vector = index.embeddings[7].astype(np.float64)
    vector[0] += 4e-7
    client.collections[name].tamper(
        "ex-00007", embedding=(vector / np.linalg.norm(vector)).astype(np.float32)
    )
    exemplar_store.clear_memo()
    drifted, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert reason == "ok" and drifted.fingerprint != registered  # inside the tolerance
    exemplar_store.clear_memo()
    refused, reason = load_chroma_exemplar_store(
        _settings(name, index_path=path, technique_exemplar_library_fingerprint=registered),
        client,
    )
    assert refused is None and "not the pinned" in reason


def test_the_pin_applies_to_the_npz_backend_and_its_default_changes_nothing(tmp_path):
    index, _ = _library(count=60)
    path = index.save(tmp_path / "ex.npz")
    plain = _settings("", backend="npz", index_path=path)
    loaded, status = load_exemplar_index(plain, client=None)
    assert status == "active" and loaded.fingerprint == index.fingerprint
    pinned = _replace(plain, technique_exemplar_library_fingerprint=index.fingerprint)
    assert load_exemplar_index(pinned, client=None)[1] == "active"
    other = _replace(plain, technique_exemplar_library_fingerprint="ab" * 12)
    none, status = load_exemplar_index(other, client=None)
    assert none is None and "not the pinned" in status
    malformed = _replace(plain, technique_exemplar_library_fingerprint="AB" * 12)
    assert load_exemplar_index(malformed, client=None)[0] is None


def _unreadable_npz_files(tmp_path, index):
    valid = index.save(tmp_path / "valid.npz").read_bytes()
    files = {
        "garbage": b"not an npz artifact at all " * 8,
        "truncated": valid[:120],
        "empty": b"",
    }
    paths = {}
    for label, payload in files.items():
        paths[label] = tmp_path / f"{label}.npz"
        paths[label].write_bytes(payload)
    paths["no_metadata"] = tmp_path / "no_metadata.npz"
    np.savez(paths["no_metadata"], embeddings=np.zeros((2, 3), dtype=np.float32))
    paths["no_fingerprint"] = tmp_path / "no_fingerprint.npz"
    np.savez(paths["no_fingerprint"], metadata=np.array(json.dumps({"kind": "x"})))
    return paths


def test_a_present_but_unreadable_npz_is_refused_never_skipped(tmp_path):
    """Review fix: an npz whose manifest could not be read counted as no npz, so the Chroma
    library's vectors went unchecked. A present npz must be readable."""
    client, index, name = _synced(normalise_vectors=True)
    for label, path in _unreadable_npz_files(tmp_path, index).items():
        exemplar_store.clear_memo()
        store, reason = load_chroma_exemplar_store(_settings(name, index_path=path), client)
        assert store is None, label
        assert "present but" in reason and str(path) in reason, (label, reason)
        derived, how = configured_collection_name(_settings("", index_path=path))
        assert derived is None and "present but" in how, (label, how)
    assert client.writes() == []  # the loader only reads


def test_the_npz_fitted_model_is_refused_on_the_chroma_library(tmp_path):
    """A model records the fingerprint of the library it was fitted on: the shipped (npz) model
    does not run on the re-normalised Chroma library, a model fitted there does, and its
    contract records the backend without comparing it."""
    from apt_classifier.acceptance import AcceptanceModel, contract_mismatches
    from apt_classifier.config import Settings
    from apt_classifier.exemplars import load_exemplar_index
    from apt_classifier.retrieve import retrieval_contract_for
    from tests.test_acceptance import _synthetic_rows

    client, index, name = _synced(normalise_vectors=True)
    npz_path = index.save(tmp_path / "ex.npz")
    settings = Settings(
        _env_file=None,
        technique_exemplar_index_path=npz_path,
        technique_exemplar_backend="chroma",
        technique_exemplar_collection=name,
        embedding_model=MODEL,
        embedding_model_revision=REVISION,
    )
    chroma, status = load_exemplar_index(settings, client=client)
    assert status == "active" and chroma.backend == "chroma"
    assert chroma.fingerprint != index.fingerprint
    shipped = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2")
    shipped.metadata = {"exemplar_fingerprint": index.fingerprint}
    assert shipped.compatible_with("19.2", index.fingerprint)[0]
    ok, reason = shipped.compatible_with("19.2", chroma.fingerprint)
    assert not ok and index.fingerprint in reason and chroma.fingerprint in reason
    contract = retrieval_contract_for(settings, chroma)
    assert contract["exemplar_fingerprint"] == chroma.fingerprint
    assert contract["exemplar_backend"] == "chroma"
    refit = AcceptanceModel.fit(*_synthetic_rows(), attack_version="19.2", contract=contract)
    refit.metadata = {"exemplar_fingerprint": chroma.fingerprint, "exemplar_backend": "chroma"}
    assert refit.compatible_with("19.2", chroma.fingerprint, contract) == (True, "ok")
    assert not refit.compatible_with("19.2", index.fingerprint, contract)[0]
    # the backend is recorded, never compared: the fingerprint is the library's identity
    assert (
        contract_mismatches(contract, {**contract, "exemplar_backend": "npz"}, exemplars=True) == []
    )


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
    assert client.collections[name].metadata["fingerprint"] == index.fingerprint


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


def test_the_verified_download_is_memoised_per_client_and_collection(tmp_path):
    client, index, name = _synced(normalise_vectors=True)
    first, _ = load_chroma_exemplar_store(_settings(name), client)
    gets = client.methods().count("get")
    second, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    assert client.methods().count("get") == gets  # no second download
    assert second.index is first.index and second is not first
    assert second.fingerprint == first.fingerprint  # the memo holds the library fingerprint
    # another tolerance or a source npz to compare with is verified afresh
    path = index.save(tmp_path / "ex.npz")
    third, _ = load_chroma_exemplar_store(_settings(name, index_path=path), client)
    assert client.methods().count("get") > gets and third.index is not first.index
    assert third.fingerprint == first.fingerprint
    other = FakeClient()
    _synced(client=other, index=index, name=name)
    fourth, _ = load_chroma_exemplar_store(_settings(name), other)
    assert fourth.index is not first.index  # another client downloads its own copy


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


def test_sync_verifies_a_renormalising_server_within_tolerance(tmp_path, monkeypatch):
    """The real server's behaviour: the sync passes within tolerance, reports how far the
    library is from the npz and its own fingerprint, and --verify-only checks the existing
    collection again without writing."""
    index, _ = _library(count=300)
    path = index.save(tmp_path / "ex.npz")
    monkeypatch.setenv("TECHNIQUE_EXEMPLAR_INDEX_PATH", str(path))
    monkeypatch.setenv("EMBEDDING_MODEL", MODEL)
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", REVISION)
    client = FakeClient(normalise_vectors=True)
    name = exemplar_collection_name(index.fingerprint, 32, 400)
    report_path = tmp_path / "sync.json"
    shared = "--allow-shared-output"
    assert sync_main(["--report", str(report_path), "--recall-k", "64", shared], client=client) == 0
    report = json.loads(report_path.read_text())
    assert report["ok"] and report["action"] == "created and verified"
    library = report["library"]
    assert library["backend"] == "chroma" and library["collection"] == name
    assert library["source_fingerprint"] == index.fingerprint == report["source_fingerprint"]
    assert report["library_fingerprint"] == library["fingerprint"] != index.fingerprint
    assert not library["bit_identical_to_npz"] and library["split"] == "train"
    assert library["identical_rows"] < library["rows"] == len(index)
    assert 0.0 < library["max_abs_diff"] <= 1e-6 and library["max_norm_deviation"] <= 1e-6
    assert library["norm_min"] <= 1.0 <= library["norm_max"]
    assert library["max_cosine_change"] < 1e-5
    lists = library["neighbour_lists"]["64"]
    assert lists["queries"] == len(index) and lists["window"] == 144
    assert 0.99 <= lists["share_identical_order"] <= lists["share_identical_set"] <= 1.0
    passed = {item["check"]: item["passed"] for item in report["checks"]}
    assert passed["vectors within 1e-06 of the npz"] and passed["vector norms within 1e-06 of 1"]
    # verify-only on the existing collection: nothing recreated, nothing written
    client.calls.clear()
    verify_path = tmp_path / "verify.json"
    assert sync_main(["--verify-only", "--report", str(verify_path), shared], client=client) == 0
    assert client.writes() == []
    again = json.loads(verify_path.read_text())
    assert again["action"] == "verified" and again["library_fingerprint"] == library["fingerprint"]
    # beyond the tolerance it fails (exit 1) and says how far the vectors are
    vector = index.embeddings[4].astype(np.float64)
    vector[0] += 1e-4
    client.collections[name].tamper(
        "ex-00004", embedding=(vector / np.linalg.norm(vector)).astype(np.float32)
    )
    failed_path = tmp_path / "failed.json"
    assert sync_main(["--verify-only", "--report", str(failed_path), shared], client=client) == 1
    failed = json.loads(failed_path.read_text())
    assert not failed["ok"] and "differ from the source npz" in failed["checks"][0]["detail"]
    assert failed["diagnosis"]["deviation"]["max_abs_diff"] > 1e-6
    # a tighter --tolerance refuses even the untouched re-normalised rows
    exemplar_store.clear_memo()
    other = FakeClient(normalise_vectors=True)
    tight = ["--report", str(tmp_path / "tight.json"), "--tolerance", "1e-12", shared]
    assert sync_main(tight, client=other) == 1
    assert {call[0] for call in other.writes()} == {"create_collection", "add"}


def test_sync_verify_compares_the_library_fingerprint_with_a_registered_value(
    tmp_path, monkeypatch
):
    """Review fix: the check "library fingerprint recomputed from the vectors Chroma returned"
    compared the fingerprint with itself. A re-verify after the server changed a stored vector
    within the tolerance passed with a new fingerprint and no warning. The fingerprint is now
    compared with a second download and with --expect-library-fingerprint (default
    TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT); without one the report warns."""
    import scripts.sync_exemplars_to_chroma as sync_module

    index, _ = _library(count=300)
    path = index.save(tmp_path / "ex.npz")
    monkeypatch.setenv("TECHNIQUE_EXEMPLAR_INDEX_PATH", str(path))
    monkeypatch.setenv("EMBEDDING_MODEL", MODEL)
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", REVISION)
    monkeypatch.delenv("TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT", raising=False)
    client = FakeClient(normalise_vectors=True)
    name = exemplar_collection_name(index.fingerprint, 32, 400)
    shared = "--allow-shared-output"

    def run(label, *extra):
        report_path = tmp_path / f"{label}.json"
        arguments = ["--verify-only", "--report", str(report_path), shared, *extra]
        code = sync_main(arguments, client=client)
        return code, json.loads(report_path.read_text())

    first_path = tmp_path / "first.json"
    assert sync_main(["--report", str(first_path), shared], client=client) == 0
    first = json.loads(first_path.read_text())
    registered = first["library_fingerprint"]
    labels = [item["check"] for item in first["checks"]]
    assert not any("recomputed" in label for label in labels)  # the tautology is gone
    second = {item["check"]: item for item in first["checks"]}[
        "a second download of the collection gives the same library fingerprint"
    ]
    assert second["passed"] and second["detail"] == {"first": registered, "second": registered}
    assert first["library_fingerprint_expected"] is None
    assert (
        len(first["warnings"]) == 1
        and "not compared with an expected value" in (first["warnings"][0])
    )
    # the registered value passes and silences the warning
    code, pinned = run("pinned", "--expect-library-fingerprint", registered)
    assert code == 0 and pinned["ok"] and pinned["warnings"] == []
    assert {item["check"]: item["passed"] for item in pinned["checks"]}[
        "library fingerprint is the expected one (--expect-library-fingerprint)"
    ]
    # the server changes one stored vector by 4e-7 (inside the 1e-6 tolerance)
    vector = index.embeddings[7].astype(np.float64)
    vector[0] += 4e-7
    client.collections[name].tamper(
        "ex-00007", embedding=(vector / np.linalg.norm(vector)).astype(np.float32)
    )
    code, unpinned = run("unpinned")
    assert code == 0 and unpinned["library_fingerprint"] != registered  # within tolerance
    assert unpinned["warnings"]  # but it says the fingerprint was compared with nothing
    code, drifted = run("drifted", "--expect-library-fingerprint", registered)
    assert code == 1 and not drifted["ok"]
    failed = [item for item in drifted["checks"] if not item["passed"]]
    assert [item["check"] for item in failed] == [
        "library fingerprint is the expected one (--expect-library-fingerprint)"
    ]
    assert failed[0]["detail"] == {
        "library": drifted["library_fingerprint"],
        "expected": registered,
    }
    # the setting is the default expectation (T1.env pins it)
    monkeypatch.setenv("TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT", registered)
    code, from_settings = run("from_settings")
    assert code == 1 and from_settings["library_fingerprint_expected"] == registered
    monkeypatch.delenv("TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT")
    with pytest.raises(SystemExit):
        run("malformed", "--expect-library-fingerprint", "XYZ")
    # a second download that disagrees with the first fails the verification
    real_read = sync_module.read_exemplar_collection

    def other_vectors(collection, metadata):
        library = real_read(collection, metadata)
        library.embeddings[0, 0] = np.float32(library.embeddings[0, 0] + 1e-3)
        library.metadata["fingerprint"] = library.compute_fingerprint()
        return library

    monkeypatch.setattr(sync_module, "read_exemplar_collection", other_vectors)
    code, unstable = run(
        "unstable", "--expect-library-fingerprint", unpinned["library_fingerprint"]
    )
    assert code == 1
    assert [item["check"] for item in unstable["checks"] if not item["passed"]] == [
        "a second download of the collection gives the same library fingerprint"
    ]
    assert client.writes() == [
        call for call in client.writes() if call[0] in {"create_collection", "add"}
    ]


def test_a_fit_on_the_chroma_library_records_it(tmp_path):
    """fit_acceptance_model.py --exemplar-backend chroma: the artifact names the Chroma library
    (its own fingerprint), the backend and the npz it was copied from."""
    from apt_classifier.config import Settings
    from scripts.fit_acceptance_model import exemplar_fit_metadata

    client, index, name = _synced(normalise_vectors=True)
    path = index.save(tmp_path / "ex.npz")
    settings = Settings(
        _env_file=None,
        technique_exemplar_index_path=path,
        technique_exemplar_backend="chroma",
        technique_exemplar_collection=name,
        embedding_model=MODEL,
        embedding_model_revision=REVISION,
    )
    chroma, status = load_exemplar_index(settings, client=client)
    assert status == "active"
    recorded = exemplar_fit_metadata(settings, chroma)
    assert recorded["exemplar_fingerprint"] == chroma.fingerprint != index.fingerprint
    assert recorded["exemplar_backend"] == "chroma"
    exemplars = recorded["exemplars"]
    assert exemplars["backend"] == "chroma" and exemplars["collection"] == name
    assert exemplars["source_fingerprint"] == index.fingerprint
    assert exemplars["store"]["fingerprint"] == chroma.fingerprint
    npz = exemplar_fit_metadata(settings, index)
    assert npz["exemplar_backend"] == "npz" and npz["exemplar_fingerprint"] == index.fingerprint
    assert npz["exemplars"]["source_fingerprint"] == index.fingerprint
    assert exemplar_fit_metadata(settings, None) == {
        "exemplar_fingerprint": None,
        "exemplar_backend": None,
        "exemplars": None,
    }
