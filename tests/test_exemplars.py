"""Exemplar index: labelled real-report sentences as technique-mapping evidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier.acceptance import (
    FEATURE_NAMES,
    AcceptanceModel,
    candidate_features,
    load_acceptance_model,
)
from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.exemplars import (
    EVIDENCE_SOURCE,
    ExemplarIndex,
    empty_features,
    exemplar_kwargs,
    exemplar_query_text,
    load_exemplar_index,
    parent_map,
)
from apt_classifier.exemplars import (
    FEATURE_NAMES as EXEMPLAR_FEATURES,
)
from apt_classifier.retrieve import retrieve_techniques
from tests.test_retrieve import _two_hit_setup

WORDS = ("powershell", "lsass", "schedule", "phishing", "mimikatz", "download")


class WordEmbedder:
    """Deterministic bag-of-words embedder over a tiny vocabulary (plus a bias dimension)."""

    def embed(self, texts, batch_size=None):
        rows = []
        for text in texts:
            lowered = str(text).casefold()
            rows.append([1.0 if word in lowered else 0.0 for word in WORDS] + [0.3])
        return rows


def _index(rows=None):
    rows = rows or [
        ("the actor ran powershell to download a payload", ["T1059.001", "T1105"], "tram"),
        ("powershell was used for execution", ["T1059.001"], "expert"),
        ("credentials were dumped from lsass memory with mimikatz", ["T1003.001"], "tram"),
        ("a scheduled task gave persistence", ["T1053.005"], "tram"),
        ("spear phishing emails delivered the loader", ["T1566.001"], "derived_procedures"),
        ("Powershell was used for execution", ["T1059"], "tram"),  # identical text: merged
    ]
    return ExemplarIndex.build(
        rows,
        WordEmbedder(),
        embedding_model="test-model",
        embedding_revision="r1",
        attack_version="t",
    )


def _parents():
    return {"T1059.001": "T1059", "T1003.001": "T1003", "T1053.005": "T1053"}


def test_build_merges_identical_texts_and_fingerprints():
    index = _index()
    assert len(index) == 5  # the two "powershell was used for execution" rows merged
    merged = index.labels[index.texts.index("powershell was used for execution")]
    assert merged == ("T1059", "T1059.001")
    assert index.fingerprint == index.compute_fingerprint() and len(index.fingerprint) == 24
    with pytest.raises(ValueError):
        ExemplarIndex.build(
            [("no labels", [], "tram")],
            WordEmbedder(),
            embedding_model="m",
            embedding_revision=None,
            attack_version="t",
        )


def test_votes_weight_neighbours_credit_parents_and_skip_duplicates():
    index = _index()
    # not identical to any exemplar in the toy vocabulary (an identical one counts as a copy)
    query = WordEmbedder().embed(["powershell downloaded a payload after a phishing email"])
    votes = index.query_vectors(
        query, parent_of=_parents(), k=4, tau=0.05, duplicate_similarity=0.999
    )[0]
    assert votes.ranked[0] == "T1059.001"
    assert votes.votes["T1059.001"] > votes.votes.get("T1003.001", 0.0)
    # a sub-technique's vote also counts for its parent (hierarchy-relaxed share)
    assert votes.family["T1059"] >= votes.votes["T1059.001"]
    assert votes.best_text["T1105"].startswith("the actor ran powershell")
    assert 0 < votes.top_similarity <= 1.0 and votes.neighbours == 4
    features = votes.features("T1059.001")
    assert set(features) == set(EXEMPLAR_FEATURES)
    assert features["ex_rank_inv"] == 1.0 and features["ex_maxsim_rel"] == pytest.approx(
        0.0, abs=1e-6
    )
    absent = votes.features("T9999")
    assert (
        absent["ex_vote"] == 0.0
        and absent["ex_maxsim_rel"] == -0.5
        and absent["ex_rank_inv"] == 0.0
    )
    assert votes.features("T1105", exemplar_only=True)["exemplar_only"] == 1.0
    # an exemplar identical to the query is treated as a copy and never votes
    copy = WordEmbedder().embed(["credentials were dumped from lsass memory with mimikatz"])
    strict = index.query_vectors(copy, parent_of=_parents(), k=4, duplicate_similarity=0.99)[0]
    assert strict.excluded_duplicates >= 1
    assert strict.max_similarity.get("T1003.001", 0.0) < 0.99
    # nothing but duplicates -> empty votes, not an error
    only = ExemplarIndex.build(
        [("lsass", ["T1003.001"], "tram")],
        WordEmbedder(),
        embedding_model="m",
        embedding_revision=None,
        attack_version="t",
    )
    empty = only.query_vectors(WordEmbedder().embed(["lsass"]), duplicate_similarity=0.5)[0]
    assert empty.votes == {} and empty.excluded_duplicates == 1 and empty.top_similarity == 0.0


def test_save_load_round_trip_and_fail_closed_loader(tmp_path):
    index = _index()
    path = index.save(tmp_path / "ex.npz")
    loaded = ExemplarIndex.load(path)
    assert loaded is not None and loaded.fingerprint == index.fingerprint
    assert loaded.texts == index.texts and np.allclose(loaded.embeddings, index.embeddings)
    assert ExemplarIndex.load(tmp_path / "missing.npz") is None
    settings = SimpleNamespace(
        technique_exemplars_enabled=True,
        technique_exemplar_index_path=path,
        embedding_model="test-model",
        embedding_model_revision="r1",
    )
    assert load_exemplar_index(settings)[1] == "active"
    assert load_exemplar_index(SimpleNamespace(technique_exemplars_enabled=False))[1] == "disabled"
    assert (
        "not found"
        in load_exemplar_index(
            SimpleNamespace(
                **{**settings.__dict__, "technique_exemplar_index_path": tmp_path / "x.npz"}
            )
        )[1]
    )
    assert (
        "incompatible"
        in load_exemplar_index(
            SimpleNamespace(**{**settings.__dict__, "embedding_model": "other"})
        )[1]
    )
    assert (
        "incompatible"
        in load_exemplar_index(
            SimpleNamespace(**{**settings.__dict__, "embedding_model_revision": "r2"})
        )[1]
    )
    # tampered contents fail the fingerprint check
    tampered = ExemplarIndex(
        index.embeddings * 0.5, index.labels, index.texts, index.sources, dict(index.metadata)
    )
    bad = tampered.save(tmp_path / "bad.npz")
    assert (
        "invalid"
        in load_exemplar_index(
            SimpleNamespace(**{**settings.__dict__, "technique_exemplar_index_path": bad})
        )[1]
    )


def test_retrieval_attaches_exemplar_evidence_and_appends_exemplar_only_candidates():
    kb, store, ledger = _two_hit_setup()
    kb.techniques["T1105"] = Technique(id="T1105", name="Ingress Tool Transfer")
    kb.techniques["T1059"] = Technique(id="T1059", name="Command and Scripting Interpreter")
    store.embedder = WordEmbedder()
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="powershell downloaded a payload and read lsass",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    index = _index()
    settings = SimpleNamespace(
        technique_exemplar_k=4,
        technique_exemplar_tau=0.05,
        technique_exemplar_duplicate_similarity=0.999,
        technique_exemplar_new_candidates=2,
    )
    result = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", **exemplar_kwargs(settings, index)
    )
    by_id = {c.technique_id: c for c in result.candidates}
    assert result.diagnostics["exemplars"] == "active"
    # every candidate carries exemplar evidence (possibly a zero vote) so features are complete
    assert all(c.evidence_for(EVIDENCE_SOURCE) is not None for c in result.candidates)
    assert by_id["T1059.001"].evidence_for(EVIDENCE_SOURCE).score > 0
    # techniques only the exemplars suggest are appended after the retrieved pool and flagged;
    # a cue-only candidate (no dense or lexical hit either) is not an exemplar-only one
    appended = [
        c for c in result.candidates if c.evidence_for(EVIDENCE_SOURCE).metadata["exemplar_only"]
    ]
    assert appended
    assert all(
        c.evidence_for("dense") is None and c.evidence_for("lexical") is None for c in appended
    )
    assert result.diagnostics["exemplar_candidates_added"] == len(appended) <= 2
    retrieved_ranks = [c.fused_rank for c in result.candidates if c not in appended]
    assert min(c.fused_rank for c in appended) > max(retrieved_ranks)
    features = candidate_features(
        appended[0], list(result.candidates), benign_context=False, sparse_context=False
    )
    assert features["exemplar_only"] == 1.0 and features["dense_present"] == 0.0
    assert set(EXEMPLAR_FEATURES) <= set(FEATURE_NAMES)
    # without an index nothing changes and features are the empty defaults
    plain = retrieve_techniques(ledger, store, kb, retrieval_mode="dense")
    assert plain.diagnostics["exemplars"] == "off"
    assert all(c.evidence_for(EVIDENCE_SOURCE) is None for c in plain.candidates)
    feats = candidate_features(
        plain.candidates[0], list(plain.candidates), benign_context=False, sparse_context=False
    )
    assert {k: feats[k] for k in EXEMPLAR_FEATURES} == empty_features()


def test_acceptance_model_fitted_with_exemplars_refuses_to_run_without_them():
    kb, store, ledger = _two_hit_setup()
    store.embedder = WordEmbedder()
    index = _index()
    rows = [{name: float(i % 3) for name in FEATURE_NAMES} for i in range(12)]
    model = AcceptanceModel.fit(rows, [i % 2 for i in range(12)], attack_version="")
    model.metadata = {"exemplar_fingerprint": index.fingerprint}
    assert model.requires_exemplars
    assert model.compatible_with("", index.fingerprint)[0]
    ok, reason = model.compatible_with("", None)
    assert not ok and "exemplar index" in reason
    # retrieval disables it (fail closed) when the index is missing, with a visible reason
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", acceptance_model=model)
    assert "acceptance_model_disabled" in result.diagnostics
    assert result.diagnostics["acceptance_model"] is False
    fine = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", acceptance_model=model, exemplar_index=index
    )
    assert "acceptance_model_disabled" not in fine.diagnostics


def test_parent_map_reads_the_kb():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="x"),
            "T1059.001": Technique(
                id="T1059.001", name="y", is_subtechnique=True, parent_id="T1059"
            ),
        },
        groups={},
    )
    assert parent_map(kb) == {"T1059": None, "T1059.001": "T1059"}


def test_exemplar_query_drops_record_schema_but_keeps_prose():
    record = "command_line: whoami /all. pid: 5512. process_name: C:\\Windows\\System32\\whoami.exe"
    assert exemplar_query_text(record) == "whoami /all. C:\\Windows\\System32\\whoami.exe"
    assert exemplar_query_text(
        "dst_ip: 10.0.0.5. guid: {0123abcd-0000-1111-2222-333344445555}"
    ) == ("10.0.0.5")
    # prose, a single field, a record behind a sentence, and all-identifier records stay as they are
    for text in (
        "The actor ran whoami /all to list privileges.",
        "Note: the actor used cmd. Later: they left.",
        "command_line: whoami /all",
        "The event said: pid: 5 ppid: 4",
        "pid: 5512. ppid: 4",
    ):
        assert exemplar_query_text(text) == text
    # retrieval queries the exemplars with the value-only text
    index = _index()
    calls = []

    class Recording(WordEmbedder):
        def embed(self, texts, batch_size=None):
            calls.extend(texts)
            return super().embed(texts, batch_size)

    kb, store, _ = _two_hit_setup()
    store.embedder = Recording()
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="command_line: powershell -enc AAA. pid: 42",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    retrieve_techniques(ledger, store, kb, retrieval_mode="dense", exemplar_index=index)
    assert "powershell -enc AAA" in calls


def test_loader_falls_back_to_the_no_exemplar_model_when_the_index_is_missing(tmp_path):
    index = _index()
    rows = [{name: float(i % 3) for name in FEATURE_NAMES} for i in range(12)]
    labels = [i % 2 for i in range(12)]
    primary = AcceptanceModel.fit(rows, labels, attack_version="t")
    primary.metadata = {"exemplar_fingerprint": index.fingerprint}
    fallback = AcceptanceModel.fit(rows, labels, attack_version="t")
    primary_path = primary.save(tmp_path / "primary.json")
    fallback_path = fallback.save(tmp_path / "fallback.json")
    kb = SimpleNamespace(version="t")
    settings = SimpleNamespace(
        acceptance_model_enabled=True,
        acceptance_model_path=primary_path,
        acceptance_model_fallback_path=fallback_path,
    )
    model, status = load_acceptance_model(settings, kb, index)
    assert status == "active" and model.requires_exemplars
    model, status = load_acceptance_model(settings, kb, None)
    assert model is not None and not model.requires_exemplars
    assert status.startswith("fallback without exemplars (incompatible artifact")
    # a fallback that itself needs exemplars is refused, and a missing one leaves the old reason
    settings.acceptance_model_fallback_path = primary_path.with_name("missing.json")
    assert load_acceptance_model(settings, kb, None)[0] is None
    primary.save(tmp_path / "also_needs.json")
    settings.acceptance_model_fallback_path = tmp_path / "also_needs.json"
    model, status = load_acceptance_model(settings, kb, None)
    assert model is None and status.startswith("incompatible artifact")
    assert (
        load_acceptance_model(SimpleNamespace(acceptance_model_enabled=False), kb)[1] == "disabled"
    )


# ----- refactor guard: a frozen copy of the original query loop -----
def _reference_query_vectors(index, vectors, *, parent_of=None, k=64, tau=0.05, dup=0.95):
    """``ExemplarIndex.query_vectors`` as it was before the backend refactor (2026-10-02),
    copied verbatim; the refactored code must match it bit for bit."""
    from apt_classifier.exemplars import ExemplarVotes, _unit

    queries = np.asarray(vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if not len(queries):
        return []
    queries = _unit(queries)
    similarities = queries @ index.embeddings.T
    parents = parent_of or {}
    k = max(1, int(k))
    tau = max(1e-4, float(tau))
    window = min(len(index), k * 2 + 16)
    results = []
    for row in similarities:
        if window < len(row):
            candidates = np.argpartition(-row, window - 1)[:window]
        else:
            candidates = np.arange(len(row))
        ordered = candidates[np.argsort(-row[candidates], kind="stable")]
        excluded = int(np.sum(row[ordered] >= dup))
        chosen = [int(j) for j in ordered if row[j] < dup][:k]
        if not chosen:
            results.append(ExemplarVotes(excluded_duplicates=excluded, k=k))
            continue
        sims = np.asarray([row[j] for j in chosen], dtype=np.float64)
        weights = np.exp((sims - sims[0]) / tau)
        weights = weights / weights.sum()
        votes, family, maxsim, counts, best_text, best_source = {}, {}, {}, {}, {}, {}
        for weight, sim, j in zip(weights, sims, chosen, strict=True):
            for label in index.labels[j]:
                votes[label] = votes.get(label, 0.0) + float(weight)
                counts[label] = counts.get(label, 0) + 1
                family[label] = family.get(label, 0.0) + float(weight)
                parent = parents.get(label)
                if parent:
                    family[parent] = family.get(parent, 0.0) + float(weight)
                if label not in maxsim or sim > maxsim[label]:
                    maxsim[label] = float(sim)
                    best_text[label] = index.texts[j]
                    best_source[label] = index.sources[j]
        ranked = sorted(votes, key=lambda technique_id: (-votes[technique_id], technique_id))
        results.append(
            ExemplarVotes(
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
        )
    return results


def _random_library(seed=3, count=400, dim=24):
    """Random unit vectors with exact duplicates (ties) and near-copies around cosine 0.95."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    if count > 41:
        vectors[10] = vectors[11] = vectors[12]  # an exact-tie group
        vectors[40] = vectors[41]
    for base in range(100, min(160, count - 100)):  # near-copies around the 0.95 duplicate cut
        noise = rng.normal(size=dim).astype(np.float32)
        noise -= noise.dot(vectors[base]) / vectors[base].dot(vectors[base]) * vectors[base]
        noise *= np.linalg.norm(vectors[base]) / np.linalg.norm(noise)
        angle = np.arccos(0.95 + rng.uniform(-0.004, 0.004))
        vectors[base + 100] = np.cos(angle) * vectors[base] + np.sin(angle) * noise
    pool = ["T1059", "T1059.001", "T1059.003", "T1003", "T1003.001", "T1105", "T1566.001", "T1053"]
    labels = tuple(
        tuple(sorted(set(rng.choice(pool, size=int(rng.integers(1, 4)), replace=True))))
        for _ in range(count)
    )
    index = ExemplarIndex(
        embeddings=np.asarray(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True), dtype=np.float32
        ),
        labels=labels,
        texts=tuple(f"text {row}" for row in range(count)),
        sources=tuple(("tram", "expert", "derived_procedures")[row % 3] for row in range(count)),
        metadata={"kind": "technique_exemplar_index", "schema_version": 1},
    )
    index.metadata["fingerprint"] = index.compute_fingerprint()
    queries = np.concatenate(
        [
            index.embeddings[:30],  # self queries: similarity 1 is always a duplicate
            index.embeddings[100:130] * 1.0,  # near-copies sit around the 0.95 cut
            rng.normal(size=(40, dim)).astype(np.float32),
            index.embeddings[10:13],  # exact ties
        ]
    )
    return index, queries


def _vote_bits(votes):
    from scripts.exemplar_backend_equivalence import _bits

    return [_bits([getattr(item, name) for name in item.__dataclass_fields__]) for item in votes]


@pytest.mark.parametrize(
    "k,tau,dup",
    [(64, 0.05, 0.95), (4, 0.05, 0.95), (16, 0.08, 0.9), (256, 0.03, 0.95), (500, 0.05, 1.01)],
)
def test_refactored_query_matches_the_frozen_reference_bit_for_bit(k, tau, dup):
    index, queries = _random_library()
    parents = {"T1059.001": "T1059", "T1059.003": "T1059", "T1003.001": "T1003"}
    expected = _reference_query_vectors(index, queries, parent_of=parents, k=k, tau=tau, dup=dup)
    actual = index.query_vectors(queries, parent_of=parents, k=k, tau=tau, duplicate_similarity=dup)
    assert _vote_bits(actual) == _vote_bits(expected)
    # one query at a time as well (a 1-D vector), and the empty batch
    single = index.query_vectors(
        queries[0], parent_of=parents, k=k, tau=tau, duplicate_similarity=dup
    )
    assert _vote_bits(single) == _vote_bits(
        _reference_query_vectors(index, queries[0], parent_of=parents, k=k, tau=tau, dup=dup)
    )
    assert index.query_vectors(np.zeros((0, queries.shape[1]), dtype=np.float32)) == []


def test_allowed_rows_superset_is_identical_and_a_missing_member_is_a_counted_miss():
    from apt_classifier.exemplars import window_size

    index, queries = _random_library()
    reference = index.query_vectors(queries, k=32)
    windows = index.exact_windows(queries, k=32)
    assert all(len(window) == window_size(len(index), 32) == 80 for window in windows)
    rng = np.random.default_rng(1)
    superset = [
        np.concatenate([window, rng.choice(len(index), size=50)])[::-1] for window in windows
    ]
    stats = {"recall_misses": 0}
    assert _vote_bits(index.query_vectors(queries, k=32, allowed_rows=superset, stats=stats)) == (
        _vote_bits(reference)
    )
    assert stats["recall_misses"] == 0
    # a candidate search that lost one member of two windows: counted, votes still exact
    lossy = [window.copy() for window in windows]
    lossy[0] = lossy[0][1:]
    lossy[5] = lossy[5][:-1]
    stats = {"recall_misses": 0}
    assert _vote_bits(index.query_vectors(queries, k=32, allowed_rows=lossy, stats=stats)) == (
        _vote_bits(reference)
    )
    assert stats["recall_misses"] == 2
    with pytest.raises(ValueError):
        index.query_vectors(queries, k=32, allowed_rows=lossy[:3])
    with pytest.raises(ValueError):
        index.query_vectors(queries, vote_norm="softmax")


def test_label_split_divides_each_neighbours_weight_by_its_label_count():
    index = _index()
    query = WordEmbedder().embed(["powershell downloaded a payload after a phishing email"])
    plain = index.query_vectors(query, parent_of=_parents(), k=4, duplicate_similarity=0.999)[0]
    split = index.query_vectors(
        query, parent_of=_parents(), k=4, duplicate_similarity=0.999, vote_norm="label_split"
    )[0]
    # the multi-label exemplar (T1059.001 + T1105) gives each label half its weight
    assert sum(split.votes.values()) == pytest.approx(1.0)
    assert sum(plain.votes.values()) > 1.0
    assert split.votes["T1105"] < plain.votes["T1105"]
    assert split.counts == plain.counts and split.max_similarity == plain.max_similarity


def test_npz_index_signature_and_backend_dispatch(tmp_path):
    index = _index()
    path = index.save(tmp_path / "ex.npz")
    assert index.backend == "npz" and index.search == "exact" and index.search_class == "exact"
    assert index.signature() == {
        "backend": "npz",
        "search": "exact",
        "search_class": "exact",
        "fingerprint": index.fingerprint,
        "count": len(index),
    }
    base = {
        "technique_exemplars_enabled": True,
        "technique_exemplar_index_path": path,
        "embedding_model": "test-model",
        "embedding_model_revision": "r1",
    }
    # the default backend is the npz, whatever client is passed
    loaded, status = load_exemplar_index(SimpleNamespace(**base), client=object())
    assert status == "active" and loaded.backend == "npz"
    unknown = SimpleNamespace(**base, technique_exemplar_backend="sqlite")
    assert load_exemplar_index(unknown)[0] is None
    assert "unknown exemplar backend" in load_exemplar_index(unknown)[1]
    # no client: strict chroma gives no library, auto falls back to the npz with the reason
    strict, reason = load_exemplar_index(
        SimpleNamespace(**base, technique_exemplar_backend="chroma")
    )
    assert strict is None and "no Chroma client" in reason
    auto, status = load_exemplar_index(SimpleNamespace(**base, technique_exemplar_backend="auto"))
    assert auto is not None and auto.backend == "npz"
    assert status.startswith("active (npz fallback: no Chroma client")


# ----- tuning knobs: vote normalisation, per-sentence votes, the legacy contract -----


def test_pool_votes_combines_queries_and_recomputes_ranks():
    from apt_classifier.exemplars import ExemplarVotes, pool_votes

    whole = ExemplarVotes(
        top_similarity=0.8,
        votes={"T1": 0.6, "T2": 0.4},
        family={"T1": 0.6, "T2": 0.4, "T0": 0.6},
        max_similarity={"T1": 0.8, "T2": 0.7},
        counts={"T1": 3, "T2": 2},
        ranks={"T1": 1, "T2": 2},
        best_text={"T1": "whole t1", "T2": "whole t2"},
        best_source={"T1": "tram", "T2": "tram"},
        neighbours=5,
        excluded_duplicates=1,
        k=4,
    )
    sentence = ExemplarVotes(
        top_similarity=0.9,
        votes={"T2": 0.9, "T3": 0.1},
        family={"T2": 0.9, "T3": 0.1},
        max_similarity={"T2": 0.9, "T3": 0.5},
        counts={"T2": 4, "T3": 1},
        ranks={"T2": 1, "T3": 2},
        best_text={"T2": "sentence t2", "T3": "sentence t3"},
        best_source={"T2": "expert", "T3": "expert"},
        neighbours=4,
        excluded_duplicates=0,
        k=4,
    )
    pooled = pool_votes([whole, sentence], "max")
    assert pooled.votes == {"T1": 0.6, "T2": 0.9, "T3": 0.1}
    assert pooled.ranked == ["T2", "T1", "T3"]
    assert pooled.counts == {"T1": 3, "T2": 4, "T3": 1}
    assert pooled.family["T0"] == 0.6
    assert pooled.top_similarity == 0.9 and pooled.max_similarity["T2"] == 0.9
    assert pooled.best_text["T2"] == "sentence t2" and pooled.best_source["T1"] == "tram"
    assert pooled.neighbours == 5 and pooled.excluded_duplicates == 1 and pooled.k == 4
    mean = pool_votes([whole, sentence], "mean")
    assert mean.votes["T1"] == pytest.approx(0.3) and mean.votes["T2"] == pytest.approx(0.65)
    assert mean.counts["T2"] == pytest.approx(3.0)
    assert mean.ranked == ["T2", "T1", "T3"]
    assert pool_votes([whole], "max") is whole
    with pytest.raises(ValueError):
        pool_votes([whole, sentence], "off")


def test_exemplar_kwargs_carry_the_new_knobs_with_shipped_defaults():
    kwargs = exemplar_kwargs(SimpleNamespace(), None)
    assert kwargs["exemplar_vote_norm"] == "none" and kwargs["exemplar_segment_votes"] == "off"
    tuned = exemplar_kwargs(
        SimpleNamespace(
            technique_exemplar_vote_norm="label_split", technique_exemplar_segment_votes="max"
        ),
        None,
    )
    assert tuned["exemplar_vote_norm"] == "label_split"
    assert tuned["exemplar_segment_votes"] == "max"


class _RecordingIndex:
    """Wraps an index and records each query_vectors call (count of vectors, keyword names)."""

    def __init__(self, index):
        self.index = index
        self.calls = []

    @property
    def fingerprint(self):
        return self.index.fingerprint

    def query_vectors(self, vectors, **kwargs):
        self.calls.append((len(vectors), sorted(kwargs)))
        return self.index.query_vectors(vectors, **kwargs)


LONG_OBSERVATION = (
    "During the intrusion the operators relied on a long chain of tools across several hosts "
    "over many days. First they ran powershell to download a second stage from their server. "
    "Later they dumped credentials from lsass memory with mimikatz and created a scheduled "
    "task for persistence on the domain controller."
)


def _segment_setup():
    kb, store, _ = _two_hit_setup()
    for technique_id, name in (
        ("T1105", "Ingress Tool Transfer"),
        ("T1053.005", "Scheduled Task"),
        ("T1566.001", "Spearphishing Attachment"),
        ("T1059", "Command and Scripting Interpreter"),
    ):
        kb.techniques[technique_id] = Technique(id=technique_id, name=name)
    store.embedder = WordEmbedder()
    ledger = EvidenceLedger(
        observations=[Observation(text=LONG_OBSERVATION, provenance=Provenance(source="free_text"))]
    )
    return kb, store, ledger


def test_segment_votes_query_each_sentence_and_off_keeps_the_shipped_query():
    kb, store, ledger = _segment_setup()
    settings = SimpleNamespace(
        technique_exemplar_k=4,
        technique_exemplar_tau=0.05,
        technique_exemplar_duplicate_similarity=0.999,
        technique_exemplar_new_candidates=2,
    )
    off_index = _RecordingIndex(_index())
    off = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", **exemplar_kwargs(settings, off_index)
    )
    # one query for the whole text, and no vote_norm argument at the default
    assert off_index.calls == [(1, ["duplicate_similarity", "k", "parent_of", "tau"])]
    assert "exemplar_segment_votes" not in off.diagnostics
    pooled_index = _RecordingIndex(_index())
    pooled = retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        **{**exemplar_kwargs(settings, pooled_index), "exemplar_segment_votes": "max"},
    )
    assert len(pooled_index.calls) == 2 and pooled_index.calls[1][0] >= 2
    assert pooled.diagnostics["exemplar_segment_votes"] == "max"

    # the scheduled-task sentence now votes for its technique on its own, and the exemplar-only
    # candidates follow the pooled ranking
    def vote(result, technique_id):
        candidate = next(c for c in result.candidates if c.technique_id == technique_id)
        return candidate.evidence_for(EVIDENCE_SOURCE).metadata["features"]["ex_vote"]

    def exemplar_only(result):
        return [
            c.technique_id
            for c in result.candidates
            if c.evidence_for(EVIDENCE_SOURCE).metadata["exemplar_only"]
        ]

    assert vote(pooled, "T1053.005") > vote(off, "T1053.005")
    assert exemplar_only(off) == ["T1105", "T1059"]
    assert exemplar_only(pooled) == ["T1059", "T1105"]
    # label_split is passed through to the library
    split_index = _RecordingIndex(_index())
    retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        **{**exemplar_kwargs(settings, split_index), "exemplar_vote_norm": "label_split"},
    )
    assert "vote_norm" in split_index.calls[0][1]


def test_shipped_acceptance_model_gives_identical_results_on_the_legacy_path():
    """The shipped artifact (read only) has no stored contract: with the default settings, and
    with any knob that existed before contracts, retrieval_kwargs runs it exactly as the old
    explicit arguments ran the unconstrained model."""
    import copy
    from pathlib import Path

    from apt_classifier.config import Settings
    from apt_classifier.presets import apply_preset
    from apt_classifier.retrieve import LexicalTechniqueIndex, lexical_fields, retrieval_kwargs
    from tests.test_retrieve import old_serving_kwargs

    path = Path(__file__).resolve().parents[1] / "data/models/acceptance/acceptance_model.json"
    if not path.is_file():
        pytest.skip("shipped acceptance artifact is not present")
    shipped = AcceptanceModel.load(path)
    if shipped.contract_origin != "legacy":
        # the shipped artifact carried no contract until the ATT&CK 20.0 refit (2026-10-21); the
        # legacy path it exercised is covered by the synthetic legacy artifacts of this module
        pytest.skip("the shipped acceptance artifact records its contract")
    unconstrained = copy.deepcopy(shipped)
    unconstrained.contract = None
    kb, store, _ = _segment_setup()
    index = _index()
    index.metadata["fingerprint"] = shipped.exemplar_fingerprint  # stands in for the library
    settings = apply_preset(Settings(_env_file=None))
    lexical = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(settings))
    for text in (
        "powershell downloaded a payload and read lsass",
        LONG_OBSERVATION,
        "Routine maintenance used powershell to download updates",
    ):
        ledger = EvidenceLedger(
            observations=[Observation(text=text, provenance=Provenance(source="free_text"))]
        )
        new = retrieve_techniques(
            ledger,
            store,
            kb,
            **retrieval_kwargs(settings, exemplar_index=index, lexical_index=lexical),
            acceptance_model=shipped,
        )
        old = retrieve_techniques(
            copy.deepcopy(ledger),
            store,
            kb,
            **old_serving_kwargs(settings, index, lexical),
            acceptance_model=unconstrained,
        )
        assert new.diagnostics["acceptance_model_active"] is True
        assert "acceptance_model_disabled" not in new.diagnostics
        assert new.to_dict() == old.to_dict()
    # a knob that existed before contracts runs it as before (unconstrained), one new with
    # contracts refuses it
    kwargs = retrieval_kwargs(settings, exemplar_index=index, lexical_index=lexical)
    older = retrieve_techniques(
        ledger, store, kb, **{**kwargs, "exemplar_k": 128}, acceptance_model=shipped
    )
    before = retrieve_techniques(
        copy.deepcopy(ledger),
        store,
        kb,
        **{**old_serving_kwargs(settings, index, lexical), "exemplar_k": 128},
        acceptance_model=unconstrained,
    )
    assert "acceptance_model_disabled" not in older.diagnostics
    assert older.to_dict() == before.to_dict()
    tuned = retrieve_techniques(
        ledger,
        store,
        kb,
        **{**kwargs, "exemplar_vote_norm": "label_split"},
        acceptance_model=shipped,
    )
    assert (
        "exemplar_vote_norm 'label_split' (fitted with 'none')"
        in (tuned.diagnostics["acceptance_model_disabled"])
    )
