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
