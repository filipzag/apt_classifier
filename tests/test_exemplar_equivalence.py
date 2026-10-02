"""The exemplar backend equivalence harness (levels L1-L4) on a fake Chroma client."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from apt_classifier import exemplar_store
from apt_classifier.exemplar_store import load_chroma_exemplar_store
from scripts.exemplar_backend_equivalence import (
    CachedEmbedder,
    compare_mapping_runs,
    compare_vote_lists,
    recall_level,
    refused_set,
    run_vote_levels,
    vote_differences,
)
from tests.test_exemplar_store import _library, _settings, _synced


@pytest.fixture(autouse=True)
def _fresh_memo():
    exemplar_store.clear_memo()
    yield
    exemplar_store.clear_memo()


def _stores():
    client, index, name = _synced()
    exact, _ = load_chroma_exemplar_store(_settings(name), client)
    hnsw, _ = load_chroma_exemplar_store(_settings(name, search="hnsw"), client)
    return client, index, name, exact, hnsw


def test_vote_levels_pass_when_the_copy_is_identical():
    client, index, name, exact, hnsw = _stores()
    _, queries = _library()
    report = run_vote_levels(
        index,
        exact,
        hnsw,
        {"self": index.embeddings, "queries": queries},
        levels=("L1", "L2", "L3"),
        parents={"T1059.001": "T1059"},
        k=64,
        tau=0.05,
        duplicate_similarity=0.95,
        batch_size=37,
        wide_k=256,
        wide_k_sets=("queries",),
    )
    assert all(entry["passed"] for level in report.values() for entry in level.values())
    assert (
        report["L1"]["self"]["queries"] == len(index) and report["L1"]["self"]["differences"] == 0
    )
    assert report["L2"]["queries"]["wide_k"]["window"] == 400  # 2*256+16 capped at the library
    assert report["L2"]["queries"]["default_k"]["recall_misses"] == 0
    # raw recall@window is informational: float32 vs float64 order at the window edge can swap
    assert report["L2"]["self"]["default_k"]["raw_recall_at_window_mean"] > 0.999
    assert client.writes() == []


def test_recall_level_reports_misses_while_the_votes_stay_identical():
    client, index, name, exact, hnsw = _stores()
    _, queries = _library()
    window = index.exact_windows(queries[:1], k=64)[0]
    client.collections[name].hidden_ids = {f"ex-{int(row):05d}" for row in window[:3]}
    recall = recall_level(index, hnsw, queries, k=64, batch_size=50)
    assert not recall["passed"] and recall["recall_misses"] >= 1
    assert recall["raw_recall_at_window_min"] < 1.0
    report = run_vote_levels(
        index,
        exact,
        hnsw,
        {"queries": queries},
        levels=("L3",),
        parents={},
        k=64,
        tau=0.05,
        duplicate_similarity=0.95,
        batch_size=50,
    )
    assert report["L3"]["queries"]["passed"]  # Chroma proposes candidates; votes stay exact


def test_vote_comparison_is_bitwise():
    index, queries = _library()
    votes = index.query_vectors(queries[:3])
    nudged = dataclasses.replace(votes[1], top_similarity=np.nextafter(votes[1].top_similarity, 2))
    assert vote_differences(votes[1], nudged) == ["top_similarity"]
    reordered = dataclasses.replace(votes[2], votes=dict(reversed(list(votes[2].votes.items()))))
    if len(votes[2].votes) > 1:
        assert vote_differences(votes[2], reordered) == ["votes"]  # insertion order counts
    summary = compare_vote_lists(votes, [votes[0], nudged, votes[2]])
    assert summary["differences"] == 1 and summary["examples"][0]["query"] == 1


def _run_block(probability=0.4, state="auto_accepted"):
    return {
        "runtime_seconds": 1.0,
        "pipeline_fingerprint": "x",
        "pooled": {"metrics": {"micro_f1": 0.5}},
        "datasets": {"tram": {"metrics": {"micro_f1": 0.5}}},
        "cases": [
            {
                "case_id": "c1",
                "ranked_technique_ids": ["T1"],
                "accepted_technique_ids": ["T1"],
                "expected_in_pool": True,
                "expected_accepted": True,
                "candidates": [
                    {"technique_id": "T1", "state": state, "acceptance_probability": probability}
                ],
            }
        ],
    }


def test_mapping_runs_compare_rows_and_metrics_but_not_timings():
    base = _run_block()
    same = _run_block()
    same["runtime_seconds"] = 9.0
    same["pipeline_fingerprint"] = "y"  # a different backend signature
    assert compare_mapping_runs(base, same)["passed"]
    flipped = _run_block(probability=float(np.nextafter(0.4, 1.0)))
    result = compare_mapping_runs(base, flipped)
    assert not result["passed"] and result["examples"][0]["fields"] == ["candidates"]
    other_metrics = _run_block()
    other_metrics["datasets"]["tram"]["metrics"]["micro_f1"] = 0.51
    assert not compare_mapping_runs(base, other_metrics)["metrics_equal"]


def test_cached_embedder_embeds_each_text_once_and_persists(tmp_path):
    calls = []

    class Inner:
        def embed(self, texts, batch_size=None):
            calls.append(list(texts))
            return [[float(len(text)), 0.5, -1.0] for text in texts]

    path = tmp_path / "cache.npz"
    cached = CachedEmbedder(Inner, path)
    first = cached.embed(["a", "bb", "a"])
    second = cached.embed(["bb", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]] and first[0] == first[2] and second[0] == first[1]
    cached.save()
    reloaded = CachedEmbedder(None, path)  # no embedder: everything must come from the cache
    assert reloaded.loaded == 3 and reloaded.embed(["ccc", "a"]) == [second[1], first[0]]
    with pytest.raises(RuntimeError):
        reloaded.embed(["missing"])


def test_clinic_and_elastic_sets_are_refused():
    assert refused_set("evidenceforge:clinic")
    assert refused_set("evidenceforge:mfg", ["/data/evidenceforge_clinic", "scenario.yaml"])
    assert refused_set("elastic")
    assert (
        refused_set("evidenceforge:branch-office", ["/data/evidenceforge_output", "s.yaml"]) is None
    )


# ----- L4 end to end (fake Chroma, fake dense store, fake embedder, a tiny contract-less model)
def _l4_main(tmp_path, monkeypatch, *, ignore_contract_at_load=False, **settings_changes):
    """Run ``main --levels L4`` the way R11 does, on fakes only; returns (exit code, L4 block)."""
    import json
    import zlib

    import apt_classifier.acceptance as acceptance
    import apt_classifier.config as config
    import apt_classifier.embeddings as embeddings
    import apt_classifier.vectorstore as vectorstore
    import scripts.exemplar_backend_equivalence as equivalence
    from apt_classifier.acceptance import FEATURE_NAMES, AcceptanceModel
    from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from apt_classifier.vectorstore import VectorHit
    from tests.test_exemplar_store import MODEL, REVISION

    client, index, name = _synced(index=_library(count=200)[0])
    npz_path = index.save(tmp_path / "exemplars.npz")
    technique_ids = sorted({label for labels in index.labels for label in labels} | {"T1566"})
    kb = AttackKnowledgeBase(
        version="19.2",
        techniques={
            tid: Technique(
                id=tid,
                name=f"technique {tid}",
                is_subtechnique="." in tid,
                parent_id=tid.split(".")[0] if "." in tid else None,
            )
            for tid in technique_ids
        },
    )
    kb.save(tmp_path / "kb" / "attack_kb.json")
    # a model saved the way the shipped one is: exemplar features, no stored contract
    rng = np.random.default_rng(5)
    rows, labels = [], []
    for _ in range(200):
        dense = float(rng.uniform(0.3, 0.95))
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        row.update({"dense_score": dense, "dense_present": 1.0})
        rows.append(row)
        labels.append(int(dense + rng.normal(0, 0.05) > 0.7))
    model = AcceptanceModel.fit(rows, labels, attack_version="19.2", threshold=0.5)
    model.metadata = {"exemplar_fingerprint": index.fingerprint}
    model_path = tmp_path / "acceptance.json"
    model_path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    settings = config.Settings(
        _env_file=None,
        data_dir=tmp_path,
        technique_exemplar_index_path=npz_path,
        acceptance_model_path=model_path,
        acceptance_model_fallback_path=None,
        embedding_model=MODEL,
        embedding_model_revision=REVISION,
        **settings_changes,
    )
    monkeypatch.setattr(config, "Settings", lambda *args, **kwargs: settings)

    rows_by_text = {str(text): row for row, text in enumerate(index.texts)}

    class Embedder:
        def embed(self, texts, batch_size=None, **_):
            out = []
            for text in texts:
                row = rows_by_text.get(str(text).strip())
                if row is not None:
                    out.append(index.embeddings[row].tolist())
                    continue
                seed = zlib.crc32(str(text).encode())
                vector = np.random.default_rng(seed).normal(size=index.embeddings.shape[1])
                out.append((vector / np.linalg.norm(vector)).astype(np.float32).tolist())
            return out

    class SentenceTransformerEmbedder:
        @classmethod
        def from_settings(cls, *args, **kwargs):
            return Embedder()

    class DenseStore:
        def __init__(self, settings, embedder, client=None):
            self.settings, self.embedder, self.client = settings, embedder, client

        def query_techniques(self, texts, *, n_results, fields=None, **kwargs):
            out = []
            for text in texts:
                row = rows_by_text.get(str(text).strip(), 0)
                out.append(
                    [
                        VectorHit(
                            f"{tid}:{rank}",
                            tid,
                            {"technique_id": tid, "name": tid, "field": "description"},
                            0.2 + 0.05 * rank,
                            0.8 - 0.05 * rank,
                        )
                        for rank, tid in enumerate(index.labels[row])
                    ]
                )
            return out

        def query_group_profile(self, texts, *, document_type, n_results):
            return []

        def query_group_behaviors(self, texts, *, n_results):
            return [[] for _ in texts]

    cases = [
        TechniqueEvaluationCase(
            case_id=f"dev-{row}",
            document_id=f"dev-{row}",
            text=str(index.texts[row]),
            expected_technique_ids=tuple(index.labels[row]),
            source=f"tumeteor/mitre-ttp-mapping:{('tram', 'expert')[row % 2]}",
        )
        for row in range(0, 200, 25)
    ]
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", SentenceTransformerEmbedder)
    monkeypatch.setattr(vectorstore, "ChromaVectorStore", DenseStore)
    monkeypatch.setattr(equivalence, "load_set_cases", lambda name, **_: list(cases))
    if ignore_contract_at_load:
        # the loader as it was before it received the runtime contract
        real = acceptance.load_acceptance_model
        monkeypatch.setattr(
            acceptance,
            "load_acceptance_model",
            lambda settings, kb, source, contract=None: real(settings, kb, source),
        )
    output = tmp_path / "equivalence.json"
    code = equivalence.main(
        [
            "--collection",
            name,
            "--sets",
            "self",
            "--levels",
            "L4",
            "--l4-sets",
            "public:dev",
            "--output",
            str(output),
            "--allow-shared-output",
            "--overwrite",
        ],
        client=client,
    )
    assert not client.writes()
    return code, json.loads(output.read_text(encoding="utf-8"))["levels"]["L4"]


def test_l4_passes_only_while_the_model_actually_scores(tmp_path, monkeypatch):
    code, level = _l4_main(tmp_path, monkeypatch)
    assert code == 0 and level["passed"]
    for entry in level["arms"].values():
        assert entry["acceptance_model_status"] == "active" and entry["model_active"]
        assert entry["model_scored"] and entry["model_refused_cases"] == 0
        assert entry["probability_rows"] > 0


def test_l4_fails_when_a_setting_breaks_the_model_contract(tmp_path, monkeypatch):
    """Regression: with a knob the model was not fitted with, every arm ran the rule-based gate,
    the arms matched each other and L4 passed while reporting the model active."""
    code, level = _l4_main(tmp_path, monkeypatch, technique_dense_overfetch=90)
    assert code == 1 and not level["passed"]
    for entry in level["arms"].values():
        assert "dense_overfetch 90 (fitted with 0)" in entry["acceptance_model_status"]
        assert not entry["model_active"]


def test_l4_fails_when_the_model_is_refused_inside_retrieval(tmp_path, monkeypatch):
    """Even if the load status says active, a model refused per call fails L4."""
    code, level = _l4_main(
        tmp_path, monkeypatch, ignore_contract_at_load=True, technique_dense_overfetch=90
    )
    assert code == 1 and not level["passed"]
    for entry in level["arms"].values():
        assert entry["model_active"]  # what the load status alone claimed
        assert not entry["model_scored"] and entry["probability_rows"] == 0
        assert entry["model_refused_cases"] > 0
        assert any("dense_overfetch 90" in reason for reason in entry["model_refused_reasons"])
