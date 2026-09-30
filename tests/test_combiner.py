"""Learned attribution combiner (apt_classifier/combiner.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.combiner import (
    FEATURE_NAMES,
    NULL_FEATURE_NAMES,
    AttributionCombiner,
    CombinerList,
    apply_combiner_order,
    combiner_inputs,
    load_attribution_combiner,
    pipeline_fingerprint,
)
from apt_classifier.fusion import reciprocal_rank_fusion
from apt_classifier.separation import discriminate
from tests.test_separation import _context, _kb


def _result(name: str, ordered: list[tuple[str, float]]) -> ClassifierResult:
    entries = tuple(
        ClassifierEvidence(classifier=name, group_id=gid, score=score, rank=rank)
        for rank, (gid, score) in enumerate(ordered, 1)
    )
    return ClassifierResult(classifier=name, applicable=True, entries=entries)


def _case(observed=None, software=None):
    kb = _kb()
    results = (
        _result("bernoulli_naive_bayes", [("G1", -1.0), ("G2", -2.0), ("G3", -5.0)]),
        _result("idf_tfidf_cosine", [("G1", 0.8), ("G3", 0.5), ("G2", 0.4)]),
        _result("software_tools", [("G1", 2.0)]),
        _result("group_description", [("G3", 0.3), ("G2", 0.2), ("G1", 0.1)]),
    )
    weights = {
        name: 1.0
        for name in (
            "bernoulli_naive_bayes",
            "idf_tfidf_cosine",
            "software_tools",
            "group_description",
        )
    }
    context = _context(
        observed_techniques=observed if observed is not None else {"T1": 0.9, "T2": 1.0},
        software_ids=software if software is not None else {"S_mal"},
        eligible_group_ids={"G1", "G2", "G3"},
    )
    fusion = reciprocal_rank_fusion(results, weights, 20, eligible_group_ids={"G1", "G2", "G3"})
    basis = discriminate(kb, fusion, context, top_k=3)
    return kb, results, weights, context, fusion, basis


def test_feature_rows_are_deterministic_and_cover_the_schema():
    kb, results, weights, context, fusion, basis = _case()
    first = combiner_inputs(
        results, fusion, basis, context, kb, top_n=3, weights=weights, truth="G1"
    )
    second = combiner_inputs(
        results, fusion, basis, context, kb, top_n=3, weights=weights, truth="G1"
    )
    assert first.group_ids == tuple(e.group_id for e in fusion.entries[:3])
    assert first.features.shape == (3, len(FEATURE_NAMES))
    assert np.array_equal(first.features, second.features) and first.regime == second.regime
    row = dict(zip(FEATURE_NAMES, first.features[first.truth_index], strict=True))
    assert row["software_tools_present"] == 1.0 and row["software_tools_inv_rank"] == 1.0
    assert row["bernoulli_naive_bayes_score_rel"] == pytest.approx(1.0)  # top NB score
    assert row["rrf_rel"] == pytest.approx(1.0) and row["rrf_inv_rank"] == 1.0
    assert 0.0 < row["matched_share"] <= 1.0
    assert row["exclusive_software"] == 1.0  # S_mal is documented for G1 alone
    assert first.regime["applicable_software"] == 1.0 and first.regime["software_rarity_max"] > 0
    # a group absent from a classifier gets zeros for that classifier
    g2 = dict(zip(FEATURE_NAMES, first.features[first.group_ids.index("G2")], strict=True))
    assert g2["software_tools_present"] == 0.0 and g2["software_tools_inv_rank"] == 0.0
    # interactions are products of the base and the regime value
    assert row["rrf_inv_rank_x_log_accepted"] == pytest.approx(
        row["rrf_inv_rank"] * first.regime["log_accepted"]
    )
    # round trip through dicts
    assert CombinerList.from_dict(first.to_dict()).features.tolist() == first.features.tolist()
    # without software the regime says so
    _, _, _, context2, fusion2, basis2 = _case(software=set())
    bare = combiner_inputs(results, fusion2, basis2, context2, kb, top_n=3, weights=weights)
    assert bare.regime["software_rarity_max"] == 0.0 and bare.regime["applicable_software"] == 1.0


def _synthetic_lists(rng: np.random.Generator, count: int, *, null_share: float = 0.0):
    """Lists whose truth is decided by one feature (rrf_rel) plus noise."""
    lists = []
    for index in range(count):
        n = 5
        features = rng.normal(size=(n, len(FEATURE_NAMES)))
        signal = FEATURE_NAMES.index("rrf_rel")
        truth_index = int(rng.integers(0, n))
        features[truth_index, signal] += 3.0
        regime = {name: float(rng.uniform(0, 1)) for name in NULL_FEATURE_NAMES[1:]}
        is_null = rng.uniform() < null_share
        if is_null:
            features[truth_index, signal] -= 3.0
            regime["log_accepted"] = 0.05  # thin evidence marks the null lists
        lists.append(
            CombinerList(
                tuple(f"G{index}_{i}" for i in range(n)),
                features,
                regime,
                None if is_null else f"G{index}_{truth_index}",
            )
        )
    return lists


def test_conditional_logit_recovers_a_synthetic_signal_and_sums_to_one():
    rng = np.random.default_rng(3)
    train = _synthetic_lists(rng, 300, null_share=0.2)
    model = AttributionCombiner.fit(train, attack_version="t", pipeline_fingerprint="fp", l2=0.5)
    assert model.weights[FEATURE_NAMES.index("rrf_rel")] > 1.0
    assert model.null_weights is not None and len(model.null_weights) == len(NULL_FEATURE_NAMES)
    test = _synthetic_lists(np.random.default_rng(4), 100)
    hits = 0
    for lst in test:
        probabilities, null_probability = model.probabilities(lst)
        assert sum(probabilities.values()) + (null_probability or 0.0) == pytest.approx(1.0)
        hits += max(probabilities, key=probabilities.get) == lst.truth
    assert hits >= 80  # one informative feature among 73 noisy ones under L2
    # the null option responds to the regime
    thin = _synthetic_lists(np.random.default_rng(5), 50, null_share=1.0)
    null_mean = np.mean([model.probabilities(lst)[1] for lst in thin])
    rich_mean = np.mean([model.probabilities(lst)[1] for lst in test])
    assert null_mean > rich_mean
    # without the null option probabilities live on the candidates only
    plain = AttributionCombiner.fit(
        train, attack_version="t", pipeline_fingerprint="fp", null_option=False
    )
    probabilities, null_probability = plain.probabilities(test[0])
    assert null_probability is None and sum(probabilities.values()) == pytest.approx(1.0)


def test_artifact_round_trip_and_rejections(tmp_path):
    rng = np.random.default_rng(1)
    model = AttributionCombiner.fit(
        _synthetic_lists(rng, 60), attack_version="19.2", pipeline_fingerprint="abc"
    )
    path = model.save(tmp_path / "c.json")
    loaded = AttributionCombiner.load(path)
    assert loaded is not None and np.allclose(loaded.weights, model.weights)
    assert loaded.metadata["lists"] == 60 and loaded.top_n == 20
    payload = json.loads(path.read_text())
    for broken in (
        {**payload, "kind": "other"},
        {**payload, "feature_names": payload["feature_names"][:-1]},
        {**payload, "feature_schema_version": 99},
        {**payload, "scales": [0.0] * len(FEATURE_NAMES)},
        {**payload, "weights": [1.0]},
        {**payload, "null_feature_names": ["x"]},
    ):
        with pytest.raises(ValueError):
            AttributionCombiner.from_dict(broken)
    assert AttributionCombiner.load(tmp_path / "missing.json") is None


def test_loader_fails_closed(tmp_path):
    rng = np.random.default_rng(2)
    model = AttributionCombiner.fit(
        _synthetic_lists(rng, 40), attack_version="19.2", pipeline_fingerprint="fp1"
    )
    path = model.save(tmp_path / "c.json")
    kb = SimpleNamespace(version="19.2")
    settings = SimpleNamespace(attribution_combiner_enabled=True, attribution_combiner_path=path)
    loaded, status = load_attribution_combiner(settings, kb, fingerprint="fp1")
    assert loaded is not None and status == "active"
    assert (
        load_attribution_combiner(
            SimpleNamespace(attribution_combiner_enabled=False), kb, fingerprint="fp1"
        )[1]
        == "disabled"
    )
    assert (
        "not found"
        in load_attribution_combiner(
            SimpleNamespace(
                attribution_combiner_enabled=True, attribution_combiner_path=tmp_path / "x.json"
            ),
            kb,
            fingerprint="fp1",
        )[1]
    )
    assert (
        "incompatible"
        in load_attribution_combiner(settings, SimpleNamespace(version="20.0"), fingerprint="fp1")[
            1
        ]
    )
    assert "fingerprint" in load_attribution_combiner(settings, kb, fingerprint="other")[1]
    (tmp_path / "bad.json").write_text("{not json")
    assert (
        "invalid"
        in load_attribution_combiner(
            SimpleNamespace(
                attribution_combiner_enabled=True, attribution_combiner_path=tmp_path / "bad.json"
            ),
            kb,
            fingerprint="fp1",
        )[1]
    )


def test_pipeline_fingerprint_depends_on_weights_and_sources(tmp_path):
    a = pipeline_fingerprint(rrf_k=20, weights={"x": 1.0}, family_weights={"f": 2.0})
    b = pipeline_fingerprint(rrf_k=20, weights={"x": 1.5}, family_weights={"f": 2.0})
    c = pipeline_fingerprint(rrf_k=20, weights={"x": 1.0}, family_weights={"f": 2.0})
    assert a == c and a != b
    src = tmp_path / "s.py"
    src.write_text("a = 1")
    d = pipeline_fingerprint(rrf_k=20, weights={}, family_weights={}, sources=[src])
    src.write_text("a = 2")
    assert d != pipeline_fingerprint(rrf_k=20, weights={}, family_weights={}, sources=[src])


def test_combiner_order_keeps_the_top_n_set_and_the_tail():
    kb, results, weights, context, fusion, basis = _case()
    probabilities = {"G3": 0.7, "G2": 0.2, "G1": 0.1}
    reordered = apply_combiner_order(fusion, probabilities, blend=1.0, top_n=2)
    head = [e.group_id for e in fusion.entries[:2]]
    assert {e.group_id for e in reordered.entries[:2]} == set(head)
    assert [e.group_id for e in reordered.entries[2:]] == [e.group_id for e in fusion.entries[2:]]
    assert reordered.entries[0].relative_score == pytest.approx(100.0)
    assert [e.rank for e in reordered.entries] == list(range(1, len(fusion.entries) + 1))
    scores = [e.score for e in reordered.entries]
    assert scores == sorted(scores, reverse=True)
    # blend 0 is the RRF order and object
    assert apply_combiner_order(fusion, probabilities, blend=0.0, top_n=3) is fusion
    # a full blend puts the most probable candidate first
    full = apply_combiner_order(fusion, probabilities, blend=1.0, top_n=3)
    assert full.entries[0].group_id == "G3"
    assert (
        full.entries[0].classifier_ranks
        == next(e for e in fusion.entries if e.group_id == "G3").classifier_ranks
    )


def test_an_artifact_can_use_a_subset_of_the_current_features():
    """A schema-1 artifact (no source_country features) keeps scoring on its own columns."""
    rng = np.random.default_rng(6)
    lists = _synthetic_lists(rng, 60)
    model = AttributionCombiner.fit(lists, attack_version="t", pipeline_fingerprint="fp")
    keep = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith("source_country_")]
    payload = model.to_dict()
    payload["feature_schema_version"] = 1
    payload["feature_names"] = [FEATURE_NAMES[i] for i in keep]
    for key in ("weights", "means", "scales"):
        payload[key] = [payload[key][i] for i in keep]
    older = AttributionCombiner.from_dict(payload)
    assert older.feature_names == tuple(FEATURE_NAMES[i] for i in keep)
    lst = lists[0]
    expected = (lst.features[:, keep] - older.means) / older.scales @ older.weights
    utilities, _ = older.utilities(lst)
    assert np.allclose(utilities, expected)
    assert older.to_dict()["feature_names"] == payload["feature_names"]
    with pytest.raises(ValueError):
        AttributionCombiner.from_dict(
            {**payload, "feature_names": [*payload["feature_names"][:-1], "nope"]}
        )
    with pytest.raises(ValueError):
        AttributionCombiner.from_dict({**payload, "feature_schema_version": 0})
