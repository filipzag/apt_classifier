"""PyTorch group network (apt_classifier/group_mlp.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique  # noqa: E402
from apt_classifier.group_mlp import (  # noqa: E402
    GroupMLP,
    MLPFeatures,
    classification_metrics,
    fit_group_mlp,
    group_tools,
    load_group_mlp,
    prominence_log_prior,
    training_cases,
)

SMALL = {
    "samples_per_group": 60,
    "size_min": 1,
    "size_max": 3,
    "hidden": [32],
    "epochs": 12,
    "resample_every": 4,
    "batch_size": 64,
    "lr": 1e-2,
    "precision_min": 0.9,
    "precision_max": 1.0,
}


def _kb() -> AttackKnowledgeBase:
    techniques = {f"T{i}": Technique(id=f"T{i}", name=f"Technique {i}") for i in range(1, 13)}
    techniques["T1.001"] = Technique(id="T1.001", name="Sub", is_subtechnique=True, parent_id="T1")
    groups = {
        "G1": Group(
            id="G1",
            name="Alpha",
            description="Alpha is a Chinese espionage group.",
            technique_ids=["T1.001", "T2", "T3"],
            software_ids=["S1"],
        ),
        "G2": Group(
            id="G2",
            name="Bravo",
            description="Bravo.",
            technique_ids=["T4", "T5", "T6"],
            software_ids=["S2"],
        ),
        "G3": Group(
            id="G3", name="Charlie", description="Charlie.", technique_ids=["T7", "T8", "T9"]
        ),
        "G4": Group(
            id="G4", name="Delta", description="Delta.", technique_ids=["T10", "T11", "T12"]
        ),
        "G5": Group(id="G5", name="Echo", description="Echo.", technique_ids=["T2"]),
    }
    software = {
        "S1": Software(id="S1", name="Tool", technique_ids=["T12"], group_ids=["G1"]),
        "S2": Software(id="S2", name="Implant", technique_ids=[], group_ids=["G2"]),
    }
    kb = AttackKnowledgeBase(version="t1", techniques=techniques, groups=groups, software=software)
    kb.recompute_statistics()
    return kb


def _fit(**overrides):
    return fit_group_mlp(_kb(), params={**SMALL, **overrides}, device="cpu")


def test_training_cases_layout_and_size_independent_of_profile():
    kb = _kb()
    layout = MLPFeatures.from_kb(kb)
    space = layout.space
    assert layout.software == ("S1", "S2") and layout.size == space.size + 2
    features, groups = training_cases(kb, layout, {**SMALL, "samples_per_group": 30})
    assert features.shape == (4 * 30, layout.size)  # G5 documents one technique: skipped
    assert set(groups) == {"G1", "G2", "G3", "G4"}
    n = len(space.techniques)
    # exactly one motivation block entry set (a value or "unknown"), same for targets
    assert (features[:, n : n + len(space.motivations) + 1].sum(1) >= 1).all()
    assert (features[:, n + len(space.motivations) + 1 : space.size].sum(1) == 1).all()
    # a sub-technique always sets its parent
    sub, parent = space.techniques.index("T1.001"), space.techniques.index("T1")
    assert (features[features[:, sub] == 1, parent] == 1).all()
    # true techniques are capped by size_max (+ parents); precision >= 0.9 adds at most one
    assert features[:, :n].sum(1).max() <= 3 + 1 + 1


def test_software_techniques_and_names_enter_the_group_cases():
    kb = _kb()
    assert group_tools(kb)["G1"] == [("S1", ("T12",))]
    assert group_tools(kb)["G2"] == [("S2", ())]
    layout = MLPFeatures.from_kb(kb)
    space = layout.space
    s1, s2 = space.size, space.size + 1
    features, groups = training_cases(
        kb,
        layout,
        {
            **SMALL,
            "software_rate": 1.0,
            "software_keep": 1.0,
            "size_max": 4,
            "software_named": 1.0,
            "software_false_rate": 0.0,
        },
    )
    labels = np.asarray(groups)
    assert features[labels == "G1"][:, space.techniques.index("T12")].any()
    assert features[labels == "G1"][:, s1].all() and features[labels == "G2"][:, s2].all()
    assert not features[labels == "G3"][:, space.size :].any()  # no software, no false name
    none, groups = training_cases(
        kb, layout, {**SMALL, "software_rate": 0.0, "software_false_rate": 0.0}
    )
    assert not none[np.asarray(groups) == "G1"][:, space.techniques.index("T12")].any()
    assert not none[:, space.size :].any()
    # a false name is another group's software, never the group's own
    noisy, groups = training_cases(
        kb, layout, {**SMALL, "software_rate": 0.0, "software_false_rate": 1.0}
    )
    labels = np.asarray(groups)
    assert not noisy[labels == "G1"][:, s1].any() and noisy[labels == "G3"][:, space.size :].any()


def test_fit_predicts_the_documented_group():
    model = _fit()
    assert model.group_ids == ("G1", "G2", "G3", "G4")
    for group, techniques in {"G2": ["T4", "T5", "T6"], "G4": ["T10", "T11"]}.items():
        probabilities = model.group_probabilities(model.features.vector(techniques))
        assert max(probabilities, key=probabilities.get) == group
        assert abs(sum(probabilities.values()) - 1.0) < 1e-5


def test_prior_shifts_towards_prominent_groups():
    kb = _kb()
    kb.groups["G3"].external_references = [{"source_name": f"r{i}"} for i in range(50)]
    kb.recompute_statistics()
    prior = prominence_log_prior(kb, ("G1", "G2", "G3", "G4"))
    assert prior.argmax() == 2 and abs(float(prior.mean())) < 1e-6
    model = fit_group_mlp(kb, params=SMALL, device="cpu")
    empty = model.features.vector([])
    flat = model.predict(empty, prior_power=0.0)[0]
    tilted = model.predict(empty, prior_power=3.0)[0]
    assert tilted[2] > flat[2]


def test_save_load_round_trip_and_fail_closed(tmp_path):
    model = _fit(prior_power=1.5)
    path = model.save(tmp_path / "mlp.npz")
    loaded = GroupMLP.load(path, device="cpu")
    assert loaded is not None and loaded.fingerprint == model.fingerprint
    assert loaded.prior_power == 1.5
    vector = model.features.vector(["T7", "T8"], ["S1"])
    np.testing.assert_allclose(loaded.predict(vector), model.predict(vector), atol=1e-6)
    ok, status = load_group_mlp(path, _kb(), device="cpu")
    assert ok is not None and status == "active (cpu)"

    assert load_group_mlp(tmp_path / "missing.npz", _kb())[0] is None
    other = _kb()
    other.version = "t2"
    model_other, status = load_group_mlp(path, other, device="cpu")
    assert model_other is None and "incompatible" in status

    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["w1"] = arrays["w1"] + 1.0
    np.savez_compressed(tmp_path / "tampered.npz", **arrays)
    tampered, status = load_group_mlp(tmp_path / "tampered.npz", _kb(), device="cpu")
    assert tampered is None and "fingerprint" in status

    manifest = json.loads(str(arrays["manifest"]))
    manifest["kind"] = "profile_forest"
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(tmp_path / "kind.npz", **arrays)
    assert "not a group MLP" in load_group_mlp(tmp_path / "kind.npz", _kb())[1]


def test_named_software_points_to_its_group_and_members_average():
    model = _fit(members=2, software_named=1.0, software_rate=1.0, software_false_rate=0.0)
    assert model.members == ["m0", "m1"]
    probabilities = model.group_probabilities(model.features.vector([], ["S2"]))
    assert max(probabilities, key=probabilities.get) == "G2"
    context = SimpleNamespace(
        observed_techniques={"T4": 1.0}, software_ids={"S2"}, strategic_context=None
    )
    vector = model.context_vector(context)
    assert vector[model.space.size + 1] == 1 and vector.shape == (model.features.size,)
    single = GroupMLP(
        features=model.features,
        group_ids=model.group_ids,
        weights={k: v for k, v in model.weights.items() if k.startswith("m0.")},
        hidden=model.hidden,
        device="cpu",
    )
    both = model.logits(vector)
    assert not np.allclose(both, single.logits(vector))
    with pytest.raises(ValueError, match="features"):
        model.logits(np.zeros(model.features.size - 1))


def test_classification_metrics():
    truth = ["A", "A", "B", "C"]
    ranked = [["A", "B"], ["B", "A"], ["B", "A"], ["A", "C"]]
    metrics = classification_metrics(truth, ranked)
    assert metrics["top1"] == 0.5 and metrics["top5"] == 1.0
    assert metrics["mrr"] == round((1 + 0.5 + 1 + 0.5) / 4, 4)
    # A: tp1 fp1 fn1 -> F1 .5 ; B: tp1 fp1 fn0 -> .6667 ; C: tp0 -> 0
    assert metrics["per_class"]["B"]["f1"] == 0.6667
    assert metrics["macro_f1"] == round((0.5 + 2 / 3 + 0.0) / 3, 4)
    assert classification_metrics([], []) == {"cases": 0}


def test_real_cases_are_learned_and_recorded():
    kb = _kb()
    probe = MLPFeatures.from_kb(kb).vector(["T1", "T4", "T7"])  # one technique from three groups
    plain = fit_group_mlp(kb, params=SMALL, device="cpu")
    assert plain.metadata["real_cases"] == 0
    taught = fit_group_mlp(
        kb,
        params={**SMALL, "real_weight": 40},
        device="cpu",
        real_cases=(np.stack([probe]), ["G4"]),
    )
    assert taught.metadata["real_cases"] == 1
    assert taught.group_ids == plain.group_ids
    assert taught.predict(probe)[0][3] > plain.predict(probe)[0][3]
