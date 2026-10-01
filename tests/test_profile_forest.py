"""Random-forest group and source-country models (apt_classifier/profile_forest.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique  # noqa: E402
from apt_classifier.classifiers.base import ClassifierContext  # noqa: E402
from apt_classifier.classifiers.profile_forest import (  # noqa: E402
    ProfileForestClassifier,
    SourceCountryClassifier,
)
from apt_classifier.evidence.models import StrategicContext  # noqa: E402
from apt_classifier.profile_forest import (  # noqa: E402
    FeatureSpace,
    ProfileForest,
    canonical_context,
    fit_profile_forest,
    group_profiles,
    load_profile_forest,
    training_cases,
)

SMALL = {
    "samples_per_group": 40,
    "group_trees": 12,
    "country_trees": 12,
    "group_min_samples_leaf": 1,
    "country_min_samples_leaf": 1,
}


def _kb() -> AttackKnowledgeBase:
    techniques = {f"T{i}": Technique(id=f"T{i}", name=f"Technique {i}") for i in range(1, 13)}
    techniques["T1.001"] = Technique(id="T1.001", name="Sub", is_subtechnique=True, parent_id="T1")
    groups = {
        "G1": Group(
            id="G1",
            name="Alpha",
            description="Alpha is a suspected Chinese threat group conducting espionage against Taiwan.",
            technique_ids=["T1.001", "T2", "T3"],
        ),
        "G2": Group(
            id="G2",
            name="Bravo",
            description="Bravo is a Russia-based group whose attacks against Ukraine are destructive.",
            technique_ids=["T4", "T5", "T6"],
        ),
        "G3": Group(
            id="G3",
            name="Charlie",
            description="Charlie is a financially motivated group targeting the United States.",
            technique_ids=["T7", "T8", "T9"],
        ),
        "G4": Group(
            id="G4",
            name="Delta",
            description="Delta is an Iranian-sponsored threat group that targets Israel.",
            technique_ids=["T10", "T11", "T12"],
        ),
        "G5": Group(id="G5", name="Echo", description="Echo is a group.", technique_ids=["T2"]),
        "G6": Group(
            id="G6",
            name="Foxtrot",
            description="Foxtrot is an Iranian threat group.",
            technique_ids=["T10", "T12"],
        ),
    }
    kb = AttackKnowledgeBase(version="t1", techniques=techniques, groups=groups)
    kb.recompute_statistics()
    return kb


def test_profiles_read_techniques_motivation_targets_and_origin():
    profiles = group_profiles(_kb())
    assert profiles["G1"].techniques == frozenset({"T1.001", "T1", "T2", "T3"})  # parent added
    assert profiles["G1"].country == "China" and profiles["G1"].country_class == "China"
    assert "espionage" in profiles["G1"].motivations and "taiwan" in profiles["G1"].targets
    assert profiles["G2"].country_class == "Russia" and "destruction" in profiles["G2"].motivations
    assert profiles["G3"].country_class == "unattributed"
    assert "financial gain" in profiles["G3"].motivations
    assert profiles["G4"].country_class == "Iran"


def test_feature_vector_and_training_cases():
    kb = _kb()
    space = FeatureSpace.from_kb(kb)
    row = space.vector(["T1.001"], ["espionage"], ["taiwan"])
    index = {t: i for i, t in enumerate(space.techniques)}
    assert row[index["T1.001"]] == 1 and row[index["T1"]] == 1  # a sub-technique sets its parent
    base = len(space.techniques)
    assert row[base + space.motivations.index("espionage")] == 1
    assert row[base + len(space.motivations)] == 0  # motivation known
    empty = space.vector(["T2"])
    assert empty[base + len(space.motivations)] == 1 and empty[-1] == 1  # both unknown flags
    x, groups, classes = training_cases(kb, space, {"samples_per_group": 10, "seed": 3})
    again, _, _ = training_cases(kb, space, {"samples_per_group": 10, "seed": 3})
    assert np.array_equal(x, again) and x.shape == (50, space.size)
    assert "G5" not in groups  # fewer than two documented techniques
    assert set(classes) == {"China", "Russia", "Iran", "unattributed"}


def test_fit_save_load_round_trip_and_fail_closed_loader(tmp_path):
    kb = _kb()
    model = fit_profile_forest(kb, params=SMALL, device="cpu")
    assert set(model.group_ids) == {"G1", "G2", "G3", "G4", "G6"}
    path = model.save(tmp_path / "forest.npz")
    loaded = ProfileForest.load(path, device="cpu")
    vector = model.space.vector(["T4", "T5"])
    assert loaded.fingerprint == model.fingerprint
    assert np.array_equal(loaded.predict(vector)[0], model.predict(vector)[0])
    assert (
        max(loaded.group_probabilities(vector), key=loaded.group_probabilities(vector).get) == "G2"
    )
    countries = loaded.country_probabilities(vector)
    assert max(countries, key=countries.get) == "Russia"
    settings = SimpleNamespace(
        profile_forest_enabled=True, profile_forest_path=path, profile_forest_device="cpu"
    )
    assert load_profile_forest(settings, kb)[1] == "active (cpu)"
    assert load_profile_forest(SimpleNamespace(profile_forest_enabled=False), kb)[1] == "disabled"
    missing = SimpleNamespace(**{**settings.__dict__, "profile_forest_path": tmp_path / "x.npz"})
    assert "not found" in load_profile_forest(missing, kb)[1]
    other_version = AttackKnowledgeBase(version="t2", techniques=kb.techniques, groups=kb.groups)
    assert "incompatible" in load_profile_forest(settings, other_version)[1]
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["group_leaf_prob"] = arrays["group_leaf_prob"] * 0.5
    np.savez(tmp_path / "tampered.npz", **arrays)
    tampered = SimpleNamespace(
        **{**settings.__dict__, "profile_forest_path": tmp_path / "tampered.npz"}
    )
    assert "invalid" in load_profile_forest(tampered, kb)[1]


def test_prediction_falls_back_to_the_cpu_when_the_gpu_is_full(monkeypatch):
    import torch

    import apt_classifier.profile_forest as module

    kb = _kb()
    model = fit_profile_forest(kb, params=SMALL, device="cpu")
    vector = model.space.vector(["T4", "T5"])
    expected = model.predict(vector)
    real = module.ForestPredictor

    def predictor(arrays, device):
        if str(device).startswith("cuda"):
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(arrays, device)

    monkeypatch.setattr(module, "ForestPredictor", predictor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model._group_predictor = model._country_predictor = None
    model.device = "cuda"  # loaded on a GPU that another process has since filled
    groups, countries = model.predict(vector)
    assert model.device == "cpu"
    assert np.array_equal(groups, expected[0]) and np.array_equal(countries, expected[1])


def _context(kb, techniques, strategic=None):
    context = ClassifierContext(
        kb=kb,
        observed_techniques={t: 0.9 for t in techniques},
        technique_observations={t: ("o1",) for t in techniques},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
    )
    context.strategic_context = strategic or StrategicContext()
    return context


def test_classifiers_vote_share_one_prediction_and_abstain_without_a_model(monkeypatch):
    kb = _kb()
    model = fit_profile_forest(kb, params=SMALL, device="cpu")
    calls = []
    original = model.predict
    monkeypatch.setattr(model, "predict", lambda v: calls.append(1) or original(v))
    context = _context(kb, ["T10", "T11"], StrategicContext(target_geographies=["Israel"]))
    groups = ProfileForestClassifier(model).classify(context)
    countries = SourceCountryClassifier(model).classify(context)
    assert len(calls) == 1  # both members reuse one forest prediction
    assert groups.applicable and groups.entries[0].group_id == "G4"
    assert countries.applicable and context.source_country["top"] == "Iran"
    by_group = {entry.group_id: entry for entry in countries.entries}
    # every group is scored with its class probability; groups of one class tie on rank
    assert by_group["G4"].rank == 1 and by_group["G4"].signal_ids[0].startswith(
        "source_country:Iran"
    )
    assert by_group["G6"].rank == 1 and by_group["G6"].score == by_group["G4"].score
    assert all(entry.score > 0 for entry in countries.entries)  # zero-probability classes abstain
    assert not ProfileForestClassifier(None).classify(context).applicable
    empty = _context(kb, [])
    assert not SourceCountryClassifier(model).classify(empty).applicable
    assert empty.source_country is None


def test_analyst_context_is_mapped_to_the_vocabulary():
    motivations, targets = canonical_context(
        StrategicContext(
            motivations=["Espionage", "bananas"], target_geographies=["U.S.", "Taiwan"]
        )
    )
    assert motivations == ("espionage",)
    assert targets == ("united states", "taiwan")
