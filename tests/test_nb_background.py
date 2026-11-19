"""The Naive Bayes group-background correction (apt_classifier/nb_background.py)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.classifiers.base import ClassifierContext
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier
from apt_classifier.classifiers.registry import default_classifiers
from apt_classifier.config import Settings
from apt_classifier.nb_background import (
    NAIVE_BAYES_NAME,
    BackgroundNaiveBayesClassifier,
    NaiveBayesBackground,
    case_z,
    fit_background,
    load_nb_background_for,
    nb_background_bound_fingerprint,
    with_nb_background,
)

SETTINGS = SimpleNamespace(nb_alpha=0.6, nb_beta=0.05, negative_evidence_weight=0.3)


def _kb() -> AttackKnowledgeBase:
    techniques = {f"T{n}": Technique(id=f"T{n}", name=f"Technique {n}") for n in range(1, 15)}
    groups = {
        # a long technique list that covers part of every report
        "G1": Group(id="G1", name="Big", technique_ids=[f"T{n}" for n in range(1, 11)]),
        "G2": Group(id="G2", name="Small", technique_ids=["T1", "T2", "T11"]),
        "G3": Group(id="G3", name="Mid", technique_ids=["T3", "T4", "T12"]),
        "G4": Group(id="G4", name="Other", technique_ids=["T5", "T6", "T13", "T14"]),
    }
    kb = AttackKnowledgeBase(techniques=techniques, groups=groups, version="20.0")
    kb.recompute_statistics()
    return kb


def _context(kb, techniques, eligible=None) -> ClassifierContext:
    return ClassifierContext(
        kb=kb,
        observed_techniques={technique: 0.9 for technique in techniques},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
        eligible_group_ids=eligible,
    )


# reports of the three focused groups, each with some of the big group's techniques mixed in
REFERENCE = [
    ["T1", "T2", "T11", "T7"],
    ["T1", "T11", "T8"],
    ["T2", "T11", "T9", "T10"],
    ["T3", "T4", "T12", "T7"],
    ["T3", "T12", "T8"],
    ["T4", "T12", "T9", "T1"],
    ["T5", "T6", "T13", "T10"],
    ["T5", "T13", "T14", "T2"],
    ["T6", "T14", "T13", "T3"],
] * 3


def _background(kb, **kwargs) -> NaiveBayesBackground:
    member = BernoulliNaiveBayesClassifier()
    contexts = [_context(kb, case) for case in REFERENCE]
    return fit_background(kb, member, contexts, SETTINGS, min_reference_cases=5, **kwargs)


def test_case_z_and_the_reference_correction():
    assert case_z({}) == {}
    assert case_z({"A": 1.0, "B": 1.0}) == {"A": 0.0, "B": 0.0}
    z = case_z({"A": 3.0, "B": 1.0})
    assert z == {"A": 1.0, "B": -1.0}
    background = NaiveBayesBackground(
        attack_version="20.0",
        group_profiles={},
        parent_digest="",
        params={},
        groups={"A": (0.5, 0.25, 30), "B": (-1.0, 1.0, 3)},
        min_reference_cases=20,
    )
    corrected = background.correct({"A": 3.0, "B": 1.0})
    assert corrected["A"] == pytest.approx((1.0 - 0.5) / 0.25)
    # too few reference cases: the case z-score stands
    assert corrected["B"] == pytest.approx(-1.0)


def test_the_big_profile_stops_leading_a_focused_report():
    kb = _kb()
    member = BernoulliNaiveBayesClassifier()
    report = _context(kb, ["T1", "T2", "T11", "T5", "T6"])
    plain = member.classify(report)
    assert plain.entries[0].group_id == "G1"
    corrected = BackgroundNaiveBayesClassifier(member, _background(kb)).classify(report)
    assert corrected.classifier == NAIVE_BAYES_NAME
    assert corrected.entries[0].group_id == "G2"
    assert {entry.group_id for entry in corrected.entries} == {e.group_id for e in plain.entries}


def test_artifact_round_trip_and_the_loader_fails_closed(tmp_path):
    kb = _kb()
    background = _background(kb, sources=[{"path": "design41.json", "cases": 27}])
    path = background.save(tmp_path / "nb.json")
    assert NaiveBayesBackground.load(path) == background
    settings = SimpleNamespace(
        **vars(SETTINGS), naive_bayes_background_enabled=False, naive_bayes_background_path=path
    )
    assert load_nb_background_for(settings, kb) == (None, "disabled")
    settings.naive_bayes_background_enabled = True
    model, status = load_nb_background_for(settings, kb)
    assert status == "active" and model == background
    # a removed-actor fold's knowledge base (one group deleted) still loads
    reduced = copy.deepcopy(kb)
    del reduced.groups["G4"]
    reduced.recompute_statistics()
    assert load_nb_background_for(settings, reduced)[1] == "active"
    # a changed technique list, another ATT&CK version or other parameters refuse
    changed = copy.deepcopy(kb)
    changed.groups["G2"].technique_ids.append("T14")
    assert "technique lists changed" in load_nb_background_for(settings, changed)[1]
    other = copy.deepcopy(kb)
    other.version = "21.0"
    assert "ATT&CK 20.0" in load_nb_background_for(settings, other)[1]
    assert (
        "parameters"
        in load_nb_background_for(SimpleNamespace(**{**vars(settings), "nb_alpha": 0.7}), kb)[1]
    )
    # an edited artifact fails its fingerprint; a missing one is reported
    payload = json.loads(path.read_text())
    payload["groups"]["G1"][0] += 1.0
    path.write_text(json.dumps(payload))
    assert "invalid artifact" in load_nb_background_for(settings, kb)[1]
    settings.naive_bayes_background_path = tmp_path / "missing.json"
    assert "not found" in load_nb_background_for(settings, kb)[1]


def test_only_the_naive_bayes_member_is_wrapped_and_the_fingerprint_binds_the_reference():
    kb = _kb()
    members = default_classifiers()
    assert with_nb_background(members, None) is members
    background = _background(kb)
    wrapped = with_nb_background(members, background)
    assert [m.name for m in wrapped] == [m.name for m in members]
    for before, after in zip(members, wrapped, strict=True):
        if before.name == NAIVE_BAYES_NAME:
            assert isinstance(after, BackgroundNaiveBayesClassifier) and after.member is before
        else:
            assert after is before
    bound = nb_background_bound_fingerprint("base", background)
    assert bound != "base" and bound != nb_background_bound_fingerprint("other", background)
    shifted = NaiveBayesBackground(
        **{**{f: getattr(background, f) for f in background.__dataclass_fields__}, "groups": {}}
    )
    assert nb_background_bound_fingerprint("base", shifted) != bound


def test_off_by_default():
    assert Settings.model_fields["naive_bayes_background_enabled"].default is False


def test_fit_script_refuses_report_sets_and_evaluation_seeds(tmp_path):
    from scripts import fit_nb_background as script

    for benchmark, seed in (("attribution_cti_hal_reports", 7), ("attribution_text_heldout", 109)):
        path = tmp_path / f"{benchmark}_{seed}.json"
        path.write_text(json.dumps({"benchmark": benchmark, "parameters": {"seed": seed}}))
        with pytest.raises(SystemExit):
            script._load(path, (script.FOLD_BENCHMARK,), script.DESIGN_SEEDS)  # noqa: SLF001
