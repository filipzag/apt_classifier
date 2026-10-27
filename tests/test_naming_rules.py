"""Independent-inputs naming rules (prereg attribution_independent_inputs_2026-10-27): the
technique list counts as one evidence family, a single name must beat the combiner's null option,
and the offline replay applies both exactly."""

from __future__ import annotations

from apt_classifier.attribution import (
    AttributionStage,
    assess_attribution,
    supporting_families,
)
from apt_classifier.config import Settings
from tests.test_attribution import (
    _candidate,
    _cluster_thresholds,
    _context,
    _fusion,
    _fusion_entry,
    _separation_kb,
)

EXCLUSIVE = {"T1": 1.0, "T2": 1.0}  # T1 is Alpha's alone: the leader clears the separation gate
FAMILIES = {
    "bernoulli_naive_bayes": "technique_overlap",
    "campaign_profile": "campaign",
    "source_country": "origin",
    "software_tools": "software",
    "text_group": "learned_profile",
}
TECHNIQUES_ONLY = {"bernoulli_naive_bayes": 1, "source_country": 1, "campaign_profile": 2}


def _decide(ranks, *, combiner=None, **overrides):
    fusion = _fusion(
        [
            _fusion_entry("G1", 70.0, rank_std=1.0, classifier_ranks=ranks),
            _fusion_entry("G2", 60.0, rank=2, rank_std=1.0, classifier_ranks={"x": 2}),
        ]
    )
    return assess_attribution(
        _separation_kb(),
        fusion,
        _context(techniques=EXCLUSIVE),
        thresholds=_cluster_thresholds(**overrides),
        classifier_families=FAMILIES,
        combiner=combiner,
    )


def test_technique_list_readings_count_as_one_family():
    assert supporting_families(TECHNIQUES_ONLY, FAMILIES) == {
        "technique_overlap",
        "origin",
        "campaign",
    }
    assert supporting_families(TECHNIQUES_ONLY, FAMILIES, technique_list_one_family=True) == {
        "technique_list"
    }
    # the text members never count, and a rank past the support limit is no support
    assert supporting_families({"text_group": 1, "software_tools": 11}, FAMILIES) == set()


def test_a_name_on_the_technique_list_alone_is_held_and_a_second_input_names_it():
    named = _decide(TECHNIQUES_ONLY)
    assert named.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    held = _decide(TECHNIQUES_ONLY, technique_list_one_family=True)
    assert held.stage is AttributionStage.UNRESOLVED_ACTIVITY_CLUSTER
    assert held.leading_group_id == "G1"
    assert "1 independent families below minimum 2" in held.rationale
    with_software = _decide(
        {**TECHNIQUES_ONLY, "software_tools": 1}, technique_list_one_family=True
    )
    assert with_software.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def test_a_single_name_must_beat_the_null_option():
    ranks = {"bernoulli_naive_bayes": 1, "software_tools": 1}
    weak = {"leader_probability": 0.36, "null_probability": 0.62}
    assert _decide(ranks, combiner=weak).stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    held = _decide(ranks, combiner=weak, hypothesis_above_null=True)
    assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "combiner probability 0.36 not above its none-of-these probability 0.62" in (
        held.rationale
    )
    strong = {"leader_probability": 0.60, "null_probability": 0.30}
    named = _decide(ranks, combiner=strong, hypothesis_above_null=True)
    assert named.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    # without a null probability (no active combiner) the rule does not apply
    alone = {"leader_probability": 0.36}
    assert _decide(ranks, combiner=alone, hypothesis_above_null=True).stage is (
        AttributionStage.KNOWN_GROUP_HYPOTHESIS
    )
    # a held leader can still be named through its documented cluster
    clustered = _decide(
        ranks, combiner={**weak, "cluster": _candidate()}, hypothesis_above_null=True
    )
    assert clustered.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS


def test_both_rules_are_on_by_default():
    settings = Settings(_env_file=None)
    assert settings.attribution_technique_list_one_family is True
    assert settings.attribution_hypothesis_above_null is True


def _row(stage, leading, ranks, probability, null, *, truth="G1", accept=(), cluster=None):
    return {
        "group_id": truth,
        "accept": list(accept),
        "stage": stage,
        "leading": leading,
        "top3_detail": [{"group_id": leading, "classifier_ranks": ranks}],
        "combiner_leader_probability": probability,
        "combiner_null_probability": null,
        "cluster_candidate": cluster,
        "accepted_techniques": ["T1", "T2"],
        "decisive_software": {"factor": 1.0},
        "leader_guard_fired": False,
        "named_cluster": [],
    }


def test_replay_applies_both_rules_and_counts_what_they_held():
    from scripts.naming_rules_replay import replay

    named = "known_group_hypothesis"
    two_inputs = {"bernoulli_naive_bayes": 1, "software_tools": 1}
    artifact = {
        "fingerprint_inputs": {"thresholds": _cluster_thresholds().__dict__},
        "known": [
            _row(named, "G1", two_inputs, 0.9, 0.05),
            _row(named, "G1", TECHNIQUES_ONLY, 0.95, 0.01),  # correct, technique list alone
            _row(named, "G2", two_inputs, 0.36, 0.62),  # wrong, below the null option
            _row(named, "G3", two_inputs, 0.9, 0.05, accept=["G3"]),  # an accepted alias group
            _row(named, "G1", two_inputs, 0.36, 0.62, cluster=_candidate()),  # to its cluster
        ],
        "unknown": [_row(named, "G4", TECHNIQUES_ONLY, 0.9, 0.05, truth="G9")],
    }
    off = replay(artifact, False, False)
    assert off["correct"] == 4 and len(off["wrong"]) == 1 and len(off["removed"]) == 1
    both = replay(artifact, True, True)
    assert both["correct"] == 3 and both["wrong"] == [] and both["removed"] == []
    assert len(both["held_correct"]) == 2 and len(both["held_bad"]) == 2
    assert both["cluster_fallback"] == ["known:G1 as ['G1', 'G2'] (correct)"]
    stages = {name: result["stage"] for name, result in both["predictions"].items()}
    assert stages["unknown:G9"] == "unresolved_activity_cluster"
