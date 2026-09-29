from types import SimpleNamespace

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult, build_result
from apt_classifier.fusion import kendall_tau, reciprocal_rank_fusion


def result(name, order, applicable=True):
    entries = tuple(
        ClassifierEvidence(name, group_id, float(len(order) - index), index + 1)
        for index, group_id in enumerate(order)
    )
    return ClassifierResult(name, applicable, entries if applicable else ())


def test_weighted_rrf_matches_hand_calculation():
    first = result("first", ["G1", "G2", "G3"])
    second = result("second", ["G2", "G1", "G3"])
    fused = reciprocal_rank_fusion((first, second), {"first": 2.0, "second": 1.0}, k=60)
    assert fused.entries[0].group_id == "G1"
    assert abs(fused.entries[0].score - (2 / 61 + 1 / 62)) < 1e-12


def test_abstention_does_not_shift_ranking():
    first = result("first", ["G1", "G2", "G3"])
    baseline = reciprocal_rank_fusion((first,))
    with_abstention = reciprocal_rank_fusion((first, result("empty", [], False)))
    assert baseline.to_dict() == with_abstention.to_dict()


def test_kendall_tau_reverse_order_is_negative_one():
    first = {"G1": 1, "G2": 2, "G3": 3}
    second = {"G1": 3, "G2": 2, "G3": 1}
    assert kendall_tau(first, second) == -1.0


def test_eligible_groups_exclude_context_only_candidate():
    technical = result("technical", ["G1", "G2"])
    context = result("strategic_context", ["G3", "G1"])
    fused = reciprocal_rank_fusion((technical, context), eligible_group_ids={"G1", "G2"})
    assert [entry.group_id for entry in fused.entries] == ["G1", "G2"]
    assert "G3" not in fused.by_group()


def test_eligibility_filter_recomputes_classifier_ranks():
    technical = result("technical", ["G0", "G1", "G2"])
    fused = reciprocal_rank_fusion((technical,), eligible_group_ids={"G1", "G2"})
    assert [entry.group_id for entry in fused.entries] == ["G1", "G2"]
    assert fused.entries[0].classifier_ranks == {"technical": 1}
    assert abs(fused.entries[0].score - 1 / 61) < 1e-12


def test_equal_classifier_scores_receive_equal_ranks():
    context = SimpleNamespace(
        kb=SimpleNamespace(groups={"G1": object(), "G2": object()}, group_techniques={}),
        observed_techniques={},
    )
    output = build_result("equal", context, {"G1": 0.5, "G2": 0.5}, applicable=True)
    assert [entry.rank for entry in output.entries] == [1, 1]


def test_classifier_family_budget_prevents_correlated_technique_double_counting():
    first_technique = result("technique_a", ["G1", "G2"])
    second_technique = result("technique_b", ["G1", "G2"])
    specific_behavior = result("specific_behavior", ["G2", "G1"])
    fused = reciprocal_rank_fusion(
        (first_technique, second_technique, specific_behavior),
        classifier_families={
            "technique_a": "technique_overlap",
            "technique_b": "technique_overlap",
        },
        family_weights={"technique_overlap": 1.0},
        weights={"specific_behavior": 2.0},
    )
    assert fused.entries[0].group_id == "G2"
    assert fused.effective_weights == {
        "specific_behavior": 2.0,
        "technique_a": 0.5,
        "technique_b": 0.5,
    }


def test_independent_motivation_vote_can_rerank_technically_supported_candidates():
    technique_a = result("technique_a", ["G1", "G2"])
    technique_b = result("technique_b", ["G1", "G2"])
    motivation = result("motivation_alignment", ["G2"])
    fused = reciprocal_rank_fusion(
        (technique_a, technique_b, motivation),
        weights={"motivation_alignment": 1.50},
        eligible_group_ids={"G1", "G2"},
        classifier_families={
            "technique_a": "technique_overlap",
            "technique_b": "technique_overlap",
        },
        family_weights={"technique_overlap": 1.0},
    )
    assert fused.entries[0].group_id == "G2"
    assert fused.effective_weights["motivation_alignment"] == 1.50


def test_motivation_family_budget_must_not_silently_cap_configured_weight():
    motivation = result("motivation_alignment", ["G1", "G2"])
    fused = reciprocal_rank_fusion(
        (motivation,),
        weights={"motivation_alignment": 1.50},
        classifier_families={"motivation_alignment": "motivation"},
        family_weights={"motivation": 1.50},
    )
    assert fused.effective_weights["motivation_alignment"] == 1.50
    capped = reciprocal_rank_fusion(
        (motivation,),
        weights={"motivation_alignment": 1.50},
        classifier_families={"motivation_alignment": "motivation"},
        family_weights={"motivation": 1.00},
    )
    assert capped.effective_weights["motivation_alignment"] == 1.00


def test_family_budget_does_not_amplify_single_low_weight_classifier():
    description = result("description", ["G1", "G2"])
    fused = reciprocal_rank_fusion(
        (description,),
        weights={"description": 0.4},
        classifier_families={"description": "context"},
        family_weights={"context": 1.0},
    )
    assert fused.effective_weights == {"description": 0.4}


def test_actor_behavior_family_budget_caps_correlated_actor_classifiers():
    behavior_a = result("specific_behavior", ["G1", "G2"])
    behavior_b = result("semantic_centroid", ["G2", "G1"])
    fused = reciprocal_rank_fusion(
        (behavior_a, behavior_b),
        classifier_families={
            "specific_behavior": "actor_behavior",
            "semantic_centroid": "actor_behavior",
        },
        family_weights={"actor_behavior": 1.50},
    )
    assert fused.effective_weights == {
        "specific_behavior": 0.75,
        "semantic_centroid": 0.75,
    }


def test_software_family_isolated_from_technique_overlap_family():
    software = result("software_tools", ["G1", "G2"])
    technique = result("idf_tfidf_cosine", ["G2", "G1"])
    fused = reciprocal_rank_fusion(
        (software, technique),
        weights={"software_tools": 1.20, "idf_tfidf_cosine": 1.00},
        classifier_families={
            "software_tools": "software",
            "idf_tfidf_cosine": "technique_overlap",
        },
        family_weights={
            "software": 1.20,
            "technique_overlap": 1.00,
        },
    )
    assert fused.effective_weights == {
        "software_tools": 1.20,
        "idf_tfidf_cosine": 1.00,
    }


def test_tie_breaker_prefers_less_prominent_group_over_alphabetical():
    from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
    from apt_classifier.fusion import reciprocal_rank_fusion

    result = ClassifierResult(
        classifier="idf_tfidf_cosine",
        applicable=True,
        entries=(
            ClassifierEvidence(group_id="G0007", score=0.9, rank=1, classifier="idf_tfidf_cosine"),
            ClassifierEvidence(group_id="G1030", score=0.9, rank=1, classifier="idf_tfidf_cosine"),
        ),
    )
    alphabetical = reciprocal_rank_fusion((result,), {}, 60)
    assert [e.group_id for e in alphabetical.entries] == ["G0007", "G1030"]
    fused = reciprocal_rank_fusion((result,), {}, 60, tie_breaker={"G0007": 40.0, "G1030": 3.0})
    assert [e.group_id for e in fused.entries] == ["G1030", "G0007"]
