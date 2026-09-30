from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.fusion import reciprocal_rank_fusion
from apt_classifier.origin import assess_origin_and_threat_type


def fusion_for(*group_ids):
    result = ClassifierResult(
        "test",
        True,
        tuple(
            ClassifierEvidence("test", group_id, float(len(group_ids) - index), index + 1)
            for index, group_id in enumerate(group_ids)
        ),
    )
    return reciprocal_rank_fusion((result,))


def test_country_and_apt_like_assessment_uses_documented_origin(synthetic_kb):
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G1"))
    assert assessment.origin.top_country == "China"
    assert assessment.origin.hypotheses[0].supporting_group_ids == ("G1",)
    assert assessment.origin.confidence == "medium"
    assert assessment.threat_type.likely_apt is True
    assert assessment.threat_type.label == "APT-like state/espionage activity"


def test_close_candidate_ranking_caps_origin_confidence(synthetic_kb):
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G1", "G2"))
    assert assessment.origin.top_country == "China"
    assert assessment.origin.confidence == "low"


def test_financially_motivated_group_is_not_forced_into_apt_label(synthetic_kb):
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G2"))
    assert assessment.origin.top_country is None
    assert assessment.threat_type.likely_apt is False
    assert assessment.threat_type.label == "Likely financially motivated criminal intrusion"


def test_target_country_is_not_mistaken_for_actor_origin(synthetic_kb):
    synthetic_kb.groups[
        "G3"
    ].description = (
        "Charlie targeted Chinese government and military organizations using destructive malware."
    )
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G3"))
    assert assessment.origin.hypotheses == ()
    assert assessment.origin.confidence == "insufficient"


def test_dprk_origin_does_not_also_match_south_korea(synthetic_kb):
    synthetic_kb.groups["G1"].description = (
        "Alpha is a Democratic People's Republic of Korea (DPRK)-based cyber espionage group. "
        "It initially targeted South Korean government agencies."
    )
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G1"))
    assert [item.country for item in assessment.origin.hypotheses] == ["North Korea"]


def test_state_group_with_financial_operations_remains_apt_like(synthetic_kb):
    synthetic_kb.groups["G1"].description = (
        "Alpha is a Chinese state-sponsored espionage group that also conducts "
        "financially motivated ransomware operations."
    )
    assessment = assess_origin_and_threat_type(synthetic_kb, fusion_for("G1"))
    assert assessment.threat_type.likely_apt is True
    assert assessment.threat_type.label == "APT-like with mixed state and criminal objectives"


def test_empty_fusion_returns_indeterminate_assessment(synthetic_kb):
    assessment = assess_origin_and_threat_type(synthetic_kb, reciprocal_rank_fusion(()))
    assert assessment.origin.top_country is None
    assert assessment.threat_type.likely_apt is None
    assert assessment.threat_type.label == "Indeterminate"


def test_self_description_and_operating_phrases_resolve_origin(synthetic_kb):
    from apt_classifier.origin import _documented_origins, documented_origins

    def origin(text):
        return sorted(_documented_origins([text], ("Charlie", "Charlie Team")))

    assert origin("Charlie is a suspected Iranian threat group active since 2013.") == ["Iran"]
    assert origin("Charlie is an Iranian-sponsored threat group.") == ["Iran"]
    assert origin("Charlie is a North Korea-aligned threat group.") == ["North Korea"]
    assert origin(
        "Charlie is a People's Republic of China (PRC)-aligned cyberespionage actor."
    ) == ["China"]
    assert origin("Charlie Team is a Türkiye-linked threat actor.") == ["Turkey"]
    assert origin("Charlie is a group believed to be operating out of Iran.") == ["Iran"]
    assert origin("Charlie is a cyber espionage group with likely Chinese origins.") == ["China"]
    assert origin("Charlie has a suspected nexus to the Iranian government.") == ["Iran"]
    # the per-KB table reads the group's own profile
    synthetic_kb.groups["G3"].description = "Charlie is a Lebanese threat group."
    assert list(documented_origins(synthetic_kb, "G3")) == ["Lebanon"]


def test_language_regions_and_other_groups_are_not_origin():
    from apt_classifier.origin import _documented_origins

    def origin(text):
        return sorted(_documented_origins([text], ("Charlie",)))

    assert origin("Charlie is a Russian-speaking cybercriminal group.") == []
    assert origin("Charlie is a suspected South American threat group.") == []  # not the US
    # the adjective describes another group, not Charlie
    assert origin("Charlie's name comes from Sidewinder, a suspected Indian threat group.") == []
    assert origin("Charlie targeted Chinese government entities and Iranian dissidents.") == []
    assert origin("Charlie is a group that targets Chinese threat actors.") == []
