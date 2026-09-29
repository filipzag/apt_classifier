import pytest

from apt_classifier.classifiers.base import ClassifierContext
from apt_classifier.classifiers.campaign import CampaignClassifier
from apt_classifier.classifiers.group_text import GroupTextClassifier
from apt_classifier.classifiers.idf_overlap import IDFOverlapClassifier
from apt_classifier.classifiers.jaccard import WeightedJaccardClassifier
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier, posterior
from apt_classifier.classifiers.semantic_centroid import SemanticCentroidClassifier
from apt_classifier.classifiers.software import SoftwareClassifier
from apt_classifier.classifiers.specific_behavior import SpecificBehaviorClassifier
from apt_classifier.classifiers.strategic_context import (
    MotivationClassifier,
    StrategicContextClassifier,
)
from apt_classifier.evidence.models import StrategicContext


def context(kb):
    return ClassifierContext(
        kb=kb,
        observed_techniques={"T1": 1.0, "T2": 0.8},
        technique_observations={"T1": ("o1",), "T2": ("o2",)},
        negative_techniques={"T3"},
        semantic_profile_scores={"G1": 0.9, "G2": 0.5, "G3": 0.1},
        group_text_scores={"G1": 0.7, "G2": 0.4, "G3": 0.2},
        software_ids={"S1"},
        narrative="PowerShell and credential access",
    )


def test_all_classifiers_rank_expected_group_first(synthetic_kb):
    ctx = context(synthetic_kb)
    classifiers = (
        BernoulliNaiveBayesClassifier(),
        IDFOverlapClassifier(),
        WeightedJaccardClassifier(),
        SemanticCentroidClassifier(),
        GroupTextClassifier(),
        CampaignClassifier(),
        SoftwareClassifier(),
    )
    for classifier in classifiers:
        result = classifier.classify(ctx)
        assert result.applicable
        assert result.entries[0].group_id == "G1", classifier.name
        assert result.entries[0].rank == 1


def test_idf_overlap_omits_groups_without_positive_overlap(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T1": 1.0}
    result = IDFOverlapClassifier().classify(ctx)
    assert [entry.group_id for entry in result.entries] == ["G1"]
    assert all(entry.score > 0 for entry in result.entries)


def test_negative_evidence_changes_bayes_order(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T2": 1.0}
    ctx.negative_techniques = {"T1"}
    result = BernoulliNaiveBayesClassifier(negative_weight=1.0).classify(ctx)
    assert result.entries[0].group_id == "G2"
    probabilities = posterior(result, temperature=2.0)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_software_abstains_without_named_software(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.software_ids = set()
    result = SoftwareClassifier().classify(ctx)
    assert not result.applicable
    assert result.entries == ()


def test_campaign_classifier_only_votes_for_positive_campaign_matches(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T1": 1.0}
    result = CampaignClassifier().classify(ctx)
    assert result.applicable
    assert [entry.group_id for entry in result.entries] == ["G1"]


def test_campaign_classifier_abstains_without_overlap(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T4": 1.0}
    result = CampaignClassifier().classify(ctx)
    assert not result.applicable
    assert result.entries == ()


def test_software_classifier_abstains_when_named_software_has_no_group_link(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.software_ids = {"S-UNKNOWN"}
    result = SoftwareClassifier().classify(ctx)
    assert not result.applicable
    assert result.entries == ()


def test_strategic_context_ranks_matching_technically_supported_group(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.strategic_context = StrategicContext(
        target_geographies=["US"],
        target_industries=["defense"],
        motivations=["intelligence collection"],
        objectives=["steal intellectual property"],
        suspected_origins=["China"],
    )
    result = StrategicContextClassifier().classify(ctx)
    assert result.applicable
    assert result.entries[0].group_id == "G1"
    assert {
        "target_geographies:united states",
        "target_industries:defense",
        "objectives:intellectual property theft",
        "suspected_origins:china",
    } <= set(result.entries[0].signal_ids)
    assert all(not signal.startswith("motivations:") for signal in result.entries[0].signal_ids)


def test_motivation_alignment_survives_unmatched_other_context_fields(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T2": 1.0}
    ctx.strategic_context = StrategicContext(
        target_geographies=["Japan"],
        target_industries=["healthcare"],
        motivations=["financial gain"],
        objectives=["surveillance"],
    )
    result = MotivationClassifier().classify(ctx)
    assert result.applicable
    assert result.entries[0].group_id == "G2"
    assert result.entries[0].signal_ids == ("motivations:financial gain",)


def test_motivation_alignment_cannot_introduce_unsupported_group(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T1": 1.0}
    ctx.software_ids = set()
    ctx.strategic_context = StrategicContext(motivations=["financial gain"])
    result = MotivationClassifier().classify(ctx)
    assert not result.applicable
    assert result.entries == ()


def test_strategic_context_cannot_introduce_group_without_technical_support(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.observed_techniques = {"T1": 1.0}
    ctx.strategic_context = StrategicContext(
        target_geographies=["Europe"],
        target_industries=["financial services"],
        motivations=["financial gain"],
        objectives=["ransomware extortion"],
    )
    result = StrategicContextClassifier().classify(ctx)
    assert not result.applicable
    assert result.entries == ()


def test_specific_behavior_classifier_uses_actor_relationship_matches(synthetic_kb):
    ctx = context(synthetic_kb)
    ctx.specific_behavior_scores = {"G1": 0.62, "G2": 0.91}
    ctx.specific_behavior_signals = {
        "G1": ("G1:actor-behavior:T1:0000",),
        "G2": ("G2:actor-behavior:T2:0000",),
    }
    result = SpecificBehaviorClassifier().classify(ctx)
    assert result.applicable
    assert result.entries[0].group_id == "G2"
    assert result.entries[0].signal_ids == ("G2:actor-behavior:T2:0000",)


def test_hierarchy_ancestor_match_supports_group_with_parent_only(synthetic_kb):
    """Observed sub-technique should support a group that documents only the parent."""
    from types import SimpleNamespace

    kb = synthetic_kb
    kb.techniques["T1"] = SimpleNamespace(
        id="T1", name="One", tactics=("execution",), platforms=("Windows",), parent_id="T0"
    )
    kb.techniques["T0"] = SimpleNamespace(
        id="T0", name="Parent", tactics=("execution",), platforms=("Windows",)
    )
    kb.group_techniques["G3"] = {"T0", "T3", "T4"}
    ctx = context(kb)
    ctx.observed_techniques = {"T1": 1.0}
    ctx.technique_observations = {"T1": ("o1",)}
    result = IDFOverlapClassifier().classify(ctx)
    assert "G3" in result.by_group()
    entry = result.by_group()["G3"]
    assert entry.score > 0
    assert any(c.relationship == "ancestor" for c in entry.matched_techniques)


def test_hierarchy_exact_policy_disables_ancestor_matching(synthetic_kb):
    """Under exact_only policy, sub-technique should not match parent-only group."""
    from types import SimpleNamespace

    kb = synthetic_kb
    kb.techniques["T1"] = SimpleNamespace(
        id="T1", name="One", tactics=("execution",), platforms=("Windows",), parent_id="T0"
    )
    kb.techniques["T0"] = SimpleNamespace(
        id="T0", name="Parent", tactics=("execution",), platforms=("Windows",)
    )
    kb.group_techniques["G3"] = {"T0", "T3", "T4"}
    ctx = context(kb)
    ctx.observed_techniques = {"T1": 1.0}
    ctx.technique_observations = {"T1": ("o1",)}
    ctx.parent_policy = "exact_only"
    result = IDFOverlapClassifier().classify(ctx)
    assert "G3" not in result.by_group()


def test_hierarchy_non_duplicative_single_contribution(synthetic_kb):
    """A sub-technique and its parent both observed should not double-count for one group."""
    from types import SimpleNamespace

    kb = synthetic_kb
    kb.techniques["T1"] = SimpleNamespace(
        id="T1", name="One", tactics=("execution",), platforms=("Windows",), parent_id="T0"
    )
    kb.techniques["T0"] = SimpleNamespace(
        id="T0", name="Parent", tactics=("execution",), platforms=("Windows",)
    )
    kb.group_techniques["G1"] = {"T0", "T1", "T2"}
    ctx = context(kb)
    ctx.observed_techniques = {"T1": 1.0, "T0": 1.0}
    ctx.technique_observations = {"T1": ("o1",), "T0": ("o2",)}
    result = IDFOverlapClassifier().classify(ctx)
    entry = result.by_group()["G1"]
    # Both techniques contribute, but each only once (exact match for both)
    assert len(entry.matched_techniques) == 2
    relationships = {c.relationship for c in entry.matched_techniques}
    assert relationships == {"exact"}


def test_software_classifier_weights_tools_by_rarity():
    from types import SimpleNamespace

    from apt_classifier.classifiers.base import ClassifierContext
    from apt_classifier.classifiers.software import SoftwareClassifier

    groups = {f"G{i}": SimpleNamespace(id=f"G{i}") for i in range(1, 11)}
    # S_common is documented for 9 of 10 groups, S_rare for one.
    software_groups = {"S_common": [f"G{i}" for i in range(1, 10)], "S_rare": ["G1"]}
    group_software = {gid: [] for gid in groups}
    for sid, gids in software_groups.items():
        for gid in gids:
            group_software[gid].append(sid)
    kb = SimpleNamespace(
        groups=groups, software_groups=software_groups, group_software=group_software, techniques={}
    )
    context = ClassifierContext(
        kb=kb,
        observed_techniques={},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids={"S_common", "S_rare"},
        narrative="",
    )
    weighted = {e.group_id: e.score for e in SoftwareClassifier().classify(context).entries}
    plain = {
        e.group_id: e.score for e in SoftwareClassifier(use_idf=False).classify(context).entries
    }
    # G1 links both tools -> full score either way; G2 links only the commodity tool.
    assert weighted["G1"] == pytest.approx(1.0) and plain["G1"] == pytest.approx(1.0)
    assert plain["G2"] == pytest.approx(0.5)
    assert weighted["G2"] < 0.35  # commodity tool alone is worth much less than half
    assert weighted["G2"] > 0.0


def test_software_classifier_caps_admin_utilities_but_not_malware():
    from types import SimpleNamespace

    from apt_classifier.classifiers.base import ClassifierContext
    from apt_classifier.classifiers.software import SoftwareClassifier

    groups = {f"G{i}": SimpleNamespace(id=f"G{i}") for i in range(1, 41)}
    # one bespoke implant (1 group) and two sparsely documented admin utilities (4 groups each)
    software_groups = {
        "S_implant": ["G1"],
        "S_arp": ["G2", "G3", "G4", "G5"],
        "S_cmd": ["G2", "G6", "G7", "G8"],
    }
    software = {
        "S_implant": SimpleNamespace(software_type="malware"),
        "S_arp": SimpleNamespace(software_type="tool"),
        "S_cmd": SimpleNamespace(software_type="tool"),
    }
    group_software = {gid: [] for gid in groups}
    for sid, gids in software_groups.items():
        for gid in gids:
            group_software[gid].append(sid)
    kb = SimpleNamespace(
        groups=groups,
        software_groups=software_groups,
        group_software=group_software,
        software=software,
        techniques={},
    )
    context = ClassifierContext(
        kb=kb,
        observed_techniques={},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids={"S_implant", "S_arp", "S_cmd"},
        narrative="",
    )
    capped = {e.group_id: e.rank for e in SoftwareClassifier().classify(context).entries}
    uncapped = {
        e.group_id: e.rank
        for e in SoftwareClassifier(tool_weight_cap=float("inf")).classify(context).entries
    }
    # without the cap the two utilities outweigh the implant; with it the implant decides
    assert uncapped["G2"] == 1 and uncapped["G1"] == 2
    assert capped["G1"] == 1 and capped["G2"] == 2


def test_decisive_software_factor_triggers_on_bespoke_malware_only():
    from types import SimpleNamespace

    from apt_classifier.classifiers.software import decisive_software_factor

    kb = SimpleNamespace(
        software={
            "S_bespoke": SimpleNamespace(software_type="malware"),
            "S_shared": SimpleNamespace(software_type="malware"),
            "S_common": SimpleNamespace(software_type="malware"),
            "S_tool": SimpleNamespace(software_type="tool"),
        },
        software_groups={
            "S_bespoke": ["G1"],
            "S_shared": ["G1", "G2", "G3"],
            "S_common": [f"G{i}" for i in range(1, 30)],
            "S_tool": ["G1"],
        },
    )
    assert decisive_software_factor(kb, {"S_bespoke", "S_tool"}, boost=2.0) == (3.0, ("S_bespoke",))
    assert decisive_software_factor(kb, {"S_shared"}, boost=2.0) == (2.0, ("S_shared",))
    assert decisive_software_factor(kb, {"S_common", "S_tool"}, boost=2.0) == (1.0, ())
    assert decisive_software_factor(kb, {"S_bespoke"}, boost=0.0) == (1.0, ())
    assert decisive_software_factor(kb, set(), boost=2.0) == (1.0, ())


def test_soft_techniques_feed_ranking_but_not_the_accepted_set(synthetic_kb):
    from apt_classifier.classifiers.base import ClassifierContext, ranking_techniques
    from apt_classifier.classifiers.idf_overlap import IDFOverlapClassifier
    from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier

    kb = synthetic_kb
    group_techniques = {gid: set(tids) for gid, tids in kb.group_techniques.items()}
    # pick a technique documented for exactly one group as the soft candidate
    only = next(
        (tid, gid)
        for gid, tids in group_techniques.items()
        for tid in tids
        if sum(tid in other for other in group_techniques.values()) == 1
    )
    soft_tid, soft_gid = only
    context = ClassifierContext(
        kb=kb,
        observed_techniques={},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
        soft_techniques={soft_tid: 0.4},
        soft_technique_observations={soft_tid: ("obs-1",)},
    )
    assert ranking_techniques(context) == {soft_tid: 0.4}
    idf = IDFOverlapClassifier().classify(context)
    assert idf.applicable and idf.entries[0].group_id == soft_gid
    contribution = idf.entries[0].matched_techniques[0]
    assert contribution.technique_id == soft_tid and contribution.observation_ids == ("obs-1",)
    nb = BernoulliNaiveBayesClassifier().classify(context)
    assert nb.applicable and nb.entries[0].group_id == soft_gid
    # an accepted technique overrides a soft one with the same id
    context.observed_techniques = {soft_tid: 1.0}
    assert ranking_techniques(context)[soft_tid] == 1.0
    assert nb.input_count == 0  # input_count still counts accepted techniques only


def test_software_rarity_weights_match_the_classifier_and_cap_tools():
    from types import SimpleNamespace

    from apt_classifier.classifiers.base import ClassifierContext
    from apt_classifier.classifiers.software import SoftwareClassifier, software_rarity_weights

    groups = {f"G{i}": SimpleNamespace(id=f"G{i}") for i in range(1, 11)}
    kb = SimpleNamespace(
        groups=groups,
        software_groups={"S_mal": ["G1"], "S_tool": ["G1", "G2"]},
        group_software={"G1": ["S_mal", "S_tool"], "G2": ["S_tool"]},
        software={
            "S_mal": SimpleNamespace(software_type="malware"),
            "S_tool": SimpleNamespace(software_type="tool"),
        },
        techniques={},
    )
    weights = software_rarity_weights(kb, {"S_mal", "S_tool"}, tool_weight_cap=1.5)
    assert weights["S_tool"] == 1.5
    assert weights["S_mal"] > 2.0
    context = ClassifierContext(
        kb=kb,
        observed_techniques={},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids={"S_mal", "S_tool"},
    )
    assert SoftwareClassifier(tool_weight_cap=1.5)._weights(context) == weights
