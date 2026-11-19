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


def _negative_tie_context(groups: int = 12) -> ClassifierContext:
    """Every group documents T1 and exactly one of the negated techniques N00..N11, so every
    Naive Bayes score is mathematically equal."""
    from types import SimpleNamespace

    negatives = [f"N{index:02d}" for index in range(groups)]
    kb = SimpleNamespace(
        groups={f"G{index:02d}": SimpleNamespace() for index in range(groups)},
        group_techniques={f"G{index:02d}": ["T1", negatives[index]] for index in range(groups)},
        techniques={tid: SimpleNamespace(parent_id=None) for tid in ["T1", *negatives]},
    )
    return ClassifierContext(
        kb=kb,
        observed_techniques={"T1": 1.0},
        technique_observations={"T1": ("o1",)},
        negative_techniques=set(negatives),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
    )


def test_negative_evidence_keeps_equal_groups_tied():
    """Summing the negative term over the set in iteration order split these exact ties by one
    ulp, and which groups came first followed PYTHONHASHSEED; counting keeps them tied."""
    from math import log

    classifier = BernoulliNaiveBayesClassifier()
    result = classifier.classify(_negative_tie_context())
    assert len({entry.score for entry in result.entries}) == 1
    assert {entry.rank for entry in result.entries} == {1}
    assert [entry.group_id for entry in result.entries] == [f"G{i:02d}" for i in range(12)]
    expected = (
        log(1 / 12)
        + log(classifier.alpha)
        + classifier.negative_weight * (log(1 - classifier.alpha) + 11 * log(1 - classifier.beta))
    )
    assert result.entries[0].score == pytest.approx(expected, abs=1e-12)


def test_negative_evidence_does_not_depend_on_the_hash_seed():
    from tests.child_python import run_child

    script = (
        "from tests.test_classifiers import BernoulliNaiveBayesClassifier, _negative_tie_context\n"
        "result = BernoulliNaiveBayesClassifier().classify(_negative_tie_context())\n"
        "print([(entry.group_id, entry.rank, entry.score.hex()) for entry in result.entries])\n"
    )
    outputs = {run_child(script, seed=seed) for seed in ("0", "1", "2", "3")}
    assert len(outputs) == 1


def _hash_order_context() -> ClassifierContext:
    """A synthetic KB large enough for summation order to matter, with every input the
    hardened classifiers sum over built from a set (as scripts/ensemble_benchmark.py builds
    observed_techniques), so its iteration order follows PYTHONHASHSEED."""
    import random
    from types import SimpleNamespace

    rng = random.Random(5)
    parents = [f"T1{index:03d}" for index in range(40)]
    children = [f"{parent}.00{n}" for parent in parents[:20] for n in (1, 2)]
    technique_ids = parents + children
    techniques = {tid: SimpleNamespace(parent_id=None) for tid in parents}
    techniques.update({tid: SimpleNamespace(parent_id=tid.split(".")[0]) for tid in children})
    idf = {tid: 1.0 + (index * 0.6180339887498949) % 3.7 for index, tid in enumerate(technique_ids)}
    groups = [f"G{index:02d}" for index in range(20)]
    group_techniques = {gid: rng.sample(technique_ids, 25) for gid in groups}
    software = {f"S{index}": SimpleNamespace(software_type="malware") for index in range(40)}
    software_groups = {sid: rng.sample(groups, rng.randint(1, 12)) for sid in software}
    group_software = {
        gid: [sid for sid, owners in software_groups.items() if gid in owners] for gid in groups
    }
    campaigns = {f"C{index}": rng.sample(technique_ids, 18) for index in range(6)}
    observed_ids = set(rng.sample(technique_ids, 30))
    inferred_ids = set(rng.sample(technique_ids, 15))
    kb = SimpleNamespace(
        groups={gid: SimpleNamespace() for gid in groups},
        group_techniques=group_techniques,
        techniques=techniques,
        idf=idf,
        software=software,
        software_groups=software_groups,
        group_software=group_software,
        campaign_techniques=campaigns,
        campaign_groups={cid: rng.sample(groups, 2) for cid in campaigns},
    )
    confidence = {tid: 0.35 + (index * 0.37) % 0.6 for index, tid in enumerate(technique_ids)}
    context = ClassifierContext(
        kb=kb,
        observed_techniques={tid: confidence[tid] for tid in observed_ids},
        technique_observations={tid: ("o",) for tid in observed_ids},
        negative_techniques=set(rng.sample(technique_ids, 6)),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(rng.sample(sorted(software), 20)),
    )
    context.cyberre_techniques = {tid: confidence[tid] for tid in inferred_ids}
    return context


HASH_ORDER_SCRIPT = """
import builtins, functools, sys
if sys.argv[1] == "left_fold":
    # Python 3.11's sum() of floats is a plain left fold (3.12 compensates rounding errors)
    builtin_sum = builtins.sum
    def left_fold(values, start=0):
        return functools.reduce(lambda total, value: total + value, values, start)
    builtins.sum = left_fold
from apt_classifier.classifiers.campaign import CampaignClassifier
from apt_classifier.classifiers.cyberre import CyberREClassifier
from apt_classifier.classifiers.idf_overlap import IDFOverlapClassifier
from apt_classifier.classifiers.jaccard import WeightedJaccardClassifier
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier, posterior
from apt_classifier.classifiers.software import SoftwareClassifier
from tests.test_classifiers import _hash_order_context
context = _hash_order_context()
classifiers = (
    BernoulliNaiveBayesClassifier(), IDFOverlapClassifier(), IDFOverlapClassifier(0.5),
    WeightedJaccardClassifier(), WeightedJaccardClassifier("dice"), CampaignClassifier(),
    SoftwareClassifier(), CyberREClassifier(),
)
for classifier in classifiers:
    result = classifier.classify(context)
    assert result.entries, classifier.name
    print(classifier.name, [(e.group_id, e.rank, e.score.hex()) for e in result.entries])
nb = BernoulliNaiveBayesClassifier().classify(context)
print("posterior", sorted((g, p.hex()) for g, p in posterior(nb).items()))
"""


def test_classifier_scores_do_not_depend_on_hash_or_summation_order():
    """Every float sum in these classifiers is math.fsum (exactly rounded), so the scores are
    bit-identical under any PYTHONHASHSEED and do not rely on CPython 3.12's compensated sum().
    Under a left-fold sum the pre-fsum code varied across these seeds for Naive Bayes, IDF,
    Jaccard, dice, campaign and software; under 3.12's sum() for Naive Bayes."""
    from tests.child_python import run_child

    outputs = {
        run_child(HASH_ORDER_SCRIPT, mode, seed=seed)
        for seed in ("0", "1", "2", "3")
        for mode in ("builtin", "left_fold")
    }
    assert len(outputs) == 1


def test_overlap_classifiers_tie_groups_with_the_same_terms():
    """GA and GB document different techniques whose IDF values are the same multiset in
    opposite orders, and every technique is observed: each classifier sees the same terms for
    both groups in a different order and must score them bit-identically. Under 3.12's sum()
    only the Naive Bayes case failed before fsum; the left-fold mode of the hash-order test
    above covers IDF, Jaccard and dice."""
    from types import SimpleNamespace

    values = [1e-3, 3.3, 0.1, 2.9, 0.7, 1.1]
    first = [f"T{index}" for index in range(6)]
    second = [f"T{index}" for index in range(6, 12)]
    kb = SimpleNamespace(
        groups={"GA": SimpleNamespace(), "GB": SimpleNamespace()},
        group_techniques={"GA": first, "GB": second},
        techniques={tid: SimpleNamespace(parent_id=None) for tid in first + second},
        idf=dict(zip(first, values, strict=True)) | dict(zip(second, values[::-1], strict=True)),
    )
    context = ClassifierContext(
        kb=kb,
        observed_techniques={tid: 0.9 for tid in first + second},
        technique_observations={tid: ("o",) for tid in first + second},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
    )
    for classifier in (
        BernoulliNaiveBayesClassifier(),
        IDFOverlapClassifier(),
        WeightedJaccardClassifier(),
        WeightedJaccardClassifier("dice"),
    ):
        entries = classifier.classify(context).entries
        assert len(entries) == 2 and entries[0].score == entries[1].score, classifier.name
        assert entries[0].rank == entries[1].rank == 1, classifier.name


def test_children_by_parent_is_rebuilt_for_a_new_attack_release():
    """The parent -> sub-technique map is cached per technique table: a newly loaded knowledge
    base, an ATT&CK version bump on the same object or a resized table each rebuild it (tables
    are not edited in place otherwise)."""
    from types import SimpleNamespace

    from apt_classifier.classifiers.base import children_by_parent

    def kb(version, extra=()):
        techniques = {"T1": SimpleNamespace(parent_id=None)}
        techniques.update({tid: SimpleNamespace(parent_id="T1") for tid in ("T1.001", *extra)})
        return SimpleNamespace(version=version, techniques=techniques)

    release_a = kb("19.2")
    first = children_by_parent(release_a)
    assert first == {"T1": frozenset({"T1.001"})}
    assert children_by_parent(release_a) is first  # served from the cache
    release_b = kb("20.0", extra=("T1.002",))  # a new release, loaded as a new knowledge base
    assert children_by_parent(release_b) == {"T1": frozenset({"T1.001", "T1.002"})}
    release_a.version = "19.3"  # updated in place: same table object, new version
    release_a.techniques["T1.001"].parent_id = None
    assert children_by_parent(release_a) == {}
    release_a.techniques["T2.001"] = SimpleNamespace(parent_id="T2")  # resized table
    assert children_by_parent(release_a) == {"T2": frozenset({"T2.001"})}
    assert children_by_parent(SimpleNamespace()) == {}


def test_hierarchy_helpers_use_the_documented_children():
    """The two lookups that read the cached parent -> children map: a group documenting only
    sub-techniques matches an observed parent through its lowest documented child (0.85), and
    a negated parent counts against such a group at 0.85."""
    from types import SimpleNamespace

    from apt_classifier.classifiers.base import hierarchy_matches_for, negative_hierarchy_factor

    kb = SimpleNamespace(
        version="test",
        techniques={
            "T1": SimpleNamespace(parent_id=None),
            "T1.001": SimpleNamespace(parent_id="T1"),
            "T1.002": SimpleNamespace(parent_id="T1"),
            "T2": SimpleNamespace(parent_id=None),
        },
        group_techniques={"G1": ["T1.002", "T1.001"], "G2": ["T1"], "G3": ["T2"]},
        groups={"G1": SimpleNamespace(), "G2": SimpleNamespace(), "G3": SimpleNamespace()},
    )

    def context(policy="ancestor_compatible"):
        return ClassifierContext(
            kb=kb,
            observed_techniques={"T1": 1.0, "T1.001": 1.0},
            technique_observations={},
            negative_techniques=set(),
            semantic_profile_scores={},
            group_text_scores={},
            software_ids=set(),
            parent_policy=policy,
        )

    g1 = hierarchy_matches_for(context(), "G1")
    assert (g1["T1"].group_technique_id, g1["T1"].relationship, g1["T1"].factor) == (
        "T1.001",
        "descendant",
        0.85,
    )
    assert g1["T1.001"].relationship == "exact"
    g2 = hierarchy_matches_for(context(), "G2")
    assert (g2["T1.001"].relationship, g2["T1.001"].factor) == ("ancestor", 0.95)
    assert hierarchy_matches_for(context(), "G3") == {}
    assert set(hierarchy_matches_for(context("exact_only"), "G1")) == {"T1.001"}
    assert negative_hierarchy_factor(context(), "G1", "T1") == 0.85
    assert negative_hierarchy_factor(context(), "G1", "T1.001") == 1.0
    assert negative_hierarchy_factor(context(), "G2", "T1.001") == 0.0
    assert negative_hierarchy_factor(context(), "G3", "T1") == 0.0
    assert negative_hierarchy_factor(context("exact_only"), "G1", "T1") == 0.0


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
