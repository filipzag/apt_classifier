"""Documented-overlap clusters (apt_classifier/overlap_cluster.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.attack.related import description_links, related_groups
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.overlap_cluster import (
    LINK_DESCRIPTION,
    LINK_NESTING,
    LINK_SAME_ACTIVITY,
    LINK_SHARED_SOFTWARE,
    cluster_candidate,
    cluster_label,
    containment,
    link_kind,
    select_twin,
    twin_link,
    unit_totals,
)
from apt_classifier.separation import DiscriminationWeights, discriminate


@pytest.fixture(autouse=True)
def alpha_beta_same_activity(monkeypatch):
    """Review Beta's description link to Alpha as the same activity: the real table holds ATT&CK
    ids only, and an unreviewed description link fails closed even when the profiles nest."""
    from apt_classifier.attack import same_activity

    monkeypatch.setitem(
        same_activity.SAME_ACTIVITY,
        frozenset(("G1", "G2")),
        "ATT&CK: Beta shares tooling with Alpha",
    )


def _kb() -> AttackKnowledgeBase:
    kb = AttackKnowledgeBase(
        version="t",
        techniques={tid: Technique(id=tid, name=tid) for tid in ("T1", "T2", "T3", "T4", "T5")},
        groups={
            "G1": Group(id="G1", name="Alpha", technique_ids=["T1", "T2", "T3"]),
            # Beta's description names Alpha, and its profile lies inside Alpha's
            "G2": Group(
                id="G2",
                name="Beta",
                description="Beta shares tooling with Alpha.",
                technique_ids=["T2", "T3"],
            ),
            "G3": Group(id="G3", name="Gamma", technique_ids=["T4", "T2"]),
            "G4": Group(id="G4", name="Delta", technique_ids=["T5"]),
        },
    )
    kb.recompute_statistics()
    return kb


def _scope_kb() -> AttackKnowledgeBase:
    """Alpha leads; Epsilon only shares rare malware with it, Zeta's description names it and
    Eta shares malware and nests inside it."""
    kb = AttackKnowledgeBase(
        version="scope",
        techniques={
            tid: Technique(id=tid, name=tid)
            for tid in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
        },
        groups={
            "G1": Group(
                id="G1",
                name="Alpha",
                technique_ids=["T1", "T2", "T3", "T4"],
                software_ids=["S1", "S2"],
            ),
            "G5": Group(id="G5", name="Epsilon", technique_ids=["T1", "T5"], software_ids=["S1"]),
            "G6": Group(
                id="G6",
                name="Zeta",
                description="Zeta has been linked to Alpha by several vendors.",
                technique_ids=["T2", "T6"],
            ),
            "G7": Group(id="G7", name="Eta", technique_ids=["T3", "T4"], software_ids=["S2"]),
            "G8": Group(id="G8", name="Theta", technique_ids=["T7", "T8"]),
        },
        software={
            "S1": Software(id="S1", name="KillDisk", software_type="malware"),
            "S2": Software(id="S2", name="Backdoor", software_type="malware"),
        },
    )
    kb.recompute_statistics()
    return kb


@pytest.fixture
def zeta_same_activity(monkeypatch):
    """Review Zeta's description link to Alpha as the same activity (the real table holds ATT&CK
    ids only)."""
    from apt_classifier.attack import same_activity

    monkeypatch.setitem(
        same_activity.SAME_ACTIVITY,
        frozenset(("G1", "G6")),
        "ATT&CK: Zeta has been linked to Alpha",
    )


def test_link_kind_tells_same_activity_nesting_mentions_and_shared_malware_apart(monkeypatch):
    kb = _scope_kb()
    # an unreviewed description mention is only a mention, never the same activity
    assert link_kind(kb, "G1", "G6") == LINK_DESCRIPTION
    assert link_kind(kb, "G1", "G7") == LINK_NESTING
    assert link_kind(kb, "G1", "G5") == LINK_SHARED_SOFTWARE
    assert link_kind(kb, "G1", "G8") is None
    # description links stay symmetric and are a subset of the related-group table
    links = description_links(kb)
    assert "G1" in links["G6"] and "G6" in links["G1"]
    assert set(links["G1"]) <= set(related_groups(kb)["G1"])
    from apt_classifier.attack import same_activity

    monkeypatch.setitem(same_activity.SAME_ACTIVITY, frozenset(("G1", "G6")), "same")
    assert link_kind(kb, "G1", "G6") == LINK_SAME_ACTIVITY
    assert link_kind(kb, "G6", "G1") == LINK_SAME_ACTIVITY  # either direction
    # a reviewed "distinct" verdict also vetoes nesting profiles (APT30 / Naikon nest fully)
    monkeypatch.setitem(same_activity.NOT_SAME_ACTIVITY, frozenset(("G1", "G7")), "distinct")
    assert link_kind(kb, "G1", "G7") == LINK_SHARED_SOFTWARE


def test_twin_link_per_scope(zeta_same_activity):
    kb = _scope_kb()
    # a pair that only shares rare malware and does not separate: full only
    assert twin_link(kb, "G1", "G5", separation=0.1, scope="full")["link"] == LINK_SHARED_SOFTWARE
    assert twin_link(kb, "G1", "G5", separation=0.1, scope="documented") is None
    assert twin_link(kb, "G1", "G5", separation=0.1, scope="nested") is None
    # a reviewed same-activity link qualifies under documented and full, not nested
    same = twin_link(kb, "G1", "G6", separation=0.1, scope="documented")
    assert same["link"] == LINK_SAME_ACTIVITY
    assert same["reason"] == "ATT&CK: Zeta has been linked to Alpha"
    # the reason does not depend on the separation or the scope
    assert twin_link(kb, "G1", "G6", separation=-0.3, scope="full")["reason"] == same["reason"]
    assert twin_link(kb, "G1", "G6", separation=0.1, scope="nested") is None
    # every scope still needs nesting or a pair separation below the overlap bar
    assert twin_link(kb, "G1", "G6", separation=0.6, scope="documented") is None
    nested = twin_link(kb, "G1", "G7", separation=0.9, scope="nested")
    assert nested["nested"] and nested["link"] == LINK_NESTING
    assert "profiles nest (100%)" in nested["reason"]
    with pytest.raises(ValueError):
        twin_link(kb, "G1", "G6", separation=0.1, scope="everything")


def test_an_unreviewed_mention_is_no_twin_under_the_documented_scope():
    kb = _scope_kb()
    assert twin_link(kb, "G1", "G6", separation=0.1, scope="documented") is None
    assert twin_link(kb, "G1", "G6", separation=0.1, scope="full")["link"] == LINK_DESCRIPTION


def test_documented_scope_skips_a_heavier_shared_malware_twin(zeta_same_activity):
    kb = _scope_kb()
    # Epsilon (shared malware) carries more window mass than Zeta (same activity); both are
    # within the overlap bar of the leader, so both qualify under the full scope
    context = _context({"T1": 1.0, "T2": 0.9, "T5": 1.0, "T6": 0.5, "T7": 0.2})
    discrimination = discriminate(kb, _fusion("G1", "G5", "G6", "G8"), context, top_k=4)
    totals = {item.group_id: item.total for item in discrimination.scores}
    assert totals["G5"] > totals["G6"] > 0
    full = select_twin(kb, discrimination, scope="full")
    documented = select_twin(kb, discrimination)
    assert full is not None and full["group_id"] == "G5"
    assert documented is not None and documented["group_id"] == "G6"
    assert documented["link"] == LINK_SAME_ACTIVITY
    candidate = cluster_candidate(kb, discrimination, context, probabilities={"G1": 0.5})
    assert candidate is not None and candidate.members == ("G1", "G6")
    assert candidate.to_dict()["link"] == LINK_SAME_ACTIVITY
    # without the same-activity twin in the window the default scope finds none
    alone = discriminate(kb, _fusion("G1", "G5", "G8"), context, top_k=3)
    assert select_twin(kb, alone) is None and select_twin(kb, alone, scope="full") is not None
    with pytest.raises(ValueError):
        select_twin(kb, alone, scope="everything")


def test_the_reviewed_table_covers_every_description_link_of_the_real_knowledge_base():
    from pathlib import Path

    from apt_classifier.attack.kb import load_knowledge_base
    from apt_classifier.attack.same_activity import (
        NOT_SAME_ACTIVITY,
        REVIEWED_ATTACK_VERSION,
        SAME_ACTIVITY,
        unreviewed_description_pairs,
    )

    assert not set(SAME_ACTIVITY) & set(NOT_SAME_ACTIVITY)
    path = Path(__file__).resolve().parents[1] / "data" / "kb" / "attack_kb.json"
    if not path.exists():
        pytest.skip("ATT&CK knowledge base not ingested")
    kb = load_knowledge_base(path)
    if str(kb.version) != REVIEWED_ATTACK_VERSION:
        pytest.skip(
            f"knowledge base is ATT&CK {kb.version}, review was for {REVIEWED_ATTACK_VERSION}"
        )
    assert unreviewed_description_pairs(kb) == []
    # every reviewed same-activity pair is still description-linked, and nothing nests outside it
    reviewed = [pair for pair in SAME_ACTIVITY if pair <= set(kb.groups)]  # not test fixtures
    assert len(reviewed) == 13
    for pair in reviewed:
        first, second = sorted(pair)
        assert link_kind(kb, first, second) == LINK_SAME_ACTIVITY, pair
    # a reviewed false match is no link at all (Tonto Team / BITTER, "Operation Bitter Biscuit")
    assert link_kind(kb, "G0131", "G1002") is None
    assert link_kind(kb, "G0034", "G0082") == LINK_SHARED_SOFTWARE  # Sandworm Team / APT38
    assert link_kind(kb, "G1003", "G1031") == LINK_DESCRIPTION  # Ember Bear / Saint Bear
    assert link_kind(kb, "G0013", "G0019") == LINK_DESCRIPTION  # APT30 / Naikon nest, distinct


def _fusion(*ids: str) -> FusionResult:
    entries = tuple(
        FusionEntry(gid, 1.0 / (20 + rank), 100.0 - rank, rank, {}, 1, 1, 0.0, False)
        for rank, gid in enumerate(ids, 1)
    )
    return FusionResult(entries, ("c",), {}, 20)


def _context(techniques):
    return SimpleNamespace(
        observed_techniques=techniques,
        negative_techniques=set(),
        software_ids=set(),
        specific_behavior_matches=(),
        parent_policy="ancestor_compatible",
        decisive_software={},
    )


def test_singleton_units_equal_the_gate_discrimination():
    kb = _kb()
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    fusion = _fusion("G1", "G2", "G3", "G4")
    reference = discriminate(kb, fusion, context, top_k=4)
    totals = unit_totals(kb, [(gid,) for gid in ("G1", "G2", "G3", "G4")], context)
    assert totals == pytest.approx([item.total for item in reference.scores])


def test_evidence_the_pair_shares_is_exclusive_to_the_merged_unit():
    kb = _kb()
    context = _context({"T3": 1.0})
    single = unit_totals(kb, [("G1",), ("G2",), ("G4",)], context)
    merged = unit_totals(kb, [("G1", "G2"), ("G4",)], context)
    # T3 is documented by both Alpha and Beta: shared between them, exclusive to the pair
    assert merged[0] > single[0] and merged[0] > single[1]
    assert merged[1] == 0.0


def test_twin_is_a_related_group_that_nests_or_does_not_separate():
    kb = _kb()
    assert containment(kb, "G1", "G2") == 1.0
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    discrimination = discriminate(kb, _fusion("G1", "G2", "G3"), context, top_k=3)
    twin = select_twin(kb, discrimination)
    assert twin is not None and twin["group_id"] == "G2" and twin["nested"]
    assert twin["reason"] == "ATT&CK: Beta shares tooling with Alpha"
    # an unrelated leader has no twin
    unrelated = discriminate(kb, _fusion("G3", "G4"), context, top_k=2)
    assert select_twin(kb, unrelated) is None
    # the twin must be in the window
    assert select_twin(kb, discriminate(kb, _fusion("G1", "G3"), context, top_k=2)) is None


def test_cluster_candidate_merges_the_pair_and_sums_the_probabilities():
    kb = _kb()
    context = _context({"T1": 0.9, "T2": 1.0, "T3": 0.7, "T4": 0.8})
    discrimination = discriminate(kb, _fusion("G1", "G2", "G3"), context, top_k=3)
    candidate = cluster_candidate(
        kb, discrimination, context, probabilities={"G1": 0.6, "G2": 0.25, "G3": 0.1}
    )
    assert candidate is not None and candidate.members == ("G1", "G2")
    assert candidate.probability == pytest.approx(0.85)
    assert candidate.runner_up_id == "G3"
    expected = unit_totals(kb, [("G1", "G2"), ("G3",)], context, DiscriminationWeights())
    assert candidate.discriminative == pytest.approx(expected[0])
    assert -1.0 <= candidate.separation <= 1.0
    record = candidate.to_dict()
    assert record["members"] == ["G1", "G2"] and record["nested"] is True
    # without a combiner there is no probability
    assert cluster_candidate(kb, discrimination, context, probabilities=None).probability is None
    assert cluster_label(kb, ("G1", "G2")) == "Alpha / Beta (documented overlap)"


def test_split_related_keeps_only_same_activity_links_as_one_cluster(zeta_same_activity):
    from api.serializers import _related_groups
    from apt_classifier.overlap_cluster import split_related

    kb = _scope_kb()
    related = related_groups(kb)["G1"]
    same, other = split_related(kb, "G1", related)
    assert {gid for gid, _ in same} == {"G6", "G7"}  # reviewed same activity, nesting
    assert {(gid, kind) for gid, _, kind in other} == {("G5", LINK_SHARED_SOFTWARE)}
    item = SimpleNamespace(related_leading_groups=related, leading_group_id="G1")
    payload = {row["group_id"]: row for row in _related_groups(kb, item)}
    assert payload["G6"]["link"] == "same_activity"
    assert payload["G6"]["reason"] == "ATT&CK: Zeta has been linked to Alpha"
    assert payload["G7"]["link"] == "same_activity"  # nesting reads as the same activity
    assert payload["G5"]["link"] == LINK_SHARED_SOFTWARE and "KillDisk" in payload["G5"]["reason"]
    assert (
        _related_groups(kb, SimpleNamespace(related_leading_groups={}, leading_group_id="G1")) == []
    )


def test_a_narrower_scope_can_select_a_lighter_nested_twin():
    # Epsilon only shares malware and carries more mass; Eta nests inside Alpha. The full scope
    # picks Epsilon (not nested, so no nested-twin hold); the default picks Eta, whose nesting
    # holds a single name back when the cluster fails (attribution.assess_attribution)
    kb = _scope_kb()
    context = _context({"T1": 1.0, "T4": 0.3, "T7": 0.2})
    discrimination = discriminate(kb, _fusion("G1", "G5", "G7", "G8"), context, top_k=4)
    totals = {item.group_id: item.total for item in discrimination.scores}
    assert totals["G5"] > totals["G7"] > 0
    full = select_twin(kb, discrimination, scope="full")
    documented = select_twin(kb, discrimination)
    assert full["group_id"] == "G5" and not full["nested"]
    assert documented["group_id"] == "G7" and documented["nested"]
    assert documented["link"] == LINK_NESTING


def test_split_related_follows_the_scope_and_the_named_cluster(zeta_same_activity):
    from apt_classifier.overlap_cluster import recorded_cluster_scope, split_related

    kb = _scope_kb()
    related = related_groups(kb)["G1"]
    full_same, full_other = split_related(kb, "G1", related, scope="full")
    assert {gid for gid, _ in full_same} == {"G5", "G6", "G7"} and full_other == []
    nested_same, nested_other = split_related(kb, "G1", related, scope="nested")
    assert {gid for gid, _ in nested_same} == {"G7"}
    assert {gid for gid, _, _ in nested_other} == {"G5", "G6"}
    # a named cluster's twin is always reported with the leader
    same, other = split_related(kb, "G1", related, named_cluster=("G1", "G5"))
    assert "G5" in {gid for gid, _ in same} and "G5" not in {gid for gid, _, _ in other}
    # the scope an analysis selected its twin under is read back from the combiner state
    state = SimpleNamespace(context=SimpleNamespace(combiner={"cluster_scope": "full"}))
    assert recorded_cluster_scope(state) == "full"
    assert recorded_cluster_scope(SimpleNamespace(context=SimpleNamespace(combiner=None))) == (
        "documented"
    )
    assert recorded_cluster_scope(SimpleNamespace(), default="nested") == "nested"


def test_unreviewed_description_links_and_false_matches_fail_closed(monkeypatch):
    from apt_classifier.attack import same_activity
    from apt_classifier.overlap_cluster import link_reason, split_related

    kb = _scope_kb()
    # Eta nests inside Alpha; once Eta's description names Alpha without a review, the nesting
    # route closes (the text may call them distinct)
    kb.groups["G7"].description = "Eta is not Alpha."
    kb.version = "scope-2"  # new stamp for the memoized link tables
    assert link_kind(kb, "G1", "G7") == LINK_DESCRIPTION
    assert link_reason(kb, "G1", "G7").endswith("(not reviewed)")
    # reviewed as a false match and related only through that description: no link at all
    monkeypatch.setitem(same_activity.NOT_SAME_ACTIVITY, frozenset(("G1", "G6")), "false match: x")
    assert link_kind(kb, "G1", "G6") is None
    same, other = split_related(kb, "G1", related_groups(kb)["G1"])
    assert "G6" not in {gid for gid, _ in same} | {gid for gid, _, _ in other}
    # a reviewed distinct verdict shows with the reason
    monkeypatch.setitem(same_activity.NOT_SAME_ACTIVITY, frozenset(("G1", "G7")), "distinct: y")
    assert link_reason(kb, "G1", "G7").endswith("(reviewed: distinct: y)")
