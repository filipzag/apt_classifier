"""The analyst's Top-three view (apt_classifier/top3_view.py): display only."""

from __future__ import annotations

from types import SimpleNamespace

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.attribution import AttributionStage
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.separation import (
    CandidateDiscrimination,
    DiscriminationResult,
    DiscriminationWeights,
    discriminate,
)
from apt_classifier.top3_view import build_top3_view, top3_summary


def _kb() -> AttackKnowledgeBase:
    kb = AttackKnowledgeBase(
        version="t",
        techniques={
            tid: Technique(id=tid, name=f"Technique {tid}") for tid in ["T1", "T2", "T3", "T4"]
        },
        groups={
            "G0046": Group(
                id="G0046",
                name="FIN7",
                description="FIN7 is a financially motivated criminal group.",
                technique_ids=["T1", "T2", "T3"],
                software_ids=["S0030", "S0002"],
            ),
            "G0008": Group(
                id="G0008",
                name="Carbanak",
                description="Carbanak is a cybercriminal group that shares tooling with FIN7.",
                technique_ids=["T1", "T2"],
                software_ids=["S0030"],
            ),
            "G0016": Group(
                id="G0016",
                name="APT29",
                description=(
                    "APT29 is a threat group that has been attributed to Russia's Foreign "
                    "Intelligence Service and conducts cyber espionage."
                ),
                technique_ids=["T1", "T4"],
                software_ids=["S0002"],
            ),
        },
        software={
            "S0030": Software(id="S0030", name="Carbanak", software_type="malware"),
            "S0002": Software(id="S0002", name="Mimikatz", software_type="tool"),
        },
    )
    kb.recompute_statistics()
    return kb


def _fusion(*ids: str) -> FusionResult:
    scores = [1.0 / (20 + i) for i in range(1, len(ids) + 1)]
    return FusionResult(
        tuple(
            FusionEntry(gid, score, 100.0 * score / scores[0], rank, {}, 1, 1, 0.0, False)
            for rank, (gid, score) in enumerate(zip(ids, scores, strict=True), 1)
        ),
        ("c",),
        {},
        20,
    )


def _state(kb, order, *, discrimination=None, stage="known_group_shortlist"):
    fusion = _fusion(*order)
    context = SimpleNamespace(
        kb=kb,
        observed_techniques={"T1": 1.0, "T3": 0.9, "T4": 0.8},
        negative_techniques=set(),
        software_ids={"S0030"},
        specific_behavior_matches=(),
        parent_policy="ancestor_compatible",
        decisive_software={},
    )
    context.discrimination = discrimination or discriminate(kb, fusion, context, top_k=3)
    context.combiner = {
        "probabilities": {order[0]: 0.6, order[1]: 0.3},
        "rrf_ranks": {order[0]: 2, order[1]: 1},
    }
    decision = SimpleNamespace(
        stage=AttributionStage(stage),
        runner_up_group_id=context.discrimination.runner_up_id,
        separation=context.discrimination.separation,
    )
    return SimpleNamespace(
        fusion=fusion,
        deterministic_fusion=fusion,
        context=context,
        attribution_decision=decision,
        ledger=SimpleNamespace(observations=[]),
    )


def _manual(*items):
    weights = DiscriminationWeights()
    scores = tuple(
        CandidateDiscrimination(gid, 0.0, technique, behavior, 0.0, 0.0, technique + 2 * behavior)
        for gid, technique, behavior in items
    )
    return DiscriminationResult(scores=scores, top_k=len(scores), weights=weights)


def test_documented_overlap_pair_is_reported_as_one_cluster():
    kb = _kb()
    view = build_top3_view(kb, _state(kb, ["G0046", "G0008", "G0016"]))
    assert view["state"] == "overlap"
    assert view["other"]["group_id"] == "G0008"
    pair = next(p for p in view["pairs"] if {p["a"], p["b"]} == {"G0046", "G0008"})
    assert pair["related_reason"] and pair["overlap_cluster"]
    assert pair["containment"] == 1.0  # Carbanak's documented profile lies inside FIN7's
    # only FIN7 documents T3 among the three; only APT29 documents T4
    fin7_vs_apt29 = next(p for p in view["pairs"] if {p["a"], p["b"]} == {"G0046", "G0016"})
    assert [t["technique_id"] for t in fin7_vs_apt29["a_only"]["techniques"]] == ["T3"]
    assert [t["technique_id"] for t in fin7_vs_apt29["b_only"]["techniques"]] == ["T4"]
    assert fin7_vs_apt29["a_only"]["software"] == [{"id": "S0030", "name": "Carbanak"}]
    # per-candidate totals are the recorded gate discrimination, not a recomputation
    recorded = {item.group_id: item.total for item in view_state_totals(kb)}
    for candidate in view["candidates"]:
        assert candidate["channels"]["total"] == round(recorded[candidate["group_id"]], 4)


def view_state_totals(kb):
    return _state(kb, ["G0046", "G0008", "G0016"]).context.discrimination.scores


def test_attack_documentation_is_labelled_per_group():
    kb = _kb()
    view = build_top3_view(kb, _state(kb, ["G0046", "G0008", "G0016"]))
    documented = {c["group_id"]: c["documented"] for c in view["candidates"]}
    assert documented["G0046"]["type"] == "criminal"
    assert "financial gain" in documented["G0046"]["motivations"]
    assert documented["G0016"]["type"] == "state-sponsored"
    assert "espionage" in documented["G0016"]["motivations"]
    assert [o["country"] for o in documented["G0016"]["origins"]] == ["Russia"]
    assert documented["G0046"]["origins"] == []


def test_states_for_split_evidence_and_thin_evidence():
    kb = _kb()
    # APT29 leads; FIN7 has clearly more exclusive evidence: the evidence pulls both ways
    split = _manual(("G0016", 0.2, 0.0), ("G0046", 2.0, 0.0), ("G0008", 0.0, 0.0))
    view = build_top3_view(kb, _state(kb, ["G0016", "G0046", "G0008"], discrimination=split))
    assert view["state"] == "split" and view["other"]["group_id"] == "G0046"
    assert view["contest"]["promoted"] == "G0046" and not view["contest"]["blocked"]
    assert "rarely right" in view["summary"]

    # same, but actor behaviour favours the leader: still split, and the summary says why
    pulls = _manual(("G0016", 0.0, 0.6), ("G0046", 2.6, 0.0), ("G0008", 0.0, 0.0))
    view = build_top3_view(kb, _state(kb, ["G0016", "G0046", "G0008"], discrimination=pulls))
    assert view["state"] == "split" and view["contest"]["blocked"]
    assert "behavior evidence favours APT29" in view["summary"]

    thin = _manual(("G0016", 0.4, 0.0), ("G0046", 0.3, 0.0), ("G0008", 0.0, 0.0))
    view = build_top3_view(kb, _state(kb, ["G0016", "G0046", "G0008"], discrimination=thin))
    assert view["state"] == "thin"

    named = build_top3_view(
        kb, _state(kb, ["G0046", "G0008", "G0016"], stage="known_group_hypothesis")
    )
    assert named["state"] == "named"
    assert set(top3_summary(named)) == {"state", "label", "summary", "leader", "other"}


def test_view_never_changes_the_state_it_reads():
    kb = _kb()
    state = _state(kb, ["G0046", "G0008", "G0016"])
    before = ([e.group_id for e in state.fusion.entries], state.context.discrimination)
    build_top3_view(kb, state)
    assert ([e.group_id for e in state.fusion.entries], state.context.discrimination) == before


def test_empty_shortlist():
    kb = _kb()
    state = _state(kb, ["G0046", "G0008", "G0016"])
    state.fusion = state.deterministic_fusion = _fusion()
    assert build_top3_view(kb, state)["state"] == "empty"


def test_each_candidate_lists_its_documented_overlap_with_rank():
    kb = _kb()
    view = build_top3_view(kb, _state(kb, ["G0046", "G0016", "G0008"]))
    fin7 = next(c for c in view["candidates"] if c["group_id"] == "G0046")
    assert [(r["group_id"], r["rank"]) for r in fin7["related"]] == [("G0008", 3)]
    assert fin7["related"][0]["reason"]
    apt29 = next(c for c in view["candidates"] if c["group_id"] == "G0016")
    assert apt29["related"] == []
