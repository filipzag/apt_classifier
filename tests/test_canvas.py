from types import SimpleNamespace

from apt_classifier.canvas import build_analyst_canvas, build_candidate_dossier
from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.evidence.models import (
    EvidenceLedger,
    Observation,
    Provenance,
    StrategicContext,
    TechniqueMatch,
)
from apt_classifier.fusion import reciprocal_rank_fusion
from apt_classifier.origin import assess_origin_and_threat_type
from apt_classifier.retrieve import BehaviorMatch
from apt_classifier.threat_intel import ThreatIntelItem, ThreatIntelMatch, ThreatIntelSnapshot


def test_canvas_connects_evidence_techniques_tools_groups_context_and_country(synthetic_kb):
    action = Observation(
        text="ContosoLoader used PowerShell -enc SQBFAFgA.",
        provenance=Provenance(source="free_text"),
    )
    tool = Observation(
        text="RedKit was identified.",
        kind="software",
        metadata={"software_id": "S1"},
        provenance=Provenance(source="ioc"),
    )
    ledger = EvidenceLedger(
        observations=[action, tool],
        strategic_context=StrategicContext(
            target_geographies=["United States"],
            target_industries=["defense"],
            motivations=["espionage"],
        ),
    )
    retrieval = SimpleNamespace(
        observed_techniques={"T1": 0.91, "T4": 0.6},
        technique_observations={"T1": (action.id,), "T4": (action.id,)},
        matches=(
            TechniqueMatch(
                technique_id="T1",
                technique_name="One",
                score=0.91,
                observation_ids=[action.id],
                matched_text="Actor-specific command behavior.",
            ),
        ),
        specific_behavior_matches=(
            BehaviorMatch(
                group_id="G1",
                observation_id=action.id,
                technique_id="T1",
                score=0.94,
                matched_text="Alpha used ContosoLoader with PowerShell -enc SQBFAFgA.",
                vector_id="G1:actor-behavior:T1:0000",
            ),
        ),
    )
    context = SimpleNamespace(
        software_ids={"S1"},
        negative_techniques={"T3"},
    )
    technical = ClassifierResult(
        "specific_behavior",
        True,
        (
            ClassifierEvidence(
                "specific_behavior",
                "G1",
                0.94,
                1,
                signal_ids=("G1:actor-behavior:T1:0000",),
            ),
            ClassifierEvidence("specific_behavior", "G2", 0.5, 2),
        ),
    )
    strategic = ClassifierResult(
        "strategic_context",
        True,
        (
            ClassifierEvidence(
                "strategic_context",
                "G1",
                1.0,
                1,
                signal_ids=(
                    "target_geographies:united states",
                    "target_industries:defense",
                    "motivations:espionage",
                ),
            ),
        ),
    )
    campaign = ClassifierResult(
        "campaign_profile",
        True,
        (ClassifierEvidence("campaign_profile", "G1", 0.5, 1, signal_ids=("C1",)),),
    )
    fusion = reciprocal_rank_fusion((technical, strategic, campaign))
    intel_item = ThreatIntelItem(
        id="intel-1",
        source="CISA",
        kind="campaign_advisory",
        title="Alpha campaign advisory",
        summary="CVE overlap",
        published_at="2026-09-01T00:00:00+00:00",
        url="https://www.cisa.gov/example",
        verification="authoritative",
    )
    intel_snapshot = ThreatIntelSnapshot(
        fetched_at="2026-09-01T01:00:00+00:00",
        items=(intel_item,),
        sources=(),
    )
    intel_match = ThreatIntelMatch(
        item_id=intel_item.id,
        score=9.0,
        reasons=("CVE overlap: CVE-2026-1234", "Ranked actor reference: Alpha"),
        group_ids=("G1",),
        observation_ids=(action.id,),
        verification="authoritative",
    )
    state = SimpleNamespace(
        ledger=ledger,
        retrieval=retrieval,
        context=context,
        classifier_results=(technical, strategic, campaign),
        fusion=fusion,
        origin_assessment=assess_origin_and_threat_type(synthetic_kb, fusion),
        threat_intel_snapshot=intel_snapshot,
        threat_intel_matches=(intel_match,),
    )

    canvas = build_analyst_canvas(synthetic_kb, state, top_groups=2)
    node_kinds = {node.kind for node in canvas.nodes}
    relationships = {edge.relationship for edge in canvas.edges}

    assert {
        "evidence",
        "technique",
        "tool",
        "group",
        "campaign",
        "context",
        "country",
        "live_intel",
    } <= node_kinds
    assert {
        "maps_to_technique",
        "documented_technique",
        "specific_behavior",
        "identified_tool",
        "documented_tool",
        "campaign_link",
        "context_match",
        "documented_origin",
        "live_actor_reference",
        "live_indicator_overlap",
    } <= relationships


def test_candidate_dossier_separates_support_and_unmatched_evidence(synthetic_kb):
    action = Observation(text="Observed T1 and T4.", provenance=Provenance(source="test"))
    ledger = EvidenceLedger(observations=[action])
    retrieval = SimpleNamespace(
        observed_techniques={"T1": 0.9, "T4": 0.8},
        technique_observations={"T1": (action.id,), "T4": (action.id,)},
        specific_behavior_matches=(),
    )
    context = SimpleNamespace(software_ids=set(), negative_techniques={"T2"})
    result = ClassifierResult(
        "test",
        True,
        (ClassifierEvidence("test", "G1", 1.0, 1),),
    )
    fusion = reciprocal_rank_fusion((result,))
    state = SimpleNamespace(
        ledger=ledger,
        retrieval=retrieval,
        context=context,
        classifier_results=(result,),
        fusion=fusion,
        origin_assessment=assess_origin_and_threat_type(synthetic_kb, fusion),
    )

    dossier = build_candidate_dossier(synthetic_kb, state, "G1")

    assert [item["technique_id"] for item in dossier["matched_techniques"]] == ["T1"]
    assert [item["technique_id"] for item in dossier["unmatched_techniques"]] == ["T4"]
    assert dossier["contradicting_techniques"] == ["T2"]
    assert dossier["rank"] == 1
