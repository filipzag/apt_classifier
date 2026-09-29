from types import SimpleNamespace

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.diamond import build_diamond_model
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance, StrategicContext
from apt_classifier.fusion import reciprocal_rank_fusion
from apt_classifier.origin import assess_origin_and_threat_type


def _state(synthetic_kb, *, strategic_context=None):
    action = Observation(
        text="PowerShell used RedKit to access the target environment.",
        provenance=Provenance(source="free_text"),
    )
    ioc = Observation(
        text="Observed DOMAIN indicator c2.example.test.",
        kind="ioc",
        metadata={"indicator_type": "domain", "value": "c2.example.test"},
        provenance=Provenance(source="ioc", source_name="analyst-input"),
    )
    technical = ClassifierResult(
        "specific_behavior",
        True,
        (
            ClassifierEvidence("specific_behavior", "G1", 0.9, 1),
            ClassifierEvidence("specific_behavior", "G2", 0.4, 2),
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
                signal_ids=("target_geographies:united states", "target_industries:defense"),
            ),
        ),
    )
    fusion = reciprocal_rank_fusion((technical, strategic))
    ledger = EvidenceLedger(
        observations=[action, ioc],
        strategic_context=strategic_context or StrategicContext(),
    )
    retrieval = SimpleNamespace(
        observed_techniques={"T1": 0.9},
        technique_observations={"T1": (action.id,)},
        matches=(),
        specific_behavior_matches=(),
    )
    context = SimpleNamespace(software_ids={"S1"})
    return SimpleNamespace(
        ledger=ledger,
        retrieval=retrieval,
        context=context,
        classifier_results=(technical, strategic),
        fusion=fusion,
        origin_assessment=assess_origin_and_threat_type(synthetic_kb, fusion),
        threat_intel_snapshot=None,
        threat_intel_matches=(),
    )


def test_diamond_model_maps_four_vertices_and_provenance(synthetic_kb):
    state = _state(
        synthetic_kb,
        strategic_context=StrategicContext(
            target_geographies=["United States"], target_industries=["defense"]
        ),
    )
    model = build_diamond_model(synthetic_kb, state, top_groups=2)

    assert {node.vertex for node in model.nodes} == {
        "adversary",
        "capability",
        "infrastructure",
        "victim",
    }
    assert any(node.status == "hypothesis" for node in model.nodes_for("adversary"))
    assert any(node.label.startswith("domain:") for node in model.nodes_for("infrastructure"))
    assert any("United States" in node.label for node in model.nodes_for("victim"))
    relationships = {edge.relationship for edge in model.edges}
    assert "documented_capability" in relationships
    assert "targeting_context_match" in relationships
    assert "documented_tool_association" in relationships
    assert "not proof of attribution" in model.nodes_for("adversary")[0].details


def test_diamond_model_keeps_missing_victimology_explicit(synthetic_kb):
    model = build_diamond_model(synthetic_kb, _state(synthetic_kb), top_groups=1)
    victim_nodes = model.nodes_for("victim")
    assert len(victim_nodes) == 1
    assert victim_nodes[0].status == "unknown"
    assert "not specified" in victim_nodes[0].label
    assert any("Victim geography" not in caveat for caveat in model.caveats)


def test_diamond_model_serializes_for_audit(synthetic_kb):
    model = build_diamond_model(synthetic_kb, _state(synthetic_kb), top_groups=1)
    payload = model.to_dict()
    assert payload["nodes"]
    assert payload["edges"]
    assert payload["caveats"]
    assert all("vertex" in node for node in payload["nodes"])
