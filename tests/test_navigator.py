"""Tests for ATT&CK Navigator layer export."""

from apt_classifier.attack.kb import (
    Analytic,
    AttackKnowledgeBase,
    DataComponent,
    DetectionStrategy,
    LogSource,
    Technique,
)
from apt_classifier.navigator import (
    build_checked_absent_layer,
    build_observed_layer,
    build_telemetry_gap_layer,
    export_layers,
)
from apt_classifier.telemetry import TelemetryCoverage, TelemetryProfile


def _kb() -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        version="19.2",
        techniques={
            "T1055.011": Technique(
                id="T1055.011",
                name="Extra Window Memory Injection",
                description="Inject code into EWM.",
                tactics=["defense-evasion"],
            ),
            "T1059.001": Technique(
                id="T1059.001",
                name="PowerShell",
                description="Adversaries may abuse PowerShell.",
                tactics=["execution"],
            ),
        },
        groups={},
        software={},
        campaigns={},
        detection_strategies={
            "DET0217": DetectionStrategy(
                id="DET0217",
                name="EWM Detection",
                analytic_ids=["AN0608"],
                technique_ids=["T1055.011"],
            ),
        },
        analytics={
            "AN0608": Analytic(
                id="AN0608",
                name="Analytic 0608",
                description="Monitor Sysmon Event 10.",
                platforms=["Windows"],
                log_source_refs=[
                    LogSource(name="WinEventLog:Sysmon", channel="EventCode=10"),
                ],
                data_component_ids=["DC0035"],
            ),
        },
        data_components={
            "DC0035": DataComponent(
                id="DC0035",
                name="Process Access",
                description="Process access events.",
                log_sources=[LogSource(name="WinEventLog:Sysmon", channel="EventCode=10")],
            ),
        },
    )


def test_observed_layer_includes_techniques_with_scores():
    kb = _kb()
    layer = build_observed_layer(kb, {"T1055.011": 0.85, "T1059.001": 1.0})
    assert layer["name"] == "Observed Techniques"
    assert layer["domain"] == "enterprise-attack"
    technique_ids = [t["techniqueID"] for t in layer["techniques"]]
    assert technique_ids == ["T1055.011", "T1059.001"]
    assert layer["techniques"][0]["score"] == 85
    assert layer["techniques"][1]["score"] == 100


def test_checked_absent_layer_marks_techniques():
    kb = _kb()
    layer = build_checked_absent_layer(kb, {"T1059.001"})
    assert layer["name"] == "Checked Absent Techniques"
    assert len(layer["techniques"]) == 1
    assert layer["techniques"][0]["techniqueID"] == "T1059.001"
    assert layer["techniques"][0]["color"] == "#ff6666"


def test_telemetry_gap_layer_shows_techniques_with_unknown_coverage():
    kb = _kb()
    profile = TelemetryProfile()
    layer = build_telemetry_gap_layer(kb, ["T1055.011", "T1059.001"], profile)
    # T1055.011 has analytics with unknown coverage → gap
    # T1059.001 has no analytics → no gap
    technique_ids = [t["techniqueID"] for t in layer["techniques"]]
    assert "T1055.011" in technique_ids
    assert "T1059.001" not in technique_ids


def test_telemetry_gap_layer_omits_when_coverage_available():
    kb = _kb()
    profile = TelemetryProfile()
    profile.set_coverage("WinEventLog:Sysmon", "EventCode=10", TelemetryCoverage.AVAILABLE)
    layer = build_telemetry_gap_layer(kb, ["T1055.011"], profile)
    assert layer["techniques"] == []


def test_export_layers_returns_all_three():
    kb = _kb()
    profile = TelemetryProfile()
    layers = export_layers(
        kb,
        observed_techniques={"T1055.011": 0.85},
        negative_techniques={"T1059.001"},
        candidate_technique_ids=["T1055.011", "T1059.001"],
        profile=profile,
    )
    assert set(layers.keys()) == {"observed", "checked_absent", "telemetry_gaps"}
    assert layers["observed"]["techniques"][0]["techniqueID"] == "T1055.011"
    assert layers["checked_absent"]["techniques"][0]["techniqueID"] == "T1059.001"
    assert any(t["techniqueID"] == "T1055.011" for t in layers["telemetry_gaps"]["techniques"])
