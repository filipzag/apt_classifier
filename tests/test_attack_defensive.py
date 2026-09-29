"""Tests for ATT&CK Detection Strategy, Analytic, and Data Component parsing."""

from apt_classifier.attack.kb import AttackKnowledgeBase, LogSource
from apt_classifier.attack.parse import parse_attack_bundle

_DETECTION_BUNDLE = {
    "type": "bundle",
    "id": "bundle--test",
    "objects": [
        {
            "type": "identity",
            "id": "identity--test",
            "name": "Test",
            "identity_class": "organization",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
        },
        {
            "type": "marking-definition",
            "id": "marking-definition--test",
            "definition_type": "statement",
            "definition": {"statement": "Copyright"},
            "created": "2024-01-01T00:00:00Z",
        },
        {
            "type": "x-mitre-collection",
            "id": "x-mitre-collection--test",
            "name": "Test Collection v19.2",
            "x_mitre_version": "19.2",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "attack-pattern",
            "id": "attack-pattern--T1055-011",
            "name": "Extra Window Memory Injection",
            "description": "Adversaries inject code into EWM.",
            "x_mitre_detection": "Monitor for SetWindowLong.",
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "T1055.011",
                    "url": "https://attack.mitre.org/techniques/T1055/011",
                }
            ],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "defense-evasion"}
            ],
            "x_mitre_platforms": ["Windows"],
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "x-mitre-detection-strategy",
            "id": "x-mitre-detection-strategy--det0217",
            "name": "Detection Strategy for EWM Injection",
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "DET0217",
                    "url": "https://attack.mitre.org/detectionstrategies/DET0217",
                }
            ],
            "x_mitre_analytic_refs": ["x-mitre-analytic--an0608"],
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "x-mitre-analytic",
            "id": "x-mitre-analytic--an0608",
            "name": "Analytic 0608",
            "description": "Monitor Sysmon Event 10 for cross-process SetWindowLong.",
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "AN0608",
                    "url": "https://attack.mitre.org/detectionstrategies/DET0217#AN0608",
                }
            ],
            "x_mitre_platforms": ["Windows"],
            "x_mitre_log_source_references": [
                {
                    "x_mitre_data_component_ref": "x-mitre-data-component--dc0035",
                    "name": "WinEventLog:Sysmon",
                    "channel": "EventCode=10",
                },
                {
                    "x_mitre_data_component_ref": "x-mitre-data-component--dc0021",
                    "name": "WinEventLog:Security",
                    "channel": "EventCode=4688",
                },
            ],
            "x_mitre_mutable_elements": [
                {"field": "ProcessWhitelist", "description": "Known safe processes"},
            ],
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "x-mitre-data-component",
            "id": "x-mitre-data-component--dc0035",
            "name": "Process Access",
            "description": "Process access events from Sysmon.",
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "DC0035",
                    "url": "https://attack.mitre.org/datacomponents/DC0035",
                }
            ],
            "x_mitre_log_sources": [
                {"name": "WinEventLog:Sysmon", "channel": "EventCode=10"},
            ],
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "x-mitre-data-component",
            "id": "x-mitre-data-component--dc0021",
            "name": "Process Creation",
            "description": "Process creation events.",
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "DC0021",
                    "url": "https://attack.mitre.org/datacomponents/DC0021",
                }
            ],
            "x_mitre_log_sources": [
                {"name": "WinEventLog:Security", "channel": "EventCode=4688"},
            ],
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "created_by_ref": "identity--test",
            "object_marking_refs": ["marking-definition--test"],
        },
        {
            "type": "relationship",
            "id": "relationship--det0217-t1055",
            "source_ref": "x-mitre-detection-strategy--det0217",
            "target_ref": "attack-pattern--T1055-011",
            "relationship_type": "detects",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
        },
    ],
}


def _parse() -> AttackKnowledgeBase:
    return parse_attack_bundle(_DETECTION_BUNDLE, version="19.2")


def test_detection_strategies_parsed():
    kb = _parse()
    assert "DET0217" in kb.detection_strategies
    ds = kb.detection_strategies["DET0217"]
    assert ds.name == "Detection Strategy for EWM Injection"
    assert ds.technique_ids == ["T1055.011"]
    assert ds.analytic_ids == ["AN0608"]


def test_analytics_parsed_with_log_sources():
    kb = _parse()
    assert "AN0608" in kb.analytics
    an = kb.analytics["AN0608"]
    assert an.description.startswith("Monitor Sysmon")
    assert an.platforms == ["Windows"]
    assert len(an.log_source_refs) == 2
    assert an.log_source_refs[0] == LogSource(name="WinEventLog:Sysmon", channel="EventCode=10")
    assert an.data_component_ids == ["DC0035", "DC0021"]
    assert an.mutable_elements == [
        {"field": "ProcessWhitelist", "description": "Known safe processes"}
    ]


def test_data_components_parsed_with_log_sources():
    kb = _parse()
    assert "DC0035" in kb.data_components
    dc = kb.data_components["DC0035"]
    assert dc.name == "Process Access"
    assert dc.log_sources == [LogSource(name="WinEventLog:Sysmon", channel="EventCode=10")]


def test_technique_detection_strategies_index():
    kb = _parse()
    result = kb.technique_detection_strategies
    assert result["T1055.011"] == ["DET0217"]


def test_kb_serialization_roundtrip_preserves_defensive_objects():
    import tempfile
    from pathlib import Path

    from apt_classifier.attack.kb import load_knowledge_base, save_knowledge_base

    kb = _parse()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    save_knowledge_base(kb, path)
    loaded = load_knowledge_base(path)
    Path(path).unlink()
    assert "DET0217" in loaded.detection_strategies
    assert "AN0608" in loaded.analytics
    assert "DC0035" in loaded.data_components
    an = loaded.analytics["AN0608"]
    assert an.log_source_refs[0].name == "WinEventLog:Sysmon"
    assert an.data_component_ids == ["DC0035", "DC0021"]
    dc = loaded.data_components["DC0035"]
    assert dc.log_sources[0].channel == "EventCode=10"


def test_backward_compat_loading_without_defensive_objects():
    import json
    import tempfile
    from pathlib import Path

    from apt_classifier.attack.kb import AttackKnowledgeBase

    old_data = {
        "version": "15.1",
        "techniques": {"T1059": {"id": "T1059", "name": "PowerShell"}},
        "groups": {},
        "software": {},
        "campaigns": {},
        "idf": {},
        "priors": {},
        "prominence": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(old_data, f)
        path = f.name
    kb = AttackKnowledgeBase.load(path)
    Path(path).unlink()
    assert kb.detection_strategies == {}
    assert kb.analytics == {}
    assert kb.data_components == {}
    assert "T1059" in kb.techniques
