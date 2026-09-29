"""Tests for telemetry-aware collection guidance and negative-evidence validation."""

from apt_classifier.attack.kb import (
    Analytic,
    AttackKnowledgeBase,
    DataComponent,
    DetectionStrategy,
    LogSource,
)
from apt_classifier.telemetry import (
    TelemetryCoverage,
    TelemetryProfile,
    all_log_sources_for_technique,
    negative_answer_is_valid,
    technique_collection_steps,
)


def _kb_with_detection() -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        version="19.2",
        techniques={},
        groups={},
        software={},
        campaigns={},
        detection_strategies={
            "DET0217": DetectionStrategy(
                id="DET0217",
                name="EWM Injection Detection",
                analytic_ids=["AN0608"],
                technique_ids=["T1055.011"],
            ),
        },
        analytics={
            "AN0608": Analytic(
                id="AN0608",
                name="Analytic 0608",
                description="Monitor Sysmon Event 10 for cross-process SetWindowLong.",
                platforms=["Windows"],
                log_source_refs=[
                    LogSource(name="WinEventLog:Sysmon", channel="EventCode=10"),
                    LogSource(name="WinEventLog:Security", channel="EventCode=4688"),
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


def test_collection_steps_generated_from_analytics():
    kb = _kb_with_detection()
    steps = technique_collection_steps(kb, "T1055.011")
    assert len(steps) == 1
    step = steps[0]
    assert step.technique_id == "T1055.011"
    assert step.analytic_id == "AN0608"
    assert step.description.startswith("Monitor Sysmon")
    assert len(step.log_sources) == 2
    assert step.log_sources[0].name == "WinEventLog:Sysmon"
    assert step.log_sources[0].channel == "EventCode=10"
    assert step.log_sources[0].coverage is TelemetryCoverage.UNKNOWN


def test_collection_steps_empty_for_unknown_technique():
    kb = _kb_with_detection()
    assert technique_collection_steps(kb, "T9999") == []


def test_profile_sets_log_source_coverage():
    kb = _kb_with_detection()
    profile = TelemetryProfile()
    profile.set_coverage("WinEventLog:Sysmon", "EventCode=10", TelemetryCoverage.AVAILABLE)
    steps = technique_collection_steps(kb, "T1055.011", profile)
    assert steps[0].log_sources[0].coverage is TelemetryCoverage.AVAILABLE
    assert steps[0].log_sources[1].coverage is TelemetryCoverage.UNKNOWN


def test_negative_answer_valid_when_telemetry_available():
    kb = _kb_with_detection()
    profile = TelemetryProfile()
    profile.set_coverage("WinEventLog:Sysmon", "EventCode=10", TelemetryCoverage.AVAILABLE)
    assert negative_answer_is_valid("T1055.011", kb, profile) is True


def test_negative_answer_invalid_when_telemetry_unknown():
    kb = _kb_with_detection()
    profile = TelemetryProfile()
    assert negative_answer_is_valid("T1055.011", kb, profile) is False


def test_negative_answer_invalid_without_profile():
    kb = _kb_with_detection()
    assert negative_answer_is_valid("T1055.011", kb, None) is False


def test_all_log_sources_returns_distinct():
    kb = _kb_with_detection()
    sources = all_log_sources_for_technique(kb, "T1055.011")
    assert len(sources) == 2
    assert (sources[0].name, sources[0].channel) == ("WinEventLog:Sysmon", "EventCode=10")
    assert (sources[1].name, sources[1].channel) == ("WinEventLog:Security", "EventCode=4688")


def test_data_component_id_resolved_in_collection_step():
    kb = _kb_with_detection()
    steps = technique_collection_steps(kb, "T1055.011")
    # The first log source (WinEventLog:Sysmon, EventCode=10) should resolve to DC0035
    assert steps[0].log_sources[0].data_component_id == "DC0035"


def test_build_classifier_context_downgrades_negative_without_telemetry():
    from apt_classifier.evidence.models import (
        EvidenceLedger,
        EvidenceStatus,
        Observation,
        Provenance,
        QuestionAnswer,
    )
    from apt_classifier.retrieve import RetrievalResult, build_classifier_context

    kb = _kb_with_detection()
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Process injection observed",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    ledger.question_answers.append(
        QuestionAnswer(
            question_id="q1",
            question="Was T1055.011 observed?",
            status=EvidenceStatus.NEGATIVE,
            technique_id="T1055.011",
        )
    )
    retrieval = RetrievalResult(
        matches=(),
        observed_techniques={},
        technique_observations={},
        semantic_profile_scores={},
        group_text_scores={},
        specific_behavior_scores={},
        specific_behavior_signals={},
        specific_behavior_matches=(),
        candidates=(),
        mapping_mode="hybrid",
        parent_policy="ancestor_compatible",
        diagnostics={},
    )
    # Without telemetry profile: negative is accepted (backward compatible)
    context = build_classifier_context(kb, ledger, retrieval)
    assert "T1055.011" in context.negative_techniques

    # With telemetry profile but no available sources: negative is downgraded
    profile = TelemetryProfile()
    context = build_classifier_context(kb, ledger, retrieval, telemetry_profile=profile)
    assert "T1055.011" not in context.negative_techniques

    # With available telemetry: negative is accepted
    profile.set_coverage("WinEventLog:Sysmon", "EventCode=10", TelemetryCoverage.AVAILABLE)
    context = build_classifier_context(kb, ledger, retrieval, telemetry_profile=profile)
    assert "T1055.011" in context.negative_techniques


def test_build_classifier_context_no_detection_data_accepts_all_negatives():
    from apt_classifier.evidence.models import (
        EvidenceLedger,
        EvidenceStatus,
        Observation,
        Provenance,
        QuestionAnswer,
    )
    from apt_classifier.retrieve import RetrievalResult, build_classifier_context

    # KB without detection strategies — telemetry validation is skipped
    kb = AttackKnowledgeBase(version="19.2")
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Some activity",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    ledger.question_answers.append(
        QuestionAnswer(
            question_id="q1",
            question="Was T1055.011 observed?",
            status=EvidenceStatus.NEGATIVE,
            technique_id="T1055.011",
        )
    )
    retrieval = RetrievalResult(
        matches=(),
        observed_techniques={},
        technique_observations={},
        semantic_profile_scores={},
        group_text_scores={},
        specific_behavior_scores={},
        specific_behavior_signals={},
        specific_behavior_matches=(),
        candidates=(),
        mapping_mode="hybrid",
        parent_policy="ancestor_compatible",
        diagnostics={},
    )
    profile = TelemetryProfile()
    context = build_classifier_context(kb, ledger, retrieval, telemetry_profile=profile)
    assert "T1055.011" in context.negative_techniques
