from types import SimpleNamespace

from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.questions import (
    QuestionCandidate,
    information_gain,
    interview_stop_reason,
    parse_interview_answer,
    select_question_with_llm,
    shortlist_questions,
    should_stop,
)
from apt_classifier.session import AttributionEngine


def test_information_gain_prefers_balanced_split():
    posterior = {f"G{i}": 0.25 for i in range(1, 5)}
    groups = {
        "G1": {"Tsplit", "Tall"},
        "G2": {"Tsplit", "Tall"},
        "G3": {"Tall"},
        "G4": {"Tall"},
    }
    split_gain, _ = information_gain(posterior, groups, "Tsplit")
    all_gain, _ = information_gain(posterior, groups, "Tall")
    assert split_gain > all_gain
    assert split_gain > 0


def test_shortlist_excludes_answered_and_builds_investigation_leads():
    kb = SimpleNamespace(
        group_techniques={"G1": {"T1", "T2"}, "G2": {"T2", "T3"}},
        techniques={
            "T1": SimpleNamespace(name="One"),
            "T2": SimpleNamespace(
                name="Hidden execution",
                tactics=("execution",),
                platforms=("Windows",),
                detection="Review process creation events and command-line telemetry.",
            ),
            "T3": SimpleNamespace(name="Three"),
        },
    )
    candidates = shortlist_questions(kb, {"G1": 0.6, "G2": 0.4}, {"T1"})
    assert {candidate.technique_id for candidate in candidates} == {"T2", "T3"}
    lead = next(candidate for candidate in candidates if candidate.technique_id == "T2")
    assert lead.profile_support == 1.0
    assert lead.supporting_group_count == 2
    assert lead.investigation_priority > lead.information_gain
    assert lead.hypothesis.startswith("Unconfirmed investigation hypothesis:")
    assert any("Windows" in step for step in lead.investigation_steps)
    assert any("process creation" in step.casefold() for step in lead.investigation_steps)


def test_target_os_keeps_other_operating_systems_out_of_the_interview():
    kb = SimpleNamespace(
        group_techniques={"G1": {"TWIN", "TCRON", "TJOB"}, "G2": {"TWIN", "TCLOUD", "TJOB"}},
        techniques={
            "TWIN": SimpleNamespace(name="Scheduled Task", platforms=("Windows",)),
            "TCRON": SimpleNamespace(name="Cron", platforms=("Linux", "macOS")),
            "TJOB": SimpleNamespace(
                name="Scheduled Task/Job", platforms=("Containers", "Linux", "Windows", "macOS")
            ),
            "TCLOUD": SimpleNamespace(name="Cloud Accounts", platforms=("IaaS", "SaaS")),
        },
    )
    posterior = {"G1": 0.6, "G2": 0.4}
    assert {c.technique_id for c in shortlist_questions(kb, posterior, set())} == {
        "TWIN",
        "TCRON",
        "TJOB",
        "TCLOUD",
    }
    linux = shortlist_questions(kb, posterior, set(), platforms=["Linux"])
    # Windows-only is never asked; cloud techniques are not an OS question
    assert {c.technique_id for c in linux} == {"TCRON", "TJOB", "TCLOUD"}
    job = next(c for c in linux if c.technique_id == "TJOB")
    scope_step = next(s for s in job.investigation_steps if s.startswith("Scope the search"))
    assert scope_step == "Scope the search to Linux hosts and containers."


def test_scope_step_names_places_to_search_not_platform_labels():
    kb = SimpleNamespace(
        group_techniques={"G1": {"TPRE", "TALL", "TESX"}},
        techniques={
            "TPRE": SimpleNamespace(
                name="Tool", platforms=("PRE",), tactics=("resource-development",)
            ),
            "TALL": SimpleNamespace(
                name="Phishing",
                platforms=(
                    "Identity Provider",
                    "Linux",
                    "Office Suite",
                    "SaaS",
                    "Windows",
                    "macOS",
                ),
            ),
            "TESX": SimpleNamespace(name="Hypervisor CLI", platforms=("ESXi", "Linux")),
        },
    )
    steps = {
        c.technique_id: c.investigation_steps for c in shortlist_questions(kb, {"G1": 1.0}, set())
    }
    # pre-compromise techniques happen outside the environment: no "PRE systems"
    assert steps["TPRE"][0].startswith("Pre-compromise activity outside the environment")
    assert not any("PRE" in step for step in steps["TPRE"])
    assert (
        "Scope the search to Windows, Linux and macOS hosts, SaaS applications, office suites "
        "(Microsoft 365, Google Workspace) and identity providers." in steps["TALL"]
    )
    assert "Scope the search to Linux hosts and ESXi hypervisors." in steps["TESX"]


def test_shortlist_does_not_reask_intake_confirmed_archive_utility():
    kb = SimpleNamespace(
        group_techniques={"G1": {"T1560", "T1560.001"}, "G2": {"T1560.001", "T1041"}},
        techniques={
            "T1560": SimpleNamespace(name="Archive Collected Data"),
            "T1560.001": SimpleNamespace(name="Archive via Utility"),
            "T1041": SimpleNamespace(name="Exfiltration Over C2 Channel"),
        },
    )
    candidates = shortlist_questions(
        kb,
        {"G1": 0.6, "G2": 0.4},
        {"T1560", "T1560.001"},
    )
    assert "T1560.001" not in {candidate.technique_id for candidate in candidates}


def test_windows_investigation_guidance_includes_channels_and_event_ids():
    kb = SimpleNamespace(
        group_techniques={
            "G1": {"T1059.001", "T1053.005"},
            "G2": {"T1059.001"},
        },
        techniques={
            "T1059.001": SimpleNamespace(
                name="PowerShell",
                tactics=("execution",),
                platforms=("Windows",),
                detection="Monitor PowerShell execution.",
            ),
            "T1053.005": SimpleNamespace(
                name="Scheduled Task/Job: Scheduled Task",
                tactics=("persistence",),
                platforms=("Windows",),
                detection="Monitor task creation and modification.",
            ),
        },
    )
    candidates = shortlist_questions(kb, {"G1": 0.6, "G2": 0.4}, set())
    powershell = next(item for item in candidates if item.technique_id == "T1059.001")
    scheduled_task = next(item for item in candidates if item.technique_id == "T1053.005")
    powershell_steps = " ".join(powershell.investigation_steps)
    scheduled_task_steps = " ".join(scheduled_task.investigation_steps)
    assert "Microsoft-Windows-PowerShell/Operational" in powershell_steps
    assert "4103" in powershell_steps and "4104" in powershell_steps
    assert "Security 4688" in powershell_steps and "Sysmon 1" in powershell_steps
    assert "Security Event IDs 4698" in scheduled_task_steps
    assert "TaskScheduler/Operational" in scheduled_task_steps
    assert "106" in scheduled_task_steps and "140" in scheduled_task_steps


def test_llm_selection_preserves_deterministic_investigation_guidance():
    kb = SimpleNamespace(
        group_techniques={"G1": {"T1"}, "G2": {"T2"}},
        techniques={
            "T1": SimpleNamespace(name="One", tactics=("execution",), platforms=("Windows",)),
            "T2": SimpleNamespace(name="Two"),
        },
    )
    candidates = shortlist_questions(kb, {"G1": 0.6, "G2": 0.4}, set())
    expected = next(candidate for candidate in candidates if candidate.technique_id == "T1")
    client = SimpleNamespace(
        complete=lambda *args, **kwargs: SimpleNamespace(
            parsed={"technique_id": "T1", "question": "What did execution telemetry show?"}
        )
    )
    selected = select_question_with_llm(
        client, candidates, "Observed activity", model="test/model", effort="low"
    )
    assert selected is not None
    assert selected.question == "What did execution telemetry show?"
    assert selected.hypothesis == expected.hypothesis
    assert selected.investigation_steps == expected.investigation_steps
    assert selected.investigation_priority == expected.investigation_priority


def test_llm_question_is_told_the_target_os_only_when_one_is_set():
    candidates = shortlist_questions(
        SimpleNamespace(group_techniques={"G1": {"T1"}}, techniques={}), {"G1": 1.0}, set()
    )
    prompts: list[str] = []

    def complete(messages, **kwargs):
        prompts.append(messages[-1]["content"])
        return SimpleNamespace(parsed={"technique_id": "T1", "question": "Seen?"})

    client = SimpleNamespace(complete=complete)
    select_question_with_llm(client, candidates, "Observed", model="test/model", effort="low")
    select_question_with_llm(
        client,
        candidates,
        "Observed",
        model="test/model",
        effort="low",
        target_platforms=["Linux", "ESXi"],
    )
    assert "affected hosts" not in prompts[0]
    assert "The affected hosts run Linux, ESXi" in prompts[1]
    assert (
        prompts[1].replace(
            "The affected hosts run Linux, ESXi: phrase the question for those operating systems.\n\n",
            "",
        )
        == prompts[0]
    )


def test_shortlist_only_contains_techniques_supported_by_weighted_candidates():
    kb = SimpleNamespace(
        group_techniques={"G1": {"T1"}, "G2": {"T2"}},
        techniques={"T1": SimpleNamespace(name="One"), "T2": SimpleNamespace(name="Two")},
    )
    candidates = shortlist_questions(kb, {"G1": 1.0}, set())
    assert [candidate.technique_id for candidate in candidates] == ["T1"]


def test_first_question_is_not_suppressed_by_actionability_threshold():
    candidate = QuestionCandidate(
        technique_id="T1",
        technique_name="One",
        information_gain=0.0,
        probability_yes=0.1,
        default_question="Did you observe One?",
        investigation_priority=0.025,
    )
    posterior = {"G1": 0.5, "G2": 0.5}
    assert interview_stop_reason(posterior, [candidate], 0) is None
    assert not should_stop(posterior, [candidate], 0)
    assert interview_stop_reason(posterior, [candidate], 1) == "low_priority"
    assert should_stop(posterior, [candidate], 1)


def test_stop_reason_distinguishes_limit_from_no_candidates():
    assert interview_stop_reason({}, [], 0, max_questions=10) == "no_candidates"
    assert interview_stop_reason({}, [], 10, max_questions=10) == "question_limit"


def test_stopping_rule_continues_for_actionable_hidden_behavior():
    kb = SimpleNamespace(
        group_techniques={"G1": {"T1"}, "G2": {"T2"}},
        techniques={"T1": SimpleNamespace(name="One"), "T2": SimpleNamespace(name="Two")},
    )
    posterior = {"G1": 0.7, "G2": 0.3}
    candidates = shortlist_questions(kb, posterior, set())
    assert not should_stop(posterior, candidates, 0)
    assert should_stop(posterior, candidates, 10)


def test_chat_answer_parser_supports_quick_and_typed_replies():
    assert parse_interview_answer("yes") is EvidenceStatus.POSITIVE
    assert parse_interview_answer("Observed") is EvidenceStatus.POSITIVE
    assert parse_interview_answer("no") is EvidenceStatus.NEGATIVE
    assert parse_interview_answer("not observed") is EvidenceStatus.NEGATIVE
    assert parse_interview_answer("I don't know") is EvidenceStatus.UNKNOWN
    assert parse_interview_answer("not visible in our telemetry") is EvidenceStatus.NOT_VISIBLE
    assert parse_interview_answer("maybe later") is None


def test_engine_resets_stale_interview_to_original_retrieval_evidence():
    engine = object.__new__(AttributionEngine)
    engine.rescore = lambda state: state
    state = SimpleNamespace(
        interview_revision=1,
        interview_complete=True,
        interview_stop_reason="question_limit",
        current_question=SimpleNamespace(),
        ledger=SimpleNamespace(question_answers=[SimpleNamespace(technique_id="T2")]),
        retrieval=SimpleNamespace(
            observed_techniques={"T1": 0.8}, technique_observations={"T1": ("obs-1",)}
        ),
        context=SimpleNamespace(
            observed_techniques={"T1": 0.8, "T2": 1.0},
            technique_observations={"T1": ("obs-1",), "T2": ("answer-1",)},
            negative_techniques={"T3"},
        ),
    )
    assert engine.ensure_interview_current(state)
    assert state.ledger.question_answers == []
    assert state.context.observed_techniques == {"T1": 0.8}
    assert state.context.technique_observations == {"T1": ("obs-1",)}
    assert state.context.negative_techniques == set()
    assert state.current_question is None
    assert state.interview_complete is False
    assert state.interview_stop_reason is None


def test_engine_builds_interview_weights_from_fused_candidates():
    engine = object.__new__(AttributionEngine)
    state = SimpleNamespace(
        fusion=SimpleNamespace(
            entries=(
                SimpleNamespace(group_id="G1", relative_score=100.0),
                SimpleNamespace(group_id="G2", relative_score=50.0),
            )
        ),
        bayes_posterior={"G1": 0.5, "G2": 0.5},
    )
    assert engine.interview_hypothesis_weights(state) == {"G1": 0.8, "G2": 0.2}


def test_engine_marks_interview_complete_when_stopping_rule_is_met(synthetic_kb):
    engine = object.__new__(AttributionEngine)
    engine.kb = synthetic_kb
    engine.settings = SimpleNamespace(nb_alpha=0.6, nb_beta=0.05, max_questions=10)
    engine.llm = SimpleNamespace()
    state = SimpleNamespace(
        bayes_posterior={"G1": 0.7, "G2": 0.2, "G3": 0.1},
        ledger=SimpleNamespace(question_answers=[]),
        context=SimpleNamespace(
            observed_techniques={"T1": 1.0, "T2": 1.0, "T3": 1.0, "T4": 1.0}, narrative=""
        ),
        current_question=None,
        interview_complete=False,
    )
    assert engine.next_question(state) is None
    assert state.interview_complete is True


def test_select_question_locally_uses_top_priority_lead_and_default_phrasing():
    from apt_classifier.questions import QuestionCandidate, select_question_locally

    def candidate(tid, priority):
        return QuestionCandidate(
            technique_id=tid,
            technique_name=tid,
            information_gain=0.1,
            probability_yes=0.4,
            default_question=f"Did you find {tid}?",
            profile_support=0.5,
            supporting_group_count=2,
            candidate_group_count=4,
            investigation_priority=priority,
            hypothesis="h",
            investigation_steps=["look"],
        )

    assert select_question_locally([]) is None
    chosen = select_question_locally([candidate("T1003", 0.9), candidate("T1059", 0.3)])
    assert chosen is not None
    assert chosen.technique_id == "T1003"
    assert chosen.question == "Did you find T1003?"
    assert chosen.investigation_steps == ["look"]
