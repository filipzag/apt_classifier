from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.evidence.models import (
    EvidenceLedger,
    Observation,
    Provenance,
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueRetrievalEvidence,
    TechniqueReviewAction,
    TechniqueReviewDecision,
)
from apt_classifier.platforms import set_ledger_platforms
from apt_classifier.retrieve import (
    LexicalTechniqueIndex,
    _specific_behavior_results,
    apply_candidate_decisions,
    build_classifier_context,
    retrieve_techniques,
)
from apt_classifier.vectorstore import VectorHit


class FakeStore:
    def query_techniques(self, texts, *, n_results):
        assert n_results >= 10
        return [
            [
                VectorHit(
                    "v1",
                    "Adversaries may execute PowerShell commands.",
                    {
                        "technique_id": "T1059.001",
                        "name": "PowerShell",
                        "parent_id": "T1059",
                        "field": "description",
                    },
                    0.2,
                    0.8,
                ),
                VectorHit(
                    "noise",
                    "Low confidence text",
                    {"technique_id": "T0000", "name": "Noise", "parent_id": ""},
                    0.8,
                    0.2,
                ),
            ]
            for _ in texts
        ]

    def query_group_profile(self, texts, *, document_type, n_results):
        if not texts:
            return []
        score = 0.7 if document_type == "technique_profile" else 0.6
        return [
            VectorHit(
                f"G1:{document_type}",
                "profile",
                {"group_id": "G1", "document_type": document_type},
                1 - score,
                score,
            )
        ]

    def query_group_behaviors(self, texts, *, n_results):
        return [
            [
                VectorHit(
                    "G1:actor-behavior:T1059.001:0000",
                    "Group One executed ContosoLoader with PowerShell -enc SQBFAFgA.",
                    {
                        "group_id": "G1",
                        "technique_id": "T1059.001",
                        "document_type": "actor_behavior",
                    },
                    0.1,
                    0.9,
                )
            ]
            for _ in texts
        ]


def test_retrieval_threshold_rollup_and_context():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Word spawned ContosoLoader with PowerShell -enc SQBFAFgA.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(
        ledger,
        FakeStore(),
        kb,
        threshold=0.45,
        retrieval_mode="legacy_dense",
        parent_policy="legacy_expand",
    )
    assert abs(result.observed_techniques["T1059.001"] - 0.9) < 1e-12
    # Legacy hierarchy roll-up remains visible in matches, but it is not a
    # second independently observed technique for downstream scoring.
    assert "T1059" not in result.observed_techniques
    assert "T0000" not in result.observed_techniques
    assert result.semantic_profile_scores == {"G1": 0.7}
    assert abs(result.specific_behavior_scores["G1"] - 0.9) < 1e-12
    assert result.specific_behavior_matches[0].technique_id == "T1059.001"
    context = build_classifier_context(kb, ledger, result)
    assert context.group_text_scores == {"G1": 0.6}
    assert abs(context.specific_behavior_scores["G1"] - 0.9) < 1e-12
    assert context.technique_observations["T1059.001"] == (ledger.observations[0].id,)


class EmptyGroupStore:
    def query_group_profile(self, texts, *, document_type, n_results):
        return []

    def query_group_behaviors(self, texts, *, n_results):
        return [[] for _ in texts]


class EmptyVectorStore(EmptyGroupStore):
    def query_techniques(self, texts, *, n_results):
        return [[] for _ in texts]


class PlatformStore(EmptyGroupStore):
    def query_techniques(self, texts, *, n_results):
        return [
            [
                VectorHit(
                    "linux",
                    "Linux behavior",
                    {"technique_id": "TLINUX", "name": "Linux Only", "platforms": "Linux"},
                    0.05,
                    0.95,
                ),
                VectorHit(
                    "windows",
                    "Windows behavior",
                    {"technique_id": "TWIN", "name": "Windows Only", "platforms": "Windows"},
                    0.30,
                    0.70,
                ),
                VectorHit(
                    "far",
                    "Distant Windows behavior",
                    {"technique_id": "TFAR", "name": "Far Match", "platforms": "Windows"},
                    0.50,
                    0.50,
                ),
            ]
            for _ in texts
        ]


class HybridStore(EmptyGroupStore):
    def __init__(self):
        self.fields = None

    def query_techniques(self, texts, *, n_results, fields=None):
        self.fields = fields
        return [
            [
                VectorHit(
                    "T1059.001:description:0000",
                    "Adversaries may abuse PowerShell commands and scripts.",
                    {
                        "technique_id": "T1059.001",
                        "name": "PowerShell",
                        "parent_id": "T1059",
                        "field": "description",
                        "platforms": "Windows",
                    },
                    0.18,
                    0.82,
                )
            ]
            for _ in texts
        ]


def test_hybrid_retrieval_fuses_dense_and_lexical_evidence_without_detection_text():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001",
                name="PowerShell",
                description="Adversaries may abuse PowerShell commands and scripts.",
                is_subtechnique=True,
                parent_id="T1059",
                platforms=["Windows"],
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    store = HybridStore()
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="hybrid")
    candidate = next(item for item in result.candidates if item.technique_id == "T1059.001")
    assert store.fields == ("name", "description", "procedure", "synthetic")
    assert {item.source for item in candidate.evidence} == {"dense", "lexical", "explicit_cue"}
    assert candidate.state.value == "auto_accepted"
    assert set(result.observed_techniques) == {"T1059.001"}


def test_rejected_mapping_is_removed_without_becoming_negative_incident_evidence():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001",
                name="PowerShell",
                description="Adversaries may abuse PowerShell commands and scripts.",
            )
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, HybridStore(), kb, retrieval_mode="hybrid")
    ledger.record_technique_review(
        TechniqueReviewDecision(
            observation_id=ledger.observations[0].id,
            technique_id="T1059.001",
            action=TechniqueReviewAction.REJECT,
        )
    )
    reviewed = apply_candidate_decisions(result, ledger, kb)
    context = build_classifier_context(kb, ledger, reviewed)
    assert reviewed.observed_techniques == {}
    assert context.negative_techniques == set()


def test_ancestor_compatible_policy_keeps_parent_as_relationship_not_second_feature():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Word spawned ContosoLoader with PowerShell -enc SQBFAFgA.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(
        ledger,
        FakeStore(),
        kb,
        threshold=0.45,
        parent_policy="ancestor_compatible",
    )
    assert set(result.observed_techniques) == {"T1059.001"}
    assert result.matches[0].parent_technique_id == "T1059"


def test_explicit_archive_utility_cue_marks_subtechnique_as_observed():
    kb = AttackKnowledgeBase(
        techniques={
            "T1560": Technique(id="T1560", name="Archive Collected Data"),
            "T1560.001": Technique(
                id="T1560.001",
                name="Archive via Utility",
                is_subtechnique=True,
                parent_id="T1560",
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1560.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="The attacker used the zip utility to archive collected data.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )

    class ArchiveStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v1",
                        "Archive via Utility",
                        {
                            "technique_id": "T1560.001",
                            "name": "Archive via Utility",
                            "parent_id": "T1560",
                            "field": "name",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.60,
                    )
                ]
            ]

    result = retrieve_techniques(ledger, ArchiveStore(), kb, retrieval_mode="legacy_dense")
    assert "T1560.001" in result.observed_techniques
    assert "T1560" not in result.observed_techniques
    explicit = next(match for match in result.matches if match.technique_id == "T1560.001")
    assert explicit.metadata["source"] == "explicit_technique_cue"
    assert explicit.observation_ids == [ledger.observations[0].id]


def test_windows_logs_filter_other_platforms_and_distant_vector_hits():
    kb = AttackKnowledgeBase(
        techniques={
            technique_id: Technique(id=technique_id, name=name)
            for technique_id, name in {
                "TLINUX": "Linux Only",
                "TWIN": "Windows Only",
                "TFAR": "Far Match",
            }.items()
        },
        groups={"G1": Group(id="G1", name="Group One")},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="A security event recorded a specific behavior.",
                kind="winlog",
                provenance=Provenance(source="winlog"),
            )
        ]
    )
    result = retrieve_techniques(ledger, PlatformStore(), kb)
    assert set(result.observed_techniques) == {"TWIN"}
    assert abs(result.observed_techniques["TWIN"] - 0.665) < 1e-12


class WindowsFirstStore(EmptyGroupStore):
    def query_techniques(self, texts, *, n_results):
        return [
            [
                VectorHit(
                    "win",
                    "Recurring job",
                    {"technique_id": "TWIN", "platforms": "Windows"},
                    0.05,
                    0.95,
                ),
                VectorHit(
                    "lin",
                    "Recurring job",
                    {"technique_id": "TLINUX", "platforms": "Linux"},
                    0.08,
                    0.92,
                ),
                VectorHit(
                    "cloud",
                    "Recurring job",
                    {"technique_id": "TCLOUD", "platforms": "IaaS,Windows"},
                    0.10,
                    0.90,
                ),
            ]
            for _ in texts
        ]


def _platform_kb() -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        techniques={
            "TWIN": Technique(id="TWIN", name="Windows Only", platforms=["Windows"]),
            "TLINUX": Technique(id="TLINUX", name="Linux Only", platforms=["Linux"]),
            "TCLOUD": Technique(
                id="TCLOUD", name="Cloud Or Windows", platforms=["IaaS", "Windows"]
            ),
            "T1053.005": Technique(id="T1053.005", name="Scheduled Task", platforms=["Windows"]),
        },
        groups={"G1": Group(id="G1", name="Group One")},
    )


def _ids(result) -> list[str]:
    return [c.technique_id for c in sorted(result.candidates, key=lambda c: c.fused_rank)]


def test_case_target_os_keeps_only_techniques_documented_for_it():
    kb = _platform_kb()
    ledger = EvidenceLedger(observations=[Observation(text="The operator ran a recurring job.")])
    assert _ids(retrieve_techniques(ledger, WindowsFirstStore(), kb, retrieval_mode="dense")) == [
        "TWIN",
        "TLINUX",
        "TCLOUD",
    ]
    set_ledger_platforms(ledger, ["Linux"])
    scoped = retrieve_techniques(ledger, WindowsFirstStore(), kb, retrieval_mode="dense")
    # the Windows-only technique leaves the pool; the cloud one is not an OS question
    assert _ids(scoped) == ["TLINUX", "TCLOUD"]
    assert scoped.diagnostics["target_platforms"] == ["Linux"]
    assert scoped.diagnostics["platform_scopes"] == {
        ledger.observations[0].id: {"platforms": ["Linux"], "source": "case"}
    }


def test_operating_system_named_in_the_observation_overrides_the_case():
    kb = _platform_kb()
    ledger = EvidenceLedger(
        observations=[Observation(text="On the Linux server the operator ran a recurring job.")]
    )
    set_ledger_platforms(ledger, ["Windows"])
    off = retrieve_techniques(ledger, WindowsFirstStore(), kb, retrieval_mode="dense")
    assert _ids(off) == ["TWIN", "TCLOUD"]
    named = retrieve_techniques(
        ledger, WindowsFirstStore(), kb, retrieval_mode="dense", platform_from_text=True
    )
    assert _ids(named) == ["TLINUX", "TCLOUD"]
    assert named.diagnostics["platform_scopes"][ledger.observations[0].id]["source"] == "text"


def test_target_os_also_filters_cue_candidates():
    kb = _platform_kb()
    ledger = EvidenceLedger(
        observations=[Observation(text="The operator created a scheduled task for persistence.")]
    )
    plain = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="dense")
    assert "T1053.005" in _ids(plain)
    set_ledger_platforms(ledger, ["Linux"])
    scoped = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="dense")
    assert "T1053.005" not in _ids(scoped)
    assert scoped.diagnostics["platform_dropped_candidates"] == 1


class AggregationStore(EmptyGroupStore):
    def query_techniques(self, texts, *, n_results):
        scores = (0.9, 0.5) if len(texts) == 2 else (0.8,)
        return [
            [
                VectorHit(
                    f"hit-{index}",
                    "Matched behavior",
                    {"technique_id": "T1", "name": "Technique One", "platforms": "Windows"},
                    1 - score,
                    score,
                )
            ]
            for index, score in enumerate(scores)
        ]


def test_aggregation_blends_independent_scores_without_occurrence_boost():
    kb = AttackKnowledgeBase(
        techniques={"T1": Technique(id="T1", name="Technique One")},
        groups={"G1": Group(id="G1", name="Group One")},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(text="First observed behavior.", provenance=Provenance(source="free_text")),
            Observation(
                text="Second observed behavior.", provenance=Provenance(source="free_text")
            ),
        ]
    )
    result = retrieve_techniques(ledger, AggregationStore(), kb, retrieval_mode="legacy_dense")
    # The weaker second retrieval is retained for analyst review but does not
    # vote below the calibrated acceptance floor.
    assert abs(result.observed_techniques["T1"] - 0.9) < 1e-12
    repeated = EvidenceLedger(
        observations=[
            Observation(
                text="Repeated observed behavior.",
                provenance=Provenance(source="free_text", occurrence_count=20),
            )
        ]
    )
    repeated_result = retrieve_techniques(
        repeated, AggregationStore(), kb, retrieval_mode="legacy_dense"
    )
    assert repeated_result.observed_techniques["T1"] == 0.8


def test_specific_behavior_abstains_for_crowded_generic_matches():
    observation = Observation(
        text="A scheduled task was created for persistence.",
        provenance=Provenance(source="free_text"),
    )
    hits = [
        VectorHit(
            f"G{index}:actor-behavior:T1053.005:0000",
            "The group created a scheduled task for persistence.",
            {
                "group_id": f"G{index}",
                "technique_id": "T1053.005",
                "document_type": "actor_behavior",
            },
            1 - score,
            score,
        )
        for index, score in enumerate((0.95, 0.94, 0.93, 0.92), 1)
    ]
    scores, signals, matches = _specific_behavior_results([observation], [hits], threshold=0.55)
    assert scores == {}
    assert signals == {}
    assert matches == []


def test_specific_behavior_rejects_semantically_distinct_but_generic_match():
    observation = Observation(
        text="PowerShell executed an encoded command.",
        provenance=Provenance(source="free_text"),
    )
    hits = [
        VectorHit(
            f"G{index}:actor-behavior:T1059.001:0000",
            document,
            {
                "group_id": f"G{index}",
                "technique_id": "T1059.001",
                "document_type": "actor_behavior",
            },
            1 - score,
            score,
        )
        for index, (score, document) in enumerate(
            (
                (0.91, "The group has executed encoded PowerShell commands."),
                (0.71, "The group used PowerShell."),
                (0.68, "The group ran a script."),
                (0.64, "The group executed commands."),
            ),
            1,
        )
    ]
    scores, _, matches = _specific_behavior_results([observation], [hits], threshold=0.55)
    assert scores == {}
    assert matches == []


def test_specific_behavior_keeps_distinct_match():
    observation = Observation(
        text="Rundll32 invoked MiniDump against LSASS memory.",
        provenance=Provenance(source="free_text"),
    )
    hits = [
        VectorHit(
            f"G{index}:actor-behavior:T1003.001:0000",
            document,
            {
                "group_id": f"G{index}",
                "technique_id": "T1003.001",
                "document_type": "actor_behavior",
            },
            1 - score,
            score,
        )
        for index, (score, document) in enumerate(
            (
                (0.92, "The group used rundll32 MiniDump to dump LSASS memory."),
                (0.72, "The group accessed credentials."),
                (0.69, "The group dumped process memory."),
                (0.66, "The group collected passwords."),
            ),
            1,
        )
    ]
    scores, signals, matches = _specific_behavior_results([observation], [hits], threshold=0.55)
    assert set(scores) == {"G1"}
    assert signals["G1"] == ("G1:actor-behavior:T1003.001:0000",)
    assert matches[0].group_id == "G1"


def test_name_token_bonus_returns_zero_for_no_match():
    from types import SimpleNamespace

    from apt_classifier.retrieve import _name_token_bonus

    kb = SimpleNamespace(
        techniques={
            "T1059.001": SimpleNamespace(name="PowerShell"),
        }
    )
    assert _name_token_bonus(kb, "T1059.001", "no relevant terms here") == 0.0


def test_name_token_bonus_requires_two_tokens():
    from types import SimpleNamespace

    from apt_classifier.retrieve import _name_token_bonus

    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="Software"),
        }
    )
    assert _name_token_bonus(kb, "T1", "software was used") == 0.0


def test_name_token_bonus_returns_positive_for_match():
    from types import SimpleNamespace

    from apt_classifier.retrieve import _name_token_bonus

    kb = SimpleNamespace(
        techniques={
            "T1059.001": SimpleNamespace(name="PowerShell Command Interpreter"),
        }
    )
    bonus = _name_token_bonus(kb, "T1059.001", "powershell command was executed")
    assert bonus > 0.0


def test_name_token_bonus_gives_extra_for_exact_tool_name():
    from types import SimpleNamespace

    from apt_classifier.retrieve import _name_token_bonus

    kb = SimpleNamespace(
        techniques={
            "T1059.001": SimpleNamespace(name="PowerShell Command"),
        }
    )
    bonus = _name_token_bonus(kb, "T1059.001", "powershell was used")
    assert bonus >= 0.015


def test_benign_indicator_keeps_similarity_only_candidate_visible_but_unaccepted():
    kb = AttackKnowledgeBase(
        techniques={
            "T1098.001": Technique(
                id="T1098.001",
                name="Additional Cloud Credentials",
                is_subtechnique=True,
                parent_id="T1098",
                platforms=["Windows"],
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1098.001"])},
    )

    class StrongStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v1",
                        "Adversaries may reset account credentials.",
                        {
                            "technique_id": "T1098.001",
                            "name": "Additional Cloud Credentials",
                            "parent_id": "T1098",
                            "field": "description",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.92,
                    )
                ]
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="A user reset their own password through the self-service portal.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, StrongStore(), kb, retrieval_mode="hybrid")
    candidate = next(c for c in result.candidates if c.technique_id == "T1098.001")
    assert candidate.state is TechniqueCandidateState.NEEDS_REVIEW
    assert "benign-context" in candidate.state_reason
    assert result.observed_techniques == {}
    assert any(m.technique_id == "T1098.001" for m in result.matches)


def test_benign_phrase_does_not_globally_abstain_from_supported_cue():
    """A broad benign phrase cannot suppress independently supported evidence."""
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="The administrator logged in and used PowerShell -enc to run a command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="hybrid")
    assert "T1059.001" in result.observed_techniques


def test_explicit_cue_boosts_existing_candidate_but_does_not_inject():
    """Explicit cues boost candidates already retrieved by dense/lexical.
    Injected candidates (not retrieved) are needs_review, not auto_accepted."""
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T0000": Technique(id="T0000", name="Unrelated Technique"),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )

    class PowerShellStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v1",
                        "PowerShell commands and scripts",
                        {
                            "technique_id": "T1059.001",
                            "name": "PowerShell",
                            "parent_id": "T1059",
                            "field": "description",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.70,
                    )
                ]
            ]

    result = retrieve_techniques(ledger, PowerShellStore(), kb, retrieval_mode="hybrid")
    # T1059.001 was retrieved by dense, so the cue boosts and auto-accepts it
    assert "T1059.001" in result.observed_techniques
    # The score should be boosted but not forced to 1.0
    score = result.observed_techniques["T1059.001"]
    assert 0.7 < score <= 0.9
    # T0000 was not retrieved, so it should not appear
    assert "T0000" not in result.observed_techniques


def test_explicit_cue_only_candidate_requires_review():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="hybrid")
    candidate = next(item for item in result.candidates if item.technique_id == "T1059.001")
    # The lexical index independently corroborates this explicit cue.
    assert candidate.state.value == "auto_accepted"
    assert candidate.evidence_for("explicit_cue") is not None
    assert "T1059.001" in result.observed_techniques


def test_hierarchy_expansion_adds_parent_when_multiple_children_retrieved():
    kb = AttackKnowledgeBase(
        techniques={
            "T1055": Technique(id="T1055", name="Process Injection"),
            "T1055.001": Technique(
                id="T1055.001", name="DLL Injection", is_subtechnique=True, parent_id="T1055"
            ),
            "T1055.003": Technique(
                id="T1055.003",
                name="Thread Execution Hijacking",
                is_subtechnique=True,
                parent_id="T1055",
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1055"])},
    )

    class MultiChildStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v1",
                        "DLL injection into a process",
                        {
                            "technique_id": "T1055.001",
                            "name": "DLL Injection",
                            "parent_id": "T1055",
                            "field": "description",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.90,
                    ),
                    VectorHit(
                        "v2",
                        "Thread execution hijacking",
                        {
                            "technique_id": "T1055.003",
                            "name": "Thread Execution Hijacking",
                            "parent_id": "T1055",
                            "field": "description",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.85,
                    ),
                ]
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="A DLL was loaded into another process and its thread execution was redirected.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(
        ledger, MultiChildStore(), kb, retrieval_mode="hybrid", threshold=0.45
    )
    candidate_ids = {c.technique_id for c in result.candidates}
    assert "T1055" in candidate_ids
    parent = next(c for c in result.candidates if c.technique_id == "T1055")
    assert parent.evidence_for("dense") is None
    assert parent.state is TechniqueCandidateState.NEEDS_REVIEW
    # A derived parent never votes in attribution and never displaces the
    # directly observed child mapping.
    assert "T1055" not in result.observed_techniques
    # Exactly one pure-retrieval mapping is auto-accepted per atomic observation.
    assert len(result.observed_techniques) == 1
    assert next(iter(result.observed_techniques)).startswith("T1055.")
    child_scores = [m.score for m in result.matches if m.technique_id.startswith("T1055.")]
    parent_match = next(m for m in result.matches if m.technique_id == "T1055")
    assert max(child_scores) > parent_match.score
    assert parent_match.metadata["mapping_state"] == "needs_review"


def test_lexical_only_mode_accepts_strong_lexical_candidates():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001",
                name="PowerShell",
                description="Adversaries may abuse PowerShell commands.",
                is_subtechnique=True,
                parent_id="T1059",
                platforms=["Windows"],
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="lexical")
    assert "T1059.001" in result.observed_techniques


def test_cross_validation_bonus_boosts_dense_lexical_agreement():
    from types import SimpleNamespace

    from apt_classifier.retrieve import _build_observation_candidates

    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="Technique One", platforms=["Windows"]),
        }
    )
    dense = {
        "T1": VectorHit(
            "d1", "dense match", {"technique_id": "T1", "field": "description"}, 0.1, 0.80
        ),
    }
    lexical = {
        "T1": VectorHit(
            "l1", "lexical match", {"technique_id": "T1", "field": "description"}, 0.1, 5.0
        ),
    }
    obs = SimpleNamespace(
        id="obs-1",
        text="technique one was observed",
        metadata={},
        provenance=SimpleNamespace(source="free_text"),
        kind="free_text",
    )
    candidates = _build_observation_candidates(obs, kb, dense, lexical, candidate_pool=12)
    assert len(candidates) > 0
    assert candidates[0].state.value == "needs_review"


def test_sparse_observation_is_never_accepted_on_similarity_alone():
    kb = AttackKnowledgeBase(
        techniques={
            "T1016": Technique(
                id="T1016", name="System Network Configuration Discovery", platforms=["Windows"]
            )
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1016"])},
    )

    class StrongStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v1",
                        "Adversaries may look for details about the network configuration.",
                        {
                            "technique_id": "T1016",
                            "name": "System Network Configuration Discovery",
                            "parent_id": "",
                            "field": "description",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.95,
                    )
                ]
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(text="source_ip: 10.77.10.35", provenance=Provenance(source="free_text"))
        ]
    )
    result = retrieve_techniques(ledger, StrongStore(), kb, retrieval_mode="hybrid")
    candidate = next(c for c in result.candidates if c.technique_id == "T1016")
    assert candidate.state is TechniqueCandidateState.NEEDS_REVIEW
    assert "too sparse" in candidate.state_reason
    assert result.observed_techniques == {}


def test_lone_procedure_hit_at_rank_one_is_accepted_by_the_rule_gate():
    kb = AttackKnowledgeBase(
        techniques={
            "T1047": Technique(
                id="T1047", name="Windows Management Instrumentation", platforms=["Windows"]
            )
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1047"])},
    )

    class ProcedureStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "p1",
                        "the adversary has utilized WMIC to log into remote hosts.",
                        {
                            "technique_id": "T1047",
                            "name": "Windows Management Instrumentation",
                            "parent_id": "",
                            "field": "procedure",
                            "platforms": "Windows",
                        },
                        0.1,
                        0.90,
                    )
                ]
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="robocopy.exe D:\\ERP\\Data \\\\ERP-01\\Backups /MIR (Robocopy.exe, run by admin on ERP-01).",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    result = retrieve_techniques(ledger, ProcedureStore(), kb, retrieval_mode="hybrid")
    candidate = next(c for c in result.candidates if c.technique_id == "T1047")
    # The former procedure-only exclusion promoted lower-ranked candidates instead and
    # measured worse on the held-out public benchmark (F1 0.232 vs 0.240); a strong
    # rank-1 procedure hit is now accepted by the rule gate. Discounting command-like
    # procedure matches is the job of the calibrated acceptance model.
    assert candidate.state is TechniqueCandidateState.AUTO_ACCEPTED
    assert candidate.evidence_for("dense").field == "procedure"
    assert set(result.observed_techniques) == {"T1047"}


def test_cues_disabled_removes_regex_evidence_and_injection():
    """``cues_enabled=False`` must run pure retrieval: no explicit-cue or structured-signal
    evidence, no cue-only injected candidates, and the diagnostics record the switch."""
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059.001"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell executed an encoded command.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    with_cues = retrieve_techniques(ledger, EmptyVectorStore(), kb, retrieval_mode="hybrid")
    assert any(item.evidence_for("explicit_cue") for item in with_cues.candidates)
    without = retrieve_techniques(
        ledger, EmptyVectorStore(), kb, retrieval_mode="hybrid", cues_enabled=False
    )
    assert without.diagnostics["cues_enabled"] is False
    assert not any(item.evidence_for("explicit_cue") for item in without.candidates)
    assert not any(item.evidence_for("structured_signal") for item in without.candidates)
    # Only lexical retrieval remains for this store, and lexical-only evidence is never
    # a primary acceptance on its own.
    assert "T1059.001" not in without.observed_techniques


def test_lexical_index_includes_anonymized_procedure_sentences():
    kb = AttackKnowledgeBase(
        techniques={
            "T1047": Technique(id="T1047", name="Windows Management Instrumentation"),
        },
        groups={
            "G1": Group(
                id="G1",
                name="Group One",
                technique_ids=["T1047"],
                technique_behaviors={"T1047": ["Group One used wmic to run robocopy on hosts."]},
            )
        },
    )
    from apt_classifier.retrieve import LexicalTechniqueIndex

    index = LexicalTechniqueIndex.from_kb(kb)
    procedure_docs = [doc for doc in index.documents if doc.field == "procedure"]
    assert procedure_docs and "the adversary" in procedure_docs[0].text.lower()
    assert "group one" not in procedure_docs[0].text.lower()
    hits = index.query("robocopy launched through wmic", n_results=3)
    assert hits and hits[0].metadata["technique_id"] == "T1047"
    assert hits[0].metadata["field"] == "procedure"
    legacy = LexicalTechniqueIndex.from_kb(kb, fields=("name", "description"))
    assert not [doc for doc in legacy.documents if doc.field == "procedure"]


class SegmentStore(EmptyGroupStore):
    """Returns a different technique for each sentence so segmentation is observable."""

    def __init__(self):
        self.queries: list[list[str]] = []

    def query_techniques(self, texts, *, n_results, **kwargs):
        self.queries.append(list(texts))
        hits = []
        for text in texts:
            if "powershell" in text.lower():
                tid, name = "T1059.001", "PowerShell"
            elif "lsass" in text.lower():
                tid, name = "T1003.001", "LSASS Memory"
            else:
                hits.append([])
                continue
            hits.append(
                [
                    VectorHit(
                        f"{tid}:description:0000",
                        f"{name} behaviour",
                        {"technique_id": tid, "name": name, "field": "description"},
                        0.15,
                        0.85,
                    )
                ]
            )
        return hits


def test_long_observations_are_retrieved_per_sentence_and_merged():
    from apt_classifier.retrieve import _observation_segments

    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={},
    )
    first = (
        "The operators first established a foothold on the workstation and then launched "
        "PowerShell with a long series of obfuscated arguments that decoded a second stage "
        "loader from a remote host over HTTPS during the night."
    )
    second = (
        "Several hours later the same account created a memory dump of the LSASS process "
        "on the domain controller and staged the file under a temporary directory for later "
        "collection by the operators."
    )
    text = f"{first} {second}"
    assert len(_observation_segments(text)) == 2
    assert _observation_segments("short PowerShell note") == ["short PowerShell note"]
    store = SegmentStore()
    ledger = EvidenceLedger(
        observations=[Observation(text=text, provenance=Provenance(source="free_text"))]
    )
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="dense")
    assert store.queries == [[first, second]]
    ids = {candidate.technique_id for candidate in result.candidates}
    assert {"T1059.001", "T1003.001"} <= ids
    assert result.diagnostics["segmented_observations"] == 1
    whole = retrieve_techniques(
        EvidenceLedger(
            observations=[Observation(text=first, provenance=Provenance(source="free_text"))]
        ),
        SegmentStore(),
        kb,
        retrieval_mode="dense",
    )
    assert whole.diagnostics["segmented_observations"] == 0


class AcceptEverythingAbove:
    """Stub acceptance model: accept when the candidate's dense score clears a bar."""

    threshold = 0.5

    def __init__(self, bar: float):
        self.bar = bar
        self.seen: list[dict[str, float]] = []

    def accepts(self, features):
        self.seen.append(features)
        return features["dense_score"] >= self.bar


def test_acceptance_model_replaces_one_primary_rule():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={},
    )

    class TwoHitStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "a",
                        "x",
                        {"technique_id": "T1059.001", "name": "PowerShell", "field": "description"},
                        0.1,
                        0.9,
                    ),
                    VectorHit(
                        "b",
                        "y",
                        {
                            "technique_id": "T1003.001",
                            "name": "LSASS Memory",
                            "field": "description",
                        },
                        0.18,
                        0.82,
                    ),
                ]
                for _ in texts
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="The loader ran scripts and harvested credentials from security processes.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    rule_based = retrieve_techniques(ledger, TwoHitStore(), kb, retrieval_mode="dense")
    assert set(rule_based.observed_techniques) == {"T1059.001"}
    model = AcceptEverythingAbove(0.8)
    learned = retrieve_techniques(
        ledger, TwoHitStore(), kb, retrieval_mode="dense", acceptance_model=model
    )
    assert set(learned.observed_techniques) == {"T1059.001", "T1003.001"}
    assert learned.diagnostics["acceptance_model_active"] is True
    assert all("fused_margin_to_top" in row for row in model.seen)
    reasons = {c.technique_id: c.state_reason for c in learned.candidates}
    assert "calibrated acceptance model" in reasons["T1003.001"]
    strict = retrieve_techniques(
        ledger,
        TwoHitStore(),
        kb,
        retrieval_mode="dense",
        acceptance_model=AcceptEverythingAbove(0.95),
    )
    assert strict.observed_techniques == {}


def test_inverted_lexical_index_matches_brute_force_bm25():
    import math
    from collections import Counter

    from apt_classifier.retrieve import LexicalTechniqueIndex, _lexical_tokens

    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001",
                name="PowerShell",
                description="Adversaries may abuse PowerShell commands and scripts. PowerShell is powerful.",
            ),
            "T1003.001": Technique(
                id="T1003.001",
                name="LSASS Memory",
                description="Adversaries may attempt to access credential material stored in LSASS memory.",
            ),
            "T1021.002": Technique(
                id="T1021.002",
                name="SMB/Windows Admin Shares",
                description="Adversaries may use valid accounts to interact with remote network shares using SMB.",
            ),
        },
        groups={
            "G1": Group(
                id="G1",
                name="Group One",
                technique_ids=["T1059.001"],
                technique_behaviors={
                    "T1059.001": ["Group One ran PowerShell scripts to stage tools."]
                },
            )
        },
    )
    index = LexicalTechniqueIndex.from_kb(kb)
    assert index.postings and len(index.norms) == len(index.documents)

    def brute_force(text: str) -> dict[str, float]:
        query_counts = Counter(_lexical_tokens(text))
        n = len(index.documents)
        out: dict[str, float] = {}
        for document in index.documents:
            counts = Counter(document.tokens)
            norm = 1.2 * (0.25 + 0.75 * len(document.tokens) / max(index.average_length, 1.0))
            score = 0.0
            for token, qf in query_counts.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = index.document_frequencies.get(token, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.2 / (tf + norm) * (1.0 + math.log(qf))
            if score > 0:
                out[document.id] = score
        return out

    for text in (
        "powershell scripts executed powershell",
        "credential material from lsass memory",
        "remote shares over smb with valid accounts",
        "nothing relevant here",
    ):
        hits = index.query(text, n_results=50)
        expected = brute_force(text)
        assert {hit.id: hit.similarity for hit in hits} == pytest.approx(expected)
        assert [hit.id for hit in hits] == sorted(expected, key=lambda i: (-expected[i], i))


def test_hierarchy_expansion_preserves_reranker_order():
    from apt_classifier.retrieve import _expand_hierarchy_candidates

    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059.003": Technique(
                id="T1059.003",
                name="Windows Command Shell",
                is_subtechnique=True,
                parent_id="T1059",
            ),
        },
        groups={},
    )

    def candidate(tid, fused_rank, fused_score, reranker_rank=None):
        evidence = []
        if reranker_rank is not None:
            evidence.append(
                TechniqueRetrievalEvidence(
                    source="reranker", text="t", score=1.0 / reranker_rank, rank=reranker_rank
                )
            )
        return TechniqueCandidate(
            observation_id="obs-1",
            technique_id=tid,
            technique_name=tid,
            parent_technique_id="T1059",
            is_subtechnique=True,
            fused_score=fused_score,
            fused_rank=fused_rank,
            evidence=evidence,
        )

    observation = SimpleNamespace(id="obs-1", text="x")
    # Reranker put the fused runner-up first.
    reranked = [candidate("T1059.003", 1, 0.010, 1), candidate("T1059.001", 2, 0.020, 2)]
    expanded = _expand_hierarchy_candidates(observation, kb, reranked)
    order = [item.technique_id for item in expanded]
    assert order[:2] == ["T1059.003", "T1059.001"]
    assert [item.fused_rank for item in expanded] == [1, 2, 3]
    assert order[2] == "T1059"  # derived parent sorts after reranked children
    # Without reranker evidence the fused score still decides.
    plain = [candidate("T1059.003", 1, 0.010), candidate("T1059.001", 2, 0.020)]
    assert [c.technique_id for c in _expand_hierarchy_candidates(observation, kb, plain)][:2] == [
        "T1059.001",
        "T1059.003",
    ]


def test_parent_policy_reaches_the_classifier_context():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
        },
        groups={"G1": Group(id="G1", name="Group One", technique_ids=["T1059"])},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Word spawned ContosoLoader with PowerShell -enc SQBFAFgA.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    for policy in ("ancestor_compatible", "exact_only"):
        result = retrieve_techniques(ledger, FakeStore(), kb, parent_policy=policy)
        context = build_classifier_context(kb, ledger, result)
        assert context.parent_policy == policy


def _two_sentence_kb():
    return AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={},
    )


LONG_FIRST = (
    "The operators first established a foothold on the workstation and then launched "
    "PowerShell with a long series of obfuscated arguments that decoded a second stage "
    "loader from a remote host over HTTPS during the night."
)
LONG_SECOND = (
    "Several hours later the same account created a memory dump of the LSASS process "
    "on the domain controller and staged the file under a temporary directory for later "
    "collection by the operators."
)


def test_segmented_observation_judges_each_sentence_on_its_own_pool():
    kb = _two_sentence_kb()
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text=f"{LONG_FIRST} {LONG_SECOND}", provenance=Provenance(source="free_text")
            )
        ]
    )
    result = retrieve_techniques(
        ledger, SegmentStore(), kb, retrieval_mode="dense", segment_acceptance=True
    )
    by_id = {c.technique_id: c for c in result.candidates}
    # Each sentence contributed one candidate tagged with its segment and a segment rank of 1.
    assert by_id["T1059.001"].evidence[0].metadata["segment_index"] == 0
    assert by_id["T1003.001"].evidence[0].metadata["segment_index"] == 1
    assert by_id["T1003.001"].evidence[0].metadata["segment_rank"] == 1
    assert "LSASS" in by_id["T1003.001"].evidence[0].metadata["segment_text"]
    # Under the rule gate both sentences' rank-1 candidates are accepted (one primary per
    # sentence), where a single merged pool would have accepted only the global rank 1.
    assert set(result.observed_techniques) == {"T1059.001", "T1003.001"}
    # The merged observation-level order is still reported (ranks 1 and 2 across sentences).
    assert {by_id["T1059.001"].fused_rank, by_id["T1003.001"].fused_rank} == {1, 2}


def test_segment_groups_attach_derived_parents_to_their_children():
    from apt_classifier.retrieve import _segment_groups

    def cand(tid, seg, parent=None, hierarchy=False):
        evidence = (
            [TechniqueRetrievalEvidence(source="hierarchy_parent", score=0.5, rank=1)]
            if hierarchy
            else [
                TechniqueRetrievalEvidence(
                    source="dense", score=0.8, rank=1, metadata={"segment_index": seg}
                )
            ]
        )
        return TechniqueCandidate(
            observation_id="o",
            technique_id=tid,
            technique_name=tid,
            parent_technique_id=parent,
            fused_score=0.01,
            fused_rank=1,
            evidence=evidence,
        )

    groups = _segment_groups(
        [
            cand("T1003.001", 1, "T1003"),
            cand("T1003.002", 1, "T1003"),
            cand("T1059.001", 0, "T1059"),
            cand("T1003", None, hierarchy=True),
        ]
    )
    assert {c.technique_id for c in groups[1]} == {"T1003.001", "T1003.002", "T1003"}
    assert {c.technique_id for c in groups[0]} == {"T1059.001"}


def test_hierarchy_consistency_accepts_parent_over_indistinguishable_siblings():
    from apt_classifier.retrieve import _apply_hierarchy_consistency

    def child(tid, dense, state=TechniqueCandidateState.NEEDS_REVIEW, cue=False):
        evidence = [
            TechniqueRetrievalEvidence(source="dense", field="description", score=dense, rank=1)
        ]
        if cue:
            evidence.append(TechniqueRetrievalEvidence(source="explicit_cue", score=0.85, rank=1))
        return TechniqueCandidate(
            observation_id="o",
            technique_id=tid,
            technique_name=tid,
            parent_technique_id="T1003",
            is_subtechnique=True,
            fused_score=0.02,
            fused_rank=1,
            evidence=evidence,
            state=state,
        )

    parent = TechniqueCandidate(
        observation_id="o",
        technique_id="T1003",
        technique_name="OS Credential Dumping",
        fused_score=0.005,
        fused_rank=3,
        evidence=[TechniqueRetrievalEvidence(source="hierarchy_parent", score=0.5, rank=1)],
    )
    # Two plausible siblings, neither accepted -> parent accepted, children stay in review.
    out = _apply_hierarchy_consistency(
        [child("T1003.001", 0.66), child("T1003.003", 0.64), parent],
        dense_floor=0.60,
        relative_margin=0.08,
        fill_unaccepted=True,
    )
    states = {c.technique_id: c.state for c in out}
    assert states["T1003"] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states["T1003.001"] is TechniqueCandidateState.NEEDS_REVIEW
    # One sibling accepted but not separable (lead 0.02 < 0.08) -> swapped for the parent.
    out = _apply_hierarchy_consistency(
        [
            child("T1003.001", 0.78, TechniqueCandidateState.AUTO_ACCEPTED),
            child("T1003.003", 0.76),
            parent,
        ],
        dense_floor=0.60,
        relative_margin=0.08,
    )
    states = {c.technique_id: c.state for c in out}
    assert states["T1003"] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states["T1003.001"] is TechniqueCandidateState.NEEDS_REVIEW
    # Clear leader (lead 0.20) -> untouched.
    out = _apply_hierarchy_consistency(
        [
            child("T1003.001", 0.90, TechniqueCandidateState.AUTO_ACCEPTED),
            child("T1003.003", 0.70),
            parent,
        ],
        dense_floor=0.60,
        relative_margin=0.08,
    )
    states = {c.technique_id: c.state for c in out}
    assert states["T1003.001"] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states["T1003"] is TechniqueCandidateState.NEEDS_REVIEW
    # A cue-backed accepted child is a deliberate choice and is never replaced.
    out = _apply_hierarchy_consistency(
        [
            child("T1003.001", 0.70, TechniqueCandidateState.AUTO_ACCEPTED, cue=True),
            child("T1003.003", 0.69),
            parent,
        ],
        dense_floor=0.60,
        relative_margin=0.08,
    )
    states = {c.technique_id: c.state for c in out}
    assert states["T1003.001"] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states["T1003"] is TechniqueCandidateState.NEEDS_REVIEW


def test_corroboration_bonus_rewards_independent_observations():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True)
        },
        groups={},
    )
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="PowerShell launched an encoded stager on host A",
                provenance=Provenance(source="free_text"),
            ),
            Observation(
                text="PowerShell launched an encoded stager on host B",
                provenance=Provenance(source="free_text"),
            ),
        ]
    )
    plain = retrieve_techniques(ledger, SegmentStore(), kb, retrieval_mode="dense")
    boosted = retrieve_techniques(
        ledger, SegmentStore(), kb, retrieval_mode="dense", corroboration_bonus=0.1
    )
    assert len(plain.technique_observations["T1059.001"]) == 2
    assert boosted.observed_techniques["T1059.001"] == pytest.approx(
        min(1.0, plain.observed_techniques["T1059.001"] + 0.1)
    )
    single = retrieve_techniques(
        EvidenceLedger(observations=[ledger.observations[0]]),
        SegmentStore(),
        kb,
        retrieval_mode="dense",
        corroboration_bonus=0.1,
    )
    assert single.observed_techniques["T1059.001"] == pytest.approx(
        plain.observed_techniques["T1059.001"]
    )


def test_include_synthetic_controls_dense_and_lexical_fields():
    from apt_classifier.retrieve import LexicalTechniqueIndex

    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True)
        },
        groups={},
    )
    synthetic = {"T1059.001": ["An encoded stager ran through powershell.exe on the endpoint."]}
    with_synthetic = LexicalTechniqueIndex.from_kb(kb, synthetic=synthetic)
    assert any(doc.field == "synthetic" for doc in with_synthetic.documents)
    hits = with_synthetic.query("encoded stager powershell endpoint", n_results=3)
    assert hits and hits[0].metadata["field"] == "synthetic"
    without = LexicalTechniqueIndex.from_kb(kb, fields=("name", "description"), synthetic=synthetic)
    assert not any(doc.field == "synthetic" for doc in without.documents)

    class RecordingStore(EmptyGroupStore):
        fields = None

        def query_techniques(self, texts, *, n_results, fields=None):
            RecordingStore.fields = fields
            return [[] for _ in texts]

    ledger = EvidenceLedger(
        observations=[
            Observation(text="PowerShell ran a stager", provenance=Provenance(source="free_text"))
        ]
    )
    retrieve_techniques(
        ledger, RecordingStore(), kb, retrieval_mode="hybrid", lexical_index=without
    )
    assert RecordingStore.fields == ("name", "description", "procedure", "synthetic")
    retrieve_techniques(
        ledger,
        RecordingStore(),
        kb,
        retrieval_mode="hybrid",
        lexical_index=without,
        include_synthetic=False,
    )
    assert RecordingStore.fields == ("name", "description", "procedure")


class ProbabilityModel:
    """Stub calibrated model: probability is a fixed number per technique id."""

    threshold = 0.5

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def probability(self, features):
        self.calls += 1
        # dense_score identifies the candidate in the two-hit store below
        return self.probabilities[round(features["dense_score"], 2)]

    def accepts(self, features):
        return self.probability(features) >= self.threshold


def _two_hit_setup():
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(id="T1059.001", name="PowerShell", is_subtechnique=True),
            "T1003.001": Technique(id="T1003.001", name="LSASS Memory", is_subtechnique=True),
        },
        groups={},
    )

    class TwoHitStore(EmptyGroupStore):
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "a",
                        "x",
                        {"technique_id": "T1059.001", "name": "PowerShell", "field": "description"},
                        0.1,
                        0.9,
                    ),
                    VectorHit(
                        "b",
                        "y",
                        {
                            "technique_id": "T1003.001",
                            "name": "LSASS Memory",
                            "field": "description",
                        },
                        0.18,
                        0.82,
                    ),
                ]
                for _ in texts
            ]

    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="The loader ran scripts and harvested credentials from security processes.",
                provenance=Provenance(source="free_text"),
            )
        ]
    )
    return kb, TwoHitStore(), ledger


def test_rank_by_acceptance_orders_candidates_by_probability_and_stores_it():
    kb, store, ledger = _two_hit_setup()
    # fusion puts PowerShell (0.9) first; the model is more confident in LSASS (0.82)
    model = ProbabilityModel({0.9: 0.3, 0.82: 0.8})
    fused = retrieve_techniques(ledger, store, kb, retrieval_mode="dense", acceptance_model=model)
    assert [c.technique_id for c in sorted(fused.candidates, key=lambda c: c.fused_rank)] == [
        "T1059.001",
        "T1003.001",
    ]
    assert fused.diagnostics["rank_by_acceptance"] is False
    reordered = retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        acceptance_model=ProbabilityModel({0.9: 0.3, 0.82: 0.8}),
        rank_by_acceptance=True,
    )
    order = [c.technique_id for c in sorted(reordered.candidates, key=lambda c: c.fused_rank)]
    assert order == ["T1003.001", "T1059.001"]
    assert reordered.diagnostics["rank_by_acceptance"] is True
    # the acceptance decision itself is unchanged by the reorder
    assert set(reordered.observed_techniques) == set(fused.observed_techniques) == {"T1003.001"}
    stored = {
        c.technique_id: c.evidence_for("acceptance_model").score for c in reordered.candidates
    }
    assert stored == {"T1003.001": 0.8, "T1059.001": 0.3}


def test_rank_by_acceptance_is_idempotent_across_finalization_passes():
    from apt_classifier.retrieve import finalize_retrieval_candidates

    kb, store, ledger = _two_hit_setup()
    model = ProbabilityModel({0.9: 0.3, 0.82: 0.8})
    first = retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        acceptance_model=model,
        rank_by_acceptance=True,
    )
    calls_after_first = model.calls
    # Second pass (as after verification): fused ranks are now the reordered ones, so
    # recomputing features would drift; the stored probability must be reused instead.
    second = finalize_retrieval_candidates(
        first,
        ledger,
        kb,
        retrieval_mode="dense",
        threshold=0.45,
        acceptance_threshold=0.75,
        relative_margin=0.08,
        acceptance_model=model,
        rank_by_acceptance=True,
    )
    assert model.calls == calls_after_first
    assert [(c.technique_id, c.fused_rank, c.state) for c in second.candidates] == [
        (c.technique_id, c.fused_rank, c.state) for c in first.candidates
    ]
    assert all(
        len([e for e in c.evidence if e.source == "acceptance_model"]) == 1
        for c in second.candidates
    )


def test_rank_by_acceptance_keeps_unscored_candidates_after_scored_ones():
    from apt_classifier.evidence.models import TechniqueCandidate, TechniqueRetrievalEvidence
    from apt_classifier.retrieve import rank_candidates_by_acceptance

    def cand(tid, fused_score, rank, probability=None, source="dense"):
        evidence = [TechniqueRetrievalEvidence(source=source, score=0.5, rank=1)]
        if probability is not None:
            evidence.append(
                TechniqueRetrievalEvidence(source="acceptance_model", score=probability)
            )
        return TechniqueCandidate(
            observation_id="obs-1",
            technique_id=tid,
            technique_name=tid,
            parent_technique_id=None,
            is_subtechnique=False,
            fused_score=fused_score,
            fused_rank=rank,
            evidence=evidence,
        )

    parent = cand("T1059", 0.9, 1, source="hierarchy_parent")  # derived, never scored
    low = cand("T1105", 0.8, 2, probability=0.2)
    high = cand("T1003", 0.7, 3, probability=0.9)
    ordered = rank_candidates_by_acceptance([parent, low, high])
    assert [(c.technique_id, c.fused_rank) for c in ordered] == [
        ("T1003", 1),
        ("T1105", 2),
        ("T1059", 3),
    ]


def test_soft_technique_evidence_extracts_near_misses_only():
    from apt_classifier.retrieve import build_classifier_context, soft_technique_evidence

    kb, store, ledger = _two_hit_setup()
    result = retrieve_techniques(
        ledger,
        store,
        kb,
        retrieval_mode="dense",
        acceptance_model=ProbabilityModel({0.9: 0.3, 0.82: 0.8}),
    )
    assert set(result.observed_techniques) == {"T1003.001"}  # p=0.8 accepted, p=0.3 not
    soft, ids = soft_technique_evidence(
        result, observed=dict(result.observed_techniques), negative=set(), min_probability=0.2
    )
    assert soft == {"T1059.001": pytest.approx(0.3)}
    assert ids["T1059.001"] == (ledger.observations[0].id,)
    # below the floor, or disabled, nothing is soft
    assert soft_technique_evidence(
        result, observed=dict(result.observed_techniques), negative=set(), min_probability=0.5
    ) == ({}, {})
    assert soft_technique_evidence(
        result, observed=dict(result.observed_techniques), negative=set(), min_probability=0.0
    ) == ({}, {})
    # accepted candidates never become soft even when not listed as observed; negative
    # techniques are excluded; the weight scales the probability and is capped below 1.0
    assert soft_technique_evidence(
        result, observed={}, negative={"T1059.001"}, min_probability=0.2, weight=2.0
    ) == ({}, {})
    soft, _ = soft_technique_evidence(
        result, observed={}, negative=set(), min_probability=0.2, weight=5.0
    )
    assert soft == {"T1059.001": pytest.approx(0.95)}
    context = build_classifier_context(kb, ledger, result, soft_min_probability=0.2)
    assert context.soft_techniques == {"T1059.001": pytest.approx(0.3)}
    assert set(context.observed_techniques) == {"T1003.001"}
    # contrastive near misses use the same qualification but live on their own field
    assert context.contrast_techniques == {}
    context = build_classifier_context(kb, ledger, result, contrast_min_probability=0.2)
    assert context.soft_techniques == {}
    assert context.contrast_techniques == {"T1059.001": pytest.approx(0.3)}
    assert context.contrast_technique_observations["T1059.001"] == (ledger.observations[0].id,)


def test_accept_by_corroboration_requires_independent_observations():
    from apt_classifier.evidence.models import TechniqueCandidate, TechniqueRetrievalEvidence
    from apt_classifier.retrieve import accept_by_corroboration

    def cand(obs, tid, probability, state=TechniqueCandidateState.NEEDS_REVIEW, parent=False):
        evidence = [TechniqueRetrievalEvidence(source="dense", score=0.6, rank=1)]
        if parent:
            evidence = [TechniqueRetrievalEvidence(source="hierarchy_parent", score=0.5, rank=1)]
        if probability is not None:
            evidence.append(
                TechniqueRetrievalEvidence(source="acceptance_model", score=probability)
            )
        return TechniqueCandidate(
            observation_id=obs,
            technique_id=tid,
            technique_name=tid,
            parent_technique_id=None,
            is_subtechnique=False,
            fused_score=0.5,
            fused_rank=1,
            evidence=evidence,
            state=state,
        )

    candidates = [
        cand("o1", "T1105", 0.3),
        cand("o2", "T1105", 0.25),  # same technique, second observation -> corroborated
        cand("o1", "T1027", 0.3),
        cand("o1", "T1027", 0.4),  # same observation twice does not count as corroboration
        cand("o3", "T1059", 0.1),
        cand("o4", "T1059", 0.9),  # one below the floor -> only one supporting observation
        cand("o5", "T1003", 0.5, state=TechniqueCandidateState.AUTO_REJECTED),
        cand("o6", "T1003", 0.5),
        cand("o7", "T1566", 0.5, parent=True),
        cand("o8", "T1566", 0.5, parent=True),
    ]
    output, accepted = accept_by_corroboration(candidates, min_observations=2, min_probability=0.2)
    states = {(c.observation_id, c.technique_id): c.state for c in output}
    assert accepted == 2
    assert states[("o1", "T1105")] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states[("o2", "T1105")] is TechniqueCandidateState.AUTO_ACCEPTED
    assert states[("o1", "T1027")] is TechniqueCandidateState.NEEDS_REVIEW
    assert states[("o4", "T1059")] is TechniqueCandidateState.NEEDS_REVIEW
    assert states[("o6", "T1003")] is TechniqueCandidateState.NEEDS_REVIEW  # rejected twin
    assert states[("o7", "T1566")] is TechniqueCandidateState.NEEDS_REVIEW  # derived parents
    assert "corroborated across 2 observations" in next(
        c.state_reason for c in output if c.technique_id == "T1105"
    )
    assert accept_by_corroboration(candidates, min_observations=0, min_probability=0.2)[1] == 0


def test_classifier_context_carries_behaviour_matches():
    from apt_classifier.retrieve import build_classifier_context

    kb, store, ledger = _two_hit_setup()
    result = retrieve_techniques(ledger, store, kb, retrieval_mode="dense")
    context = build_classifier_context(kb, ledger, result)
    assert context.specific_behavior_matches == tuple(result.specific_behavior_matches or ())
    assert context.discrimination is None


# ----- shared retrieval arguments, runtime guards and the retrieval contract -----


def test_cue_tables_are_pinned():
    """The regex cue tables are fixed while retrieval is tuned (docs 21.3.15): a cue added to
    fix a benchmark case would leak into every reported number. Change this pin only with a
    deliberate, separately measured cue change."""
    import hashlib
    import json

    from apt_classifier import retrieve, technique_signals

    def pattern(value):
        return None if value is None else [value.pattern, int(value.flags)]

    payload = {
        "explicit": [
            [technique_id, pattern(regex)]
            for technique_id, regex in retrieve._EXPLICIT_TECHNIQUE_CUES.items()
        ],
        "benign": pattern(retrieve._BENIGN_INDICATORS),
        "signals": [
            [
                signal.technique_id,
                pattern(signal.pattern),
                signal.rationale,
                pattern(signal.benign_guard),
                bool(signal.requires_context),
            ]
            for signal in technique_signals.SIGNALS
        ],
        "adversarial_context": pattern(technique_signals._ADVERSARIAL_CONTEXT),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == "cfff3df110338f1378f0b6f1e14bbe2f4cd3b639d39effee53065324589e04ed"


def old_serving_kwargs(settings, exemplar_index, lexical_index):
    """Frozen copy of the arguments ``AttributionEngine.map_techniques`` passed to
    ``retrieve_techniques`` before ``retrieval_kwargs`` existed (2026-10-02, minus the call-site
    reranker, finalize and acceptance model)."""
    return {
        "threshold": settings.similarity_threshold,
        "top_k": settings.retrieval_top_k,
        "behavior_threshold": settings.behavior_similarity_threshold,
        "behavior_top_k": settings.behavior_retrieval_top_k,
        "retrieval_mode": getattr(settings, "technique_retrieval_mode", "hybrid"),
        "candidate_pool": getattr(settings, "technique_candidate_pool", 12),
        "include_detection": getattr(settings, "technique_include_detection", False),
        "parent_policy": getattr(settings, "technique_parent_policy", "ancestor_compatible"),
        "lexical_index": lexical_index,
        "acceptance_threshold": getattr(settings, "technique_acceptance_threshold", 0.75),
        "relative_margin": getattr(settings, "technique_relative_margin", 0.08),
        "rank_by_acceptance": bool(getattr(settings, "technique_rank_by_acceptance", False)),
        "corroboration_accept_observations": int(
            getattr(settings, "technique_corroboration_accept_observations", 0)
        ),
        "corroboration_accept_min_probability": float(
            getattr(settings, "technique_corroboration_accept_min_probability", 0.2)
        ),
        "rrf_k": int(getattr(settings, "technique_rrf_k", 60)),
        "corroboration_bonus": float(getattr(settings, "technique_corroboration_bonus", 0.0)),
        "segment_acceptance": bool(getattr(settings, "technique_segment_acceptance", False)),
        "hierarchy_parent_mode": str(getattr(settings, "technique_hierarchy_parent_mode", "off")),
        "include_synthetic": bool(getattr(settings, "technique_include_synthetic", False)),
        "platform_from_text": bool(getattr(settings, "technique_platform_from_text", False)),
        "exemplar_index": exemplar_index,
        "exemplar_k": int(getattr(settings, "technique_exemplar_k", 64)),
        "exemplar_tau": float(getattr(settings, "technique_exemplar_tau", 0.05)),
        "exemplar_duplicate_similarity": float(
            getattr(settings, "technique_exemplar_duplicate_similarity", 0.95)
        ),
        "exemplar_new_candidates": int(getattr(settings, "technique_exemplar_new_candidates", 0)),
    }


NEW_RETRIEVAL_KNOBS = {
    "dense_relative_margin",
    "dense_overfetch",
    "exemplar_vote_norm",
    "exemplar_segment_votes",
}


def _serving_settings(**changes):
    from apt_classifier.config import Settings
    from apt_classifier.presets import apply_preset

    return apply_preset(Settings(_env_file=None)).model_copy(update=changes)


def test_retrieval_kwargs_at_the_defaults_equal_the_old_serving_arguments():
    import inspect

    from apt_classifier.retrieve import retrieval_kwargs

    index, lexical = object(), object()
    for settings in (
        _serving_settings(),
        _serving_settings(technique_mapping_preset="conservative", similarity_threshold=0.55),
    ):
        new = retrieval_kwargs(settings, exemplar_index=index, lexical_index=lexical)
        old = old_serving_kwargs(settings, index, lexical)
        assert set(new) - set(old) == NEW_RETRIEVAL_KNOBS
        for key, value in old.items():
            assert new[key] == value and type(new[key]) is type(value), key
    # the new knobs default to retrieve_techniques' own defaults, so passing them changes nothing
    defaults = inspect.signature(retrieve_techniques).parameters
    new = retrieval_kwargs(_serving_settings(), exemplar_index=index, lexical_index=lexical)
    assert {key: new[key] for key in NEW_RETRIEVAL_KNOBS} == {
        key: defaults[key].default for key in NEW_RETRIEVAL_KNOBS
    }
    # a settings object without the fields gets the serving call's old fallbacks
    bare = SimpleNamespace(
        similarity_threshold=0.45,
        retrieval_top_k=30,
        behavior_similarity_threshold=0.55,
        behavior_retrieval_top_k=20,
    )
    new = retrieval_kwargs(bare, exemplar_index=None, lexical_index=None)
    assert {key: new[key] for key in old_serving_kwargs(bare, None, None)} == old_serving_kwargs(
        bare, None, None
    )
    assert {key: new[key] for key in NEW_RETRIEVAL_KNOBS} == {
        key: defaults[key].default for key in NEW_RETRIEVAL_KNOBS
    }


def test_lexical_fields_follow_the_serving_index():
    from apt_classifier.retrieve import default_lexical_fields, lexical_fields

    assert lexical_fields(_serving_settings()) == ("name", "description", "procedure")
    assert lexical_fields(_serving_settings(technique_include_synthetic=True))[-1] == "synthetic"
    # detection text is never indexed lexically by the engine (only retrieve's own default)
    assert "detection" not in lexical_fields(_serving_settings(technique_include_detection=True))
    assert default_lexical_fields(include_synthetic=False, include_detection=True)[-1] == (
        "detection"
    )
    kb = AttackKnowledgeBase(techniques={"T1": Technique(id="T1", name="One")}, groups={})
    index = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(_serving_settings()))
    assert index.fields == ("name", "description", "procedure")


def _hybrid_setup():
    from apt_classifier.retrieve import lexical_fields

    kb, store, ledger = _two_hit_setup()
    lexical = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(_serving_settings()))
    return kb, store, ledger, lexical


def test_new_retrieval_knobs_at_their_defaults_change_nothing():
    from apt_classifier.retrieve import retrieval_kwargs

    kb, store, ledger, lexical = _hybrid_setup()
    settings = _serving_settings()
    old = retrieve_techniques(
        ledger, store, kb, **old_serving_kwargs(settings, None, lexical), finalize=True
    )
    new = retrieve_techniques(
        ledger,
        store,
        kb,
        **retrieval_kwargs(settings, exemplar_index=None, lexical_index=lexical),
        finalize=True,
    )
    assert new.to_dict() == old.to_dict()

    class Recording(type(store)):
        depths: list[int] = []

        def query_techniques(self, texts, *, n_results, **kwargs):
            self.depths.append(n_results)
            return super().query_techniques(texts, n_results=n_results, **kwargs)

    recording = Recording()
    retrieve_techniques(ledger, recording, kb, retrieval_mode="dense", top_k=30)
    deep = retrieve_techniques(
        ledger, recording, kb, retrieval_mode="dense", top_k=30, dense_overfetch=150
    )
    assert recording.depths == [90, 150]
    assert deep.diagnostics["dense_overfetch"] == 150
    # a tighter margin drops the second technique (0.82 is 0.08 below the best 0.90)
    tight = retrieve_techniques(
        ledger, store, kb, retrieval_mode="dense", dense_relative_margin=0.05
    )
    assert [c.technique_id for c in tight.candidates if c.evidence_for("dense")] == ["T1059.001"]
    assert tight.diagnostics["dense_relative_margin"] == 0.05
    assert "dense_relative_margin" not in new.diagnostics
    with pytest.raises(ValueError, match="segment votes"):
        retrieve_techniques(ledger, store, kb, exemplar_segment_votes="sum")


def _candidate(observation_id, technique_id, *, parent=None, p=None, **evidence):
    items = []
    for source, (score, rank) in evidence.items():
        items.append(
            TechniqueRetrievalEvidence(source=source, field="description", score=score, rank=rank)
        )
    if p is not None:
        items.append(TechniqueRetrievalEvidence(source="acceptance_model", score=p))
    return TechniqueCandidate(
        observation_id=observation_id,
        technique_id=technique_id,
        technique_name=technique_id,
        parent_technique_id=parent,
        is_subtechnique=parent is not None,
        fused_score=0.1,
        fused_rank=len(items),
        evidence=items,
    )


def test_candidate_gate_flags_match_the_gate():
    from apt_classifier.retrieve import candidate_gate_flags, context_gate_flags

    text = "The actor ran an encoded PowerShell command on the host"
    cue = _candidate("o", "T1", dense=(0.5, 1), explicit_cue=(0.85, 1))
    weak_cue = _candidate("o", "T2", dense=(0.3, 1), explicit_cue=(0.85, 1))
    lexical_cue = _candidate("o", "T3", lexical=(2.0, 3), structured_signal=(0.9, 1))
    parent = _candidate("o", "T4", hierarchy_parent=(0.5, 1))
    flags = candidate_gate_flags(cue, text, threshold=0.45)
    assert flags == {
        "benign": False,
        "sparse": False,
        "hierarchy_parent": False,
        "cue_supported": True,
    }
    assert candidate_gate_flags(weak_cue, text, threshold=0.45)["cue_supported"] is False
    assert candidate_gate_flags(lexical_cue, text, threshold=0.45)["cue_supported"] is True
    assert candidate_gate_flags(parent, text, threshold=0.45)["hierarchy_parent"] is True
    assert context_gate_flags("routine maintenance of the host") == {
        "benign": True,
        "sparse": False,
    }
    assert context_gate_flags("powershell")["sparse"] is True
    # a precomputed context is used as given
    assert candidate_gate_flags(cue, "", threshold=0.45, context={"benign": True, "sparse": True})[
        "benign"
    ]


def test_runtime_emulation_reproduces_the_accepted_set():
    """What the fit and the harness score (p, candidate_gate_flags, suppressed parents) gives
    exactly the accepted techniques of finalize_retrieval_candidates (plan check P3)."""
    from apt_classifier.retrieve import (
        RetrievalResult,
        candidate_gate_flags,
        finalize_retrieval_candidates,
        runtime_gate_masks,
        suppressed_parent_flags,
    )

    observations = [
        Observation(observation_id="o1", text="The actor executed payloads and moved laterally"),
        Observation(
            observation_id="o2", text="Routine maintenance ran the script on several servers"
        ),
        Observation(observation_id="o3", text="powershell download"),
    ]
    candidates = [
        _candidate("o1", "T1059.001", parent="T1059", p=0.70, dense=(0.80, 1)),
        _candidate("o1", "T1059", p=0.90, hierarchy_parent=(0.5, 1)),
        # exemplar-only parent of a visible child: accepted by p, never materialized
        _candidate("o1", "T1003", p=0.95, exemplar=(0.4, 1)),
        _candidate("o1", "T1003.001", parent="T1003", p=0.30, dense=(0.60, 2)),
        _candidate("o1", "T1105", p=0.10, lexical=(3.0, 2), explicit_cue=(0.85, 1)),
        _candidate("o1", "T1021", p=0.55, dense=(0.50, 3)),
        _candidate("o2", "T1047", p=0.90, dense=(0.70, 1)),
        _candidate("o2", "T1053", p=0.05, dense=(0.60, 2), structured_signal=(0.9, 1)),
        _candidate("o3", "T1086", p=0.90, dense=(0.70, 1)),
    ]
    ledger = EvidenceLedger(observations=observations)
    kb = AttackKnowledgeBase(techniques={}, groups={})
    texts = {item.observation_id: item.text for item in observations}
    flags = [
        candidate_gate_flags(candidate, texts[candidate.observation_id], threshold=0.45)
        for candidate in candidates
    ]
    for row, suppressed in zip(flags, suppressed_parent_flags(candidates), strict=True):
        row["parent_suppressed"] = suppressed
    blocked, forced = runtime_gate_masks(flags)
    probabilities = [float(c.evidence_for("acceptance_model").score) for c in candidates]
    for threshold in (0.05, 0.2, 0.5, 0.6, 0.75, 0.92, 0.99):
        model = SimpleNamespace(threshold=threshold, probability=lambda features: 0.0)
        retrieval = RetrievalResult(
            matches=(),
            observed_techniques={},
            technique_observations={},
            semantic_profile_scores={},
            group_text_scores={},
            specific_behavior_scores={},
            specific_behavior_signals={},
            specific_behavior_matches=(),
            candidates=tuple(candidates),
            parent_policy="ancestor_compatible",
        )
        final = finalize_retrieval_candidates(
            retrieval,
            ledger,
            kb,
            retrieval_mode="hybrid",
            threshold=0.45,
            acceptance_threshold=0.75,
            relative_margin=0.08,
            acceptance_model=model,
        )
        emulated = {
            candidate.technique_id
            for candidate, p, never, always in zip(
                candidates, probabilities, blocked, forced, strict=True
            )
            if always or (not never and p >= threshold)
        }
        assert emulated == set(final.observed_techniques), threshold
    assert flags[2]["parent_suppressed"] and not flags[0]["parent_suppressed"]


def test_retrieval_refuses_a_model_whose_contract_it_does_not_meet():
    from apt_classifier.acceptance import AcceptanceModel, retrieval_contract
    from apt_classifier.retrieve import retrieval_kwargs

    kb, store, ledger, lexical = _hybrid_setup()
    rows = [dict.fromkeys(["dense_score"], 0.9), dict.fromkeys(["dense_score"], 0.2)]
    fitted = AcceptanceModel.fit(rows * 5, [1, 0] * 5, attack_version="t")
    legacy = AcceptanceModel.from_dict(fitted.to_dict())  # an artifact without a contract
    assert legacy.contract_origin == "legacy" and not legacy.requires_exemplars
    kwargs = retrieval_kwargs(_serving_settings(), exemplar_index=None, lexical_index=lexical)
    meets = retrieve_techniques(ledger, store, kb, **kwargs, acceptance_model=legacy)
    assert meets.diagnostics["acceptance_model"] is True
    assert "acceptance_model_disabled" not in meets.diagnostics
    # the in-memory model has no contract and runs anywhere (as before contracts)
    free = retrieve_techniques(
        ledger, store, kb, **{**kwargs, "candidate_pool": 16}, acceptance_model=fitted
    )
    assert free.diagnostics["acceptance_model"] is True
    # a model with a stored contract is held to every key of it
    contract = retrieval_contract(kwargs, None)
    stored = AcceptanceModel.fit(rows * 5, [1, 0] * 5, attack_version="t", contract=contract)
    stored = AcceptanceModel.from_dict(stored.to_dict())
    assert stored.contract_origin == "artifact"
    # the legacy artifact only to the knobs new with contracts; the older ones (pool, mode,
    # synthetic sentences) ran it before contracts and still do, exactly as unconstrained
    for change, key, legacy_refused in (
        ({"candidate_pool": 16}, "candidate_pool 16 (fitted with 12)", False),
        ({"retrieval_mode": "dense"}, "retrieval_mode", False),
        ({"include_synthetic": True}, "include_synthetic", False),
        ({"dense_relative_margin": 0.2}, "dense_relative_margin", True),
        ({"dense_overfetch": 120}, "dense_overfetch 120 (fitted with 0)", True),
    ):
        plain = retrieve_techniques(ledger, store, kb, **{**kwargs, **change})
        unconstrained = retrieve_techniques(
            ledger, store, kb, **{**kwargs, **change}, acceptance_model=fitted
        )
        for model, refused_expected in ((stored, True), (legacy, legacy_refused)):
            result = retrieve_techniques(
                ledger, store, kb, **{**kwargs, **change}, acceptance_model=model
            )
            if refused_expected:
                assert result.diagnostics["acceptance_model"] is False
                assert key in result.diagnostics["acceptance_model_disabled"]
                assert [c.model_dump() for c in result.candidates] == [
                    c.model_dump() for c in plain.candidates
                ]
            else:
                assert "acceptance_model_disabled" not in result.diagnostics, change
                assert [c.model_dump() for c in result.candidates] == [
                    c.model_dump() for c in unconstrained.candidates
                ]
    # exemplar knobs never refuse a model fitted without exemplars
    exemplar_change = retrieve_techniques(
        ledger, store, kb, **{**kwargs, "exemplar_k": 128}, acceptance_model=legacy
    )
    assert exemplar_change.diagnostics["acceptance_model"] is True
