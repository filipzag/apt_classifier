from types import SimpleNamespace

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.evidence.extract import extract_attack_actions
from apt_classifier.evidence.iocs import (
    ambiguous_software_aliases,
    extract_iocs,
    match_software_aliases,
)
from apt_classifier.evidence.models import (
    EvidenceLedger,
    Observation,
    Provenance,
    StrategicContext,
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueRetrievalEvidence,
    TechniqueReviewAction,
    TechniqueReviewDecision,
    TechniqueVerification,
    TechniqueVerificationVerdict,
)
from apt_classifier.llm import LLMResponse, LLMUsage


class FakeClient:
    settings = SimpleNamespace(extract_model="openai/gpt-5.6-sol", extract_effort="low")

    def __init__(self):
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        parsed = {
            "actions": [
                {
                    "sentence": "PowerShell executed an encoded command",
                    "source_lines": [2],
                    "raw_excerpt": "PowerShell ran -enc SQBFAFgA",
                }
            ]
        }
        return LLMResponse("{}", parsed, LLMUsage(10, 5, 0.001), kwargs["model"], {})


def test_ledger_deduplicates_and_preserves_provenance():
    ledger = EvidenceLedger()
    ledger.add_observation(
        Observation(
            text="Process created a file.",
            provenance=Provenance(source="winlog", line_numbers=[1], event_id=11),
        )
    )
    ledger.add_observation(
        Observation(
            text="Process created a file.",
            provenance=Provenance(source="winlog", line_numbers=[4], event_id=11),
        )
    )
    assert len(ledger.observations) == 1
    assert ledger.observations[0].line_numbers == [1, 4]
    assert ledger.observations[0].occurrence_count == 2
    assert EvidenceLedger.from_json(ledger.to_json()).to_dict() == ledger.to_dict()


def test_technique_candidates_and_reviews_round_trip_without_conflating_rejection():
    ledger = EvidenceLedger()
    candidate = TechniqueCandidate(
        observation_id="obs-1",
        technique_id="t1059.001",
        technique_name="PowerShell",
        parent_technique_id="t1059",
        is_subtechnique=True,
        fused_score=0.75,
        fused_rank=1,
        evidence=[
            TechniqueRetrievalEvidence(
                source="dense",
                document_id="T1059.001:description:0000",
                field="description",
                text="Adversaries may abuse PowerShell commands.",
                score=0.82,
                rank=1,
            )
        ],
        verification=TechniqueVerification(
            verdict=TechniqueVerificationVerdict.SUPPORTED,
            reason="The behavior directly names PowerShell.",
            support_span="PowerShell",
            model="openai/gpt-5.6-sol",
        ),
        state=TechniqueCandidateState.AUTO_ACCEPTED,
    )
    decision = ledger.record_technique_review(
        TechniqueReviewDecision(
            observation_id=candidate.observation_id,
            technique_id=candidate.technique_id,
            action=TechniqueReviewAction.REJECT,
            rationale="The event was benign administration.",
        )
    )
    assert candidate.technique_id == "T1059.001"
    assert candidate.parent_technique_id == "T1059"
    assert decision.action is TechniqueReviewAction.REJECT
    assert ledger.question_answers == []
    assert EvidenceLedger.from_json(ledger.to_json()).technique_reviews == [decision]


def test_record_technique_review_replaces_prior_decision_for_same_mapping():
    ledger = EvidenceLedger()
    ledger.record_technique_review(
        TechniqueReviewDecision(
            observation_id="obs-1",
            technique_id="T1059.001",
            action=TechniqueReviewAction.REJECT,
        )
    )
    replacement = ledger.record_technique_review(
        TechniqueReviewDecision(
            observation_id="obs-1",
            technique_id="T1059.001",
            action=TechniqueReviewAction.REPLACE,
            replacement_technique_id="T1059.003",
        )
    )
    assert ledger.technique_reviews == [replacement]


def test_llm_extraction_is_atomic_and_provenanced():
    client = FakeClient()
    observations = extract_attack_actions(
        "Word spawned a child.\nPowerShell ran -enc SQBFAFgA", client
    )
    assert len(observations) == 1
    assert observations[0].text.endswith(".")
    assert observations[0].line_numbers == [2]
    assert client.calls[0][1]["seed"] == 0


def test_iocs_and_software_aliases_are_extracted(synthetic_kb):
    observations = extract_iocs(
        "RedKit contacted evil.example at 8.8.8.8 and produced " + "a" * 64,
        kb=synthetic_kb,
    )
    kinds = {observation.kind for observation in observations}
    assert {"ioc", "software"} <= kinds
    software = next(observation for observation in observations if observation.kind == "software")
    assert software.metadata["software_id"] == "S1"


def _alias_kb():
    return AttackKnowledgeBase(
        version="t",
        techniques={
            "T1": Technique(
                id="T1",
                name="Phishing",
                description=(
                    "Adversaries may clone a login page. The page captures credentials. "
                    "Operators ran mimikatz. A calendar event can start a timer."
                ),
            ),
        },
        groups={
            "G1": Group(
                id="G1",
                name="Lotus",
                description="The group used Mimikatz and [Elise](https://x) (Citation: page).",
                technique_behaviors={"T1": ["The group ran Mimikatz, then netsh."]},
            )
        },
        software={
            "S1": Software(
                id="S1",
                name="Elise",
                software_type="malware",
                aliases=["Elise", "Page"],
                description="Elise, also known as Page. Page is a backdoor.",
            ),
            "S2": Software(
                id="S2", name="Mimikatz", software_type="tool", description="Mimikatz dumps."
            ),
            "S3": Software(id="S3", name="netsh", software_type="tool", description="netsh."),
            "S4": Software(id="S4", name="CALENDAR", software_type="malware"),
        },
    )


def test_ordinary_word_aliases_match_only_in_their_exact_spelling():
    kb = _alias_kb()
    # ATT&CK prose uses "page" and "calendar" as words; code spans, citations, link targets and
    # the software's own description are not evidence; Mimikatz is mostly written as a name
    assert ambiguous_software_aliases(kb) == frozenset({"Page", "CALENDAR"})

    def ids(text):
        return [(m.software_id, m.matched_alias) for m in match_software_aliases(text, kb)]

    assert ids("The skimmer sat on the payment page.") == []
    assert ids("They deployed the Page backdoor.") == [("S1", "Page")]
    assert ids("Page was dropped to disk.") == []  # a leading capital at a sentence start
    assert ids("CALENDAR beaconed out.") == [("S4", "CALENDAR")]  # all caps stays informative
    assert ids("a calendar invite") == []
    # names that are not ordinary words and lowercase command names keep matching any case
    assert ids("they ran mimikatz and NETSH") == [("S2", "mimikatz"), ("S3", "NETSH")]
    assert ids("Elise was loaded") == [("S1", "Elise")]
    # knowledge bases without prose (or none) change nothing
    assert ambiguous_software_aliases(None) == frozenset()
    bare = AttackKnowledgeBase(software={"S1": Software(id="S1", name="Page")})
    assert ambiguous_software_aliases(bare) == frozenset()
    assert [m.software_id for m in match_software_aliases("the page", bare)] == ["S1"]


def test_strategic_context_normalizes_lists_and_round_trips():
    ledger = EvidenceLedger(
        strategic_context=StrategicContext(
            target_geographies="US, Ukraine\nUS",
            target_industries=["Energy", "energy", "Government"],
            motivations="espionage; disruption",
            objectives="Steal research and development documents",
            suspected_origins="China",
        )
    )
    assert ledger.strategic_context.target_geographies == ["US", "Ukraine"]
    assert ledger.strategic_context.target_industries == ["Energy", "Government"]
    assert ledger.strategic_context.motivations == ["espionage", "disruption"]
    assert EvidenceLedger.from_json(ledger.to_json()).to_dict() == ledger.to_dict()
