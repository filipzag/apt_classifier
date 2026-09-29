import json
from pathlib import Path

import pytest

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.fusion import reciprocal_rank_fusion
from apt_classifier.llm import LLMResponse, LLMUsage, OpenRouterError
from apt_classifier.review.context import build_review_payload
from apt_classifier.review.pipeline import run_review_pipeline


class PipelineClient:
    def __init__(self):
        self.calls = []
        self.ach_count = 0

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        schema_name = kwargs["schema_name"]
        if schema_name == "ach_review":
            blinded = self.ach_count == 0
            self.ach_count += 1
            ids = ["Candidate A", "Candidate B"] if blinded else ["G1", "G2"]
            parsed = {
                "summary": "Limited evidence.",
                "ranking": [
                    {
                        "candidate_id": candidate_id,
                        "rank": index,
                        "confidence": "low",
                        "justification": "Structured evidence rank.",
                    }
                    for index, candidate_id in enumerate(ids, 1)
                ],
                "evidence_matrix": [
                    {
                        "candidate_id": candidate_id,
                        "supporting": ["T1"],
                        "contradicting": [],
                        "missing": ["More telemetry"],
                    }
                    for candidate_id in ids
                ],
                "classifier_disagreements": [],
                "uncertainties": ["Sparse evidence"],
                "disconfirmation_tests": ["Collect endpoint telemetry"],
            }
        elif schema_name == "devils_advocate":
            parsed = {
                "leading_candidate": "G1",
                "counter_case": "The evidence is not unique.",
                "contradictory_evidence": ["T3 is unmatched"],
                "missing_evidence": ["No infrastructure evidence"],
                "alternative_candidates": ["G2"],
                "collection_actions": ["Collect network data"],
            }
        else:
            parsed = {
                "headline": "G1 is the leading relative hypothesis",
                "assessment": "Evidence remains insufficient for conclusive attribution.",
                "confidence_label": "low",
                "ranking": [
                    {
                        "candidate_id": "G2",
                        "rank": 1,
                        "confidence": "low",
                        "justification": "Review challenge to the baseline.",
                    },
                    {
                        "candidate_id": "G1",
                        "rank": 2,
                        "confidence": "low",
                        "justification": "Plausible alternative after review.",
                    },
                ],
                "competing_hypotheses": ["G2"],
                "evidence_matrix": [
                    {
                        "candidate_id": "G1",
                        "supporting": ["T1"],
                        "contradicting": [],
                        "missing": ["Infrastructure"],
                    }
                ],
                "counter_case_response": "The counter-case is material.",
                "fame_bias_assessment": "No measured promotion.",
                "limitations": ["ATT&CK reporting is incomplete"],
                "disconfirmation_tests": ["Find contradictory tooling"],
            }
        return LLMResponse("{}", parsed, LLMUsage(10, 10, 0.001), kwargs["model"], {})


def test_full_review_pipeline_isolated_stages_and_audit(synthetic_kb, tmp_path):
    ledger = EvidenceLedger(
        case_id="case-test",
        observations=[
            Observation(text="Observed technique one.", provenance=Provenance(source="test"))
        ],
    )
    result = ClassifierResult(
        "test",
        True,
        (
            ClassifierEvidence("test", "G1", 1.0, 1),
            ClassifierEvidence("test", "G2", 0.5, 2),
            ClassifierEvidence("test", "G3", 0.1, 3),
        ),
    )
    fusion = reciprocal_rank_fusion((result,))
    client = PipelineClient()
    outcome = run_review_pipeline(
        client,
        synthetic_kb,
        ledger,
        (result,),
        fusion,
        model="anthropic/claude-sonnet-5",
        top_n=2,
        audit_dir=tmp_path,
    )
    assert len(client.calls) == 4
    assert [call[1]["schema_name"] for call in client.calls] == [
        "ach_review",
        "ach_review",
        "devils_advocate",
        "advisory_review_reconciliation",
    ]
    assert outcome.final.parsed["ranking"][0]["candidate_id"] == "G1"
    assert outcome.final.parsed["advisory_ranking"][0]["candidate_id"] == "G2"
    assert outcome.final.parsed["rank_changes"]
    assert "authoritative" in outcome.final.parsed["review_role"]
    assert Path(outcome.audit_path).is_file()
    assert "Candidate A" in outcome.blinded.prompt
    assert "Alpha" not in outcome.blinded.prompt
    reconciliation_prompt = client.calls[-1][0][1]["content"]
    assert "advisory review" in reconciliation_prompt
    assert "not a replacement attribution engine" in reconciliation_prompt


def test_review_payload_excludes_bulk_raw_data_and_reference_citations(synthetic_kb):
    synthetic_kb.groups["G1"].external_references = [
        {"description": "large citation", "url": "https://example.invalid"}
    ]
    synthetic_kb.software["S1"].external_references = [
        {"description": "software citation", "url": "https://example.invalid"}
    ]
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Observed technique one.",
                provenance=Provenance(
                    source="winlog",
                    raw_excerpt="RAW-SECRET-LOG-LINE" * 1000,
                    raw_excerpts=["RAW-SECRET-LOG-LINE" * 1000],
                ),
            )
        ]
    )
    synthetic_kb.groups["G1"].technique_behaviors = {
        "T1": ["Alpha used rundll32 MiniDump to dump LSASS memory."]
    }
    result = ClassifierResult(
        "specific_behavior",
        True,
        (
            ClassifierEvidence(
                "specific_behavior",
                "G1",
                1.0,
                1,
                signal_ids=("G1:actor-behavior:T1:0000",),
            ),
            ClassifierEvidence("specific_behavior", "G2", 0.5, 2),
        ),
    )
    fusion = reciprocal_rank_fusion((result,))
    payload = build_review_payload(synthetic_kb, ledger, (result,), fusion, top_n=2)
    serialized = json.dumps(payload.to_dict())
    assert "RAW-SECRET-LOG-LINE" not in serialized
    assert "external_references" not in serialized
    assert payload.reference["actor_behaviors"]["G1:actor-behavior:T1:0000"] == {
        "group_id": "G1",
        "technique_id": "T1",
        "text": "Alpha used rundll32 MiniDump to dump LSASS memory.",
    }
    assert len(serialized) < 20_000


def test_review_pipeline_identifies_failing_stage(synthetic_kb):
    class FailingClient:
        def complete(self, messages, **kwargs):
            raise OpenRouterError("invalid structured response")

    ledger = EvidenceLedger(
        observations=[Observation(text="Observed T1.", provenance=Provenance(source="test"))]
    )
    result = ClassifierResult(
        "test",
        True,
        (ClassifierEvidence("test", "G1", 1.0, 1),),
    )
    fusion = reciprocal_rank_fusion((result,))
    with pytest.raises(OpenRouterError, match="blinded ACH review failed"):
        run_review_pipeline(
            FailingClient(),
            synthetic_kb,
            ledger,
            (result,),
            fusion,
            model="test/model",
        )


def test_review_payload_includes_technique_specificity(synthetic_kb):
    """The review payload should include documented specificity for matched techniques."""
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Observed technique one.",
                provenance=Provenance(source="test"),
            )
        ]
    )
    result = ClassifierResult(
        "test",
        True,
        (ClassifierEvidence("test", "G1", 1.0, 1, matched_techniques=()),),
    )
    fusion = reciprocal_rank_fusion((result,))
    payload = build_review_payload(synthetic_kb, ledger, (result,), fusion, top_n=2)
    # G1 candidate should have technique_specificity for matched techniques
    g1_candidate = next(c for c in payload.candidates if c["candidate_id"] == "G1")
    assert "technique_specificity" in g1_candidate
    # T1 is used by G1 only (1/3 groups) → group-specific
    if "T1" in g1_candidate["technique_specificity"]:
        t1_spec = g1_candidate["technique_specificity"]["T1"]
        assert t1_spec["group_count"] == 1
        assert t1_spec["total_groups"] == 3
        assert t1_spec["is_group_specific"] is True


def test_review_payload_includes_evidence_quality_from_provenance(synthetic_kb):
    """Evidence quality ratings from provenance metadata should be included."""
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="Observed technique one.",
                provenance=Provenance(
                    source="edr",
                    metadata={
                        "reliability": "A",
                        "credibility": "1",
                        "independence_key": "sensor-edr-001",
                        "rating_rationale": "EDR sensor with confirmed detection",
                    },
                ),
            )
        ]
    )
    result = ClassifierResult(
        "test",
        True,
        (ClassifierEvidence("test", "G1", 1.0, 1, matched_techniques=()),),
    )
    fusion = reciprocal_rank_fusion((result,))
    payload = build_review_payload(synthetic_kb, ledger, (result,), fusion, top_n=2)
    obs = payload.evidence["observations"][0]
    assert "evidence_quality" in obs
    assert obs["evidence_quality"]["reliability"] == "A"
    assert obs["evidence_quality"]["credibility"] == "1"
    assert obs["evidence_quality"]["independence_key"] == "sensor-edr-001"
