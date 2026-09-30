import inspect
from types import SimpleNamespace

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.cyberre import CYBERRE_CLASSIFIER_NAME
from apt_classifier.llm import LLMResponse, LLMUsage
from apt_classifier.review.context import ReviewPayload
from apt_classifier.review.pipeline import run_review_pipeline
from apt_classifier.review.reviewer import run_ach_review


class CapturingClient:
    def __init__(self):
        self.messages = None

    def complete(self, messages, **kwargs):
        self.messages = messages
        parsed = {
            "summary": "Evidence is limited.",
            "ranking": [
                {
                    "candidate_id": "Candidate A",
                    "rank": 1,
                    "confidence": "low",
                    "justification": "Only one candidate supplied.",
                }
            ],
            "evidence_matrix": [
                {
                    "candidate_id": "Candidate A",
                    "supporting": ["T1"],
                    "contradicting": [],
                    "missing": ["Additional telemetry"],
                }
            ],
            "classifier_disagreements": [],
            "uncertainties": ["Sparse evidence"],
            "disconfirmation_tests": ["Collect network telemetry"],
        }
        return LLMResponse("{}", parsed, LLMUsage(), kwargs["model"], {})


def test_reviewer_receives_only_explicit_typed_payload():
    forbidden_prior = "SECRET PRIOR LLM REASONING"
    forbidden_vote = "LLM VOTE SAID G9999"
    payload = ReviewPayload(
        evidence={"observations": [{"text": "Observed T1"}]},
        candidates=({"candidate_id": "Candidate A", "known_techniques": ["T1"]},),
        classifier_results=(),
        fusion={"entries": [{"group_id": "Candidate A", "rank": 1}]},
        reference={"techniques": {"T1": {"name": "Technique"}}},
    )
    client = CapturingClient()
    stage = run_ach_review(
        client,
        payload,
        model="anthropic/claude-sonnet-5",
        effort="max",
        blinded=True,
    )
    prompt = client.messages[1]["content"]
    assert forbidden_prior not in prompt
    assert forbidden_vote not in prompt
    assert "Candidate A" in prompt
    assert stage.parsed["ranking"][0]["candidate_id"] == "Candidate A"
    parameters = inspect.signature(run_review_pipeline).parameters
    assert "message_history" not in parameters
    assert "zero_shot" not in parameters


def test_review_payload_excludes_the_cyberre_llm_classifier():
    """The review payload must not include the CyberRE-LLM classifier result."""
    from apt_classifier.fusion import FusionEntry, FusionResult

    cyberre_result = ClassifierResult(
        classifier=CYBERRE_CLASSIFIER_NAME,
        applicable=True,
        entries=(
            ClassifierEvidence(
                classifier=CYBERRE_CLASSIFIER_NAME,
                group_id="G1",
                score=3.0,
                rank=1,
                signal_ids=("cyberre:T1:rank:1",),
            ),
        ),
        input_count=1,
    )
    other_result = ClassifierResult(
        classifier="idf_tfidf_cosine",
        applicable=True,
        entries=(
            ClassifierEvidence(
                classifier="idf_tfidf_cosine",
                group_id="G1",
                score=1.0,
                rank=1,
            ),
        ),
        input_count=1,
    )
    kb = SimpleNamespace(
        groups={"G1": SimpleNamespace(id="G1", name="Group One", aliases=(), description="desc")},
        techniques={},
        software={},
        campaigns={},
        group_techniques={"G1": set()},
        group_software={"G1": set()},
        campaign_groups={},
        prominence={"G1": 1.0},
    )
    ledger = SimpleNamespace(
        observations=[],
        technique_matches=[],
        question_answers=[],
        strategic_context=SimpleNamespace(
            target_geographies=[],
            target_industries=[],
            motivations=[],
            objectives=[],
            suspected_origins=[],
        ),
        case_id="test",
    )
    fusion = FusionResult(
        entries=(
            FusionEntry(
                group_id="G1",
                score=1.0,
                relative_score=100.0,
                rank=1,
                classifier_ranks={},
                rank_min=1,
                rank_max=1,
                rank_std=0.0,
                contested=False,
            ),
        ),
        active_classifiers=("idf_tfidf_cosine",),
        correlations={},
        k=60,
        effective_weights={"idf_tfidf_cosine": 1.0},
    )
    filtered = tuple(
        r
        for r in (cyberre_result, other_result)
        if r.classifier
        != CYBERRE_CLASSIFIER_NAME  # the filter AttributionEngine.run_review applies
    )
    from apt_classifier.review.context import build_review_payload

    payload = build_review_payload(kb, ledger, filtered, fusion)
    classifier_names = [r["classifier"] for r in payload.classifier_results]
    assert CYBERRE_CLASSIFIER_NAME not in classifier_names
    assert "idf_tfidf_cosine" in classifier_names
