from types import SimpleNamespace

from apt_classifier.classifiers.base import ClassifierContext
from apt_classifier.classifiers.cyberre import CyberREClassifier
from apt_classifier.cyberre import run_cyberre_scan, verify_technique_candidates
from apt_classifier.evidence.models import (
    Observation,
    Provenance,
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueRetrievalEvidence,
    TechniqueVerificationVerdict,
)
from apt_classifier.llm import LLMResponse, LLMUsage


class CyberREClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        prompt = messages[-1]["content"]
        if kwargs["schema_name"] == "cyberre_technique_filter":
            parsed = {
                "relevant": "T1" in prompt,
                "reason": "The sentence directly names the behavior."
                if "T1" in prompt
                else "No direct support.",
            }
        else:
            parsed = {
                "ranking": [{"technique_id": "T1", "rank": 1, "reason": "Direct behavior match."}]
            }
        return LLMResponse("{}", parsed, LLMUsage(10, 5, 0.001), kwargs["model"], {})


class VerifierClient:
    def __init__(self, decisions=None, error=None):
        self.decisions = decisions or []
        self.error = error
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise RuntimeError(self.error)
        parsed = {"decisions": self.decisions}
        return LLMResponse("{}", parsed, LLMUsage(20, 10, 0.002), kwargs["model"], {})


class FakeVectorStore:
    """Returns only the supplied candidate technique IDs as vector hits."""

    def __init__(self, hit_ids, parent_ids=None):
        self.hit_ids = list(hit_ids)
        self.parent_ids = dict(parent_ids or {})
        self.queries = []

    def query_techniques(self, texts, *, n_results=10):
        self.queries.append((texts, n_results))
        hits = []
        for index, technique_id in enumerate(self.hit_ids[:n_results]):
            hits.append(
                SimpleNamespace(
                    id=technique_id,
                    document=f"doc-{technique_id}",
                    distance=0.1 * index,
                    similarity=0.9 - 0.05 * index,
                    metadata={
                        "technique_id": technique_id,
                        "name": technique_id,
                        "parent_id": self.parent_ids.get(technique_id, ""),
                    },
                )
            )
        return [hits]


def candidate(technique_id, rank, state=TechniqueCandidateState.NEEDS_REVIEW):
    return TechniqueCandidate(
        observation_id="obs-1",
        technique_id=technique_id,
        technique_name=technique_id,
        fused_score=1 / (60 + rank),
        fused_rank=rank,
        evidence=[
            TechniqueRetrievalEvidence(
                source="dense",
                text=f"Description for {technique_id}",
                score=0.7,
                rank=rank,
            )
        ],
        state=state,
    )


def test_candidate_verifier_supports_multiple_techniques_and_rejects_unrelated():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(description="Runs PowerShell."),
            "T2": SimpleNamespace(description="Creates scheduled tasks."),
            "T3": SimpleNamespace(description="Captures clipboard data."),
        }
    )
    observation = SimpleNamespace(
        id="obs-1",
        text="PowerShell created a scheduled task.",
        provenance=SimpleNamespace(raw_excerpt="PowerShell created a scheduled task."),
    )
    client = VerifierClient(
        [
            {
                "technique_id": "T1",
                "verdict": "supported",
                "support_span": "PowerShell",
                "reason": "PowerShell is explicit.",
            },
            {
                "technique_id": "T2",
                "verdict": "supported",
                "support_span": "scheduled task",
                "reason": "Task creation is explicit.",
            },
            {
                "technique_id": "T3",
                "verdict": "unsupported",
                "support_span": "",
                "reason": "No clipboard behavior.",
            },
            {
                "technique_id": "T9999",
                "verdict": "supported",
                "support_span": "PowerShell",
                "reason": "Not supplied.",
            },
        ]
    )
    result = verify_technique_candidates(
        client,
        kb,
        observation,
        [candidate("T1", 1), candidate("T2", 2), candidate("T3", 3)],
        model="openai/gpt-5.6-sol",
        effort="low",
        scope="all",
    )
    states = {item.technique_id: item.state for item in result.candidates}
    assert states == {
        "T1": TechniqueCandidateState.AUTO_ACCEPTED,
        "T2": TechniqueCandidateState.AUTO_ACCEPTED,
        "T3": TechniqueCandidateState.AUTO_REJECTED,
    }
    assert result.usage.prompt_tokens == 20
    assert client.calls[0][1]["schema_name"] == "technique_candidate_verification"


def test_candidate_verifier_invalid_support_span_becomes_uncertain():
    kb = SimpleNamespace(techniques={"T1": SimpleNamespace(description="Runs PowerShell.")})
    observation = SimpleNamespace(
        id="obs-1",
        text="PowerShell executed a command.",
        provenance=SimpleNamespace(raw_excerpt="PowerShell executed a command."),
    )
    client = VerifierClient(
        [
            {
                "technique_id": "T1",
                "verdict": "supported",
                "support_span": "clipboard capture",
                "reason": "Invalid span.",
            }
        ]
    )
    result = verify_technique_candidates(
        client,
        kb,
        observation,
        [candidate("T1", 1)],
        model="openai/gpt-5.6-sol",
        effort="low",
        scope="all",
    )
    verified = result.candidates[0]
    assert verified.state is TechniqueCandidateState.NEEDS_REVIEW
    assert verified.verification.verdict is TechniqueVerificationVerdict.UNCERTAIN


def test_candidate_verifier_failure_is_visible_and_nonfatal():
    kb = SimpleNamespace(techniques={"T1": SimpleNamespace(description="Runs PowerShell.")})
    observation = SimpleNamespace(
        id="obs-1",
        text="PowerShell executed a command.",
        provenance=SimpleNamespace(raw_excerpt="PowerShell executed a command."),
    )
    result = verify_technique_candidates(
        VerifierClient(error="service unavailable"),
        kb,
        observation,
        [candidate("T1", 1)],
        model="openai/gpt-5.6-sol",
        effort="low",
        scope="all",
    )
    assert result.error == "service unavailable"
    assert result.candidates[0].verification.verdict is TechniqueVerificationVerdict.ERROR
    assert result.candidates[0].state is TechniqueCandidateState.NEEDS_REVIEW


def test_cyberre_runs_filter_then_rank_with_provenance():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
        }
    )
    observation = Observation(
        text="PowerShell executed a command.",
        provenance=Provenance(source="free_text"),
    )
    client = CyberREClient()
    result = run_cyberre_scan(
        client,
        kb,
        observation,
        model="openai/gpt-5.6-sol",
        effort="low",
    )
    assert result.filter_calls == 2
    assert [item.technique_id for item in result.filtered] == ["T1"]
    assert result.ranking[0].technique_id == "T1"
    assert result.ranking[0].rank == 1
    assert result.observation_id == observation.id
    assert result.technique_scores() == {"T1": 1.0}
    assert len(client.calls) == 3
    assert client.calls[-1][1]["schema_name"] == "cyberre_technique_ranking"


def test_cyberre_prefilter_narrows_candidate_space():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
            "T3": SimpleNamespace(name="Clipboard Data", description="Reads clipboard."),
            "T4": SimpleNamespace(name="Data Encrypted", description="Encrypts data."),
        }
    )
    observation = Observation(
        text="PowerShell executed a command.",
        provenance=Provenance(source="free_text"),
    )
    store = FakeVectorStore(["T1", "T3"])
    client = CyberREClient()
    result = run_cyberre_scan(
        client,
        kb,
        observation,
        model="openai/gpt-5.6-sol",
        effort="low",
        prefilter_store=store,
        prefilter_top_k=2,
    )
    assert store.queries and store.queries[0][1] == 2
    assert result.filter_calls == 2
    assert [item.technique_id for item in result.filtered] == ["T1"]
    assert result.ranking[0].technique_id == "T1"
    assert len(client.calls) == 3


def test_cyberre_prefilter_reads_technique_id_from_vector_metadata():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
            "T3": SimpleNamespace(name="Clipboard Data", description="Reads clipboard."),
        }
    )
    observation = Observation(
        text="PowerShell executed a command.",
        provenance=Provenance(source="free_text"),
    )
    store = SimpleNamespace(
        query_techniques=lambda texts, n_results: [
            [
                SimpleNamespace(
                    id="T1:description:0000",
                    metadata={"technique_id": "T1", "parent_id": ""},
                ),
                SimpleNamespace(
                    id="T2:description:0000",
                    metadata={"technique_id": "T2", "parent_id": ""},
                ),
            ]
        ]
    )
    client = CyberREClient()
    result = run_cyberre_scan(
        client,
        kb,
        observation,
        model="openai/gpt-5.6-sol",
        effort="low",
        prefilter_store=store,
        prefilter_top_k=2,
    )
    assert result.filter_calls == 2
    assert len(client.calls) == 3


def test_cyberre_prefilter_includes_parent_techniques():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T1059": SimpleNamespace(name="Command and Scripting", description="Parent technique."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
        }
    )
    observation = Observation(
        text="PowerShell executed a command.",
        provenance=Provenance(source="free_text"),
    )
    store = FakeVectorStore(["T1"], parent_ids={"T1": "T1059"})
    client = CyberREClient()
    result = run_cyberre_scan(
        client,
        kb,
        observation,
        model="openai/gpt-5.6-sol",
        effort="low",
        prefilter_store=store,
        prefilter_top_k=1,
    )
    assert result.filter_calls == 2
    assert {item.technique_id for item in result.filtered} == {"T1", "T1059"}


def test_cyberre_without_prefilter_uses_full_kb():
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
            "T3": SimpleNamespace(name="Clipboard Data", description="Reads clipboard."),
        }
    )
    observation = Observation(
        text="PowerShell executed a command.",
        provenance=Provenance(source="free_text"),
    )
    client = CyberREClient()
    result = run_cyberre_scan(
        client,
        kb,
        observation,
        model="openai/gpt-5.6-sol",
        effort="low",
        prefilter_store=None,
    )
    assert result.filter_calls == 3
    assert len(client.calls) == 4


def test_cyberre_batch_scan_loops_over_all_observations():
    from apt_classifier.config import Settings
    from apt_classifier.evidence.models import EvidenceLedger, EvidenceStatus
    from apt_classifier.session import AttributionEngine

    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(name="PowerShell", description="Runs PowerShell."),
            "T2": SimpleNamespace(name="Scheduled Task", description="Creates a scheduled task."),
        },
        groups={},
    )
    observations = [
        Observation(
            text="PowerShell executed a command.",
            provenance=Provenance(source="free_text"),
        ),
        Observation(
            text="PowerShell spawned a child process.",
            provenance=Provenance(source="free_text"),
        ),
        Observation(
            text="8.8.8.8",
            kind="ioc",
            provenance=Provenance(source="free_text"),
        ),
    ]
    for observation in observations:
        observation.status = EvidenceStatus.POSITIVE
    ledger = EvidenceLedger(observations=observations)
    state = SimpleNamespace(
        ledger=ledger,
        cyberre_results={},
        context=SimpleNamespace(
            cyberre_techniques={},
            cyberre_technique_observations={},
        ),
    )
    client = CyberREClient()
    store = FakeVectorStore(["T1", "T2"])
    engine = AttributionEngine.__new__(AttributionEngine)
    engine.llm = client
    engine.kb = kb
    engine.store = store
    engine.settings = Settings()
    engine.classifiers = []
    engine.rescore = lambda s: s
    results = engine.run_cyberre_batch(state, use_prefilter=True)
    assert set(results.keys()) == {observations[0].id, observations[1].id}
    assert state.context.cyberre_techniques
    assert "T1" in state.context.cyberre_techniques


def test_cyberre_classifier_ranks_only_groups_with_inferred_technique_overlap(synthetic_kb):
    context = ClassifierContext(
        kb=synthetic_kb,
        observed_techniques={"T1": 1.0},
        technique_observations={"T1": ("deterministic",)},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
        cyberre_techniques={"T1": 1.0, "T4": 0.5},
        cyberre_technique_observations={"T1": ("obs-1",), "T4": ("obs-1",)},
    )
    result = CyberREClassifier().classify(context)
    assert result.applicable
    assert result.entries
    assert all(entry.group_id in {"G1", "G2", "G3"} for entry in result.entries)
    assert result.entries[0].matched_techniques
    assert result.entries[0].matched_techniques[0].observation_ids == ("obs-1",)


def test_cyberre_classifier_abstains_without_scan_results(synthetic_kb):
    context = ClassifierContext(
        kb=synthetic_kb,
        observed_techniques={"T1": 1.0},
        technique_observations={},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
    )
    result = CyberREClassifier().classify(context)
    assert not result.applicable
    assert result.entries == ()
