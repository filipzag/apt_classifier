from types import SimpleNamespace

from apt_classifier.evidence.local_extract import local_observations
from apt_classifier.evidence.standardise import CHUNK_SIZE, standardise_observations


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(parsed=answer)


TEXT = (
    "The actor sent spearphishing emails with a malicious Word attachment. "
    "Once opened, a macro launched PowerShell with an encoded command. "
    "This report was written for the security team."
)


def test_each_sentence_is_rewritten_one_for_one_and_keeps_its_original():
    observations = local_observations(TEXT)
    client = FakeClient(
        [
            {
                "sentences": [
                    {
                        "index": 1,
                        "sentence": "The actor sent spearphishing emails with a Word attachment.",
                    },
                    {
                        "index": 2,
                        "sentence": "A macro launched PowerShell with an encoded command.",
                    },
                ]
            }
        ]
    )
    diagnostics = {}
    result = standardise_observations(
        observations, client, model="local/qwen3.5:4b", diagnostics=diagnostics
    )
    assert len(result) == len(observations) == 3
    assert result[1].text == "A macro launched PowerShell with an encoded command."
    assert result[1].provenance.metadata["original_sentence"] == observations[1].text
    assert result[1].provenance.metadata["normalization"] == "local_llm_sentence"
    assert result[1].provenance.line_numbers == observations[1].provenance.line_numbers
    # sentence 3 was omitted by the model: the split sentence stays
    assert result[2] == observations[2]
    assert diagnostics["standardisation"] == {
        "model": "local/qwen3.5:4b",
        "rewritten": 2,
        "kept": 1,
        "failed_calls": 0,
    }
    kwargs = client.calls[0][1]
    assert kwargs["effort"] == "low" and kwargs["model"] == "local/qwen3.5:4b"


def test_degenerate_rewrites_and_failed_calls_keep_the_split():
    observations = local_observations(TEXT)
    client = FakeClient([{"sentences": [{"index": 1, "sentence": "None."}]}])
    result = standardise_observations(observations, client, model="local/qwen3.5:4b")
    assert result == observations
    diagnostics = {}
    result = standardise_observations(
        observations,
        FakeClient([RuntimeError("Local LLM request failed")]),
        model="local/qwen3.5:4b",
        diagnostics=diagnostics,
    )
    assert result == observations
    assert diagnostics["standardisation"]["failed_calls"] == 1


def test_long_reports_are_rewritten_in_chunks():
    text = " ".join(
        f"The actor copied file{i}.dat to the staging share." for i in range(CHUNK_SIZE + 5)
    )
    observations = local_observations(text)
    assert len(observations) == CHUNK_SIZE + 5
    client = FakeClient([{"sentences": []}, {"sentences": []}])
    standardise_observations(observations, client, model="local/qwen3.5:4b")
    assert len(client.calls) == 2


def test_service_rewrites_prose_with_a_local_model_and_extracts_with_openrouter():
    from apt_classifier.llm import OpenRouterClient
    from apt_classifier.service import CaseService, IntakePayload

    class Settings:
        extract_model = "openai/gpt-5.6-sol"
        local_llm_enabled = True
        local_llm_fallback_model = "local/qwen3.5:4b"

        def require_api_key(self):
            raise RuntimeError("OPENROUTER_API_KEY is required.")

    client = OpenRouterClient(Settings())
    rewrites = {
        "sentences": [
            {"index": 1, "sentence": "The actor sent spearphishing emails with a Word attachment."}
        ]
    }
    client.complete = lambda *args, **kwargs: SimpleNamespace(parsed=rewrites)
    engine = SimpleNamespace(settings=Settings(), kb=SimpleNamespace(techniques={}), llm=client)
    diagnostics = {}
    ledger = CaseService._build_ledger(
        SimpleNamespace(engine=engine), IntakePayload(narrative=TEXT), diagnostics
    )
    texts = [observation.text for observation in ledger.observations]
    assert texts[0] == "The actor sent spearphishing emails with a Word attachment."
    assert len(texts) == 3
    assert diagnostics["normalization"] == "local_llm_sentences"
