from apt_classifier.classifiers.llm_zeroshot import (
    ZERO_SHOT_CLASSIFIER_NAME,
    ZeroShotResult,
)
from apt_classifier.llm import LLMResponse, LLMUsage


class ZeroShotClient:
    """Fake LLM client that returns a fixed zero-shot ranking."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        parsed = {"ranking": self.ranking}
        return LLMResponse("{}", parsed, LLMUsage(10, 5, 0.001), kwargs["model"], {})


def test_zero_shot_result_converts_to_classifier_result():
    result = ZeroShotResult(
        ranking=(
            {"candidate_id": "G1", "rank": 1, "reason": "best fit"},
            {"candidate_id": "G2", "rank": 2, "reason": "partial"},
        ),
        model="openai/gpt-5.6-sol",
        usage=LLMUsage(),
        prompt="",
        response="",
    )
    cr = result.to_classifier_result()
    assert cr.classifier == ZERO_SHOT_CLASSIFIER_NAME
    assert cr.applicable
    assert len(cr.entries) == 2
    assert cr.entries[0].group_id == "G1"
    assert cr.entries[0].rank == 1
    assert cr.entries[1].group_id == "G2"
    assert cr.entries[1].rank == 2
    assert cr.entries[0].signal_ids == ("llm_zero_shot:rank:1",)


def test_zero_shot_result_empty_ranking_is_not_applicable():
    result = ZeroShotResult(
        ranking=(),
        model="openai/gpt-5.6-sol",
        usage=LLMUsage(),
        prompt="",
        response="",
    )
    cr = result.to_classifier_result()
    assert not cr.applicable
    assert cr.entries == ()


def test_zero_shot_to_classifier_result_filters_invalid_entries():
    result = ZeroShotResult(
        ranking=(
            {"candidate_id": "G1", "rank": 1, "reason": ""},
            {"candidate_id": "", "rank": 2, "reason": ""},
            {"candidate_id": "G2", "rank": 3, "reason": ""},
        ),
        model="openai/gpt-5.6-sol",
        usage=LLMUsage(),
        prompt="",
        response="",
    )
    cr = result.to_classifier_result()
    assert [e.group_id for e in cr.entries] == ["G1", "G2"]
