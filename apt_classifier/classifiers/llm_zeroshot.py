import json
from dataclasses import asdict, dataclass
from typing import Any

from apt_classifier.classifiers.base import ClassifierEvidence, ClassifierResult
from apt_classifier.llm import LLMUsage, OpenRouterClient
from apt_classifier.review.context import ReviewPayload

ZERO_SHOT_CLASSIFIER_NAME = "llm_zero_shot"


@dataclass(frozen=True)
class ZeroShotResult:
    ranking: tuple[dict[str, Any], ...]
    model: str
    usage: LLMUsage
    prompt: str
    response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking": list(self.ranking),
            "model": self.model,
            "usage": asdict(self.usage),
            "prompt": self.prompt,
            "response": self.response,
        }

    def to_classifier_result(self) -> ClassifierResult:
        """Convert the LLM ranking into a voting ClassifierResult for RRF."""
        entries = []
        for index, item in enumerate(self.ranking, 1):
            group_id = item.get("candidate_id")
            if not isinstance(group_id, str) or not group_id:
                continue
            entries.append(
                ClassifierEvidence(
                    classifier=ZERO_SHOT_CLASSIFIER_NAME,
                    group_id=group_id,
                    score=float(len(self.ranking) - index + 1),
                    rank=index,
                    signal_ids=(f"llm_zero_shot:rank:{index}",),
                )
            )
        return ClassifierResult(
            classifier=ZERO_SHOT_CLASSIFIER_NAME,
            applicable=bool(entries),
            entries=tuple(entries),
            input_count=len(self.ranking),
        )


def run_zero_shot_guess(
    client: OpenRouterClient,
    payload: ReviewPayload,
    *,
    model: str,
    effort: str = "medium",
) -> ZeroShotResult:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "rank": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "rank", "reason"],
                },
            }
        },
        "required": ["ranking"],
    }
    zero_shot_input = {
        "evidence": payload.evidence,
        "candidates": payload.candidates,
        "reference": payload.reference,
    }
    prompt = (
        "Independently rank only the supplied threat-group candidates against the incident evidence. "
        "Use threat-intelligence knowledge, but do not claim conclusive attribution. This opinion is "
        "a bounded, low-weight vote in the ensemble; it is withheld from the independent review and "
        "cannot introduce candidates or raise the attribution stage.\n\n"
        + json.dumps(zero_shot_input, sort_keys=True, separators=(",", ":"))
    )
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are a cyber-threat intelligence analyst providing a bounded second opinion.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        effort=effort,
        schema=schema,
        schema_name="zero_shot_ranking",
        max_tokens=5000,
        seed=83 if model.startswith("openai/") else None,
    )
    parsed = response.parsed if isinstance(response.parsed, dict) else {}
    allowed = {candidate["candidate_id"] for candidate in payload.candidates}
    items = [
        item
        for item in parsed.get("ranking", [])
        if isinstance(item, dict) and item.get("candidate_id") in allowed
    ]
    items.sort(key=lambda item: (int(item.get("rank") or 999), str(item.get("candidate_id"))))
    return ZeroShotResult(tuple(items), model, response.usage, prompt, response.content)
