import json
from dataclasses import asdict, dataclass
from typing import Any

from apt_classifier.llm import LLMUsage, OpenRouterClient
from apt_classifier.review.context import ReviewPayload

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "rank": {"type": "integer"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "justification": {"type": "string"},
                },
                "required": ["candidate_id", "rank", "confidence", "justification"],
            },
        },
        "evidence_matrix": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "supporting": {"type": "array", "items": {"type": "string"}},
                    "contradicting": {"type": "array", "items": {"type": "string"}},
                    "missing": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidate_id", "supporting", "contradicting", "missing"],
            },
        },
        "classifier_disagreements": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "disconfirmation_tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "ranking",
        "evidence_matrix",
        "classifier_disagreements",
        "uncertainties",
        "disconfirmation_tests",
    ],
}


@dataclass(frozen=True)
class ReviewStage:
    name: str
    model: str
    prompt: str
    response: str
    parsed: dict[str, Any]
    usage: LLMUsage

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["usage"] = asdict(self.usage)
        return value


def run_ach_review(
    client: OpenRouterClient,
    payload: ReviewPayload,
    *,
    model: str,
    effort: str,
    blinded: bool,
) -> ReviewStage:
    identity_instruction = (
        "Candidate identities are intentionally blinded. Do not guess real identities."
        if blinded
        else "Candidate identities are visible; use victimology and geopolitical context only when supported."
    )
    prompt = (
        "Perform an independent Analysis of Competing Hypotheses review of this incident. Treat the "
        "RRF result as the deterministic baseline to audit, not as a claim of attribution. Evaluate "
        "supporting, contradicting, and missing evidence for every candidate. Explicitly explain "
        "classifier disagreement and identify concerns with the baseline without presenting a review "
        "rank as a replacement score. Ranks are advisory relative hypotheses, not attribution "
        "probabilities. Do not introduce candidates not supplied. "
        f"{identity_instruction}\n\nCASE DATA:\n"
        + json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"))
    )
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are an independent cyber-threat intelligence reviewer applying ACH.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        effort=effort,
        schema=REVIEW_SCHEMA,
        schema_name="ach_review",
        max_tokens=10000,
    )
    parsed = dict(response.parsed) if isinstance(response.parsed, dict) else {}
    allowed = [candidate["candidate_id"] for candidate in payload.candidates]
    parsed["ranking"] = _normalize_ranking(parsed.get("ranking"), allowed)
    return ReviewStage(
        "blinded_review" if blinded else "unblinded_review",
        model,
        prompt,
        response.content,
        parsed,
        response.usage,
    )


def _normalize_ranking(value: Any, allowed: list[str]) -> list[dict[str, Any]]:
    supplied: dict[str, dict[str, Any]] = {}
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", ""))
            if candidate_id in allowed and candidate_id not in supplied:
                supplied[candidate_id] = item
    ordered = sorted(
        supplied.values(),
        key=lambda item: (
            int(item.get("rank") or len(allowed) + 1),
            allowed.index(str(item["candidate_id"])),
        ),
    )
    ordered.extend(
        {
            "candidate_id": candidate_id,
            "rank": len(ordered) + 1,
            "confidence": "low",
            "justification": "Not ranked by the reviewer.",
        }
        for candidate_id in allowed
        if candidate_id not in supplied
    )
    for index, item in enumerate(ordered, 1):
        item["rank"] = index
    return ordered
