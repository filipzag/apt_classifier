import json
from typing import Any

from apt_classifier.llm import OpenRouterClient
from apt_classifier.review.context import ReviewPayload
from apt_classifier.review.reviewer import ReviewStage

ADVOCATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "leading_candidate": {"type": "string"},
        "counter_case": {"type": "string"},
        "contradictory_evidence": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "alternative_candidates": {"type": "array", "items": {"type": "string"}},
        "collection_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "leading_candidate",
        "counter_case",
        "contradictory_evidence",
        "missing_evidence",
        "alternative_candidates",
        "collection_actions",
    ],
}


def run_devils_advocate(
    client: OpenRouterClient,
    payload: ReviewPayload,
    leading_candidate: str,
    *,
    model: str,
    effort: str,
) -> ReviewStage:
    prompt = (
        "Build the strongest evidence-based case AGAINST the supplied leading candidate. Focus on "
        "unmatched observations, missing expected tradecraft, shared commodity techniques, and better "
        "alternatives. Do not defend the leader and do not invent evidence.\n\n"
        f"LEADING CANDIDATE: {leading_candidate}\nCASE DATA:\n"
        + json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"))
    )
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are an independent red-team reviewer of cyber attribution hypotheses.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        effort=effort,
        schema=ADVOCATE_SCHEMA,
        schema_name="devils_advocate",
        max_tokens=6000,
    )
    parsed = dict(response.parsed) if isinstance(response.parsed, dict) else {}
    return ReviewStage("devils_advocate", model, prompt, response.content, parsed, response.usage)
