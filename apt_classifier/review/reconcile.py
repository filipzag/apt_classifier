import json
from typing import Any

from apt_classifier.llm import OpenRouterClient
from apt_classifier.review.bias import FameBiasReport
from apt_classifier.review.context import ReviewPayload
from apt_classifier.review.reviewer import ReviewStage, _normalize_ranking

FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "assessment": {"type": "string"},
        "confidence_label": {"type": "string", "enum": ["low", "medium", "high"]},
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
        "competing_hypotheses": {"type": "array", "items": {"type": "string"}},
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
        "counter_case_response": {"type": "string"},
        "fame_bias_assessment": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "disconfirmation_tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "headline",
        "assessment",
        "confidence_label",
        "ranking",
        "competing_hypotheses",
        "evidence_matrix",
        "counter_case_response",
        "fame_bias_assessment",
        "limitations",
        "disconfirmation_tests",
    ],
}


def run_reconciliation(
    client: OpenRouterClient,
    payload: ReviewPayload,
    blinded_review: ReviewStage,
    unblinded_review: ReviewStage,
    bias: FameBiasReport,
    advocate: ReviewStage,
    *,
    model: str,
    effort: str,
) -> ReviewStage:
    review_inputs = {
        "case": payload.to_dict(),
        "blinded_review": blinded_review.parsed,
        "unblinded_review": unblinded_review.parsed,
        "fame_bias": bias.to_dict(),
        "devils_advocate": advocate.parsed,
    }
    prompt = (
        "Review the independent blinded and unblinded ACH reviews against the deterministic ensemble "
        "and the devil's-advocate counter-case. This is an advisory review of the RRF result, not a "
        "replacement attribution engine. Explicitly address the counter-case and any promotion that "
        "occurred only after identities were revealed. You may propose an alternative ordering only "
        "for review discussion, and every proposed rank change needs an evidence-based justification "
        "tied to supplied observations, classifier output, or documented ATT&CK relationships. Do not "
        "call any candidate decisive, rejected, proven, or final; do not claim that the review changes "
        "the ensemble result. Describe all candidates as relative hypotheses, never conclusive "
        "attribution or calibrated probability.\n\nINPUTS:\n"
        + json.dumps(review_inputs, sort_keys=True, separators=(",", ":"))
    )
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are the final independent cyber-attribution review chair.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        effort=effort,
        schema=FINAL_SCHEMA,
        schema_name="advisory_review_reconciliation",
        max_tokens=12000,
    )
    parsed = dict(response.parsed) if isinstance(response.parsed, dict) else {}
    allowed = [candidate["candidate_id"] for candidate in payload.candidates]
    advisory_ranking = _normalize_ranking(parsed.get("ranking"), allowed)
    deterministic_ranking = _deterministic_ranking(payload, allowed)
    parsed["advisory_ranking"] = advisory_ranking
    parsed["deterministic_ranking"] = deterministic_ranking
    parsed["rank_changes"] = _rank_changes(deterministic_ranking, advisory_ranking)
    parsed["ranking"] = deterministic_ranking
    parsed["review_role"] = "advisory_review; deterministic RRF ranking remains authoritative"
    return ReviewStage("reconciliation", model, prompt, response.content, parsed, response.usage)


def _deterministic_ranking(payload: ReviewPayload, allowed: list[str]) -> list[dict[str, Any]]:
    entries = payload.fusion.get("entries", [])
    entry_by_id = {
        str(entry.get("group_id")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("group_id") in allowed
    }
    ranking = []
    for index, candidate_id in enumerate(allowed, 1):
        entry = entry_by_id.get(candidate_id, {})
        ranking.append(
            {
                "candidate_id": candidate_id,
                "rank": index,
                "confidence": "low",
                "justification": (
                    "Deterministic family-aware RRF baseline; independent review does not "
                    "alter this ensemble order."
                ),
                "rrf_score": entry.get("score"),
                "rrf_relative_score": entry.get("relative_score"),
            }
        )
    return ranking


def _rank_changes(
    deterministic: list[dict[str, Any]], advisory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline = {item["candidate_id"]: item["rank"] for item in deterministic}
    proposed = {item["candidate_id"]: item["rank"] for item in advisory}
    return [
        {
            "candidate_id": candidate_id,
            "rrf_rank": baseline[candidate_id],
            "advisory_rank": proposed[candidate_id],
            "change": baseline[candidate_id] - proposed[candidate_id],
        }
        for candidate_id in baseline
        if baseline[candidate_id] != proposed.get(candidate_id, baseline[candidate_id])
    ]
