import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from apt_classifier.classifiers.base import ClassifierResult
from apt_classifier.fusion import FusionResult
from apt_classifier.llm import OpenRouterClient, OpenRouterError
from apt_classifier.review.advocate import run_devils_advocate
from apt_classifier.review.bias import FameBiasReport, extract_ranks, measure_fame_bias
from apt_classifier.review.blinding import blind_payload, create_blinding_map
from apt_classifier.review.context import build_review_payload
from apt_classifier.review.reconcile import run_reconciliation
from apt_classifier.review.reviewer import ReviewStage, run_ach_review


@dataclass(frozen=True)
class ReviewOutcome:
    blinded: ReviewStage
    unblinded: ReviewStage
    fame_bias: FameBiasReport
    advocate: ReviewStage
    final: ReviewStage
    audit_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blinded": self.blinded.to_dict(),
            "unblinded": self.unblinded.to_dict(),
            "fame_bias": self.fame_bias.to_dict(),
            "advocate": self.advocate.to_dict(),
            "final": self.final.to_dict(),
            "audit_path": self.audit_path,
        }


def run_review_pipeline(
    client: OpenRouterClient,
    kb: Any,
    evidence_ledger: Any,
    classifier_results: tuple[ClassifierResult, ...],
    fusion: FusionResult,
    *,
    model: str,
    effort: str = "max",
    top_n: int = 10,
    audit_dir: Path | None = None,
) -> ReviewOutcome:
    if not fusion.entries:
        raise ValueError("Cannot review an empty fusion result")
    payload = build_review_payload(kb, evidence_ledger, classifier_results, fusion, top_n=top_n)
    group_ids = [candidate["candidate_id"] for candidate in payload.candidates]
    mapping = create_blinding_map(group_ids)
    blinded_payload = blind_payload(payload, kb, mapping)
    try:
        blinded = run_ach_review(client, blinded_payload, model=model, effort=effort, blinded=True)
    except OpenRouterError as exc:
        raise OpenRouterError(f"blinded ACH review failed: {exc}") from exc
    try:
        unblinded = run_ach_review(client, payload, model=model, effort=effort, blinded=False)
    except OpenRouterError as exc:
        raise OpenRouterError(f"unblinded ACH review failed: {exc}") from exc
    blinded_ranks = extract_ranks(blinded.parsed, mapping.label_to_group)
    unblinded_ranks = extract_ranks(unblinded.parsed)
    prominence = {
        group_id: float(getattr(kb, "prominence", {}).get(group_id, 0.0)) for group_id in group_ids
    }
    fame_bias = measure_fame_bias(blinded_ranks, unblinded_ranks, prominence)
    try:
        advocate = run_devils_advocate(
            client,
            payload,
            fusion.entries[0].group_id,
            model=model,
            effort=effort,
        )
    except OpenRouterError as exc:
        raise OpenRouterError(f"devil's-advocate review failed: {exc}") from exc
    try:
        final = run_reconciliation(
            client,
            payload,
            blinded,
            unblinded,
            fame_bias,
            advocate,
            model=model,
            effort=effort,
        )
    except OpenRouterError as exc:
        raise OpenRouterError(f"final reconciliation review failed: {exc}") from exc
    outcome = ReviewOutcome(blinded, unblinded, fame_bias, advocate, final)
    if audit_dir is None:
        return outcome
    audit_path = _write_audit(audit_dir, evidence_ledger, fusion, outcome)
    return ReviewOutcome(blinded, unblinded, fame_bias, advocate, final, str(audit_path))


def _write_audit(
    audit_dir: Path, evidence_ledger: Any, fusion: FusionResult, outcome: ReviewOutcome
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(getattr(evidence_ledger, "case_id", "") or uuid4())
    safe_case_id = "".join(
        character for character in case_id if character.isalnum() or character in "-_"
    )
    path = audit_dir / f"{safe_case_id}.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": case_id,
        "pre_review_fusion": fusion.to_dict(),
        "review": outcome.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
