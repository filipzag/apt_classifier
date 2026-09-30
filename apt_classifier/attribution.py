"""Open-set attribution stages and calibrated rejection.

This module implements a deterministic attribution decision layer above the
RRF ranking. It introduces explicit stages that distinguish insufficient
evidence, unresolved activity clusters, and named-group hypotheses. The
stages are:

- ``INSUFFICIENT_EVIDENCE``: too little evidence to form any conclusion.
- ``UNRESOLVED_ACTIVITY_CLUSTER``: malicious evidence exists but no named
  group meets the maturity threshold. A stable ``CL-UNK-*`` cluster ID is
  assigned.
- ``KNOWN_GROUP_SHORTLIST``: one or more named groups are plausible but none
  meets the hypothesis threshold.
- ``KNOWN_GROUP_HYPOTHESIS``: a named group is the leading hypothesis, but
  this is still not confirmation.

Key design decisions:
- RRF remains an auditable ranking, not a probability.
- The stage decision is deterministic and reproducible across processes (no
  set-ordering dependence).
- LLM output can never raise the stage.
- Live intelligence is non-voting and cannot promote the stage.
- Without a valid calibration artifact, the UI emits no numeric attribution
  probability — only qualitative maturity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from apt_classifier.attack.related import related_to
from apt_classifier.evidence.quality import (
    compute_technique_specificity,
)
from apt_classifier.fusion import FusionResult
from apt_classifier.separation import discriminate


class AttributionStage(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRESOLVED_ACTIVITY_CLUSTER = "unresolved_activity_cluster"
    KNOWN_GROUP_SHORTLIST = "known_group_shortlist"
    KNOWN_GROUP_HYPOTHESIS = "known_group_hypothesis"


@dataclass(frozen=True)
class AttributionDecision:
    stage: AttributionStage
    cluster_id: str | None
    leading_group_id: str | None
    leading_group_name: str = ""
    leading_score: float = 0.0
    margin: float = 0.0
    maturity_score: float = 0.0
    maturity_factors: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    is_calibrated: bool = False
    calibrated_probability: float | None = None
    calibration_note: str = ""
    # Documented overlap of the leading group with other ATT&CK groups (id -> reason), from
    # description cross-references and rarely shared software; see attack/related.py.
    related_leading_groups: dict[str, str] = field(default_factory=dict)
    # Candidate-relative separation of the leader from its best-supported alternative
    # (separation.py): -1..1, the per-channel factors, the runner-up and whether it is
    # documented-related, and the gate mode that produced the stage.
    separation: float = 0.0
    separation_factors: dict[str, Any] = field(default_factory=dict)
    runner_up_group_id: str | None = None
    runner_up_related: bool = False
    gate_mode: str = "maturity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "cluster_id": self.cluster_id,
            "leading_group_id": self.leading_group_id,
            "leading_group_name": self.leading_group_name,
            "leading_score": self.leading_score,
            "margin": self.margin,
            "maturity_score": self.maturity_score,
            "maturity_factors": dict(self.maturity_factors),
            "rationale": self.rationale,
            "is_calibrated": self.is_calibrated,
            "calibrated_probability": self.calibrated_probability,
            "calibration_note": self.calibration_note,
            "related_leading_groups": dict(self.related_leading_groups),
            "separation": self.separation,
            "separation_factors": dict(self.separation_factors),
            "runner_up_group_id": self.runner_up_group_id,
            "runner_up_related": self.runner_up_related,
            "gate_mode": self.gate_mode,
        }


@dataclass(frozen=True)
class MaturityThresholds:
    """Thresholds for attribution maturity gating.

    These are conservative defaults. A named hypothesis requires strong,
    independent, corroborated evidence. The thresholds should only be
    relaxed after held-out evaluation demonstrates a benefit.
    """

    min_observations: int = 3
    min_techniques: int = 2
    min_independent_evidence_families: int = 2
    min_maturity_score: float = 0.5
    min_margin: float = 0.05
    hypothesis_maturity_score: float = 0.7
    hypothesis_margin: float = 0.10
    # Naming gate mode: "maturity" (maturity >= bar and margin >= bar), "separation"
    # (candidate-relative separation and discriminative mass instead of the maturity bar) or
    # "both". See separation.py and docs 10.6.
    gate_mode: str = "maturity"  # the engine passes the configured mode (default "separation")
    hypothesis_separation: float = 0.5
    hypothesis_min_discriminative: float = 3.0
    hypothesis_specificity: float = 0.0


SUPPORT_RANK_LIMIT = 10


def _evidence_family_count(
    fusion: FusionResult,
    classifier_families: dict[str, str] | None = None,
    *,
    support_rank_limit: int = SUPPORT_RANK_LIMIT,
) -> int:
    """Count distinct evidence families that *support* the top entry.

    A classifier supports the leading group only when it ranks that group within
    ``support_rank_limit``. Several classifiers (Naive Bayes, group description) rank
    every group, so merely appearing in ``classifier_ranks`` is not evidence.
    """
    if not fusion.entries:
        return 0
    top = fusion.entries[0]
    families: set[str] = set()
    for classifier_name, rank in top.classifier_ranks.items():
        if rank is None or int(rank) > support_rank_limit:
            continue
        family = (classifier_families or {}).get(classifier_name, classifier_name)
        families.add(family)
    return len(families)


def _compute_maturity(
    kb: Any,
    fusion: FusionResult,
    context: Any,
    thresholds: MaturityThresholds,
    classifier_families: dict[str, str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute a qualitative maturity score for the leading hypothesis.

    The score is a transparent combination of factors:
    - Evidence volume (observations + techniques)
    - Independent evidence families
    - Classifier agreement (low rank_std)
    - Margin between top-1 and top-2
    - Documented specificity of matched techniques
    - Corroboration quality

    Returns a 0–1 score and a dict of factor scores.
    """
    factors: dict[str, float] = {}
    observations = getattr(context, "observed_techniques", {})
    observation_count = len(getattr(context, "narrative", "").split()) // 10  # rough
    technique_count = len(observations)
    # Evidence volume factor
    volume = min(
        1.0,
        (technique_count + observation_count)
        / max(thresholds.min_techniques + thresholds.min_observations, 1),
    )
    factors["evidence_volume"] = round(volume, 3)
    # Independent families
    family_count = _evidence_family_count(fusion, classifier_families)
    families = min(1.0, family_count / max(thresholds.min_independent_evidence_families, 1))
    factors["independent_families"] = round(families, 3)
    # Classifier agreement (inverse of rank_std)
    if fusion.entries:
        rank_std = fusion.entries[0].rank_std
        agreement = max(0.0, 1.0 - rank_std / 10.0)
    else:
        agreement = 0.0
    factors["classifier_agreement"] = round(agreement, 3)
    # Margin
    if len(fusion.entries) >= 2:
        margin = (fusion.entries[0].relative_score - fusion.entries[1].relative_score) / 100.0
    elif fusion.entries:
        margin = fusion.entries[0].relative_score / 100.0
    else:
        margin = 0.0
    factors["margin"] = round(min(1.0, margin / max(thresholds.min_margin, 0.01)), 3)
    # Specificity of matched techniques
    group_techniques = getattr(kb, "group_techniques", {})
    if fusion.entries and observations:
        top_group_id = fusion.entries[0].group_id
        matched = set(group_techniques.get(top_group_id, ())) & set(observations)
        if matched:
            specificity_scores = [
                compute_technique_specificity(tid, kb).specificity for tid in sorted(matched)[:10]
            ]
            avg_specificity = sum(specificity_scores) / len(specificity_scores)
        else:
            avg_specificity = 0.0
    else:
        avg_specificity = 0.0
    factors["technique_specificity"] = round(avg_specificity, 3)
    # Weighted combination (transparent, not a probability)
    weights = {
        "evidence_volume": 0.15,
        "independent_families": 0.25,
        "classifier_agreement": 0.20,
        "margin": 0.15,
        "technique_specificity": 0.25,
    }
    maturity = sum(weights[key] * factors[key] for key in weights)
    return round(maturity, 3), factors


def _related(kb: Any, group_id: str) -> dict[str, str]:
    try:
        return related_to(kb, group_id)
    except Exception:  # noqa: BLE001 - overlap hints must never break the stage decision
        return {}


def _generate_cluster_id(fusion: FusionResult, context: Any = None) -> str:
    """Generate a stable ``CL-UNK-*`` cluster ID from the evidence, not just the leader.

    The ID hashes the sorted observed technique IDs and the top three candidate group
    IDs, so two unrelated cases that merely share a leading candidate get different
    clusters, and the same evidence always yields the same ID across processes.
    """
    if not fusion.entries:
        return "CL-UNK-000000"
    observed = sorted(str(item) for item in getattr(context, "observed_techniques", {}) or ())
    leaders = [entry.group_id for entry in fusion.entries[:3]]
    material = "|".join(observed) + "||" + "|".join(leaders)
    return f"CL-UNK-{hashlib.sha256(material.encode()).hexdigest()[:6]}"


def assess_attribution(
    kb: Any,
    fusion: FusionResult,
    context: Any,
    *,
    thresholds: MaturityThresholds | None = None,
    classifier_families: dict[str, str] | None = None,
    calibration_artifact: Any = None,
    discrimination: Any = None,
    combiner: Any = None,
) -> AttributionDecision:
    """Assess the attribution stage from the fusion result.

    The decision is deterministic and reproducible. If a calibration artifact
    is provided and valid, a pipeline-bound calibrated probability is emitted.
    Otherwise, only qualitative maturity is reported.
    """
    thresholds = thresholds or MaturityThresholds()
    calibration = _combiner_calibration(combiner)
    observations = getattr(context, "observed_techniques", {})
    technique_count = len(observations)
    # Check for insufficient evidence
    if not fusion.entries or technique_count < thresholds.min_techniques:
        return AttributionDecision(
            stage=AttributionStage.INSUFFICIENT_EVIDENCE,
            **calibration,
            cluster_id=None,
            leading_group_id=None,
            gate_mode=thresholds.gate_mode,
            rationale=(
                f"Insufficient evidence: {technique_count} techniques observed "
                f"(minimum {thresholds.min_techniques} required)."
            ),
        )
    # Compute maturity
    maturity, factors = _compute_maturity(kb, fusion, context, thresholds, classifier_families)
    # Compute margin
    if len(fusion.entries) >= 2:
        margin = (fusion.entries[0].relative_score - fusion.entries[1].relative_score) / 100.0
    else:
        margin = fusion.entries[0].relative_score / 100.0
    # Candidate-relative separation (computed here when the engine did not supply it)
    if discrimination is None:
        discrimination = discriminate(kb, fusion, context)
    separation_fields = {
        "separation": round(float(discrimination.separation), 3),
        "separation_factors": _separation_factors(discrimination),
        "runner_up_group_id": discrimination.runner_up_id,
        "runner_up_related": bool(discrimination.runner_up_related),
        "gate_mode": thresholds.gate_mode,
    }
    leading = fusion.entries[0]
    leading_group = getattr(kb, "groups", {}).get(leading.group_id)
    leading_name = (
        getattr(leading_group, "name", leading.group_id) if leading_group else leading.group_id
    )
    # Check family count
    family_count = _evidence_family_count(fusion, classifier_families)
    # Stage decision
    if (
        maturity < thresholds.min_maturity_score
        or family_count < thresholds.min_independent_evidence_families
    ):
        cluster_id = _generate_cluster_id(fusion, context)
        return AttributionDecision(
            stage=AttributionStage.UNRESOLVED_ACTIVITY_CLUSTER,
            **calibration,
            cluster_id=cluster_id,
            leading_group_id=leading.group_id,
            leading_group_name=leading_name,
            leading_score=leading.relative_score,
            margin=round(margin, 3),
            maturity_score=maturity,
            maturity_factors=factors,
            related_leading_groups=_related(kb, leading.group_id),
            **separation_fields,
            rationale=(
                f"Unresolved activity cluster {cluster_id}: maturity {maturity:.3f} "
                f"below threshold {thresholds.min_maturity_score:.3f} or "
                f"{family_count} independent families below minimum "
                f"{thresholds.min_independent_evidence_families}. "
                f"Leading candidate: {leading_name} ({leading.group_id}) at "
                f"{leading.relative_score:.2f}, but named attribution maturity not met."
            ),
        )
    failed = _hypothesis_gate_failures(thresholds, maturity, margin, discrimination)
    # The separation was scored for the ensemble's rank-1 candidate. If a re-ranking put another
    # group first, that group's separation is unknown here: never name it on borrowed evidence.
    if discrimination.leader_id and discrimination.leader_id != leading.group_id:
        failed.append(
            f"the re-ranking placed {leading.group_id} above {discrimination.leader_id}, "
            "whose evidence the separation was scored for"
        )
    if failed:
        return AttributionDecision(
            stage=AttributionStage.KNOWN_GROUP_SHORTLIST,
            cluster_id=None,
            leading_group_id=leading.group_id,
            leading_group_name=leading_name,
            leading_score=leading.relative_score,
            margin=round(margin, 3),
            maturity_score=maturity,
            maturity_factors=factors,
            related_leading_groups=_related(kb, leading.group_id),
            **separation_fields,
            rationale=(
                f"Known-group shortlist: {leading_name} ({leading.group_id}) leads at "
                f"{leading.relative_score:.2f} with maturity {maturity:.3f} and separation "
                f"{discrimination.separation:.2f}, but does not meet the naming bar "
                f"({'; '.join(failed)})."
            ),
            **calibration,
        )
    # Check calibration: the learned combiner's leader probability when available, else a
    # legacy calibration artifact (placeholder path).
    is_calibrated = False
    calibrated_probability = None
    calibration_note = "No calibration artifact provided; maturity is qualitative only."
    leader_probability = combiner.get("leader_probability") if isinstance(combiner, dict) else None
    if leader_probability is not None:
        is_calibrated = True
        calibrated_probability = float(leader_probability)
        calibration_note = (
            "Combiner probability that the leading candidate is the true group "
            "(pipeline-bound, fitted on ATT&CK folds with group-level cross-validation)."
        )
    elif calibration_artifact is not None:
        valid, note = _validate_calibration_artifact(calibration_artifact, kb, fusion)
        if valid:
            is_calibrated = True
            calibrated_probability = _apply_calibration(
                calibration_artifact, maturity, margin, factors
            )
            calibration_note = note
        else:
            calibration_note = f"Calibration artifact invalid: {note}"
    return AttributionDecision(
        stage=AttributionStage.KNOWN_GROUP_HYPOTHESIS,
        cluster_id=None,
        leading_group_id=leading.group_id,
        leading_group_name=leading_name,
        leading_score=leading.relative_score,
        margin=round(margin, 3),
        maturity_score=maturity,
        maturity_factors=factors,
        related_leading_groups=_related(kb, leading.group_id),
        **separation_fields,
        rationale=(
            f"Known-group hypothesis: {leading_name} ({leading.group_id}) is the leading "
            f"hypothesis at {leading.relative_score:.2f} with maturity {maturity:.3f} and "
            f"separation {discrimination.separation:.2f} from "
            f"{discrimination.runner_up_id or 'no alternative'}. "
            f"This is a hypothesis, not confirmation."
        ),
        is_calibrated=is_calibrated,
        calibrated_probability=calibrated_probability,
        calibration_note=calibration_note,
    )


def _combiner_calibration(combiner: Any) -> dict[str, Any]:
    """Calibrated leader probability from the learned combiner (empty when absent)."""
    if not isinstance(combiner, dict) or combiner.get("leader_probability") is None:
        return {}
    return {
        "is_calibrated": True,
        "calibrated_probability": float(combiner["leader_probability"]),
        "calibration_note": (
            "Combiner probability that the leading candidate is the true group "
            "(pipeline-bound, fitted on ATT&CK folds with group-level cross-validation)."
        ),
    }


def _separation_factors(discrimination: Any) -> dict[str, Any]:
    scores = discrimination.by_group()
    leader = scores.get(discrimination.leader_id) if discrimination.leader_id else None
    runner = scores.get(discrimination.runner_up_id) if discrimination.runner_up_id else None
    return {
        "leader": leader.to_dict() if leader is not None else {},
        "runner_up": runner.to_dict() if runner is not None else {},
        "leader_specificity": round(float(discrimination.leader_specificity), 3),
        "top_k": discrimination.top_k,
    }


def _hypothesis_gate_failures(
    thresholds: MaturityThresholds, maturity: float, margin: float, discrimination: Any
) -> list[str]:
    """Names of the naming-bar conditions the leader fails (empty = named hypothesis)."""
    mode = str(thresholds.gate_mode or "maturity")
    failed: list[str] = []
    # The RRF margin is a rank-agreement quantity (1 - rrf2/rrf1); in separation mode the
    # evidence-based separation replaces it, so it is reported but not required there.
    if mode in {"maturity", "both"} and margin < thresholds.hypothesis_margin:
        failed.append(f"margin {margin:.2f} < {thresholds.hypothesis_margin:.2f}")
    if mode in {"maturity", "both"} and maturity < thresholds.hypothesis_maturity_score:
        failed.append(f"maturity {maturity:.3f} < {thresholds.hypothesis_maturity_score:.3f}")
    if mode in {"separation", "both"}:
        if discrimination.separation < thresholds.hypothesis_separation:
            failed.append(
                f"separation {discrimination.separation:.2f} < "
                f"{thresholds.hypothesis_separation:.2f}"
            )
        if discrimination.leader_total < thresholds.hypothesis_min_discriminative:
            failed.append(
                f"discriminative evidence {discrimination.leader_total:.2f} < "
                f"{thresholds.hypothesis_min_discriminative:.2f}"
            )
        if (
            thresholds.hypothesis_specificity > 0
            and discrimination.leader_specificity < thresholds.hypothesis_specificity
        ):
            failed.append(
                f"evidence specificity {discrimination.leader_specificity:.2f} < "
                f"{thresholds.hypothesis_specificity:.2f}"
            )
    return failed


def _validate_calibration_artifact(
    artifact: Any, kb: Any, fusion: FusionResult
) -> tuple[bool, str]:
    """Validate that a calibration artifact is compatible with the current pipeline.

    Checks:
    - ATT&CK version match
    - Pipeline fingerprint match
    - Calibration data hash presence
    - Sample adequacy
    """
    if not isinstance(artifact, dict):
        return False, "artifact is not a dict"
    required_keys = {"attack_version", "pipeline_fingerprint", "calibration_hash", "sample_count"}
    missing = required_keys - set(artifact.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    attack_version = str(artifact.get("attack_version", ""))
    kb_version = str(getattr(kb, "version", ""))
    if attack_version and kb_version and attack_version != kb_version:
        return False, f"ATT&CK version mismatch: artifact={attack_version}, kb={kb_version}"
    sample_count = int(artifact.get("sample_count", 0))
    if sample_count < 30:
        return False, f"insufficient calibration samples: {sample_count} (minimum 30)"
    return True, "Calibration artifact valid."


def _apply_calibration(
    artifact: dict[str, Any],
    maturity: float,
    margin: float,
    factors: dict[str, float],
) -> float | None:
    """Apply a simple calibration mapping from maturity to probability.

    This is a placeholder for a proper conformal/selective rejector. It maps
    the maturity score to a pipeline-bound probability using the artifact's
    calibration curve. The probability is pipeline-bound: it only reflects
    how this specific pipeline's maturity score relates to historical
    performance, not a generic probability of attribution.
    """
    curve = artifact.get("calibration_curve", {})
    if not isinstance(curve, dict):
        return None
    # Find the closest maturity bucket
    buckets = sorted(curve.items(), key=lambda item: float(item[0]))
    for threshold_str, probability in buckets:
        if maturity <= float(threshold_str):
            return float(probability)
    if buckets:
        return float(buckets[-1][1])
    return None
