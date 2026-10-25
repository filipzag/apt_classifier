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
from apt_classifier.classifiers.base import kb_mapping
from apt_classifier.evidence.quality import (
    compute_technique_specificity,
)
from apt_classifier.fusion import FusionResult
from apt_classifier.separation import discriminate
from apt_classifier.software_questions import cluster_negative, leader_only_negative


class AttributionStage(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRESOLVED_ACTIVITY_CLUSTER = "unresolved_activity_cluster"
    KNOWN_GROUP_SHORTLIST = "known_group_shortlist"
    KNOWN_GROUP_HYPOTHESIS = "known_group_hypothesis"
    # two groups ATT&CK documents as overlapping, named together (overlap_cluster.py, docs 10.11)
    KNOWN_CLUSTER_HYPOTHESIS = "known_cluster_hypothesis"


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
    # documented-overlap cluster named instead of a single group (leader first), with the merged
    # pair's separation and mass against the rest of the gate window and its combiner probability
    named_cluster: tuple[str, ...] = ()
    cluster_separation: float = 0.0
    cluster_discriminative: float = 0.0
    cluster_probability: float | None = None
    cluster_reason: str = ""
    cluster_runner_up_id: str | None = None

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
            "named_cluster": list(self.named_cluster),
            "cluster_separation": self.cluster_separation,
            "cluster_discriminative": self.cluster_discriminative,
            "cluster_probability": self.cluster_probability,
            "cluster_reason": self.cluster_reason,
            "cluster_runner_up_id": self.cluster_runner_up_id,
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
    # Naming probability floor (prereg attribution_probability_floor_2026-10-25, docs 10.6): a
    # single name also needs the learned combiner's probability for the leader at or above this
    # (0: no floor; without an active combiner there is no probability and no floor). A leader it
    # holds can still be named through its documented cluster, which has its own probability bar.
    hypothesis_min_probability: float = 0.0
    # Documented-overlap clusters (overlap_cluster.py, docs 10.11): off unless enabled. A cluster
    # needs the separation bar, this mass (the single bar for a nested pair) and the members'
    # combined combiner probability. The scope says which ATT&CK links make a twin: "documented"
    # (ATT&CK documents the same activity, attack/same_activity.py, or the profiles nest), "full"
    # (any ATT&CK relation, shared rare malware included) or "nested" (nesting profiles only);
    # the engine selects the twin under it.
    cluster_naming: bool = False
    cluster_min_probability: float = 0.8
    cluster_min_discriminative: float = 3.5
    cluster_scope: str = "documented"
    # Evidence-volume guard (2026-10-19, prereg cribl_cluster_volume_guard): a cluster is held at
    # shortlist when the case's accepted technique count exceeds this (0: no guard), unless
    # decisive software supports the leader. The bars above were set on 3-23 techniques; a
    # commodity malware write-up with 50-95 accepted techniques clears them on mass alone.
    cluster_max_techniques: int = 0


SUPPORT_RANK_LIMIT = 10


# Members that re-read evidence other families already score: they rank, but never count as an
# independent evidence family for the naming gate. The free-text model (text_group.py) scores
# the same case narrative the technique, software and behaviour classifiers read.
NOT_INDEPENDENT_EVIDENCE = frozenset({"text_group", "behavior_text", "procedure_match"})


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
        if classifier_name in NOT_INDEPENDENT_EVIDENCE:
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
    negative_software: Any = None,
    negative_software_weight: float = 0.3,
    software_guard: dict[str, Any] | None = None,
) -> AttributionDecision:
    """Assess the attribution stage from the fusion result.

    The decision is deterministic and reproducible. If a calibration artifact
    is provided and valid, a pipeline-bound calibrated probability is emitted.
    Otherwise, only qualitative maturity is reported.

    Software interview answers (software_questions.py, docs 21.3.15 C) enter in a fixed order:
    (1) the single-name gate, which also holds the "found" rules of ``software_guard`` (a name
    must not rest on interview answers alone, and a name the interview software enabled must
    belong to a group that documents found malware); (2) the documented-cluster step on the
    discrimination before any "not found" answer, under the same rules; (3) the "not found"
    filter last (``negative_software``): it lowers only the unit holding the leader and may only
    move a hypothesis or a cluster to the shortlist, never back into step 2.
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
    # step 3's input, computed up front so every stage reports the separation it was judged on
    not_found = tuple(sorted({str(item) for item in (negative_software or ()) if item}))
    leader_effect = (
        leader_only_negative(discrimination, kb, not_found, negative_software_weight)
        if not_found and discrimination.scores
        else None
    )
    reported_separation = float(discrimination.separation)
    if leader_effect is not None:
        reported_separation = float(leader_effect.separation)
        separation_fields["separation"] = round(reported_separation, 3)
        separation_fields["separation_factors"] = {
            **separation_fields["separation_factors"],
            "negative_software": leader_effect.to_dict(),
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
    guard_fired = bool(discrimination.leader_id and discrimination.leader_id != leading.group_id)
    if guard_fired:
        failed.append(
            f"the re-ranking placed {leading.group_id} above {discrimination.leader_id}, "
            "whose evidence the separation was scored for"
        )
    floor = float(getattr(thresholds, "hypothesis_min_probability", 0.0) or 0.0)
    floor_probability = combiner.get("leader_probability") if isinstance(combiner, dict) else None
    if floor > 0 and floor_probability is not None and float(floor_probability) < floor:
        failed.append(
            f"combiner probability {float(floor_probability):.2f} below the naming floor "
            f"{floor:.2f}"
        )
    # (1) the single gate holds the "found" rules too
    guard = (
        software_guard if isinstance(software_guard, dict) and software_guard.get("found") else None
    )
    if guard is not None:
        failed.extend(_found_failures(kb, guard, (leading.group_id,), cluster=False))
    # (2) the cluster step, on the discrimination before any "not found" answer
    decisive = getattr(context, "decisive_software", None)
    decisive_factor = (
        float(decisive.get("factor", 1.0) or 1.0) if isinstance(decisive, dict) else 1.0
    )
    cluster_step, candidate = _cluster_step(
        thresholds,
        combiner,
        leading.group_id,
        guard_fired,
        technique_count=technique_count,
        decisive_factor=decisive_factor,
    )
    cluster_blocked: list[str] = []
    if cluster_step == "cluster" and candidate is not None and guard is not None:
        members = tuple(str(gid) for gid in candidate["members"])
        cluster_blocked = _found_failures(kb, guard, members, cluster=True)
        if cluster_blocked:
            cluster_step = ""
    cluster_fields = {
        "leading_group_id": leading.group_id,
        "leading_group_name": leading_name,
        "leading_score": leading.relative_score,
        "margin": round(margin, 3),
        "maturity_score": maturity,
        "maturity_factors": factors,
        "related_leading_groups": _related(kb, leading.group_id),
        **separation_fields,
        **calibration,
    }
    # a leader that clears alone keeps its single name unless its twin nests inside it
    if cluster_step == "cluster" and candidate is not None and (failed or candidate.get("nested")):
        # (3) the "not found" filter on the cluster: it can only send it to the shortlist
        demoted = _cluster_negative_failures(
            kb, thresholds, candidate, discrimination, not_found, negative_software_weight
        )
        if not demoted:
            return _cluster_decision(kb, candidate, cluster_fields)
        failed = [*failed, *demoted]
    elif not failed and candidate is not None and candidate.get("nested"):
        # the leader clears alone, but its nested documented twin can never be ruled out by the
        # leader's own evidence (all of the twin's documented techniques are the leader's too)
        twin = str(candidate["members"][1])
        failed.append(
            f"documented nested twin {twin} in the window cannot be ruled out and the pair does "
            f"not clear the cluster bars (cluster separation {candidate['separation']:.2f}, "
            f"mass {candidate['discriminative']:.2f}"
            + (f"; {'; '.join(cluster_blocked)}" if cluster_blocked else "")
            + ")"
        )
    if not failed and leader_effect is not None:
        # (3) the "not found" filter on the single name: it can only add failed conditions
        failed.extend(
            _negative_failures(
                thresholds,
                float(discrimination.separation),
                float(leader_effect.separation),
                float(discrimination.leader_total),
                float(leader_effect.total),
                minimum=float(thresholds.hypothesis_min_discriminative),
                label="the leader's",
            )
        )
    if failed and cluster_blocked:
        # say why the cluster was not named either (the nested hold-back already says it)
        joined = " ".join(failed)
        failed.extend([reason for reason in cluster_blocked if reason not in joined])
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
                f"{reported_separation:.2f}, but does not meet the naming bar "
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
            f"separation {reported_separation:.2f} from "
            f"{discrimination.runner_up_id or 'no alternative'}. "
            f"This is a hypothesis, not confirmation."
        ),
        is_calibrated=is_calibrated,
        calibrated_probability=calibrated_probability,
        calibration_note=calibration_note,
    )


def _cluster_step(
    thresholds: MaturityThresholds,
    combiner: Any,
    leader_id: str,
    guard_fired: bool,
    *,
    technique_count: int | None = None,
    decisive_factor: float = 1.0,
) -> tuple[str, dict[str, Any] | None]:
    """("cluster", candidate) when the leader's documented-overlap cluster clears its bars,
    ("", candidate) when a candidate exists but does not, ("", None) otherwise. The candidate is
    computed by the engine (overlap_cluster.cluster_candidate) and recorded on the combiner
    state; without an active combiner there is no probability and so no cluster.

    ``technique_count`` (the case's accepted techniques) and ``decisive_factor`` (the
    decisive-software factor, > 1 when group-exclusive malware is named) feed the evidence-volume
    guard: with ``cluster_max_techniques`` set, a cluster on more techniques than that is held
    unless decisive software supports the leader; the held candidate records ``volume_guard``."""
    if not thresholds.cluster_naming or thresholds.gate_mode == "maturity" or guard_fired:
        return "", None
    candidate = combiner.get("cluster") if isinstance(combiner, dict) else None
    if not isinstance(candidate, dict) or not candidate.get("members"):
        return "", None
    if str(candidate["members"][0]) != leader_id:
        return "", None
    nested = bool(candidate.get("nested"))
    if thresholds.cluster_scope == "nested" and not nested:
        return "", None
    # the engine already selects the twin under the scope; this keeps a candidate recorded under
    # a wider scope (or before the link field existed) from naming a pair ATT&CK does not
    # document as the same activity
    if thresholds.cluster_scope == "documented" and candidate.get("link") not in (
        "same_activity",
        "nesting",
    ):
        return "", None
    probability = candidate.get("probability")
    bar = (
        thresholds.hypothesis_min_discriminative
        if nested
        else max(thresholds.hypothesis_min_discriminative, thresholds.cluster_min_discriminative)
    )
    passes = (
        probability is not None
        and float(probability) >= thresholds.cluster_min_probability
        and float(candidate.get("separation", 0.0)) >= thresholds.hypothesis_separation
        and float(candidate.get("discriminative", 0.0)) >= bar
    )
    held = volume_guard_holds(thresholds, technique_count, decisive_factor)
    if passes and held is not None:
        return "", {**candidate, "volume_guard": held}
    return ("cluster" if passes else ""), candidate


def volume_guard_holds(
    thresholds: MaturityThresholds, technique_count: int | None, decisive_factor: float
) -> dict[str, Any] | None:
    """Why the evidence-volume guard holds a cluster (``None`` when it does not): more accepted
    techniques than ``cluster_max_techniques`` and no decisive software for the leader."""
    limit = int(getattr(thresholds, "cluster_max_techniques", 0) or 0)
    if limit <= 0 or technique_count is None or int(technique_count) <= limit:
        return None
    if float(decisive_factor or 1.0) > 1.0:
        return None
    return {
        "reason": "evidence volume above the cluster gate's validated range, no decisive software",
        "technique_count": int(technique_count),
        "limit": limit,
        "decisive_factor": float(decisive_factor or 1.0),
    }


def _cluster_decision(
    kb: Any, candidate: dict[str, Any], fields: dict[str, Any]
) -> AttributionDecision:
    from apt_classifier.overlap_cluster import cluster_label

    members = tuple(str(gid) for gid in candidate["members"])
    label = cluster_label(kb, members)
    runner = candidate.get("runner_up_id")
    return AttributionDecision(
        stage=AttributionStage.KNOWN_CLUSTER_HYPOTHESIS,
        cluster_id=None,
        named_cluster=members,
        cluster_separation=round(float(candidate.get("separation", 0.0)), 3),
        cluster_discriminative=round(float(candidate.get("discriminative", 0.0)), 3),
        cluster_probability=(
            round(float(candidate["probability"]), 4)
            if candidate.get("probability") is not None
            else None
        ),
        cluster_reason=str(candidate.get("reason", "")),
        cluster_runner_up_id=runner,
        rationale=(
            f"Documented-overlap cluster: {label}. ATT&CK link: {candidate.get('reason', '')}"
            f". Together the pair separates from {runner or 'the rest of the shortlist'} "
            f"(cluster separation {float(candidate.get('separation', 0.0)):.2f}, mass "
            f"{float(candidate.get('discriminative', 0.0)):.2f}, combined probability "
            f"{float(candidate.get('probability') or 0.0):.2f}); the evidence does not separate "
            "the two members. Report the cluster, not either member. This is a hypothesis, "
            "not confirmation."
        ),
        **fields,
    )


def _found_failures(
    kb: Any, guard: dict[str, Any], members: tuple[str, ...], *, cluster: bool
) -> list[str]:
    """The "found" rules for one named unit (the leader, or the cluster's members).

    - With ``requires_lead`` the unit's separation in the gate window without the interview
      software (computed by the engine) must be >= 0: a name never rests on answers alone.
    - When the case would not be named without the interview software (``named_without`` False),
      a member must document at least one found malware: a "found" only enables its own group."""
    who = "the cluster" if cluster else "the name"
    reasons: list[str] = []
    if guard.get("requires_lead", True):
        value = guard.get("cluster_separation_without" if cluster else "separation_without")
        if value is None:
            reasons.append(f"{who} cannot be checked without the interview software")
        elif float(value) < 0:
            reasons.append(
                f"{who} would rest on interview answers alone (separation {float(value):.2f} "
                "< 0 without the found software)"
            )
    if guard.get("named_without") is False:
        found = {str(item) for item in guard.get("found") or ()}
        group_software = kb_mapping(kb, "group_software")
        if not any(found & set(group_software.get(gid, ()) or ()) for gid in members):
            reasons.append(
                f"only the interview software names the case and {' / '.join(members)} "
                "documents none of the found malware"
            )
    return reasons


def _negative_failures(
    thresholds: MaturityThresholds,
    separation_before: float,
    separation: float,
    total_before: float,
    total: float,
    *,
    minimum: float,
    label: str,
) -> list[str]:
    """Naming-bar conditions that the "not found" answers newly fail (separation-type gates
    only; the maturity bar never reads the separation)."""
    if str(thresholds.gate_mode or "maturity") not in {"separation", "both"}:
        return []
    failed: list[str] = []
    bar = float(thresholds.hypothesis_separation)
    if separation_before >= bar > separation:
        failed.append(
            f"{label} separation {separation:.2f} < {bar:.2f} after the not-found interview "
            f"answers ({separation_before:.2f} before)"
        )
    if total_before >= minimum > total:
        failed.append(
            f"{label} discriminative evidence {total:.2f} < {minimum:.2f} after the not-found "
            f"interview answers ({total_before:.2f} before)"
        )
    return failed


def _cluster_negative_failures(
    kb: Any,
    thresholds: MaturityThresholds,
    candidate: dict[str, Any],
    discrimination: Any,
    not_found: tuple[str, ...],
    weight: float,
) -> list[str]:
    """Step 3 for a named cluster: the merged pair's mass and separation after the "not found"
    answers any member documents, against the same cluster bars."""
    if not not_found:
        return []
    effect = cluster_negative(candidate, discrimination, kb, not_found, weight)
    if effect.penalty <= 0:
        return []
    nested = bool(candidate.get("nested"))
    minimum = (
        thresholds.hypothesis_min_discriminative
        if nested
        else max(thresholds.hypothesis_min_discriminative, thresholds.cluster_min_discriminative)
    )
    return _negative_failures(
        thresholds,
        effect.separation_before,
        effect.separation,
        effect.total_before,
        effect.total,
        minimum=float(minimum),
        label="the cluster's",
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
