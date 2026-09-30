"""Calibrated per-candidate acceptance for technique mapping.

The rule-based gate in :mod:`apt_classifier.retrieve` accepts at most one
similarity-only mapping per observation. Public CTI sentences carry 1-4 labels each, so a
learned, per-candidate acceptance score is needed to recover more than one technique
without hand-tuned thresholds.

This module is deliberately small and dependency-free:

- :func:`candidate_features` turns a retrieval candidate (with its siblings for the same
  observation) into a fixed feature vector built only from evidence the pipeline already
  records (fused score/rank, dense similarity and field, lexical rank, margins, cue flags,
  hierarchy, benign/sparse context).
- :class:`AcceptanceModel` is an L2-regularised logistic regression fitted by Newton's
  method on the public *dev* splits (never on test), with the acceptance threshold chosen
  at a precision-constrained operating point. It is persisted as JSON with the ATT&CK
  version, a feature-schema version and dev metrics, and loads fail-closed: a mismatch
  disables the model and the rule-based gate stays in force.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.exemplars import EVIDENCE_SOURCE as EXEMPLAR_EVIDENCE_SOURCE
from apt_classifier.exemplars import FEATURE_NAMES as EXEMPLAR_FEATURE_NAMES
from apt_classifier.exemplars import features_from_evidence as exemplar_features_from_evidence

# Schema 3 (2026-09-30): exemplar features, i.e. what the nearest labelled real-report
# sentences say about the candidate (``apt_classifier.exemplars``). A model fitted with them
# records the exemplar index fingerprint and refuses to run without that index.
FEATURE_SCHEMA_VERSION = 3

# Per-technique priors (schema 2): how often the technique is labelled in the public *train*
# splits, how many ATT&CK procedure sentences it has, and how many groups document it. They
# are computed once at fit time (``compute_technique_priors``), stored inside the model
# artifact and looked up by technique id at inference, so the runtime needs no training data.
PRIOR_FEATURE_NAMES: tuple[str, ...] = (
    "technique_prior_log_freq",
    "technique_procedure_log_count",
    "technique_group_log_count",
)
FEATURE_NAMES: tuple[str, ...] = (
    "fused_score",
    "inverse_fused_rank",
    "is_rank_one",
    "fused_margin_to_top",
    "fused_margin_to_next",
    "dense_score",
    "dense_present",
    "dense_rank_one",
    "dense_margin",
    "dense_field_name",
    "dense_field_description",
    "dense_field_procedure",
    "lexical_present",
    "lexical_rank_le3",
    "lexical_rank_le5",
    "inverse_lexical_rank",
    "structured_signal",
    "explicit_cue",
    "reranker_score",
    "is_subtechnique",
    "hierarchy_parent",
    "benign_context",
    "sparse_context",
    "candidate_count",
    *PRIOR_FEATURE_NAMES,
    *EXEMPLAR_FEATURE_NAMES,
)


def compute_technique_priors(kb: Any, label_counts: Mapping[str, int]) -> dict[str, list[float]]:
    """Prior features per technique id from the knowledge base and train-split label counts."""
    from apt_classifier.vectorstore import procedure_sentences_by_technique

    groups_per_technique: dict[str, int] = {}
    for group in getattr(kb, "groups", {}).values():
        for technique_id in set(getattr(group, "technique_ids", ()) or ()):
            groups_per_technique[technique_id] = groups_per_technique.get(technique_id, 0) + 1
    procedures = procedure_sentences_by_technique(kb)
    priors: dict[str, list[float]] = {}
    for technique_id in getattr(kb, "techniques", {}):
        priors[technique_id] = [
            math.log1p(float(label_counts.get(technique_id, 0))),
            math.log1p(float(len(procedures.get(technique_id, ())))),
            math.log1p(float(groups_per_technique.get(technique_id, 0))),
        ]
    return priors


def prior_features(priors: Mapping[str, Sequence[float]], technique_id: str) -> dict[str, float]:
    """The prior feature dict for one technique; unknown techniques get zeros (count 0)."""
    values = priors.get(technique_id) or [0.0] * len(PRIOR_FEATURE_NAMES)
    return {name: float(value) for name, value in zip(PRIOR_FEATURE_NAMES, values, strict=True)}


def _score(evidence: Any) -> float:
    return float(evidence.score) if evidence is not None and evidence.score is not None else 0.0


def candidate_features(
    candidate: Any,
    siblings: Sequence[Any],
    *,
    benign_context: bool,
    sparse_context: bool,
) -> dict[str, float]:
    """Feature vector for one candidate given every candidate of the same observation."""
    ordered = sorted(siblings, key=lambda item: (item.fused_rank, item.technique_id))
    scores = [float(item.fused_score) for item in ordered]
    top_score = scores[0] if scores else 0.0
    index = next(
        (i for i, item in enumerate(ordered) if item.technique_id == candidate.technique_id),
        None,
    )
    next_score = scores[index + 1] if index is not None and index + 1 < len(scores) else 0.0
    dense_scores = sorted(
        (
            _score(item.evidence_for("dense"))
            for item in ordered
            if item.evidence_for("dense") is not None
        ),
        reverse=True,
    )
    dense_margin = (
        (dense_scores[0] - dense_scores[1])
        if len(dense_scores) > 1
        else (dense_scores[0] if dense_scores else 0.0)
    )
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    reranker = candidate.evidence_for("reranker")
    dense_field = str(dense.field or "") if dense is not None else ""
    lexical_rank = int(lexical.rank or 999) if lexical is not None else 999
    fused_score = float(candidate.fused_score)
    return {
        "fused_score": fused_score,
        "inverse_fused_rank": 1.0 / max(int(candidate.fused_rank), 1),
        "is_rank_one": 1.0 if int(candidate.fused_rank) == 1 else 0.0,
        "fused_margin_to_top": top_score - fused_score,
        "fused_margin_to_next": fused_score - next_score,
        "dense_score": _score(dense),
        "dense_present": 1.0 if dense is not None else 0.0,
        "dense_rank_one": 1.0 if dense is not None and dense.rank == 1 else 0.0,
        "dense_margin": dense_margin,
        "dense_field_name": 1.0 if dense_field == "name" else 0.0,
        "dense_field_description": 1.0 if dense_field == "description" else 0.0,
        "dense_field_procedure": 1.0 if dense_field == "procedure" else 0.0,
        "lexical_present": 1.0 if lexical is not None else 0.0,
        "lexical_rank_le3": 1.0 if lexical_rank <= 3 else 0.0,
        "lexical_rank_le5": 1.0 if lexical_rank <= 5 else 0.0,
        "inverse_lexical_rank": 1.0 / lexical_rank if lexical is not None else 0.0,
        "structured_signal": 1.0 if candidate.evidence_for("structured_signal") else 0.0,
        "explicit_cue": 1.0 if candidate.evidence_for("explicit_cue") else 0.0,
        "reranker_score": _score(reranker),
        "is_subtechnique": 1.0 if getattr(candidate, "is_subtechnique", False) else 0.0,
        "hierarchy_parent": 1.0 if candidate.evidence_for("hierarchy_parent") else 0.0,
        "benign_context": 1.0 if benign_context else 0.0,
        "sparse_context": 1.0 if sparse_context else 0.0,
        "candidate_count": float(len(ordered)),
        **exemplar_features_from_evidence(candidate.evidence_for(EXEMPLAR_EVIDENCE_SOURCE)),
    }


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iterations: int = 30
) -> tuple[np.ndarray, float]:
    """Newton-method logistic regression; returns (weights, intercept)."""
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("feature matrix and labels are misaligned")
    n, d = X.shape
    design = np.hstack([np.ones((n, 1)), X])
    weights = np.zeros(d + 1)
    penalty = l2 * np.diag(np.r_[0.0, np.ones(d)])
    for _ in range(iterations):
        z = np.clip(design @ weights, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (p - y) + penalty @ weights
        w = p * (1.0 - p) + 1e-9
        hessian = (design * w[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights[1:], float(weights[0])


@dataclass
class AcceptanceModel:
    """Logistic acceptance scorer with standardized inputs and a chosen threshold."""

    weights: np.ndarray
    intercept: float
    means: np.ndarray
    scales: np.ndarray
    threshold: float
    attack_version: str
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    technique_priors: dict[str, list[float]] = field(default_factory=dict)

    def prior_features(self, technique_id: str) -> dict[str, float]:
        return prior_features(self.technique_priors, technique_id)

    def probability(self, features: dict[str, float]) -> float:
        x = (feature_vector(features) - self.means) / self.scales
        z = float(np.clip(x @ self.weights + self.intercept, -30.0, 30.0))
        return 1.0 / (1.0 + math.exp(-z))

    def accepts(self, features: dict[str, float]) -> bool:
        return self.probability(features) >= self.threshold

    @property
    def exemplar_fingerprint(self) -> str | None:
        value = self.metadata.get("exemplar_fingerprint")
        return str(value) if value else None

    @property
    def requires_exemplars(self) -> bool:
        return self.exemplar_fingerprint is not None

    def compatible_with(
        self, attack_version: str | None, exemplar_fingerprint: str | None = None
    ) -> tuple[bool, str]:
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            return False, (
                f"feature schema {self.feature_schema_version} != {FEATURE_SCHEMA_VERSION}"
            )
        if attack_version and self.attack_version and attack_version != self.attack_version:
            return False, f"ATT&CK version {self.attack_version} != {attack_version}"
        if self.requires_exemplars and exemplar_fingerprint != self.exemplar_fingerprint:
            return False, (
                "fitted with exemplar index "
                f"{self.exemplar_fingerprint}, runtime has "
                f"{exemplar_fingerprint or 'none'} (run make exemplars, or refit)"
            )
        return True, "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "technique_acceptance_logistic",
            "created_at": datetime.now(UTC).isoformat(),
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(value) for value in self.weights],
            "intercept": self.intercept,
            "means": [float(value) for value in self.means],
            "scales": [float(value) for value in self.scales],
            "threshold": self.threshold,
            "attack_version": self.attack_version,
            "metadata": self.metadata,
            "technique_priors": {
                key: [float(v) for v in values]
                for key, values in sorted(self.technique_priors.items())
            },
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), "utf-8")
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AcceptanceModel:
        if payload.get("kind") != "technique_acceptance_logistic":
            raise ValueError("not an acceptance model artifact")
        names = tuple(payload.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError("feature names do not match the current schema")
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            intercept=float(payload["intercept"]),
            means=np.asarray(payload["means"], dtype=float),
            scales=np.asarray(payload["scales"], dtype=float),
            threshold=float(payload["threshold"]),
            attack_version=str(payload.get("attack_version", "")),
            feature_schema_version=int(payload.get("feature_schema_version", 0)),
            metadata=dict(payload.get("metadata") or {}),
            technique_priors={
                str(key): [float(v) for v in values]
                for key, values in (payload.get("technique_priors") or {}).items()
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> AcceptanceModel | None:
        source = Path(path)
        if not source.is_file():
            return None
        return cls.from_dict(json.loads(source.read_text("utf-8")))

    @classmethod
    def fit(
        cls,
        rows: Sequence[dict[str, float]],
        labels: Sequence[int],
        *,
        attack_version: str,
        l2: float = 1.0,
        threshold: float = 0.5,
        metadata: dict[str, Any] | None = None,
        technique_priors: Mapping[str, Sequence[float]] | None = None,
    ) -> AcceptanceModel:
        X = np.vstack([feature_vector(row) for row in rows])
        y = np.asarray(labels, dtype=float)
        means = X.mean(axis=0)
        scales = X.std(axis=0)
        scales[scales < 1e-9] = 1.0
        weights, intercept = fit_logistic((X - means) / scales, y, l2=l2)
        return cls(
            weights=weights,
            intercept=intercept,
            means=means,
            scales=scales,
            threshold=threshold,
            attack_version=attack_version,
            metadata=dict(metadata or {}),
            technique_priors={
                k: [float(v) for v in vals] for k, vals in (technique_priors or {}).items()
            },
        )


def choose_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    case_ids: Sequence[str],
    expected_counts: dict[str, int],
    *,
    min_precision: float = 0.5,
    grid: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Pick the threshold maximizing micro-F1 of the accepted set subject to a precision floor.

    ``expected_counts`` gives the number of expected labels per case so recall counts
    labels that never appeared as candidates (they are unrecoverable false negatives).
    """
    total_expected = sum(expected_counts.values())
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    candidates = sorted(set(grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)))
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = probs >= threshold
        tp = int((accepted & (truth == 1)).sum())
        fp = int((accepted & (truth == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_expected if total_expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        entry = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "micro_f1": f1,
            "accepted": int(accepted.sum()),
        }
        if fallback is None or f1 > fallback["micro_f1"]:
            fallback = entry
        if precision >= min_precision and (best is None or f1 > best["micro_f1"]):
            best = entry
    chosen = (
        best or fallback or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "micro_f1": 0.0}
    )
    chosen["precision_floor_met"] = best is not None
    return chosen


def load_acceptance_model(
    settings: Any, kb: Any, exemplar_index: Any = None
) -> tuple[AcceptanceModel | None, str]:
    """Load the configured acceptance model fail-closed.

    Returns ``(model, status)``; ``model`` is ``None`` with a reason whenever the feature
    is disabled, the artifact is missing, malformed, built for another ATT&CK version, or
    fitted with an exemplar index that the runtime does not have (``exemplar_index``).

    When the primary artifact cannot run, ``acceptance_model_fallback_path`` is tried: a model
    fitted the same way without exemplar features (``make fit-acceptance`` writes both). It is
    used only if it needs no exemplar index, so turning exemplars off, or losing the index,
    restores the previous calibrated gate instead of the rule-based one.
    """
    if not bool(getattr(settings, "acceptance_model_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "acceptance_model_path", None)
    if not path:
        return None, "no path configured"
    model, status = _load_compatible(path, kb, exemplar_index)
    if model is not None:
        return model, status
    fallback = getattr(settings, "acceptance_model_fallback_path", None)
    if fallback and Path(fallback) != Path(path):
        alternative, _ = _load_compatible(fallback, kb, None)
        if alternative is not None and not alternative.requires_exemplars:
            return alternative, f"fallback without exemplars ({status})"
    return None, status


def _load_compatible(path: Any, kb: Any, exemplar_index: Any) -> tuple[AcceptanceModel | None, str]:
    try:
        model = AcceptanceModel.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path}"
    ok, reason = model.compatible_with(
        str(getattr(kb, "version", "") or ""),
        getattr(exemplar_index, "fingerprint", None) if exemplar_index is not None else None,
    )
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, "active"
