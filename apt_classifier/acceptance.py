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
- :func:`retrieval_contract` records every retrieval setting the features depend on (pool,
  margins, depth, fields, platform scoping, the exemplar library and its vote settings). The
  fit stores it in the artifact; the loader and ``retrieve_techniques`` refuse a model whose
  contract the runtime does not meet (:func:`contract_mismatches`), so a changed knob can never
  silently feed a model features it was not fitted on. Artifacts from before the contract are
  held only to the knobs that are new with it (:data:`CONTRACT_1_KEYS`, :func:`legacy_contract`),
  so every configuration that ran them before still does.
- :func:`choose_threshold_runtime` picks the threshold on the accepted sets the runtime gate
  would produce (guards, cue-supported candidates, hierarchy parents), not on ``p`` alone.
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

from apt_classifier.exemplars import DEFAULT_SEGMENT_VOTES, DEFAULT_VOTE_NORM
from apt_classifier.exemplars import EVIDENCE_SOURCE as EXEMPLAR_EVIDENCE_SOURCE
from apt_classifier.exemplars import FEATURE_NAMES as EXEMPLAR_FEATURE_NAMES
from apt_classifier.exemplars import features_from_evidence as exemplar_features_from_evidence
from apt_classifier.technique_head import DEFAULT_NEW_CANDIDATES as HEAD_NEW_CANDIDATES
from apt_classifier.technique_head import DEFAULT_TOP_FAMILIES as HEAD_TOP_FAMILIES
from apt_classifier.technique_head import EVIDENCE_SOURCE as TECHNIQUE_HEAD_EVIDENCE_SOURCE
from apt_classifier.technique_head import FEATURE_NAMES as TECHNIQUE_HEAD_FEATURE_NAMES
from apt_classifier.technique_head import MULTI_FEATURE_NAMES as TECHNIQUE_HEAD_MULTI_NAMES
from apt_classifier.technique_head import features_from_evidence as head_features_from_evidence

# Schema 3 (2026-09-30): exemplar features, i.e. what the nearest labelled real-report
# sentences say about the candidate (``apt_classifier.exemplars``). A model fitted with them
# records the exemplar index fingerprint and refuses to run without that index.
# Schema 4 (2026-10-19): schema 3 plus the technique head's features
# (``apt_classifier.technique_head``). A model fitted with the head records the head's
# checkpoint fingerprint and refuses to run on another head or without one. Schema-3 artifacts
# keep loading unchanged (their feature vector is the schema-3 prefix): ``feature_names_for``.
# Schema 5 (2026-10-19): schema 4 plus the head's relative probabilities (its second answer) and
# the statement length (``technique_head.MULTI_FEATURE_NAMES``; prereg
# cribl_multi_answer_acceptance_2026-10-19).
FEATURE_SCHEMA_VERSION = 5
SUPPORTED_FEATURE_SCHEMAS: tuple[int, ...] = (3, 4, 5)

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
# schema 4 = schema 3 followed by the technique head's features
FEATURE_NAMES_V4: tuple[str, ...] = (*FEATURE_NAMES, *TECHNIQUE_HEAD_FEATURE_NAMES)
# schema 5 = schema 4 followed by the head's second-answer features
FEATURE_NAMES_V5: tuple[str, ...] = (*FEATURE_NAMES_V4, *TECHNIQUE_HEAD_MULTI_NAMES)


def feature_names_for(schema_version: int) -> tuple[str, ...]:
    """The feature vector of a schema version (``ValueError`` for an unknown one)."""
    if int(schema_version) == 3:
        return FEATURE_NAMES
    if int(schema_version) == 4:
        return FEATURE_NAMES_V4
    if int(schema_version) == 5:
        return FEATURE_NAMES_V5
    raise ValueError(f"unknown acceptance feature schema {schema_version!r}")


# ----- retrieval contract -----
# Contract 1 (2026-10-02). Keys are compared exactly; the exemplar keys only for a model fitted
# with exemplars (a no-exemplar model is compared on the others, so it can stand in whenever the
# exemplar side differs). A key a stored contract lacks is compared against its legacy value,
# i.e. the behaviour before the key existed. Version 0 is an artifact from before contracts
# (:func:`legacy_contract`): only :data:`CONTRACT_1_KEYS` are compared.
RETRIEVAL_CONTRACT_VERSION = 1
LEGACY_CONTRACT_VERSION = 0
DEFAULT_DENSE_COLLECTION = "attack_techniques"
EXEMPLAR_CONTRACT_KEYS: tuple[str, ...] = (
    "exemplar_fingerprint",
    "exemplar_k",
    "exemplar_tau",
    "exemplar_duplicate_similarity",
    "exemplar_new_candidates",
    "exemplar_vote_norm",
    "exemplar_segment_votes",
    "exemplar_search_class",
)
# Compared only for a model fitted with the technique head (``requires_technique_head``), as the
# exemplar keys are for a model fitted with exemplars; the fingerprint is the head checkpoint's.
TECHNIQUE_HEAD_CONTRACT_KEYS: tuple[str, ...] = (
    "technique_head_fingerprint",
    "technique_head_new_candidates",
    "technique_head_top_families",
)
RETRIEVAL_CONTRACT_KEYS: tuple[str, ...] = (
    "similarity_threshold",
    "top_k",
    "candidate_pool",
    "retrieval_mode",
    "parent_policy",
    "include_detection",
    "include_synthetic",
    "lexical_fields",
    "platform_from_text",
    "segment_acceptance",
    "hierarchy_parent_mode",
    "rrf_k",
    "dense_relative_margin",
    "dense_overfetch",
    "dense_collection",
    *EXEMPLAR_CONTRACT_KEYS,
    *TECHNIQUE_HEAD_CONTRACT_KEYS,
)
# Recorded in a contract but never compared: where the exemplar library was read from ("npz" or
# "chroma"). The library's identity is ``exemplar_fingerprint`` (recomputed from the vectors), so
# two backends serving bit-identical vectors are the same library; Chroma re-normalises on
# insert, so its library has its own fingerprint and a model fitted on the npz is refused there.
CONTRACT_RECORD_KEYS: tuple[str, ...] = ("exemplar_backend",)
_FLOAT_KEYS = frozenset(
    {
        "similarity_threshold",
        "dense_relative_margin",
        "exemplar_tau",
        "exemplar_duplicate_similarity",
    }
)
_INT_KEYS = frozenset(
    {
        "top_k",
        "candidate_pool",
        "rrf_k",
        "dense_overfetch",
        "exemplar_k",
        "exemplar_new_candidates",
        "technique_head_new_candidates",
        "technique_head_top_families",
    }
)
_BOOL_KEYS = frozenset(
    {
        "include_detection",
        "include_synthetic",
        "platform_from_text",
        "segment_acceptance",
    }
)
# The knobs that did not exist before contract 1. A model artifact without a stored contract
# can only have been fitted with their one legacy value, so it is held to that value; every knob
# that existed before (the presets, the Streamlit retrieval sliders, ``--modes``, the documented
# ablations such as TECHNIQUE_PLATFORM_FROM_TEXT and TECHNIQUE_INCLUDE_SYNTHETIC, the exemplar
# k/tau settings) ran such a model before contracts and keeps doing so unchanged.
CONTRACT_1_KEYS: tuple[str, ...] = (
    "dense_relative_margin",
    "dense_overfetch",
    "dense_collection",
    "exemplar_vote_norm",
    "exemplar_segment_votes",
    "exemplar_search_class",
)

# The serving values from before contracts: the value of a key that a stored contract lacks
# (contract_mismatches), and for CONTRACT_1_KEYS the one value a model artifact without a stored
# contract (fitted before 2026-10-02) can have been fitted with. For the older knobs these are the
# serving defaults only, not what such a model was fitted with: ``scripts/fit_acceptance_model.py``
# then ran retrieval with the ATT&CK synthetic sentences included (dense and lexical,
# ``include_synthetic`` defaulted to on there while serving has it off) and without
# platform-from-text scoping (the shipped model was created 2026-09-30, the scoping landed
# 2026-10-01). Such a model is not held to those knobs (:func:`legacy_contract`); a refit aligns
# fit and serving and stores its own contract.
LEGACY_CONTRACT: dict[str, Any] = {
    "contract_version": RETRIEVAL_CONTRACT_VERSION,
    "similarity_threshold": 0.45,
    "top_k": 30,
    "candidate_pool": 12,
    "retrieval_mode": "hybrid",
    "parent_policy": "ancestor_compatible",
    "include_detection": False,
    "include_synthetic": False,
    "lexical_fields": ["description", "name", "procedure"],
    "platform_from_text": True,
    "segment_acceptance": False,
    "hierarchy_parent_mode": "off",
    "rrf_k": 60,
    "dense_relative_margin": 0.12,
    "dense_overfetch": 0,
    "dense_collection": DEFAULT_DENSE_COLLECTION,
    "exemplar_fingerprint": None,
    "exemplar_k": 64,
    "exemplar_tau": 0.05,
    "exemplar_duplicate_similarity": 0.95,
    "exemplar_new_candidates": 3,
    "exemplar_vote_norm": DEFAULT_VOTE_NORM,
    "exemplar_segment_votes": DEFAULT_SEGMENT_VOTES,
    "exemplar_search_class": "exact",
    "exemplar_backend": "npz",
    # the technique head (2026-10-19): off before it existed
    "technique_head_fingerprint": None,
    "technique_head_new_candidates": HEAD_NEW_CANDIDATES,
    "technique_head_top_families": HEAD_TOP_FAMILIES,
}


def _contract_value(key: str, value: Any) -> Any:
    """One contract value in its canonical type (JSON round trips turn tuples into lists)."""
    if value is None:
        return None
    if key in _FLOAT_KEYS:
        return float(value)
    if key in _INT_KEYS:
        return int(value)
    if key in _BOOL_KEYS:
        return bool(value)
    if key == "lexical_fields":
        return sorted(str(item) for item in value)
    return str(value)


def retrieval_contract(
    kwargs: Mapping[str, Any],
    source: Any = None,
    *,
    lexical_fields: Sequence[str] | None = None,
    dense_collection: str | None = None,
    technique_head: Any = None,
) -> dict[str, Any]:
    """The retrieval settings the acceptance features depend on.

    ``kwargs`` are the keyword arguments ``retrieve_techniques`` runs with
    (:func:`apt_classifier.retrieve.retrieval_kwargs`), ``source`` the exemplar library whose
    votes the candidates carry (``None`` when exemplars are off or unavailable): its
    fingerprint is compared, its backend only recorded (:data:`CONTRACT_RECORD_KEYS`). The
    lexical fields are those of ``lexical_fields``, else of ``kwargs["lexical_index"]``, else
    the fields ``retrieve_techniques`` builds its own index with. ``technique_head`` is the
    active head client (``None`` when the head is off or unavailable): its checkpoint
    fingerprint is compared for a model fitted with the head."""
    if lexical_fields is None:
        lexical_fields = getattr(kwargs.get("lexical_index"), "fields", None) or None
    if lexical_fields is None:
        from apt_classifier.retrieve import default_lexical_fields

        lexical_fields = default_lexical_fields(
            include_synthetic=bool(kwargs["include_synthetic"]),
            include_detection=bool(kwargs["include_detection"]),
        )
    fingerprint = getattr(source, "fingerprint", None) if source is not None else None
    search_class = getattr(source, "search_class", "exact") if source is not None else None
    backend = str(getattr(source, "backend", "npz") or "npz") if source is not None else None
    values = {
        "similarity_threshold": kwargs["threshold"],
        "top_k": kwargs["top_k"],
        "candidate_pool": kwargs["candidate_pool"],
        "retrieval_mode": kwargs["retrieval_mode"],
        "parent_policy": kwargs["parent_policy"],
        "include_detection": kwargs["include_detection"],
        "include_synthetic": kwargs["include_synthetic"],
        "lexical_fields": lexical_fields,
        "platform_from_text": kwargs["platform_from_text"],
        "segment_acceptance": kwargs["segment_acceptance"],
        "hierarchy_parent_mode": kwargs["hierarchy_parent_mode"],
        "rrf_k": kwargs["rrf_k"],
        "dense_relative_margin": kwargs["dense_relative_margin"],
        "dense_overfetch": kwargs["dense_overfetch"],
        "dense_collection": dense_collection or DEFAULT_DENSE_COLLECTION,
        "exemplar_fingerprint": str(fingerprint) if fingerprint else None,
        "exemplar_k": kwargs["exemplar_k"],
        "exemplar_tau": kwargs["exemplar_tau"],
        "exemplar_duplicate_similarity": kwargs["exemplar_duplicate_similarity"],
        "exemplar_new_candidates": kwargs["exemplar_new_candidates"],
        "exemplar_vote_norm": kwargs.get("exemplar_vote_norm", DEFAULT_VOTE_NORM),
        "exemplar_segment_votes": kwargs.get("exemplar_segment_votes", DEFAULT_SEGMENT_VOTES),
        "exemplar_search_class": search_class,
        "technique_head_fingerprint": (
            str(getattr(technique_head, "fingerprint", "") or "") or None
            if technique_head is not None
            else None
        ),
        "technique_head_new_candidates": kwargs.get(
            "technique_head_new_candidates", HEAD_NEW_CANDIDATES
        ),
        "technique_head_top_families": kwargs.get("technique_head_top_families", HEAD_TOP_FAMILIES),
    }
    return {
        "contract_version": RETRIEVAL_CONTRACT_VERSION,
        **{key: _contract_value(key, values[key]) for key in RETRIEVAL_CONTRACT_KEYS},
        # recorded, not compared (CONTRACT_RECORD_KEYS)
        "exemplar_backend": backend,
    }


def legacy_contract(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a model artifact without a stored contract is held to (contract version 0): the
    knobs new with contract 1 at their one legacy value, and its own exemplar fingerprint (which
    ``compatible_with`` and ``retrieve_techniques`` check as they did before contracts). The
    older knobs are left out on purpose: such a model ran under any of their values before."""
    contract: dict[str, Any] = {"contract_version": LEGACY_CONTRACT_VERSION}
    contract.update({key: LEGACY_CONTRACT[key] for key in CONTRACT_1_KEYS})
    fingerprint = dict(metadata or {}).get("exemplar_fingerprint")
    contract["exemplar_fingerprint"] = str(fingerprint) if fingerprint else None
    return contract


def contract_mismatches(
    fitted: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    exemplars: bool,
    technique_head: bool = False,
) -> list[str]:
    """How ``runtime`` departs from the contract a model was fitted under (empty when it meets
    it). ``exemplars=False`` (a model fitted without exemplars) skips the exemplar keys and
    ``technique_head=False`` (a model fitted without the technique head) the head keys; a
    version-0 contract (:func:`legacy_contract`) is compared on :data:`CONTRACT_1_KEYS` only."""
    try:
        # strict: a missing or malformed version must never read as the looser version 0
        version = int(fitted.get("contract_version", RETRIEVAL_CONTRACT_VERSION))
    except (TypeError, ValueError):
        return ["contract version is unreadable"]
    if version > RETRIEVAL_CONTRACT_VERSION:
        return [
            f"contract version {version} is newer than this code ({RETRIEVAL_CONTRACT_VERSION})"
        ]
    keys = CONTRACT_1_KEYS if version == LEGACY_CONTRACT_VERSION else RETRIEVAL_CONTRACT_KEYS
    problems: list[str] = []
    for key in keys:
        if not exemplars and key in EXEMPLAR_CONTRACT_KEYS:
            continue
        if not technique_head and key in TECHNIQUE_HEAD_CONTRACT_KEYS:
            continue
        try:
            expected = _contract_value(key, fitted.get(key, LEGACY_CONTRACT.get(key)))
            actual = _contract_value(key, runtime.get(key, LEGACY_CONTRACT.get(key)))
        except (TypeError, ValueError):
            # a malformed value never matches (fail closed instead of raising at load)
            problems.append(f"{key} has an unreadable value")
            continue
        if expected != actual:
            problems.append(f"{key} {actual!r} (fitted with {expected!r})")
    return problems


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
        # schema 4; a schema-3 model reads its own prefix (feature_vector with its names)
        **head_features_from_evidence(candidate.evidence_for(TECHNIQUE_HEAD_EVIDENCE_SOURCE)),
    }


def feature_vector(features: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in names], dtype=float)


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
    # the retrieval contract the model is held to (retrieval_contract); None for a model fitted
    # in memory without one (unconstrained). contract_origin is "artifact" when the file stored
    # it and "legacy" when it predates contracts (legacy_contract, not written back on save).
    contract: dict[str, Any] | None = None
    contract_origin: str | None = None

    def prior_features(self, technique_id: str) -> dict[str, float]:
        return prior_features(self.technique_priors, technique_id)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature vector this model reads (its schema's names)."""
        return feature_names_for(self.feature_schema_version)

    def probability(self, features: dict[str, float]) -> float:
        x = (feature_vector(features, self.feature_names) - self.means) / self.scales
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

    @property
    def technique_head_fingerprint(self) -> str | None:
        value = self.metadata.get("technique_head_fingerprint")
        return str(value) if value else None

    @property
    def requires_technique_head(self) -> bool:
        """Fitted with the technique head's features active: runs only on that head."""
        return self.technique_head_fingerprint is not None

    @property
    def second_min_probability(self) -> float | None:
        """The single-answer cap's bar for a second calibrated acceptance per observation
        (``metadata["gate"]``, chosen by the fit on dev); ``None`` when the model has no cap
        (every artifact fitted before 2026-10-19)."""
        gate = self.metadata.get("gate")
        if not isinstance(gate, Mapping) or not gate.get("single_answer"):
            return None
        try:
            return float(gate.get("second_min_probability", 1.0))
        except (TypeError, ValueError):
            return None

    @property
    def sibling_min_ratio(self) -> float | None:
        """The sibling-replacement ratio (``metadata["gate"]["sibling_min_ratio"]``, chosen by the
        fit on dev): the head's preferred member of a family replaces an accepted sibling when
        the head prefers it by at least this factor; ``None`` when the model carries no rule."""
        gate = self.metadata.get("gate")
        if not isinstance(gate, Mapping):
            return None
        value = gate.get("sibling_min_ratio")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def head_first_min_probability(self) -> float | None:
        """The head-first bar (``metadata["gate"]["head_first_min_probability"]``, chosen by
        the fit on dev): a candidate the technique head gives at least this joint probability is
        accepted outright; ``None`` when the model carries no such rule."""
        gate = self.metadata.get("gate")
        if not isinstance(gate, Mapping):
            return None
        value = gate.get("head_first_min_probability")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def compatible_with(
        self,
        attack_version: str | None,
        exemplar_fingerprint: str | None = None,
        contract: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Whether the model may run here. ``contract`` is the runtime's retrieval contract
        (:func:`retrieval_contract`); without it, or for a model without one, only the schema,
        the ATT&CK version and the exemplar fingerprint are checked (the technique head a model
        was fitted with is part of its contract; ``retrieve_techniques`` checks it as well)."""
        if self.feature_schema_version not in SUPPORTED_FEATURE_SCHEMAS:
            return False, (
                f"feature schema {self.feature_schema_version} not in {SUPPORTED_FEATURE_SCHEMAS}"
            )
        if self.requires_technique_head and self.feature_schema_version < 4:
            return False, "fitted with the technique head but without its features (schema < 4)"
        if attack_version and self.attack_version and attack_version != self.attack_version:
            return False, f"ATT&CK version {self.attack_version} != {attack_version}"
        if self.requires_exemplars and exemplar_fingerprint != self.exemplar_fingerprint:
            return False, (
                "fitted with exemplar index "
                f"{self.exemplar_fingerprint}, runtime has "
                f"{exemplar_fingerprint or 'none'} (run make exemplars, or refit)"
            )
        if contract is not None and self.contract is not None:
            problems = contract_mismatches(
                self.contract,
                contract,
                exemplars=self.requires_exemplars,
                technique_head=self.requires_technique_head,
            )
            if problems:
                return False, "retrieval contract not met: " + "; ".join(problems)
        elif self.requires_technique_head and contract is not None:
            runtime_head = contract.get("technique_head_fingerprint")
            if runtime_head != self.technique_head_fingerprint:
                return False, (
                    f"fitted with technique head {self.technique_head_fingerprint}, runtime has "
                    f"{runtime_head or 'none'} (start the head service, or use another model)"
                )
        return True, "ok"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "technique_acceptance_logistic",
            "created_at": datetime.now(UTC).isoformat(),
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
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
        if self.contract is not None and self.contract_origin != "legacy":
            payload["retrieval_contract"] = dict(self.contract)
        return payload

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
        try:
            expected = feature_names_for(int(payload.get("feature_schema_version", 0)))
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
        if names != expected:
            raise ValueError("feature names do not match the artifact's feature schema")
        metadata = dict(payload.get("metadata") or {})
        stored = payload.get("retrieval_contract")
        if stored is not None and not isinstance(stored, Mapping):
            raise ValueError("retrieval_contract is not an object")
        if stored is not None:
            try:
                stored_version = int(stored.get("contract_version", RETRIEVAL_CONTRACT_VERSION))
            except (TypeError, ValueError):
                stored_version = None  # refused as unreadable by contract_mismatches
            if stored_version == LEGACY_CONTRACT_VERSION:
                # version 0 means "no stored contract" and is compared loosely; never from a file
                raise ValueError("a stored retrieval_contract cannot have version 0")
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            intercept=float(payload["intercept"]),
            means=np.asarray(payload["means"], dtype=float),
            scales=np.asarray(payload["scales"], dtype=float),
            threshold=float(payload["threshold"]),
            attack_version=str(payload.get("attack_version", "")),
            feature_schema_version=int(payload.get("feature_schema_version", 0)),
            metadata=metadata,
            technique_priors={
                str(key): [float(v) for v in values]
                for key, values in (payload.get("technique_priors") or {}).items()
            },
            contract=dict(stored) if stored is not None else legacy_contract(metadata),
            contract_origin="artifact" if stored is not None else "legacy",
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
        contract: Mapping[str, Any] | None = None,
        feature_schema_version: int = 3,
    ) -> AcceptanceModel:
        """Fit on feature rows. ``feature_schema_version`` picks the feature vector: 3 (the
        default, the shipped artifacts' shape) ignores the technique head's features, 4 reads
        them (a fit with the head active, ``scripts/fit_acceptance_model.py``)."""
        names = feature_names_for(feature_schema_version)
        X = np.vstack([feature_vector(row, names) for row in rows])
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
            feature_schema_version=int(feature_schema_version),
            metadata=dict(metadata or {}),
            technique_priors={
                k: [float(v) for v in vals] for k, vals in (technique_priors or {}).items()
            },
            contract=dict(contract) if contract is not None else None,
            contract_origin="artifact" if contract is not None else None,
        )


@dataclass
class HeadAwareAcceptance:
    """Two acceptance models behind one gate: the schema-4 model fitted with the technique head
    judges the candidates of an observation the head scored (they carry ``technique_head``
    evidence), the shipped model everything else (log-derived observations, which the head does
    not read, and observations whose request failed). Built by :func:`load_acceptance_model`
    when the head is active and ``acceptance_model_head_path`` loads; ``retrieve_techniques``
    picks the sub-model per candidate through :meth:`model_for`. The identity it reports (the
    contract, fingerprints, threshold, metadata) is the head model's, so the runtime checks hold
    the pair to the head it was fitted with."""

    head_model: AcceptanceModel
    plain_model: AcceptanceModel

    def model_for(self, candidate: Any) -> AcceptanceModel:
        evidence = getattr(candidate, "evidence_for", None)
        if callable(evidence) and evidence(TECHNIQUE_HEAD_EVIDENCE_SOURCE) is not None:
            return self.head_model
        return self.plain_model

    @property
    def threshold(self) -> float:
        return self.head_model.threshold

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.head_model.metadata,
            "plain_model": {
                "threshold": self.plain_model.threshold,
                "feature_schema_version": self.plain_model.feature_schema_version,
                "pipeline_fingerprint": self.plain_model.metadata.get("pipeline_fingerprint"),
            },
        }

    @property
    def contract(self) -> dict[str, Any] | None:
        return self.head_model.contract

    @property
    def contract_origin(self) -> str | None:
        return self.head_model.contract_origin

    @property
    def feature_schema_version(self) -> int:
        return self.head_model.feature_schema_version

    @property
    def requires_exemplars(self) -> bool:
        return self.head_model.requires_exemplars or self.plain_model.requires_exemplars

    @property
    def exemplar_fingerprint(self) -> str | None:
        return self.head_model.exemplar_fingerprint or self.plain_model.exemplar_fingerprint

    @property
    def requires_technique_head(self) -> bool:
        return True

    @property
    def technique_head_fingerprint(self) -> str | None:
        return self.head_model.technique_head_fingerprint

    @property
    def second_min_probability(self) -> float | None:
        return self.head_model.second_min_probability

    @property
    def head_first_min_probability(self) -> float | None:
        return self.head_model.head_first_min_probability

    @property
    def sibling_min_ratio(self) -> float | None:
        return self.head_model.sibling_min_ratio

    def prior_features(self, technique_id: str) -> dict[str, float]:
        return self.head_model.prior_features(technique_id)

    def probability(self, features: dict[str, float]) -> float:
        return self.head_model.probability(features)

    def accepts(self, features: dict[str, float]) -> bool:
        return self.head_model.accepts(features)

    def compatible_with(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        return self.head_model.compatible_with(*args, **kwargs)


def composite_problems(head_model: AcceptanceModel, plain_model: AcceptanceModel) -> list[str]:
    """Why ``head_model`` and ``plain_model`` may not share a gate (empty when they may): the
    head model must be fitted with the head, the plain one without, on the same exemplar library
    and under the same retrieval settings apart from the head's."""
    problems: list[str] = []
    if not head_model.requires_technique_head:
        problems.append("the technique head model was not fitted with the head")
    if plain_model.requires_technique_head:
        problems.append("the plain model was fitted with the technique head")
    if (
        head_model.requires_exemplars
        and plain_model.requires_exemplars
        and head_model.exemplar_fingerprint != plain_model.exemplar_fingerprint
    ):
        problems.append("the two models were fitted on different exemplar libraries")
    if head_model.contract is not None and plain_model.contract is not None:
        problems.extend(
            f"retrieval contract: {problem}"
            for problem in contract_mismatches(
                plain_model.contract,
                head_model.contract,
                exemplars=plain_model.requires_exemplars,
                technique_head=False,
            )
        )
    return problems


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


def choose_threshold_runtime(
    probabilities: Sequence[float],
    labels: Sequence[int],
    case_ids: Sequence[str],
    expected_counts: Mapping[str, int],
    *,
    blocked: Sequence[bool],
    forced: Sequence[bool],
    technique_ids: Sequence[str],
    min_precision: float = 0.5,
    grid: Iterable[float] | None = None,
    second_min_probability: float | None = None,
    head_probabilities: Sequence[float] | None = None,
    head_first_min_probability: float | None = None,
    sibling_min_ratio: float | None = None,
) -> dict[str, Any]:
    """:func:`choose_threshold` scored on the accepted sets the runtime gate produces.

    ``sibling_min_ratio`` with ``head_probabilities`` emulates sibling replacement (prereg
    cribl_sibling_replacement_2026-10-20): per (case, family) the row the head gives the
    highest probability replaces an accepted sibling on probability alone when its head
    probability is at least the ratio times the sibling's (``technique_ids`` give the family).

    ``head_first_min_probability`` with ``head_probabilities`` (the technique head's
    ``th_technique_p`` per row) emulates the head-first rule: an eligible row at or above the
    bar is accepted whatever its calibrated probability (prereg cribl_head_first_2026-10-20).

    ``second_min_probability`` emulates the single-answer cap (``retrieve.
    _assign_states_for_group``): per case only the most probable row accepted on probability
    stays, another only when its probability reaches the bar; forced rows are never capped.

    Per candidate row, ``forced`` rows are accepted at any threshold (cue-supported) and
    ``blocked`` rows never on probability alone (benign or sparse context, a hierarchy parent,
    or a parent ``_materialize_candidates`` leaves out); the rest are accepted when
    ``p >= t``. A forced row must not also be blocked by a hierarchy or materialization rule:
    the caller folds those into ``forced`` (``apt_classifier.retrieve.runtime_gate_masks``).
    Accepted techniques are counted once per case (``technique_ids``), as the case's accepted
    set is; recall counts every expected label, including those never retrieved. This emulates
    the local gate with ``hierarchy_parent_mode="off"``, no reranker and single-observation
    cases (the fit data), where corroboration across observations cannot apply."""
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    blocked_mask = np.asarray(blocked, dtype=bool)
    forced_mask = np.asarray(forced, dtype=bool)
    if not (
        len(probs)
        == len(truth)
        == len(case_ids)
        == len(blocked_mask)
        == len(forced_mask)
        == len(technique_ids)
    ):
        raise ValueError("rows, labels, cases, guards and technique ids are misaligned")
    pair_index: dict[tuple[str, str], int] = {}
    pairs = np.asarray(
        [
            pair_index.setdefault((str(case), str(technique)), len(pair_index))
            for case, technique in zip(case_ids, technique_ids, strict=True)
        ],
        dtype=np.int64,
    )
    pair_truth = np.zeros(len(pair_index), dtype=bool)
    if len(pairs):
        np.logical_or.at(pair_truth, pairs, truth == 1)
    total_expected = sum(int(value) for value in expected_counts.values())
    eligible = ~blocked_mask
    candidates = sorted(set(grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)))
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    capper = (
        _SingleAnswerCap(case_ids, probs, float(second_min_probability))
        if second_min_probability is not None
        else None
    )
    head_first_rows = np.zeros(len(probs), dtype=bool)
    if head_first_min_probability is not None and head_probabilities is not None:
        head = np.asarray(list(head_probabilities), dtype=float)
        if len(head) != len(probs):
            raise ValueError("head probabilities and rows are misaligned")
        head_first_rows = eligible & (head >= float(head_first_min_probability))
    swapper = (
        _SiblingSwap(
            case_ids,
            technique_ids,
            np.asarray(list(head_probabilities), dtype=float),
            float(sibling_min_ratio),
            eligible,
        )
        if sibling_min_ratio is not None and head_probabilities is not None
        else None
    )
    for threshold in candidates:
        on_probability = eligible & (probs >= threshold)
        if capper is not None:
            on_probability = capper.keep(on_probability)
        if swapper is not None:
            on_probability = swapper.swap(on_probability)
        accepted_rows = forced_mask | on_probability | head_first_rows
        accepted = np.zeros(len(pair_index), dtype=bool)
        accepted[pairs[accepted_rows]] = True
        tp = int((accepted & pair_truth).sum())
        fp = int((accepted & ~pair_truth).sum())
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
    chosen["rule"] = "runtime"
    chosen["forced_rows"] = int(forced_mask.sum())
    chosen["blocked_rows"] = int(blocked_mask.sum())
    chosen["second_min_probability"] = (
        float(second_min_probability) if second_min_probability is not None else None
    )
    chosen["head_first_min_probability"] = (
        float(head_first_min_probability) if head_first_min_probability is not None else None
    )
    chosen["sibling_min_ratio"] = (
        float(sibling_min_ratio) if sibling_min_ratio is not None else None
    )
    return chosen


class _SiblingSwap:
    """Per (case, family) the row with the highest head probability; an accepted sibling on
    probability alone is replaced by it when the head prefers it by the ratio (the runtime rule
    of ``retrieve._apply_sibling_replacement``)."""

    def __init__(
        self,
        case_ids: Sequence[str],
        technique_ids: Sequence[str],
        head: np.ndarray,
        ratio: float,
        eligible: np.ndarray,
    ) -> None:
        keys = [
            f"{case}\0{str(technique).split('.', 1)[0]}"
            for case, technique in zip(case_ids, technique_ids, strict=True)
        ]
        index: dict[str, int] = {}
        group = np.asarray([index.setdefault(key, len(index)) for key in keys])
        order = np.lexsort((-head, group))
        best = np.zeros(len(index), dtype=np.int64)
        seen = np.zeros(len(index), dtype=bool)
        for row in order:
            g = group[row]
            if not seen[g] and eligible[row]:
                best[g], seen[g] = row, True
        self.best_row = np.where(seen[group], best[group], np.arange(len(head)))
        self.head = head
        self.ratio = ratio

    def swap(self, accepted: np.ndarray) -> np.ndarray:
        if not len(accepted):
            return accepted
        rows = np.arange(len(accepted))
        best = self.best_row
        moved = (
            accepted
            & (best != rows)
            & (self.head[best] >= self.ratio * self.head)
            & (self.head[best] > 0)
        )
        out = accepted & ~moved
        out[best[moved]] = True
        return out


class _SingleAnswerCap:
    """Per case, keep the most probable row accepted on probability plus the rows at or above
    the bar (the runtime cap of ``_assign_states_for_group``), for any accepted mask."""

    def __init__(self, case_ids: Sequence[str], probs: np.ndarray, bar: float) -> None:
        index: dict[str, int] = {}
        self.case = np.asarray([index.setdefault(str(c), len(index)) for c in case_ids])
        self.probs = probs
        self.bar = bar
        # rows ordered by case, most probable first; the start of each case in that order
        self.order = np.lexsort((-probs, self.case))
        sorted_cases = self.case[self.order]
        starts = np.zeros(len(index), dtype=np.int64)
        if len(sorted_cases):
            first = np.r_[True, sorted_cases[1:] != sorted_cases[:-1]]
            starts[sorted_cases[first]] = np.flatnonzero(first)
        self.starts = starts

    def keep(self, accepted: np.ndarray) -> np.ndarray:
        if not len(accepted):
            return accepted
        acc_sorted = accepted[self.order]
        before = np.r_[0, np.cumsum(acc_sorted)[:-1]]
        case_sorted = self.case[self.order]
        # the first accepted row of its case: no accepted row between the case start and it
        first = acc_sorted & (before - before[self.starts[case_sorted]] == 0)
        kept_sorted = first | (acc_sorted & (self.probs[self.order] >= self.bar))
        kept = np.zeros_like(accepted)
        kept[self.order] = kept_sorted
        return kept


def load_acceptance_model(
    settings: Any,
    kb: Any,
    exemplar_index: Any = None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> tuple[AcceptanceModel | None, str]:
    """Load the configured acceptance model fail-closed.

    Returns ``(model, status)``; ``model`` is ``None`` with a reason whenever the feature
    is disabled, the artifact is missing, malformed, built for another ATT&CK version, or
    fitted with an exemplar index that the runtime does not have (``exemplar_index``), or, when
    ``contract`` (the runtime's :func:`retrieval_contract`, from
    :func:`apt_classifier.retrieve.retrieval_contract_for`) is given, fitted under another one.

    When the primary artifact cannot run, ``acceptance_model_fallback_path`` is tried: a model
    fitted the same way without exemplar features (``make fit-acceptance`` writes both). It is
    used only if it needs no exemplar index, so turning exemplars off, or losing the index,
    restores the previous calibrated gate instead of the rule-based one. Its contract is
    compared without the exemplar keys, so it also stands in when only an exemplar setting
    differs from the primary's contract. A fallback without a stored contract that would stand
    in for a primary that stores one (a refit) must also meet the serving defaults it ran under
    (:data:`LEGACY_CONTRACT`, non-exemplar keys): the refit's tuned settings never reach a model
    fitted before them. Without a stored contract on either side nothing changes.

    When the runtime contract carries an active technique head (``technique_head_fingerprint``)
    and ``acceptance_model_head_path`` names a model fitted with that head, the loaded model is
    paired with it in a :class:`HeadAwareAcceptance`: the head model judges the observations the
    head scored, the loaded one the rest. The pair is refused (the loaded model alone stays)
    when the head model does not load, was fitted on another head or library, or under other
    retrieval settings (:func:`composite_problems`); without an active head the head path is
    never read, so a stopped service restores the shipped gate exactly.
    """
    if not bool(getattr(settings, "acceptance_model_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "acceptance_model_path", None)
    if not path:
        return None, "no path configured"
    model, status = _load_compatible(path, kb, exemplar_index, contract)
    if model is None:
        fallback = getattr(settings, "acceptance_model_fallback_path", None)
        if fallback and Path(fallback) != Path(path):
            alternative, _ = _load_compatible(fallback, kb, None, contract)
            if alternative is not None and not alternative.requires_exemplars:
                problems = _legacy_stand_in_problems(path, alternative, contract)
                if problems:
                    return None, f"{status}; fallback refused: " + "; ".join(problems)
                model, status = alternative, f"fallback without exemplars ({status})"
    if model is None:
        return None, status
    return _with_head_model(settings, kb, exemplar_index, contract, model, status)


def _with_head_model(
    settings: Any,
    kb: Any,
    exemplar_index: Any,
    contract: Mapping[str, Any] | None,
    model: AcceptanceModel,
    status: str,
) -> tuple[AcceptanceModel | HeadAwareAcceptance, str]:
    """``model`` paired with the technique head model when the head is active and the pair is
    sound (:class:`HeadAwareAcceptance`), else ``model`` with the reason appended."""
    head_path = getattr(settings, "acceptance_model_head_path", None)
    runtime_head = dict(contract or {}).get("technique_head_fingerprint")
    if not head_path or not runtime_head or model.requires_technique_head:
        return model, status
    if Path(head_path) == Path(getattr(settings, "acceptance_model_path", "") or ""):
        return model, status
    head_model, head_status = _load_compatible(head_path, kb, exemplar_index, contract)
    if head_model is None:
        return model, f"{status}; technique head model not used: {head_status}"
    problems = composite_problems(head_model, model)
    if problems:
        return model, f"{status}; technique head model refused: " + "; ".join(problems)
    return HeadAwareAcceptance(head_model, model), f"{status} + technique head model"


def _legacy_stand_in_problems(
    primary_path: Any, alternative: AcceptanceModel, contract: Mapping[str, Any] | None
) -> list[str]:
    """Why a contract-less fallback may not stand in for a primary that stores a contract
    (empty when it may: no runtime contract, a fallback with its own contract, or a primary
    without one or unreadable, i.e. the behaviour before contracts)."""
    if contract is None or alternative.contract_origin != "legacy":
        return []
    try:
        primary = AcceptanceModel.load(primary_path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return []
    if primary is None or primary.contract_origin != "artifact":
        return []
    return [
        f"{problem} for a fallback without a stored contract"
        for problem in contract_mismatches(LEGACY_CONTRACT, contract, exemplars=False)
    ]


def _load_compatible(
    path: Any, kb: Any, exemplar_index: Any, contract: Mapping[str, Any] | None = None
) -> tuple[AcceptanceModel | None, str]:
    try:
        model = AcceptanceModel.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path}"
    ok, reason = model.compatible_with(
        str(getattr(kb, "version", "") or ""),
        getattr(exemplar_index, "fingerprint", None) if exemplar_index is not None else None,
        contract,
    )
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, "active"
