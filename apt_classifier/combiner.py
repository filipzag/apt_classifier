"""Learned attribution combiner: a listwise (conditional-logit) model over the shortlist.

Reciprocal-rank fusion weights every classifier the same way in every case. The evidence
regime differs a lot between cases, though: on real reports technique recall is low and the
technique-profile classifiers collapse while software and actor-behaviour evidence decide the
answer; on rich cases the opposite holds. This module scores each shortlist candidate from
*features only* (per-classifier ranks and relative scores, the fused rank, the candidate-relative
discrimination channels, documented-profile facts, and interactions with the case's evidence
regime), never from the group's identity, so a removed or unseen group needs no special case.

The model is a conditional logit fitted by Newton's method in numpy (the same conventions as
``apt_classifier.acceptance``): for every training list the true candidate competes against the
other shortlisted candidates (and, optionally, a "no candidate" option built from the regime
features) under a softmax. Its output orders the shortlist and gives a calibrated probability
that the leader is the true group. The artifact is JSON, pinned to the ATT&CK version and to a
fingerprint of the fusion pipeline, and loads fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from apt_classifier.classifiers.base import hierarchy_matches_for, kb_mapping
from apt_classifier.classifiers.software import COMMODITY_TOOL_WEIGHT_CAP, software_rarity_weights
from apt_classifier.fusion import FusionEntry, FusionResult, filter_eligible_results
from apt_classifier.separation import DiscriminationResult, technique_document_counts

# 2: the random-forest source-country vote joins the features; 3: per-member features for the
# optional members (OPTIONAL_CLASSIFIERS; zero whenever one does not vote, so an artifact fitted
# without them keeps its own columns); 4: the knowledge store's two members join them
FEATURE_SCHEMA_VERSION = 4
KIND = "attribution_combiner_conditional_logit"
DISTINCTIVE_MAX_GROUPS = 10
EXCLUSIVE_SOFTWARE_MAX_GROUPS = 4
SUPPORT_RANK_LIMIT = 10

CLASSIFIERS: tuple[str, ...] = (
    "bernoulli_naive_bayes",
    "idf_tfidf_cosine",
    "weighted_jaccard",
    "semantic_centroid",
    "specific_behavior",
    "software_tools",
    "campaign_profile",
    "group_description",
    "strategic_context",
    "motivation_alignment",
    "source_country",
)
# members that vote only behind a weight (default 0): their columns are zero when they are off
OPTIONAL_CLASSIFIERS: tuple[str, ...] = (
    "group_mlp",
    "text_group",
    "behavior_text",
    "procedure_match",
    # schema 4 (2026-11-04): the knowledge store's members (knowledge_store/members.py)
    "store_text",
    "store_software",
)
RRF_FEATURES = ("rrf_rel", "rrf_inv_rank", "rrf_support", "rrf_rank_std", "rrf_contested")
DISC_FEATURES = (
    "disc_technique",
    "disc_behavior",
    "disc_software",
    "disc_soft",
    "disc_negative",
    "disc_total_rel",
)
CANDIDATE_FEATURES = (
    "matched_share",
    "matched_distinctive_share",
    "exclusive_software",
    "log_group_techniques",
    "log_prominence",
)
REGIME_FEATURES = (
    "log_accepted",
    "distinctive_share",
    "software_rarity_max",
    "behavior_present",
    "log_eligible",
    "applicable_software",
    "applicable_behavior",
)
INTERACTION_BASES = (
    "bernoulli_naive_bayes",
    "idf_tfidf_cosine",
    "weighted_jaccard",
    "semantic_centroid",
    "specific_behavior",
    "software_tools",
    "group_description",
    "rrf",
    "disc_total",
)
INTERACTION_REGIMES = ("distinctive_share", "software_rarity_max", "log_accepted")


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for name in CLASSIFIERS:
        names += [f"{name}_present", f"{name}_inv_rank", f"{name}_score_rel"]
    names += list(RRF_FEATURES) + list(DISC_FEATURES) + list(CANDIDATE_FEATURES)
    names += [
        f"{base}_inv_rank_x_{regime}"
        for base in INTERACTION_BASES
        for regime in INTERACTION_REGIMES
    ]
    # schema 3: the optional members, after every schema-2 column so older artifacts keep theirs
    for name in OPTIONAL_CLASSIFIERS:
        names += [f"{name}_present", f"{name}_inv_rank", f"{name}_score_rel"]
    names += [
        f"{name}_inv_rank_x_{regime}"
        for name in OPTIONAL_CLASSIFIERS
        for regime in INTERACTION_REGIMES
    ]
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _feature_names()
NULL_FEATURE_NAMES: tuple[str, ...] = ("null_bias", *REGIME_FEATURES)


@dataclass(frozen=True)
class CombinerList:
    """One case: the shortlisted candidates in fused order with their feature rows."""

    group_ids: tuple[str, ...]
    features: np.ndarray  # (n_candidates, len(FEATURE_NAMES))
    regime: dict[str, float]
    truth: str | None = None  # a group id, or None for "no documented group" (unknown arm)

    @property
    def truth_index(self) -> int | None:
        if self.truth is None or self.truth not in self.group_ids:
            return None
        return self.group_ids.index(self.truth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_ids": list(self.group_ids),
            "features": self.features.tolist(),
            "regime": dict(self.regime),
            "truth": self.truth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CombinerList:
        return cls(
            group_ids=tuple(str(g) for g in payload["group_ids"]),
            features=np.asarray(payload["features"], dtype=float).reshape(
                len(payload["group_ids"]), len(FEATURE_NAMES)
            ),
            regime={str(k): float(v) for k, v in dict(payload.get("regime") or {}).items()},
            truth=payload.get("truth"),
        )


# ----------------------------------------------------------------------------- features


def _entry_maps(active: Sequence[Any]) -> dict[str, dict[str, tuple[int, float, float]]]:
    """classifier -> group -> (rank, score, top score of that classifier)."""
    maps: dict[str, dict[str, tuple[int, float, float]]] = {}
    for result in active:
        top = max((float(e.score) for e in result.entries), default=0.0)
        maps[result.classifier] = {
            e.group_id: (int(e.rank), float(e.score), top) for e in result.entries
        }
    return maps


def _score_rel(classifier: str, score: float, top: float) -> float:
    if classifier == "bernoulli_naive_bayes":
        return float(math.exp(max(-60.0, min(0.0, score - top))))
    if top > 0:
        return float(max(0.0, min(1.0, score / top)))
    return 0.0


def combiner_inputs(
    results: Sequence[Any],
    fusion: FusionResult,
    discrimination: DiscriminationResult | None,
    context: Any,
    kb: Any,
    *,
    top_n: int,
    weights: Mapping[str, float] | None = None,
    truth: str | None = None,
    tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
) -> CombinerList:
    """Feature rows for the top-N fused candidates (fused order). Deterministic."""
    candidates = list(fusion.entries[: max(1, int(top_n))])
    if not candidates:
        return CombinerList((), np.zeros((0, len(FEATURE_NAMES))), {}, truth)
    eligible = getattr(context, "eligible_group_ids", None)
    active = filter_eligible_results(results, weights or {}, eligible)
    maps = _entry_maps(active)
    observed: dict[str, float] = dict(getattr(context, "observed_techniques", {}) or {})
    software_ids = set(getattr(context, "software_ids", set()) or set())
    total_groups = len(kb_mapping(kb, "groups")) or 1
    doc_counts = technique_document_counts(kb)
    group_techniques = kb_mapping(kb, "group_techniques")
    group_software = kb_mapping(kb, "group_software")
    software_groups = kb_mapping(kb, "software_groups")
    prominence = kb_mapping(kb, "prominence")
    view = SimpleNamespace(
        kb=kb,
        parent_policy=str(getattr(context, "parent_policy", "ancestor_compatible")),
        observed_techniques=observed,
    )

    # ---- regime (list level) ----
    distinctive_observed = [t for t in observed if doc_counts.get(t, 0) <= DISTINCTIVE_MAX_GROUPS]
    rarity_max = 0.0
    if software_ids:
        rarity = software_rarity_weights(kb, software_ids, tool_weight_cap=tool_weight_cap)
        max_weight = math.log((total_groups + 1) / 2.0) + 1.0
        rarity_max = max((min(1.0, float(v) / max_weight) for v in rarity.values()), default=0.0)
    behaviour_matches = tuple(getattr(context, "specific_behavior_matches", ()) or ())
    regime = {
        "log_accepted": math.log1p(len(observed)),
        "distinctive_share": (len(distinctive_observed) / len(observed)) if observed else 0.0,
        "software_rarity_max": rarity_max,
        "behavior_present": 1.0 if behaviour_matches else 0.0,
        "log_eligible": math.log1p(len(eligible) if eligible is not None else len(fusion.entries)),
        "applicable_software": 1.0 if maps.get("software_tools") else 0.0,
        "applicable_behavior": 1.0 if maps.get("specific_behavior") else 0.0,
    }

    # ---- per-candidate ----
    by_group = discrimination.by_group() if discrimination is not None else {}
    disc_max = max((item.total for item in by_group.values()), default=0.0)
    rrf_max = max(float(candidates[0].score), 1e-12)
    rows: list[list[float]] = []
    for entry in candidates:
        gid = entry.group_id
        values: dict[str, float] = {}
        for name in (*CLASSIFIERS, *OPTIONAL_CLASSIFIERS):
            hit = maps.get(name, {}).get(gid)
            values[f"{name}_present"] = 1.0 if hit else 0.0
            values[f"{name}_inv_rank"] = (1.0 / hit[0]) if hit else 0.0
            values[f"{name}_score_rel"] = _score_rel(name, hit[1], hit[2]) if hit else 0.0
        values["rrf_rel"] = float(entry.score) / rrf_max
        values["rrf_inv_rank"] = 1.0 / max(1, int(entry.rank))
        values["rrf_support"] = float(
            sum(1 for r in entry.classifier_ranks.values() if r <= SUPPORT_RANK_LIMIT)
        )
        values["rrf_rank_std"] = float(entry.rank_std)
        values["rrf_contested"] = 1.0 if entry.contested else 0.0
        item = by_group.get(gid)
        values["disc_technique"] = float(item.technique) if item else 0.0
        values["disc_behavior"] = float(item.behavior) if item else 0.0
        values["disc_software"] = float(item.software) if item else 0.0
        values["disc_soft"] = float(getattr(item, "soft", 0.0)) if item else 0.0
        values["disc_negative"] = float(item.negative) if item else 0.0
        values["disc_total_rel"] = (float(item.total) / disc_max) if item and disc_max > 0 else 0.0
        matches = (
            hierarchy_matches_for(view, gid, observed_techniques=set(observed)) if observed else {}
        )
        matched = [t for t, m in matches.items() if float(m.factor) > 0]
        values["matched_share"] = (len(matched) / len(observed)) if observed else 0.0
        values["matched_distinctive_share"] = (
            sum(1 for t in matched if doc_counts.get(t, 0) <= DISTINCTIVE_MAX_GROUPS)
            / len(observed)
            if observed
            else 0.0
        )
        documented_software = set(group_software.get(gid, ()) or ())
        exclusive = [
            s
            for s in software_ids
            if s in documented_software
            and len(software_groups.get(s, ()) or ()) <= EXCLUSIVE_SOFTWARE_MAX_GROUPS
        ]
        values["exclusive_software"] = (len(exclusive) / len(software_ids)) if software_ids else 0.0
        values["log_group_techniques"] = math.log1p(len(group_techniques.get(gid, ()) or ()))
        values["log_prominence"] = math.log1p(max(0.0, float(prominence.get(gid, 0) or 0)))
        bases = {name: values[f"{name}_inv_rank"] for name in (*CLASSIFIERS, *OPTIONAL_CLASSIFIERS)}
        bases["rrf"] = values["rrf_inv_rank"]
        bases["disc_total"] = values["disc_total_rel"]
        for base in (*INTERACTION_BASES, *OPTIONAL_CLASSIFIERS):
            for regime_name in INTERACTION_REGIMES:
                values[f"{base}_inv_rank_x_{regime_name}"] = bases[base] * regime[regime_name]
        rows.append([float(values[name]) for name in FEATURE_NAMES])
    return CombinerList(
        tuple(entry.group_id for entry in candidates),
        np.asarray(rows, dtype=float),
        regime,
        truth,
    )


# ----------------------------------------------------------------------------- model


def _softmax(u: np.ndarray) -> np.ndarray:
    z = u - float(np.max(u))
    e = np.exp(z)
    return e / float(np.sum(e))


def _null_row(regime: Mapping[str, float]) -> np.ndarray:
    return np.asarray([1.0, *(float(regime.get(name, 0.0)) for name in REGIME_FEATURES)])


@dataclass
class AttributionCombiner:
    weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    attack_version: str
    pipeline_fingerprint: str
    null_weights: np.ndarray | None = None
    top_n: int = 20
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    # the features this artifact was fitted on, a subset of FEATURE_NAMES in any order: a newer
    # schema only adds features, so an older artifact keeps working on its own columns
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def _columns(self) -> list[int]:
        index = {name: position for position, name in enumerate(FEATURE_NAMES)}
        return [index[name] for name in self.feature_names]

    # ----- inference -----
    def _design(self, lst: CombinerList) -> tuple[np.ndarray, np.ndarray | None]:
        features = lst.features
        if tuple(self.feature_names) != FEATURE_NAMES and features.size:
            features = features[:, self._columns]
        elif not features.size:
            features = np.zeros((features.shape[0], len(self.feature_names)))
        candidate = (features - self.means) / self.scales
        null = _null_row(lst.regime) if self.null_weights is not None else None
        return candidate, null

    def utilities(self, lst: CombinerList) -> tuple[np.ndarray, float | None]:
        candidate, null = self._design(lst)
        u = candidate @ self.weights if len(candidate) else np.zeros(0)
        u_null = (
            float(null @ self.null_weights)
            if null is not None and self.null_weights is not None
            else None
        )
        return u, u_null

    def probabilities(self, lst: CombinerList) -> tuple[dict[str, float], float | None]:
        """(probability per candidate, probability of no candidate or None)."""
        if not lst.group_ids:
            return {}, (1.0 if self.null_weights is not None else None)
        u, u_null = self.utilities(lst)
        stacked = np.concatenate([u, [u_null]]) if u_null is not None else u
        p = _softmax(stacked)
        candidates = {gid: float(p[i]) for i, gid in enumerate(lst.group_ids)}
        return candidates, (float(p[-1]) if u_null is not None else None)

    # ----- persistence -----
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "null_feature_names": list(NULL_FEATURE_NAMES) if self.null_weights is not None else [],
            "weights": [float(v) for v in self.weights],
            "null_weights": [float(v) for v in self.null_weights]
            if self.null_weights is not None
            else None,
            "means": [float(v) for v in self.means],
            "scales": [float(v) for v in self.scales],
            "top_n": int(self.top_n),
            "attack_version": self.attack_version,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttributionCombiner:
        if payload.get("kind") != KIND:
            raise ValueError("not an attribution combiner artifact")
        version = int(payload.get("feature_schema_version", 0))
        if not 1 <= version <= FEATURE_SCHEMA_VERSION:
            raise ValueError("feature schema version is not supported")
        names = tuple(payload.get("feature_names", ()))
        known = set(FEATURE_NAMES)
        if not names or len(set(names)) != len(names) or any(name not in known for name in names):
            raise ValueError("feature names do not match the current schema")
        null_names = tuple(payload.get("null_feature_names") or ())
        null_weights = payload.get("null_weights")
        if null_weights is not None and null_names != NULL_FEATURE_NAMES:
            raise ValueError("null-option feature names do not match the current schema")
        weights = np.asarray(payload["weights"], dtype=float)
        means = np.asarray(payload["means"], dtype=float)
        scales = np.asarray(payload["scales"], dtype=float)
        if (
            weights.shape != (len(names),)
            or means.shape != weights.shape
            or scales.shape != weights.shape
        ):
            raise ValueError("weight vectors have the wrong shape")
        if np.any(scales <= 0):
            raise ValueError("scales must be positive")
        null = np.asarray(null_weights, dtype=float) if null_weights is not None else None
        if null is not None and null.shape != (len(NULL_FEATURE_NAMES),):
            raise ValueError("null-option weights have the wrong shape")
        return cls(
            weights=weights,
            means=means,
            scales=scales,
            attack_version=str(payload.get("attack_version", "")),
            pipeline_fingerprint=str(payload.get("pipeline_fingerprint", "")),
            null_weights=null,
            top_n=int(payload.get("top_n", 20)),
            feature_schema_version=version,
            metadata=dict(payload.get("metadata") or {}),
            feature_names=names,
        )

    @classmethod
    def load(cls, path: str | Path) -> AttributionCombiner | None:
        source = Path(path)
        if not source.is_file():
            return None
        return cls.from_dict(json.loads(source.read_text("utf-8")))

    def compatible_with(self, attack_version: str, pipeline_fingerprint: str) -> tuple[bool, str]:
        if self.attack_version and attack_version and self.attack_version != attack_version:
            return False, f"ATT&CK {self.attack_version} != {attack_version}"
        if (
            self.pipeline_fingerprint
            and pipeline_fingerprint
            and self.pipeline_fingerprint != pipeline_fingerprint
        ):
            return False, "pipeline fingerprint changed (refit the combiner)"
        return True, "ok"

    # ----- fitting -----
    @classmethod
    def fit(
        cls,
        lists: Sequence[CombinerList],
        *,
        attack_version: str,
        pipeline_fingerprint: str,
        l2: float = 1.0,
        iterations: int = 50,
        null_option: bool = True,
        top_n: int = 20,
        metadata: Mapping[str, Any] | None = None,
    ) -> AttributionCombiner:
        """Conditional logit by Newton's method. Lists whose truth is a group must contain it;
        lists with ``truth=None`` train the null option (skipped when ``null_option`` is off)."""
        usable = [
            lst
            for lst in lists
            if lst.group_ids and (lst.truth is None or lst.truth_index is not None)
        ]
        if not null_option:
            usable = [lst for lst in usable if lst.truth is not None]
        if not usable:
            raise ValueError("no usable training lists")
        stacked = np.vstack([lst.features for lst in usable])
        means = stacked.mean(axis=0)
        scales = stacked.std(axis=0)
        scales = np.where(scales > 1e-9, scales, 1.0)
        d_c = len(FEATURE_NAMES)
        d_n = len(NULL_FEATURE_NAMES) if null_option else 0
        designs: list[tuple[np.ndarray, int]] = []
        for lst in usable:
            cand = (lst.features - means) / scales
            block = np.hstack([cand, np.zeros((len(cand), d_n))]) if d_n else cand
            if null_option:
                null = np.concatenate([np.zeros(d_c), _null_row(lst.regime)])
                block = np.vstack([block, null[None, :]])
            target = len(cand) if lst.truth is None else int(lst.truth_index)  # type: ignore[arg-type]
            designs.append((block, target))
        d = d_c + d_n
        w = np.zeros(d)
        penalty = np.full(d, float(l2))
        if d_n:
            penalty[d_c] = 0.0  # the null bias is not shrunk
        for _ in range(max(1, iterations)):
            gradient = np.zeros(d)
            hessian = np.zeros((d, d))
            for block, target in designs:
                u = block @ w
                p = _softmax(u)
                gradient += block[target] - block.T @ p
                weighted = block * p[:, None]
                hessian -= block.T @ weighted - np.outer(block.T @ p, block.T @ p)
            gradient -= penalty * w
            hessian -= np.diag(penalty)
            step = np.linalg.solve(hessian, -gradient)

            # damped Newton: shrink the step when the (penalised) log-likelihood does not improve
            def objective(vector: np.ndarray) -> float:
                total = 0.0
                for block, target in designs:
                    u = block @ vector
                    total += float(
                        u[target] - (np.max(u) + math.log(np.sum(np.exp(u - np.max(u)))))
                    )
                return total - 0.5 * float(np.sum(penalty * vector * vector))

            base_value = objective(w)
            scale = 1.0
            while scale > 1e-4 and objective(w + scale * step) < base_value:
                scale *= 0.5
            w = w + scale * step
            if float(np.max(np.abs(scale * step))) < 1e-7:
                break
        return cls(
            weights=w[:d_c],
            null_weights=w[d_c:] if d_n else None,
            means=means,
            scales=scales,
            attack_version=attack_version,
            pipeline_fingerprint=pipeline_fingerprint,
            top_n=top_n,
            metadata={
                "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "lists": len(usable),
                "null_lists": sum(1 for lst in usable if lst.truth is None),
                "l2": float(l2),
                **dict(metadata or {}),
            },
        )


# ----------------------------------------------------------------------------- ordering


def apply_combiner_order(
    fusion: FusionResult,
    probabilities: Mapping[str, float],
    *,
    blend: float = 1.0,
    top_n: int = 20,
) -> FusionResult:
    """Re-order the top-N by ``(1 - blend) * rrf_rel + blend * P``; the top-N set and the
    tail are unchanged, scores are mapped into the head's original score range so
    ``relative_score`` stays comparable and no tail entry can overtake the head."""
    entries = list(fusion.entries)
    head = entries[: max(1, int(top_n))]
    tail = entries[len(head) :]
    if len(head) < 2 or blend <= 0:
        return fusion
    rrf_max = float(head[0].score) or 1.0
    rrf_min = float(head[-1].score)
    keyed = [
        (
            (1.0 - blend) * (float(e.score) / rrf_max)
            + blend * float(probabilities.get(e.group_id, 0.0)),
            e.rank,
            e,
        )
        for e in head
    ]
    keyed.sort(key=lambda item: (-item[0], item[1]))
    key_max = keyed[0][0]
    key_min = keyed[-1][0]
    span = key_max - key_min
    new_head: list[FusionEntry] = []
    for position, (key, _, entry) in enumerate(keyed):
        share = (key - key_min) / span if span > 0 else 1.0 - position / max(1, len(keyed) - 1)
        score = rrf_min + (rrf_max - rrf_min) * share
        new_head.append(replace(entry, score=score))
    ordered = new_head + tail
    maximum = float(ordered[0].score) if ordered else 1.0
    return replace(
        fusion,
        entries=tuple(
            replace(
                entry,
                rank=rank,
                relative_score=100.0 * float(entry.score) / maximum if maximum else 0.0,
            )
            for rank, entry in enumerate(ordered, 1)
        ),
    )


# ----------------------------------------------------------------------------- loading


def pipeline_fingerprint(
    *,
    rrf_k: int,
    weights: Mapping[str, float],
    family_weights: Mapping[str, float],
    sources: Iterable[Path] | None = None,
) -> str:
    """Hash of the fusion pipeline the combiner was fitted against: the fusion and separation
    modules, every classifier module, and the deterministic weight settings. The LLM vote and
    CyberRE weights are excluded on purpose: the features read only the deterministic
    classifiers, and a production run (vote on) must accept an artifact fitted by the local
    pipeline (vote off). Feature changes in this module are guarded by the schema version."""
    root = Path(__file__).resolve().parent
    files = (
        list(sources)
        if sources is not None
        else [
            root / "fusion.py",
            root / "separation.py",
            *sorted((root / "classifiers").glob("*.py")),
        ]
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"")

    def deterministic(mapping: Mapping[str, float]) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in sorted(mapping.items())
            if "llm" not in key and "cyberre" not in key
        }

    digest.update(
        json.dumps(
            {
                "rrf_k": int(rrf_k),
                "weights": deterministic(weights),
                "family_weights": deterministic(family_weights),
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:24]


def load_attribution_combiner(
    settings: Any, kb: Any, *, fingerprint: str
) -> tuple[AttributionCombiner | None, str]:
    """Load the configured combiner fail-closed: ``(model, status)``."""
    if not bool(getattr(settings, "attribution_combiner_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "attribution_combiner_path", None)
    if not path:
        return None, "no path configured"
    try:
        model = AttributionCombiner.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path}"
    ok, reason = model.compatible_with(str(getattr(kb, "version", "") or ""), fingerprint)
    if not ok:
        return None, f"incompatible artifact: {reason}"
    configured = int(getattr(settings, "attribution_combiner_top_n", model.top_n) or model.top_n)
    if configured != int(model.top_n):
        # a model fitted on top-N lists scores only that window; another N silently changes
        # which candidates it can reorder
        return None, (
            f"incompatible artifact: top_n (fitted on {model.top_n}, configured {configured})"
        )
    return model, "active"
