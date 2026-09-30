"""Random-forest group and source-country models over mapped techniques, motivation and target.

Both models are fitted from the ATT&CK knowledge base alone, never from benchmark or case data:

- the **group forest** predicts which documented group an evidence set belongs to;
- the **country forest** predicts the source-country class: China, Russia, Iran, North Korea,
  another state, or ``unattributed`` (no documented state origin). A group's class comes from
  ``origin.documented_origins``; groups documenting one of the four countries among several
  (Group5, POLONIUM) take that country.

Features are 0/1: one per ATT&CK technique (a sub-technique also sets its parent), one per
canonical motivation and target geography from ``classifiers.strategic_context`` plus an
"unknown" flag for each. Training cases are synthetic partial observations of each group's
documented profile: a random 15-70% of its techniques, false positives drawn by how many groups
document a technique (precision 45-95%), and motivation / one target geography each given 40%
of the time (5% of those replaced by a random value). That mimics what the mapping pipeline
delivers and what an analyst may or may not enter.

Training and inference run on the GPU when one is available (``torch_forest``). The artifact is
an ``.npz`` without pickle: the trees, the feature space, the class lists and a JSON manifest
pinned to the ATT&CK version and integrity-checked by a fingerprint, loaded fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.classifiers import strategic_context as _victimology
from apt_classifier.memo import ObjectMemo
from apt_classifier.origin import documented_origin_table
from apt_classifier.torch_forest import ForestArrays, ForestPredictor, fit_forest, resolve_device

SCHEMA_VERSION = 1
KIND = "profile_forest"
MAIN_COUNTRIES = ("China", "Russia", "Iran", "North Korea")
OTHER_STATE = "other_state"
UNATTRIBUTED = "unattributed"
MOTIVATIONS: tuple[str, ...] = tuple(_victimology._ALIASES["motivations"])
TARGETS: tuple[str, ...] = tuple(_victimology._ALIASES["target_geographies"])
DEFAULT_PARAMS: dict[str, Any] = {
    "samples_per_group": 300,
    "keep_min": 0.15,
    "keep_max": 0.70,
    "precision_min": 0.45,
    "precision_max": 0.95,
    "context_rate": 0.40,
    "context_noise": 0.05,
    "min_group_techniques": 2,
    "group_trees": 200,
    "group_min_samples_leaf": 2,
    "country_trees": 300,
    "country_min_samples_leaf": 3,
    "country_class_weight": "balanced",
    "max_features": "sqrt",
    "seed": 7,
}
_PROFILE_CACHE: ObjectMemo[dict[str, GroupProfile]] = ObjectMemo()


@dataclass(frozen=True)
class GroupProfile:
    techniques: frozenset[str]
    motivations: frozenset[str]
    targets: frozenset[str]
    country: str | None
    country_class: str


def _parent(kb: Any, technique_id: str) -> str | None:
    technique = (getattr(kb, "techniques", {}) or {}).get(technique_id)
    return getattr(technique, "parent_id", None) if technique is not None else None


def country_class(country: str | None) -> str:
    if country in MAIN_COUNTRIES:
        return str(country)
    return OTHER_STATE if country else UNATTRIBUTED


def group_profiles(kb: Any) -> dict[str, GroupProfile]:
    """Documented techniques (with parents), motivations, targets and source country per group."""
    groups = getattr(kb, "groups", {}) or {}
    stamp = (
        str(getattr(kb, "version", "") or ""),
        len(groups),
        len(getattr(kb, "techniques", {}) or {}),
    )
    cached = _PROFILE_CACHE.get(kb, stamp)
    if cached is not None:
        return cached
    origins = documented_origin_table(kb)
    profiles: dict[str, GroupProfile] = {}
    for group_id, group in groups.items():
        # the same technique source the classifiers read (kb.group_techniques), else the group
        techniques = set(
            (getattr(kb, "group_techniques", {}) or {}).get(group_id, ())
            or getattr(group, "technique_ids", ())
            or ()
        )
        techniques |= {p for t in list(techniques) if (p := _parent(kb, t))}
        texts = _victimology._profile_texts(kb, group_id)
        motivation_corpus = _victimology._field_corpus("motivations", texts)
        motivations = {
            value
            for value in MOTIVATIONS
            if _victimology._matches(
                "motivations",
                texts,
                motivation_corpus,
                _victimology._aliases_for("motivations", value)[1],
            )
        }
        target_corpus = _victimology._field_corpus("target_geographies", texts)
        targets = {
            value
            for value in TARGETS
            if _victimology._matches(
                "target_geographies",
                texts,
                target_corpus,
                _victimology._aliases_for("target_geographies", value)[1],
            )
        }
        documented = origins.get(group_id, {})
        main = [country for country in documented if country in MAIN_COUNTRIES]
        if len(documented) == 1:
            country: str | None = next(iter(documented))
        elif len(main) == 1:
            country = main[0]
        else:
            country = None
        profiles[group_id] = GroupProfile(
            techniques=frozenset(techniques),
            motivations=frozenset(motivations),
            targets=frozenset(targets),
            country=country,
            country_class=country_class(country),
        )
    return _PROFILE_CACHE.put(kb, stamp, profiles)


@dataclass(frozen=True)
class FeatureSpace:
    techniques: tuple[str, ...]
    motivations: tuple[str, ...] = MOTIVATIONS
    targets: tuple[str, ...] = TARGETS
    parents: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_kb(cls, kb: Any) -> FeatureSpace:
        techniques = tuple(sorted((getattr(kb, "techniques", {}) or {}).keys()))
        parents = {t: p for t in techniques if (p := _parent(kb, t))}
        return cls(techniques=techniques, parents=parents)

    @property
    def size(self) -> int:
        return len(self.techniques) + len(self.motivations) + 1 + len(self.targets) + 1

    @property
    def _index(self) -> dict[str, int]:
        return {technique: index for index, technique in enumerate(self.techniques)}

    def vector(
        self,
        techniques: Iterable[str],
        motivations: Iterable[str] = (),
        targets: Iterable[str] = (),
    ) -> np.ndarray:
        index = self._index
        row = np.zeros(self.size, dtype=np.uint8)
        for technique in techniques:
            for item in (technique, self.parents.get(technique)):
                if item in index:
                    row[index[item]] = 1
        base = len(self.techniques)
        chosen = [self.motivations.index(m) for m in motivations if m in self.motivations]
        if chosen:
            row[[base + position for position in chosen]] = 1
        else:
            row[base + len(self.motivations)] = 1
        base += len(self.motivations) + 1
        chosen = [self.targets.index(t) for t in targets if t in self.targets]
        if chosen:
            row[[base + position for position in chosen]] = 1
        else:
            row[base + len(self.targets)] = 1
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "techniques": list(self.techniques),
            "motivations": list(self.motivations),
            "targets": list(self.targets),
            "parents": dict(sorted(self.parents.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpace:
        return cls(
            techniques=tuple(str(t) for t in payload["techniques"]),
            motivations=tuple(str(m) for m in payload["motivations"]),
            targets=tuple(str(t) for t in payload["targets"]),
            parents={str(k): str(v) for k, v in (payload.get("parents") or {}).items()},
        )


def training_cases(
    kb: Any, space: FeatureSpace, params: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    """Synthetic partial observations of every group's documented profile (seeded)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(int(p["seed"]))
    profiles = group_profiles(kb)
    index = {technique: position for position, technique in enumerate(space.techniques)}
    popularity = np.zeros(len(space.techniques), dtype=np.float64)
    for profile in profiles.values():
        for technique in profile.techniques:
            if technique in index:
                popularity[index[technique]] += 1
    rows: list[np.ndarray] = []
    groups: list[str] = []
    classes: list[str] = []
    for group_id in sorted(profiles):
        profile = profiles[group_id]
        own = sorted(t for t in profile.techniques if t in index)
        if len(own) < int(p["min_group_techniques"]):
            continue
        weights = popularity.copy()
        weights[[index[t] for t in own]] = 0.0
        total = weights.sum()
        weights = weights / total if total > 0 else None
        motivations = sorted(profile.motivations)
        targets = sorted(profile.targets)
        for _ in range(int(p["samples_per_group"])):
            keep = max(1, int(round(len(own) * rng.uniform(p["keep_min"], p["keep_max"]))))
            kept = list(rng.choice(own, size=min(keep, len(own)), replace=False))
            precision = rng.uniform(p["precision_min"], p["precision_max"])
            false_positives = int(round(len(kept) * (1 - precision) / precision))
            if false_positives and weights is not None:
                false_positives = min(false_positives, int((weights > 0).sum()))
                kept += [
                    space.techniques[i]
                    for i in rng.choice(
                        len(space.techniques), size=false_positives, replace=False, p=weights
                    )
                ]
            given_motivations: tuple[str, ...] = ()
            if motivations and rng.uniform() < p["context_rate"]:
                given_motivations = tuple(motivations)
                if rng.uniform() < p["context_noise"]:
                    given_motivations = (MOTIVATIONS[int(rng.integers(len(MOTIVATIONS)))],)
            given_targets: tuple[str, ...] = ()
            if targets and rng.uniform() < p["context_rate"]:
                given_targets = (targets[int(rng.integers(len(targets)))],)
                if rng.uniform() < p["context_noise"]:
                    given_targets = (TARGETS[int(rng.integers(len(TARGETS)))],)
            rows.append(space.vector(kept, given_motivations, given_targets))
            groups.append(group_id)
            classes.append(profile.country_class)
    if not rows:
        raise ValueError("no group has enough documented techniques to train on")
    return np.stack(rows), groups, classes


class ProfileForest:
    """Group and country forests with their feature space; predictions run on ``device``."""

    def __init__(
        self,
        *,
        space: FeatureSpace,
        group_ids: tuple[str, ...],
        group_forest: ForestArrays,
        country_classes: tuple[str, ...],
        country_forest: ForestArrays,
        group_classes: Mapping[str, str],
        metadata: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.space = space
        self.group_ids = tuple(group_ids)
        self.group_forest = group_forest
        self.country_classes = tuple(country_classes)
        self.country_forest = country_forest
        self.group_classes = dict(group_classes)  # group id -> country class (every KB group)
        self.metadata = dict(metadata or {})
        self.device = resolve_device(device)
        self._group_predictor: ForestPredictor | None = None
        self._country_predictor: ForestPredictor | None = None

    # ------------------------------------------------------------------ prediction
    def _predictors(self) -> tuple[ForestPredictor, ForestPredictor]:
        if self._group_predictor is None or self._country_predictor is None:
            self._group_predictor = ForestPredictor(self.group_forest, self.device)
            self._country_predictor = ForestPredictor(self.country_forest, self.device)
        return self._group_predictor, self._country_predictor

    def predict(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        group_predictor, country_predictor = self._predictors()
        return group_predictor.predict_proba(vectors), country_predictor.predict_proba(vectors)

    def group_probabilities(self, vector: np.ndarray) -> dict[str, float]:
        probabilities, _ = self.predict(vector)
        return {
            group_id: float(value)
            for group_id, value in zip(self.group_ids, probabilities[0], strict=True)
            if value > 0
        }

    def country_probabilities(self, vector: np.ndarray) -> dict[str, float]:
        _, probabilities = self.predict(vector)
        return {
            name: float(value)
            for name, value in zip(self.country_classes, probabilities[0], strict=True)
        }

    # ------------------------------------------------------------------ persistence
    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for arrays in (self.group_forest, self.country_forest):
            for name, value in sorted(arrays.to_npz_dict("").items()):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(
            json.dumps(
                {
                    "space": self.space.to_dict(),
                    "groups": list(self.group_ids),
                    "countries": list(self.country_classes),
                    "group_classes": dict(sorted(self.group_classes.items())),
                    "schema": SCHEMA_VERSION,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()[:24]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "space": self.space.to_dict(),
            "group_ids": list(self.group_ids),
            "country_classes": list(self.country_classes),
            "group_classes": dict(sorted(self.group_classes.items())),
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }
        arrays = {
            **self.group_forest.to_npz_dict("group_"),
            **self.country_forest.to_npz_dict("country_"),
            "manifest": np.asarray(json.dumps(manifest, sort_keys=True)),
        }
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> ProfileForest | None:
        source = Path(path)
        if not source.is_file():
            return None
        with np.load(source, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            if manifest.get("kind") != KIND:
                raise ValueError(f"not a profile forest artifact: {manifest.get('kind')!r}")
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema {manifest.get('schema_version')!r}")
            model = cls(
                space=FeatureSpace.from_dict(manifest["space"]),
                group_ids=tuple(manifest["group_ids"]),
                group_forest=ForestArrays.from_npz(data, "group_"),
                country_classes=tuple(manifest["country_classes"]),
                country_forest=ForestArrays.from_npz(data, "country_"),
                group_classes=manifest["group_classes"],
                metadata=manifest.get("metadata") or {},
                device=device,
            )
        if model.group_forest.n_classes != len(
            model.group_ids
        ) or model.country_forest.n_classes != len(model.country_classes):
            raise ValueError("class lists do not match the forests")
        if (
            model.group_forest.n_features != model.space.size
            or model.country_forest.n_features != model.space.size
        ):
            raise ValueError("feature space does not match the forests")
        if manifest.get("fingerprint") != model.fingerprint:
            raise ValueError("fingerprint mismatch (artifact modified or corrupted)")
        return model

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        version = str(getattr(kb, "version", "") or "")
        fitted = str(self.metadata.get("attack_version") or "")
        if fitted and version and fitted != version:
            return (
                False,
                f"fitted on ATT&CK {fitted}, runtime has {version} (refit: make fit-forest)",
            )
        return True, "ok"


def fit_profile_forest(
    kb: Any,
    *,
    params: Mapping[str, Any] | None = None,
    device: str | None = None,
    reuse_group_forest: ProfileForest | None = None,
) -> ProfileForest:
    """Fit both forests on ``kb`` (GPU when available); returns the model in memory.

    ``reuse_group_forest`` keeps that model's group forest instead of refitting it (used when
    only the country forest votes, e.g. per removed actor in the benchmarks)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    space = FeatureSpace.from_kb(kb)
    started = time.time()
    features, groups, classes = training_cases(kb, space, p)
    cases_seconds = time.time() - started
    group_ids = tuple(sorted(set(groups)))
    group_index = {group_id: position for position, group_id in enumerate(group_ids)}
    country_names = tuple(sorted(set(classes)))
    country_index = {name: position for position, name in enumerate(country_names)}
    resolved = resolve_device(device)
    t0 = time.time()
    if reuse_group_forest is not None:
        group_ids = reuse_group_forest.group_ids
        group_forest = reuse_group_forest.group_forest
    else:
        group_forest = fit_forest(
            features,
            np.asarray([group_index[g] for g in groups]),
            len(group_ids),
            n_estimators=int(p["group_trees"]),
            max_features=p["max_features"],
            min_samples_leaf=int(p["group_min_samples_leaf"]),
            seed=int(p["seed"]),
            device=resolved,
        )
    group_seconds = time.time() - t0
    t0 = time.time()
    country_forest = fit_forest(
        features,
        np.asarray([country_index[c] for c in classes]),
        len(country_names),
        n_estimators=int(p["country_trees"]),
        max_features=p["max_features"],
        min_samples_leaf=int(p["country_min_samples_leaf"]),
        class_weight=p["country_class_weight"],
        seed=int(p["seed"]) + 1,
        device=resolved,
    )
    country_seconds = time.time() - t0
    profiles = group_profiles(kb)
    return ProfileForest(
        space=space,
        group_ids=group_ids,
        group_forest=group_forest,
        country_classes=country_names,
        country_forest=country_forest,
        group_classes={group_id: profile.country_class for group_id, profile in profiles.items()},
        metadata={
            "attack_version": str(getattr(kb, "version", "") or ""),
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "params": p,
            "device": resolved,
            "training_cases": int(features.shape[0]),
            "group_forest_reused": reuse_group_forest is not None,
            "seconds": {
                "cases": round(cases_seconds, 1),
                "group_forest": round(group_seconds, 1),
                "country_forest": round(country_seconds, 1),
            },
            "group_forest_nodes": group_forest.n_nodes,
            "country_forest_nodes": country_forest.n_nodes,
            "country_groups": {
                name: sum(1 for profile in profiles.values() if profile.country_class == name)
                for name in country_names
            },
        },
        device=resolved,
    )


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    """The artifact's JSON manifest without loading the trees (``None`` when absent)."""
    source = Path(path)
    if not source.is_file():
        return None
    with np.load(source, allow_pickle=False) as data:
        return json.loads(str(data["manifest"]))


def load_profile_forest(settings: Any, kb: Any) -> tuple[ProfileForest | None, str]:
    """Load the configured artifact fail-closed: ``(model, status)``."""
    if not bool(getattr(settings, "profile_forest_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "profile_forest_path", None)
    if not path:
        return None, "no path configured"
    try:
        manifest = read_manifest(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        return None, f"invalid artifact: {error}"
    if manifest is None:
        return None, f"artifact not found: {path} (run make fit-forest)"
    fitted = str((manifest.get("metadata") or {}).get("attack_version") or "")
    runtime = str(getattr(kb, "version", "") or "")
    if fitted and runtime and fitted != runtime:  # decided before the trees are read
        return None, f"incompatible artifact: fitted on ATT&CK {fitted}, runtime has {runtime}"
    try:
        model = ProfileForest.load(path, device=getattr(settings, "profile_forest_device", None))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path} (run make fit-forest)"
    ok, reason = model.compatible_with(kb)
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, f"active ({model.device})"


def canonical_context(strategic_context: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The analyst's motivations and target geographies mapped to the model's vocabulary."""
    motivations = []
    for value in getattr(strategic_context, "motivations", ()) or ():
        canonical, _ = _victimology._aliases_for("motivations", str(value))
        if canonical in MOTIVATIONS:
            motivations.append(canonical)
    targets = []
    for value in getattr(strategic_context, "target_geographies", ()) or ():
        canonical, _ = _victimology._aliases_for("target_geographies", str(value))
        if canonical in TARGETS:
            targets.append(canonical)
    return tuple(dict.fromkeys(motivations)), tuple(dict.fromkeys(targets))


def evidence_vector(context: Any, space: FeatureSpace) -> np.ndarray:
    """Feature row for a classifier context: accepted techniques plus the analyst's context."""
    motivations, targets = canonical_context(getattr(context, "strategic_context", None))
    return space.vector(getattr(context, "observed_techniques", {}) or {}, motivations, targets)
