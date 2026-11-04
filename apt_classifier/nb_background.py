"""Group-background correction of the Naive Bayes member (profile-size bias, 2026-11-04).

The Naive Bayes member scores a group by the observed techniques alone: a documented technique
adds log(alpha), an undocumented one log(beta), and a documented technique the report does not
mention costs nothing. A group with a long ATT&CK technique list therefore covers most of any
long report and leads it (Kimsuky, 130 techniques, led 18 of the 58 development-report misses).

The correction asks how unusual a group's score is for that group. It z-scores the member's
scores across the groups of the case (so short and long inputs share one scale), then z-scores
each group against the mean and spread of its own case-level scores on a fixed reference set.
The reference is the attribution benchmark's design folds (seeds 41 and 53), built from ATT&CK
procedure sentences only, never benchmark reports.

Development result (scratchpad replay, member alone, the 68 vendor/campaign/advisory reports;
the ATT&CK folds keep the true group's technique list and cannot judge a technique-list member):
top-1 10 -> 23 of 68, MRR 0.298 -> 0.415, 14 gained and 1 lost; Kimsuky as wrong leader 18 -> 0.
The same correction lowers the IDF cosine and Jaccard members, so it applies to Naive Bayes only.

Off by default (``NAIVE_BAYES_BACKGROUND_ENABLED``). The module lives outside ``classifiers/`` so
that the combiner pipeline fingerprint stays unchanged while it is off; when it is on the
fingerprint binds this module and the reference artifact, so it needs a combiner fitted with it.
The artifact is pinned to the ATT&CK version, the group technique table and the member's
parameters, and the loader fails closed (the shipped Naive Bayes keeps voting).
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apt_classifier.classifiers.base import ClassifierContext, ClassifierResult, build_result

NAIVE_BAYES_NAME = "bernoulli_naive_bayes"
SCHEMA_VERSION = 1
# a group needs this many reference cases before its own mean and spread are used
MIN_REFERENCE_CASES = 20
_VARIANCE_FLOOR = 1e-9


def case_z(scores: Mapping[str, float]) -> dict[str, float]:
    """The member's scores z-scored across the groups of one case."""
    if not scores:
        return {}
    values = list(scores.values())
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values) or 1.0
    return {group_id: (score - mean) / spread for group_id, score in scores.items()}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def group_profiles(kb: Any) -> dict[str, str]:
    """Hash of each group's technique list, which its score depends on."""
    return {
        group_id: _digest(sorted(technique_ids))
        for group_id, technique_ids in sorted((kb.group_techniques or {}).items())
    }


def parent_digest(kb: Any) -> str:
    """Hash of the technique parent links (the member's hierarchy matches)."""
    techniques = getattr(kb, "techniques", {}) or {}
    return _digest(
        {
            technique_id: getattr(technique, "parent_id", None)
            for technique_id, technique in sorted(techniques.items())
        }
    )


def member_params(settings: Any) -> dict[str, float]:
    """The Naive Bayes parameters the reference was scored with."""
    return {
        "alpha": float(getattr(settings, "nb_alpha", 0.6)),
        "beta": float(getattr(settings, "nb_beta", 0.05)),
        "negative_weight": float(getattr(settings, "negative_evidence_weight", 0.3)),
    }


@dataclass(frozen=True)
class NaiveBayesBackground:
    attack_version: str
    # group_id -> hash of its technique list, and the hash of the technique parent links
    group_profiles: dict[str, str]
    parent_digest: str
    params: dict[str, float]
    # group_id -> (mean, spread, reference cases) of its case-level z-scores
    groups: dict[str, tuple[float, float, int]]
    sources: tuple[dict[str, Any], ...] = ()
    min_reference_cases: int = MIN_REFERENCE_CASES
    created_at: str = field(default="", compare=False)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self._content(), sort_keys=True).encode())
        return digest.hexdigest()[:24]

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "attack_version": self.attack_version,
            "group_profiles": dict(sorted(self.group_profiles.items())),
            "parent_digest": self.parent_digest,
            "params": {key: float(value) for key, value in sorted(self.params.items())},
            "groups": {
                group_id: [float(mean), float(spread), int(count)]
                for group_id, (mean, spread, count) in sorted(self.groups.items())
            },
            "sources": list(self.sources),
            "min_reference_cases": int(self.min_reference_cases),
        }

    def correct(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Case z-score, then each group's z-score against its reference distribution. A group
        with too few reference cases keeps its case z-score."""
        output = {}
        for group_id, value in case_z(scores).items():
            reference = self.groups.get(group_id)
            if reference is None or reference[2] < self.min_reference_cases:
                output[group_id] = value
                continue
            mean, spread, _ = reference
            output[group_id] = (value - mean) / spread
        return output

    def to_json(self) -> dict[str, Any]:
        return {
            **self._content(),
            "created_at": self.created_at,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> NaiveBayesBackground:
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"schema {payload.get('schema_version')} (expected {SCHEMA_VERSION})")
        model = cls(
            attack_version=str(payload["attack_version"]),
            group_profiles={str(k): str(v) for k, v in payload["group_profiles"].items()},
            parent_digest=str(payload["parent_digest"]),
            params={key: float(value) for key, value in payload["params"].items()},
            groups={
                str(group_id): (float(values[0]), float(values[1]), int(values[2]))
                for group_id, values in payload["groups"].items()
            },
            sources=tuple(payload.get("sources") or ()),
            min_reference_cases=int(payload.get("min_reference_cases", MIN_REFERENCE_CASES)),
            created_at=str(payload.get("created_at", "")),
        )
        if model.fingerprint != payload.get("fingerprint"):
            raise ValueError("fingerprint mismatch (artifact modified or corrupted)")
        return model

    @classmethod
    def load(cls, path: str | Path) -> NaiveBayesBackground | None:
        source = Path(path)
        if not source.is_file():
            return None
        return cls.from_json(json.loads(source.read_text(encoding="utf-8")))

    def compatible_with(self, kb: Any, settings: Any) -> tuple[bool, str]:
        """The pinned knowledge base or one with groups removed (the benchmarks' removed-actor
        folds); any other change to a technique list or a parent link refuses."""
        version = str(getattr(kb, "version", "") or "")
        if self.attack_version != version:
            return False, f"ATT&CK {self.attack_version} (knowledge base {version})"
        if self.parent_digest != parent_digest(kb):
            return False, "technique parent links changed (rebuild the reference)"
        changed = sorted(
            group_id
            for group_id, profile in group_profiles(kb).items()
            if self.group_profiles.get(group_id) != profile
        )
        if changed:
            return False, f"technique lists changed or new for {', '.join(changed[:5])}"
        if self.params != member_params(settings):
            return (
                False,
                f"Naive Bayes parameters {self.params} (configured {member_params(settings)})",
            )
        return True, "active"


def fit_background(
    kb: Any,
    member: Any,
    contexts: Iterable[ClassifierContext],
    settings: Any,
    *,
    sources: Iterable[Mapping[str, Any]] = (),
    min_reference_cases: int = MIN_REFERENCE_CASES,
) -> NaiveBayesBackground:
    """Each group's mean and spread of its case-level z-scores over the reference contexts."""
    sums: dict[str, float] = {}
    squares: dict[str, float] = {}
    counts: dict[str, int] = {}
    for context in contexts:
        result = member.classify(context)
        if not result.applicable:
            continue
        for group_id, value in case_z({e.group_id: e.score for e in result.entries}).items():
            sums[group_id] = sums.get(group_id, 0.0) + value
            squares[group_id] = squares.get(group_id, 0.0) + value * value
            counts[group_id] = counts.get(group_id, 0) + 1
    groups = {}
    for group_id, count in sorted(counts.items()):
        mean = sums[group_id] / count
        variance = max(squares[group_id] / count - mean * mean, _VARIANCE_FLOOR)
        groups[group_id] = (mean, variance**0.5, count)
    return NaiveBayesBackground(
        attack_version=str(getattr(kb, "version", "") or ""),
        group_profiles=group_profiles(kb),
        parent_digest=parent_digest(kb),
        params=member_params(settings),
        groups=groups,
        sources=tuple(dict(source) for source in sources),
        min_reference_cases=min_reference_cases,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


class BackgroundNaiveBayesClassifier:
    """The Naive Bayes member with its scores corrected by the group background; same name, so
    fusion weights, families and combiner columns treat it as the Naive Bayes member."""

    name = NAIVE_BAYES_NAME

    def __init__(self, member: Any, background: NaiveBayesBackground) -> None:
        self.member = member
        self.background = background

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        result = self.member.classify(context)
        if not result.applicable:
            return result
        scores = self.background.correct({e.group_id: e.score for e in result.entries})
        return build_result(self.name, context, scores, applicable=True)


def nb_background_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "naive_bayes_background_enabled", False))


def load_nb_background_for(settings: Any, kb: Any) -> tuple[NaiveBayesBackground | None, str]:
    """The configured reference for the engine, fail-closed: ``(model, status)``."""
    if not nb_background_enabled(settings):
        return None, "disabled"
    path = getattr(settings, "naive_bayes_background_path", "")
    try:
        model = NaiveBayesBackground.load(path)
    except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path}"
    ok, reason = model.compatible_with(kb, settings)
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, "active"


def with_nb_background(
    classifiers: tuple[Any, ...], background: NaiveBayesBackground | None
) -> tuple[Any, ...]:
    """The member list with the Naive Bayes member wrapped (unchanged without a background)."""
    if background is None:
        return classifiers
    return tuple(
        BackgroundNaiveBayesClassifier(member, background)
        if getattr(member, "name", "") == NAIVE_BAYES_NAME
        else member
        for member in classifiers
    )


def nb_background_bound_fingerprint(base: str, model: NaiveBayesBackground) -> str:
    """The combiner pipeline fingerprint with the correction on: the base fingerprint mixed with
    the reference fingerprint and this module's source."""
    digest = hashlib.sha256()
    digest.update(base.encode())
    digest.update(model.fingerprint.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:24]
