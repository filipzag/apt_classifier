"""Calibrated confidence labels for the top three (display only, never a decision).

The naming gate (``attribution.py``) decides whether a group is named. A confidence label only
states how often cases like this one were right on ATT&CK test folds: "strong single answer",
"one of these 2/3" or "open". Three logistic regressions read what the decision already
recorded -- the learned combiner's probabilities for the first three places and its "no
candidate" probability, the gate's separation, the number of accepted techniques and whether
named software supports the leader -- and give P(leader right), P(truth in the top two) and
P(truth in the top three). ``scripts/fit_confidence_labels.py`` fits them on the design folds
(seeds 41 and 53, removed actors count as wrong) and picks the thresholds on out-of-fold
predictions with one-sided Clopper-Pearson bounds. A strong-only artifact (``thresholds["set"]``
null: no set threshold met its bound and the fit was told to keep the strong label) never shows
"one of these"; a cluster the gate named is still shown as its pair.

Nothing here changes the stage, the ranking, the gate window or an export, and the engine never
imports this module (like ``top3_view``). A label can say "strong" where the gate holds back,
and on real reports it runs overconfident: a name from the gate always takes the headline. The
artifact is JSON, pinned to the ATT&CK version and to the exact combiner artifact whose
probabilities it reads, and loads fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIND = "vestige_confidence_labels"
SCHEMA_VERSION = 1
FEATURE_NAMES: tuple[str, ...] = (
    "logit_p",
    "separation",
    "logit_null",
    "log_accepted",
    "leader_software",
)
# leader: p = P1; top2: p = min(P1 + P2, 1); top3: p = min(P1 + P2 + P3, 1)
MODELS: tuple[str, ...] = ("leader", "top2", "top3")
LOGIT_CLIP = 1e-4

INSUFFICIENT_STAGE = "insufficient_evidence"
NAMED_STAGE = "known_group_hypothesis"
# a documented-overlap cluster named as a whole (decision.named_cluster: group ids, leader first)
CLUSTER_STAGE = "known_cluster_hypothesis"

# the evaluation sets the analyst-facing counts come from (written by the fit script)
FOLD_EVALUATION = "design_oof"
REAL_REPORT_EVALUATION = "cti_hal_anonymized"
NOTE = "Calibrated on ATT&CK test folds; real reports run overconfident."


def _logit(p: float, clip: float) -> float:
    p = min(max(float(p), clip), 1.0 - clip)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class ConfidenceInputs:
    """What a label reads from a decision; identical for a live state and a benchmark row."""

    # combiner probability of places 1-3 as the engine records it (4 decimals; 0 when absent)
    probabilities: tuple[float, float, float]
    separation: float
    null_probability: float
    n_accepted: int
    leader_software: int

    def vector(
        self, model: str, *, p: float | None = None, clip: float = LOGIT_CLIP
    ) -> tuple[float, ...]:
        """Feature vector in FEATURE_NAMES order; ``p`` overrides the summed probability."""
        if p is None:
            p = min(sum(self.probabilities[: MODELS.index(model) + 1]), 1.0)
        return (
            _logit(p, clip),
            float(self.separation),
            _logit(self.null_probability, clip),
            math.log(max(int(self.n_accepted), 1)),
            float(self.leader_software),
        )


def _active_combiner(context: Any) -> dict[str, Any] | None:
    combiner = getattr(context, "combiner", None)
    if not isinstance(combiner, dict) or combiner.get("status") != "active":
        return None
    if combiner.get("null_probability") is None or combiner.get("leader_probability") is None:
        return None
    return combiner


def _threshold(value: Any) -> float | None:
    return None if value is None else float(value)


def _three(values: Sequence[float]) -> tuple[float, float, float]:
    padded = [float(v) for v in values[:3]] + [0.0] * (3 - min(3, len(values)))
    return padded[0], padded[1], padded[2]


def features(state: Any) -> ConfidenceInputs | None:
    """Inputs from a live analysis state; None without an active combiner or a shortlist."""
    context = getattr(state, "context", None)
    combiner = _active_combiner(context)
    fusion = getattr(state, "deterministic_fusion", None) or getattr(state, "fusion", None)
    entries = list(getattr(fusion, "entries", ())[:3])
    if combiner is None or not entries:
        return None
    probabilities = combiner.get("probabilities") or {}
    decision = getattr(state, "attribution_decision", None)
    return ConfidenceInputs(
        probabilities=_three(
            [float(combiner["leader_probability"])]
            + [float(probabilities.get(entry.group_id, 0.0) or 0.0) for entry in entries[1:]]
        ),
        separation=float(getattr(decision, "separation", 0.0) or 0.0),
        null_probability=float(combiner["null_probability"]),
        n_accepted=len(getattr(context, "observed_techniques", {}) or {}),
        leader_software=int("software_tools" in (entries[0].classifier_ranks or {})),
    )


def features_from_row(row: Mapping[str, Any]) -> ConfidenceInputs | None:
    """The same inputs from an attribution-benchmark row (offline replay)."""
    if row.get("combiner_status") != "active":
        return None
    leader = row.get("combiner_leader_probability")
    null = row.get("combiner_null_probability")
    detail = list(row.get("top3_detail") or [])
    if leader is None or null is None or not detail:
        return None
    accepted = row.get("accepted_confidences")
    if accepted is None:
        accepted = row.get("accepted_techniques") or ()
    return ConfidenceInputs(
        probabilities=_three(
            [float(leader)]
            + [float(item.get("combiner_probability") or 0.0) for item in detail[1:3]]
        ),
        separation=float(row.get("separation") or 0.0),
        null_probability=float(null),
        n_accepted=len(accepted),
        leader_software=int("software_tools" in (detail[0].get("classifier_ranks") or {})),
    )


def row_shortlist(row: Mapping[str, Any]) -> tuple[list[str], dict[str, float]]:
    """(first three group ids, their combiner probabilities) of a benchmark row."""
    detail = list(row.get("top3_detail") or [])[:3]
    return (
        [str(item["group_id"]) for item in detail],
        {str(item["group_id"]): float(item.get("combiner_probability") or 0.0) for item in detail},
    )


# ----------------------------------------------------------------------------- model


@dataclass
class ConfidenceCalibrator:
    intercepts: dict[str, float]
    weights: dict[str, tuple[float, ...]]
    # "strong" (leader) and "set" (top two / top three; None in a strong-only artifact)
    thresholds: dict[str, float | None]
    attack_version: str
    combiner_fingerprint: str
    combiner_sha256: str  # sha256[:24] of the combiner artifact file
    logit_clip: float = LOGIT_CLIP
    selection: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)

    def probability(self, model: str, vector: Sequence[float]) -> float:
        z = self.intercepts[model] + sum(
            w * x for w, x in zip(self.weights[model], vector, strict=True)
        )
        return _sigmoid(z)

    def predict(self, inputs: ConfidenceInputs) -> dict[str, float]:
        """Calibrated P(leader right), P(truth in top two), P(truth in top three)."""
        return {
            model: self.probability(model, inputs.vector(model, clip=self.logit_clip))
            for model in MODELS
        }

    def measured(self) -> dict[str, Any]:
        """Per-label counts the analyst sees (from the fit script's evaluation, never typed)."""
        out: dict[str, Any] = {}
        for key, name in ((FOLD_EVALUATION, "folds"), (REAL_REPORT_EVALUATION, "real_reports")):
            item = self.evaluation.get(key)
            labels = item.get("labels") if isinstance(item, dict) else None
            if not isinstance(labels, dict):
                out[name] = None
                continue
            out[name] = {
                "source": str(item.get("description", key)),
                "rows": int(item.get("rows", 0) or 0),
                "strong": {
                    "n": int(labels["strong"]["n"]),
                    "right": int(labels["strong"]["right"]),
                },
                "set": {"n": int(labels["set"]["n"]), "right": int(labels["set"]["right"])},
                "open": {
                    "n": int(labels["open"]["n"]),
                    "truth_in_top3": int(labels["open"]["truth_in_top3"]),
                },
            }
        return out

    # ----- persistence -----
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "logit_clip": float(self.logit_clip),
            "models": {
                model: {
                    "intercept": float(self.intercepts[model]),
                    "weights": [float(w) for w in self.weights[model]],
                }
                for model in MODELS
            },
            "thresholds": {key: _threshold(self.thresholds.get(key)) for key in ("strong", "set")},
            "selection": dict(self.selection),
            "attack_version": self.attack_version,
            "combiner": {
                "pipeline_fingerprint": self.combiner_fingerprint,
                "artifact_sha256_24": self.combiner_sha256,
            },
            "provenance": dict(self.provenance),
            "evaluation": dict(self.evaluation),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfidenceCalibrator:
        if payload.get("kind") != KIND:
            raise ValueError("not a confidence-label artifact")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema version is not supported")
        if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("feature names do not match the current schema")
        clip = float(payload.get("logit_clip", LOGIT_CLIP))
        if not 0.0 < clip < 0.5:
            raise ValueError("logit clip must lie in (0, 0.5)")
        models = payload.get("models") or {}
        intercepts: dict[str, float] = {}
        weights: dict[str, tuple[float, ...]] = {}
        for model in MODELS:
            entry = models.get(model)
            if not isinstance(entry, Mapping):
                raise ValueError(f"model {model} is missing")
            vector = entry.get("weights")
            if not isinstance(vector, Sequence) or len(vector) != len(FEATURE_NAMES):
                raise ValueError(f"weights of {model} have the wrong shape")
            values = (float(entry["intercept"]), *(float(w) for w in vector))
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"weights of {model} are not finite")
            intercepts[model] = values[0]
            weights[model] = tuple(values[1:])
        stored = payload.get("thresholds") or {}
        # the set threshold may be null (a strong-only artifact); the key must still be there
        thresholds = {"strong": float(stored["strong"]), "set": _threshold(stored["set"])}
        if not all(value is None or 0.0 < value < 1.0 for value in thresholds.values()):
            raise ValueError("thresholds must lie in (0, 1)")
        combiner = payload.get("combiner") or {}
        fingerprint = str(combiner.get("pipeline_fingerprint") or "")
        digest = str(combiner.get("artifact_sha256_24") or "")
        attack_version = str(payload.get("attack_version") or "")
        if not fingerprint or not digest or not attack_version:
            raise ValueError("the ATT&CK version or the combiner pin is missing")
        return cls(
            intercepts=intercepts,
            weights=weights,
            thresholds=thresholds,
            attack_version=attack_version,
            combiner_fingerprint=fingerprint,
            combiner_sha256=digest,
            logit_clip=clip,
            selection=dict(payload.get("selection") or {}),
            provenance=dict(payload.get("provenance") or {}),
            evaluation=dict(payload.get("evaluation") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceCalibrator | None:
        source = Path(path)
        if not source.is_file():
            return None
        return cls.from_dict(json.loads(source.read_text("utf-8")))

    def compatible_with(
        self, attack_version: str, combiner: Any, combiner_path: str | Path | None
    ) -> tuple[bool, str]:
        if self.attack_version != attack_version:
            return False, f"ATT&CK {self.attack_version} != {attack_version}"
        if combiner is None:
            return False, "no active learned combiner (the labels read its probabilities)"
        if str(getattr(combiner, "pipeline_fingerprint", "") or "") != self.combiner_fingerprint:
            return False, "combiner pipeline fingerprint changed (refit the calibration)"
        try:
            digest = combiner_sha256(combiner_path) if combiner_path else ""
        except OSError:
            digest = ""
        if digest != self.combiner_sha256:
            return False, "combiner artifact changed (refit the calibration)"
        return True, "ok"


def combiner_sha256(path: str | Path) -> str:
    """sha256[:24] of a combiner artifact file (the same digest the benchmark records)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]


def load_confidence_labels(
    settings: Any, kb: Any, combiner: Any, combiner_path: str | Path | None
) -> tuple[ConfidenceCalibrator | None, str]:
    """Load the configured calibration fail-closed: ``(model, status)``."""
    if not bool(getattr(settings, "confidence_labels_enabled", False)):
        return None, "disabled"
    path = getattr(settings, "confidence_labels_path", None)
    if not path:
        return None, "no path configured"
    try:
        model = ConfidenceCalibrator.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path}"
    ok, reason = model.compatible_with(
        str(getattr(kb, "version", "") or ""), combiner, combiner_path
    )
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, "active"


# ----------------------------------------------------------------------------- labels


def decide(
    calibrator: ConfidenceCalibrator,
    inputs: ConfidenceInputs | None,
    *,
    stage: str | None,
    group_ids: Sequence[str],
    probabilities: Mapping[str, float],
    named_cluster: Sequence[str] = (),
) -> dict[str, Any]:
    """The label rule, shared by the live view and the offline evaluation.

    insufficient evidence or no shortlist -> open; a named two-group cluster -> that pair
    (top-two model on the pair's summed probability, never "strong"); P(leader) >= strong ->
    strong single answer; else P(top two) >= set -> one of these 2, else P(top three) >= set ->
    one of these 3; otherwise open. Without a set threshold (strong-only) the last two never apply."""
    predicted = calibrator.predict(inputs) if inputs is not None else None
    strong = float(calibrator.thresholds["strong"])  # type: ignore[arg-type]
    shown = calibrator.thresholds.get("set")
    set_bar = math.inf if shown is None else float(shown)
    ids = [str(gid) for gid in group_ids][:3]
    cluster = [str(gid) for gid in named_cluster or ()]
    label, key, members, probability = "open", "open", [], None
    if inputs is None or predicted is None or stage == INSUFFICIENT_STAGE or not ids:
        pass
    elif stage == CLUSTER_STAGE and len(cluster) == 2:
        summed = min(sum(float(probabilities.get(gid, 0.0) or 0.0) for gid in cluster), 1.0)
        probability = calibrator.probability(
            "top2", inputs.vector("top2", p=summed, clip=calibrator.logit_clip)
        )
        label, key, members = "set", "named_cluster", cluster
    elif stage != CLUSTER_STAGE and predicted["leader"] >= strong:
        label, key, members, probability = "strong", "strong", ids[:1], predicted["leader"]
    elif len(ids) >= 2 and predicted["top2"] >= set_bar:
        label, key, members, probability = "set", "one_of_these", ids[:2], predicted["top2"]
    elif len(ids) >= 3 and predicted["top3"] >= set_bar:
        label, key, members, probability = "set", "one_of_these", ids[:3], predicted["top3"]
    if stage == NAMED_STAGE and label != "strong":
        key = "named_lower"  # the name rests on exclusive evidence, the calibration is lower

    def rounded(value: float | None) -> float | None:
        return round(float(value), 4) if value is not None else None

    return {
        "label": label,
        "text_key": key,
        "probability": rounded(probability),
        "p_leader": rounded(predicted["leader"]) if predicted else None,
        "p_top2": rounded(predicted["top2"]) if predicted else None,
        "p_top3": rounded(predicted["top3"]) if predicted else None,
        "set_size": len(members),
        "group_ids": members,
    }


def confidence_label(
    calibrator: ConfidenceCalibrator | None, state: Any, *, kb: Any = None
) -> dict[str, Any] | None:
    """The label block of the Top-three view, or None (no calibration, combiner inactive or
    no "no candidate" probability recorded)."""
    if calibrator is None or _active_combiner(getattr(state, "context", None)) is None:
        return None
    fusion = getattr(state, "deterministic_fusion", None) or getattr(state, "fusion", None)
    entries = list(getattr(fusion, "entries", ())[:3])
    combiner = state.context.combiner
    decision = getattr(state, "attribution_decision", None)
    stage = getattr(getattr(decision, "stage", None), "value", None)
    block = decide(
        calibrator,
        features(state),
        stage=stage,
        group_ids=[entry.group_id for entry in entries],
        probabilities=combiner.get("probabilities") or {},
        named_cluster=tuple(getattr(decision, "named_cluster", ()) or ()),
    )
    groups = getattr(kb, "groups", {}) or {}
    block["members"] = [
        {"group_id": gid, "name": str(getattr(groups.get(gid), "name", gid) or gid)}
        for gid in block["group_ids"]
    ]
    block["thresholds"] = dict(calibrator.thresholds)
    block["measured"] = calibrator.measured()
    block["note"] = NOTE
    return block
