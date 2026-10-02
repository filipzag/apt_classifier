from __future__ import annotations

import hashlib
import json
import math
import random
import re
from ast import literal_eval
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TechniqueEvaluationCase:
    case_id: str
    document_id: str
    text: str
    expected_technique_ids: tuple[str, ...]
    source: str = "local"
    published_at: str | None = None
    label_quality: str = "reviewed"
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TechniqueEvaluationCase:
        labels = value.get("expected_technique_ids", value.get("labels", ()))
        return cls(
            case_id=str(value.get("case_id") or value.get("id") or _case_id(value)),
            document_id=str(value.get("document_id") or value.get("case_id") or _case_id(value)),
            text=str(value.get("text") or value.get("text1") or "").strip(),
            expected_technique_ids=_labels(labels),
            source=str(value.get("source") or "local"),
            published_at=_optional_text(value.get("published_at")),
            label_quality=str(value.get("label_quality") or "reviewed"),
            split=_optional_text(value.get("split")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class TechniquePrediction:
    case_id: str
    ranked_technique_ids: tuple[str, ...]
    accepted_technique_ids: tuple[str, ...]

    @classmethod
    def from_ranked(
        cls,
        case_id: str,
        ranked: Sequence[str],
        accepted: Sequence[str] | None = None,
    ) -> TechniquePrediction:
        ranked_ids = tuple(dict.fromkeys(str(item).upper() for item in ranked if str(item)))
        accepted_ids = tuple(
            dict.fromkeys(
                str(item).upper() for item in (accepted if accepted is not None else ranked)
            )
        )
        return cls(case_id, ranked_ids, accepted_ids)


@dataclass(frozen=True, slots=True)
class AttributionEvaluationCase:
    case_id: str
    document_id: str
    expected_group_id: str | None
    observed_technique_ids: tuple[str, ...] = ()
    source: str = "local"
    published_at: str | None = None
    label_quality: str = "reviewed"

    @property
    def is_unknown(self) -> bool:
        return not self.expected_group_id

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttributionEvaluationCase:
        group_id = _optional_text(value.get("expected_group_id", value.get("group_id")))
        if group_id and group_id.casefold() in {"unknown", "unattributed", "none"}:
            group_id = None
        return cls(
            case_id=str(value.get("case_id") or value.get("id") or _case_id(value)),
            document_id=str(value.get("document_id") or value.get("case_id") or _case_id(value)),
            expected_group_id=group_id,
            observed_technique_ids=_labels(value.get("observed_technique_ids", ())),
            source=str(value.get("source") or "local"),
            published_at=_optional_text(value.get("published_at")),
            label_quality=str(value.get("label_quality") or "reviewed"),
        )


@dataclass(frozen=True, slots=True)
class AttributionPrediction:
    case_id: str
    ranked_group_ids: tuple[str, ...]
    named_attribution: bool


@dataclass(frozen=True, slots=True)
class LabelRemap:
    mapped: tuple[str, ...]
    unscorable: tuple[str, ...]
    replacements: dict[str, str]


@dataclass(frozen=True, slots=True)
class AttackIdMap:
    replacements: dict[str, str]

    @classmethod
    def from_bundle(cls, source: str | Path | dict[str, Any]) -> AttackIdMap:
        bundle = (
            source
            if isinstance(source, dict)
            else json.loads(Path(source).read_text(encoding="utf-8"))
        )
        objects = [item for item in bundle.get("objects", ()) if isinstance(item, dict)]
        by_stix = {str(item["id"]): item for item in objects if item.get("id")}
        by_external = {
            str(item["id"]): external_id
            for item in objects
            if item.get("id") and (external_id := _attack_id(item))
        }
        replacements: dict[str, str] = {}
        for item in objects:
            if item.get("type") != "relationship" or item.get("relationship_type") != "revoked-by":
                continue
            source_id = by_external.get(str(item.get("source_ref") or ""))
            target_id = by_external.get(str(item.get("target_ref") or ""))
            target = by_stix.get(str(item.get("target_ref") or ""))
            if source_id and target_id and target and not target.get("revoked"):
                replacements[source_id] = target_id
        return cls(replacements)

    def resolve(self, technique_id: str, known_ids: Iterable[str]) -> str | None:
        known = set(known_ids)
        current = technique_id.upper()
        visited: set[str] = set()
        while current not in visited:
            if current in known:
                return current
            visited.add(current)
            replacement = self.replacements.get(current)
            if replacement is None:
                return None
            current = replacement
        return None

    def remap(self, labels: Iterable[str], known_ids: Iterable[str]) -> LabelRemap:
        known = set(known_ids)
        mapped: list[str] = []
        unscorable: list[str] = []
        replacements: dict[str, str] = {}
        for raw in labels:
            original = str(raw).upper()
            resolved = self.resolve(original, known)
            if resolved is None:
                unscorable.append(original)
                continue
            mapped.append(resolved)
            if resolved != original:
                replacements[original] = resolved
        return LabelRemap(
            tuple(dict.fromkeys(mapped)),
            tuple(dict.fromkeys(unscorable)),
            replacements,
        )


def load_jsonl_cases(
    path: str | Path,
    case_type: type[TechniqueEvaluationCase] | type[AttributionEvaluationCase],
) -> list[TechniqueEvaluationCase] | list[AttributionEvaluationCase]:
    output = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain an object")
        case = case_type.from_dict(value)
        if not case.text.strip() if isinstance(case, TechniqueEvaluationCase) else False:
            raise ValueError(f"line {line_number} has empty text")
        output.append(case)
    return output


def grouped_split(
    cases: Sequence[TechniqueEvaluationCase],
    *,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, tuple[TechniqueEvaluationCase, ...]]:
    if train_fraction <= 0 or dev_fraction < 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("split fractions must leave a non-empty test fraction")
    documents = sorted({case.document_id for case in cases})
    random.Random(seed).shuffle(documents)
    train_end = round(len(documents) * train_fraction)
    dev_end = train_end + round(len(documents) * dev_fraction)
    labels = {
        document_id: "train" if index < train_end else "dev" if index < dev_end else "test"
        for index, document_id in enumerate(documents)
    }
    return {
        split: tuple(case for case in cases if labels[case.document_id] == split)
        for split in ("train", "dev", "test")
    }


def time_ordered_split(
    cases: Sequence[AttributionEvaluationCase],
    *,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
) -> dict[str, tuple[AttributionEvaluationCase, ...]]:
    if train_fraction <= 0 or dev_fraction < 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("split fractions must leave a non-empty test fraction")
    by_document: dict[str, list[AttributionEvaluationCase]] = {}
    for case in cases:
        by_document.setdefault(case.document_id, []).append(case)
    ordered = sorted(
        by_document,
        key=lambda document_id: (
            min(_date_key(item.published_at) for item in by_document[document_id]),
            document_id,
        ),
    )
    train_end = round(len(ordered) * train_fraction)
    dev_end = train_end + round(len(ordered) * dev_fraction)
    labels = {
        document_id: "train" if index < train_end else "dev" if index < dev_end else "test"
        for index, document_id in enumerate(ordered)
    }
    return {
        split: tuple(case for case in cases if labels[case.document_id] == split)
        for split in ("train", "dev", "test")
    }


def duplicate_text_groups(cases: Sequence[TechniqueEvaluationCase]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for case in cases:
        normalized = re.sub(r"[^a-z0-9]+", " ", case.text.casefold()).strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        groups.setdefault(digest, []).append(case.case_id)
    return {
        digest: tuple(case_ids) for digest, case_ids in groups.items() if len(set(case_ids)) > 1
    }


def evaluate_technique_predictions(
    cases: Sequence[TechniqueEvaluationCase],
    predictions: Sequence[TechniquePrediction],
    *,
    kb: Any = None,
) -> dict[str, Any]:
    by_case = {prediction.case_id: prediction for prediction in predictions}
    labels = sorted({item for case in cases for item in case.expected_technique_ids})
    true_positive = false_positive = false_negative = 0
    parent_tp = parent_fp = parent_fn = 0
    per_label = {label: [0, 0, 0] for label in labels}
    positive_count = negative_count = 0
    hit_counts = {1: 0, 3: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    negative_false_positives = 0
    abstained = 0
    correct_abstentions = 0
    exact_matches = 0
    for case in cases:
        prediction = by_case.get(case.case_id, TechniquePrediction(case.case_id, (), ()))
        expected = set(case.expected_technique_ids)
        accepted = set(prediction.accepted_technique_ids)
        ranked = list(prediction.ranked_technique_ids)
        if not accepted:
            abstained += 1
            if not expected:
                correct_abstentions += 1
        if not expected:
            negative_count += 1
            negative_false_positives += int(bool(accepted))
            exact_matches += int(not accepted)
            continue
        positive_count += 1
        exact_matches += int(accepted == expected)
        exact_tp = len(expected & accepted)
        exact_fp = len(accepted - expected)
        exact_fn = len(expected - accepted)
        true_positive += exact_tp
        false_positive += exact_fp
        false_negative += exact_fn
        # Only labels present in the expected or accepted set can change a count;
        # iterating the full label vocabulary per case made this O(cases x labels).
        for label in (expected | accepted) & per_label.keys():
            per_label[label][0] += int(label in expected and label in accepted)
            per_label[label][1] += int(label not in expected and label in accepted)
            per_label[label][2] += int(label in expected and label not in accepted)
        relevant_ranks = [index for index, label in enumerate(ranked, 1) if label in expected]
        reciprocal_ranks.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
        for cutoff in hit_counts:
            hit_counts[cutoff] += int(any(label in expected for label in ranked[:cutoff]))
        ndcg_values.append(_ndcg(ranked[:10], expected))
        matched = _hierarchy_matching(accepted, expected, kb)
        parent_tp += matched
        parent_fp += len(accepted) - matched
        parent_fn += len(expected) - matched
    precision, recall, f1 = _prf(true_positive, false_positive, false_negative)
    parent_precision, parent_recall, parent_f1 = _prf(parent_tp, parent_fp, parent_fn)
    macro = [_prf(*counts) for counts in per_label.values()]
    return {
        "case_count": len(cases),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "hit_at_1": _ratio(hit_counts[1], positive_count),
        "hit_at_3": _ratio(hit_counts[3], positive_count),
        "hit_at_10": _ratio(hit_counts[10], positive_count),
        "mrr": _mean(reciprocal_ranks),
        "ndcg_at_10": _mean(ndcg_values),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_precision": _mean([item[0] for item in macro]),
        "macro_recall": _mean([item[1] for item in macro]),
        "macro_f1": _mean([item[2] for item in macro]),
        "parent_aware_precision": parent_precision,
        "parent_aware_recall": parent_recall,
        "parent_aware_f1": parent_f1,
        "exact_set_match": _ratio(exact_matches, len(cases)),
        "zero_label_false_positive_rate": _ratio(negative_false_positives, negative_count),
        "abstention_precision": _ratio(correct_abstentions, abstained),
        "abstention_recall": _ratio(correct_abstentions, negative_count),
        "per_label": {
            label: {
                "precision": _prf(*counts)[0],
                "recall": _prf(*counts)[1],
                "f1": _prf(*counts)[2],
                "support": counts[0] + counts[2],
            }
            for label, counts in per_label.items()
        },
    }


def bootstrap_technique_metrics(
    cases: Sequence[TechniqueEvaluationCase],
    predictions: Sequence[TechniquePrediction],
    *,
    kb: Any = None,
    samples: int = 1000,
    seed: int = 7,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Bootstrap confidence intervals over cases for canonical scalar metrics.

    Resampling is case-level and deterministic for a given seed.  Per-label
    dictionaries and integer counts are intentionally excluded; their support
    can disappear in a bootstrap replicate and the canonical point estimates
    already contain those details.
    """
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    point = evaluate_technique_predictions(cases, predictions, kb=kb)
    metric_names = tuple(key for key, value in point.items() if isinstance(value, float))
    if not cases:
        return {
            name: {
                "estimate": float(point[name]),
                "lower": 0.0,
                "upper": 0.0,
                "confidence": confidence,
            }
            for name in metric_names
        }

    prediction_by_case = {prediction.case_id: prediction for prediction in predictions}
    rng = random.Random(seed)
    estimates = {name: [] for name in metric_names}
    for _ in range(samples):
        sampled_cases = [cases[rng.randrange(len(cases))] for _ in cases]
        sampled_predictions = [
            prediction_by_case[case.case_id]
            for case in sampled_cases
            if case.case_id in prediction_by_case
        ]
        metrics = evaluate_technique_predictions(sampled_cases, sampled_predictions, kb=kb)
        for name in metric_names:
            estimates[name].append(float(metrics[name]))

    tail = (1.0 - confidence) / 2.0
    lower_index = min(int(tail * samples), samples - 1)
    upper_index = min(int((1.0 - tail) * samples), samples - 1)
    return {
        name: {
            "estimate": float(point[name]),
            "lower": sorted(values)[lower_index],
            "upper": sorted(values)[upper_index],
            "confidence": confidence,
        }
        for name, values in estimates.items()
    }


def evaluate_attribution_predictions(
    cases: Sequence[AttributionEvaluationCase],
    predictions: Sequence[AttributionPrediction],
) -> dict[str, Any]:
    by_case = {prediction.case_id: prediction for prediction in predictions}
    known = [case for case in cases if not case.is_unknown]
    unknown = [case for case in cases if case.is_unknown]
    top1 = top5 = 0
    reciprocal_ranks: list[float] = []
    named_known = named_known_correct = 0
    for case in known:
        prediction = by_case.get(case.case_id, AttributionPrediction(case.case_id, (), False))
        expected = str(case.expected_group_id)
        rank = (
            prediction.ranked_group_ids.index(expected) + 1
            if expected in prediction.ranked_group_ids
            else None
        )
        top1 += int(rank == 1)
        top5 += int(rank is not None and rank <= 5)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if prediction.named_attribution:
            named_known += 1
            named_known_correct += int(rank == 1)
    unknown_rejected = sum(
        not by_case.get(
            case.case_id, AttributionPrediction(case.case_id, (), False)
        ).named_attribution
        for case in unknown
    )
    unknown_named = len(unknown) - unknown_rejected
    named_total = named_known + unknown_named
    named_errors = (named_known - named_known_correct) + unknown_named
    return {
        "case_count": len(cases),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "top1": _ratio(top1, len(known)),
        "top5": _ratio(top5, len(known)),
        "mrr": _mean(reciprocal_ranks),
        "unknown_recall": _ratio(unknown_rejected, len(unknown)),
        "false_named_attribution_rate": _ratio(unknown_named, len(unknown)),
        "named_coverage": _ratio(named_total, len(cases)),
        "named_risk": _ratio(named_errors, named_total),
    }


def bootstrap_interval(
    values: Sequence[Any],
    metric: Callable[[Sequence[Any]], float],
    *,
    samples: int = 1000,
    seed: int = 7,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = random.Random(seed)
    estimates = sorted(
        metric([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = estimates[min(int(tail * samples), samples - 1)]
    upper = estimates[min(int((1.0 - tail) * samples), samples - 1)]
    return float(metric(values)), float(lower), float(upper)


def source_fingerprint(paths: Iterable[str | Path]) -> str:
    """SHA-256 over the contents of the given source files, in sorted path order.

    Benchmark artifacts include this so that changes to gating or cue code are visible
    even when the configured settings are unchanged.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def pipeline_fingerprint(payload: dict[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode()).hexdigest()


def write_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    material = dict(payload)
    material["artifact_hash"] = pipeline_fingerprint(material)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(material, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _labels(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = literal_eval(stripped)
            value = parsed if isinstance(parsed, list | tuple | set) else [parsed]
        else:
            value = re.split(r"[,;\s]+", stripped)
    return tuple(
        dict.fromkeys(str(item).strip().upper() for item in (value or ()) if str(item).strip())
    )


def _attack_id(item: dict[str, Any]) -> str | None:
    for reference in item.get("external_references", ()):
        if isinstance(reference, dict) and reference.get("external_id"):
            identifier = str(reference["external_id"]).upper()
            if identifier.startswith(("T", "G", "S", "C", "DET", "AN", "DC")):
                return identifier
    return None


def _case_id(value: dict[str, Any]) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"eval-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _date_key(value: str | None) -> tuple[int, str]:
    if not value:
        return 1, ""
    try:
        return 0, datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return 0, value


def _ndcg(ranked: Sequence[str], expected: set[str]) -> float:
    gain = sum(
        1.0 / math.log2(index + 1) for index, item in enumerate(ranked, 1) if item in expected
    )
    ideal = sum(
        1.0 / math.log2(index + 1) for index in range(1, min(len(expected), len(ranked)) + 1)
    )
    return gain / ideal if ideal else 0.0


def _hierarchy_compatible(predicted: str, expected: str, kb: Any) -> bool:
    if predicted == expected:
        return True
    techniques = getattr(kb, "techniques", {}) if kb is not None else {}
    predicted_item = techniques.get(predicted)
    expected_item = techniques.get(expected)
    predicted_parent = getattr(predicted_item, "parent_id", None)
    expected_parent = getattr(expected_item, "parent_id", None)
    return predicted_parent == expected or expected_parent == predicted


def _hierarchy_matching(accepted: set[str], expected: set[str], kb: Any) -> int:
    """Size of a maximum one-to-one matching between accepted and expected ids, where two ids
    may pair when ``_hierarchy_compatible`` (equal, or parent and sub-technique). Augmenting
    paths over sorted ids: the count never depends on set iteration order (PYTHONHASHSEED),
    which a first-fit pairing did."""
    if not accepted or not expected:
        return 0
    expected_ids = sorted(expected)
    candidates = {
        predicted_id: [
            expected_id
            for expected_id in expected_ids
            if _hierarchy_compatible(predicted_id, expected_id, kb)
        ]
        for predicted_id in sorted(accepted)
    }
    owner: dict[str, str] = {}

    def augment(predicted_id: str, visited: set[str]) -> bool:
        for expected_id in candidates[predicted_id]:
            if expected_id in visited:
                continue
            visited.add(expected_id)
            if expected_id not in owner or augment(owner[expected_id], visited):
                owner[expected_id] = predicted_id
                return True
        return False

    return sum(augment(predicted_id, set()) for predicted_id in candidates)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
