from dataclasses import asdict, dataclass, field, replace
from math import sqrt
from statistics import fmean

from apt_classifier.classifiers.base import ClassifierResult, rerank


@dataclass(frozen=True)
class FusionEntry:
    group_id: str
    score: float
    relative_score: float
    rank: int
    classifier_ranks: dict[str, int]
    rank_min: int
    rank_max: int
    rank_std: float
    contested: bool


@dataclass(frozen=True)
class FusionResult:
    entries: tuple[FusionEntry, ...]
    active_classifiers: tuple[str, ...]
    correlations: dict[str, float]
    k: int
    effective_weights: dict[str, float] = field(default_factory=dict)

    def by_group(self) -> dict[str, FusionEntry]:
        return {entry.group_id: entry for entry in self.entries}

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [asdict(entry) for entry in self.entries],
            "active_classifiers": list(self.active_classifiers),
            "correlations": self.correlations,
            "k": self.k,
            "effective_weights": self.effective_weights,
        }


def _rerank_result(result: ClassifierResult) -> ClassifierResult:
    return rerank(result)


def reciprocal_rank_fusion(
    results: tuple[ClassifierResult, ...] | list[ClassifierResult],
    weights: dict[str, float] | None = None,
    k: int = 60,
    eligible_group_ids: set[str] | None = None,
    classifier_families: dict[str, str] | None = None,
    family_weights: dict[str, float] | None = None,
    tie_breaker: dict[str, float] | None = None,
) -> FusionResult:
    """Family-aware reciprocal rank fusion.

    ``tie_breaker`` maps group ids to a value sorted ascending on equal fused scores
    (the engine passes ATT&CK prominence, so the *less*-reported group wins a tie);
    without it ties fall back to the group id, which favours older ``G####`` ids.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    weights = weights or {}
    active = [
        result
        for result in results
        if result.applicable and result.entries and weights.get(result.classifier, 1.0) > 0
    ]
    if eligible_group_ids is not None:
        active = [
            replace(
                result,
                entries=tuple(
                    entry for entry in result.entries if entry.group_id in eligible_group_ids
                ),
            )
            for result in active
        ]
        active = [result for result in active if result.entries]
        active = [_rerank_result(result) for result in active]
    if not active:
        return FusionResult((), (), {}, k)
    effective_weights = _effective_classifier_weights(
        active,
        weights,
        classifier_families or {},
        family_weights or {},
    )
    active = [result for result in active if effective_weights.get(result.classifier, 0.0) > 0]
    if not active:
        return FusionResult((), (), {}, k)
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for result in active:
        weight = effective_weights[result.classifier]
        for entry in result.entries:
            scores[entry.group_id] = scores.get(entry.group_id, 0.0) + weight / (k + entry.rank)
            ranks.setdefault(entry.group_id, {})[result.classifier] = entry.rank
    breaker = tie_breaker or {}
    ordered = sorted(
        scores.items(), key=lambda item: (-item[1], float(breaker.get(item[0], 0.0)), item[0])
    )
    maximum = ordered[0][1] if ordered else 1.0
    group_count = len(ordered)
    entries = []
    for rank, (group_id, score) in enumerate(ordered, 1):
        values = list(ranks[group_id].values())
        mean = fmean(values)
        std = sqrt(fmean([(value - mean) ** 2 for value in values])) if len(values) > 1 else 0.0
        rank_min = min(values)
        rank_max = max(values)
        entries.append(
            FusionEntry(
                group_id=group_id,
                score=score,
                relative_score=100.0 * score / maximum if maximum else 0.0,
                rank=rank,
                classifier_ranks=ranks[group_id],
                rank_min=rank_min,
                rank_max=rank_max,
                rank_std=std,
                contested=rank_max - rank_min >= max(5, round(group_count * 0.15)),
            )
        )
    correlations = pairwise_kendall(tuple(active))
    return FusionResult(
        tuple(entries),
        tuple(result.classifier for result in active),
        correlations,
        k,
        dict(sorted(effective_weights.items())),
    )


def _effective_classifier_weights(
    active: list[ClassifierResult],
    weights: dict[str, float],
    classifier_families: dict[str, str],
    family_weights: dict[str, float],
) -> dict[str, float]:
    effective = {result.classifier: float(weights.get(result.classifier, 1.0)) for result in active}
    members_by_family: dict[str, list[str]] = {}
    for classifier, family in classifier_families.items():
        if classifier in effective:
            members_by_family.setdefault(family, []).append(classifier)
    for family, members in members_by_family.items():
        if family not in family_weights:
            continue
        total = sum(effective[classifier] for classifier in members)
        budget = max(float(family_weights[family]), 0.0)
        if total <= 0 or total <= budget:
            continue
        for classifier in members:
            effective[classifier] = budget * effective[classifier] / total
    return effective


def kendall_tau(first: dict[str, int], second: dict[str, int]) -> float:
    groups = sorted(set(first) & set(second))
    if len(groups) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            product = (first[left] - first[right]) * (second[left] - second[right])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 0.0


def pairwise_kendall(results: tuple[ClassifierResult, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    rank_maps = {
        result.classifier: {entry.group_id: entry.rank for entry in result.entries}
        for result in results
    }
    names = sorted(rank_maps)
    for index, first in enumerate(names):
        output[f"{first}|{first}"] = 1.0
        for second in names[index + 1 :]:
            value = kendall_tau(rank_maps[first], rank_maps[second])
            output[f"{first}|{second}"] = value
            output[f"{second}|{first}"] = value
    return output
