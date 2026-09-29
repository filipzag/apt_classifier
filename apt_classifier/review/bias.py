from dataclasses import asdict, dataclass
from math import sqrt
from statistics import fmean

from apt_classifier.fusion import kendall_tau


@dataclass(frozen=True)
class FameBiasReport:
    blinded_ranks: dict[str, int]
    unblinded_ranks: dict[str, int]
    promotion_when_named: dict[str, int]
    kendall_tau: float
    prominence_correlation: float
    largest_promotions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_fame_bias(
    blinded_ranks: dict[str, int],
    unblinded_ranks: dict[str, int],
    prominence: dict[str, float],
) -> FameBiasReport:
    common = sorted(set(blinded_ranks) & set(unblinded_ranks))
    promotion = {
        group_id: blinded_ranks[group_id] - unblinded_ranks[group_id] for group_id in common
    }
    correlation = _pearson(
        [float(prominence.get(group_id, 0.0)) for group_id in common],
        [float(promotion[group_id]) for group_id in common],
    )
    largest = tuple(sorted(common, key=lambda group_id: (-promotion[group_id], group_id))[:3])
    return FameBiasReport(
        blinded_ranks={group_id: blinded_ranks[group_id] for group_id in common},
        unblinded_ranks={group_id: unblinded_ranks[group_id] for group_id in common},
        promotion_when_named=promotion,
        kendall_tau=kendall_tau(blinded_ranks, unblinded_ranks),
        prominence_correlation=correlation,
        largest_promotions=largest,
    )


def extract_ranks(
    parsed: dict[str, object], id_map: dict[str, str] | None = None
) -> dict[str, int]:
    output: dict[str, int] = {}
    ranking = parsed.get("ranking", [])
    if not isinstance(ranking, list):
        return output
    for index, item in enumerate(ranking, 1):
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("candidate_id", ""))
        group_id = (id_map or {}).get(candidate, candidate)
        if group_id:
            output[group_id] = int(item.get("rank") or index)
    return output


def _pearson(first: list[float], second: list[float]) -> float:
    if len(first) < 2 or len(first) != len(second):
        return 0.0
    first_mean = fmean(first)
    second_mean = fmean(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_variance = sum((value - first_mean) ** 2 for value in first)
    second_variance = sum((value - second_mean) ** 2 for value in second)
    denominator = sqrt(first_variance * second_variance)
    return numerator / denominator if denominator else 0.0
