from math import fsum

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    contributions_for,
    group_ids,
    group_techniques,
    hierarchy_matches,
    idf_for,
    ranking_techniques,
)


class WeightedJaccardClassifier:
    name = "weighted_jaccard"

    def __init__(self, metric: str = "jaccard") -> None:
        if metric not in {"jaccard", "dice"}:
            raise ValueError("metric must be jaccard or dice")
        self.metric = metric

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        observed = {
            tid: confidence * idf_for(context, tid)
            for tid, confidence in ranking_techniques(context).items()
        }
        scores: dict[str, float] = {}
        evidence = {}
        for group_id in group_ids(context):
            known = group_techniques(context, group_id)
            group = {tid: idf_for(context, tid) for tid in known}
            matches = hierarchy_matches(context, group_id)
            # fsum throughout: exactly rounded, so no score depends on set or dict iteration order
            intersection = fsum(
                min(observed[tid], group.get(match.group_technique_id, 0.0)) * match.factor
                for tid, match in matches.items()
            )
            union = set(observed) | known
            if self.metric == "dice":
                denominator = fsum([*observed.values(), *group.values()])
                score = 2.0 * intersection / denominator if denominator else 0.0
            else:
                denominator = fsum(
                    max(observed.get(tid, 0.0), group.get(tid, 0.0)) for tid in union
                )
                score = intersection / denominator if denominator else 0.0
            if score > 0:
                scores[group_id] = score
                weights = {
                    tid: min(observed[tid], group.get(match.group_technique_id, 0.0)) * match.factor
                    for tid, match in matches.items()
                }
                evidence[group_id] = contributions_for(context, group_id, weights)
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(observed),
            contributions=evidence,
        )
