from math import fsum, sqrt

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


class IDFOverlapClassifier:
    """IDF-weighted cosine between the evidence techniques and a group's documented list.

    ``length_norm`` damps the group-size normalisation (pivoted length normalisation as in
    BM25's ``b``): with 1.0 the score divides by the full group norm, which penalises groups
    with long ATT&CK technique lists whenever only a few techniques were observed; with 0.5
    it divides by the square root of that norm, and with 0 the group size is ignored.
    """

    name = "idf_tfidf_cosine"

    def __init__(self, length_norm: float = 1.0) -> None:
        self.length_norm = max(0.0, min(1.0, float(length_norm)))

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        evidence_techniques = ranking_techniques(context)
        # fsum throughout: exactly rounded, so no score depends on set or dict iteration order
        observed_norm = sqrt(
            fsum(
                (confidence * idf_for(context, tid)) ** 2
                for tid, confidence in evidence_techniques.items()
            )
        )
        scores: dict[str, float] = {}
        evidence = {}
        for group_id in group_ids(context):
            known = group_techniques(context, group_id)
            group_norm = sqrt(fsum(idf_for(context, tid) ** 2 for tid in known)) ** self.length_norm
            matches = hierarchy_matches(context, group_id)
            weights = {
                tid: evidence_techniques[tid] * idf_for(context, tid) ** 2 * match.factor
                for tid, match in matches.items()
            }
            denominator = observed_norm * group_norm
            score = fsum(weights.values()) / denominator if denominator else 0.0
            if score <= 0.0:
                continue
            scores[group_id] = score
            evidence[group_id] = contributions_for(context, group_id, weights)
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(evidence_techniques),
            contributions=evidence,
        )
