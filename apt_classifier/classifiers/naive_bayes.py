from math import exp

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
    group_techniques,
    hierarchy_matches,
    prior_for,
    ranking_techniques,
    safe_log,
)


class BernoulliNaiveBayesClassifier:
    name = "bernoulli_naive_bayes"

    def __init__(self, alpha: float = 0.6, beta: float = 0.05, negative_weight: float = 0.3):
        if not 0 < beta < alpha < 1:
            raise ValueError("Expected 0 < beta < alpha < 1")
        self.alpha = alpha
        self.beta = beta
        self.negative_weight = negative_weight

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        evidence = ranking_techniques(context)
        applicable = bool(evidence or context.negative_techniques)
        scores: dict[str, float] = {}
        for group_id in group_ids(context):
            known = group_techniques(context, group_id)
            matches = hierarchy_matches(context, group_id)
            score = safe_log(prior_for(context, group_id))
            for technique_id, confidence in evidence.items():
                match = matches.get(technique_id)
                if match is not None:
                    theta = self.beta + (self.alpha - self.beta) * match.factor
                else:
                    theta = self.beta
                score += float(confidence) * safe_log(theta)
            for technique_id in context.negative_techniques:
                theta = self.alpha if technique_id in known else self.beta
                score += self.negative_weight * safe_log(1.0 - theta)
            scores[group_id] = score
        return build_result(self.name, context, scores, applicable=applicable)


def posterior(result: ClassifierResult, temperature: float = 1.0) -> dict[str, float]:
    if not result.applicable or not result.entries:
        return {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = {entry.group_id: entry.score / temperature for entry in result.entries}
    maximum = max(scaled.values())
    weights = {group_id: exp(score - maximum) for group_id, score in scaled.items()}
    total = sum(weights.values())
    return {group_id: value / total for group_id, value in weights.items()}
