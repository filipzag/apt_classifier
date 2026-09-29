from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
)


class SemanticCentroidClassifier:
    name = "semantic_centroid"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        scores = {
            group_id: score
            for group_id in group_ids(context)
            if (score := context.semantic_profile_scores.get(group_id, 0.0)) > 0
        }
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(context.semantic_profile_scores),
        )
