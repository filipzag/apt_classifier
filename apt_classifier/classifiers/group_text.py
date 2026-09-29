from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
)


class GroupTextClassifier:
    name = "group_description"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        scores = {
            group_id: context.group_text_scores.get(group_id, 0.0)
            for group_id in group_ids(context)
        }
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(context.group_text_scores and context.narrative.strip()),
        )
