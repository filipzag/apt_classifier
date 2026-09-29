from apt_classifier.classifiers.base import ClassifierContext, ClassifierResult, build_result


class SpecificBehaviorClassifier:
    name = "specific_behavior"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        raw_scores = getattr(context, "specific_behavior_scores", {})
        raw_signals = getattr(context, "specific_behavior_signals", {})
        scores = {
            group_id: float(score) for group_id, score in raw_scores.items() if float(score) > 0
        }
        signals = {group_id: tuple(raw_signals.get(group_id, ())) for group_id in scores}
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(scores),
            signals=signals,
        )
