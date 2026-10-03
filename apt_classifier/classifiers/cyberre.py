from math import fsum

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierEvidence,
    ClassifierResult,
    TechniqueContribution,
    group_ids,
    group_techniques,
    idf_for,
)
from apt_classifier.cyberre import CYBERRE_CLASSIFIER_NAME


class CyberREClassifier:
    name = CYBERRE_CLASSIFIER_NAME

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        inferred = dict(getattr(context, "cyberre_techniques", {}))
        if not inferred:
            return ClassifierResult(self.name, False, (), 0)
        observation_ids = getattr(context, "cyberre_technique_observations", {})
        scores: dict[str, float] = {}
        contributions: dict[str, tuple[TechniqueContribution, ...]] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for group_id in group_ids(context):
            overlap = sorted(
                set(inferred) & group_techniques(context, group_id),
                key=lambda technique_id: (-inferred[technique_id], technique_id),
            )
            if not overlap:
                continue
            group_contributions = tuple(
                TechniqueContribution(
                    technique_id=technique_id,
                    weight=float(inferred[technique_id]) * idf_for(context, technique_id),
                    confidence=float(inferred[technique_id]),
                    observation_ids=tuple(observation_ids.get(technique_id, ())),
                )
                for technique_id in overlap
            )
            scores[group_id] = fsum(item.weight for item in group_contributions)
            contributions[group_id] = group_contributions
            signals[group_id] = tuple(
                f"cyberre:{technique_id}:rank:{round(1.0 / inferred[technique_id])}"
                for technique_id in overlap
                if inferred[technique_id] > 0
            )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        entries = tuple(
            ClassifierEvidence(
                classifier=self.name,
                group_id=group_id,
                score=score,
                rank=index,
                matched_techniques=contributions[group_id],
                unmatched_observed=tuple(
                    sorted(set(inferred) - group_techniques(context, group_id))
                ),
                signal_ids=signals[group_id],
            )
            for index, (group_id, score) in enumerate(ranked, 1)
        )
        return ClassifierResult(self.name, bool(entries), entries, len(inferred))
