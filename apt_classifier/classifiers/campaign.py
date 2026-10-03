from math import fsum

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    TechniqueContribution,
    build_result,
    group_ids,
    hierarchy_matches_for,
    idf_for,
    kb_mapping,
)


class CampaignClassifier:
    name = "campaign_profile"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        campaign_techniques = kb_mapping(context.kb, "campaign_techniques")
        campaign_groups = kb_mapping(context.kb, "campaign_groups")
        observed = {
            tid: confidence * idf_for(context, tid)
            for tid, confidence in context.observed_techniques.items()
        }
        valid_groups = set(group_ids(context))
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        evidence: dict[str, tuple[TechniqueContribution, ...]] = {}
        for campaign_id, raw_techniques in campaign_techniques.items():
            techniques = set(raw_techniques)
            matches = hierarchy_matches_for(
                context, None, observed_techniques=observed, known_techniques=techniques
            )
            campaign_known_matches = {
                tid: match
                for tid, match in matches.items()
                if match.group_technique_id in techniques
            }
            # fsum: exactly rounded, so the score does not depend on set or dict iteration order
            numerator = fsum(
                min(observed[tid], idf_for(context, match.group_technique_id)) * match.factor
                for tid, match in campaign_known_matches.items()
            )
            union = set(observed) | techniques
            denominator = fsum(
                max(
                    observed.get(tid, 0.0),
                    idf_for(context, tid) if tid in techniques else 0.0,
                )
                for tid in union
            )
            campaign_score = numerator / denominator if denominator else 0.0
            if campaign_score <= 0:
                continue
            for group_id in campaign_groups.get(campaign_id, ()):
                if group_id not in valid_groups or campaign_score <= scores.get(group_id, 0.0):
                    continue
                scores[group_id] = campaign_score
                signals[group_id] = (campaign_id,)
                evidence[group_id] = tuple(
                    TechniqueContribution(
                        tid,
                        min(observed[tid], idf_for(context, match.group_technique_id))
                        * match.factor,
                        context.observed_techniques[tid],
                        tuple(context.technique_observations.get(tid, ())),
                        match.group_technique_id,
                        match.relationship,
                    )
                    for tid, match in sorted(campaign_known_matches.items())
                )
        applicable = bool(scores)
        return build_result(
            self.name,
            context,
            scores,
            applicable=applicable,
            contributions=evidence,
            signals=signals,
        )
