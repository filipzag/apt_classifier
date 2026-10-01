"""Ensemble members backed by the random-forest models in ``apt_classifier.profile_forest``.

``profile_forest`` votes with the group forest's probability for each group. ``source_country``
votes for each group with the country forest's probability of that group's documented source
country class (China, Russia, Iran, North Korea, another state, or unattributed), so every group
is scored and groups without a documented state origin are not disadvantaged. Both abstain
without a loaded model or without accepted techniques, and neither can make a group eligible:
RRF still gates on the deterministic technical classifiers.
"""

from __future__ import annotations

from typing import Any

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
)

PROFILE_FOREST_NAME = "profile_forest"
SOURCE_COUNTRY_NAME = "source_country"


def _predict(
    model: Any, context: ClassifierContext
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Group and country probabilities for this context, computed once per fusion pass."""
    from apt_classifier.profile_forest import evidence_vector

    if model is None or not context.observed_techniques:
        return None
    vector = evidence_vector(context, model.space)
    key = (id(model), vector.tobytes())
    cached = getattr(context, "profile_forest_prediction", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    groups, countries = model.predict(vector)
    group_probabilities = {
        group_id: float(value)
        for group_id, value in zip(model.group_ids, groups[0], strict=True)
        if value > 0
    }
    country_probabilities = {
        name: float(value) for name, value in zip(model.country_classes, countries[0], strict=True)
    }
    context.profile_forest_prediction = (key, group_probabilities, country_probabilities)
    return group_probabilities, country_probabilities


class ProfileForestClassifier:
    name = PROFILE_FOREST_NAME

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        prediction = _predict(self.model, context)
        if prediction is None:
            return build_result(self.name, context, {}, applicable=False)
        allowed = set(group_ids(context))
        scores = {group_id: p for group_id, p in prediction[0].items() if group_id in allowed}
        signals = {group_id: (f"profile_forest:p={p:.3f}",) for group_id, p in scores.items()}
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)


class SourceCountryClassifier:
    name = SOURCE_COUNTRY_NAME

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        prediction = _predict(self.model, context)
        if prediction is None:
            context.source_country = None
            return build_result(self.name, context, {}, applicable=False)
        countries = prediction[1]
        top = max(countries, key=lambda name: (countries[name], name)) if countries else None
        context.source_country = {"probabilities": countries, "top": top}
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for group_id in group_ids(context):
            country_class = self.model.group_classes.get(group_id)
            probability = countries.get(country_class, 0.0) if country_class else 0.0
            if probability > 0:
                scores[group_id] = probability
                signals[group_id] = (f"source_country:{country_class}:{probability:.2f}",)
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)
