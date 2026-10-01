from typing import Any

from apt_classifier.classifiers.base import Classifier, ClassifierContext, ClassifierResult
from apt_classifier.classifiers.campaign import CampaignClassifier
from apt_classifier.classifiers.cyberre import CyberREClassifier
from apt_classifier.classifiers.group_text import GroupTextClassifier
from apt_classifier.classifiers.idf_overlap import IDFOverlapClassifier
from apt_classifier.classifiers.jaccard import WeightedJaccardClassifier
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier
from apt_classifier.classifiers.profile_forest import (
    ProfileForestClassifier,
    SourceCountryClassifier,
)
from apt_classifier.classifiers.semantic_centroid import SemanticCentroidClassifier
from apt_classifier.classifiers.software import COMMODITY_TOOL_WEIGHT_CAP, SoftwareClassifier
from apt_classifier.classifiers.specific_behavior import SpecificBehaviorClassifier
from apt_classifier.classifiers.strategic_context import (
    MotivationClassifier,
    StrategicContextClassifier,
)


def default_classifiers(
    alpha: float = 0.6,
    beta: float = 0.05,
    negative_weight: float = 0.3,
    *,
    software_idf: bool = True,
    software_tool_weight_cap: float = COMMODITY_TOOL_WEIGHT_CAP,
    idf_length_norm: float = 1.0,
    profile_forest: Any = None,
    source_country_score_all: bool = False,
) -> tuple[Classifier, ...]:
    return (
        BernoulliNaiveBayesClassifier(alpha, beta, negative_weight),
        IDFOverlapClassifier(length_norm=idf_length_norm),
        WeightedJaccardClassifier(),
        SpecificBehaviorClassifier(),
        SemanticCentroidClassifier(),
        GroupTextClassifier(),
        StrategicContextClassifier(),
        MotivationClassifier(),
        CampaignClassifier(),
        SoftwareClassifier(use_idf=software_idf, tool_weight_cap=software_tool_weight_cap),
        CyberREClassifier(),
        ProfileForestClassifier(profile_forest),
        SourceCountryClassifier(profile_forest, score_all=source_country_score_all),
    )


def run_classifiers(
    context: ClassifierContext, classifiers: tuple[Classifier, ...]
) -> tuple[ClassifierResult, ...]:
    return tuple(classifier.classify(context) for classifier in classifiers)
