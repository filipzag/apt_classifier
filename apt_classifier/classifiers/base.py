from dataclasses import dataclass, field, replace
from math import log
from typing import Any, Protocol


@dataclass(frozen=True)
class TechniqueContribution:
    technique_id: str
    weight: float
    confidence: float
    observation_ids: tuple[str, ...] = ()
    matched_group_technique_id: str | None = None
    relationship: str = "exact"


@dataclass(frozen=True)
class HierarchyMatch:
    observed_technique_id: str
    group_technique_id: str
    relationship: str
    factor: float


@dataclass(frozen=True)
class ClassifierEvidence:
    classifier: str
    group_id: str
    score: float
    rank: int
    matched_techniques: tuple[TechniqueContribution, ...] = ()
    unmatched_observed: tuple[str, ...] = ()
    signal_ids: tuple[str, ...] = ()
    applicable: bool = True


@dataclass(frozen=True)
class ClassifierResult:
    classifier: str
    applicable: bool
    entries: tuple[ClassifierEvidence, ...]
    input_count: int = 0

    def by_group(self) -> dict[str, ClassifierEvidence]:
        return {entry.group_id: entry for entry in self.entries}


@dataclass
class ClassifierContext:
    kb: Any
    observed_techniques: dict[str, float]
    technique_observations: dict[str, tuple[str, ...]]
    negative_techniques: set[str]
    semantic_profile_scores: dict[str, float]
    group_text_scores: dict[str, float]
    software_ids: set[str]
    narrative: str = ""
    cyberre_techniques: dict[str, float] = field(default_factory=dict)
    cyberre_technique_observations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    parent_policy: str = "ancestor_compatible"
    eligible_group_ids: set[str] | None = None
    group_eligibility_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    decisive_software: dict[str, Any] = field(default_factory=dict)
    # Near-miss technique candidates (still NEEDS_REVIEW) weighted by their calibrated
    # acceptance probability. They feed the group ranking only; the accepted set, the
    # interview and the naming gate use ``observed_techniques``.
    soft_techniques: dict[str, float] = field(default_factory=dict)
    soft_technique_observations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Actor-behaviour matches (group, observation, technique, similarity, matched text) and the
    # candidate-relative discrimination result of the last fusion (see separation.py).
    specific_behavior_matches: tuple[Any, ...] = ()
    discrimination: Any = None
    # near-miss candidates that only enter the candidate-relative separation (never a classifier)
    contrast_techniques: dict[str, float] = field(default_factory=dict)
    contrast_technique_observations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # random-forest country estimate for the last scoring pass ({probabilities, top} or None) and
    # the cached forest prediction it came from (see classifiers/profile_forest.py)
    source_country: dict[str, Any] | None = None
    profile_forest_prediction: Any = None
    # learned combiner output for the last fusion: probabilities, leader/null probability, status
    combiner: Any = None


class Classifier(Protocol):
    name: str

    def classify(self, context: ClassifierContext) -> ClassifierResult: ...


def kb_mapping(kb: Any, name: str) -> dict[str, Any]:
    value = getattr(kb, name, {})
    return value if isinstance(value, dict) else {}


def ranking_techniques(context: ClassifierContext) -> dict[str, float]:
    """Techniques the group classifiers rank on: accepted ones at their confidence plus
    soft (near-miss) ones at their probability; an accepted technique always wins."""
    soft = getattr(context, "soft_techniques", None) or {}
    if not soft:
        return context.observed_techniques
    return {**soft, **context.observed_techniques}


def technique_observation_ids(context: ClassifierContext, technique_id: str) -> tuple[str, ...]:
    ids = context.technique_observations.get(technique_id)
    if ids:
        return tuple(ids)
    return tuple(
        (getattr(context, "soft_technique_observations", None) or {}).get(technique_id, ())
    )


def group_ids(context: ClassifierContext) -> list[str]:
    known = set(kb_mapping(context.kb, "groups"))
    if context.eligible_group_ids is not None:
        known &= set(context.eligible_group_ids)
    return sorted(known)


def group_techniques(context: ClassifierContext, group_id: str) -> set[str]:
    return set(kb_mapping(context.kb, "group_techniques").get(group_id, ()))


def hierarchy_matches_for(
    context: ClassifierContext,
    group_id: str | None,
    observed_techniques: dict[str, float] | set[str] | None = None,
    known_techniques: set[str] | None = None,
) -> dict[str, HierarchyMatch]:
    observed = (
        set(observed_techniques)
        if observed_techniques is not None
        else set(ranking_techniques(context))
    )
    known = (
        set(known_techniques)
        if known_techniques is not None
        else group_techniques(context, group_id or "")
    )
    techniques = kb_mapping(context.kb, "techniques")
    children_by_parent: dict[str, set[str]] = {}
    for technique_id, technique in techniques.items():
        parent_id = _record_value(technique, "parent_id", None)
        if parent_id:
            children_by_parent.setdefault(str(parent_id), set()).add(technique_id)
    output: dict[str, HierarchyMatch] = {}
    for technique_id in observed:
        if technique_id in known:
            output[technique_id] = HierarchyMatch(
                technique_id,
                technique_id,
                "exact",
                1.0,
            )
            continue
        if context.parent_policy == "exact_only":
            continue
        technique = techniques.get(technique_id)
        parent_id = _record_value(technique, "parent_id", None)
        if parent_id and str(parent_id) in known:
            output[technique_id] = HierarchyMatch(
                technique_id,
                str(parent_id),
                "ancestor",
                0.95,
            )
            continue
        documented_children = sorted(children_by_parent.get(technique_id, set()) & known)
        if documented_children:
            output[technique_id] = HierarchyMatch(
                technique_id,
                documented_children[0],
                "descendant",
                0.85,
            )
    return output


def hierarchy_matches(context: ClassifierContext, group_id: str) -> dict[str, HierarchyMatch]:
    return hierarchy_matches_for(context, group_id)


def negative_hierarchy_factor(
    context: ClassifierContext,
    group_id: str,
    technique_id: str,
) -> float:
    known = group_techniques(context, group_id)
    if technique_id in known:
        return 1.0
    if context.parent_policy == "exact_only":
        return 0.0
    techniques = kb_mapping(context.kb, "techniques")
    for known_id in known:
        if str(_record_value(techniques.get(known_id), "parent_id", "")) == technique_id:
            return 0.85
    return 0.0


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def idf_for(context: ClassifierContext, technique_id: str) -> float:
    return float(kb_mapping(context.kb, "idf").get(technique_id, 1.0))


def prior_for(context: ClassifierContext, group_id: str) -> float:
    priors = kb_mapping(context.kb, "group_priors") or kb_mapping(context.kb, "priors")
    if group_id in priors:
        return max(float(priors[group_id]), 1e-12)
    count = max(len(group_ids(context)), 1)
    return 1.0 / count


def contributions_for(
    context: ClassifierContext, group_id: str, weights: dict[str, float] | None = None
) -> tuple[TechniqueContribution, ...]:
    matches = hierarchy_matches(context, group_id)
    output = []
    for technique_id in sorted(matches):
        match = matches[technique_id]
        confidence = float(ranking_techniques(context).get(technique_id, 0.0))
        if weights is None:
            base_weight = confidence * idf_for(context, technique_id) * match.factor
        else:
            base_weight = weights.get(technique_id, 0.0)
        output.append(
            TechniqueContribution(
                technique_id,
                float(base_weight),
                confidence,
                technique_observation_ids(context, technique_id),
                match.group_technique_id,
                match.relationship,
            )
        )
    return tuple(output)


def build_result(
    classifier: str,
    context: ClassifierContext,
    scores: dict[str, float],
    *,
    applicable: bool,
    contributions: dict[str, tuple[TechniqueContribution, ...]] | None = None,
    signals: dict[str, tuple[str, ...]] | None = None,
) -> ClassifierResult:
    if not applicable:
        return ClassifierResult(classifier, False, (), len(context.observed_techniques))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    entries = []
    observed = set(context.observed_techniques)
    previous_score: float | None = None
    dense_rank = 0
    for group_id, score in ranked:
        numeric_score = float(score)
        if previous_score is None or numeric_score != previous_score:
            dense_rank += 1
            previous_score = numeric_score
        matched = set(hierarchy_matches(context, group_id))
        entries.append(
            ClassifierEvidence(
                classifier=classifier,
                group_id=group_id,
                score=numeric_score,
                rank=dense_rank,
                matched_techniques=(
                    contributions_for(context, group_id)
                    if contributions is None
                    else contributions.get(group_id, ())
                ),
                unmatched_observed=tuple(sorted(observed - matched)),
                signal_ids=(signals or {}).get(group_id, ()),
            )
        )
    return ClassifierResult(classifier, True, tuple(entries), len(context.observed_techniques))


def rerank(result: ClassifierResult) -> ClassifierResult:
    ordered = sorted(result.entries, key=lambda entry: (-entry.score, entry.group_id))
    reranked = []
    previous_score: float | None = None
    dense_rank = 0
    for entry in ordered:
        if previous_score is None or entry.score != previous_score:
            dense_rank += 1
            previous_score = entry.score
        reranked.append(replace(entry, rank=dense_rank))
    return replace(result, entries=tuple(reranked))


def safe_log(value: float) -> float:
    return log(max(value, 1e-12))
