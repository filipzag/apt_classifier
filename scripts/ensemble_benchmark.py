"""Held-out group self-consistency ensemble benchmark.

For each ATT&CK group we hide a fraction of its documented techniques, present
the remaining techniques as observations, run the full classifier ensemble, and
measure whether the originating group is recovered. This is a self-consistency
check (does the ensemble reproduce the documented tradecraft table?) and is NOT
real-world attribution accuracy.

Superseded as the attribution gate by ``scripts/attribution_benchmark.py``, which feeds
anonymized procedure *text* through the real pipeline (retrieval, acceptance, eligibility,
family weights, stage decision) with per-fold leak control and an unknown-actor arm. This
script assumes perfect technique mapping, runs only three voting classifiers with a
campaign weight (0.15) that differs from production (0.00), and lets about 19% of a
group's held-out techniques re-enter through campaign overlap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.classifiers.base import ClassifierContext
from apt_classifier.classifiers.campaign import CampaignClassifier
from apt_classifier.classifiers.group_text import GroupTextClassifier
from apt_classifier.classifiers.idf_overlap import IDFOverlapClassifier
from apt_classifier.classifiers.jaccard import WeightedJaccardClassifier
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier
from apt_classifier.classifiers.semantic_centroid import SemanticCentroidClassifier
from apt_classifier.classifiers.software import SoftwareClassifier
from apt_classifier.fusion import reciprocal_rank_fusion

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "data" / "kb" / "attack_kb.json"

CLASSIFIERS = (
    BernoulliNaiveBayesClassifier(),
    IDFOverlapClassifier(),
    WeightedJaccardClassifier(),
    SemanticCentroidClassifier(),
    GroupTextClassifier(),
    CampaignClassifier(),
    SoftwareClassifier(),
)


def evaluate_group(
    kb: AttackKnowledgeBase,
    group_id: str,
    observed_techniques: set[str],
    *,
    include_campaign: bool,
) -> dict[str, int]:
    observed = {tid: 1.0 for tid in observed_techniques}
    technique_observations = {tid: ("synthetic",) for tid in observed_techniques}
    context = ClassifierContext(
        kb=kb,
        observed_techniques=observed,
        technique_observations=technique_observations,
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
        narrative="",
    )
    results = tuple(classifier.classify(context) for classifier in CLASSIFIERS)
    weights = {} if include_campaign else {"campaign_profile": 0.0}
    if include_campaign:
        weights = {"campaign_profile": 0.15}
    fusion = reciprocal_rank_fusion(
        results,
        weights,
        classifier_families={
            "bernoulli_naive_bayes": "technique_overlap",
            "idf_tfidf_cosine": "technique_overlap",
            "weighted_jaccard": "technique_overlap",
            "semantic_centroid": "technique_overlap",
            "specific_behavior": "actor_behavior",
            "software_tools": "software",
            "campaign_profile": "campaign",
        },
        family_weights={
            "technique_overlap": 1.0,
            "actor_behavior": 1.5,
            "software": 1.2,
            "campaign": 1.0,
        },
    )
    return {entry.group_id: entry.rank for entry in fusion.entries}


def run(coverage: float, seed: int, include_campaign: bool) -> dict[str, Any]:
    kb = AttackKnowledgeBase.load(KB_PATH)
    rng = random.Random(seed)
    eligible = [group_id for group_id, group in kb.groups.items() if len(group.technique_ids) >= 5]
    per_classifier_top1: dict[str, int] = defaultdict(int)
    per_classifier_top5: dict[str, int] = defaultdict(int)
    rrf_top1 = 0
    rrf_top5 = 0
    reciprocal_ranks = []
    total = 0
    for group_id in eligible:
        techniques = sorted(kb.groups[group_id].technique_ids)
        rng.shuffle(techniques)
        observed_count = max(1, int(round(coverage * len(techniques))))
        observed = set(techniques[:observed_count])
        ranks = evaluate_group(kb, group_id, observed, include_campaign=include_campaign)
        if group_id not in ranks:
            continue
        total += 1
        rank = ranks[group_id]
        if rank == 1:
            rrf_top1 += 1
        if rank <= 5:
            rrf_top5 += 1
        reciprocal_ranks.append(1.0 / rank)
        for classifier in CLASSIFIERS:
            cls_result = classifier.classify(_context_for(kb, observed))
            if not cls_result.applicable:
                continue
            entry = next((e for e in cls_result.entries if e.group_id == group_id), None)
            if entry is None:
                continue
            if entry.rank == 1:
                per_classifier_top1[classifier.name] += 1
            if entry.rank <= 5:
                per_classifier_top5[classifier.name] += 1
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return {
        "coverage": coverage,
        "include_campaign": include_campaign,
        "eligible_groups": len(eligible),
        "evaluated": total,
        "rrf_top1": rrf_top1,
        "rrf_top5": rrf_top5,
        "rrf_mrr": round(mrr, 4),
        "per_classifier_top1": dict(per_classifier_top1),
        "per_classifier_top5": dict(per_classifier_top5),
    }


def _context_for(kb: AttackKnowledgeBase, observed: set[str]) -> ClassifierContext:
    return ClassifierContext(
        kb=kb,
        observed_techniques={tid: 1.0 for tid in observed},
        technique_observations={tid: ("synthetic",) for tid in observed},
        negative_techniques=set(),
        semantic_profile_scores={},
        group_text_scores={},
        software_ids=set(),
        narrative="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    with_campaign = run(args.coverage, args.seed, include_campaign=True)
    without_campaign = run(args.coverage, args.seed, include_campaign=False)
    report = {"with_campaign": with_campaign, "without_campaign": without_campaign}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, result in report.items():
            print(f"\n=== {label} ===")
            print(f"  eligible groups: {result['eligible_groups']}")
            print(f"  evaluated:        {result['evaluated']}")
            print(f"  RRF top-1:        {result['rrf_top1']}/{result['evaluated']}")
            print(f"  RRF top-5:        {result['rrf_top5']}/{result['evaluated']}")
            print(f"  RRF MRR:          {result['rrf_mrr']}")
            print("  per-classifier top-1:")
            for name, value in sorted(result["per_classifier_top1"].items()):
                print(f"    {name:24s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
