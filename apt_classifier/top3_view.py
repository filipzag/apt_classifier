"""The analyst's Top-three view: what separates the first three candidates (display only).

Built from what the decision recorded -- the naming gate's discrimination (separation.py), the
learned combiner's probabilities and RRF ranks and the stage decision -- plus ATT&CK's own
documentation of each group (origin, threat type and motivation, labelled as documentation,
never as inference). Nothing here re-scores the case or feeds the ranking; the engine never
imports this module. The state thresholds below are display labels (docs 10.10), not gates.

When the combiner's leader holds clearly less exclusive evidence than another top-three member,
the view says the evidence pulls both ways rather than which group is right: on the fresh fold
seeds 41 and 53 the leader was right in 1 of 8 such cases and the exclusive-evidence favourite in
2 (docs 21.3.13), so the leader is unreliable there but moving the favourite up does not help.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from apt_classifier.attack.related import related_to
from apt_classifier.classifiers.base import kb_mapping

# The motivation classifier's own vocabulary and corpus rules, so "ATT&CK documents: espionage"
# means exactly what that classifier would match in the group's description and campaigns.
from apt_classifier.classifiers.strategic_context import (
    _ALIASES as _STRATEGIC_ALIASES,
)
from apt_classifier.classifiers.strategic_context import (
    _contains,
    _field_corpus,
)
from apt_classifier.classifiers.strategic_context import (
    _profile_texts as _strategic_profile_texts,
)
from apt_classifier.origin import documented_origins, documented_threat_type
from apt_classifier.separation import distinguishing_techniques
from apt_classifier.top3 import PREREGISTERED_MARGIN, PREREGISTERED_TOLERANCE, decide_promotion

CHANNELS = ("technique", "behavior", "software", "soft", "negative", "total")
# largest top-three discriminative mass at or below this: too little exclusive evidence
THIN_EVIDENCE = 1.5
# documented-overlap pairs whose profiles nest or whose evidence does not separate them
OVERLAP_CONTAINMENT = 0.8
OVERLAP_SEPARATION = 0.5
# a profile in the largest tenth of ATT&CK explains more techniques by default
BROAD_PROFILE_QUANTILE = 0.9
ONE_SIDED_LIMIT = 8

STATE_LABELS = {
    "named": "Named",
    "overlap": "Documented overlap",
    "split": "Evidence pulls both ways",
    "thin": "Too little exclusive evidence",
    "not_separated": "Leader not separated",
    "empty": "No shortlist",
}


def pair_separation(own: float, other: float, floor: float) -> float:
    """Candidate-relative separation of one group over another, as the gate computes it."""
    denominator = max(own, other, floor)
    return max(-1.0, min(1.0, (own - other) / denominator)) if denominator > 0 else 0.0


def documented_motivations(kb: Any, group_id: str) -> list[str]:
    """Motivations ATT&CK's description and linked campaigns state for the group."""
    texts = _strategic_profile_texts(kb, group_id)
    if not texts:
        return []
    corpus = _field_corpus("motivations", texts)
    return [
        canonical
        for canonical, aliases in _STRATEGIC_ALIASES["motivations"].items()
        if any(_contains(corpus, alias) for alias in (canonical, *aliases))
    ]


def _name(kb: Any, group_id: str) -> str:
    group = getattr(kb, "groups", {}).get(group_id)
    return str(getattr(group, "name", group_id)) if group else group_id


def build_top3_view(
    kb: Any,
    state: Any,
    *,
    floor: float = 1.0,
    separation_bar: float = 0.5,
    min_discriminative: float = 3.0,
    behavior_threshold: float = 0.55,
) -> dict[str, Any]:
    """Head-to-head view of the decision's first three candidates and a plain-language state."""
    fusion = state.deterministic_fusion or state.fusion
    context = state.context
    decision = getattr(state, "attribution_decision", None)
    entries = list(fusion.entries[:3])
    thresholds = {
        "separation": separation_bar,
        "min_discriminative": min_discriminative,
        "clear_lead": PREREGISTERED_MARGIN,
        "tolerance": PREREGISTERED_TOLERANCE,
        "thin_evidence": THIN_EVIDENCE,
    }
    if not entries:
        return {
            "state": "empty",
            "label": STATE_LABELS["empty"],
            "summary": "No candidate groups yet.",
            "candidates": [],
            "pairs": [],
            "thresholds": thresholds,
        }
    ids = [entry.group_id for entry in entries]
    names = {gid: _name(kb, gid) for gid in ids}
    discrimination = getattr(context, "discrimination", None)
    scored = discrimination.by_group() if discrimination is not None else {}
    combiner = context.combiner if isinstance(getattr(context, "combiner", None), dict) else {}
    probabilities = combiner.get("probabilities") or {}
    rrf_ranks = combiner.get("rrf_ranks") or {}
    group_techniques = kb_mapping(kb, "group_techniques")
    group_software = kb_mapping(kb, "group_software")
    sizes = sorted(len(set(values)) for values in group_techniques.values())
    broad_size = sizes[int(BROAD_PROFILE_QUANTILE * (len(sizes) - 1))] if sizes else 0
    software_ids = set(getattr(context, "software_ids", ()) or ())
    software = getattr(kb, "software", {}) or {}
    techniques = getattr(kb, "techniques", {}) or {}

    def mass(group_id: str) -> float:
        item = scored.get(group_id)
        return float(item.total) if item is not None else 0.0

    fused_rank = {entry.group_id: int(entry.rank) for entry in fusion.entries}
    candidates = []
    for entry in entries:
        gid = entry.group_id
        item = scored.get(gid)
        profile = set(group_techniques.get(gid, ()))
        candidates.append(
            {
                "group_id": gid,
                "name": names[gid],
                "rank": int(entry.rank),
                "relative_score": round(float(entry.relative_score), 1),
                "channels": {key: round(float(getattr(item, key)), 4) for key in CHANNELS}
                if item is not None
                else None,
                "combiner_probability": probabilities.get(gid),
                "rrf_rank": rrf_ranks.get(gid),
                "profile_size": len(profile),
                "broad_profile": bool(broad_size) and len(profile) >= broad_size,
                "named_software": [
                    {"id": sid, "name": getattr(software.get(sid), "name", sid)}
                    for sid in sorted(software_ids & set(group_software.get(gid, ()) or ()))
                ],
                # groups ATT&CK documents as overlapping with this one, wherever they rank
                "related": sorted(
                    (
                        {
                            "group_id": other,
                            "name": _name(kb, other),
                            "reason": reason,
                            "rank": fused_rank.get(other),
                        }
                        for other, reason in related_to(kb, gid).items()
                    ),
                    key=lambda item: (item["rank"] is None, item["rank"] or 0, item["group_id"]),
                ),
                "documented": {
                    "origins": [
                        {"country": country, "sentence": sentence[:300]}
                        for country, sentence in sorted(documented_origins(kb, gid).items())
                    ],
                    **documented_threat_type(kb, gid),
                    "motivations": documented_motivations(kb, gid),
                },
            }
        )

    # documented-by among exactly these three (hierarchy-aware, accepted techniques only)
    rows = distinguishing_techniques(
        kb, replace(fusion, entries=tuple(entries)), context, top_k=len(entries)
    )
    best_behavior: dict[str, dict[str, float]] = {}
    for match in getattr(context, "specific_behavior_matches", ()) or ():
        gid = str(getattr(match, "group_id", ""))
        if gid not in names:
            continue
        per_observation = best_behavior.setdefault(str(match.observation_id), {})
        per_observation[gid] = max(per_observation.get(gid, 0.0), float(match.score))
    observation_text = {item.id: item.text for item in state.ledger.observations}

    def one_sided(own: str, other: str) -> dict[str, Any]:
        only = [row for row in rows if own in row.documented_by and other not in row.documented_by]
        only.sort(key=lambda row: (-row.specificity * row.confidence, row.technique_id))
        own_software = set(group_software.get(own, ()) or ())
        other_software = set(group_software.get(other, ()) or ())
        wins = [
            observation
            for observation, scores in sorted(best_behavior.items())
            if scores.get(own, 0.0) > max(scores.get(other, 0.0), behavior_threshold)
        ]
        return {
            "techniques": [
                {
                    "technique_id": row.technique_id,
                    "name": getattr(techniques.get(row.technique_id), "name", "") or "",
                    "confidence": round(row.confidence, 3),
                    "specificity": round(row.specificity, 3),
                }
                for row in only[:ONE_SIDED_LIMIT]
            ],
            "technique_count": len(only),
            "software": [
                {"id": sid, "name": getattr(software.get(sid), "name", sid)}
                for sid in sorted(software_ids & own_software - other_software)
            ],
            "behavior_wins": [
                {"observation_id": oid, "text": str(observation_text.get(oid, oid))[:160]}
                for oid in wins[:3]
            ],
            "behavior_win_count": len(wins),
        }

    pairs = []
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            separation = pair_separation(mass(first), mass(second), floor)
            reason = related_to(kb, first).get(second)
            profile_a = set(group_techniques.get(first, ()))
            profile_b = set(group_techniques.get(second, ()))
            smaller = min(len(profile_a), len(profile_b))
            containment = len(profile_a & profile_b) / smaller if smaller else 0.0
            pairs.append(
                {
                    "a": first,
                    "b": second,
                    "a_name": names[first],
                    "b_name": names[second],
                    "separation": round(separation, 4),
                    "favours": first if separation > 0 else second if separation < 0 else None,
                    "related_reason": reason,
                    "containment": round(containment, 3),
                    "overlap_cluster": bool(reason)
                    and (
                        containment >= OVERLAP_CONTAINMENT or abs(separation) < OVERLAP_SEPARATION
                    ),
                    "a_only": one_sided(first, second),
                    "b_only": one_sided(second, first),
                }
            )

    leader = ids[0]
    named = decision is not None and decision.stage.value == "known_group_hypothesis"
    # does another top-three member hold clearly more exclusive evidence than the leader?
    contest = decide_promotion(ids, discrimination)
    overlap = next(
        (pair for pair in pairs if pair["a"] == leader and pair["overlap_cluster"]), None
    )
    largest = max((mass(gid) for gid in ids), default=0.0)
    runner_up = getattr(decision, "runner_up_group_id", None) if decision is not None else None
    runner_name = _name(kb, runner_up) if runner_up else "the alternative"
    if runner_up and runner_up not in names:
        runner_name += " (outside the top three)"
    separation = float(getattr(decision, "separation", 0.0) or 0.0) if decision else 0.0
    if named:
        state_key = "named"
        summary = (
            f"{names[leader]} is named: its exclusive evidence clears the naming bar against "
            f"{runner_name} (separation {separation:.2f})."
        )
    elif overlap is not None:
        state_key = "overlap"
        summary = (
            f"ATT&CK documents an overlap between {names[leader]} and {overlap['b_name']} "
            f"({overlap['related_reason']}). Treat them as one cluster: the evidence here does "
            "not separate them reliably."
        )
    elif contest is not None:
        state_key = "split"
        rival = names[contest.promoted]
        summary = (
            f"The classifiers together put {names[leader]} first, but {rival} holds clearly more "
            f"evidence that only it explains (+{contest.gap:.2f})"
            + (
                f", while the {contest.advantage_channel} evidence favours {names[leader]}"
                if contest.blocked
                else ""
            )
            + ". On held-out test folds the leader was rarely right in this situation and the "
            "other group only somewhat more often: treat both as open and collect evidence that "
            "only one of them documents."
        )
    elif largest <= THIN_EVIDENCE:
        state_key = "thin"
        summary = (
            "Too little exclusive evidence to separate the three (largest discriminative mass "
            f"{largest:.2f}). Collect more specific evidence: techniques or software only one "
            "of them documents."
        )
    else:
        state_key = "not_separated"
        summary = (
            f"{names[leader]} leads, but its exclusive evidence does not clear the naming bar "
            f"against {runner_name} (separation {separation:.2f} < {separation_bar:.2f})."
        )
    if state_key == "overlap" and overlap is not None:
        other = overlap["b"]
    elif state_key == "split" and contest is not None:
        other = contest.promoted
    else:
        other = runner_up
    return {
        "state": state_key,
        "label": STATE_LABELS[state_key],
        "summary": summary,
        "leader": {"group_id": leader, "name": names[leader]},
        "other": {"group_id": other, "name": _name(kb, other)} if other else None,
        "contest": contest.to_dict() if contest is not None else None,
        "candidates": candidates,
        "pairs": pairs,
        # the channel weights the gate scored the mass with (for the stacked bars)
        "weights": {
            key: float(getattr(discrimination.weights, key))
            for key in ("technique", "behavior", "software", "soft", "negative")
        }
        if discrimination is not None
        else None,
        "display_order_differs": [e.group_id for e in state.fusion.entries[:3]] != ids,
        "combiner_note": (
            "Learned combiner probability (fitted on ATT&CK folds): how often a leader with these "
            "features was right there. Not a probability that the true group is one of these "
            "three."
        ),
        "thresholds": thresholds,
    }


def top3_summary(view: dict[str, Any]) -> dict[str, Any]:
    """The light part of the view carried in every case payload (Assessment badge), with the
    calibrated confidence label the service adds (confidence_labels.py; None when off)."""
    return {
        key: view.get(key) for key in ("state", "label", "summary", "leader", "other", "confidence")
    }
