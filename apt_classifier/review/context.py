from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from apt_classifier.classifiers.base import ClassifierResult
from apt_classifier.fusion import FusionResult


@dataclass(frozen=True)
class ReviewPayload:
    evidence: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    classifier_results: tuple[dict[str, Any], ...]
    fusion: dict[str, Any]
    reference: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": serialize(self.evidence),
            "candidates": serialize(self.candidates),
            "classifier_results": serialize(self.classifier_results),
            "fusion": serialize(self.fusion),
            "reference": serialize(self.reference),
        }


def build_review_payload(
    kb: Any,
    evidence_ledger: Any,
    classifier_results: tuple[ClassifierResult, ...],
    fusion: FusionResult,
    *,
    top_n: int = 10,
) -> ReviewPayload:
    candidate_ids = [entry.group_id for entry in fusion.entries[:top_n]]
    groups = getattr(kb, "groups", {})
    techniques = getattr(kb, "techniques", {})
    software = getattr(kb, "software", {})
    campaigns = getattr(kb, "campaigns", {})
    group_techniques = getattr(kb, "group_techniques", {})
    group_software = getattr(kb, "group_software", {})
    campaign_groups = getattr(kb, "campaign_groups", {})
    ledger_raw = serialize(evidence_ledger)
    observed_technique_ids = (
        {
            str(item.get("technique_id"))
            for item in ledger_raw.get("technique_matches", [])
            if isinstance(item, dict) and item.get("technique_id")
        }
        if isinstance(ledger_raw, dict)
        else set()
    )
    for result in classifier_results:
        for entry in result.entries:
            observed_technique_ids.update(
                contribution.technique_id for contribution in entry.matched_techniques
            )
            observed_technique_ids.update(entry.unmatched_observed)
    campaigns_by_group: dict[str, list[str]] = {group_id: [] for group_id in candidate_ids}
    for campaign_id, linked_groups in campaign_groups.items():
        for group_id in set(linked_groups):
            if group_id in campaigns_by_group:
                campaigns_by_group[group_id].append(campaign_id)
    candidates = []
    for group_id in candidate_ids:
        profile = _minimal_group(groups.get(group_id))
        # Compute documented specificity for matched techniques
        matched_techniques = sorted(
            observed_technique_ids & set(group_techniques.get(group_id, ()))
        )
        specificity_scores = {}
        for technique_id in matched_techniques[:20]:
            try:
                from apt_classifier.evidence.quality import compute_technique_specificity

                score = compute_technique_specificity(technique_id, kb)
                specificity_scores[technique_id] = {
                    "group_count": score.group_count,
                    "total_groups": score.total_groups,
                    "specificity": round(score.specificity, 3),
                    "is_group_specific": score.is_group_specific,
                    "note": score.note,
                }
            except Exception:
                pass
        candidates.append(
            {
                "candidate_id": group_id,
                "profile": profile,
                "known_techniques": sorted(group_techniques.get(group_id, ())),
                "matched_observed_techniques": matched_techniques,
                "unmatched_observed_techniques": sorted(
                    observed_technique_ids - set(group_techniques.get(group_id, ()))
                ),
                "known_software": sorted(group_software.get(group_id, ())),
                "campaigns": sorted(campaigns_by_group.get(group_id, ())),
                "prominence": float(getattr(kb, "prominence", {}).get(group_id, 0.0)),
                "technique_specificity": specificity_scores,
            }
        )
    compact_results = []
    referenced_techniques: set[str] = set()
    referenced_software: set[str] = set()
    referenced_campaigns: set[str] = set()
    referenced_behaviors: set[str] = set()
    for result in classifier_results:
        if not result.applicable:
            compact_results.append(
                {"classifier": result.classifier, "applicable": False, "entries": []}
            )
            continue
        selected = [entry for entry in result.entries if entry.group_id in candidate_ids]
        compact_results.append(
            {
                "classifier": result.classifier,
                "applicable": True,
                "entries": [_compact_classifier_entry(entry) for entry in selected],
            }
        )
        for entry in selected:
            referenced_techniques.update(
                contribution.technique_id for contribution in entry.matched_techniques
            )
            referenced_techniques.update(entry.unmatched_observed)
            referenced_software.update(signal for signal in entry.signal_ids if signal in software)
            referenced_campaigns.update(
                signal for signal in entry.signal_ids if signal in campaigns
            )
            referenced_behaviors.update(
                signal for signal in entry.signal_ids if ":actor-behavior:" in signal
            )
    if isinstance(ledger_raw, dict):
        referenced_techniques.update(
            str(item.get("technique_id"))
            for item in ledger_raw.get("technique_matches", [])
            if isinstance(item, dict) and item.get("technique_id")
        )
        referenced_software.update(
            str(item.get("metadata", {}).get("software_id"))
            for item in ledger_raw.get("observations", [])
            if isinstance(item, dict)
            and isinstance(item.get("metadata"), dict)
            and item["metadata"].get("software_id")
        )
    referenced_techniques = {item for item in referenced_techniques if item in techniques}
    referenced_software = {item for item in referenced_software if item in software}
    referenced_campaigns = {item for item in referenced_campaigns if item in campaigns}
    reference = {
        "techniques": {
            technique_id: _minimal_technique(techniques.get(technique_id))
            for technique_id in sorted(referenced_techniques)
        },
        "software": {
            software_id: _minimal_software(software.get(software_id))
            for software_id in sorted(referenced_software)
        },
        "campaigns": {
            campaign_id: _minimal_campaign(campaigns.get(campaign_id))
            for campaign_id in sorted(referenced_campaigns)
        },
        "actor_behaviors": {
            signal_id: behavior
            for signal_id in sorted(referenced_behaviors)
            if (behavior := _actor_behavior_reference(groups, signal_id)) is not None
        },
    }
    compact_fusion = fusion.to_dict()
    compact_fusion["entries"] = [
        entry for entry in compact_fusion["entries"] if entry["group_id"] in candidate_ids
    ]
    return ReviewPayload(
        evidence=_neutral_evidence(evidence_ledger),
        candidates=tuple(candidates),
        classifier_results=tuple(compact_results),
        fusion=compact_fusion,
        reference=reference,
    )


def _compact_classifier_entry(entry: Any) -> dict[str, Any]:
    return {
        "group_id": entry.group_id,
        "score": entry.score,
        "rank": entry.rank,
        "matched_techniques": [
            contribution.technique_id for contribution in entry.matched_techniques
        ],
        "signal_ids": list(entry.signal_ids),
    }


def serialize(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {key: serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): serialize(item) for key, item in vars(value).items()}
    return str(value)


def _neutral_evidence(ledger: Any) -> dict[str, Any]:
    raw = serialize(ledger)
    if not isinstance(raw, dict):
        return {"observations": raw}
    observations = []
    for item in raw.get("observations", []):
        if not isinstance(item, dict):
            continue
        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        obs_entry = {
            "observation_id": item.get("observation_id"),
            "text": item.get("text"),
            "status": item.get("status"),
            "kind": item.get("kind"),
            "provenance": {
                key: provenance.get(key)
                for key in (
                    "source",
                    "source_name",
                    "line_numbers",
                    "occurrence_count",
                    "event_id",
                )
                if provenance.get(key) not in (None, "", [])
            },
            "metadata": {
                key: metadata.get(key)
                for key in ("software_id", "ioc_type", "platform")
                if metadata.get(key) not in (None, "", [])
            },
        }
        # Include evidence quality ratings from provenance metadata
        prov_meta = (
            provenance.get("metadata", {}) if isinstance(provenance.get("metadata"), dict) else {}
        )
        quality_fields = {
            "reliability": prov_meta.get("reliability"),
            "credibility": prov_meta.get("credibility"),
            "independence_key": prov_meta.get("independence_key"),
            "rating_rationale": prov_meta.get("rating_rationale"),
        }
        quality_entry = {k: v for k, v in quality_fields.items() if v is not None}
        if quality_entry:
            obs_entry["evidence_quality"] = quality_entry
        observations.append(obs_entry)
    technique_matches = _compact_technique_matches(raw.get("technique_matches", []))
    answers = []
    for item in raw.get("question_answers", []):
        if isinstance(item, dict):
            answers.append(
                {
                    key: item.get(key)
                    for key in (
                        "question_id",
                        "question",
                        "status",
                        "technique_id",
                        "software_id",
                        "answer_text",
                    )
                    if item.get(key) not in (None, "", [])
                }
            )
    return {
        "case_id": raw.get("case_id"),
        "observations": observations,
        "technique_matches": technique_matches,
        "question_answers": answers,
        "strategic_context": raw.get("strategic_context", {}),
    }


def _compact_technique_matches(items: Any) -> list[dict[str, Any]]:
    by_technique: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("technique_id"):
            continue
        technique_id = str(item["technique_id"])
        observation_ids = {
            str(value) for value in item.get("observation_ids", []) if str(value).strip()
        }
        current = by_technique.get(technique_id)
        if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
            by_technique[technique_id] = {
                "technique_id": technique_id,
                "technique_name": item.get("technique_name"),
                "score": item.get("score", 0.0),
                "observation_ids": sorted(observation_ids),
                "parent_technique_id": item.get("parent_technique_id"),
                "matched_text": _limited_text(item.get("matched_text"), 500),
            }
        elif current is not None:
            current["observation_ids"] = sorted(
                {*current.get("observation_ids", []), *observation_ids}
            )
    return sorted(
        by_technique.values(),
        key=lambda item: (-float(item.get("score", 0.0)), str(item["technique_id"])),
    )[:50]


def _actor_behavior_reference(groups: dict[str, Any], signal_id: str) -> dict[str, str] | None:
    try:
        group_id, remainder = signal_id.split(":actor-behavior:", 1)
        technique_id, raw_index = remainder.rsplit(":", 1)
        index = int(raw_index)
    except (ValueError, TypeError):
        return None
    group = groups.get(group_id)
    raw = serialize(group)
    if not isinstance(raw, dict):
        return None
    behaviors = raw.get("technique_behaviors")
    if not isinstance(behaviors, dict):
        return None
    sentences = behaviors.get(technique_id)
    if not isinstance(sentences, list) or not 0 <= index < len(sentences):
        return None
    return {
        "group_id": group_id,
        "technique_id": technique_id,
        "text": _limited_text(sentences[index], 1000),
    }


def _minimal_group(group: Any) -> dict[str, Any]:
    raw = serialize(group)
    if not isinstance(raw, dict):
        return {"value": raw}
    return {
        key: _limited_text(value, 5000) if key == "description" else value
        for key, value in raw.items()
        if key in {"id", "name", "aliases", "description"}
    }


def _minimal_software(item: Any) -> dict[str, Any]:
    raw = serialize(item)
    if not isinstance(raw, dict):
        return {"value": raw}
    allowed = {"id", "name", "aliases", "software_type", "platforms"}
    return {key: value for key, value in raw.items() if key in allowed}


def _minimal_campaign(campaign: Any) -> dict[str, Any]:
    raw = serialize(campaign)
    if not isinstance(raw, dict):
        return {"value": raw}
    return {
        key: _limited_text(value, 3000) if key == "description" else value
        for key, value in raw.items()
        if key in {"id", "name", "aliases", "description", "first_seen", "last_seen"}
    }


def _limited_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"


def _minimal_technique(technique: Any) -> dict[str, Any]:
    raw = serialize(technique)
    if not isinstance(raw, dict):
        return {"value": raw}
    allowed = {"id", "external_id", "name", "tactics", "platforms", "parent_id"}
    return {key: value for key, value in raw.items() if key in allowed}
