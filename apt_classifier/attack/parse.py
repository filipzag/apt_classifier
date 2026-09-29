import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apt_classifier.attack.kb import (
    Analytic,
    AttackKnowledgeBase,
    Campaign,
    DataComponent,
    DataSource,
    DetectionStrategy,
    Group,
    LogSource,
    Software,
    Technique,
)

_CITATION_RE = re.compile(
    r"\s*\((?:citation|citations):(?:[^()]|\([^()]*\))*\)|\s*\[(?:citation|citations):[^]]*]",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)]\([^)]+\)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'(\[])?[A-Z0-9])|\n+")
_ID_PREFIXES = {
    "attack-pattern": "T",
    "intrusion-set": "G",
    "malware": "S",
    "tool": "S",
    "campaign": "C",
    "x-mitre-detection-strategy": "DET",
    "x-mitre-analytic": "AN",
    "x-mitre-data-component": "DC",
    "x-mitre-data-source": "DS",
}


def strip_attack_citations(text: str) -> str:
    cleaned = _CITATION_RE.sub("", text or "")
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = strip_attack_citations(text)
    if not cleaned:
        return []
    sentences: list[str] = []
    for part in _SENTENCE_BOUNDARY_RE.split(cleaned):
        sentence = re.sub(r"^(?:[-*•]+|#{1,6})\s*", "", part).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def technique_sentence_fields(name: str, description: str, detection: str) -> dict[str, list[str]]:
    clean_name = strip_attack_citations(name)
    return {
        "name": [clean_name] if clean_name else [],
        "description": split_sentences(description),
        "detection": split_sentences(detection),
    }


def _active(item: Mapping[str, Any]) -> bool:
    return not bool(item.get("revoked")) and not bool(item.get("x_mitre_deprecated"))


def _attack_id(item: Mapping[str, Any]) -> str | None:
    references = item.get("external_references", [])
    if not isinstance(references, Sequence) or isinstance(references, str):
        return None
    for reference in references:
        if (
            isinstance(reference, Mapping)
            and reference.get("source_name") == "mitre-attack"
            and reference.get("external_id")
        ):
            return str(reference["external_id"])
    prefix = _ID_PREFIXES.get(str(item.get("type", "")))
    if prefix:
        for reference in references:
            if not isinstance(reference, Mapping):
                continue
            external_id = str(reference.get("external_id", ""))
            if external_id.startswith(prefix):
                return external_id
    return None


def _references(item: Mapping[str, Any]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    values = item.get("external_references", [])
    if not isinstance(values, Sequence) or isinstance(values, str):
        return references
    for value in values:
        if not isinstance(value, Mapping):
            continue
        reference = {
            key: str(value[key])
            for key in ("source_name", "external_id", "url", "description")
            if value.get(key) is not None
        }
        if reference:
            references.append(reference)
    return references


def _strings(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        items = [value] if isinstance(value, str) else value if isinstance(value, Sequence) else []
        for item in items:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
    return result


def _version(
    bundle: Mapping[str, Any], objects: list[Mapping[str, Any]], fallback: str | None
) -> str:
    direct = bundle.get("x_mitre_version") or bundle.get("version")
    if direct:
        return str(direct)
    for item in objects:
        if item.get("type") != "x-mitre-collection":
            continue
        value = item.get("x_mitre_version") or item.get("version")
        if value:
            return str(value)
        match = re.search(r"\bv(\d+(?:\.\d+)+)\b", str(item.get("name", "")), re.IGNORECASE)
        if match:
            return match.group(1)
    return fallback or "unknown"


def load_attack_bundle(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ATT&CK STIX bundle must be a JSON object")
    return data


def parse_attack_bundle(
    source: str | Path | Mapping[str, Any], *, version: str | None = None
) -> AttackKnowledgeBase:
    bundle = load_attack_bundle(source)
    raw_objects = bundle.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("ATT&CK STIX bundle has no objects list")
    objects = [item for item in raw_objects if isinstance(item, Mapping) and _active(item)]
    by_stix_id = {
        str(item["id"]): item
        for item in objects
        if item.get("id") and item.get("type") != "relationship"
    }
    stix_to_attack_id: dict[str, str] = {}
    techniques: dict[str, Technique] = {}
    groups: dict[str, Group] = {}
    software: dict[str, Software] = {}
    campaigns: dict[str, Campaign] = {}
    detection_strategies: dict[str, DetectionStrategy] = {}
    analytics: dict[str, Analytic] = {}
    data_components: dict[str, DataComponent] = {}
    data_sources: dict[str, DataSource] = {}

    # Pre-populate the STIX→ATT&CK ID map so that cross-references (e.g.
    # analytic → data-component) resolve regardless of iteration order.
    for item in by_stix_id.values():
        attack_id = _attack_id(item)
        if attack_id:
            stix_to_attack_id[str(item["id"])] = attack_id

    for item in by_stix_id.values():
        item_type = str(item.get("type", ""))
        attack_id = _attack_id(item)
        if not attack_id:
            continue
        stix_id = str(item["id"])
        stix_to_attack_id[stix_id] = attack_id
        name = strip_attack_citations(str(item.get("name", "")))
        description = strip_attack_citations(str(item.get("description", "")))
        references = _references(item)
        if item_type == "attack-pattern":
            detection = strip_attack_citations(str(item.get("x_mitre_detection", "")))
            techniques[attack_id] = Technique(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                detection=detection,
                platforms=sorted(_strings(item.get("x_mitre_platforms", []))),
                tactics=sorted(
                    {
                        str(phase.get("phase_name"))
                        for phase in item.get("kill_chain_phases", [])
                        if isinstance(phase, Mapping) and phase.get("phase_name")
                    }
                ),
                is_subtechnique=bool(item.get("x_mitre_is_subtechnique")),
                external_references=references,
                sentences=technique_sentence_fields(name, description, detection),
            )
        elif item_type == "intrusion-set":
            groups[attack_id] = Group(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                aliases=_strings(item.get("aliases", []), item.get("x_mitre_aliases", [])),
                external_references=references,
            )
        elif item_type in {"malware", "tool"}:
            software[attack_id] = Software(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                software_type=item_type,
                description=description,
                aliases=_strings(item.get("aliases", []), item.get("x_mitre_aliases", [])),
                platforms=sorted(_strings(item.get("x_mitre_platforms", []))),
                external_references=references,
            )
        elif item_type == "campaign":
            campaigns[attack_id] = Campaign(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                aliases=_strings(item.get("aliases", []), item.get("x_mitre_aliases", [])),
                first_seen=str(item["first_seen"]) if item.get("first_seen") else None,
                last_seen=str(item["last_seen"]) if item.get("last_seen") else None,
                external_references=references,
            )
        elif item_type == "x-mitre-detection-strategy":
            detection_strategies[attack_id] = DetectionStrategy(
                id=attack_id,
                name=name,
                stix_id=stix_id,
            )
        elif item_type == "x-mitre-analytic":
            log_source_refs: list[LogSource] = []
            data_component_ids: list[str] = []
            for ref in item.get("x_mitre_log_source_references", []):
                if not isinstance(ref, Mapping):
                    continue
                dc_stix = str(ref.get("x_mitre_data_component_ref", ""))
                dc_id = stix_to_attack_id.get(dc_stix, "")
                if dc_id:
                    data_component_ids.append(dc_id)
                log_source_refs.append(
                    LogSource(
                        name=str(ref.get("name", "")),
                        channel=str(ref.get("channel", "")),
                    )
                )
            analytics[attack_id] = Analytic(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                platforms=sorted(_strings(item.get("x_mitre_platforms", []))),
                log_source_refs=log_source_refs,
                data_component_ids=data_component_ids,
                mutable_elements=[
                    {
                        "field": str(me.get("field", "")),
                        "description": str(me.get("description", "")),
                    }
                    for me in item.get("x_mitre_mutable_elements", [])
                    if isinstance(me, Mapping)
                ],
            )
        elif item_type == "x-mitre-data-component":
            log_sources: list[LogSource] = []
            for ls in item.get("x_mitre_log_sources", []):
                if not isinstance(ls, Mapping):
                    continue
                log_sources.append(
                    LogSource(
                        name=str(ls.get("name", "")),
                        channel=str(ls.get("channel", "")),
                    )
                )
            data_components[attack_id] = DataComponent(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                log_sources=log_sources,
            )
        elif item_type == "x-mitre-data-source":
            data_sources[attack_id] = DataSource(
                id=attack_id,
                name=name,
                stix_id=stix_id,
                description=description,
                platforms=sorted(_strings(item.get("x_mitre_platforms", []))),
                collection_layers=sorted(_strings(item.get("x_mitre_collection_layers", []))),
            )

    group_techniques = {key: set() for key in groups}
    group_behaviors: dict[str, dict[str, list[str]]] = {key: {} for key in groups}
    group_software = {key: set() for key in groups}
    group_campaigns = {key: set() for key in groups}
    software_techniques = {key: set() for key in software}
    software_groups = {key: set() for key in software}
    campaign_techniques = {key: set() for key in campaigns}
    campaign_groups = {key: set() for key in campaigns}

    for relationship in objects:
        if relationship.get("type") != "relationship":
            continue
        source_stix = str(relationship.get("source_ref", ""))
        target_stix = str(relationship.get("target_ref", ""))
        if source_stix not in by_stix_id or target_stix not in by_stix_id:
            continue
        source_id = stix_to_attack_id.get(source_stix)
        target_id = stix_to_attack_id.get(target_stix)
        if not source_id or not target_id:
            continue
        source_type = str(by_stix_id[source_stix].get("type", ""))
        target_type = str(by_stix_id[target_stix].get("type", ""))
        relationship_type = relationship.get("relationship_type")
        if (
            relationship_type == "subtechnique-of"
            and source_id in techniques
            and target_id in techniques
        ):
            techniques[source_id].parent_id = target_id
        elif (
            relationship_type == "uses"
            and source_type == "intrusion-set"
            and target_type == "attack-pattern"
            and source_id in groups
            and target_id in techniques
        ):
            group_techniques[source_id].add(target_id)
            behavior_sentences = group_behaviors[source_id].setdefault(target_id, [])
            for sentence in split_sentences(str(relationship.get("description", ""))):
                if sentence not in behavior_sentences:
                    behavior_sentences.append(sentence)
        elif (
            relationship_type == "uses"
            and source_type == "intrusion-set"
            and target_type in {"malware", "tool"}
            and source_id in groups
            and target_id in software
        ):
            group_software[source_id].add(target_id)
            software_groups[target_id].add(source_id)
        elif (
            relationship_type == "uses"
            and source_type == "campaign"
            and target_type == "attack-pattern"
            and source_id in campaigns
            and target_id in techniques
        ):
            campaign_techniques[source_id].add(target_id)
        elif (
            relationship_type == "attributed-to"
            and source_type == "campaign"
            and target_type == "intrusion-set"
            and source_id in campaigns
            and target_id in groups
        ):
            campaign_groups[source_id].add(target_id)
            group_campaigns[target_id].add(source_id)
        elif (
            relationship_type == "uses"
            and source_type in {"malware", "tool"}
            and target_type == "attack-pattern"
            and source_id in software
            and target_id in techniques
        ):
            software_techniques[source_id].add(target_id)
        elif (
            relationship_type == "detects"
            and source_type == "x-mitre-detection-strategy"
            and target_type == "attack-pattern"
            and source_id in detection_strategies
            and target_id in techniques
        ):
            detection_strategies[source_id].technique_ids.append(target_id)

    for group_id, group in groups.items():
        group.technique_ids = sorted(group_techniques[group_id])
        group.technique_behaviors = {
            technique_id: group_behaviors[group_id][technique_id]
            for technique_id in sorted(group_behaviors[group_id])
            if group_behaviors[group_id][technique_id]
        }
        group.software_ids = sorted(group_software[group_id])
        group.campaign_ids = sorted(group_campaigns[group_id])
    for software_id, item in software.items():
        item.technique_ids = sorted(software_techniques[software_id])
        item.group_ids = sorted(software_groups[software_id])
    for campaign_id, campaign in campaigns.items():
        campaign.technique_ids = sorted(campaign_techniques[campaign_id])
        campaign.group_ids = sorted(campaign_groups[campaign_id])

    # Resolve detection-strategy → analytic references (STIX IDs → ATT&CK IDs).
    for ds in detection_strategies.values():
        raw_ds = by_stix_id.get(ds.stix_id)
        if raw_ds is None:
            continue
        for analytic_stix in raw_ds.get("x_mitre_analytic_refs", []):
            analytic_id = stix_to_attack_id.get(str(analytic_stix))
            if analytic_id and analytic_id in analytics:
                ds.analytic_ids.append(analytic_id)
        ds.technique_ids = sorted(set(ds.technique_ids))
        ds.analytic_ids = sorted(set(ds.analytic_ids))

    knowledge_base = AttackKnowledgeBase(
        version=_version(bundle, objects, version),
        techniques=dict(sorted(techniques.items())),
        groups=dict(sorted(groups.items())),
        software=dict(sorted(software.items())),
        campaigns=dict(sorted(campaigns.items())),
        detection_strategies=dict(sorted(detection_strategies.items())),
        analytics=dict(sorted(analytics.items())),
        data_components=dict(sorted(data_components.items())),
        data_sources=dict(sorted(data_sources.items())),
    )
    knowledge_base.recompute_statistics()
    return knowledge_base


strip_citations = strip_attack_citations
parse_bundle = parse_attack_bundle
