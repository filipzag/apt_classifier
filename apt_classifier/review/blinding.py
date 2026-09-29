import re
from dataclasses import dataclass
from typing import Any

from apt_classifier.review.context import ReviewPayload, serialize


@dataclass(frozen=True)
class BlindingMap:
    group_to_label: dict[str, str]
    label_to_group: dict[str, str]


def create_blinding_map(group_ids: list[str]) -> BlindingMap:
    group_to_label = {
        group_id: f"Candidate {_label(index)}" for index, group_id in enumerate(group_ids)
    }
    return BlindingMap(group_to_label, {label: group for group, label in group_to_label.items()})


def blind_payload(payload: ReviewPayload, kb: Any, mapping: BlindingMap) -> ReviewPayload:
    sensitive = _sensitive_terms(kb, set(mapping.group_to_label))
    replacements = {**{term: "[REDACTED]" for term in sensitive}, **mapping.group_to_label}
    raw = payload.to_dict()
    reduced = {
        "evidence": raw["evidence"],
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "known_techniques": candidate.get("known_techniques", []),
                "prominence": None,
            }
            for candidate in raw["candidates"]
        ],
        "classifier_results": raw["classifier_results"],
        "fusion": raw["fusion"],
        "reference": {"techniques": raw["reference"].get("techniques", {})},
    }
    transformed = _transform(reduced, _redaction_rules(replacements))
    candidates = transformed["candidates"]
    reference = transformed["reference"]
    return ReviewPayload(
        evidence=transformed["evidence"],
        candidates=tuple(candidates),
        classifier_results=tuple(transformed["classifier_results"]),
        fusion=transformed["fusion"],
        reference=reference,
    )


@dataclass(frozen=True)
class RedactionRules:
    exact: dict[str, str]
    folded: dict[str, str]
    pattern: re.Pattern[str] | None


def redact_text(text: str, replacements: dict[str, str]) -> str:
    return _redact(text, _redaction_rules(replacements))


def _redaction_rules(replacements: dict[str, str]) -> RedactionRules:
    exact = {source.casefold(): target for source, target in replacements.items()}
    folded = {source: target for source, target in exact.items() if len(source.strip()) >= 3}
    pattern = (
        re.compile(
            "|".join(re.escape(source) for source in sorted(folded, key=len, reverse=True)),
            flags=re.IGNORECASE,
        )
        if folded
        else None
    )
    return RedactionRules(exact, folded, pattern)


def _redact(text: str, rules: RedactionRules) -> str:
    if text.casefold() in rules.exact:
        return rules.exact[text.casefold()]
    if rules.pattern is None:
        return text
    return rules.pattern.sub(lambda match: rules.folded[match.group().casefold()], text)


def _transform(value: Any, rules: RedactionRules) -> Any:
    if isinstance(value, str):
        return _redact(value, rules)
    if isinstance(value, dict):
        return {_transform(str(key), rules): _transform(item, rules) for key, item in value.items()}
    if isinstance(value, list):
        return [_transform(item, rules) for item in value]
    if isinstance(value, tuple):
        return tuple(_transform(item, rules) for item in value)
    return value


def _sensitive_terms(kb: Any, candidate_ids: set[str]) -> set[str]:
    terms = set(candidate_ids)
    groups = getattr(kb, "groups", {})
    software = getattr(kb, "software", {})
    campaigns = getattr(kb, "campaigns", {})
    group_software = getattr(kb, "group_software", {})
    campaign_groups = getattr(kb, "campaign_groups", {})
    for group_id in candidate_ids:
        terms.update(_names(groups.get(group_id)))
        for software_id in group_software.get(group_id, ()):
            terms.add(str(software_id))
            terms.update(_names(software.get(software_id)))
    for campaign_id, linked_groups in campaign_groups.items():
        if candidate_ids & set(linked_groups):
            terms.add(str(campaign_id))
            terms.update(_names(campaigns.get(campaign_id)))
    return {term for term in terms if isinstance(term, str) and len(term.strip()) >= 3}


def _names(value: Any) -> set[str]:
    raw = serialize(value)
    if not isinstance(raw, dict):
        return set()
    output = set()
    for key in ("name", "id", "external_id"):
        if raw.get(key):
            output.add(str(raw[key]))
    for key in ("aliases", "external_references"):
        for item in raw.get(key, []) or []:
            if isinstance(item, str):
                output.add(item)
            elif isinstance(item, dict):
                for field in ("external_id", "url", "source_name", "description"):
                    if item.get(field):
                        output.add(str(item[field]))
    return output


def _label(index: int) -> str:
    label = ""
    current = index
    while True:
        current, remainder = divmod(current, 26)
        label = chr(65 + remainder) + label
        if current == 0:
            return label
        current -= 1
