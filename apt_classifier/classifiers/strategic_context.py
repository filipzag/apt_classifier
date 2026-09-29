import re
from typing import Any

from apt_classifier.classifiers.base import (
    ClassifierContext,
    ClassifierResult,
    build_result,
    group_ids,
    group_techniques,
    kb_mapping,
)
from apt_classifier.evidence.models import StrategicContext

_FIELD_WEIGHTS = {
    "target_geographies": 0.8,
    "target_industries": 1.0,
    "objectives": 1.2,
    "suspected_origins": 0.6,
}

_ALIASES = {
    "target_geographies": {
        "united states": ("united states", "u s", "us", "usa", "american"),
        "united kingdom": ("united kingdom", "u k", "uk", "britain", "british"),
        "china": ("china", "chinese", "prc"),
        "russia": ("russia", "russian"),
        "iran": ("iran", "iranian"),
        "israel": ("israel", "israeli"),
        "ukraine": ("ukraine", "ukrainian"),
        "india": ("india", "indian"),
        "pakistan": ("pakistan", "pakistani"),
        "north korea": ("north korea", "north korean", "dprk"),
        "south korea": ("south korea", "south korean", "republic of korea"),
        "japan": ("japan", "japanese"),
        "taiwan": ("taiwan", "taiwanese"),
        "europe": ("europe", "european"),
        "middle east": ("middle east", "middle eastern"),
        "southeast asia": ("southeast asia", "south east asia"),
        "north america": ("north america", "north american"),
        "latin america": ("latin america", "latin american", "south america"),
        "africa": ("africa", "african"),
        "australia": ("australia", "australian"),
    },
    "target_industries": {
        "government": ("government", "public sector", "ministry", "ministries"),
        "defense": ("defense", "defence", "military", "armed forces"),
        "financial services": (
            "financial",
            "finance",
            "bank",
            "banking",
            "financial institution",
        ),
        "energy": ("energy", "oil", "gas", "petrochemical", "electric utility"),
        "healthcare": ("healthcare", "health care", "medical", "hospital"),
        "technology": (
            "technology",
            "software",
            "information technology",
            "it sector",
            "semiconductor",
        ),
        "telecommunications": ("telecommunications", "telecom", "telco"),
        "manufacturing": ("manufacturing", "manufacturer", "industrial"),
        "education": ("education", "university", "universities", "college", "academic"),
        "aerospace": ("aerospace", "aviation", "airline"),
        "transportation": (
            "transportation",
            "transport",
            "shipping",
            "maritime",
            "logistics",
        ),
        "critical infrastructure": ("critical infrastructure", "ics", "scada"),
        "media": ("media", "journalist", "news organization"),
        "nonprofit": ("nonprofit", "non profit", "ngo", "non governmental organization"),
        "retail": ("retail", "ecommerce", "e commerce"),
        "hospitality": ("hospitality", "hotel"),
        "legal": ("legal", "law firm"),
        "pharmaceutical": ("pharmaceutical", "pharma", "biotechnology", "biotech"),
    },
    "motivations": {
        "espionage": (
            "espionage",
            "intelligence collection",
            "intelligence gathering",
            "state sponsored",
        ),
        "financial gain": (
            "financial gain",
            "financially motivated",
            "profit",
            "monetize",
        ),
        "disruption": ("disruption", "disruptive", "sabotage", "destabilize"),
        "destruction": ("destruction", "destructive", "wiper", "wipe"),
        "hacktivism": ("hacktivism", "hacktivist", "ideological", "political protest"),
        "influence": ("influence operation", "information operation", "propaganda"),
        "intellectual property theft": (
            "intellectual property theft",
            "proprietary information",
            "trade secret",
        ),
    },
    "objectives": {
        "data theft": (
            "data theft",
            "steal data",
            "stolen data",
            "exfiltrate",
            "exfiltration",
            "sensitive information",
        ),
        "intellectual property theft": (
            "intellectual property",
            "proprietary information",
            "trade secret",
            "source code",
            "research and development",
            "blueprint",
        ),
        "credential theft": (
            "credential theft",
            "steal credentials",
            "credential harvesting",
            "password theft",
        ),
        "ransomware extortion": (
            "ransomware",
            "extortion",
            "ransom",
            "monetize access",
        ),
        "financial theft": ("financial theft", "steal funds", "bank fraud", "payment fraud"),
        "disruption": ("disrupt operations", "operational disruption", "denial of service"),
        "destruction": ("destroy", "destruction", "destructive", "wiper", "wipe systems"),
        "surveillance": ("surveillance", "monitor communications", "track individuals"),
        "persistent access": (
            "persistent access",
            "maintain access",
            "long term access",
            "foothold",
        ),
        "public disclosure": ("data leak", "leak data", "public disclosure", "doxing"),
    },
    "suspected_origins": {
        "china": ("china", "chinese", "china nexus", "prc"),
        "russia": ("russia", "russian", "russia nexus"),
        "iran": ("iran", "iranian", "iran nexus"),
        "north korea": ("north korea", "north korean", "dprk"),
        "united states": ("united states", "u s", "american"),
        "israel": ("israel", "israeli"),
        "pakistan": ("pakistan", "pakistani"),
        "india": ("india", "indian"),
        "vietnam": ("vietnam", "vietnamese"),
    },
}

_TARGET_CUES = (
    "against",
    "compromis",
    "target",
    "victim",
)
_ORIGIN_CUES = (
    "affiliat",
    "attribut",
    "based",
    "government",
    "linked",
    "nexus",
    "sponsor",
    "state",
)


class MotivationClassifier:
    name = "motivation_alignment"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        strategic = getattr(context, "strategic_context", StrategicContext())
        values = strategic.motivations
        if not values:
            return build_result(self.name, context, {}, applicable=False)
        eligible = _technically_supported_group_ids(context)
        if not eligible:
            return build_result(self.name, context, {}, applicable=False)
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for group_id in sorted(eligible):
            texts = _profile_texts(context.kb, group_id)
            if not texts:
                continue
            corpus = _field_corpus("motivations", texts)
            matched = []
            for value in values:
                canonical, aliases = _aliases_for("motivations", value)
                if _matches("motivations", texts, corpus, aliases):
                    matched.append(f"motivations:{canonical}")
            if matched:
                scores[group_id] = len(matched) / len(values)
                signals[group_id] = tuple(dict.fromkeys(matched))
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(scores),
            signals=signals,
        )


class StrategicContextClassifier:
    name = "strategic_context"

    def classify(self, context: ClassifierContext) -> ClassifierResult:
        strategic = getattr(context, "strategic_context", StrategicContext())
        supplied = {
            field_name: values
            for field_name, values in strategic.field_values().items()
            if field_name != "motivations"
        }
        if not any(supplied.values()):
            return build_result(self.name, context, {}, applicable=False)
        eligible = _technically_supported_group_ids(context)
        if not eligible:
            return build_result(self.name, context, {}, applicable=False)
        supplied_field_count = sum(bool(values) for values in supplied.values())
        required_matched_fields = 1 if supplied_field_count == 1 else 2
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for group_id in sorted(eligible):
            texts = _profile_texts(context.kb, group_id)
            if not texts:
                continue
            numerator = 0.0
            denominator = 0.0
            matched_fields = 0
            matched: list[str] = []
            for field_name, values in supplied.items():
                if not values:
                    continue
                weight = _FIELD_WEIGHTS[field_name]
                denominator += weight
                corpus = _field_corpus(field_name, texts)
                field_matches = 0
                for value in values:
                    canonical, aliases = _aliases_for(field_name, value)
                    if _matches(field_name, texts, corpus, aliases):
                        field_matches += 1
                        matched.append(f"{field_name}:{canonical}")
                if field_matches:
                    matched_fields += 1
                numerator += weight * field_matches / len(values)
            score = numerator / denominator if denominator else 0.0
            if score > 0 and matched_fields >= required_matched_fields:
                scores[group_id] = score
                signals[group_id] = tuple(dict.fromkeys(matched))
        return build_result(
            self.name,
            context,
            scores,
            applicable=bool(scores),
            signals=signals,
        )


def _technically_supported_group_ids(context: ClassifierContext) -> set[str]:
    observed = set(context.observed_techniques)
    group_software = kb_mapping(context.kb, "group_software")
    return {
        group_id
        for group_id in group_ids(context)
        if observed & group_techniques(context, group_id)
        or context.software_ids & set(group_software.get(group_id, ()))
    }


def _profile_texts(kb: Any, group_id: str) -> list[str]:
    groups = getattr(kb, "groups", {})
    campaigns = getattr(kb, "campaigns", {})
    group = groups.get(group_id)
    if group is None:
        return []
    texts = [_record_value(group, "description")]
    campaign_ids = set(_record_value(group, "campaign_ids", ()))
    for campaign_id, linked_groups in getattr(kb, "campaign_groups", {}).items():
        if group_id in set(linked_groups):
            campaign_ids.add(campaign_id)
    texts.extend(
        _record_value(campaigns.get(campaign_id), "description") for campaign_id in campaign_ids
    )
    return [str(text) for text in texts if str(text).strip()]


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _field_corpus(field_name: str, texts: list[str]) -> str:
    sentences = [
        sentence
        for text in texts
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if sentence.strip()
    ]
    if field_name in {"target_geographies", "target_industries"}:
        selected = []
        for sentence in sentences:
            normalized = _normalize(sentence)
            indexes = [normalized.find(cue) for cue in _TARGET_CUES if cue in normalized]
            if indexes:
                selected.append(normalized[min(indexes) :])
    elif field_name == "suspected_origins":
        selected = [sentence for sentence in sentences if _has_cue(sentence, _ORIGIN_CUES)]
    else:
        selected = sentences
    return _normalize(" ".join(selected))


def _has_cue(value: str, cues: tuple[str, ...]) -> bool:
    normalized = _normalize(value)
    return any(cue in normalized for cue in cues)


def _aliases_for(field_name: str, value: str) -> tuple[str, tuple[str, ...]]:
    normalized = _normalize(value)
    for canonical, raw_aliases in _ALIASES.get(field_name, {}).items():
        aliases = tuple(dict.fromkeys((_normalize(canonical), *map(_normalize, raw_aliases))))
        if any(_contains(normalized, alias) or _contains(alias, normalized) for alias in aliases):
            return canonical, aliases
    return normalized, (normalized,)


def _matches(field_name: str, texts: list[str], corpus: str, aliases: tuple[str, ...]) -> bool:
    if field_name != "suspected_origins":
        return any(_contains(corpus, alias) for alias in aliases)
    source = _normalize(" ".join(texts))
    for alias in aliases:
        patterns = (
            f"{alias} affiliated",
            f"{alias} based",
            f"{alias} government",
            f"{alias} linked",
            f"{alias} nexus",
            f"{alias} sponsor",
            f"{alias} state",
            f"affiliated with {alias}",
            f"attributed to {alias}",
            f"based in {alias}",
            f"linked to {alias}",
            f"support of {alias}",
        )
        if any(_contains(source, pattern) for pattern in patterns):
            return True
    return False


def _contains(corpus: str, phrase: str) -> bool:
    normalized = _normalize(phrase)
    return bool(normalized) and f" {normalized} " in f" {corpus} "


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()
