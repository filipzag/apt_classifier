import re
from dataclasses import asdict, dataclass
from typing import Any

from apt_classifier.attack.parse import split_sentences
from apt_classifier.fusion import FusionResult
from apt_classifier.memo import ObjectMemo

_COUNTRY_ALIASES = {
    "Belarus": ("belarus", "belarusian"),
    "China": (
        "people s republic of china prc",
        "people s republic of china",
        "china",
        "chinese",
        "prc",
    ),
    "India": ("india", "indian"),
    "Iran": ("iran", "iranian"),
    "Israel": ("israel", "israeli"),
    "Lebanon": ("lebanon", "lebanese"),
    "North Korea": ("north korea", "north korean", "dprk"),
    "Pakistan": ("pakistan", "pakistani"),
    "Russia": ("russia", "russian"),
    "South Korea": ("south korea", "south korean", "rok"),
    "Syria": ("syria", "syrian"),
    "Turkey": ("turkey", "turkish", "türkiye", "turkiye"),
    "Ukraine": ("ukraine", "ukrainian"),
    "United States": ("united states", "u s", "american"),
    "Vietnam": ("vietnam", "vietnamese"),
}

_STATE_MARKERS = {
    "cyber espionage": 1.0,
    "espionage": 0.8,
    "state sponsored": 1.0,
    "state affiliated": 1.0,
    "government backed": 1.0,
    "government sponsored": 1.0,
    "nation state": 1.0,
    "intelligence service": 1.0,
    "military intelligence": 1.0,
    "main intelligence directorate": 1.0,
    "general staff": 0.9,
    "reconnaissance general bureau": 1.0,
    "ministry of": 0.8,
    "state nexus": 0.7,
}

_CRIMINAL_MARKERS = {
    "financially motivated": 1.0,
    "cybercriminal": 1.0,
    "criminal group": 0.9,
    "ecrime": 0.9,
    "ransomware": 0.8,
    "extortion": 0.8,
    "payment card": 0.8,
    "bank fraud": 0.8,
    "financial theft": 0.8,
    "cryptocurrency theft": 0.8,
    "monetize": 0.6,
}


@dataclass(frozen=True)
class OriginEvidence:
    group_id: str
    group_name: str
    country: str
    candidate_rank: int
    contribution: float
    excerpt: str


@dataclass(frozen=True)
class CountryHypothesis:
    country: str
    relative_support: float
    supporting_group_ids: tuple[str, ...]
    evidence: tuple[OriginEvidence, ...]


@dataclass(frozen=True)
class OriginAssessment:
    hypotheses: tuple[CountryHypothesis, ...]
    confidence: str
    candidate_coverage: float
    caveats: tuple[str, ...]

    @property
    def top_country(self) -> str | None:
        return (
            self.hypotheses[0].country
            if self.hypotheses and self.confidence != "insufficient"
            else None
        )


@dataclass(frozen=True)
class ThreatTypeAssessment:
    label: str
    likely_apt: bool | None
    apt_like_support: float
    criminal_support: float
    unresolved_support: float
    confidence: str
    apt_supporting_group_ids: tuple[str, ...]
    criminal_supporting_group_ids: tuple[str, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class OriginAndThreatAssessment:
    origin: OriginAssessment
    threat_type: ThreatTypeAssessment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_origin_and_threat_type(
    kb: Any, fusion: FusionResult, *, top_n: int = 10
) -> OriginAndThreatAssessment:
    entries = tuple(fusion.entries[:top_n])
    if not entries:
        return OriginAndThreatAssessment(_empty_origin(), _empty_threat_type())
    maximum_score = max(max(float(entry.score), 0.0) for entry in entries)
    if maximum_score <= 0:
        return OriginAndThreatAssessment(_empty_origin(), _empty_threat_type())
    raw_weights = {
        entry.group_id: (max(float(entry.score), 0.0) / maximum_score) ** 2 / entry.rank
        for entry in entries
    }
    total_weight = sum(raw_weights.values())
    candidate_weights = {
        group_id: weight / total_weight for group_id, weight in raw_weights.items()
    }
    origin = _assess_origin(kb, entries, candidate_weights)
    threat_type = _assess_threat_type(kb, entries, candidate_weights)
    return OriginAndThreatAssessment(origin, threat_type)


def _assess_origin(
    kb: Any, entries: tuple[Any, ...], candidate_weights: dict[str, float]
) -> OriginAssessment:
    groups = getattr(kb, "groups", {})
    support: dict[str, float] = {}
    evidence: dict[str, list[OriginEvidence]] = {}
    resolved_weight = 0.0
    for entry in entries:
        group = groups.get(entry.group_id)
        if group is None:
            continue
        origins = documented_origins(kb, entry.group_id)
        if not origins:
            continue
        candidate_weight = candidate_weights[entry.group_id]
        resolved_weight += candidate_weight
        contribution = candidate_weight / len(origins)
        for country, excerpt in origins.items():
            support[country] = support.get(country, 0.0) + contribution
            evidence.setdefault(country, []).append(
                OriginEvidence(
                    group_id=entry.group_id,
                    group_name=str(_record_value(group, "name", entry.group_id)),
                    country=country,
                    candidate_rank=entry.rank,
                    contribution=contribution,
                    excerpt=excerpt,
                )
            )
    total_resolved = sum(support.values())
    hypotheses = tuple(
        CountryHypothesis(
            country=country,
            relative_support=100.0 * value / total_resolved,
            supporting_group_ids=tuple(
                dict.fromkeys(item.group_id for item in evidence.get(country, ()))
            ),
            evidence=tuple(evidence.get(country, ())),
        )
        for country, value in sorted(support.items(), key=lambda item: (-item[1], item[0]))
    )
    if not hypotheses or resolved_weight < 0.20:
        confidence = "insufficient"
    elif hypotheses[0].relative_support >= 65.0 and resolved_weight >= 0.50:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "medium" and _ranking_is_weak(entries):
        confidence = "low"
    return OriginAssessment(
        hypotheses=hypotheses,
        confidence=confidence,
        candidate_coverage=100.0 * resolved_weight,
        caveats=(
            "Origin support comes from public ATT&CK attribution reporting, not source-IP geolocation.",
            "Infrastructure relays, false flags, shared tooling, and reporting bias can misidentify origin.",
            "An unresolved result is preferable to inferring victim geography as attacker origin.",
        ),
    )


def _assess_threat_type(
    kb: Any, entries: tuple[Any, ...], candidate_weights: dict[str, float]
) -> ThreatTypeAssessment:
    apt_support = 0.0
    criminal_support = 0.0
    unresolved_support = 0.0
    apt_groups: list[str] = []
    criminal_groups: list[str] = []
    for entry in entries:
        profile = _normalize(" ".join(_profile_texts(kb, entry.group_id)))
        state_strength = min(
            sum(weight for marker, weight in _STATE_MARKERS.items() if marker in profile), 1.0
        )
        criminal_strength = min(
            sum(weight for marker, weight in _CRIMINAL_MARKERS.items() if marker in profile), 1.0
        )
        weight = candidate_weights[entry.group_id]
        if state_strength == 0 and criminal_strength == 0:
            unresolved_support += weight
            continue
        combined = state_strength + criminal_strength
        apt_part = weight * state_strength / combined
        criminal_part = weight * criminal_strength / combined
        apt_support += apt_part
        criminal_support += criminal_part
        if apt_part:
            apt_groups.append(entry.group_id)
        if criminal_part:
            criminal_groups.append(entry.group_id)
    apt_percent = 100.0 * apt_support
    criminal_percent = 100.0 * criminal_support
    unresolved_percent = 100.0 * unresolved_support
    if apt_percent >= 55.0 and apt_percent >= criminal_percent + 15.0:
        label = "APT-like state/espionage activity"
        likely_apt: bool | None = True
        dominant = apt_percent
    elif apt_percent >= 35.0 and criminal_percent >= 35.0:
        label = "APT-like with mixed state and criminal objectives"
        likely_apt = True
        dominant = max(apt_percent, criminal_percent)
    elif criminal_percent >= 55.0 and criminal_percent >= apt_percent + 15.0:
        label = "Likely financially motivated criminal intrusion"
        likely_apt = False
        dominant = criminal_percent
    else:
        label = "Indeterminate / mixed"
        likely_apt = None
        dominant = max(apt_percent, criminal_percent)
    resolved_percent = apt_percent + criminal_percent
    if resolved_percent < 35.0:
        confidence = "insufficient"
    elif dominant >= 75.0:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "medium" and _ranking_is_weak(entries):
        confidence = "low"
    return ThreatTypeAssessment(
        label=label,
        likely_apt=likely_apt,
        apt_like_support=apt_percent,
        criminal_support=criminal_percent,
        unresolved_support=unresolved_percent,
        confidence=confidence,
        apt_supporting_group_ids=tuple(dict.fromkeys(apt_groups)),
        criminal_supporting_group_ids=tuple(dict.fromkeys(criminal_groups)),
        rationale=(
            "APT-like support requires documented state sponsorship, espionage, government backing, or intelligence-service links.",
            "Criminal support requires documented financial motivation, ransomware, fraud, extortion, or e-crime activity.",
            "Technical sophistication or persistence alone is not treated as proof of state sponsorship.",
        ),
    )


def _ranking_is_weak(entries: tuple[Any, ...]) -> bool:
    if not entries:
        return True
    if bool(getattr(entries[0], "contested", False)):
        return True
    if len(entries) < 2:
        return False
    return float(entries[0].relative_score) - float(entries[1].relative_score) < 10.0


# More origin phrasings ATT&CK uses (2026-09-30), which lift the groups with one documented source
# country from 53 to 91 of 176 without losing any. A country adjective counts only in the
# group's own self-description ("APT1 is a Chinese threat group", "Magic Hound is an
# Iranian-sponsored threat group"): elsewhere "Chinese" is as likely to describe victims or
# another group. Language never counts ("Russian-speaking"), and "South American" is a region,
# not the United States, because the adjective must directly follow the article or qualifier.
_NOT_LANGUAGE = r"(?!\s+(?:speaking|speakers|language|languages)\b)"
_QUALIFIER = r"(?:(?:suspected|likely|probable|possible|alleged|reportedly)\s+)?"
_LINK = (
    r"(?:\s+(?:state|government|military))?"
    r"(?:\s+(?:sponsored|backed|linked|aligned|related|affiliated|nexus|based))?"
)
_ACTOR_WORDS = (
    r"(?:\s+(?:cyber\s*espionage|cyberespionage|cyber|espionage|advanced\s+persistent|state"
    r"|threat|apt|hacking|activity|intrusion))*"
)
_ACTOR_NOUN = r"\s+(?:group|actor|entity|apt|team|unit|adversary|operator|operators)\b"
_CITATION_RE = re.compile(r"\(Citation:[^)]*\)")
_ORIGIN_TABLE_CACHE: ObjectMemo[dict[str, dict[str, str]]] = ObjectMemo()


def _extra_origin_patterns(alias: str) -> tuple[str, ...]:
    escaped = re.escape(alias)
    return (
        rf"\b(?:operat(?:e|es|ing)|based)\s+(?:out\s+of|from)\s+(?:the\s+)?"
        rf"(?:[a-z]+\s+province\s+of\s+)?{escaped}\b{_NOT_LANGUAGE}",
        rf"\b{escaped}{_NOT_LANGUAGE}\s+origins?\b",
        rf"\b(?:nexus|ties|link|links|connection|connections)\s+to\s+(?:the\s+)?{escaped}"
        rf"{_NOT_LANGUAGE}(?:\s+(?:government|state|intelligence|military))?\b",
        rf"\bassociated\s+with\s+(?:the\s+)?{escaped}{_NOT_LANGUAGE}"
        rf"\s+(?:government|state|military|intelligence)\b",
    )


def _self_description_pattern(names: tuple[str, ...], alias: str) -> str:
    who = "|".join(re.escape(name) for name in names)
    return (
        rf"^(?:{who})\s+(?:is|are|was|were)\s+(?:(?:a|an|the)\s+)?{_QUALIFIER}"
        rf"{re.escape(alias)}{_NOT_LANGUAGE}{_LINK}{_ACTOR_WORDS}{_ACTOR_NOUN}"
    )


def _documented_origins(texts: list[str], names: tuple[str, ...] = ()) -> dict[str, str]:
    """Countries a group's profile states as its origin, with the stating sentence.

    ``names`` (the group's name and aliases) enable the self-description pattern.
    """
    normalized_names = tuple(
        sorted(
            {_normalize(name) for name in names if len(_normalize(name)) >= 2},
            key=len,
            reverse=True,
        )
    )
    found: dict[str, str] = {}
    for text in texts:
        for sentence in split_sentences(_CITATION_RE.sub("", text)):
            normalized = _normalize(sentence)
            for country, aliases in _COUNTRY_ALIASES.items():
                if country in found:
                    continue
                for alias in aliases:
                    candidate = _normalize(alias)
                    if f" {candidate} " not in f" {normalized} ":
                        continue  # every pattern needs the alias itself; skip the regex work
                    if (
                        _origin_pattern(normalized, candidate)
                        or any(
                            re.search(pattern, normalized)
                            for pattern in _extra_origin_patterns(candidate)
                        )
                        or (
                            normalized_names
                            and re.search(
                                _self_description_pattern(normalized_names, candidate), normalized
                            )
                        )
                    ):
                        found[country] = sentence.strip()
                        break
    return found


def documented_origin_table(kb: Any) -> dict[str, dict[str, str]]:
    """``{group_id: {country: stating sentence}}`` for every group (memoised per KB)."""
    groups = getattr(kb, "groups", {}) or {}
    stamp = (
        str(getattr(kb, "version", "") or ""),
        len(groups),
        len(getattr(kb, "campaigns", {}) or {}),
    )
    cached = _ORIGIN_TABLE_CACHE.get(kb, stamp)
    if cached is not None:
        return cached
    table = {}
    for group_id, group in groups.items():
        names = (
            str(_record_value(group, "name", "")),
            *(_record_value(group, "aliases", ()) or ()),
        )
        table[group_id] = _documented_origins(_profile_texts(kb, group_id), tuple(names))
    return _ORIGIN_TABLE_CACHE.put(kb, stamp, table)


def documented_origins(kb: Any, group_id: str) -> dict[str, str]:
    return dict(documented_origin_table(kb).get(group_id, {}))


def _origin_pattern(sentence: str, alias: str) -> bool:
    escaped = re.escape(alias)
    patterns = (
        rf"\b{escaped}(?:\s+[a-z0-9]+){{0,2}}\s+(?:based|nexus|state\s+sponsored|state\s+affiliated|government\s+backed|government\s+sponsored|military\s+intelligence|intelligence\s+service|ministry)\b",
        rf"\b(?:based|originating|operating)(?:\s+[a-z0-9]+){{0,2}}\s+(?:in|from)\s+(?:the\s+)?{escaped}\b",
        rf"\b(?:attributed|linked|affiliated|connected)(?:\s+[a-z0-9]+){{0,3}}\s+(?:to|with)\s+(?:the\s+)?{escaped}\b",
        rf"\bsupport(?:ing)?\s+(?:the\s+)?{escaped}\s+interests\b",
        rf"\b(?:ministry|military\s+intelligence|intelligence\s+service)(?:\s+[a-z0-9]+){{0,4}}\s+(?:of|for)\s+(?:the\s+)?{escaped}\b",
    )
    return any(re.search(pattern, sentence) for pattern in patterns)


def _profile_texts(kb: Any, group_id: str) -> list[str]:
    groups = getattr(kb, "groups", {})
    campaigns = getattr(kb, "campaigns", {})
    group = groups.get(group_id)
    if group is None:
        return []
    texts = [str(_record_value(group, "description", ""))]
    campaign_ids = set(_record_value(group, "campaign_ids", ()))
    for campaign_id, linked_groups in getattr(kb, "campaign_groups", {}).items():
        if group_id in set(linked_groups):
            campaign_ids.add(campaign_id)
    texts.extend(
        str(_record_value(campaigns.get(campaign_id), "description", ""))
        for campaign_id in sorted(campaign_ids)
    )
    return [text for text in texts if text.strip()]


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


def _empty_origin() -> OriginAssessment:
    return OriginAssessment(
        hypotheses=(),
        confidence="insufficient",
        candidate_coverage=0.0,
        caveats=("No technically supported candidate origin could be resolved.",),
    )


def _empty_threat_type() -> ThreatTypeAssessment:
    return ThreatTypeAssessment(
        label="Indeterminate",
        likely_apt=None,
        apt_like_support=0.0,
        criminal_support=0.0,
        unresolved_support=100.0,
        confidence="insufficient",
        apt_supporting_group_ids=(),
        criminal_supporting_group_ids=(),
        rationale=("No candidate profile evidence was available.",),
    )
