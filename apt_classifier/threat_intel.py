import hashlib
import html
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CISA_ADVISORIES_URL = "https://www.cisa.gov/cybersecurity-advisories/all.xml"
RANSOMWARE_LIVE_URL = "https://api.ransomware.live/v2/recentvictims"
CISA_KEV_CATALOG_URL = "https://www.cisa.gov/known-exploited-vulnerabilities-catalog"

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_INDUSTRY_TERMS = {
    "aerospace": ("aerospace", "aviation"),
    "defense": ("defense", "defence", "military"),
    "education": ("education", "university", "college"),
    "energy": ("energy", "oil", "gas", "electric utility"),
    "financial services": ("financial", "finance", "banking", "bank"),
    "government": ("government", "public sector", "ministry"),
    "healthcare": ("healthcare", "health care", "hospital", "medical"),
    "manufacturing": ("manufacturing", "manufacturer", "industrial"),
    "technology": ("technology", "software", "information technology"),
    "telecommunications": ("telecommunications", "telecom", "telco"),
    "transportation": ("transportation", "shipping", "maritime", "logistics"),
}
_COUNTRY_EQUIVALENTS = {
    "united states": {"united states", "us", "usa"},
    "united kingdom": {"united kingdom", "uk", "gb"},
    "canada": {"canada", "ca"},
    "china": {"china", "cn"},
    "france": {"france", "fr"},
    "germany": {"germany", "de"},
    "india": {"india", "in"},
    "italy": {"italy", "it"},
    "japan": {"japan", "jp"},
    "south korea": {"south korea", "kr"},
    "taiwan": {"taiwan", "tw"},
    "ukraine": {"ukraine", "ua"},
}


@dataclass(frozen=True)
class ThreatIntelItem:
    id: str
    source: str
    kind: str
    title: str
    summary: str
    published_at: str
    url: str
    verification: str
    actors: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()
    cves: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreatIntelSourceStatus:
    source: str
    url: str
    ok: bool
    item_count: int
    error: str | None = None


@dataclass(frozen=True)
class ThreatIntelSnapshot:
    fetched_at: str
    items: tuple[ThreatIntelItem, ...]
    sources: tuple[ThreatIntelSourceStatus, ...]

    def item_by_id(self) -> dict[str, ThreatIntelItem]:
        return {item.id: item for item in self.items}

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at,
            "items": [item.to_dict() for item in self.items],
            "sources": [asdict(source) for source in self.sources],
        }


@dataclass(frozen=True)
class ThreatIntelMatch:
    item_id: str
    score: float
    reasons: tuple[str, ...]
    group_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    verification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublicThreatIntelClient:
    def __init__(self, timeout: float = 20.0, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "APT-Attribution-Assistant/0.1 public-feed-reader"},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(
        self,
        *,
        days: int = 30,
        include_ransomware_claims: bool = True,
        now: datetime | None = None,
        max_items_per_source: int = 100,
    ) -> ThreatIntelSnapshot:
        current = _as_utc(now or datetime.now(UTC))
        items: list[ThreatIntelItem] = []
        statuses: list[ThreatIntelSourceStatus] = []
        sources = [
            (
                "CISA KEV",
                CISA_KEV_URL,
                lambda: parse_cisa_kev(
                    self._get_json(CISA_KEV_URL),
                    now=current,
                    days=days,
                    limit=max_items_per_source,
                ),
            ),
            (
                "CISA Alerts & Advisories",
                CISA_ADVISORIES_URL,
                lambda: parse_cisa_advisories(
                    self._get_text(CISA_ADVISORIES_URL),
                    now=current,
                    days=days,
                    limit=max_items_per_source,
                ),
            ),
        ]
        if include_ransomware_claims:
            sources.append(
                (
                    "ransomware.live",
                    RANSOMWARE_LIVE_URL,
                    lambda: parse_ransomware_claims(
                        self._get_json(RANSOMWARE_LIVE_URL),
                        now=current,
                        days=days,
                        limit=max_items_per_source,
                    ),
                )
            )
        for source_name, url, loader in sources:
            try:
                source_items = loader()
            except Exception as exc:
                statuses.append(
                    ThreatIntelSourceStatus(
                        source=source_name,
                        url=url,
                        ok=False,
                        item_count=0,
                        error=_safe_error(exc),
                    )
                )
                continue
            items.extend(source_items)
            statuses.append(
                ThreatIntelSourceStatus(
                    source=source_name,
                    url=url,
                    ok=True,
                    item_count=len(source_items),
                )
            )
        deduplicated = {item.id: item for item in items}
        ordered = tuple(
            sorted(
                deduplicated.values(),
                key=lambda item: (item.published_at, item.source, item.id),
                reverse=True,
            )
        )
        return ThreatIntelSnapshot(current.isoformat(), ordered, tuple(statuses))

    def _get_json(self, url: str) -> Any:
        return self._request(url).json()

    def _get_text(self, url: str) -> str:
        return self._request(url).text

    def _request(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._client.get(url)
                if len(response.content) > 20_000_000:
                    raise ValueError("public feed exceeded the 20 MB response limit")
                if response.status_code not in {408, 429} and response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = RuntimeError(f"HTTP {response.status_code}")
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt == 0:
                time.sleep(0.5)
        raise RuntimeError(_safe_error(last_error or RuntimeError("request failed")))


def parse_cisa_kev(
    payload: Any,
    *,
    now: datetime | None = None,
    days: int = 30,
    limit: int = 100,
) -> tuple[ThreatIntelItem, ...]:
    current = _as_utc(now or datetime.now(UTC))
    records = payload.get("vulnerabilities", []) if isinstance(payload, dict) else []
    items: list[ThreatIntelItem] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        published = _parse_datetime(record.get("dateAdded"))
        if not _recent(published, current, days):
            continue
        cve = str(record.get("cveID") or "").upper().strip()
        name = _compact(record.get("vulnerabilityName"))
        vendor = _compact(record.get("vendorProject"))
        product = _compact(record.get("product"))
        summary = _compact(record.get("shortDescription"))
        action = _compact(record.get("requiredAction"))
        if action:
            summary = f"{summary} Required action: {action}".strip()
        tags = ["active-exploitation"]
        if str(record.get("knownRansomwareCampaignUse") or "").casefold() == "known":
            tags.append("known-ransomware-use")
        title = f"{cve}: {name}" if cve else name
        items.append(
            ThreatIntelItem(
                id=_item_id("cisa-kev", cve or title, published.isoformat()),
                source="CISA KEV",
                kind="active_exploitation",
                title=title,
                summary=_limited(summary, 2500),
                published_at=published.isoformat(),
                url=CISA_KEV_CATALOG_URL,
                verification="authoritative",
                cves=(cve,) if cve else (),
                products=tuple(value for value in (f"{vendor} {product}".strip(),) if value),
                tags=tuple(tags),
            )
        )
    return tuple(sorted(items, key=lambda item: item.published_at, reverse=True)[:limit])


def parse_cisa_advisories(
    content: str | bytes,
    *,
    now: datetime | None = None,
    days: int = 30,
    limit: int = 100,
) -> tuple[ThreatIntelItem, ...]:
    current = _as_utc(now or datetime.now(UTC))
    raw = content if isinstance(content, bytes) else content.encode("utf-8")
    root = ET.fromstring(raw)
    items: list[ThreatIntelItem] = []
    for record in root.findall(".//item"):
        title = _compact(record.findtext("title"))
        summary = _clean_html(record.findtext("description") or "")
        published = _parse_datetime(
            record.findtext("pubDate") or record.findtext("{http://purl.org/dc/elements/1.1/}date")
        )
        if not title or not _recent(published, current, days):
            continue
        link = _safe_url(
            record.findtext("link") or "",
            {"cisa.gov", "www.cisa.gov"},
        )
        combined = f"{title} {summary}"
        cves = tuple(sorted({match.upper() for match in _CVE_RE.findall(combined)}))
        items.append(
            ThreatIntelItem(
                id=_item_id("cisa-advisory", link or title, published.isoformat()),
                source="CISA Alerts & Advisories",
                kind=_cisa_advisory_kind(combined),
                title=title,
                summary=_limited(summary, 3000),
                published_at=published.isoformat(),
                url=link,
                verification="authoritative",
                industries=_extract_industries(combined),
                cves=cves,
                tags=("cisa-advisory",),
            )
        )
    return tuple(sorted(items, key=lambda item: item.published_at, reverse=True)[:limit])


def parse_ransomware_claims(
    payload: Any,
    *,
    now: datetime | None = None,
    days: int = 30,
    limit: int = 100,
) -> tuple[ThreatIntelItem, ...]:
    current = _as_utc(now or datetime.now(UTC))
    records = payload if isinstance(payload, list) else []
    items: list[ThreatIntelItem] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        published = _parse_datetime(record.get("discovered"))
        if not _recent(published, current, days):
            continue
        victim = _compact(record.get("victim")) or "Unnamed victim"
        actor = _compact(record.get("group")) or "Unknown ransomware operator"
        industry = _compact(record.get("activity"))
        country = _compact(record.get("country")).upper()
        domain = _compact(record.get("domain"))
        summary = _clean_html(str(record.get("description") or ""))
        if domain:
            summary = f"Victim domain: {domain}. {summary}".strip()
        url = _safe_url(
            str(record.get("url") or ""),
            {"ransomware.live", "www.ransomware.live"},
        )
        items.append(
            ThreatIntelItem(
                id=_item_id("ransomware-live", actor, victim, published.isoformat()),
                source="ransomware.live",
                kind="claimed_breach",
                title=f"{actor} claim: {victim}",
                summary=_limited(summary, 2000),
                published_at=published.isoformat(),
                url=url,
                verification="unverified_extortion_claim",
                actors=(actor,),
                countries=(country,) if country else (),
                industries=(industry,) if industry else (),
                products=(domain,) if domain else (),
                tags=("ransomware", "unverified-claim"),
            )
        )
    return tuple(sorted(items, key=lambda item: item.published_at, reverse=True)[:limit])


def match_threat_intel_to_case(
    kb: Any,
    state: Any,
    snapshot: ThreatIntelSnapshot,
    *,
    limit: int = 50,
) -> tuple[ThreatIntelMatch, ...]:
    observations = tuple(getattr(state.ledger, "observations", ()))
    case_cves = {
        match.upper() for observation in observations for match in _CVE_RE.findall(observation.text)
    }
    candidate_aliases: dict[str, tuple[str, ...]] = {}
    groups = getattr(kb, "groups", {})
    for entry in getattr(state.fusion, "entries", ())[:15]:
        group = groups.get(entry.group_id)
        aliases = {
            _compact(_record_value(group, "name", "")),
            *(_compact(value) for value in _record_value(group, "aliases", ())),
        }
        candidate_aliases[entry.group_id] = tuple(
            sorted((alias for alias in aliases if len(alias) >= 4), key=len, reverse=True)
        )
    software_aliases: dict[str, tuple[str, ...]] = {}
    software = getattr(kb, "software", {})
    for software_id in getattr(state.context, "software_ids", ()):
        item = software.get(software_id)
        aliases = {
            _compact(_record_value(item, "name", "")),
            *(_compact(value) for value in _record_value(item, "aliases", ())),
        }
        software_aliases[software_id] = tuple(alias for alias in aliases if len(alias) >= 4)
    strategic = getattr(state.ledger, "strategic_context", None)
    target_countries = tuple(strategic.target_geographies) if strategic is not None else ()
    target_industries = tuple(strategic.target_industries) if strategic is not None else ()
    matches: list[ThreatIntelMatch] = []
    for item in snapshot.items:
        item_text = _normalize(
            " ".join(
                (
                    item.title,
                    item.summary,
                    *item.actors,
                    *item.products,
                    *item.industries,
                )
            )
        )
        score = 0.0
        reasons: list[str] = []
        group_ids: list[str] = []
        observation_ids: set[str] = set()
        cve_overlap = case_cves & set(item.cves)
        if cve_overlap:
            score += 5.0
            reasons.append(f"CVE overlap: {', '.join(sorted(cve_overlap))}")
            for observation in observations:
                if cve_overlap & {value.upper() for value in _CVE_RE.findall(observation.text)}:
                    observation_ids.add(observation.id)
        for group_id, aliases in candidate_aliases.items():
            matched_alias = next(
                (alias for alias in aliases if _contains_phrase(item_text, alias)), None
            )
            if matched_alias:
                group_ids.append(group_id)
                reasons.append(f"Ranked actor reference: {matched_alias}")
        if group_ids:
            score += 4.0
        matched_tools = []
        for aliases in software_aliases.values():
            matched_alias = next(
                (alias for alias in aliases if _contains_phrase(item_text, alias)), None
            )
            if matched_alias:
                matched_tools.append(matched_alias)
                for observation in observations:
                    if _contains_phrase(_normalize(observation.text), matched_alias):
                        observation_ids.add(observation.id)
        if matched_tools:
            score += 3.0
            reasons.append(f"Named tool: {', '.join(sorted(set(matched_tools)))}")
        matched_products = [
            product
            for product in item.products
            if len(product) >= 4
            and any(
                _contains_phrase(_normalize(observation.text), product)
                for observation in observations
            )
        ]
        if matched_products:
            score += 2.0
            reasons.append(f"Product overlap: {', '.join(matched_products)}")
            for observation in observations:
                if any(
                    _contains_phrase(_normalize(observation.text), product)
                    for product in matched_products
                ):
                    observation_ids.add(observation.id)
        industry_matches = [
            value
            for value in target_industries
            if _industry_context_match(value, item.industries, item_text)
        ]
        if industry_matches:
            score += 1.0
            reasons.append(f"Target industry: {', '.join(industry_matches)}")
        country_matches = [
            value for value in target_countries if _country_context_match(value, item.countries)
        ]
        if country_matches:
            score += 0.5
            reasons.append(f"Target geography: {', '.join(country_matches)}")
        if score >= 1.0:
            matches.append(
                ThreatIntelMatch(
                    item_id=item.id,
                    score=score,
                    reasons=tuple(dict.fromkeys(reasons)),
                    group_ids=tuple(dict.fromkeys(group_ids)),
                    observation_ids=tuple(sorted(observation_ids)),
                    verification=item.verification,
                )
            )
    return tuple(
        sorted(
            matches,
            key=lambda item: (
                -item.score,
                snapshot.item_by_id()[item.item_id].published_at,
                item.item_id,
            ),
        )[:limit]
    )


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _recent(value: datetime | None, now: datetime, days: int) -> bool:
    return value is not None and now - timedelta(days=max(days, 1)) <= value <= now + timedelta(
        days=1
    )


def _clean_html(value: str) -> str:
    return _compact(html.unescape(_TAG_RE.sub(" ", value)))


def _compact(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _limited(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit].rstrip()}…"


def _normalize(value: str) -> str:
    return _compact(re.sub(r"[^a-z0-9]+", " ", value.casefold()))


def _contains_phrase(corpus: str, phrase: str) -> bool:
    normalized = _normalize(phrase)
    return bool(normalized) and f" {normalized} " in f" {corpus} "


def _cisa_advisory_kind(value: str) -> str:
    normalized = _normalize(value)
    campaign_terms = (
        "campaign",
        "state actors",
        "state sponsored",
        "threat actors",
        "ransomware",
        "espionage",
        "malicious cyber activity",
    )
    return (
        "campaign_advisory"
        if any(term in normalized for term in campaign_terms)
        else "security_advisory"
    )


def _extract_industries(value: str) -> tuple[str, ...]:
    normalized = _normalize(value)
    return tuple(
        industry
        for industry, aliases in _INDUSTRY_TERMS.items()
        if any(_contains_phrase(normalized, alias) for alias in aliases)
    )


def _industry_context_match(value: str, industries: tuple[str, ...], item_text: str) -> bool:
    normalized = _normalize(value)
    aliases = next(
        (
            terms
            for industry, terms in _INDUSTRY_TERMS.items()
            if normalized == industry or normalized in {_normalize(term) for term in terms}
        ),
        (normalized,),
    )
    corpus = _normalize(" ".join((*industries, item_text)))
    return any(_contains_phrase(corpus, alias) for alias in aliases)


def _country_context_match(value: str, countries: tuple[str, ...]) -> bool:
    normalized = _normalize(value)
    aliases = next(
        (
            terms
            for country, terms in _COUNTRY_EQUIVALENTS.items()
            if normalized == country or normalized in terms
        ),
        {normalized},
    )
    normalized_countries = {_normalize(country) for country in countries}
    return bool(set(aliases) & normalized_countries)


def _safe_url(value: str, allowed_hosts: set[str]) -> str:
    parsed = urlparse(value.strip())
    return value.strip() if parsed.scheme == "https" and parsed.hostname in allowed_hosts else ""


def _item_id(*values: str) -> str:
    material = "\x1f".join(values)
    return f"intel-{hashlib.sha256(material.encode()).hexdigest()[:20]}"


def _record_value(record: Any, name: str, default: Any = "") -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _safe_error(error: Exception) -> str:
    return _limited(f"{type(error).__name__}: {error}", 240)
