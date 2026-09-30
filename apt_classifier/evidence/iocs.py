import ipaddress
import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.memo import ObjectMemo

_HASH_RE = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{128}|[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32})(?![0-9a-f])"
)
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_RE = re.compile(r"(?<![0-9A-Za-z])\[?[0-9A-Fa-f:]{3,}\]?(?![0-9A-Za-z:])")
_DOMAIN_RE = re.compile(
    r"(?i)(?<![\w-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,63}|xn--[a-z0-9-]{2,59})(?![\w-])"
)
_FILE_SUFFIXES = {
    "bat",
    "bin",
    "cab",
    "cmd",
    "conf",
    "config",
    "csv",
    "dat",
    "dll",
    "doc",
    "docm",
    "docx",
    "exe",
    "hta",
    "ini",
    "jar",
    "js",
    "json",
    "lnk",
    "log",
    "msi",
    "pdf",
    "ps1",
    "rar",
    "rtf",
    "scr",
    "sys",
    "tmp",
    "txt",
    "vbs",
    "xls",
    "xlsm",
    "xlsx",
    "xml",
    "yaml",
    "yml",
}
_GENERIC_ALIASES = {
    "agent",
    "at",
    "cmd",
    "http",
    "https",
    "malware",
    "net",
    "ping",
    "script",
    "service",
    "system",
    "tool",
    "windows",
}


@dataclass(frozen=True, slots=True)
class IOCMatch:
    indicator_type: str
    value: str
    line_number: int
    raw_excerpt: str
    start: int
    end: int
    subtype: str | None = None


@dataclass(frozen=True, slots=True)
class SoftwareAliasMatch:
    software_id: str | None
    name: str
    software_type: str
    matched_alias: str
    line_number: int
    raw_excerpt: str
    start: int
    end: int


def _line_details(text: str, position: int) -> tuple[int, str]:
    line_number = text.count("\n", 0, position) + 1
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end < 0:
        end = len(text)
    return line_number, text[start:end].strip()


def _hash_type(value: str) -> str:
    return {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}[len(value)]


def _ip_matches(text: str) -> list[IOCMatch]:
    matches: list[IOCMatch] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in (_IPV4_RE, _IPV6_RE):
        for match in pattern.finditer(text):
            candidate = match.group().strip("[]")
            if ":" not in candidate and pattern is _IPV6_RE:
                continue
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            line, excerpt = _line_details(text, match.start())
            matches.append(
                IOCMatch(
                    "ip",
                    address.compressed.lower(),
                    line,
                    excerpt,
                    match.start(),
                    match.end(),
                    f"ipv{address.version}",
                )
            )
    return matches


def find_iocs(text: str) -> list[IOCMatch]:
    source = str(text or "")
    matches: list[IOCMatch] = []
    for match in _HASH_RE.finditer(source):
        value = match.group().lower()
        line, excerpt = _line_details(source, match.start())
        matches.append(
            IOCMatch(
                "hash",
                value,
                line,
                excerpt,
                match.start(),
                match.end(),
                _hash_type(value),
            )
        )
    matches.extend(_ip_matches(source))
    for match in _DOMAIN_RE.finditer(source):
        value = match.group().rstrip(".").lower()
        suffix = value.rsplit(".", 1)[-1]
        if suffix in _FILE_SUFFIXES:
            continue
        try:
            ipaddress.ip_address(value)
        except ValueError:
            pass
        else:
            continue
        line, excerpt = _line_details(source, match.start())
        matches.append(
            IOCMatch(
                "domain",
                value,
                line,
                excerpt,
                match.start(),
                match.end(),
            )
        )
    return sorted(matches, key=lambda item: (item.start, item.end, item.indicator_type))


def _value(record: Any, *names: str) -> Any:
    if isinstance(record, Mapping):
        normalized = {
            re.sub(r"[^a-z0-9]", "", str(key).casefold()): value for key, value in record.items()
        }
        for name in names:
            key = re.sub(r"[^a-z0-9]", "", name.casefold())
            if key in normalized:
                return normalized[key]
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _iter_container(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, str | bytes):
        return []
    if isinstance(value, Iterable):
        return list(value)
    return []


def _software_records(kb: Any) -> list[Any]:
    if kb is None:
        return []
    containers: list[Any] = []
    for name in (
        "software",
        "softwares",
        "software_by_id",
        "softwares_by_id",
        "software_by_name",
        "objects",
    ):
        value = _value(kb, name)
        if value is not None:
            containers.append(value)
    for name in ("iter_software", "all_software", "list_software"):
        method = getattr(kb, name, None)
        if callable(method):
            with suppress(TypeError):
                containers.append(method())
    if not containers and isinstance(kb, Mapping | Iterable) and not isinstance(kb, str | bytes):
        containers.append(kb)
    records: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for container in containers:
        for record in _iter_container(container):
            record_type = str(_value(record, "software_type", "type", "kind") or "").casefold()
            if record_type and record_type not in {"malware", "tool", "software"}:
                continue
            name = str(_value(record, "name") or "").strip()
            if not name:
                continue
            identifier = str(_value(record, "attack_id", "external_id", "software_id", "id") or "")
            marker = (identifier, name.casefold())
            if marker not in seen:
                seen.add(marker)
                records.append(record)
    return records


def _external_id(record: Any) -> str | None:
    for name in ("attack_id", "external_id", "software_id", "mitre_id"):
        value = _value(record, name)
        if value:
            return str(value)
    direct = _value(record, "id")
    if direct and re.fullmatch(r"S\d{4}", str(direct), re.I):
        return str(direct).upper()
    references = _value(record, "external_references")
    for reference in _iter_container(references):
        value = _value(reference, "external_id")
        if value and re.fullmatch(r"S\d{4}", str(value), re.I):
            return str(value).upper()
    return str(direct) if direct else None


def _aliases(record: Any) -> list[str]:
    values: list[str] = []
    name = _value(record, "name")
    if name:
        values.append(str(name))
    for field in ("aliases", "x_mitre_aliases", "software_aliases"):
        aliases = _value(record, field)
        if isinstance(aliases, str):
            values.append(aliases)
        else:
            values.extend(str(alias) for alias in _iter_container(aliases))
    return list(dict.fromkeys(alias.strip() for alias in values if alias.strip()))


def _alias_pattern(alias: str, *, exact_case: bool = False) -> re.Pattern[str]:
    prefix = r"(?<!\w)" if alias[0].isalnum() or alias[0] == "_" else ""
    suffix = r"(?!\w)" if alias[-1].isalnum() or alias[-1] == "_" else ""
    return re.compile(f"{prefix}{re.escape(alias)}{suffix}", 0 if exact_case else re.I)


# Ordinary-word aliases. ATT&CK lists some software under names that are also plain English
# words ("Page" for Elise, "Wiper", "Expand", "CALENDAR"). Matched case-insensitively they fire
# on running text ("the payment page") and, for malware documented for a single group, hand that
# group decisive software evidence. Which aliases are ambiguous is read from the knowledge base
# itself rather than from a hand-written list: an alias is ambiguous when ATT&CK's own prose
# (technique, group and software descriptions and procedure sentences; code spans, citations,
# link targets and URLs removed; the software's own description excluded) uses its lowercase
# form as an ordinary word at least as often as it writes the exact name. Such an alias only
# matches in its exact spelling; a title-case one ("Page") also never at the start of a
# sentence, where a leading capital carries no information (CALENDAR and ZLib still do).
# Aliases whose canonical spelling is lowercase (netsh, certutil) are command names and keep
# the case-insensitive match.
_PROSE_TOKEN_RE = re.compile(r"(?<![\w\-./\\])[A-Za-z][A-Za-z0-9\-]*(?![\w\-]|\.\w)")
_SINGLE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_PROSE_NOISE_RE = re.compile(
    r"<code>.*?</code>|`[^`]*`|\(Citation:[^)]*\)|\]\([^)]*\)|https?://\S+", re.S
)
_SENTENCE_LEAD_CHARS = " \t\"'\u201c\u2018([{*-\u2022"
_AMBIGUOUS_ALIAS_CACHE: ObjectMemo[frozenset[str]] = ObjectMemo()


def _sentence_initial(text: str, start: int) -> bool:
    prefix = text[:start].rstrip(_SENTENCE_LEAD_CHARS)
    return not prefix or prefix[-1] in ".!?:;\n\r"


def _title_case(token: str) -> bool:
    """Only the first letter is a capital: at a sentence start that capital says nothing."""
    return token[:1].isupper() and token[1:] == token[1:].lower()


def _prose_counts(text: str, wanted: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    cleaned = _PROSE_NOISE_RE.sub(" ", text)
    for match in _PROSE_TOKEN_RE.finditer(cleaned):
        token = match.group()
        if token not in wanted:
            continue
        if _title_case(token) and _sentence_initial(cleaned, match.start()):
            continue  # a leading capital at the start of a sentence is not evidence of a name
        counts[token] = counts.get(token, 0) + 1
    return counts


def _kb_prose(kb: Any) -> list[str]:
    texts: list[str] = []
    for container_name in ("techniques", "groups"):
        container = _value(kb, container_name)
        values = container.values() if isinstance(container, Mapping) else ()
        for record in values:
            texts.append(str(_value(record, "description") or ""))
            behaviours = _value(record, "technique_behaviors")
            if isinstance(behaviours, Mapping):
                for sentences in behaviours.values():
                    texts.extend(str(sentence) for sentence in _iter_container(sentences))
    return texts


def ambiguous_software_aliases(kb: Any) -> frozenset[str]:
    """Aliases (exact spelling) that only match case-sensitively; see the note above."""
    if kb is None:
        return frozenset()
    records = _software_records(kb)
    stamp = (
        str(_value(kb, "version") or ""),
        len(records),
        len(_value(kb, "techniques") or ()),
        len(_value(kb, "groups") or ()),
    )
    cached = _AMBIGUOUS_ALIAS_CACHE.get(kb, stamp)
    if cached is not None:
        return cached
    candidates: dict[str, list[str]] = {}
    for record in records:
        for alias in _aliases(record):
            if (
                len(alias) >= 3
                and alias != alias.lower()
                and alias.casefold() not in _GENERIC_ALIASES
                and _SINGLE_TOKEN_RE.fullmatch(alias)
            ):
                candidates.setdefault(alias, []).append(str(_value(record, "description") or ""))
    if not candidates:
        return _AMBIGUOUS_ALIAS_CACHE.put(kb, stamp, frozenset())
    wanted = set(candidates) | {alias.lower() for alias in candidates}
    totals: dict[str, int] = {}
    for text in [*_kb_prose(kb), *(str(_value(r, "description") or "") for r in records)]:
        for token, count in _prose_counts(text, wanted).items():
            totals[token] = totals.get(token, 0) + count
    ambiguous: set[str] = set()
    for alias, own_descriptions in candidates.items():
        word_uses = totals.get(alias.lower(), 0)
        name_uses = totals.get(alias, 0)
        for description in own_descriptions:  # the software's own description is not evidence
            own = _prose_counts(description, {alias, alias.lower()})
            word_uses -= own.get(alias.lower(), 0)
            name_uses -= own.get(alias, 0)
        if word_uses > 0 and word_uses >= name_uses:
            ambiguous.add(alias)
    return _AMBIGUOUS_ALIAS_CACHE.put(kb, stamp, frozenset(ambiguous))


def match_software_aliases(text: str, kb: Any) -> list[SoftwareAliasMatch]:
    source = str(text or "")
    matches: list[SoftwareAliasMatch] = []
    ambiguous = ambiguous_software_aliases(kb)
    for record in _software_records(kb):
        name = str(_value(record, "name")).strip()
        software_id = _external_id(record)
        raw_type = str(_value(record, "software_type", "type", "kind") or "software")
        software_type = (
            raw_type.casefold() if raw_type.casefold() in {"malware", "tool"} else "software"
        )
        seen_spans: set[tuple[int, int]] = set()
        aliases = sorted(_aliases(record), key=lambda value: (-len(value), value.casefold()))
        for alias in aliases:
            if len(alias) < 3 or alias.casefold() in _GENERIC_ALIASES or alias.isdecimal():
                continue
            exact_case = alias in ambiguous
            for match in _alias_pattern(alias, exact_case=exact_case).finditer(source):
                start, end = match.span()
                if exact_case and _title_case(alias) and _sentence_initial(source, start):
                    continue
                if any(
                    start < seen_end and seen_start < end for seen_start, seen_end in seen_spans
                ):
                    continue
                seen_spans.add((start, end))
                line, excerpt = _line_details(source, match.start())
                matches.append(
                    SoftwareAliasMatch(
                        software_id,
                        name,
                        software_type,
                        match.group(),
                        line,
                        excerpt,
                        match.start(),
                        match.end(),
                    )
                )
    return sorted(matches, key=lambda item: (item.start, -item.end, item.name.casefold()))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8-sig")
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, Iterable):
        return "\n".join(str(item) for item in value)
    return str(value)


def extract_iocs(
    text: Any,
    kb: Any = None,
    *,
    source_name: str | None = None,
) -> list[Observation]:
    source = _as_text(text)
    ledger = EvidenceLedger()
    for match in find_iocs(source):
        if match.indicator_type == "hash" and match.subtype:
            label = match.subtype.upper()
        elif match.indicator_type == "ip" and match.subtype:
            label = match.subtype.upper().replace("IPV", "IPv")
        else:
            label = match.indicator_type
        ledger.add_observation(
            Observation(
                text=f"Observed {label} indicator {match.value}.",
                kind="ioc",
                provenance=Provenance(
                    source="ioc",
                    source_name=source_name,
                    line_numbers=[match.line_number],
                    raw_excerpt=match.raw_excerpt,
                ),
                metadata={
                    "indicator_type": match.indicator_type,
                    "indicator_subtype": match.subtype,
                    "value": match.value,
                },
            )
        )
    for match in match_software_aliases(source, kb):
        identifier = f" ({match.software_id})" if match.software_id else ""
        existing = next(
            (
                item
                for item in ledger.observations
                if item.kind == "software"
                and item.metadata.get("software_id") == match.software_id
                and item.metadata.get("software_name") == match.name
            ),
            None,
        )
        observation = Observation(
            text=f"Observed ATT&CK {match.software_type} {match.name}{identifier}.",
            kind="software",
            provenance=Provenance(
                source="ioc",
                source_name=source_name,
                line_numbers=[match.line_number],
                raw_excerpt=match.raw_excerpt,
            ),
            metadata={
                "indicator_type": "software",
                "software_id": match.software_id,
                "software_name": match.name,
                "software_type": match.software_type,
                "matched_aliases": [match.matched_alias],
            },
        )
        if existing is None:
            ledger.add_observation(observation)
        else:
            index = ledger.observations.index(existing)
            merged = ledger.add_observation(observation)
            aliases = list(
                dict.fromkeys([*existing.metadata.get("matched_aliases", []), match.matched_alias])
            )
            ledger.observations[index] = merged.model_copy(
                update={"metadata": {**merged.metadata, "matched_aliases": aliases}}
            )
    return ledger.observations


def parse_iocs(*args: Any, **kwargs: Any) -> list[Observation]:
    return extract_iocs(*args, **kwargs)


def extract_indicators(*args: Any, **kwargs: Any) -> list[Observation]:
    return extract_iocs(*args, **kwargs)


def parse(*args: Any, **kwargs: Any) -> list[Observation]:
    return extract_iocs(*args, **kwargs)
