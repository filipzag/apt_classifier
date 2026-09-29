import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance

SYSMON_EVENT_IDS = frozenset({1, 3, 7, 8, 10, 11, 12, 13, 14, 22, 23, 25})
SECURITY_EVENT_IDS = frozenset({4624, 4625, 4688, 4698, 4720, 7045, 1102})
SUPPORTED_EVENT_IDS = SYSMON_EVENT_IDS | SECURITY_EVENT_IDS


FieldSpec = tuple[str, tuple[str, ...]]
EventSpec = tuple[str, tuple[FieldSpec, ...]]


SYSMON_SPECS: dict[int, EventSpec] = {
    1: (
        "created a process",
        (
            ("Image", ("Image", "process.executable")),
            ("CommandLine", ("CommandLine", "process.command_line")),
            ("ParentImage", ("ParentImage", "process.parent.executable")),
            ("ParentCommandLine", ("ParentCommandLine", "process.parent.command_line")),
            ("User", ("User", "user.name")),
            ("IntegrityLevel", ("IntegrityLevel",)),
            ("Hashes", ("Hashes", "process.hash")),
        ),
    ),
    3: (
        "made a network connection",
        (
            ("Image", ("Image", "process.executable")),
            ("User", ("User", "user.name")),
            ("Protocol", ("Protocol", "network.transport")),
            ("Initiated", ("Initiated",)),
            ("SourceHostname", ("SourceHostname", "source.domain")),
            ("SourceIp", ("SourceIp", "source.ip")),
            ("SourcePort", ("SourcePort", "source.port")),
            ("DestinationHostname", ("DestinationHostname", "destination.domain")),
            ("DestinationIp", ("DestinationIp", "destination.ip")),
            ("DestinationPort", ("DestinationPort", "destination.port")),
        ),
    ),
    7: (
        "loaded an image",
        (
            ("Image", ("Image", "process.executable")),
            ("ImageLoaded", ("ImageLoaded", "file.path")),
            ("User", ("User", "user.name")),
            ("Signed", ("Signed",)),
            ("Signature", ("Signature",)),
            ("SignatureStatus", ("SignatureStatus",)),
            ("Hashes", ("Hashes", "file.hash")),
        ),
    ),
    8: (
        "created a remote thread",
        (
            ("SourceImage", ("SourceImage", "process.executable")),
            ("TargetImage", ("TargetImage",)),
            ("SourceUser", ("SourceUser", "user.name")),
            ("TargetUser", ("TargetUser",)),
            ("StartAddress", ("StartAddress",)),
            ("StartModule", ("StartModule",)),
            ("StartFunction", ("StartFunction",)),
        ),
    ),
    10: (
        "accessed process memory",
        (
            ("SourceImage", ("SourceImage", "process.executable")),
            ("TargetImage", ("TargetImage",)),
            ("SourceUser", ("SourceUser", "user.name")),
            ("TargetUser", ("TargetUser",)),
            ("GrantedAccess", ("GrantedAccess",)),
        ),
    ),
    11: (
        "created a file",
        (
            ("Image", ("Image", "process.executable")),
            ("TargetFilename", ("TargetFilename", "file.path")),
            ("User", ("User", "user.name")),
        ),
    ),
    12: (
        "created or deleted a registry object",
        (
            ("EventType", ("EventType",)),
            ("Image", ("Image", "process.executable")),
            ("TargetObject", ("TargetObject", "registry.path")),
            ("User", ("User", "user.name")),
        ),
    ),
    13: (
        "set a registry value",
        (
            ("EventType", ("EventType",)),
            ("Image", ("Image", "process.executable")),
            ("TargetObject", ("TargetObject", "registry.path")),
            ("Details", ("Details", "registry.data.strings")),
            ("User", ("User", "user.name")),
        ),
    ),
    14: (
        "renamed a registry object",
        (
            ("EventType", ("EventType",)),
            ("Image", ("Image", "process.executable")),
            ("TargetObject", ("TargetObject", "registry.path")),
            ("NewName", ("NewName",)),
            ("User", ("User", "user.name")),
        ),
    ),
    22: (
        "queried DNS",
        (
            ("Image", ("Image", "process.executable")),
            ("QueryName", ("QueryName", "dns.question.name")),
            ("QueryStatus", ("QueryStatus",)),
            ("QueryResults", ("QueryResults", "dns.answers.data")),
            ("User", ("User", "user.name")),
        ),
    ),
    23: (
        "deleted a file",
        (
            ("Image", ("Image", "process.executable")),
            ("TargetFilename", ("TargetFilename", "file.path")),
            ("User", ("User", "user.name")),
            ("Hashes", ("Hashes", "file.hash")),
            ("Archived", ("Archived",)),
        ),
    ),
    25: (
        "detected process tampering",
        (
            ("Image", ("Image", "process.executable")),
            ("Type", ("Type",)),
            ("User", ("User", "user.name")),
        ),
    ),
}


SECURITY_SPECS: dict[int, EventSpec] = {
    4624: (
        "recorded a successful logon",
        (
            ("TargetUser", ("TargetUserName", "user.name")),
            ("TargetDomain", ("TargetDomainName", "user.domain")),
            ("LogonType", ("LogonType", "winlog.logon.type")),
            ("IpAddress", ("IpAddress", "source.ip")),
            ("IpPort", ("IpPort", "source.port")),
            ("WorkstationName", ("WorkstationName", "source.domain")),
            ("ProcessName", ("ProcessName", "process.executable")),
            ("AuthenticationPackage", ("AuthenticationPackageName",)),
            ("ElevatedToken", ("ElevatedToken",)),
            ("SubjectUser", ("SubjectUserName",)),
        ),
    ),
    4625: (
        "recorded a failed logon",
        (
            ("TargetUser", ("TargetUserName", "user.name")),
            ("TargetDomain", ("TargetDomainName", "user.domain")),
            ("LogonType", ("LogonType", "winlog.logon.type")),
            ("IpAddress", ("IpAddress", "source.ip")),
            ("IpPort", ("IpPort", "source.port")),
            ("WorkstationName", ("WorkstationName", "source.domain")),
            ("ProcessName", ("ProcessName", "process.executable")),
            ("FailureReason", ("FailureReason",)),
            ("Status", ("Status",)),
            ("SubStatus", ("SubStatus",)),
            ("AuthenticationPackage", ("AuthenticationPackageName",)),
        ),
    ),
    4688: (
        "created a process",
        (
            ("NewProcessName", ("NewProcessName", "process.executable")),
            ("CommandLine", ("CommandLine", "ProcessCommandLine", "process.command_line")),
            ("ParentProcessName", ("ParentProcessName", "process.parent.executable")),
            ("SubjectUser", ("SubjectUserName", "user.name")),
            ("SubjectDomain", ("SubjectDomainName", "user.domain")),
            ("TargetUser", ("TargetUserName",)),
            ("TokenElevationType", ("TokenElevationType",)),
            ("MandatoryLabel", ("MandatoryLabel",)),
        ),
    ),
    4698: (
        "created a scheduled task",
        (
            ("TaskName", ("TaskName",)),
            ("TaskContent", ("TaskContent",)),
            ("SubjectUser", ("SubjectUserName", "user.name")),
            ("SubjectDomain", ("SubjectDomainName", "user.domain")),
            ("ClientProcessId", ("ClientProcessId",)),
            ("ParentProcessId", ("ParentProcessId",)),
        ),
    ),
    4720: (
        "created a user account",
        (
            ("TargetUser", ("TargetUserName", "user.name")),
            ("TargetDomain", ("TargetDomainName", "user.domain")),
            ("SamAccountName", ("SamAccountName",)),
            ("UserPrincipalName", ("UserPrincipalName",)),
            ("DisplayName", ("DisplayName",)),
            ("SubjectUser", ("SubjectUserName",)),
            ("SubjectDomain", ("SubjectDomainName",)),
        ),
    ),
    7045: (
        "installed a service",
        (
            ("ServiceName", ("ServiceName", "winlog.event_data.param1")),
            ("ImagePath", ("ImagePath", "ServiceFileName", "winlog.event_data.param2")),
            ("ServiceType", ("ServiceType", "winlog.event_data.param3")),
            ("StartType", ("StartType", "ServiceStartType", "winlog.event_data.param4")),
            ("AccountName", ("AccountName", "ServiceAccount", "winlog.event_data.param5")),
            ("SubjectUser", ("SubjectUserName", "user.name")),
        ),
    ),
    1102: (
        "cleared the audit log",
        (
            ("SubjectUser", ("SubjectUserName", "user.name")),
            ("SubjectDomain", ("SubjectDomainName", "user.domain")),
            ("SubjectLogonId", ("SubjectLogonId",)),
        ),
    ),
}


class WinlogParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WinlogRecord:
    event: Mapping[str, Any]
    line_number: int
    raw_excerpt: str


@dataclass(slots=True)
class WinlogDiagnostics:
    parsed: int = 0
    skipped: int = 0
    malformed: int = 0
    correlated: int = 0
    truncated: bool = False
    input_bytes: int = 0
    source_name: str | None = None
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "skipped": self.skipped,
            "malformed": self.malformed,
            "correlated": self.correlated,
            "truncated": self.truncated,
            "input_bytes": self.input_bytes,
            "source_name": self.source_name,
            "warnings": list(self.warnings or []),
        }


@dataclass(slots=True)
class WinlogParseResult:
    observations: list[Observation]
    diagnostics: WinlogDiagnostics


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _clean(value: Any) -> str:
    result = re.sub(r"\s+", " ", str(value)).strip()
    return result


def _mapping_scalar(value: Mapping[str, Any]) -> Any:
    normalized = {_key(str(key)): item for key, item in value.items()}
    for name in ("text", "value", "content"):
        if name in normalized and not isinstance(normalized[name], Mapping | list | tuple):
            return normalized[name]
    if len(value) == 1:
        item = next(iter(value.values()))
        if not isinstance(item, Mapping | list | tuple):
            return item
    return None


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        selected = _mapping_scalar(value)
        if selected is None:
            return None
        return _scalar(selected)
    if isinstance(value, list | tuple | set):
        items = [item for item in (_scalar(part) for part in value) if item]
        return ", ".join(dict.fromkeys(items)) or None
    result = _clean(value)
    return result or None


class _FieldBag:
    def __init__(self, event: Mapping[str, Any]) -> None:
        self.values: dict[str, str] = {}
        self._event_data(event)
        self._walk(event)

    def _add(self, name: str, value: Any) -> None:
        scalar = _scalar(value)
        if not scalar:
            return
        full = _key(name)
        leaf = _key(re.split(r"[./]", name)[-1])
        if full:
            self.values.setdefault(full, scalar)
        if leaf:
            self.values.setdefault(leaf, scalar)

    def _event_data(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                name = str(raw_key)
                normalized = _key(name)
                if normalized == "data":
                    self._data_items(item)
                elif normalized in {"eventdata", "userdata"}:
                    self._named_values(item)
                elif normalized.startswith("winlogeventdata"):
                    suffix = re.split(r"winlog[._/]event_data[._/]", name, flags=re.I)
                    self._add(suffix[-1], item)
                self._event_data(item)
        elif isinstance(value, list | tuple):
            for item in value:
                self._event_data(item)

    def _named_values(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            self._data_items(value)
            return
        for name, item in value.items():
            if _key(str(name)) == "data":
                self._data_items(item)
            else:
                scalar = _scalar(item)
                if scalar:
                    self._add(str(name), scalar)
                self._named_values(item)

    def _data_items(self, value: Any) -> None:
        items = value if isinstance(value, list | tuple) else [value]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            normalized = {_key(str(name)): part for name, part in item.items()}
            name = normalized.get("name")
            attributes = normalized.get("attributes")
            if name is None and isinstance(attributes, Mapping):
                attribute_values = {_key(str(key)): part for key, part in attributes.items()}
                name = attribute_values.get("name")
            if name is None:
                continue
            selected = None
            for candidate in ("text", "value", "content"):
                if candidate in normalized:
                    selected = normalized[candidate]
                    break
            if selected is None:
                selected = _mapping_scalar(item)
            self._add(str(name), selected)

    def _walk(self, value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            scalar = _mapping_scalar(value)
            if path and scalar is not None:
                self._add(path, scalar)
            for name, item in value.items():
                child = f"{path}.{name}" if path else str(name)
                self._walk(item, child)
        elif isinstance(value, list | tuple):
            scalar = _scalar(value)
            if path and scalar:
                self._add(path, scalar)
            for item in value:
                self._walk(item, path)
        elif path:
            self._add(path, value)

    def get(self, *names: str) -> str | None:
        for name in names:
            value = self.values.get(_key(name))
            if value is not None:
                return value
        return None


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _event_context(event: Mapping[str, Any]) -> tuple[int | None, _FieldBag, dict[str, str]]:
    bag = _FieldBag(event)
    event_id = _integer(
        bag.get(
            "winlog.event_id",
            "Event.System.EventID",
            "System.EventID",
            "EventID",
            "event_id",
            "event.code",
            "eventcode",
        )
    )
    metadata = {
        key: value
        for key, value in {
            "provider": bag.get(
                "winlog.provider_name",
                "Event.System.Provider.Name",
                "System.Provider.Name",
                "Event.System.Provider.Attributes.Name",
                "System.Provider.Attributes.Name",
                "Event.System.Provider",
                "System.Provider",
                "ProviderName",
                "Provider",
                "SourceName",
            ),
            "channel": bag.get(
                "winlog.channel", "Event.System.Channel", "System.Channel", "Channel"
            ),
            "computer": bag.get(
                "winlog.computer_name",
                "Event.System.Computer",
                "System.Computer",
                "Computer",
                "host.name",
            ),
            "timestamp": bag.get(
                "UtcTime",
                "@timestamp",
                "Event.System.TimeCreated.SystemTime",
                "System.TimeCreated.SystemTime",
                "Event.System.TimeCreated.Attributes.SystemTime",
                "System.TimeCreated.Attributes.SystemTime",
                "TimeCreated",
            ),
            "record_id": bag.get(
                "winlog.record_id",
                "Event.System.EventRecordID",
                "System.EventRecordID",
                "EventRecordID",
            ),
        }.items()
        if value is not None
    }
    return event_id, bag, metadata


def narrate_event(event: Mapping[str, Any]) -> str | None:
    event_id, bag, metadata = _event_context(event)
    if event_id not in SUPPORTED_EVENT_IDS:
        return None
    if event_id in SYSMON_EVENT_IDS:
        marker = f"{metadata.get('provider', '')} {metadata.get('channel', '')}".strip()
        if marker and "sysmon" not in marker.casefold():
            return None
        family = "Sysmon"
        action, fields = SYSMON_SPECS[event_id]
        if event_id == 10 and "lsass" in (bag.get("TargetImage") or "").casefold():
            action = "accessed LSASS memory"
    else:
        family = "Windows"
        action, fields = SECURITY_SPECS[event_id]
    computer = metadata.get("computer")
    if event_id in PROCESS_CREATION_EVENT_IDS:
        return _narrate_process_creation(family, event_id, bag, computer)
    values: list[str] = []
    if computer:
        values.append(f"Computer={computer}")
    for label, aliases in fields:
        value = bag.get(*aliases)
        if value:
            values.append(f"{label}={value}")
    details = f": {'; '.join(values)}" if values else ""
    return f"{family} Event {event_id} {action}{details}."


PROCESS_CREATION_EVENT_IDS = frozenset({1, 4688})


def _narrate_process_creation(family: str, event_id: int, bag: Any, computer: str | None) -> str:
    """Command-forward narration for process creation.

    Process-creation records are the most common structured evidence.  The
    retrieval text is the command line (the actual behavior) plus the program
    name, user, and host.  Structural boilerplate such as the event family,
    event ID, and parent image is deliberately left to metadata/provenance:
    those tokens ("Windows Event", "image", "parent process", "binary") match
    unrelated ATT&CK descriptions and crowd out the behavior itself.
    """
    image = bag.get("Image", "NewProcessName", "process.executable")
    command = bag.get("CommandLine", "ProcessCommandLine", "process.command_line")
    user = bag.get("User", "SubjectUserName", "user.name")
    program = Path(str(image).replace("\\", "/")).name if image else None
    subject = command or image
    if not subject:
        return f"{family} Event {event_id}: process created without a recorded command line."
    context = [
        item
        for item in (
            program if command and program else None,
            f"run by {user}" if user else None,
            f"on {computer}" if computer else None,
        )
        if item
    ]
    return f"{subject} ({', '.join(context)})." if context else f"{subject}."


def parse_winlog_event(
    event: Mapping[str, Any],
    *,
    line_number: int = 1,
    raw_excerpt: str | None = None,
    source_name: str | None = None,
) -> Observation | None:
    event_id, bag, metadata = _event_context(event)
    narration = narrate_event(event)
    if event_id is None or narration is None:
        return None
    raw = raw_excerpt or json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
    family = "sysmon" if event_id in SYSMON_EVENT_IDS else "windows_security"
    structured = {
        key: value
        for key, value in {
            "host": metadata.get("computer"),
            "timestamp": metadata.get("timestamp"),
            "principal": bag.get("TargetUserName", "SubjectUserName", "User", "user.name"),
            "command": bag.get("CommandLine", "ProcessCommandLine", "process.command_line"),
            "image": bag.get("Image", "NewProcessName", "ProcessName", "process.executable"),
            "src_ip": bag.get("SourceIp", "IpAddress", "source.ip"),
            "src_port": bag.get("SourcePort", "IpPort", "source.port"),
            "dst_ip": bag.get("DestinationIp", "destination.ip"),
            "dst_port": bag.get("DestinationPort", "destination.port"),
            "protocol": bag.get("Protocol", "network.transport"),
            "logon_type": bag.get("LogonType", "winlog.logon.type"),
        }.items()
        if value is not None
    }
    details: dict[str, Any] = {
        "event_id": event_id,
        "event_family": family,
        **metadata,
        **structured,
    }
    return Observation(
        text=narration,
        kind="winlog",
        provenance=Provenance(
            source="winlog",
            source_name=source_name,
            line_numbers=[line_number],
            raw_excerpt=raw,
            event_id=event_id,
            metadata=details,
        ),
        metadata=details,
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _looks_like_event(value: Mapping[str, Any]) -> bool:
    keys = {_key(str(name)) for name in value}
    if keys & {"event", "eventid", "eventcode", "wineventid", "winlogeventid"}:
        return True
    if any("eventid" in key for key in keys):
        return True
    event_id, _, _ = _event_context(value)
    return event_id is not None


def _expand(value: Any, line_number: int, raw: str) -> list[WinlogRecord]:
    if isinstance(value, list):
        records: list[WinlogRecord] = []
        for item in value:
            records.extend(_expand(item, line_number, _json(item)))
        return records
    if not isinstance(value, Mapping):
        return []
    source = value.get("_source")
    if isinstance(source, Mapping):
        return _expand(source, line_number, _json(source))
    hits = value.get("hits")
    if isinstance(hits, Mapping) and isinstance(hits.get("hits"), list):
        return _expand(hits["hits"], line_number, raw)
    for name in ("events", "Events", "records", "Records", "items", "results"):
        items = value.get(name)
        if isinstance(items, list):
            return _expand(items, line_number, raw)
    if _looks_like_event(value):
        return [WinlogRecord(value, line_number, raw)]
    return []


def _array_records(text: str, start: int) -> list[WinlogRecord]:
    decoder = json.JSONDecoder()
    position = start + 1
    records: list[WinlogRecord] = []
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == ",":
            position += 1
            continue
        if position < len(text) and text[position] == "]":
            return records
        if position >= len(text):
            raise WinlogParseError("unterminated JSON array")
        item_start = position
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            line = text.count("\n", 0, exc.pos) + 1
            raise WinlogParseError(f"invalid JSON at line {line}: {exc.msg}") from exc
        line_number = text.count("\n", 0, item_start) + 1
        records.extend(_expand(value, line_number, text[item_start:position].strip()))


def _wrapped_array_records(text: str, value: Any) -> list[WinlogRecord] | None:
    if not isinstance(value, Mapping):
        return None
    keys = [
        name
        for name in ("events", "Events", "records", "Records", "items", "results")
        if isinstance(value.get(name), list)
    ]
    hits = value.get("hits")
    if isinstance(hits, Mapping) and isinstance(hits.get("hits"), list):
        keys.append("hits")
    for name in keys:
        pattern = re.compile(rf'(?<!\\)"{re.escape(name)}"\s*:\s*\[')
        for match in pattern.finditer(text):
            records = _array_records(text, match.end() - 1)
            if records:
                return records
    return [] if keys else None


def _text_records(text: str) -> list[WinlogRecord]:
    if not text.strip():
        return []
    if text.startswith("\ufeff"):
        text = text[1:]
    offset = len(text) - len(text.lstrip(" \t\r\n"))
    first = text[offset : offset + 1]
    if first == "[":
        return _array_records(text, offset)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[WinlogRecord] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WinlogParseError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            records.extend(_expand(value, line_number, line.strip()))
        return records
    wrapped = _wrapped_array_records(text, value)
    if wrapped is not None:
        return wrapped
    line_number = text.count("\n", 0, offset) + 1
    return _expand(value, line_number, text.strip())


def load_winlog_records(data: Any) -> tuple[list[WinlogRecord], str | None]:
    inferred_name: str | None = None
    if isinstance(data, Path):
        inferred_name = str(data)
        return _text_records(data.read_text(encoding="utf-8-sig")), inferred_name
    if isinstance(data, bytes | bytearray):
        return _text_records(bytes(data).decode("utf-8-sig")), inferred_name
    if isinstance(data, str):
        candidate: Path | None = None
        if "\n" not in data and len(data) < 4096 and data[:1] not in "[{":
            try:
                candidate = Path(data)
                if not candidate.is_file():
                    candidate = None
            except OSError:
                candidate = None
        if candidate is not None:
            inferred_name = str(candidate)
            return _text_records(candidate.read_text(encoding="utf-8-sig")), inferred_name
        return _text_records(data), inferred_name
    if hasattr(data, "read"):
        value = data.read()
        if isinstance(value, bytes):
            value = value.decode("utf-8-sig")
        inferred_name = str(getattr(data, "name", "")) or None
        return _text_records(str(value)), inferred_name
    if isinstance(data, Mapping):
        return _expand(data, 1, _json(data)), inferred_name
    if isinstance(data, Iterable):
        records: list[WinlogRecord] = []
        for line_number, item in enumerate(data, 1):
            if isinstance(item, str):
                try:
                    value = json.loads(item)
                except json.JSONDecodeError as exc:
                    raise WinlogParseError(
                        f"invalid JSON in item {line_number}: {exc.msg}"
                    ) from exc
                records.extend(_expand(value, line_number, item))
            else:
                records.extend(_expand(item, line_number, _json(item)))
        return records, inferred_name
    raise TypeError("winlog input must be JSON text, bytes, a path, a file, or event records")


def _correlate_winlogs(observations: list[Observation]) -> list[Observation]:
    from collections import defaultdict
    from datetime import datetime

    groups: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    flows: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    for observation in observations:
        meta = observation.metadata
        if meta.get("event_id") == 4625:
            groups[(meta.get("host"), meta.get("principal"), meta.get("src_ip"))].append(
                observation
            )
        if meta.get("event_id") == 3:
            flows[
                (meta.get("src_ip"), meta.get("host"), meta.get("image"), meta.get("principal"))
            ].append(observation)

    def correlated(
        group: list[Observation], text: str, kind: str, details: dict[str, Any]
    ) -> Observation:
        first = group[0]
        lines = sorted({line for item in group for line in item.provenance.line_numbers})
        metadata = {"correlation_type": kind, "record_count": len(group), **details}
        return Observation(
            text=text,
            kind="correlation",
            metadata=metadata,
            provenance=Provenance(
                source="winlog_correlation",
                source_name=first.provenance.source_name,
                line_numbers=lines,
                occurrence_count=len(group),
                raw_excerpts=[
                    excerpt for item in group for excerpt in item.provenance.raw_excerpts
                ],
                metadata=metadata,
            ),
        )

    result: list[Observation] = []
    for key, group in groups.items():
        if len(group) >= 3 and any(key):
            host, principal, source = key
            result.append(
                correlated(
                    group,
                    f"Observed {len(group)} failed logons for {principal or 'an account'} on {host or 'one host'} from {source or 'one source'}.",
                    "repeated_failed_logons",
                    {"host": host, "principal": principal, "src_ip": source},
                )
            )
    for group in flows.values():
        hosts = {item.metadata.get("dst_ip") for item in group if item.metadata.get("dst_ip")}
        ports = {item.metadata.get("dst_port") for item in group if item.metadata.get("dst_port")}
        if len(group) >= 3 and (len(hosts) >= 3 or len(ports) >= 3):
            result.append(
                correlated(
                    group,
                    f"Observed {len(group)} connections spanning {len(hosts)} destination hosts and {len(ports)} destination ports.",
                    "multi_host_port_connections",
                    {"destination_hosts": len(hosts), "destination_ports": len(ports)},
                )
            )
        endpoints: dict[tuple[Any, Any], list[tuple[float, Observation]]] = defaultdict(list)
        for item in group:
            raw_time = item.metadata.get("timestamp")
            try:
                timestamp = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError):
                continue
            endpoints[(item.metadata.get("dst_ip"), item.metadata.get("dst_port"))].append(
                (timestamp, item)
            )
        for endpoint, timed in endpoints.items():
            timed.sort(key=lambda pair: pair[0])
            intervals = [right[0] - left[0] for left, right in zip(timed, timed[1:], strict=False)]
            if len(intervals) >= 3 and min(intervals) > 0:
                average = sum(intervals) / len(intervals)
                if max(intervals) - min(intervals) <= average * 0.25:
                    endpoint_group = [item for _, item in timed]
                    result.append(
                        correlated(
                            endpoint_group,
                            f"Observed {len(timed)} periodic connections to {endpoint[0]}:{endpoint[1]} with an average interval of {average:g} seconds.",
                            "periodic_network_connections",
                            {
                                "dst_ip": endpoint[0],
                                "dst_port": endpoint[1],
                                "average_interval_seconds": average,
                            },
                        )
                    )
    return result


def _bounded_data(data: Any, maximum: int) -> tuple[Any, int, bool]:
    if isinstance(data, Path):
        with data.open("rb") as stream:
            content = stream.read(maximum + 1)
        return content[:maximum], min(len(content), maximum), len(content) > maximum
    if isinstance(data, str):
        try:
            if "\n" not in data and len(data) < 4096 and Path(data).is_file():
                return _bounded_data(Path(data), maximum)
        except OSError:
            pass
        encoded = data.encode("utf-8")
        return encoded[:maximum], min(len(encoded), maximum), len(encoded) > maximum
    if isinstance(data, bytes | bytearray):
        content = bytes(data)
        return content[:maximum], min(len(content), maximum), len(content) > maximum
    if hasattr(data, "getvalue"):
        content = data.getvalue()
        if isinstance(content, str):
            content = content.encode("utf-8")
        content = bytes(content)
        return content[:maximum], min(len(content), maximum), len(content) > maximum
    if hasattr(data, "read"):
        content = data.read(maximum + 1)
        if isinstance(content, str):
            content = content.encode("utf-8")
        content = bytes(content)
        return content[:maximum], min(len(content), maximum), len(content) > maximum
    return data, 0, False


def parse_winlogs_detailed(
    data: Any,
    *,
    source_name: str | None = None,
    max_records: int = 10_000,
    max_input_bytes: int = 16 * 1024 * 1024,
    correlate: bool = True,
) -> WinlogParseResult:
    bounded, input_bytes, truncated = _bounded_data(data, max_input_bytes)
    inferred = str(getattr(data, "name", "")) or None
    if isinstance(data, Path):
        inferred = data.name
    diagnostics = WinlogDiagnostics(
        source_name=source_name or inferred,
        input_bytes=input_bytes,
        truncated=truncated,
        warnings=["input byte limit reached"] if truncated else [],
    )
    try:
        records, loaded_name = load_winlog_records(bounded)
    except WinlogParseError:
        # JSONL is recoverable line-by-line: count malformed lines and retain valid records.
        text = (
            bytes(bounded).decode("utf-8-sig", errors="replace")
            if isinstance(bounded, bytes | bytearray)
            else str(bounded)
        )
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.extend(_expand(json.loads(line), line_number, line.strip()))
            except (json.JSONDecodeError, TypeError):
                diagnostics.malformed += 1
                diagnostics.skipped += 1
        loaded_name = None
    if len(records) > max_records:
        records = records[:max_records]
        diagnostics.truncated = True
        diagnostics.warnings = [*(diagnostics.warnings or []), "record limit reached"]
    ledger = EvidenceLedger()
    parsed_observations: list[Observation] = []
    for record in records:
        observation = parse_winlog_event(
            record.event,
            line_number=record.line_number,
            raw_excerpt=record.raw_excerpt,
            source_name=source_name or inferred or loaded_name,
        )
        if observation is None:
            diagnostics.skipped += 1
        else:
            diagnostics.parsed += 1
            parsed_observations.append(observation)
            ledger.add_observation(observation)
    observations = ledger.observations
    if correlate:
        correlations = _correlate_winlogs(parsed_observations)
        diagnostics.correlated = len(correlations)
        observations.extend(correlations)
    return WinlogParseResult(observations, diagnostics)


def parse_winlogs(data: Any, **kwargs: Any) -> list[Observation]:
    return parse_winlogs_detailed(data, **kwargs).observations


def _xml_namespace(tag: str) -> str:
    """Strip XML namespace from a tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_windows_xml_legacy(
    data: Any,
    *,
    source_name: str | None = None,
) -> list[Observation]:
    """Parse Windows Event XML into Observations.

    Accepts:
    - A file path (str or Path) to an XML file
    - A string containing XML
    - A file-like object

    Uses iterparse for streaming to handle large XML files efficiently.
    Each ``<Event>`` element is converted to a flat dict and then passed
    through the standard parse_winlog_event pipeline.
    """
    import xml.etree.ElementTree as ET

    observations: list[Observation] = []

    def _event_to_dict(event_elem: Any) -> dict[str, Any]:
        """Convert an XML Event element to a flat dict."""
        record: dict[str, Any] = {}
        # Find System and EventData by local name (namespace-agnostic)
        system = None
        event_data = None
        for child in event_elem:
            tag = _xml_namespace(child.tag)
            if tag == "System":
                system = child
            elif tag == "EventData":
                event_data = child
        if system is not None:
            for child in system:
                tag = _xml_namespace(child.tag)
                if tag == "EventID":
                    try:
                        record["EventID"] = int(child.text or "0")
                    except ValueError:
                        record["EventID"] = 0
                elif tag == "Provider":
                    record["ProviderName"] = child.get("Name", "")
                elif tag == "Channel":
                    record["Channel"] = child.text or ""
                elif tag == "Computer":
                    record["Computer"] = child.text or ""
                elif tag == "TimeCreated":
                    record["TimeCreated"] = child.get("SystemTime", "")
                elif tag == "Correlation":
                    activity_id = child.get("ActivityID")
                    if activity_id:
                        record["ActivityID"] = activity_id
                elif tag == "Security":
                    user_id = child.get("UserID")
                    if user_id:
                        record["UserID"] = user_id
                elif child.text and child.text.strip():
                    record[tag] = child.text.strip()
        if event_data is not None:
            for data_elem in event_data:
                name = data_elem.get("Name", "")
                text = data_elem.text or ""
                if name:
                    record[name] = text
        return record

    # Handle file path
    if isinstance(data, str | Path):
        path = Path(data)
        if path.exists():
            src_name = source_name or path.name
            # Use iterparse for streaming. We do NOT clear child elements
            # during iteration because clearing a child (e.g., System)
            # removes its content before the parent (Event) is processed.
            # Instead, we only clear the Event element after processing.
            context = ET.iterparse(str(path), events=("end",))
            for _, elem in context:
                if _xml_namespace(elem.tag) == "Event":
                    record = _event_to_dict(elem)
                    obs = parse_winlog_event(record, source_name=src_name)
                    if obs is not None:
                        observations.append(obs)
                    elem.clear()  # Free memory after processing
            return observations
        # Treat as XML string
        data = str(data)

    if isinstance(data, str):
        import io

        src_name = source_name or "windows_xml"
        context = ET.iterparse(io.BytesIO(data.encode("utf-8")), events=("end",))
        for _, elem in context:
            if _xml_namespace(elem.tag) == "Event":
                record = _event_to_dict(elem)
                obs = parse_winlog_event(record, source_name=src_name)
                if obs is not None:
                    observations.append(obs)
                elem.clear()
        return observations

    return observations


def parse_windows_xml_detailed(
    data: Any,
    *,
    source_name: str | None = None,
    max_records: int = 10_000,
    max_input_bytes: int = 16 * 1024 * 1024,
    correlate: bool = True,
) -> WinlogParseResult:
    """Stream Windows XML with byte/record limits and recoverable diagnostics."""
    import io
    import xml.etree.ElementTree as ET

    inferred = str(getattr(data, "name", "")) or None
    close_stream = False
    path: Path | None = data if isinstance(data, Path) else None
    if isinstance(data, str) and "<" not in data[:100] and len(data) < 4096:
        try:
            candidate = Path(data)
            if candidate.is_file():
                path = candidate
        except OSError:
            pass
    if path is not None:
        inferred = path.name
        size = path.stat().st_size
        stream: Any = path.open("rb")
        close_stream = True
        input_bytes = min(size, max_input_bytes)
        byte_truncated = size > max_input_bytes
    else:
        if isinstance(data, str):
            content = data.encode("utf-8")
        elif isinstance(data, bytes | bytearray):
            content = bytes(data)
        elif hasattr(data, "getvalue"):
            content = data.getvalue()
            if isinstance(content, str):
                content = content.encode("utf-8")
            content = bytes(content)
        elif hasattr(data, "read"):
            content = data.read(max_input_bytes + 1)
            if isinstance(content, str):
                content = content.encode("utf-8")
            content = bytes(content)
        else:
            raise TypeError("Windows XML input must be bytes, text, a path, or a file")
        byte_truncated = len(content) > max_input_bytes
        content = content[:max_input_bytes]
        input_bytes = len(content)
        stream = io.BytesIO(content)

    class BoundedReader:
        def __init__(self, wrapped: Any, remaining: int) -> None:
            self.wrapped = wrapped
            self.remaining = remaining

        def read(self, size: int = -1) -> bytes:
            if self.remaining <= 0:
                return b""
            requested = self.remaining if size < 0 else min(size, self.remaining)
            value = self.wrapped.read(requested)
            self.remaining -= len(value)
            return value

    diagnostics = WinlogDiagnostics(
        source_name=source_name or inferred or "windows_xml",
        input_bytes=input_bytes,
        truncated=byte_truncated,
        warnings=["input byte limit reached"] if byte_truncated else [],
    )
    observations: list[Observation] = []

    def event_dict(event_elem: Any) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for child in event_elem:
            section = _xml_namespace(child.tag)
            if section == "System":
                for field in child:
                    tag = _xml_namespace(field.tag)
                    if tag == "Provider":
                        record["ProviderName"] = field.get("Name", "")
                    elif tag == "TimeCreated":
                        record["TimeCreated"] = field.get("SystemTime", "")
                    elif tag == "EventID":
                        record["EventID"] = field.text or ""
                    elif tag in {"Channel", "Computer", "EventRecordID"}:
                        record[tag] = field.text or ""
            elif section in {"EventData", "UserData"}:
                for field in child.iter():
                    if field is child:
                        continue
                    name = field.get("Name") or _xml_namespace(field.tag)
                    if name and field.text:
                        record[name] = field.text.strip()
        return record

    try:
        reader = BoundedReader(stream, max_input_bytes)
        for _, elem in ET.iterparse(reader, events=("end",)):
            if _xml_namespace(elem.tag) != "Event":
                continue
            if diagnostics.parsed + diagnostics.skipped >= max_records:
                diagnostics.truncated = True
                diagnostics.warnings = [*(diagnostics.warnings or []), "record limit reached"]
                break
            record = event_dict(elem)
            observation = parse_winlog_event(
                record,
                line_number=diagnostics.parsed + diagnostics.skipped + 1,
                raw_excerpt=ET.tostring(elem, encoding="unicode"),
                source_name=diagnostics.source_name,
            )
            if observation is None:
                diagnostics.skipped += 1
            else:
                diagnostics.parsed += 1
                observations.append(observation)
            elem.clear()
    except ET.ParseError as exc:
        diagnostics.malformed += 1
        diagnostics.skipped += 1
        diagnostics.warnings = [*(diagnostics.warnings or []), f"XML parse error: {exc}"]
    finally:
        if close_stream:
            stream.close()
    if correlate:
        correlations = _correlate_winlogs(observations)
        diagnostics.correlated = len(correlations)
        observations.extend(correlations)
    return WinlogParseResult(observations, diagnostics)


def parse_windows_xml(data: Any, **kwargs: Any) -> list[Observation]:
    return parse_windows_xml_detailed(data, **kwargs).observations


def parse_windows_xml_ledger(data: Any, **kwargs: Any) -> EvidenceLedger:
    return EvidenceLedger(observations=parse_windows_xml(data, **kwargs))


def parse_windows_events(data: Any, *, source_name: str | None = None) -> list[Observation]:
    return parse_winlogs(data, source_name=source_name)


def parse_winlog(data: Any, *, source_name: str | None = None) -> list[Observation]:
    return parse_winlogs(data, source_name=source_name)


def parse_winlog_ledger(data: Any, *, source_name: str | None = None) -> EvidenceLedger:
    return EvidenceLedger(observations=parse_winlogs(data, source_name=source_name))


def parse(data: Any, *, source_name: str | None = None) -> list[Observation]:
    return parse_winlogs(data, source_name=source_name)
