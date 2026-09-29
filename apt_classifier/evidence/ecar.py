"""Bounded parser and evidence-based correlator for eCAR JSONL records."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance

DEFAULT_MAX_RECORDS = 10_000
DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024

_ECAR_NARRATIONS: dict[tuple[str, str], str] = {
    ("PROCESS", "CREATE"): "created a process",
    ("PROCESS", "TERMINATE"): "terminated a process",
    ("PROCESS", "OPEN"): "opened a process handle",
    ("FLOW", "CONNECT"): "made a network connection",
    ("USER_SESSION", "LOGIN"): "recorded a login",
    ("USER_SESSION", "LOGOUT"): "recorded a logout",
    ("MODULE", "LOAD"): "loaded a module",
    ("FILE", "WRITE"): "wrote a file",
    ("FILE", "READ"): "read a file",
    ("FILE", "CREATE"): "created a file",
    ("FILE", "OPEN"): "opened a file",
    ("REGISTRY", "MODIFY"): "modified a registry value",
    ("THREAD", "REMOTE_CREATE"): "created a remote thread",
}


@dataclass(slots=True)
class EcarDiagnostics:
    parsed: int = 0
    skipped: int = 0
    malformed: int = 0
    correlated: int = 0
    truncated: bool = False
    input_bytes: int = 0
    source_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "skipped": self.skipped,
            "malformed": self.malformed,
            "correlated": self.correlated,
            "truncated": self.truncated,
            "input_bytes": self.input_bytes,
            "source_name": self.source_name,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class EcarParseResult:
    observations: list[Observation]
    diagnostics: EcarDiagnostics


def _value(record: Mapping[str, Any], props: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = props.get(name, record.get(name))
        if value not in (None, ""):
            return value
    return None


def _structured_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    props = record.get("properties")
    if not isinstance(props, Mapping):
        props = {}
    values = {
        "ecar_record_id": record.get("id"),
        "ecar_object": record.get("object"),
        "ecar_action": record.get("action"),
        "timestamp": _value(record, props, "timestamp", "timestamp_ms", "@timestamp"),
        "timestamp_ms": _value(record, props, "timestamp_ms"),
        "host": _value(record, props, "hostname", "host", "computer"),
        "principal": _value(record, props, "principal", "user", "username"),
        "command": _value(record, props, "command_line", "cmdline", "command"),
        "image": _value(record, props, "image_path", "image", "process_path"),
        "pid": _value(record, props, "pid"),
        "src_ip": _value(record, props, "src_ip", "source_ip"),
        "src_port": _value(record, props, "src_port", "source_port"),
        "dst_ip": _value(record, props, "dst_ip", "destination_ip"),
        "dst_port": _value(record, props, "dst_port", "destination_port"),
        "protocol": _value(record, props, "protocol", "transport"),
        "outcome": _value(record, props, "outcome", "result", "status"),
        "target": _value(record, props, "target", "target_user", "path"),
        "logon_type": _value(record, props, "logon_type"),
    }
    return {key: value for key, value in values.items() if value not in (None, "")}


def narrate_ecar_record(record: Mapping[str, Any]) -> str:
    metadata = _structured_metadata(record)
    obj = str(record.get("object", "")).upper()
    action = str(record.get("action", "")).upper()
    parts = []
    if metadata.get("host"):
        parts.append(f"[{metadata['host']}]")
    if (obj, action) == ("PROCESS", "CREATE") and (
        metadata.get("command") or metadata.get("image")
    ):
        # Command-forward narration; see winlogs._narrate_process_creation for
        # why structural boilerplate is kept out of the retrieval text.
        subject = str(metadata.get("command") or metadata.get("image"))
        program = Path(str(metadata.get("image", "")).replace("\\", "/")).name
        context = [
            item
            for item in (
                program if metadata.get("command") and program else None,
                f"run by {metadata['principal']}" if metadata.get("principal") else None,
                f"on {metadata['host']}" if metadata.get("host") else None,
            )
            if item
        ]
        return f"{subject} ({', '.join(context)})." if context else f"{subject}."
    else:
        parts.append(
            f"eCAR {obj} {action}: {_ECAR_NARRATIONS.get((obj, action), 'recorded an event')}"
        )
    if metadata.get("dst_ip") and metadata.get("dst_port") is not None:
        parts.append(f"Destination: {metadata['dst_ip']}:{metadata['dst_port']}")
    labels = (
        ("Principal", "principal"),
        ("PID", "pid"),
        ("Image", "image"),
        ("Command", "command"),
        ("SourceIp", "src_ip"),
        ("SourcePort", "src_port"),
        ("DestinationIp", "dst_ip"),
        ("DestinationPort", "dst_port"),
        ("Protocol", "protocol"),
        ("LogonType", "logon_type"),
        ("Outcome", "outcome"),
        ("Target", "target"),
    )
    parts.extend(
        f"{label}: {metadata[key]}"
        for label, key in labels
        if key in metadata
        and not (
            key in {"dst_ip", "dst_port"}
            and metadata.get("dst_ip")
            and metadata.get("dst_port") is not None
        )
    )
    return ". ".join(parts) + "."


def parse_ecar_record(
    record: Mapping[str, Any],
    *,
    source_name: str | None = None,
    line_number: int | None = None,
    raw_excerpt: str | None = None,
) -> Observation | None:
    obj = str(record.get("object", "")).upper()
    action = str(record.get("action", "")).upper()
    if not obj or not action:
        return None
    normalized = dict(record)
    normalized["object"] = obj
    normalized["action"] = action
    metadata = _structured_metadata(normalized)
    lines = [line_number] if line_number is not None else []
    return Observation(
        text=narrate_ecar_record(normalized),
        kind="ecar",
        metadata=metadata,
        provenance=Provenance(
            source="ecar",
            source_name=source_name or str(metadata.get("host") or "ecar"),
            line_numbers=lines,
            raw_excerpt=raw_excerpt,
            metadata=metadata,
        ),
    )


def _read_bytes(data: Any, maximum: int) -> tuple[bytes | None, str | None, bool]:
    inferred = None
    if isinstance(data, Path):
        inferred = data.name
        with data.open("rb") as stream:
            content = stream.read(maximum + 1)
        return content[:maximum], inferred, len(content) > maximum
    if isinstance(data, str):
        try:
            path = Path(data)
            if "\n" not in data and len(data) < 4096 and path.is_file():
                return _read_bytes(path, maximum)
        except OSError:
            pass
        content = data.encode("utf-8")
        return content[:maximum], None, len(content) > maximum
    if isinstance(data, bytes | bytearray):
        content = bytes(data)
        return content[:maximum], None, len(content) > maximum
    if hasattr(data, "getvalue"):
        content = data.getvalue()
        inferred = str(getattr(data, "name", "")) or None
        if isinstance(content, str):
            content = content.encode("utf-8")
        content = bytes(content)
        return content[:maximum], inferred, len(content) > maximum
    if hasattr(data, "read"):
        inferred = str(getattr(data, "name", "")) or None
        content = data.read(maximum + 1)
        if isinstance(content, str):
            content = content.encode("utf-8")
        content = bytes(content)
        return content[:maximum], inferred, len(content) > maximum
    return None, None, False


def _timestamp_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _correlation_observation(
    group: list[Observation], text: str, correlation_type: str, details: dict[str, Any]
) -> Observation:
    first = group[0]
    lines = sorted({line for item in group for line in item.provenance.line_numbers})
    sources = sorted({item.provenance.source_name or "ecar" for item in group})
    metadata = {"correlation_type": correlation_type, "record_count": len(group), **details}
    return Observation(
        text=text,
        kind="correlation",
        metadata=metadata,
        provenance=Provenance(
            source="ecar_correlation",
            source_name=first.provenance.source_name,
            line_numbers=lines,
            occurrence_count=len(group),
            raw_excerpts=[excerpt for item in group for excerpt in item.provenance.raw_excerpts],
            metadata={**metadata, "sources": sources},
        ),
    )


def correlate_ecar_observations(observations: list[Observation]) -> list[Observation]:
    """Add only correlations supported by repeated, sufficiently structured records."""
    additions: list[Observation] = []
    failed: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    flows: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    for obs in observations:
        meta = obs.metadata
        if meta.get("ecar_object") == "USER_SESSION" and str(
            meta.get("outcome", "")
        ).casefold() in {
            "fail",
            "failed",
            "failure",
            "denied",
            "false",
        }:
            failed[(meta.get("host"), meta.get("principal"), meta.get("src_ip"))].append(obs)
        if meta.get("ecar_object") == "FLOW" and meta.get("ecar_action") == "CONNECT":
            flows[
                (meta.get("src_ip"), meta.get("host"), meta.get("principal"), meta.get("image"))
            ].append(obs)
    for key, group in failed.items():
        if len(group) >= 3 and any(key):
            host, principal, src_ip = key
            additions.append(
                _correlation_observation(
                    group,
                    f"Observed {len(group)} failed logons for {principal or 'an account'} on {host or 'one host'} from {src_ip or 'one source'}.",
                    "repeated_failed_logons",
                    {"host": host, "principal": principal, "src_ip": src_ip},
                )
            )
    for key, group in flows.items():
        destinations = {
            item.metadata.get("dst_ip") for item in group if item.metadata.get("dst_ip")
        }
        ports = {
            item.metadata.get("dst_port")
            for item in group
            if item.metadata.get("dst_port") is not None
        }
        if len(group) >= 3 and (len(destinations) >= 3 or len(ports) >= 3):
            additions.append(
                _correlation_observation(
                    group,
                    f"Observed {len(group)} connections spanning {len(destinations)} destination hosts and {len(ports)} destination ports.",
                    "multi_host_port_connections",
                    {
                        "source": key[0] or key[1],
                        "destination_hosts": len(destinations),
                        "destination_ports": len(ports),
                    },
                )
            )
        endpoints: dict[tuple[Any, Any], list[Observation]] = defaultdict(list)
        for item in group:
            endpoints[(item.metadata.get("dst_ip"), item.metadata.get("dst_port"))].append(item)
        for endpoint, endpoint_group in endpoints.items():
            times = sorted(
                filter(
                    lambda value: value is not None,
                    (
                        _timestamp_number(
                            item.metadata.get("timestamp_ms") or item.metadata.get("timestamp")
                        )
                        for item in endpoint_group
                    ),
                )
            )
            if len(times) < 4:
                continue
            intervals = [
                right - left for left, right in zip(times, times[1:], strict=False) if right > left
            ]
            if (
                len(intervals) >= 3
                and statistics.mean(intervals) > 0
                and max(intervals) - min(intervals) <= statistics.mean(intervals) * 0.25
            ):
                additions.append(
                    _correlation_observation(
                        endpoint_group,
                        f"Observed {len(endpoint_group)} periodic connections to {endpoint[0]}:{endpoint[1]} with a median interval of {statistics.median(intervals):g} timestamp units.",
                        "periodic_network_connections",
                        {
                            "dst_ip": endpoint[0],
                            "dst_port": endpoint[1],
                            "median_interval": statistics.median(intervals),
                        },
                    )
                )
    return additions


def parse_ecar(
    data: Any,
    *,
    source_name: str | None = None,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    correlate: bool = True,
) -> EcarParseResult:
    diagnostics = EcarDiagnostics(source_name=source_name)
    observations: list[Observation] = []
    content, inferred, byte_truncated = _read_bytes(data, max_input_bytes)
    diagnostics.source_name = source_name or inferred
    diagnostics.truncated = byte_truncated
    if byte_truncated:
        diagnostics.warnings.append("input byte limit reached")
    items: Iterable[Any]
    if content is not None:
        diagnostics.input_bytes = len(content)
        items = content.decode("utf-8-sig", errors="replace").splitlines()
    elif isinstance(data, Mapping):
        items = [data]
    elif isinstance(data, Iterable):
        items = data
    else:
        raise TypeError("eCAR input must be bytes, text, a path, a file, a mapping, or records")
    for line_number, item in enumerate(items, 1):
        if diagnostics.parsed + diagnostics.skipped >= max_records:
            diagnostics.truncated = True
            diagnostics.warnings.append("record limit reached")
            break
        raw = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
        if isinstance(item, bytes | bytearray):
            raw = bytes(item).decode("utf-8-sig", errors="replace")
        if isinstance(raw, str) and not raw.strip():
            continue
        if isinstance(item, Mapping):
            record = item
        else:
            try:
                record = json.loads(str(raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                diagnostics.malformed += 1
                diagnostics.skipped += 1
                continue
        if not isinstance(record, Mapping):
            diagnostics.skipped += 1
            continue
        observation = parse_ecar_record(
            record,
            source_name=diagnostics.source_name,
            line_number=line_number,
            raw_excerpt=str(raw).strip(),
        )
        if observation is None:
            diagnostics.skipped += 1
        else:
            diagnostics.parsed += 1
            observations.append(observation)
    if correlate:
        correlations = correlate_ecar_observations(observations)
        diagnostics.correlated = len(correlations)
        observations.extend(correlations)
    return EcarParseResult(observations, diagnostics)


def parse_ecar_records(data: Any, **kwargs: Any) -> list[Observation]:
    return parse_ecar(data, **kwargs).observations


def parse_ecar_ledger(data: Any, **kwargs: Any) -> EvidenceLedger:
    return EvidenceLedger(observations=parse_ecar_records(data, **kwargs))
