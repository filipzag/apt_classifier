import json
import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance, StrategicContext
from apt_classifier.llm import OpenRouterClient, supports_seed

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "source_lines": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "raw_excerpt": {"type": "string"},
                },
                "required": ["sentence", "source_lines", "raw_excerpt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """Normalize the supplied incident evidence into atomic attack-action sentences. Each sentence must state exactly one concrete action, its actor or process when available, and its object or target when available. Keep executable names, command lines, paths, hosts, accounts, protocols, and security-relevant qualifiers verbatim. Split compound behavior into separate actions. Report only actions directly supported by the source. Do not infer intent, identity, attribution, malware families, ATT&CK techniques, or missing details. Ignore recommendations, product boilerplate, severity labels, and duplicated summaries. Use concise declarative sentences. For every action, return the one-based source line numbers and an exact supporting excerpt."""


_SOURCE_GUIDANCE = {
    "free_text": "Treat the source as analyst observations and incident-report prose.",
    "alert": "Treat the source as EDR or SIEM alert evidence; extract behavior, not alert metadata.",
}


class ExtractionError(RuntimeError):
    pass


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sentence(value: Any) -> str:
    result = _compact(str(value or ""))
    if result and result[-1] not in ".!?":
        result += "."
    return result


def _numbered(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines() or [text]
    return "\n".join(f"{index}: {line}" for index, line in enumerate(lines, 1)), lines


def _settings_value(client: Any, name: str, fallback: str) -> str:
    settings = getattr(client, "settings", None)
    value = getattr(settings, name, None)
    return str(value or fallback)


def _gpt_seed(model: str, seed: int | None) -> int | None:
    return seed if seed is not None and supports_seed(model) else None


def _action_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        raise ExtractionError("OpenRouter structured extraction did not return an object")
    actions = parsed.get("actions")
    if actions is None:
        actions = parsed.get("attack_actions")
    if actions is None:
        actions = parsed.get("observations")
    if not isinstance(actions, list):
        raise ExtractionError("OpenRouter structured extraction omitted the actions array")
    return actions


def _source_lines(value: Any, line_count: int) -> list[int]:
    if value is None:
        return []
    values = [value] if isinstance(value, int | str) else value
    if not isinstance(values, list | tuple | set):
        return []
    result: set[int] = set()
    for item in values:
        try:
            line = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= line <= line_count:
            result.add(line)
    return sorted(result)


def _excerpt(raw: Any, lines: list[str], line_numbers: list[int], source: str) -> str:
    selected = "\n".join(lines[line - 1] for line in line_numbers).strip()
    supplied = str(raw or "").strip()
    if supplied and supplied in source:
        return supplied
    if selected:
        return selected
    return source.strip()


def extract_attack_actions(
    text: str,
    client: OpenRouterClient,
    *,
    source_type: str = "free_text",
    source_name: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    seed: int | None = 0,
    max_tokens: int = 4096,
) -> list[Observation]:
    source = str(text or "")
    if not source.strip():
        return []
    chosen_model = model or _settings_value(client, "extract_model", "openai/gpt-5.6-sol")
    chosen_effort = effort or _settings_value(client, "extract_effort", "low")
    numbered, source_lines = _numbered(source)
    guidance = _SOURCE_GUIDANCE.get(source_type, f"Treat the source as {source_type} evidence.")
    messages = [
        {"role": "system", "content": f"{_SYSTEM_PROMPT}\n\n{guidance}"},
        {"role": "user", "content": f"Source lines:\n{numbered}"},
    ]
    response = client.complete(
        messages,
        model=chosen_model,
        effort=chosen_effort,
        schema=ACTION_SCHEMA,
        schema_name="atomic_attack_actions",
        max_tokens=max_tokens,
        seed=_gpt_seed(chosen_model, seed),
    )
    items = _action_items(response.parsed)
    ledger = EvidenceLedger()
    for item in items:
        if isinstance(item, str):
            sentence = _sentence(item)
            line_numbers: list[int] = []
            raw_excerpt: Any = ""
        elif isinstance(item, dict):
            sentence = _sentence(
                item.get("sentence")
                or item.get("attack_action")
                or item.get("action")
                or item.get("text")
            )
            line_numbers = _source_lines(
                item.get("source_lines", item.get("line_numbers")), len(source_lines)
            )
            raw_excerpt = item.get("raw_excerpt", item.get("excerpt", ""))
        else:
            continue
        if not sentence:
            continue
        if not line_numbers:
            line_numbers = [1]
        excerpt = _excerpt(raw_excerpt, source_lines, line_numbers, source)
        ledger.add_observation(
            Observation(
                text=sentence,
                kind="action",
                provenance=Provenance(
                    source=source_type,
                    source_name=source_name,
                    line_numbers=line_numbers,
                    raw_excerpt=excerpt,
                    metadata={"extract_model": chosen_model},
                ),
            )
        )
    return ledger.observations


def extract_actions(*args: Any, **kwargs: Any) -> list[Observation]:
    return extract_attack_actions(*args, **kwargs)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8-sig")
    if hasattr(value, "getvalue"):
        return _as_text(value.getvalue())
    if hasattr(value, "read"):
        return _as_text(value.read())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return "\n".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in items
    )


@dataclass(slots=True)
class StructuredInputDiagnostics:
    parsed: int = 0
    skipped: int = 0
    malformed: int = 0
    correlated: int = 0
    deduplicated: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "skipped": self.skipped,
            "malformed": self.malformed,
            "correlated": self.correlated,
            "deduplicated": self.deduplicated,
            "source_count": len(self.sources),
            "sources": list(self.sources),
        }


@dataclass(slots=True)
class StructuredParseResult:
    observations: list[Observation]
    diagnostics: StructuredInputDiagnostics


def _source_sample(value: Any, maximum: int = 65_536) -> tuple[Any, str | None]:
    name = str(getattr(value, "name", "")) or None
    if isinstance(value, Mapping):
        return value, name
    if isinstance(value, Path):
        with value.open("rb") as stream:
            return stream.read(maximum), value.name
    if isinstance(value, str):
        try:
            path = Path(value)
            if "\n" not in value and len(value) < 4096 and path.is_file():
                with path.open("rb") as stream:
                    return stream.read(maximum), path.name
        except OSError:
            pass
        return value[:maximum], name
    if isinstance(value, bytes | bytearray):
        return bytes(value[:maximum]), name
    if hasattr(value, "getvalue"):
        sample = value.getvalue()
        return sample[:maximum], name
    if hasattr(value, "read"):
        position = None
        with suppress(AttributeError, OSError):
            position = value.tell()
        sample = value.read(maximum)
        if position is not None:
            with suppress(AttributeError, OSError):
                value.seek(position)
        return sample, name
    return value, name


def detect_structured_format(value: Any, *, source_name: str | None = None) -> str:
    """Safely classify selected evidence without executing content or consuming seekable files."""
    sample, inferred = _source_sample(value)
    name = (source_name or inferred or "").casefold()
    if isinstance(sample, bytes | bytearray):
        text = bytes(sample).decode("utf-8-sig", errors="replace")
    elif isinstance(sample, str):
        text = sample
    else:
        text = ""
    stripped = text.lstrip()
    if name.endswith(".xml") or stripped.startswith("<"):
        return "windows_xml"
    candidate: Any = sample if isinstance(sample, Mapping) else None
    if isinstance(sample, list | tuple):
        candidate = sample
    if candidate is None and stripped:
        for line in stripped.splitlines():
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                try:
                    candidate = json.loads(stripped)
                except json.JSONDecodeError:
                    return "unknown"
            break
    while isinstance(candidate, list) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, Mapping):
        if "object" in candidate and "action" in candidate:
            return "ecar"
        source = candidate.get("_source")
        if isinstance(source, Mapping):
            candidate = source
        keys = {str(key).casefold().replace("_", "").replace(".", "") for key in candidate}
        serialized = json.dumps(candidate, default=str)[:16_384].casefold()
        if (
            any("eventid" in key for key in keys)
            or "winlog" in serialized
            or '"eventid"' in serialized
        ):
            return "winlogs"
        if any(key in candidate for key in ("events", "Events", "hits", "records")):
            return "winlogs"
    return "unknown"


def _selected_sources(values: Any) -> list[Any]:
    if (
        isinstance(values, list | tuple)
        and values
        and all(
            isinstance(item, str | bytes | bytearray | Path)
            or hasattr(item, "read")
            or hasattr(item, "getvalue")
            for item in values
        )
    ):
        return list(values)
    if (
        isinstance(values, Iterable)
        and not isinstance(values, str | bytes | bytearray | Mapping)
        and not hasattr(values, "read")
    ):
        return [list(values)]
    return [values]


def _deduplicate_projections(observations: list[Observation]) -> tuple[list[Observation], int]:
    result: list[Observation] = []
    signatures: dict[tuple[Any, ...], Observation] = {}
    removed = 0
    for observation in observations:
        meta = observation.metadata
        event_id = meta.get("event_id")
        is_process = event_id in {1, 4688} or (
            meta.get("ecar_object") == "PROCESS" and meta.get("ecar_action") == "CREATE"
        )
        timestamp = str(meta.get("timestamp") or meta.get("timestamp_ms") or "")
        host = str(meta.get("host") or meta.get("computer") or "").casefold()
        command = str(meta.get("command") or meta.get("image") or "").casefold()
        if not is_process or not (host and command and timestamp):
            result.append(observation)
            continue
        signature = ("process", host, command, timestamp[:19])
        previous = signatures.get(signature)
        if previous is None:
            signatures[signature] = observation
            result.append(observation)
            continue
        previous_projection = previous.metadata.get("event_family") or previous.metadata.get(
            "ecar_object"
        )
        current_projection = observation.metadata.get("event_family") or observation.metadata.get(
            "ecar_object"
        )
        if previous_projection == current_projection:
            result.append(observation)
            continue
        projections = previous.provenance.metadata.setdefault("duplicate_projections", [])
        projections.append(
            {
                "source": observation.provenance.source,
                "source_name": observation.provenance.source_name,
                "line_numbers": observation.provenance.line_numbers,
                "event_id": observation.provenance.event_id,
            }
        )
        previous.provenance.occurrence_count += observation.provenance.occurrence_count
        previous.metadata["duplicate_projection_count"] = len(projections)
        removed += 1
    return result, removed


def parse_structured_evidence(
    values: Any,
    *,
    max_records: int = 10_000,
    max_input_bytes: int = 16 * 1024 * 1024,
) -> StructuredParseResult:
    """Auto-detect and parse one or many selected structured evidence sources."""
    diagnostics = StructuredInputDiagnostics()
    observations: list[Observation] = []
    for value in _selected_sources(values):
        _, inferred = _source_sample(value)
        source_name = inferred
        detected = detect_structured_format(value, source_name=source_name)
        if detected == "ecar":
            from .ecar import parse_ecar

            parsed = parse_ecar(
                value,
                source_name=source_name,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
            )
        elif detected == "windows_xml":
            from .winlogs import parse_windows_xml_detailed

            parsed = parse_windows_xml_detailed(
                value,
                source_name=source_name,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
            )
        elif detected == "winlogs":
            from .winlogs import parse_winlogs_detailed

            parsed = parse_winlogs_detailed(
                value,
                source_name=source_name,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
            )
        else:
            diagnostics.skipped += 1
            diagnostics.sources.append(
                {
                    "source_name": source_name,
                    "format": "unknown",
                    "parsed": 0,
                    "skipped": 1,
                    "malformed": 0,
                }
            )
            continue
        detail = parsed.diagnostics.to_dict()
        detail["format"] = detected
        diagnostics.sources.append(detail)
        diagnostics.parsed += parsed.diagnostics.parsed
        diagnostics.skipped += parsed.diagnostics.skipped
        diagnostics.malformed += parsed.diagnostics.malformed
        diagnostics.correlated += parsed.diagnostics.correlated
        observations.extend(parsed.observations)
    observations, removed = _deduplicate_projections(observations)
    diagnostics.deduplicated = removed
    return StructuredParseResult(observations, diagnostics)


def extract_evidence(
    client: OpenRouterClient | None = None,
    *,
    free_text: str | None = None,
    alerts: Any = None,
    winlogs: Any = None,
    ecar: Any = None,
    windows_xml: Any = None,
    iocs: Any = None,
    kb: Any = None,
    model: str | None = None,
    effort: str | None = None,
    seed: int | None = 0,
    source_names: dict[str, str] | None = None,
    strategic_context: StrategicContext | dict[str, Any] | None = None,
    structured_files: Any = None,
    structured_diagnostics: dict[str, Any] | None = None,
    max_records: int = 10_000,
    max_input_bytes: int = 16 * 1024 * 1024,
) -> EvidenceLedger:
    names = source_names or {}
    ledger = EvidenceLedger(strategic_context=strategic_context or StrategicContext())
    if free_text and free_text.strip():
        if client is None:
            raise ValueError("an OpenRouterClient is required for free-text extraction")
        ledger.extend_observations(
            extract_attack_actions(
                free_text,
                client,
                source_type="free_text",
                source_name=names.get("free_text"),
                model=model,
                effort=effort,
                seed=seed,
            )
        )
    alert_text = _as_text(alerts)
    if alert_text.strip():
        if client is None:
            raise ValueError("an OpenRouterClient is required for alert extraction")
        ledger.extend_observations(
            extract_attack_actions(
                alert_text,
                client,
                source_type="alert",
                source_name=names.get("alerts"),
                model=model,
                effort=effort,
                seed=seed,
            )
        )
    if winlogs is not None and winlogs != "":
        from .winlogs import parse_winlogs

        ledger.extend_observations(parse_winlogs(winlogs, source_name=names.get("winlogs")))
    if ecar is not None and ecar != "":
        from .ecar import parse_ecar_records

        ledger.extend_observations(parse_ecar_records(ecar, source_name=names.get("ecar")))
    if windows_xml is not None and windows_xml != "":
        from .winlogs import parse_windows_xml

        ledger.extend_observations(
            parse_windows_xml(windows_xml, source_name=names.get("windows_xml"))
        )
    if structured_files is not None:
        structured = parse_structured_evidence(
            structured_files, max_records=max_records, max_input_bytes=max_input_bytes
        )
        ledger.extend_observations(structured.observations)
        if structured_diagnostics is not None:
            structured_diagnostics.update(structured.diagnostics.to_dict())
    ioc_text = _as_text(iocs)
    if ioc_text.strip():
        from .iocs import extract_iocs

        ledger.extend_observations(extract_iocs(ioc_text, kb=kb, source_name=names.get("iocs")))
    return ledger


@dataclass(slots=True)
class EvidenceExtractor:
    client: OpenRouterClient | None = None
    kb: Any = None
    model: str | None = None
    effort: str | None = None
    seed: int | None = 0

    def free_text(self, text: str, *, source_name: str | None = None) -> list[Observation]:
        if self.client is None:
            raise ValueError("an OpenRouterClient is required for free-text extraction")
        return extract_attack_actions(
            text,
            self.client,
            source_type="free_text",
            source_name=source_name,
            model=self.model,
            effort=self.effort,
            seed=self.seed,
        )

    def alerts(self, alerts: Any, *, source_name: str | None = None) -> list[Observation]:
        if self.client is None:
            raise ValueError("an OpenRouterClient is required for alert extraction")
        return extract_attack_actions(
            _as_text(alerts),
            self.client,
            source_type="alert",
            source_name=source_name,
            model=self.model,
            effort=self.effort,
            seed=self.seed,
        )

    def winlogs(self, data: Any, *, source_name: str | None = None) -> list[Observation]:
        from .winlogs import parse_winlogs

        return parse_winlogs(data, source_name=source_name)

    def ecar(self, data: Any, *, source_name: str | None = None) -> list[Observation]:
        from .ecar import parse_ecar_records

        return parse_ecar_records(data, source_name=source_name)

    def windows_xml(self, data: Any, *, source_name: str | None = None) -> list[Observation]:
        from .winlogs import parse_windows_xml

        return parse_windows_xml(data, source_name=source_name)

    def iocs(self, text: Any, *, source_name: str | None = None) -> list[Observation]:
        from .iocs import extract_iocs

        return extract_iocs(_as_text(text), kb=self.kb, source_name=source_name)

    def extract(self, **inputs: Any) -> EvidenceLedger:
        return extract_evidence(
            self.client,
            kb=self.kb,
            model=self.model,
            effort=self.effort,
            seed=self.seed,
            **inputs,
        )
