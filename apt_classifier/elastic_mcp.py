import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from apt_classifier.evidence.models import Observation, Provenance

ELASTIC_SEARCH_TOOL = "platform.core.search"
ELASTIC_LIST_INDICES_TOOL = "platform.core.list_indices"
ELASTIC_GENERATE_ESQL_TOOL = "platform.core.generate_esql"
ELASTIC_EXECUTE_ESQL_TOOL = "platform.core.execute_esql"
ELASTIC_READ_ONLY_TOOLS = {
    ELASTIC_SEARCH_TOOL,
    ELASTIC_LIST_INDICES_TOOL,
    ELASTIC_GENERATE_ESQL_TOOL,
    ELASTIC_EXECUTE_ESQL_TOOL,
}
_MAX_FIELD_COUNT = 40
_MAX_TEXT_LENGTH = 5000


@dataclass(frozen=True)
class MCPToolDescriptor:
    name: str
    description: str = ""
    read_only: bool | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPToolOutput:
    structured_content: Any = None
    text_content: tuple[str, ...] = ()
    is_error: bool = False


class ElasticMCPTransport(Protocol):
    def execute(
        self,
        endpoint: str,
        authorization: str,
        timeout: float,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[tuple[MCPToolDescriptor, ...], MCPToolOutput]: ...

    def execute_sequence(
        self,
        endpoint: str,
        authorization: str,
        timeout: float,
        calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[tuple[MCPToolDescriptor, ...], tuple[MCPToolOutput, ...]]: ...


class StreamableHTTPMCPTransport:
    def execute(
        self,
        endpoint: str,
        authorization: str,
        timeout: float,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[tuple[MCPToolDescriptor, ...], MCPToolOutput]:
        tools, outputs = self.execute_sequence(
            endpoint, authorization, timeout, [(tool_name, arguments)]
        )
        return tools, outputs[0]

    def execute_sequence(
        self,
        endpoint: str,
        authorization: str,
        timeout: float,
        calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[tuple[MCPToolDescriptor, ...], tuple[MCPToolOutput, ...]]:
        return asyncio.run(self._execute_sequence(endpoint, authorization, timeout, calls))

    async def _execute_sequence(
        self,
        endpoint: str,
        authorization: str,
        timeout: float,
        calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[tuple[MCPToolDescriptor, ...], tuple[MCPToolOutput, ...]]:
        headers = {"Authorization": authorization, "Content-Type": "application/json"}
        async with (
            httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
            ) as http_client,
            streamable_http_client(
                endpoint,
                http_client=http_client,
                terminate_on_close=False,
            ) as (read_stream, write_stream, _),
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=timeout),
            ) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            tools = tuple(_tool_descriptor(tool) for tool in listed.tools)
            outputs: list[MCPToolOutput] = []
            for tool_name, arguments in calls:
                _require_read_only_tool(tools, tool_name)
                result = await session.call_tool(
                    tool_name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=timeout),
                )
                texts = tuple(
                    str(content.text)
                    for content in result.content
                    if getattr(content, "type", "") == "text" and getattr(content, "text", None)
                )
                outputs.append(
                    MCPToolOutput(
                        structured_content=result.structuredContent,
                        text_content=texts,
                        is_error=bool(result.isError),
                    )
                )
        return tools, tuple(outputs)


@dataclass(frozen=True)
class ElasticEvidenceCandidate:
    id: str
    technique_id: str
    technique_name: str
    text: str
    raw_excerpt: str
    event_id: int | None = None
    timestamp: str | None = None
    index: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> Observation:
        provenance_metadata = {
            "mcp_tool": self.metadata.get("mcp_tool", ELASTIC_EXECUTE_ESQL_TOOL),
            "technique_id": self.technique_id,
            "timestamp": self.timestamp,
            "index": self.index,
            "document_id": self.document_id,
            **self.metadata,
            "review_status": "analyst_accepted",
        }
        provenance_metadata = {
            key: value for key, value in provenance_metadata.items() if value not in {None, ""}
        }
        return Observation(
            text=self.text,
            kind="elastic_mcp",
            provenance=Provenance(
                source="elasticsearch_mcp",
                source_name="Elastic Agent Builder MCP",
                raw_excerpt=self.raw_excerpt,
                event_id=self.event_id,
                metadata=provenance_metadata,
            ),
            metadata=provenance_metadata,
        )


@dataclass(frozen=True)
class ElasticEvidenceBatch:
    query_id: str
    technique_id: str
    technique_name: str
    query: str
    collected_at: str
    candidates: tuple[ElasticEvidenceCandidate, ...] = ()
    tool_name: str = ELASTIC_SEARCH_TOOL
    esql_query: str | None = None
    indices_used: tuple[str, ...] = ()
    flow: str = "search"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ElasticMCPClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        max_results: int = 20,
        transport: ElasticMCPTransport | None = None,
    ) -> None:
        self.endpoint = _validated_endpoint(endpoint)
        if not api_key.strip():
            raise ValueError("Elasticsearch MCP API key is required")
        if timeout <= 0:
            raise ValueError("Elasticsearch MCP timeout must be positive")
        if not 1 <= max_results <= 100:
            raise ValueError("Elasticsearch MCP max_results must be between 1 and 100")
        self._api_key = api_key.strip()
        self.timeout = timeout
        self.max_results = max_results
        self.transport = transport or StreamableHTTPMCPTransport()

    @classmethod
    def from_settings(cls, settings: Any) -> "ElasticMCPClient":
        secret = getattr(settings, "elastic_mcp_api_key", None)
        api_key = secret.get_secret_value() if secret is not None else ""
        return cls(
            str(getattr(settings, "elastic_mcp_url", "") or ""),
            api_key,
            timeout=float(getattr(settings, "elastic_mcp_timeout", 60.0)),
            max_results=int(getattr(settings, "elastic_mcp_max_results", 20)),
        )

    def gather(self, lead: Any, *, lookback_hours: int = 72) -> ElasticEvidenceBatch:
        query = build_evidence_query(
            lead,
            lookback_hours=lookback_hours,
            max_results=self.max_results,
        )
        technique_id = str(getattr(lead, "technique_id", ""))
        technique_name = str(getattr(lead, "technique_name", technique_id))
        query_id = _identifier("elastic-query", technique_id, query)
        collected_at = datetime.now(UTC).isoformat()
        auth = f"ApiKey {self._api_key}"
        try:
            tools = self._list_tools(auth)
            if self._has_tools(tools, ELASTIC_GENERATE_ESQL_TOOL, ELASTIC_EXECUTE_ESQL_TOOL):
                return self._gather_via_esql(
                    auth,
                    tools,
                    lead,
                    query,
                    technique_id,
                    technique_name,
                    query_id,
                    collected_at,
                    lookback_hours,
                )
            if self._has_tools(tools, ELASTIC_SEARCH_TOOL):
                return self._gather_via_search(
                    auth,
                    tools,
                    query,
                    technique_id,
                    technique_name,
                    query_id,
                    collected_at,
                )
            raise RuntimeError(
                "Elastic MCP did not expose any supported read-only search tools "
                f"(expected one of {', '.join(sorted(ELASTIC_READ_ONLY_TOOLS))})"
            )
        except Exception as exc:
            return ElasticEvidenceBatch(
                query_id=query_id,
                technique_id=technique_id,
                technique_name=technique_name,
                query=query,
                collected_at=collected_at,
                error=_safe_error(exc, self._api_key),
            )

    def _list_tools(self, auth: str) -> tuple[MCPToolDescriptor, ...]:
        tools, _ = self.transport.execute_sequence(
            self.endpoint,
            auth,
            self.timeout,
            [],
        )
        return tools

    def _gather_via_esql(
        self,
        auth: str,
        tools: tuple[MCPToolDescriptor, ...],
        lead: Any,
        natural_query: str,
        technique_id: str,
        technique_name: str,
        query_id: str,
        collected_at: str,
        lookback_hours: int,
    ) -> ElasticEvidenceBatch:
        indices: tuple[str, ...] = ()
        sequence: list[tuple[str, dict[str, Any]]] = []
        if self._has_tools(tools, ELASTIC_LIST_INDICES_TOOL):
            sequence.append((ELASTIC_LIST_INDICES_TOOL, {"query": _index_discovery_query(lead)}))
        sequence.append((ELASTIC_GENERATE_ESQL_TOOL, {"query": natural_query}))
        _, outputs = self.transport.execute_sequence(
            self.endpoint,
            auth,
            self.timeout,
            sequence,
        )
        output_index = 0
        if self._has_tools(tools, ELASTIC_LIST_INDICES_TOOL):
            indices = _extract_indices(outputs[output_index])
            output_index += 1
        esql_output = outputs[output_index]
        if esql_output.is_error:
            raise RuntimeError("Elastic MCP generate_esql tool returned an error result")
        esql_query = _extract_esql_query(esql_output)
        if not esql_query:
            raise RuntimeError("Elastic MCP generate_esql did not return a query string")
        _, exec_output = self.transport.execute_sequence(
            self.endpoint,
            auth,
            self.timeout,
            [(ELASTIC_EXECUTE_ESQL_TOOL, {"query": esql_query})],
        )
        exec_result = exec_output[0]
        if exec_result.is_error:
            raise RuntimeError("Elastic MCP execute_esql tool returned an error result")
        candidates = _evidence_candidates(
            exec_result,
            technique_id=technique_id,
            technique_name=technique_name,
            query_id=query_id,
            limit=self.max_results,
        )
        return ElasticEvidenceBatch(
            query_id=query_id,
            technique_id=technique_id,
            technique_name=technique_name,
            query=natural_query,
            collected_at=collected_at,
            candidates=candidates,
            tool_name=ELASTIC_EXECUTE_ESQL_TOOL,
            esql_query=esql_query,
            indices_used=indices,
            flow="esql",
        )

    def _gather_via_search(
        self,
        auth: str,
        tools: tuple[MCPToolDescriptor, ...],
        natural_query: str,
        technique_id: str,
        technique_name: str,
        query_id: str,
        collected_at: str,
    ) -> ElasticEvidenceBatch:
        _, outputs = self.transport.execute_sequence(
            self.endpoint,
            auth,
            self.timeout,
            [(ELASTIC_SEARCH_TOOL, {"query": natural_query})],
        )
        output = outputs[0]
        if output.is_error:
            raise RuntimeError("Elastic MCP search tool returned an error result")
        candidates = _evidence_candidates(
            output,
            technique_id=technique_id,
            technique_name=technique_name,
            query_id=query_id,
            limit=self.max_results,
        )
        return ElasticEvidenceBatch(
            query_id=query_id,
            technique_id=technique_id,
            technique_name=technique_name,
            query=natural_query,
            collected_at=collected_at,
            candidates=candidates,
            tool_name=ELASTIC_SEARCH_TOOL,
            flow="search",
        )

    @staticmethod
    def _has_tools(tools: tuple[MCPToolDescriptor, ...], *required: str) -> bool:
        available = {tool.name for tool in tools if tool.read_only is not False}
        return all(name in available for name in required)


def elastic_mcp_configured(settings: Any) -> bool:
    endpoint = str(getattr(settings, "elastic_mcp_url", "") or "").strip()
    secret = getattr(settings, "elastic_mcp_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    return bool(endpoint and key)


def build_evidence_query(
    lead: Any,
    *,
    lookback_hours: int = 72,
    max_results: int = 20,
) -> str:
    hours = min(max(int(lookback_hours), 1), 24 * 90)
    limit = min(max(int(max_results), 1), 100)
    technique_id = str(getattr(lead, "technique_id", "")).strip()
    technique_name = str(getattr(lead, "technique_name", technique_id)).strip()
    steps = tuple(
        str(step).strip() for step in getattr(lead, "investigation_steps", ()) if str(step)
    )
    guidance = "\n".join(f"- {step}" for step in steps[:8])
    return (
        f"Find at most {limit} actual event records from the past {hours} hours that could confirm "
        f"or refute behavior consistent with {technique_name} ({technique_id}). Search only data "
        "available to this MCP connection. Return event records, not recommendations or a narrative "
        "assessment. Include exact available values for timestamp, index, document ID, event code, "
        "host, user, source and destination, process image, parent process, command line, file path, "
        "hash, and original message. Do not infer missing values, attacker identity, or whether the "
        "technique occurred. Return no records when the available telemetry does not support a match. "
        f"Use these collection pivots where relevant:\n{guidance}"
    )[:8000]


def _tool_descriptor(tool: Any) -> MCPToolDescriptor:
    annotations = getattr(tool, "annotations", None)
    return MCPToolDescriptor(
        name=str(getattr(tool, "name", "")),
        description=str(getattr(tool, "description", "") or ""),
        read_only=getattr(annotations, "readOnlyHint", None),
        input_schema=dict(getattr(tool, "inputSchema", {}) or {}),
    )


def _require_read_only_tool(tools: Sequence[MCPToolDescriptor], tool_name: str) -> None:
    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        raise RuntimeError(f"Elastic MCP did not expose required tool {tool_name}")
    if tool.read_only is False:
        raise RuntimeError(f"Elastic MCP tool {tool_name} is not marked read-only")


def _require_search_tool(tools: Sequence[MCPToolDescriptor], tool_name: str) -> None:
    """Backward-compatible alias; validates the search tool specifically."""
    _require_read_only_tool(tools, tool_name)


def _extract_esql_query(output: MCPToolOutput) -> str | None:
    content = output.structured_content
    if isinstance(content, Mapping):
        for key in ("query", "esql", "esql_query", "sql"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(content, str) and content.strip():
        parsed = _parse_json(content)
        if isinstance(parsed, Mapping):
            for key in ("query", "esql", "esql_query", "sql"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return content.strip()
    for text in output.text_content:
        parsed = _parse_json(text)
        if isinstance(parsed, Mapping):
            for key in ("query", "esql", "esql_query", "sql"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if text.strip().upper().startswith(("FROM ", "SELECT ")):
            return text.strip()
    return None


def _extract_indices(output: MCPToolOutput) -> tuple[str, ...]:
    content = output.structured_content
    indices: list[str] = []
    if isinstance(content, Mapping):
        for key in ("indices", "data", "results", "items"):
            value = content.get(key)
            indices.extend(_index_names_from(value))
    if not indices:
        for text in output.text_content:
            parsed = _parse_json(text)
            if isinstance(parsed, Mapping):
                for key in ("indices", "data", "results", "items"):
                    indices.extend(_index_names_from(parsed.get(key)))
    return tuple(dict.fromkeys(indices))


def _index_names_from(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        for key in ("index", "name", "index_name"):
            name = value.get(key)
            if isinstance(name, str) and name.strip():
                return [name.strip()]
        return []
    if isinstance(value, list | tuple):
        result: list[str] = []
        for item in value:
            result.extend(_index_names_from(item))
        return result
    return []


def _index_discovery_query(lead: Any) -> str:
    technique_name = str(getattr(lead, "technique_name", getattr(lead, "technique_id", "")))
    steps = tuple(
        str(step).strip() for step in getattr(lead, "investigation_steps", ()) if str(step)
    )
    guidance = "; ".join(steps[:4])
    return (
        f"List security-relevant indices and datastreams that may contain event records "
        f"related to {technique_name}. Focus on logs, beats, security, sysmon, and audit "
        f"datastreams. Return index names and their key mapped fields. "
        f"Collection pivots: {guidance}"
    )[:2000]


def _evidence_candidates(
    output: MCPToolOutput,
    *,
    technique_id: str,
    technique_name: str,
    query_id: str,
    limit: int,
) -> tuple[ElasticEvidenceCandidate, ...]:
    records = _records(output.structured_content)
    if not records:
        for text in output.text_content:
            parsed = _parse_json(text)
            if parsed is not None:
                records.extend(_records(parsed))
            else:
                records.extend({"message": part} for part in _text_records(text))
    candidates = []
    seen = set()
    for record in records:
        candidate = _candidate_from_record(
            record,
            technique_id=technique_id,
            technique_name=technique_name,
            query_id=query_id,
        )
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return tuple(candidates)


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        if value and all(isinstance(item, Mapping) and _looks_like_event(item) for item in value):
            return [dict(item) for item in value]
        records = []
        for item in value:
            records.extend(_records(item))
        return records
    if not isinstance(value, Mapping):
        return []
    if _is_tabular(value):
        data = value.get("data", value)
        columns = [
            str(column.get("name", "")) if isinstance(column, Mapping) else str(column)
            for column in data.get("columns", [])
        ]
        return [
            {columns[index]: item for index, item in enumerate(row) if index < len(columns)}
            for row in data.get("values", data.get("rows", []))
            if isinstance(row, list | tuple)
        ]
    hits = value.get("hits")
    if isinstance(hits, Mapping) and isinstance(hits.get("hits"), list):
        return [dict(item) for item in hits["hits"] if isinstance(item, Mapping)]
    for key in ("results", "events", "items", "rows", "documents", "content", "data"):
        if key in value:
            nested = _records(value[key])
            if nested:
                return nested
    return [dict(value)] if _looks_like_event(value) else []


def _is_tabular(value: Mapping[str, Any]) -> bool:
    data = value.get("data", value)
    return (
        isinstance(data, Mapping)
        and isinstance(data.get("columns"), list)
        and isinstance(data.get("values", data.get("rows")), list)
    )


def _looks_like_event(value: Mapping[str, Any]) -> bool:
    keys = {_normalized_key(str(key)) for key in value}
    indicators = {
        "timestamp",
        "eventcode",
        "eventid",
        "message",
        "hostname",
        "processcommandline",
        "processname",
        "sourceip",
        "destinationip",
        "index",
        "id",
    }
    return bool(keys & indicators) or isinstance(value.get("_source"), Mapping)


def _candidate_from_record(
    record: Mapping[str, Any],
    *,
    technique_id: str,
    technique_name: str,
    query_id: str,
) -> ElasticEvidenceCandidate:
    source = record.get("_source") if isinstance(record.get("_source"), Mapping) else record
    flat = _flatten(source)
    index = _first(record, "_index", "index") or _flat_first(flat, "index")
    document_id = _first(record, "_id", "id") or _flat_first(flat, "id", "documentid")
    timestamp = _flat_first(flat, "timestamp", "eventcreated", "eventingested")
    event_id = _event_id(flat)
    fields = [
        f"{key}={_compact_value(value)}" for key, value in list(flat.items())[:_MAX_FIELD_COUNT]
    ]
    text = _limited("Elasticsearch event: " + "; ".join(fields), _MAX_TEXT_LENGTH)
    raw_excerpt = _limited(json.dumps(record, ensure_ascii=False, default=str), _MAX_TEXT_LENGTH)
    candidate_id = _identifier("elastic-evidence", index or "", document_id or "", raw_excerpt)
    return ElasticEvidenceCandidate(
        id=candidate_id,
        technique_id=technique_id,
        technique_name=technique_name,
        text=text,
        raw_excerpt=raw_excerpt,
        event_id=event_id,
        timestamp=timestamp,
        index=index,
        document_id=document_id,
        metadata={
            "query_id": query_id,
            "review_status": "pending_analyst_review",
            "mcp_tool": ELASTIC_EXECUTE_ESQL_TOOL,
        },
    )


def _flatten(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    if depth > 4:
        return {prefix: _compact_value(value)} if prefix else {}
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, child, depth + 1))
            if len(output) >= _MAX_FIELD_COUNT:
                break
        return output
    if isinstance(value, list | tuple):
        if all(not isinstance(item, Mapping | list | tuple) for item in value):
            return {prefix: ", ".join(_compact_value(item) for item in value)} if prefix else {}
        return {prefix: _compact_value(value)} if prefix else {}
    return {prefix: value} if prefix and value is not None else {}


def _event_id(flat: Mapping[str, Any]) -> int | None:
    value = _flat_first(flat, "eventcode", "eventid", "wineventid", "winlogeventid")
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def _flat_first(flat: Mapping[str, Any], *names: str) -> str | None:
    normalized = {_normalized_key(key): value for key, value in flat.items()}
    for name in names:
        value = normalized.get(_normalized_key(name))
        if value not in {None, ""}:
            return str(value)
    return None


def _first(value: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        item = value.get(name)
        if item is not None and item != "":
            return str(item)
    return None


def _text_records(value: str) -> list[str]:
    text = value.strip()
    if not text or text.casefold() in {"no results", "no matches", "no records"}:
        return []
    parts = [part.strip(" -*\t") for part in re.split(r"\n+", text)]
    records = [part for part in parts if len(part) >= 20]
    return records[:100] or [_limited(text, _MAX_TEXT_LENGTH)]


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _validated_endpoint(value: str) -> str:
    endpoint = value.strip()
    parsed = urlparse(endpoint)
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme not in ({"http", "https"} if local else {"https"}) or not parsed.hostname:
        raise ValueError("Elasticsearch MCP URL must use HTTPS, except for localhost development")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Elasticsearch MCP URL must not contain credentials, query, or fragment")
    if not parsed.path.endswith("/api/agent_builder/mcp"):
        raise ValueError("Elasticsearch MCP URL must end with /api/agent_builder/mcp")
    return endpoint


def _safe_error(error: Exception, secret: str) -> str:
    message = f"{type(error).__name__}: {error}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return _limited(message, 300)


def _identifier(prefix: str, *values: str) -> str:
    material = "\x1f".join(values)
    return f"{prefix}-{hashlib.sha256(material.encode()).hexdigest()[:20]}"


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _compact_value(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False, default=str)).strip()


def _limited(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit].rstrip()}…"
