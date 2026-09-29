from types import SimpleNamespace

import pytest

from apt_classifier.elastic_mcp import (
    ELASTIC_EXECUTE_ESQL_TOOL,
    ELASTIC_GENERATE_ESQL_TOOL,
    ELASTIC_LIST_INDICES_TOOL,
    ELASTIC_SEARCH_TOOL,
    ElasticMCPClient,
    MCPToolDescriptor,
    MCPToolOutput,
    build_evidence_query,
)
from apt_classifier.evidence.models import EvidenceLedger, Observation
from apt_classifier.session import AttributionEngine


class FakeTransport:
    def __init__(self, tools, outputs=None, output=None):
        self.tools = tuple(tools)
        if output is not None and outputs is None:
            outputs = [output]
        self.outputs = tuple(outputs or [])
        self.calls = []

    def execute(self, endpoint, authorization, timeout, tool_name, arguments):
        self.calls.append((endpoint, authorization, timeout, tool_name, arguments))
        return self.tools, self.outputs[len(self.calls) - 1]

    def execute_sequence(self, endpoint, authorization, timeout, calls):
        result_calls = []
        result_outputs = []
        for tool_name, arguments in calls:
            result_calls.append((endpoint, authorization, timeout, tool_name, arguments))
            idx = len(self.calls)
            self.calls.append(result_calls[-1])
            result_outputs.append(self.outputs[min(idx, len(self.outputs) - 1)])
        return self.tools, tuple(result_outputs)


def _lead():
    return SimpleNamespace(
        technique_id="T1059.001",
        technique_name="PowerShell",
        hypothesis="Unconfirmed investigation hypothesis: PowerShell may be present.",
        investigation_steps=(
            "PowerShell Operational Event IDs 4103 and 4104.",
            "Security 4688 and Sysmon 1.",
        ),
    )


def test_query_plan_is_bounded_and_uses_no_unscoped_index():
    query = build_evidence_query(_lead(), lookback_hours=48, max_results=12)
    assert "T1059.001" in query
    assert "past 48 hours" in query
    assert "at most 12" in query
    assert "actual event records" in query
    assert "Do not infer" in query


def test_client_rejects_unsafe_remote_endpoint_shapes():
    for endpoint in (
        "http://kibana.example/api/agent_builder/mcp",
        "https://user:pass@kibana.example/api/agent_builder/mcp",
        "https://kibana.example/not-mcp",
    ):
        with pytest.raises(ValueError):
            ElasticMCPClient(endpoint, "secret-api-key")


def test_client_discovers_read_only_search_tool_and_normalizes_tabular_results():
    output = MCPToolOutput(
        structured_content={
            "results": [
                {
                    "type": "tabular_data",
                    "data": {
                        "columns": [
                            {"name": "@timestamp"},
                            {"name": "host.name"},
                            {"name": "event.code"},
                            {"name": "process.command_line"},
                        ],
                        "values": [
                            ["2026-09-07T10:00:00Z", "ws-01", "4104", "powershell -enc AAA"],
                            ["2026-09-07T10:01:00Z", "ws-01", "4688", "powershell.exe"],
                        ],
                    },
                }
            ]
        }
    )
    transport = FakeTransport(
        [MCPToolDescriptor(ELASTIC_SEARCH_TOOL, read_only=True)],
        outputs=[output, output],
    )
    client = ElasticMCPClient(
        "https://kibana.example/api/agent_builder/mcp",
        "secret-api-key",
        max_results=10,
        transport=transport,
    )
    batch = client.gather(_lead(), lookback_hours=24)
    assert batch.ok
    assert batch.flow == "search"
    assert batch.tool_name == ELASTIC_SEARCH_TOOL
    assert len(batch.candidates) == 2
    assert batch.candidates[0].event_id == 4104
    assert "powershell -enc AAA" in batch.candidates[0].text
    observation = batch.candidates[0].to_observation()
    assert observation.kind == "elastic_mcp"
    assert observation.provenance.source == "elasticsearch_mcp"
    assert observation.provenance.event_id == 4104
    tool_names = [call[3] for call in transport.calls]
    assert ELASTIC_SEARCH_TOOL in tool_names


def test_client_uses_esql_flow_when_generate_and_execute_tools_available():
    list_output = MCPToolOutput(
        structured_content={"indices": ["logs-sysmon-default", ".ds-logs-system-default"]}
    )
    generate_output = MCPToolOutput(
        structured_content={
            "query": 'FROM logs-sysmon-default | WHERE event.code == "4104" | LIMIT 10'
        }
    )
    exec_output = MCPToolOutput(
        structured_content={
            "results": [
                {
                    "type": "tabular_data",
                    "data": {
                        "columns": [
                            {"name": "@timestamp"},
                            {"name": "event.code"},
                            {"name": "process.command_line"},
                        ],
                        "values": [
                            ["2026-09-07T10:00:00Z", "4104", "powershell -enc AAA"],
                        ],
                    },
                }
            ]
        }
    )
    tools = [
        MCPToolDescriptor(ELASTIC_LIST_INDICES_TOOL, read_only=True),
        MCPToolDescriptor(ELASTIC_GENERATE_ESQL_TOOL, read_only=True),
        MCPToolDescriptor(ELASTIC_EXECUTE_ESQL_TOOL, read_only=True),
    ]
    transport = FakeTransport(
        tools,
        outputs=[list_output, generate_output, exec_output],
    )
    client = ElasticMCPClient(
        "https://kibana.example/api/agent_builder/mcp",
        "secret-api-key",
        max_results=10,
        transport=transport,
    )
    batch = client.gather(_lead(), lookback_hours=24)
    assert batch.ok
    assert batch.flow == "esql"
    assert batch.tool_name == ELASTIC_EXECUTE_ESQL_TOOL
    assert batch.esql_query is not None
    assert "FROM" in batch.esql_query
    assert "logs-sysmon-default" in batch.indices_used
    assert len(batch.candidates) == 1
    assert batch.candidates[0].event_id == 4104
    tool_names = [call[3] for call in transport.calls]
    assert ELASTIC_LIST_INDICES_TOOL in tool_names
    assert ELASTIC_GENERATE_ESQL_TOOL in tool_names
    assert ELASTIC_EXECUTE_ESQL_TOOL in tool_names


def test_client_rejects_missing_or_non_read_only_search_tool_without_calling_it():
    for tools in (
        [MCPToolDescriptor(ELASTIC_EXECUTE_ESQL_TOOL, read_only=True)],
        [MCPToolDescriptor(ELASTIC_SEARCH_TOOL, read_only=False)],
    ):
        transport = FakeTransport(tools, outputs=[MCPToolOutput()])
        client = ElasticMCPClient(
            "https://kibana.example/api/agent_builder/mcp",
            "secret-api-key",
            transport=transport,
        )
        batch = client.gather(_lead())
        assert not batch.ok
        assert batch.candidates == ()
        assert any(
            name in str(batch.error)
            for name in (
                ELASTIC_SEARCH_TOOL,
                ELASTIC_GENERATE_ESQL_TOOL,
                ELASTIC_EXECUTE_ESQL_TOOL,
            )
        )


def test_client_caps_results_and_does_not_leak_api_key_in_errors():
    output = MCPToolOutput(
        structured_content={"events": [{"message": f"event {index}"} for index in range(20)]}
    )
    transport = FakeTransport(
        [MCPToolDescriptor(ELASTIC_SEARCH_TOOL, read_only=True)],
        outputs=[output, output],
    )
    client = ElasticMCPClient(
        "https://kibana.example/api/agent_builder/mcp",
        "secret-api-key",
        max_results=3,
        transport=transport,
    )
    batch = client.gather(_lead())
    assert len(batch.candidates) == 3
    assert "secret-api-key" not in str(batch)


def test_imported_mcp_observations_are_reanalyzed_and_keep_live_intel():
    engine = object.__new__(AttributionEngine)
    ledger = EvidenceLedger(observations=[Observation(text="Original evidence")])
    snapshot = SimpleNamespace()
    state = SimpleNamespace(
        ledger=ledger, weights={"specific_behavior": 1.5}, threat_intel_snapshot=snapshot
    )
    updated = SimpleNamespace(ledger=None)
    calls = []
    engine.analyze = lambda value, weights, **_: calls.append((value, weights)) or updated
    engine.attach_threat_intel = lambda value, intel: calls.append((value, intel))
    imported = Observation(text="Elasticsearch event: event.code=4104", kind="elastic_mcp")
    result = engine.import_observations(state, [imported])
    assert result is updated
    assert imported in ledger.observations
    assert calls[0] == (ledger, state.weights)
    assert calls[1] == (updated, snapshot)
