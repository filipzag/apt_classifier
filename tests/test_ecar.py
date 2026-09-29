import json

from apt_classifier.evidence.ecar import parse_ecar
from apt_classifier.evidence.extract import parse_structured_evidence


class UploadedFile:
    def __init__(self, name: str, value: bytes) -> None:
        self.name = name
        self._value = value

    def getvalue(self) -> bytes:
        return self._value


def record(**overrides):
    value = {
        "id": "event-1",
        "timestamp_ms": 1000,
        "hostname": "workstation-1",
        "object": "PROCESS",
        "action": "CREATE",
        "principal": "operator",
        "properties": {"image_path": "C:/Tools/utility.exe", "command_line": "utility.exe /q"},
    }
    value.update(overrides)
    return value


def test_streamlit_uploaded_file_bytes_and_structured_metadata():
    upload = UploadedFile("events.jsonl", (json.dumps(record()) + "\n").encode())
    result = parse_ecar(upload)
    assert result.diagnostics.parsed == 1
    observation = result.observations[0]
    assert observation.provenance.source_name == "events.jsonl"
    assert observation.provenance.metadata["command"] == "utility.exe /q"
    assert observation.provenance.metadata["host"] == "workstation-1"
    assert observation.provenance.metadata["principal"] == "operator"


def test_malformed_skipped_and_record_bound_are_reported():
    payload = b"not-json\n" + b"\n".join(
        json.dumps(record(id=str(index))).encode() for index in range(4)
    )
    result = parse_ecar(payload, max_records=2)
    assert result.diagnostics.malformed == 1
    assert result.diagnostics.skipped == 1
    assert result.diagnostics.parsed == 1
    assert result.diagnostics.truncated is True


def test_ecar_correlates_repeated_flows_but_not_one_generic_row():
    rows = []
    for index, timestamp in enumerate((1000, 2000, 3000, 4000)):
        rows.append(
            record(
                id=f"flow-{index}",
                timestamp_ms=timestamp,
                object="FLOW",
                action="CONNECT",
                properties={"src_ip": "10.0.0.2", "dst_ip": "198.51.100.7", "dst_port": 443},
            )
        )
    result = parse_ecar(rows)
    correlations = [item for item in result.observations if item.kind == "correlation"]
    assert len(correlations) == 1
    assert correlations[0].metadata["correlation_type"] == "periodic_network_connections"
    assert not [item for item in parse_ecar([rows[0]]).observations if item.kind == "correlation"]


def test_router_preserves_per_file_provenance():
    first = UploadedFile("first.jsonl", (json.dumps(record(id="one")) + "\n").encode())
    second = UploadedFile("second.jsonl", (json.dumps(record(id="two")) + "\n").encode())
    result = parse_structured_evidence([first, second])
    assert result.diagnostics.to_dict()["source_count"] == 2
    assert {item.provenance.source_name for item in result.observations} == {
        "first.jsonl",
        "second.jsonl",
    }
