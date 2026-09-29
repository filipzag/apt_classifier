import json

from apt_classifier.evidence.winlogs import (
    parse_windows_xml_detailed,
    parse_winlogs,
    parse_winlogs_detailed,
)


def test_parse_flat_sysmon_jsonl_and_dedupe():
    event = {
        "EventID": 1,
        "ProviderName": "Microsoft-Windows-Sysmon",
        "Image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        "CommandLine": "powershell.exe -enc SQBFAFgA",
        "ParentImage": "C:\\Program Files\\Microsoft Office\\WINWORD.EXE",
        "User": "DOMAIN\\analyst",
    }
    observations = parse_winlogs("\n".join((json.dumps(event), json.dumps(event))))
    assert len(observations) == 1
    observation = observations[0]
    assert observation.text.startswith("powershell.exe -enc")
    assert "run by DOMAIN\\analyst" in observation.text
    assert observation.provenance.occurrence_count == 2
    assert observation.provenance.line_numbers == [1, 2]


def test_parse_elastic_and_evtx_wrapped_events():
    events = {
        "events": [
            {
                "winlog": {
                    "event_id": 22,
                    "provider_name": "Microsoft-Windows-Sysmon",
                    "event_data": {"Image": "cmd.exe", "QueryName": "evil.example"},
                }
            },
            {
                "Event": {
                    "System": {"EventID": 4698, "Provider": {"Name": "Security"}},
                    "EventData": {
                        "Data": [
                            {"Name": "TaskName", "#text": "Updater"},
                            {"Name": "SubjectUserName", "#text": "analyst"},
                        ]
                    },
                }
            },
        ]
    }
    observations = parse_winlogs(json.dumps(events, indent=2))
    assert len(observations) == 2
    assert "queried DNS" in observations[0].text
    assert "evil.example" in observations[0].text
    assert "created a scheduled task" in observations[1].text
    assert "Updater" in observations[1].text


def test_unsupported_event_is_ignored():
    assert parse_winlogs(json.dumps({"EventID": 9999, "Message": "noise"})) == []


def test_windows_xml_accepts_uploaded_bytes_and_reports_skips():
    xml = b"""<Events><Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
      <System><Provider Name="Microsoft-Windows-Security-Auditing"/><EventID>4625</EventID>
      <TimeCreated SystemTime="2025-01-01T00:00:00Z"/><Computer>host-1</Computer></System>
      <EventData><Data Name="TargetUserName">user-1</Data><Data Name="IpAddress">10.0.0.8</Data></EventData>
    </Event><Event><System><EventID>9999</EventID></System></Event></Events>"""
    result = parse_windows_xml_detailed(xml)
    assert result.diagnostics.parsed == 1
    assert result.diagnostics.skipped == 1
    assert result.observations[0].provenance.metadata["host"] == "host-1"
    assert result.observations[0].provenance.metadata["principal"] == "user-1"


def test_failed_logon_correlation_requires_repetition():
    events = [
        {"EventID": 4625, "Computer": "host-1", "TargetUserName": "user-1", "IpAddress": "10.0.0.8"}
        for _ in range(3)
    ]
    result = parse_winlogs_detailed(events)
    correlations = [item for item in result.observations if item.kind == "correlation"]
    assert len(correlations) == 1
    assert correlations[0].metadata["correlation_type"] == "repeated_failed_logons"
    assert not [item for item in parse_winlogs(events[:1]) if item.kind == "correlation"]
