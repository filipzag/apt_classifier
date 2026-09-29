import json

from apt_classifier.evidence.winlogs import parse_winlogs


def test_sysmon_process_access_names_lsass_memory_without_calltrace_noise():
    event = {
        "EventID": 10,
        "ProviderName": "Microsoft-Windows-Sysmon",
        "SourceImage": "C:\\Tools\\Dumpert.exe",
        "TargetImage": "C:\\Windows\\System32\\lsass.exe",
        "GrantedAccess": "0x1fffff",
        "CallTrace": "very-long-call-trace-that-should-not-be-embedded",
    }
    observation = parse_winlogs(json.dumps(event))[0]
    assert "accessed LSASS memory" in observation.text
    assert "Dumpert.exe" in observation.text
    assert "very-long-call-trace" not in observation.text
