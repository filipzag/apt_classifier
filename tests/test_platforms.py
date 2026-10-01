import pytest

from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.platforms import (
    PlatformScope,
    detect_platforms,
    ledger_platforms,
    normalize_platforms,
    observation_scope,
    set_ledger_platforms,
)


def test_normalize_platforms_canonical_order_and_aliases():
    assert normalize_platforms("linux, Windows;mac os") == ["Windows", "Linux", "macOS"]
    assert normalize_platforms(["ESXi", "esx", "OSX"]) == ["macOS", "ESXi"]
    assert normalize_platforms(None) == [] and normalize_platforms("") == []
    assert normalize_platforms("Solaris") == []
    with pytest.raises(ValueError, match="Solaris"):
        normalize_platforms("Linux, Solaris", strict=True)


def test_detect_platforms_reads_operating_system_names_only():
    assert detect_platforms("They added a cron job on the Ubuntu web server.") == ["Linux"]
    assert detect_platforms("A Windows service and a Linux systemd unit.") == ["Windows", "Linux"]
    assert detect_platforms("The implant targets macOS and the ESXi hypervisors.") == [
        "macOS",
        "ESXi",
    ]
    # ordinary words, client user agents and the attacker's own distribution are not targets
    assert detect_platforms("Changes were made during the maintenance windows.") == []
    assert (
        detect_platforms("The beacon used Mozilla/5.0 (Windows NT 10.0; Win64; x64) to reach C2.")
        == []
    )
    assert detect_platforms("From a Kali Linux box the operator ran secretsdump.") == []


def test_scope_keeps_techniques_the_operating_system_says_nothing_about():
    linux = PlatformScope(("Linux",), "case")
    assert not linux.allows(["Windows"])
    assert not linux.allows(["Windows", "macOS"])
    assert linux.allows(["Linux", "macOS"])
    assert linux.allows("Windows,Linux")  # comma-joined, as stored in the vector index
    assert linux.allows(["IaaS", "SaaS"])  # cloud techniques are not an OS question
    assert linux.allows(["Windows", "Identity Provider"])  # also documented off-host
    assert linux.allows([])  # undocumented platforms never filter
    windows_logs = PlatformScope(("Windows",), "log", strict=True)
    assert not windows_logs.allows(["IaaS"])  # event logs keep the stricter rule
    assert windows_logs.allows(["Linux", "Windows"])


def test_observation_scope_precedence():
    plain = Observation(text="The attacker scheduled a recurring job for persistence.")
    named = Observation(text="The attacker added a cron job on the Linux server.")
    winlog = Observation(
        text="Linux is mentioned in this Windows event.",
        kind="winlog",
        provenance=Provenance(source="winlog"),
    )
    assert observation_scope(plain) is None
    assert observation_scope(plain, ["Linux"]) == PlatformScope(("Linux",), "case")
    # an observation's own OS overrides the case only when text detection is on
    assert observation_scope(named, ["Windows"]) == PlatformScope(("Windows",), "case")
    assert observation_scope(named, ["Windows"], from_text=True) == PlatformScope(
        ("Linux",), "text"
    )
    # the log source wins over both
    assert observation_scope(winlog, ["Linux"], from_text=True) == PlatformScope(
        ("Windows",), "log", strict=True
    )


def test_target_os_is_stored_in_ledger_metadata_and_round_trips():
    ledger = EvidenceLedger(observations=[Observation(text="The attacker ran a command.")])
    assert ledger_platforms(ledger) == []
    assert set_ledger_platforms(ledger, "linux, esxi") == ["Linux", "ESXi"]
    restored = EvidenceLedger.model_validate(ledger.to_dict())
    assert ledger_platforms(restored) == ["Linux", "ESXi"]
    assert set_ledger_platforms(restored, []) == []
    assert "target_platforms" not in restored.metadata
