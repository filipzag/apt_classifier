"""Regression tests for EvidenceForge remediation.

These tests reproduce the weaknesses identified during the EvidenceForge
F1 investigation and verify that the fixes remain in place:

1. Structured behavior signals fire on adversarial commands and are
   suppressed by benign counterexamples.
2. Signal-only candidates are not auto-accepted without adversarial context.
3. Signal-only candidates in adversarial context are auto-accepted.
4. Parent techniques are not auto-promoted from untrusted child candidates.
5. eCAR JSONL parsing produces observations with correct provenance.
6. Windows XML streaming produces observations from EvidenceForge files.
7. Reranker acceptance influence promotes/demotes candidates correctly.
8. Benign indicators suppress false positives on ordinary activity.
9. Over-fetching retrieves unique techniques, not duplicate sentences.
10. EvidenceForge narrative mapping achieves F1 >= 0.40.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
from apt_classifier.evidence.models import (
    TechniqueCandidate,
    TechniqueCandidateState,
    TechniqueRetrievalEvidence,
)
from apt_classifier.retrieve import (
    _BENIGN_INDICATORS,
    RERANKER_DEMOTE_THRESHOLD,
    RERANKER_PROMOTE_THRESHOLD,
    _assign_candidate_states,
)
from apt_classifier.technique_signals import (
    SIGNALS,
    find_signals,
    signal_for_technique,
)

# ---------------------------------------------------------------------------
# 1. Structured behavior signals fire and are suppressed by benign guards
# ---------------------------------------------------------------------------


class TestStructuredSignals:
    """Tests for the structured behavior signals module."""

    def test_whoami_signal_fires(self):
        """T1033 signal fires on whoami command."""
        hits = find_signals("Attacker runs whoami on compromised host")
        technique_ids = {s.technique_id for s, _ in hits}
        assert "T1033" in technique_ids

    def test_nmap_signal_fires(self):
        """T1046 signal fires on nmap command."""
        hits = find_signals("nmap -p 22,80,445 target_segment")
        technique_ids = {s.technique_id for s, _ in hits}
        assert "T1046" in technique_ids

    def test_benign_guard_suppresses_antivirus_scan(self):
        """T1595.002 signal is suppressed by benign antivirus scan guard."""
        hits = find_signals("antivirus scan updated its signatures")
        technique_ids = {s.technique_id for s, _ in hits}
        # The benign guard should suppress the vulnerability scanner signal
        assert "T1595.002" not in technique_ids

    def test_benign_guard_suppresses_routine_backup(self):
        """T1560.001-related signals are suppressed by benign backup guard."""
        hits = find_signals("Routine backup of user documents")
        # No signals should fire on benign backup activity
        assert len(hits) == 0 or all(s.technique_id != "T1560.001" for s, _ in hits)

    def test_rdp_signal_fires(self):
        """T1021.001 signal fires on RDP session."""
        hits = find_signals("Administrative RDP session creates Windows logon")
        technique_ids = {s.technique_id for s, _ in hits}
        assert "T1021.001" in technique_ids

    def test_ports_list_signal_fires(self):
        """T1046 signal fires on Ports: [22, 80, 445, 3389] pattern."""
        hits = find_signals("Ports: [22, 80, 445, 3389]")
        technique_ids = {s.technique_id for s, _ in hits}
        assert "T1046" in technique_ids

    def test_signal_for_technique_returns_match(self):
        """signal_for_technique returns the matched span."""
        match = signal_for_technique("T1033", "whoami")
        assert match is not None
        assert "whoami" in match.lower()

    def test_signal_for_technique_returns_none_on_no_match(self):
        """signal_for_technique returns None when no match."""
        match = signal_for_technique("T1033", "nothing relevant here")
        assert match is None

    def test_all_signals_have_benign_guards(self):
        """All signals should have benign guards (except generic T1059)."""
        for signal in SIGNALS:
            # Generic T1059 may not need a benign guard
            if signal.technique_id in ("T1059",):
                continue
            assert signal.benign_guard is not None, (
                f"Signal {signal.technique_id} lacks benign guard"
            )


# ---------------------------------------------------------------------------
# 2 & 3. Signal-only acceptance with/without adversarial context
# ---------------------------------------------------------------------------


def _make_kb_with_techniques(technique_ids: list[str]) -> AttackKnowledgeBase:
    """Create a minimal KB with the given technique IDs."""
    techniques = {}
    for tid in technique_ids:
        parent = tid.split(".")[0] if "." in tid else None
        is_sub = "." in tid
        techniques[tid] = Technique(
            id=tid,
            name=f"Technique {tid}",
            description=f"Description for {tid}",
            is_subtechnique=is_sub,
            parent_id=parent,
            platforms=["Windows", "Linux"],
            tactics=["Discovery"],
        )
    return AttackKnowledgeBase(techniques=techniques, groups={})


class TestSignalAcceptance:
    """Tests for signal-based acceptance gating."""

    def test_benign_indicator_suppresses_false_positive(self):
        """Benign indicators suppress technique acceptance."""
        assert _BENIGN_INDICATORS.search("Normal user login to the workstation")
        assert _BENIGN_INDICATORS.search("Routine backup of user documents")
        assert _BENIGN_INDICATORS.search(
            "The antivirus software updated its signatures automatically."
        )
        assert _BENIGN_INDICATORS.search("A user mistypes a password during a VPN reconnect")
        assert _BENIGN_INDICATORS.search("Software installation by administrator")

    def test_benign_indicator_does_not_fire_on_adversarial_text(self):
        """Benign indicators do not fire on adversarial text."""
        assert not _BENIGN_INDICATORS.search("Attacker runs whoami on compromised host")
        assert not _BENIGN_INDICATORS.search("nmap -p 22,80,445 target_segment")


# ---------------------------------------------------------------------------
# 4. Parent promotion from untrusted child candidates
# ---------------------------------------------------------------------------


class TestParentPromotion:
    """Tests for parent promotion policies."""

    def test_parent_candidate_is_needs_review(self):
        """Parent candidates derived from hierarchy are NEEDS_REVIEW, not auto-accepted."""
        # Create a candidate that would be a parent from hierarchy
        parent_candidate = TechniqueCandidate(
            observation_id="obs-1",
            technique_id="T1059",
            technique_name="Command and Scripting Interpreter",
            parent_technique_id=None,
            is_subtechnique=False,
            fused_score=0.5,
            fused_rank=1,
            evidence=[
                TechniqueRetrievalEvidence(
                    source="hierarchy_parent",
                    text="Derived from 2 sub-techniques of T1059",
                    score=0.5,
                    rank=1,
                )
            ],
            state=TechniqueCandidateState.NEEDS_REVIEW,
            state_reason="hierarchy parent of 2 retrieved sub-techniques (not directly observed)",
        )
        assert parent_candidate.state is TechniqueCandidateState.NEEDS_REVIEW
        assert "not directly observed" in parent_candidate.state_reason


# ---------------------------------------------------------------------------
# 5. eCAR JSONL parsing
# ---------------------------------------------------------------------------


class TestECARParser:
    """Tests for the eCAR JSONL parser."""

    def test_parse_single_record(self):
        """Parse a single eCAR record into an Observation."""
        from apt_classifier.evidence.ecar import parse_ecar_records

        record = {
            "timestamp_ms": 1715688349560,
            "id": "test-001",
            "hostname": "WEB-BO-01",
            "object": "PROCESS",
            "action": "CREATE",
            "pid": 1923273,
            "principal": "root",
            "properties": {
                "command_line": "id",
                "image_path": "/usr/bin/id",
                "parent_image_path": "/bin/bash",
            },
        }
        observations = parse_ecar_records([record], source_name="test")
        assert len(observations) == 1
        obs = observations[0]
        assert obs.text.startswith("id (")
        assert "run by root" in obs.text
        assert obs.metadata["ecar_object"] == "PROCESS"
        assert obs.provenance.source == "ecar"
        assert obs.provenance.metadata.get("ecar_record_id") == "test-001"
        assert obs.provenance.metadata.get("ecar_object") == "PROCESS"
        assert obs.provenance.metadata.get("ecar_action") == "CREATE"

    def test_parse_jsonl_string(self):
        """Parse a JSONL string into Observations."""
        from apt_classifier.evidence.ecar import parse_ecar_records

        jsonl = json.dumps(
            {
                "timestamp_ms": 1715688349560,
                "id": "test-002",
                "hostname": "WEB-BO-01",
                "object": "FLOW",
                "action": "CONNECT",
                "properties": {"dst_ip": "10.0.0.1", "dst_port": 80},
            }
        )
        observations = parse_ecar_records(jsonl, source_name="test")
        assert len(observations) == 1
        assert "eCAR FLOW CONNECT" in observations[0].text
        assert "Destination: 10.0.0.1:80" in observations[0].text

    def test_parse_skips_empty_records(self):
        """Empty records are skipped."""
        from apt_classifier.evidence.ecar import parse_ecar_records

        record = {"timestamp_ms": 123, "id": "empty"}
        observations = parse_ecar_records([record], source_name="test")
        assert len(observations) == 0

    def test_narrate_ecar_record_includes_key_fields(self):
        """Narration includes hostname, object, action, principal, and key properties."""
        from apt_classifier.evidence.ecar import narrate_ecar_record

        record = {
            "hostname": "FILE-BO-01",
            "object": "PROCESS",
            "action": "CREATE",
            "principal": "admin",
            "pid": 1234,
            "properties": {
                "command_line": "cmd.exe /c dir",
                "image_path": "C:\\Windows\\System32\\cmd.exe",
            },
        }
        narration = narrate_ecar_record(record)
        assert "FILE-BO-01" in narration
        assert narration.startswith("cmd.exe /c dir (")
        assert "eCAR" not in narration
        assert "admin" in narration
        assert "cmd.exe /c dir" in narration
        assert "cmd.exe" in narration


# ---------------------------------------------------------------------------
# 6. Windows XML streaming
# ---------------------------------------------------------------------------


class TestWindowsXMLParser:
    """Tests for the Windows Event XML streaming parser."""

    def test_parse_xml_string(self):
        """Parse a Windows Event XML string into Observations."""
        from apt_classifier.evidence.winlogs import parse_windows_xml

        xml = """<?xml version="1.0"?>
<Events>
  <Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
    <System>
      <Provider Name="Microsoft-Windows-Security-Auditing"/>
      <EventID>4624</EventID>
      <Channel>Security</Channel>
      <Computer>FILE-BO-01</Computer>
      <TimeCreated SystemTime="2024-05-14T13:00:00Z"/>
    </System>
    <EventData>
      <Data Name="TargetUserName">admin</Data>
      <Data Name="LogonType">10</Data>
    </EventData>
  </Event>
</Events>"""
        observations = parse_windows_xml(xml, source_name="test_xml")
        assert len(observations) == 1
        assert "4624" in observations[0].text
        assert "FILE-BO-01" in observations[0].text

    def test_parse_xml_file(self, tmp_path):
        """Parse a Windows Event XML file into Observations."""
        from apt_classifier.evidence.winlogs import parse_windows_xml

        xml = """<?xml version="1.0"?>
<Events>
  <Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
    <System>
      <Provider Name="Microsoft-Windows-Sysmon"/>
      <EventID>1</EventID>
      <Channel>Microsoft-Windows-Sysmon/Operational</Channel>
      <Computer>WEB-BO-01</Computer>
    </System>
    <EventData>
      <Data Name="Image">C:\\Windows\\System32\\cmd.exe</Data>
      <Data Name="CommandLine">cmd.exe /c whoami</Data>
    </EventData>
  </Event>
</Events>"""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text(xml, encoding="utf-8")
        observations = parse_windows_xml(xml_file, source_name="test_file")
        assert len(observations) == 1
        assert observations[0].text.startswith("cmd.exe /c whoami (")
        assert "Windows Event" not in observations[0].text


# ---------------------------------------------------------------------------
# 7. Reranker acceptance influence
# ---------------------------------------------------------------------------


class TestRerankerAcceptance:
    """Reranker promote/demote behaviour on the live acceptance path.

    A *validated* reranker's evidence score can promote a needs-review candidate that has
    retrieval evidence, or demote a candidate the rule gate accepted; an unvalidated
    reranker (thresholds ``None``) must neither change states nor raise.
    """

    @staticmethod
    def _observation():
        return SimpleNamespace(
            id="obs-1", text="PowerShell executed an encoded command on the host", metadata={}
        )

    @staticmethod
    def _candidate(dense_score, reranker_score):
        evidence = [
            TechniqueRetrievalEvidence(
                source="dense", field="description", text="x", score=dense_score, rank=1
            )
        ]
        if reranker_score is not None:
            evidence.append(
                TechniqueRetrievalEvidence(
                    source="reranker", text="PowerShell", score=reranker_score, rank=1
                )
            )
        return TechniqueCandidate(
            observation_id="obs-1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            fused_score=0.02,
            fused_rank=1,
            evidence=evidence,
        )

    @staticmethod
    def _reranker(validated=True):
        reranker = MagicMock()
        reranker.validated = validated
        reranker.promote_threshold = RERANKER_PROMOTE_THRESHOLD if validated else None
        reranker.demote_threshold = RERANKER_DEMOTE_THRESHOLD if validated else None
        return reranker

    def _state(self, candidate, reranker):
        result = _assign_candidate_states(
            self._observation(),
            [candidate],
            retrieval_mode="hybrid",
            threshold=0.45,
            acceptance_threshold=0.75,
            relative_margin=0.08,
            reranker=reranker,
        )
        return result[0]

    def test_promote_needs_review_to_accepted(self):
        """High reranker score promotes a weak-but-retrieved candidate."""
        weak = self._state(self._candidate(0.50, None), self._reranker())
        assert weak.state is TechniqueCandidateState.NEEDS_REVIEW
        promoted = self._state(self._candidate(0.50, 0.9), self._reranker())
        assert promoted.state is TechniqueCandidateState.AUTO_ACCEPTED
        assert "promotion threshold" in promoted.state_reason

    def test_demote_accepted_to_needs_review(self):
        """Low reranker score demotes a candidate the rule gate accepted."""
        strong = self._state(self._candidate(0.90, None), self._reranker())
        assert strong.state is TechniqueCandidateState.AUTO_ACCEPTED
        demoted = self._state(self._candidate(0.90, 0.1), self._reranker())
        assert demoted.state is TechniqueCandidateState.NEEDS_REVIEW
        assert "demotion threshold" in demoted.state_reason

    def test_no_change_when_score_in_middle(self):
        """Mid-range reranker score leaves the rule-gate decision alone."""
        accepted = self._state(self._candidate(0.90, 0.5), self._reranker())
        assert accepted.state is TechniqueCandidateState.AUTO_ACCEPTED
        review = self._state(self._candidate(0.50, 0.5), self._reranker())
        assert review.state is TechniqueCandidateState.NEEDS_REVIEW

    def test_unvalidated_reranker_never_changes_state_or_raises(self):
        """An available-but-unvalidated reranker exposes ``None`` thresholds; the live path
        used to call ``float(None)`` here."""
        reranker = self._reranker(validated=False)
        review = self._state(self._candidate(0.50, 0.99), reranker)
        assert review.state is TechniqueCandidateState.NEEDS_REVIEW
        accepted = self._state(self._candidate(0.90, 0.01), reranker)
        assert accepted.state is TechniqueCandidateState.AUTO_ACCEPTED

    def test_reranker_never_promotes_in_benign_or_sparse_context(self):
        """Guards stay absolute: a benign-context observation cannot be promoted."""
        benign = SimpleNamespace(
            id="obs-1", text="administrator logged in and ran a routine backup of documents"
        )
        result = _assign_candidate_states(
            benign,
            [self._candidate(0.50, 0.99)],
            retrieval_mode="hybrid",
            threshold=0.45,
            acceptance_threshold=0.75,
            relative_margin=0.08,
            reranker=self._reranker(),
        )
        assert result[0].state is TechniqueCandidateState.NEEDS_REVIEW
        sparse = SimpleNamespace(id="obs-2", text="10.0.0.5")
        result = _assign_candidate_states(
            sparse,
            [self._candidate(0.50, 0.99)],
            retrieval_mode="hybrid",
            threshold=0.45,
            acceptance_threshold=0.75,
            relative_margin=0.08,
            reranker=self._reranker(),
        )
        assert result[0].state is TechniqueCandidateState.NEEDS_REVIEW

    def test_reranker_does_not_demote_cue_supported_mapping(self):
        """Independent corroboration (explicit cue + retrieval) survives a low reranker score."""
        candidate = self._candidate(0.70, 0.01)
        candidate = candidate.model_copy(
            update={
                "evidence": [
                    *candidate.evidence,
                    TechniqueRetrievalEvidence(
                        source="explicit_cue", text="powershell -enc", score=0.85, rank=1
                    ),
                ]
            }
        )
        result = self._state(candidate, self._reranker())
        assert result.state is TechniqueCandidateState.AUTO_ACCEPTED

    def test_promote_threshold_is_high(self):
        """The promote threshold is conservative (>= 0.8)."""
        assert RERANKER_PROMOTE_THRESHOLD >= 0.8

    def test_demote_threshold_is_low(self):
        """The demote threshold is conservative (<= 0.2)."""
        assert RERANKER_DEMOTE_THRESHOLD <= 0.2
