"""Structured behavior signals for ATT&CK technique mapping.

These signals complement dense/lexical retrieval by recognizing broadly
recognizable command and event patterns that map to specific ATT&CK
techniques. Each signal is guarded against benign counterexamples so that
ordinary activity does not become an accepted ATT&CK label by itself.

Design principles:
- Signals produce review candidates, not auto-accepted labels. A signal
  alone never accepts a technique; it must be combined with independent
  dense or lexical support, or an analyst/verifier decision.
- Every signal has a benign guard that suppresses matches on ordinary
  activity (routine backups, normal logins, software updates, etc.).
- Signals preserve exact provenance and rationale for auditability.
- Signals are generalizable: they match command and event semantics, not
  EvidenceForge-specific strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechniqueSignal:
    """A structured behavior signal that maps observed text to a technique."""

    technique_id: str
    pattern: re.Pattern[str]
    rationale: str
    benign_guard: re.Pattern[str] | None = None
    requires_context: bool = False

    def match(self, text: str) -> str | None:
        """Return the matched span if the signal fires, else None.

              The signal is suppressed if the benign guard matches the text. When
              ``requires_context`` is set, the signal only fires when the text also
              contains adversarial context cues (attacker, compromised, malicious,
        adversary, suspicious, unauthorized, etc.).
        """
        if self.benign_guard is not None and self.benign_guard.search(text):
            return None
        match = self.pattern.search(text)
        if match is None:
            return None
        if self.requires_context and not _ADVERSARIAL_CONTEXT.search(text):
            return None
        return match.group(0)


_ADVERSARIAL_CONTEXT = re.compile(
    r"\b(?:attacker|adversary|compromised|malicious|unauthorized|suspicious"
    r"|intrusion|threat|implant|beacon|c2|c&c|command\s+and\s+control"
    r"|payload|exploit|staged|exfil|lateral|pivot|foothold)\b",
    re.IGNORECASE,
)


_BENIGN_LOGON_GUARD = re.compile(
    r"\b(?:normal\s+logon|routine\s+logon|scheduled\s+task\s+logon"
    r"|service\s+logon|system\s+logon|boot\s+logon|automatic\s+logon"
    r"|legitimate\s+(?:user|admin|logon|login)|normal\s+user\s+activity)\b",
    re.IGNORECASE,
)

_BENIGN_SCAN_GUARD = re.compile(
    r"\b(?:antivirus\s+(?:scan|update)|defender\s+(?:scan|update)|security\s+scan"
    r"|vulnerability\s+(?:management|program|assessment\s+report)"
    r"|routine\s+(?:scan|check)|scheduled\s+scan|patch\s+(?:scan|management))\b",
    re.IGNORECASE,
)

_BENIGN_FILE_GUARD = re.compile(
    r"\b(?:backup\s+(?:of|job|task|operation)|routine\s+(?:backup|archive)"
    r"|software\s+(?:installation|update|deployment)|legitimate\s+(?:file|user)\s+access"
    r"|normal\s+(?:file|user)\s+activity|document\s+(?:archive|backup))\b",
    re.IGNORECASE,
)

_BENIGN_NETWORK_GUARD = re.compile(
    r"\b(?:legitimate\s+(?:connection|traffic|user)|normal\s+(?:connection|traffic|browsing)"
    r"|software\s+update|windows\s+update|app\s+store|cdn|content\s+delivery"
    r"|routine\s+(?:connection|traffic|monitoring)|scheduled\s+(?:backup|sync))\b",
    re.IGNORECASE,
)

_BENIGN_DISCOVERY_GUARD = re.compile(
    r"\b(?:legitimate\s+(?:admin|monitoring|inventory)|routine\s+(?:check|inventory|audit)"
    r"|asset\s+(?:inventory|management)|system\s+(?:health|status)\s+check"
    r"|administrator\s+(?:check|checked|normal))\b",
    re.IGNORECASE,
)


SIGNALS: tuple[TechniqueSignal, ...] = (
    # T1595.002 — Active Scanning: Vulnerability Scanning
    TechniqueSignal(
        technique_id="T1595.002",
        pattern=re.compile(
            r"\b(?:nikto|nessus|nuclei|openvas|qualys|rapid7|burp\s+suite"
            r"|acunetix|w3af|arachni|zap|zaproxy|sqlmap|dirb|gobuster|ffuf"
            r"|nmap\s+(?:-sV|--script\s+(?:vuln|http)|--version-intensity)"
            r"|vulnerability\s+scan\w*|vuln\s+scan\w*)\b",
            re.IGNORECASE,
        ),
        rationale="vulnerability scanner tool or preset detected",
        benign_guard=_BENIGN_SCAN_GUARD,
    ),
    # T1595.001 — Active Scanning (IP blocks) — kept separate from .002
    TechniqueSignal(
        technique_id="T1595.001",
        pattern=re.compile(
            r"\b(?:nmap\s+(?:-sn|-sP|--ping)|masscan|rustscan|zmap"
            r"|ip\s+block\s+scan|network\s+sweep|host\s+sweep)\b",
            re.IGNORECASE,
        ),
        rationale="network-wide scanning tool detected",
        benign_guard=_BENIGN_SCAN_GUARD,
    ),
    # T1033 — System Owner/User Discovery
    TechniqueSignal(
        technique_id="T1033",
        pattern=re.compile(
            r"(?:^|[\s/])\bwhoami\b(?:\s+(?:/all|/groups|/priv|/user|/upn|/claims))?"
            r"|(?:^|[\s/])\bid\b(?:\s+(?:-u|-g|-G|-un))?(?!\s*=\s)"
            r"|\becho\s+(?:%USERNAME%|%USERDOMAIN%|%USERPROFILE%)"
            r"|\$env:USERNAME\b|\$env:USER\b"
            r"|\bgetenv\s+(?:USER|USERNAME|LOGNAME)\b"
            r"|\bquery\s+user\b|\bqwinsta\b",
            re.IGNORECASE,
        ),
        rationale="identity discovery command detected (whoami/id/USERNAME)",
        benign_guard=_BENIGN_DISCOVERY_GUARD,
    ),
    # T1016 — System Network Configuration Discovery
    TechniqueSignal(
        technique_id="T1016",
        pattern=re.compile(
            r"\bip\s+addr\b|\bipconfig(?:\s+/all)?\b|\bifconfig\b"
            r"|\broute\s+(?:print|show|-n)\b|\bnetsh\s+interface\s+(?:show|ipv4|ipv6)\b"
            r"|\bcat\s+/etc/(?:hosts|resolv\.conf|networks|nsswitch\.conf)\b"
            r"|\bnetsh\s+interface\s+ip\s+show\s+(?:config|addresses)\b"
            r"|\bip\s+(?:link|addr|route)\s+(?:show|list)\b",
            re.IGNORECASE,
        ),
        rationale="network configuration discovery command detected",
        benign_guard=_BENIGN_DISCOVERY_GUARD,
    ),
    # T1049 — System Network Connections Discovery
    TechniqueSignal(
        technique_id="T1049",
        pattern=re.compile(
            r"\bss\s+(?:-t|-u|-tulpn|-tan|-tunap)\b|\bss\s+-\w*[tunlp]\b"
            r"|\bnetstat\s+(?:-ano|-an|-tunlp|-tulpen|-tlnp)\b"
            r"|\bnetstat\s+-\w*[ant]\b",
            re.IGNORECASE,
        ),
        rationale="network connection enumeration command detected (ss/netstat)",
        benign_guard=_BENIGN_DISCOVERY_GUARD,
    ),
    # T1046 — Network Service Discovery
    TechniqueSignal(
        technique_id="T1046",
        pattern=re.compile(
            r"\bnmap\b(?:\s+-[sSvA]|--script|\s+\d+\.\d+\.\d+\.\d+)?"
            r"|\bmasscan\b|\brustscan\b|\bnmap\s+-p\b"
            r"|\bport\s+scan(?:ner|ning)?\b|\bservice\s+scan(?:ner|ning)?\b"
            r"|\bnc\s+-zv?\b|\bncat\s+-zv?\b"
            r"|\bscan\s+(?:target|host|subnet|network|segment)\b"
            r"|\b(?:port|service)\s+scan\w*[^.]{0,80}\bports?\s*[:=]\s*\[?\d+"
            r"|\bports?\s*[:=]\s*\[\s*\d+(?:\s*,\s*\d+){1,}\s*\]",
            re.IGNORECASE,
        ),
        rationale="network service scanning tool or action detected",
        benign_guard=_BENIGN_SCAN_GUARD,
    ),
    # T1021.001 — Remote Services: Remote Desktop Protocol
    TechniqueSignal(
        technique_id="T1021.001",
        pattern=re.compile(
            r"\bmstsc\.exe\b|\bmstsc\b(?:\s+/v:)?"
            r"|\brdp\s+(?:session|connection|login|logon|from|to)\b"
            r"|\bremote\s+desktop\s+(?:protocol|session|connection)\b"
            r"|\bLogonType\s*=?\s*10\b|\blogon\s+type\s*10\b"
            r"|\b:3389\b|\bport\s*3389\b|\b3389/tcp\b"
            r"|\bTerminalServices-LocalSessionManager\b"
            r"|\bMicrosoft-Windows-TerminalServices\b",
            re.IGNORECASE,
        ),
        rationale="RDP session or logon type 10 detected",
        benign_guard=_BENIGN_LOGON_GUARD,
    ),
    # T1083 — File and Directory Discovery
    TechniqueSignal(
        technique_id="T1083",
        pattern=re.compile(
            r"\bdir\s+/s\b|\bdir\s+/s\b|\bdir\s+-R\b"
            r"|\bls\s+-R\b|\bls\s+--recursive\b"
            r"|\bfind\s+(?:/|-name|-path|-type)\b"
            r"|\btree\s+(?:/f|/a|/h)\b"
            r"|\bGet-ChildItem\s+-Recurse\b|\bGet-ChildItem\s+-r\b"
            r"|\bdir\s+\\\\[\w.-]+\\[\w.-]+\b"
            r"|\bcmd\.exe\s+/c\s+dir\b",
            re.IGNORECASE,
        ),
        rationale="recursive file/directory enumeration command detected",
        benign_guard=_BENIGN_FILE_GUARD,
    ),
    TechniqueSignal(
        technique_id="T1560.001",
        pattern=re.compile(
            r"\bCompress-Archive\b[^\r\n]{0,240}\b(?:DestinationPath|\.zip)\b"
            r"|\b(?:7z|7za|7zr)\b[^\r\n]{0,120}\ba\b[^\r\n]{0,160}\.(?:7z|zip|rar)\b"
            r"|\b(?:tar|zip|rar|winrar|makecab|diantz)\b[^\r\n]{0,160}"
            r"\b(?:create|archive|compress|\.tar|\.tgz|\.zip|\.rar|\.cab)\b",
            re.IGNORECASE,
        ),
        rationale="archive utility command created a collection archive",
        benign_guard=_BENIGN_FILE_GUARD,
    ),
    # T1071.001 — Application Layer Protocol: Web Protocols (C2 beaconing)
    TechniqueSignal(
        technique_id="T1071.001",
        pattern=re.compile(
            r"\bbeacons?\s+(?:to|through|via|over|through\s+the\s+proxy)\b"
            r"|\bbeaconing\s+(?:to|through|via|over|through\s+the\s+proxy)\b"
            r"|\bhttps?\s+(?:beacon|callback|checkin|check-in)\b"
            r"|\bregular\s+https?\s+(?:callback|beacon|checkin)\b"
            r"|\bc2\s+over\s+https?\b|\bhttps?\s+c2\b"
            r"|\bperiodic\s+https?\s+(?:connection|request|callback)\b"
            r"|\bheartbeat\s+(?:to|through)\b",
            re.IGNORECASE,
        ),
        rationale="HTTPS beaconing or C2 callback pattern detected",
        benign_guard=_BENIGN_NETWORK_GUARD,
        requires_context=True,
    ),
    # T1078.003 — Valid Accounts: Local Accounts
    TechniqueSignal(
        technique_id="T1078.003",
        pattern=re.compile(
            r"\b(?:compromised|stolen|valid)\s+local\s+(?:admin\s+)?(?:account|credentials)"
            r"[^.]{0,100}\b(?:ssh|rdp|remote\s+(?:access|login|logon)|authenticated)\b"
            r"|\b(?:ssh|rdp|remote\s+(?:access|login|logon))\b[^.]{0,100}"
            r"\b(?:compromised|stolen|valid)\s+local\s+(?:admin\s+)?(?:account|credentials)\b",
            re.IGNORECASE,
        ),
        rationale="successful remote access with a compromised local account",
        benign_guard=_BENIGN_LOGON_GUARD,
        requires_context=True,
    ),
    # T1059.004 — Unix Shell (covers bash/sh command execution context)
    TechniqueSignal(
        technique_id="T1059.004",
        pattern=re.compile(
            r"\b(?:bash|sh|zsh|dash)\s+-c\b|\b/bin/(?:bash|sh|zsh|dash)\b"
            r"|\bsudo\s+(?:bash|sh|/bin/bash)\b",
            re.IGNORECASE,
        ),
        rationale="Unix shell command execution detected",
        benign_guard=_BENIGN_DISCOVERY_GUARD,
    ),
    # T1059 — Command and Scripting Interpreter (parent, generic)
    TechniqueSignal(
        technique_id="T1059",
        pattern=re.compile(
            r"\bcmd\.exe\s+/c\b|\bcmd\s+/c\b|\bpowershell\.exe\s+-Command\b"
            r"|\bpowershell\.exe\s+-NoProfile\b|\bpwsh\s+-Command\b",
            re.IGNORECASE,
        ),
        rationale="scripting interpreter command execution detected",
        benign_guard=_BENIGN_DISCOVERY_GUARD,
    ),
)


def find_signals(text: str) -> list[tuple[TechniqueSignal, str]]:
    """Return all signals that fire on the given text.

    Returns a list of (signal, matched_span) pairs. A signal fires only if
    its pattern matches and its benign guard does not match. Signals with
    ``requires_context`` only fire when adversarial context is present.
    """
    results: list[tuple[TechniqueSignal, str]] = []
    for signal in SIGNALS:
        matched = signal.match(text)
        if matched is not None:
            results.append((signal, matched))
    return results


def signal_for_technique(technique_id: str, text: str) -> str | None:
    """Return the matched span for a specific technique, or None."""
    for signal in SIGNALS:
        if signal.technique_id != technique_id:
            continue
        matched = signal.match(text)
        if matched is not None:
            return matched
    return None
