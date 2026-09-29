"""Curated technique retrieval benchmark.

Runs a set of analyst-style observations through the live ChromaDB technique
index and reports whether the expected ATT&CK technique appears in the top-1,
top-3, and top-10 results. Measures retrieval quality only, not attribution
accuracy. Supports comparing retrieval modes (legacy dense, hybrid, lexical).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evaluation import pipeline_fingerprint, source_fingerprint, write_artifact
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.retrieve import LexicalTechniqueIndex, retrieve_techniques
from apt_classifier.vectorstore import ChromaVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = REPO_ROOT / "data" / "kb" / "attack_kb.json"


@dataclass(frozen=True)
class CuratedCase:
    label: str
    text: str
    expected_technique_ids: tuple[str, ...]
    category: str = "positive"


CASES = (
    # Positive single-label cases
    CuratedCase(
        "PowerShell encoded command",
        "Word spawned PowerShell with an encoded command.",
        ("T1059.001",),
    ),
    CuratedCase(
        "LSASS memory access",
        "A dump utility accessed LSASS memory to extract credentials.",
        ("T1003.001",),
    ),
    CuratedCase(
        "Screen capture", "The adversary took a screenshot of the user's desktop.", ("T1113",)
    ),
    CuratedCase(
        "SMB admin shares",
        "Files were copied to an SMB Windows admin share for lateral movement.",
        ("T1021.002",),
    ),
    CuratedCase(
        "Rundll32 execution", "Suspicious DLL was loaded via rundll32.exe.", ("T1218.011",)
    ),
    CuratedCase("Keylogging", "A keylogger captured user keystrokes.", ("T1056.001",)),
    CuratedCase(
        "Archive via utility",
        "Data was compressed into a zip archive before exfiltration.",
        ("T1560.001",),
    ),
    CuratedCase(
        "DNS tunneling",
        "Suspicious DNS queries with long subdomains suggested tunneling.",
        ("T1071.004",),
    ),
    CuratedCase(
        "Process injection",
        "Code was injected into a running process via CreateRemoteThread.",
        ("T1055",),
    ),
    CuratedCase(
        "Scheduled task creation",
        "A Windows scheduled task was created for persistence.",
        ("T1053.005",),
    ),
    CuratedCase(
        "Credential dumping via registry",
        "Credentials were dumped from the SAM registry hive.",
        ("T1003.002",),
    ),
    CuratedCase(
        "Ingress tool transfer",
        "A payload was downloaded over HTTP from an external server.",
        ("T1105",),
    ),
    # Multi-label cases (one observation describes multiple behaviors)
    CuratedCase(
        "PowerShell + scheduled task",
        "PowerShell created a scheduled task and then downloaded a payload over HTTP.",
        ("T1059.001", "T1053.005", "T1105"),
        category="multi_label",
    ),
    CuratedCase(
        "Credential dump + exfil",
        "LSASS was dumped for credentials and the output was compressed into a zip archive.",
        ("T1003.001", "T1560.001"),
        category="multi_label",
    ),
    # Hard negative / non-mappable cases (should abstain or rank low)
    CuratedCase(
        "Generic admin activity",
        "An administrator logged in and checked email.",
        (),
        category="hard_negative",
    ),
    CuratedCase(
        "Benign software update",
        "The antivirus software updated its signatures automatically.",
        (),
        category="hard_negative",
    ),
    CuratedCase(
        "User password reset",
        "A user reset their own password through the self-service portal.",
        (),
        category="hard_negative",
    ),
    # Hierarchy cases: parent vs sub-technique
    CuratedCase(
        "Command and scripting interpreter (parent)",
        "The adversary used a command-line interpreter to execute scripts.",
        ("T1059",),
        category="hierarchy_parent",
    ),
    CuratedCase(
        "PowerShell sub-technique",
        "PowerShell was used to execute an encoded command.",
        ("T1059.001",),
        category="hierarchy_subtechnique",
    ),
    CuratedCase(
        "OS credential dumping (parent)",
        "The adversary dumped operating system credentials.",
        ("T1003",),
        category="hierarchy_parent",
    ),
    CuratedCase(
        "Confusing sibling sub-technique",
        "Credentials were dumped from the LSASS process memory.",
        ("T1003.001",),
        category="hierarchy_subtechnique",
    ),
    # Additional positive cases — varied phrasing and less common techniques
    CuratedCase(
        "WMI execution",
        "The adversary used wmic to execute a remote process for lateral movement.",
        ("T1047",),
    ),
    CuratedCase(
        "Registry run key persistence",
        "A malicious script was added to the Windows Registry Run key for persistence.",
        ("T1547.001",),
    ),
    CuratedCase(
        "DLL side-loading",
        "A legitimate application was used to side-load a malicious DLL.",
        ("T1574.001",),
    ),
    CuratedCase(
        "Token impersonation",
        "The adversary impersonated an access token to escalate privileges.",
        ("T1134.001",),
    ),
    CuratedCase(
        "Data exfiltration over HTTPS",
        "Stolen data was exfiltrated to an external server over HTTPS.",
        ("T1041",),
    ),
    CuratedCase(
        "Disabling security tools",
        "Windows Defender was disabled using PowerShell to evade detection.",
        ("T1687",),
    ),
    CuratedCase(
        "Credential dumping via ntds.dit",
        "The ntds.dit file was copied for offline credential extraction.",
        ("T1003.003",),
    ),
    CuratedCase(
        "Hidden window execution",
        "A payload was executed with PowerShell in a hidden window.",
        ("T1059.001",),
    ),
    # Additional multi-label cases
    CuratedCase(
        "Lateral movement + credential theft",
        "The adversary used WMI for lateral movement and dumped LSASS credentials on the remote host.",
        ("T1047", "T1003.001"),
        category="multi_label",
    ),
    CuratedCase(
        "Persistence + defense evasion",
        "A scheduled task was created to run a DLL side-loaded via rundll32.",
        ("T1053.005", "T1218.011"),
        category="multi_label",
    ),
    # Additional hard negatives
    CuratedCase(
        "Normal user login",
        "A user logged in to their workstation at the start of the workday.",
        (),
        category="hard_negative",
    ),
    CuratedCase(
        "Software installation",
        "An administrator installed a legitimate application through the software center.",
        (),
        category="hard_negative",
    ),
    CuratedCase(
        "Routine backup",
        "The system performed a scheduled backup of user documents.",
        (),
        category="hard_negative",
    ),
    # Additional hierarchy cases
    CuratedCase(
        "Scheduled task/job (parent)",
        "The adversary created scheduled tasks for execution.",
        ("T1053",),
        category="hierarchy_parent",
    ),
    CuratedCase(
        "System binary proxy execution (parent)",
        "The adversary used system binaries to proxy execution of malicious code.",
        ("T1218",),
        category="hierarchy_parent",
    ),
)


def run_mode(
    store: ChromaVectorStore,
    kb: AttackKnowledgeBase,
    lexical_index: LexicalTechniqueIndex,
    threshold: float,
    retrieval_mode: str,
    parent_policy: str = "ancestor_compatible",
    cues_enabled: bool = True,
) -> dict:
    """Run every curated case in one retrieval mode.

    Ranking metrics (top-1/3/10) are computed over all visible candidates, accepted and
    needs-review alike, so they describe retrieval order. The ``accepted_set`` block
    describes the authoritative AUTO_ACCEPTED mappings that feed attribution.
    Hard negatives are judged by ``accepted_fp_cases`` (a hard negative that accepts any
    technique is a false-positive case); ``abstain_count`` counts hard negatives with
    no visible candidate at all, which the pipeline does not aim for.
    """
    top1 = 0
    top3 = 0
    top10 = 0
    abstain_count = 0
    accepted_tp = accepted_fp = accepted_fn = 0
    accepted_negative_fp_cases = 0
    details = []
    for case in CASES:
        ledger = EvidenceLedger(
            observations=[Observation(text=case.text, provenance=Provenance(source="free_text"))]
        )
        result = retrieve_techniques(
            ledger,
            store,
            kb,
            threshold=threshold,
            retrieval_mode=retrieval_mode,
            parent_policy=parent_policy,
            lexical_index=lexical_index,
            cues_enabled=cues_enabled,
        )
        ranked = sorted(result.matches, key=lambda match: -match.score)
        seen: list[str] = []
        for match in ranked:
            if match.technique_id not in seen:
                seen.append(match.technique_id)
        accepted = set(result.observed_techniques)
        expected = set(case.expected_technique_ids)
        accepted_tp += len(accepted & expected)
        accepted_fp += len(accepted - expected)
        accepted_fn += len(expected - accepted)
        if not case.expected_technique_ids:
            abstain_count += int(not seen)
            accepted_negative_fp_cases += int(bool(accepted))
            details.append(
                {
                    "label": case.label,
                    "category": case.category,
                    "expected": [],
                    "retrieved_top5": seen[:5],
                    "accepted": sorted(accepted),
                    "abstained": not seen,
                    "accepted_fp": bool(accepted),
                }
            )
            continue
        hit1 = any(tid in seen[:1] for tid in case.expected_technique_ids)
        hit3 = any(tid in seen[:3] for tid in case.expected_technique_ids)
        hit10 = any(tid in seen[:10] for tid in case.expected_technique_ids)
        top1 += int(hit1)
        top3 += int(hit3)
        top10 += int(hit10)
        details.append(
            {
                "label": case.label,
                "category": case.category,
                "expected": list(case.expected_technique_ids),
                "retrieved_top5": seen[:5],
                "accepted": sorted(accepted),
                "top1": hit1,
                "top3": hit3,
                "top10": hit10,
            }
        )
    positive_cases = sum(1 for c in CASES if c.expected_technique_ids)
    hard_negative_cases = sum(1 for c in CASES if not c.expected_technique_ids)
    return {
        "mode": retrieval_mode,
        "cues_enabled": cues_enabled,
        "parent_policy": parent_policy,
        "total_cases": len(CASES),
        "positive_cases": positive_cases,
        "hard_negative_cases": hard_negative_cases,
        "top1": top1,
        "top3": top3,
        "top10": top10,
        "abstain_count": abstain_count,
        "abstain_rate": round(abstain_count / hard_negative_cases, 4)
        if hard_negative_cases
        else 0.0,
        "accepted_set": {
            "tp": accepted_tp,
            "fp": accepted_fp,
            "fn": accepted_fn,
            "precision": round(accepted_tp / (accepted_tp + accepted_fp), 4)
            if accepted_tp + accepted_fp
            else 0.0,
            "recall": round(accepted_tp / (accepted_tp + accepted_fn), 4)
            if accepted_tp + accepted_fn
            else 0.0,
            "micro_f1": round(2 * accepted_tp / (2 * accepted_tp + accepted_fp + accepted_fn), 4)
            if 2 * accepted_tp + accepted_fp + accepted_fn
            else 0.0,
            "accepted_fp_cases": accepted_negative_fp_cases,
        },
        "details": details,
    }


def run(
    threshold: float,
    modes: tuple[str, ...] | None = None,
    *,
    ablate_cues: bool = False,
    cues_enabled: bool = True,
) -> dict:
    settings = Settings()
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings=settings, embedder=embedder)
    kb = AttackKnowledgeBase.load(settings.kb_path)
    lexical_index = LexicalTechniqueIndex.from_kb(kb)
    modes = modes or ("legacy_dense", "hybrid")
    cue_settings = (True, False) if ablate_cues else (cues_enabled,)
    results: dict[str, dict] = {}
    for mode in modes:
        for enabled in cue_settings:
            key = mode if enabled else f"{mode}_no_cues"
            results[key] = run_mode(store, kb, lexical_index, threshold, mode, cues_enabled=enabled)
    fingerprint_inputs = {
        "attack_version": kb.version,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "threshold": threshold,
        "settings": {
            "retrieval_top_k": settings.retrieval_top_k,
            "candidate_pool": settings.technique_candidate_pool,
            "acceptance_threshold": settings.technique_acceptance_threshold,
            "relative_margin": settings.technique_relative_margin,
            "parent_policy": settings.technique_parent_policy,
        },
        "source_fingerprint": source_fingerprint(
            (
                REPO_ROOT / "apt_classifier" / "retrieve.py",
                REPO_ROOT / "apt_classifier" / "technique_signals.py",
                REPO_ROOT / "apt_classifier" / "presets.py",
                REPO_ROOT / "apt_classifier" / "vectorstore.py",
            )
        ),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "curated_retrieval",
        "case_count": len(CASES),
        "attack_version": kb.version,
        "pipeline_fingerprint": pipeline_fingerprint(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "results": results,
    }


def _print_mode(mode: str, result: dict) -> None:
    print(f"\n=== Mode: {mode} (cues {'on' if result['cues_enabled'] else 'off'}) ===")
    print(
        f"  cases: {result['total_cases']} (positive: {result['positive_cases']}, "
        f"hard-negative: {result['hard_negative_cases']})"
    )
    print(f"  top-1:  {result['top1']}/{result['positive_cases']}")
    print(f"  top-3:  {result['top3']}/{result['positive_cases']}")
    print(f"  top-10: {result['top10']}/{result['positive_cases']}")
    accepted = result["accepted_set"]
    print(
        f"  accepted-set P/R/F1: {accepted['precision']}/{accepted['recall']}/"
        f"{accepted['micro_f1']}  hard-negative accepted-FP cases: "
        f"{accepted['accepted_fp_cases']}/{result['hard_negative_cases']}"
        f"  (no-candidate abstentions: {result['abstain_count']})"
    )
    for item in result["details"]:
        if item["category"] == "hard_negative":
            print(
                f"  - [{item['category']}] {item['label']}: "
                f"accepted={item['accepted']} top5={item['retrieved_top5']}"
            )
        else:
            print(
                f"  - [{item['category']}] {item['label']}: "
                f"expected={item['expected']} top5={item['retrieved_top5']} "
                f"hit1={item.get('top1', False)} hit3={item.get('top3', False)} "
                f"accepted={item['accepted']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["legacy_dense", "hybrid"],
        choices=["legacy_dense", "hybrid", "dense", "lexical"],
    )
    parser.add_argument("--disable-cues", action="store_true")
    parser.add_argument("--ablate-cues", action="store_true", help="run with and without cues")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.threshold,
        modes=tuple(args.modes),
        ablate_cues=args.ablate_cues,
        cues_enabled=not args.disable_cues,
    )
    if args.output:
        destination = write_artifact(args.output, report)
        print(f"wrote {destination}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for mode, result in report["results"].items():
            _print_mode(mode, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
