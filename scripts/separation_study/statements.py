"""Which kinds of report statement point at the right group? (development report sets only)

Every statement of the vendor, campaign and advisory reports, anonymised at a level exactly as
the attribution benchmark reads it, is matched against ATT&CK's anonymised procedure sentences
(the behaviour-text sentence index over the text member's corpus, with the report's cited and
copied texts and its own statements hidden, as in the benchmark). A statement points at the group
of its nearest corpus sentence (best vote). Statements are tagged by what they contain (a statement
can carry several tags); per tag: how often the group it points at is the report's actor, a
permutation baseline (the actor of another report) and the reports with at least one such hit.
CPU only, about 25 minutes for two levels.

    PYTHONPATH=.:scripts .venv/bin/python scripts/separation_study/statements.py [names tradecraft]
"""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attribution_benchmark import anonymize_report_folds, load_report_folds  # noqa: E402
from report_anonymizer import ReportAnonymizer  # noqa: E402

from apt_classifier.attack.kb import load_knowledge_base  # noqa: E402
from apt_classifier.behavior_text import BehaviorTextModel  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.evidence.iocs import match_software_aliases  # noqa: E402
from apt_classifier.text_group import TextGroupModel  # noqa: E402

BENCHMARKS = REPO_ROOT / "data/benchmarks"
SETS = {
    "vendor": BENCHMARKS / "vendor/reports_hide_cited_attack20.0.jsonl",
    "campaigns": BENCHMARKS / "campaigns/reports_attack20.0.jsonl",
    "advisories": BENCHMARKS / "advisories/reports.jsonl",
}
SOFTWARE = "names ATT&CK software"
PLAIN = "none of these (plain description)"
LOLBINS = (
    r"powershell|cmd(\.exe)?|rundll32|regsvr32|mshta|wmic|schtasks|certutil|bitsadmin|vssadmin|"
    r"netsh|whoami|reg (add|query)|net (user|group|localgroup|use)|nltest|psexec|wscript|cscript|"
    r"curl|wget|bash|chmod|crontab|ssh"
)
TAGS = {
    "command or LOLBin": re.compile(
        rf"(`[^`]+`|\b({LOLBINS})\b|\s[-/][a-zA-Z]{{1,12}}\b.*\.(exe|ps1|bat))", re.I
    ),
    "file name or path": re.compile(
        r"([a-zA-Z]:\\|%\w+%|\\\\|/(tmp|var|etc|usr|home|dev|opt)/|\b[\w-]+\."
        r"(dll|exe|ps1|bat|vbs|js|lnk|zip|rar|7z|iso|docx?|xlsx?|hta|py|sh|jar|sys|tmp|dat|bin|elf)\b)",
        re.I,
    ),
    "registry key": re.compile(r"\b(HKLM|HKCU|HKEY_|CurrentVersion\\Run)", re.I),
    "network indicator": re.compile(
        r"(\b\d{1,3}(\.\d{1,3}){3}\b|https?://|hxxp|\[\.\]|"
        r"\b[\w-]+\.(com|net|org|info|xyz|top|ru|cn|ir|kp|io|cc|biz)\b|\bport \d+)",
        re.I,
    ),
    "victim sector or region": re.compile(
        r"\b(government|ministr\w*|defen[cs]e|military|energy|oil|telecom\w*|financ\w*|bank\w*|"
        r"healthcare|hospital|universit\w*|academ\w*|aerospace|manufactur\w*|embass\w*|"
        r"diplomat\w*|journalist\w*|dissident\w*|cryptocurrenc\w*|retail|hospitality|Asia|Europe|"
        r"Middle East|Africa|America)\b",
        re.I,
    ),
}


def tags(text: str, kb: Any) -> list[str]:
    found = [SOFTWARE] if match_software_aliases(text, kb) else []
    found += [name for name, pattern in TAGS.items() if pattern.search(text)]
    return found or [PLAIN]


def folds_at(kb: Any, level: str) -> dict[str, Any]:
    anonymizer = ReportAnonymizer(kb, level) if level in ("strict", "tradecraft") else None
    out = {}
    for path in SETS.values():
        loaded = load_report_folds(path, max_observations=None, anonymizer=anonymizer, kb=kb)
        loaded, _, _ = anonymize_report_folds(loaded, kb, level=level)
        out.update({fold.report_id: fold for fold in loaded if fold.group_id})
    return out


def main(argv: list[str] | None = None) -> int:
    levels = list(argv if argv is not None else sys.argv[1:]) or ["names", "tradecraft"]
    kb = load_knowledge_base(REPO_ROOT / "data/kb/attack_kb.json")
    model = BehaviorTextModel(TextGroupModel.load(Settings().text_group_path, kb))
    index = {g: i for i, g in enumerate(model.group_ids)}
    per_level = {level: folds_at(kb, level) for level in levels}
    reports = list(per_level[levels[0]].values())
    truths = [fold.group_id for fold in reports]
    rng = random.Random(0)
    # tag -> [statements, actor hits, baseline hits]
    cells = {level: defaultdict(lambda: [0, 0, 0]) for level in levels}
    hit_reports = {level: defaultdict(set) for level in levels}
    for number, fold in enumerate(reports, 1):
        hidden = model.without(kb, group_id=fold.group_id, texts=fold.hidden_raw)
        for level in levels:
            current = per_level[level].get(fold.report_id)
            if current is None:
                continue
            texts = [text for text in current.observations if text.strip()]
            for text, votes in zip(texts, hidden.votes(texts), strict=True):
                if not votes.any() or current.group_id not in index:
                    continue
                pointed = model.group_ids[int(np.argmax(votes))]
                other = rng.choice([t for t in truths if t != current.group_id])
                right = pointed == current.group_id or pointed in current.accept
                for tag in tags(text, kb):
                    cell = cells[level][tag]
                    cell[0] += 1
                    cell[1] += right
                    cell[2] += pointed == other
                    if right:
                        hit_reports[level][tag].add(current.report_id)
        print(f"  {number}/{len(reports)} {fold.report_id}", file=sys.stderr, flush=True)
    for level in levels:
        total = len(per_level[level])
        print(
            f"\n== {level}: {total} reports; the nearest ATT&CK procedure sentence is the actor's"
        )
        print(
            f"  {'statement contains':34s} {'statements':>10s} {'actor':>7s} {'baseline':>9s} "
            f"{'reports with a hit':>19s}"
        )
        ordered = sorted(cells[level].items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0]))
        for tag, (n, hits, baseline) in ordered:
            print(
                f"  {tag:34s} {n:>10d} {hits / n:>7.1%} {baseline / n:>9.1%} "
                f"{len(hit_reports[level][tag]):>12d}/{total}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
