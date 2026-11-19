"""Which member, and which kind of evidence, puts the right group first on the development reports?

Reads the runs of ``run_reports.sh`` (vendor, campaigns and advisories at the names, strict and
tradecraft anonymization levels, every member on, each member's rank of the actor recorded as
``classifier_truth_ranks``) and adds three evidence-only rankings computed from what the pipeline
extracted (``software_ids``, ``accepted_techniques``) against the ATT&CK profiles: the named
software by rarity, the techniques by IDF cosine, and the rare techniques (documented for at most
five groups). Per level: all reports, those naming ATT&CK software and those naming none. The
"ensemble" row is the plain RRF over every member (the learned combiner is inactive in these
runs), not the product's default.

    PYTHONPATH=. .venv/bin/python scripts/separation_study/members.py [--dir data/audits/separation_study]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import load_knowledge_base  # noqa: E402

SETS = ("vendor", "campaigns", "advisories")
LEVELS = ("names", "strict", "tradecraft")
OFF = "off"


class Evidence:
    """Rankings from the extracted evidence alone, against the ATT&CK profiles."""

    def __init__(self, kb: Any) -> None:
        self.techniques = {g: set(v) for g, v in kb.group_techniques.items() if v}
        self.software = {g: set(kb.group_software.get(g, ())) for g in kb.groups}
        count = len(self.techniques)
        self.df = Counter(t for v in self.techniques.values() for t in v)
        self.sdf = Counter(s for v in self.software.values() for s in v)
        self.idf = {t: math.log((count + 1) / (n + 1)) + 1 for t, n in self.df.items()}
        self.norm = {
            g: math.sqrt(sum(self.idf[t] ** 2 for t in v)) for g, v in self.techniques.items()
        }

    @staticmethod
    def _rank(scores: dict[str, float], truth: str) -> int | None:
        value = scores.get(truth, 0.0)
        if value <= 0:
            return None
        return 1 + sum(1 for other in scores.values() if other > value)

    def names_software(self, row: dict[str, Any]) -> bool:
        return any(self.sdf.get(s) for s in row.get("software_ids") or [])

    def ranks(self, row: dict[str, Any]) -> dict[str, int | None]:
        truth = row["group_id"]
        named = [s for s in row.get("software_ids") or [] if self.sdf.get(s)]
        known = {t for t in row.get("accepted_techniques") or [] if t in self.df}
        rare = {t for t in known if self.df[t] <= 5}
        out: dict[str, int | None] = {"evidence: software by rarity": None}
        if named:
            out["evidence: software by rarity"] = self._rank(
                {
                    g: sum(1 / self.sdf[s] for s in named if s in v)
                    for g, v in self.software.items()
                },
                truth,
            )
        out["evidence: techniques IDF cosine"] = None
        if known:
            case_norm = math.sqrt(sum(self.idf[t] ** 2 for t in known))
            out["evidence: techniques IDF cosine"] = self._rank(
                {
                    g: sum(self.idf[t] ** 2 for t in known & v) / (case_norm * self.norm[g])
                    for g, v in self.techniques.items()
                },
                truth,
            )
        out["evidence: rare techniques only"] = (
            self._rank(
                {g: sum(self.idf[t] for t in rare & v) for g, v in self.techniques.items()}, truth
            )
            if rare
            else None
        )
        return out


def load(directory: Path, set_name: str, level: str) -> list[dict[str, Any]]:
    path = directory / f"{set_name}_{level}.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get("known") or []
    return [row for row in rows if row.get("group_id")]


def table(rows: list[dict[str, Any]], title: str, members: list[str], evidence: Evidence) -> None:
    n = len(rows)
    ranks: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        recorded = dict(row.get("classifier_truth_ranks") or {})
        for member in members:
            ranks[member].append(recorded.get(member) if member in recorded else OFF)
        ranks["ensemble (plain RRF, every member)"].append(row.get("rank"))
        for name, value in evidence.ranks(row).items():
            ranks[name].append(value)
    print(f"\n== {title}: {n} reports")
    print(f"  {'signal':38s} {'answers':>9s} {'top-1':>8s} {'top-5':>8s} {'MRR':>6s}")
    lines = []
    for name, values in ranks.items():
        answered = [v for v in values if v not in (None, OFF)]
        applicable = [v for v in values if v != OFF]
        top1 = sum(1 for v in answered if v == 1)
        top5 = sum(1 for v in answered if v <= 5)
        mrr = sum(1 / v for v in answered) / n if n else 0.0
        lines.append(
            (
                -top1,
                -mrr,
                f"  {name:38s} {len(answered):>4d}/{len(applicable):<4d} {top1:>4d}/{n:<3d} "
                f"{top5:>4d}/{n:<3d} {mrr:6.3f}",
            )
        )
    for *_, line in sorted(lines):
        print(line)
    some = sum(
        1 for row in rows if any(v == 1 for v in (row.get("classifier_truth_ranks") or {}).values())
    )
    print(f"  some member has the actor first: {some}/{n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "data/audits/separation_study")
    args = parser.parse_args(argv)
    kb = load_knowledge_base(REPO_ROOT / "data/kb/attack_kb.json")
    evidence = Evidence(kb)
    runs = {level: [r for s in SETS for r in load(args.dir, s, level)] for level in LEVELS}
    members = sorted(
        {m for rows in runs.values() for r in rows for m in (r.get("classifier_truth_ranks") or {})}
    )
    for level, rows in runs.items():
        if not rows:
            continue
        table(rows, f"development reports, {level}", members, evidence)
        named = [r for r in rows if evidence.names_software(r)]
        table(named, f"{level}: reports naming ATT&CK software", members, evidence)
        table(
            [r for r in rows if not evidence.names_software(r)],
            f"{level}: reports naming no ATT&CK software",
            members,
            evidence,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
