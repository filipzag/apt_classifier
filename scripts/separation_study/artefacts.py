"""Do exact procedure artefacts point at the right group? (development report sets only)

Every statement of the vendor, campaign and advisory reports, anonymised at a level exactly as
the attribution benchmark reads it, is searched for exact artefacts (registry keys, file paths and
names, pipes, mutexes, service and task names, PDB paths, user agents, export names;
``apt_classifier/evidence/artefacts.py``). The same extractor runs over the text member's ATT&CK
corpus, with each report's cited and copied texts and its own statements hidden as in the
benchmark (``TextGroupModel.without``), and each artefact is credited to the groups of the
records that carry it. Groups are ranked by the summed rarity of the artefacts they share with the
report. Per level: coverage, the actor's rank, how often each kind of matched artefact belongs to
the actor against a permutation baseline, and the reports where the artefacts put the actor first
while every text member missed (their ranks come from the ``run_reports.sh`` runs). CPU, minutes.

    PYTHONPATH=.:scripts .venv/bin/python scripts/separation_study/artefacts.py \
        --runs data/audits/separation_study [--levels names strict tradecraft]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from statements import folds_at  # noqa: E402

from apt_classifier.attack.kb import load_knowledge_base  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.evidence.artefacts import KINDS, ArtefactIndex, extract_artefacts  # noqa: E402
from apt_classifier.text_group import TextGroupModel  # noqa: E402

TEXT_MEMBERS = ("text_group", "behavior_text", "procedure_match")
SETS = ("vendor", "campaigns", "advisories")


def software_aliases(kb: Any) -> frozenset[str]:
    names: set[str] = set()
    for software in (getattr(kb, "software", {}) or {}).values():
        for alias in [getattr(software, "name", None), *(getattr(software, "aliases", None) or [])]:
            if alias and len(alias) >= 4:
                names.add(re.sub(r"[^a-z0-9]", "", alias.lower()))
    return frozenset(names)


def names_software(value: str, aliases: frozenset[str]) -> bool:
    """A file or path whose name is an ATT&CK software name is a software lookup in disguise."""
    for part in re.split(r"[\\/ ]", value):
        stem = re.sub(r"[^a-z0-9]", "", part.split(".")[0].lower())
        if stem in aliases:
            return True
    return False


def member_ranks(runs: Path, level: str) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}
    for name in SETS:
        path = runs / f"{name}_{level}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["known"]:
            out[row["report_id"]] = row.get("classifier_truth_ranks") or {}
    return out


def strict_rank(scores: dict[str, float], right: set[str]) -> int | None:
    """The actor's rank with ties counted against it; ``None`` when it scores nothing."""
    best = max((scores.get(g, 0.0) for g in right), default=0.0)
    if best <= 0:
        return None
    return 1 + sum(1 for g, v in scores.items() if g not in right and v >= best)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "data/audits/separation_study")
    parser.add_argument("--levels", nargs="+", default=["names", "strict", "tradecraft"])
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/audits/separation_study/artefacts.json"
    )
    args = parser.parse_args(argv)

    kb = load_knowledge_base(REPO_ROOT / "data/kb/attack_kb.json")
    model = TextGroupModel.load(Settings().text_group_path, kb)
    if model is None:
        print("text member corpus did not load", file=sys.stderr)
        return 2
    aliases = software_aliases(kb)
    cache: dict[str, frozenset] = {}
    full = ArtefactIndex(model.records, model.group_ids, max_groups=args.max_groups, cache=cache)
    corpus_kinds = Counter(key.split(":", 1)[0] for key in full.groups_of)
    specific_kinds = Counter(key.split(":", 1)[0] for key in full.weight)
    print("== ATT&CK corpus artefacts (distinct; specific = at most", args.max_groups, "groups)")
    for kind in KINDS:
        print(f"  {kind:11s} {corpus_kinds[kind]:>6d} {specific_kinds[kind]:>6d}")

    per_level = {level: folds_at(kb, level) for level in args.levels}
    reports = list(per_level[args.levels[0]].values())
    truths = [fold.group_id for fold in reports]
    rng = random.Random(0)
    rows: dict[str, list[dict[str, Any]]] = {level: [] for level in args.levels}
    # level -> kind -> [matched artefacts, actor's, baseline's]
    kinds: dict[str, dict[str, list[int]]] = {
        level: defaultdict(lambda: [0, 0, 0]) for level in args.levels
    }
    for number, fold in enumerate(reports, 1):
        own = {t for level in args.levels for t in per_level[level][fold.report_id].observations}
        hidden = model.without(kb, group_id=fold.group_id, texts=set(fold.hidden_raw) | own)
        index = ArtefactIndex(
            hidden.records, hidden.group_ids, max_groups=args.max_groups, cache=cache
        )
        other = rng.choice([t for t in truths if t != fold.group_id])
        for level in args.levels:
            current = per_level[level][fold.report_id]
            right = {current.group_id, *(current.accept or ())}
            found = set()
            for text in current.observations:
                found |= extract_artefacts(text)
            matches = index.matches(found)
            pure = [m for m in matches if not names_software(m[0].value, aliases)]
            scores = index.scores(a for a, _, _ in matches)
            pure_scores = index.scores(a for a, _, _ in pure)
            for artefact, _weight, groups in matches:
                lookup = names_software(artefact.value, aliases)
                kind = artefact.kind + (" (software name)" if lookup else "")
                cell = kinds[level][kind]
                cell[0] += 1
                cell[1] += bool(right & groups)
                cell[2] += other in groups
            leader = max(scores, key=scores.get) if scores else None
            rows[level].append(
                {
                    "report_id": current.report_id,
                    "set": current.report_id.split("-")[0],
                    "group_id": current.group_id,
                    "artefacts": len(found),
                    "matched": len(matches),
                    "rank": strict_rank(scores, right),
                    "rank_pure": strict_rank(pure_scores, right),
                    "leader": leader,
                    "leader_pure": max(pure_scores, key=pure_scores.get) if pure_scores else None,
                    "leader_score": round(scores[leader], 3) if leader else 0.0,
                    "truth_score": round(max((scores.get(g, 0.0) for g in right), default=0.0), 3),
                    "matched_detail": [
                        {
                            "artefact": a.key(),
                            "weight": round(w, 3),
                            "actor": bool(right & g),
                            "groups": len(g),
                        }
                        for a, w, g in matches
                    ],
                }
            )
        print(f"  {number}/{len(reports)} {fold.report_id}", file=sys.stderr, flush=True)

    summary: dict[str, Any] = {}
    for level in args.levels:
        level_rows = rows[level]
        members = member_ranks(args.runs, level)
        total = len(level_rows)
        print(f"\n== {level}: {total} reports")
        print(
            f"  reports with any artefact {sum(r['artefacts'] > 0 for r in level_rows)}, "
            f"with an artefact ATT&CK holds for at most {args.max_groups} groups "
            f"{sum(r['matched'] > 0 for r in level_rows)}"
        )
        for label, key in (("all artefacts", "rank"), ("without software-name files", "rank_pure")):
            ranks = [r[key] for r in level_rows]
            top1 = sum(1 for k in ranks if k == 1)
            top5 = sum(1 for k in ranks if k and k <= 5)
            mrr = sum(1 / k for k in ranks if k) / total
            spoke = sum(1 for r in level_rows if r["leader" if key == "rank" else "leader_pure"])
            print(
                f"  {label:28s} top-1 {top1:>2d}/{total}  top-5 {top5:>2d}/{total}  MRR {mrr:.3f}"
                f"  (actor scored at all: {sum(1 for k in ranks if k)}; ranks anyone: {spoke})"
            )
        print(f"  {'matched artefact kind':32s} {'matches':>8s} {'actor':>7s} {'baseline':>9s}")
        for kind, (n, hits, base) in sorted(kinds[level].items(), key=lambda kv: -kv[1][0]):
            print(f"  {kind:32s} {n:>8d} {hits / n:>7.1%} {base / n:>9.1%}")
        rescued, wrong_lead, both = [], [], 0
        for r in level_rows:
            text = members.get(r["report_id"], {})
            text_first = any(text.get(m) == 1 for m in TEXT_MEMBERS)
            if r["rank"] == 1 and not text_first:
                rescued.append(r["report_id"])
            if r["rank"] == 1 and text_first:
                both += 1
            if r["leader"] and r["rank"] != 1:
                wrong_lead.append(
                    (r["report_id"], r["leader"], r["leader_score"], r["truth_score"])
                )
        print(
            f"  artefacts first: {both} also had a text member first; {len(rescued)} had none "
            f"({', '.join(rescued) or '-'}); runs found for {len(members)}/{total} reports"
        )
        print(f"  artefacts led with another group on {len(wrong_lead)} reports:")
        for report_id, leader, score, truth_score in sorted(wrong_lead, key=lambda x: -x[2]):
            print(f"    {report_id:28s} leader {leader} {score:6.2f}  actor {truth_score:6.2f}")
        summary[level] = {
            "reports": total,
            "with_artefacts": sum(r["artefacts"] > 0 for r in level_rows),
            "with_matched": sum(r["matched"] > 0 for r in level_rows),
            "top1": sum(1 for r in level_rows if r["rank"] == 1),
            "top1_pure": sum(1 for r in level_rows if r["rank_pure"] == 1),
            "rescued": rescued,
            "wrong_lead": len(wrong_lead),
            "kinds": {k: v for k, v in kinds[level].items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "max_groups": args.max_groups,
                "corpus_kinds": dict(corpus_kinds),
                "specific_kinds": dict(specific_kinds),
                "summary": summary,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
