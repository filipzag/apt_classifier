"""Report-level group ranking from the actor head's per-statement predictions, and the TF-IDF
text-group baseline on the same anonymised statements (prereg ``cribl_actor_head_2026-10-16.json``).

  aggregate: python group_text_eval.py aggregate --name T1 --split actor_cti_hal
             reads <name>_<split>.jsonl (family_head.py evaluate --top-kept 160) and the case file;
             a report's score for a group is the mean of the group's probability over the report's
             statements (a statement without the group in its kept list counts 0); writes
             <name>_group_text_<set>.jsonl in the prediction format (parent_rescore.py compares)
  tfidf:     python group_text_eval.py tfidf --split actor_cti_hal   (project .venv)
             the product's text-group model (apt_classifier.text_group) scoring each report's
             statements joined into one narrative; writes TFIDF_group_text_<set>.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from common import read_jsonl, write_jsonl  # noqa: E402
from group_baseline import ranked_row  # noqa: E402

AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"


def set_name(split: str) -> str:
    return split.removeprefix("actor_")


def aggregate_reports(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
) -> list[dict[str, Any]]:
    """One ranked row per report: mean probability per group over the report's statements."""
    report_of = {row["case_id"]: row["report_id"] for row in cases}
    gold_of = {row["report_id"]: row["gold_ids"] for row in cases}
    sums: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(groups, 0.0))
    counts: dict[str, int] = defaultdict(int)
    for row in predictions:
        if row.get("status") != "ok":
            continue
        report = report_of[row["case_id"]]
        counts[report] += 1
        for group, p in row["top"]:
            if group in sums[report]:
                sums[report][group] += float(p)
    out = []
    for report in sorted(counts):
        scores = {g: s / counts[report] for g, s in sums[report].items()}
        out.append(
            ranked_row(
                {"case_id": report, "dataset": "report", "gold_ids": gold_of[report]}, scores
            )
        )
    return out


def aggregate(args: argparse.Namespace) -> None:
    import json

    cases = read_jsonl(AUDITS / f"cases_{args.split}.jsonl")
    predictions = read_jsonl(AUDITS / f"{args.name}_{args.split}.jsonl")
    groups = [
        c["id"] for c in json.loads((AUDITS / "group_catalog.json").read_text())["techniques"]
    ]
    rows = aggregate_reports(cases, predictions, groups)
    path = AUDITS / f"{args.name}_group_text_{set_name(args.split)}.jsonl"
    write_jsonl(path, rows)
    ranks = [r["gold_rank"] or 10**6 for r in rows]
    statement_top1 = sum(
        1 for p in predictions if p.get("status") == "ok" and p["pick"] in p["gold_ids"]
    ) / max(1, len(predictions))
    print(
        f"{path.name}: reports {len(rows)} top-1 {sum(r == 1 for r in ranks) / len(ranks):.3f} "
        f"top-3 {sum(r <= 3 for r in ranks) / len(ranks):.3f} top-5 {sum(r <= 5 for r in ranks) / len(ranks):.3f} "
        f"MRR {sum(1 / r for r in ranks) / len(ranks):.3f} | statement top-1 {statement_top1:.3f}"
    )


def tfidf(args: argparse.Namespace) -> None:
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.text_group import TextGroupModel

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    model = TextGroupModel.load(settings.text_group_path, kb)
    if model is None:
        raise SystemExit(
            f"text-group model missing: {settings.text_group_path} (make fit-text-group)"
        )
    cases = read_jsonl(AUDITS / f"cases_{args.split}.jsonl")
    by_report: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cases:
        by_report[row["report_id"]].append(row)
    rows = []
    for report, items in sorted(by_report.items()):
        scores = model.scores(" ".join(item["text"] for item in items))
        full = dict.fromkeys(model.group_ids, 0.0)
        full.update(scores)
        rows.append(
            ranked_row(
                {"case_id": report, "dataset": "report", "gold_ids": items[0]["gold_ids"]}, full
            )
        )
    path = AUDITS / f"TFIDF_group_text_{set_name(args.split)}.jsonl"
    write_jsonl(path, rows)
    ranks = [r["gold_rank"] or 10**6 for r in rows]
    print(
        f"{path.name}: reports {len(rows)} top-1 {sum(r == 1 for r in ranks) / len(ranks):.3f} "
        f"top-3 {sum(r <= 3 for r in ranks) / len(ranks):.3f} top-5 {sum(r <= 5 for r in ranks) / len(ranks):.3f} "
        f"MRR {sum(1 / r for r in ranks) / len(ranks):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    a = commands.add_parser("aggregate")
    a.set_defaults(func=aggregate)
    a.add_argument("--name", required=True)
    a.add_argument("--split", required=True)
    t = commands.add_parser("tfidf")
    t.set_defaults(func=tfidf)
    t.add_argument("--split", required=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
