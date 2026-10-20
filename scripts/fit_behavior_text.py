"""Choose the behaviour-text member's parameters on the vendor set (development data only;
prereg ``docs/preregistration/attribution_behavior_text_2026-10-20.json``).

Scores every vendor report (its statements with actor and campaign names replaced, as the
corpus and the benchmarks replace them) with the member alone for the registered grid and
prints top-1 / top-3 / MRR per variant; the best MRR (ties: top-1, then the simplest variant)
is the dev choice to put into the ``BEHAVIOR_TEXT_*`` defaults. The text member's artifact
(``TEXT_GROUP_PATH``) supplies the corpus; nothing is written.

    PYTHONPATH=. .venv/bin/python scripts/fit_behavior_text.py [--reports data/benchmarks/vendor/reports.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.behavior_text import BehaviorTextModel  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.text_group import (  # noqa: E402
    TextGroupModel,
    _campaign_pattern,
    actor_names,
    evidence_name_pattern,
)

GRID = {
    "top_k": (10, 20, 50),
    "aggregate": ("mean", "top5"),
    "prior": (0.0, 0.25, 0.5),
}
MIN_WORDS = 3


def load_reports(path: Path, kb: Any) -> list[dict[str, Any]]:
    actors = evidence_name_pattern(actor_names(kb), before_hyphen=True)
    campaigns = _campaign_pattern(kb)
    reports = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statements = []
        for item in row.get("statements") or []:
            text = str(item.get("text") or "")
            if actors is not None:
                text = actors.sub(lambda m: "the adversary" + (m.group("possessive") or ""), text)
            if campaigns is not None:
                text = campaigns.sub("the operation", text)
            text = " ".join(text.split())
            if len(text.split()) >= MIN_WORDS:
                statements.append(text)
        if statements and row.get("group_id"):
            reports.append(
                {
                    "report_id": row["report_id"],
                    "group_id": row["group_id"],
                    "statements": statements,
                }
            )
    return reports


def evaluate(model: BehaviorTextModel, reports: list[dict[str, Any]]) -> dict[str, Any]:
    ranks: list[int | None] = []
    for report in reports:
        scores = model.scores(report["statements"])
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        rank = next((i for i, (g, _) in enumerate(ordered, 1) if g == report["group_id"]), None)
        ranks.append(rank)
    n = len(ranks)
    return {
        "reports": n,
        "top1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "top3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
        "top5": round(sum(1 for r in ranks if r and r <= 5) / n, 4),
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 4),
        "unranked": sum(1 for r in ranks if r is None),
        "ranks": ranks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=REPO_ROOT / "data" / "benchmarks" / "vendor" / "reports.jsonl",
        help="development reports (vendor set); never a test set",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    base = TextGroupModel.load(settings.text_group_path, kb)
    if base is None:
        parser.error(f"text member artifact not found: {settings.text_group_path}")
    reports = load_reports(args.reports, kb)
    print(f"{len(reports)} reports from {args.reports}; corpus {len(base.records)} sentences")
    results = []
    for top_k, aggregate, prior in itertools.product(*GRID.values()):
        params = {"top_k": top_k, "aggregate": aggregate, "prior": prior}
        model = BehaviorTextModel(base, params)
        result = evaluate(model, reports)
        results.append({"params": params, **result})
        print(
            f"top_k {top_k:3d} {aggregate:5s} prior {prior:.2f}: top-1 {result['top1']:.3f} "
            f"top-3 {result['top3']:.3f} top-5 {result['top5']:.3f} MRR {result['mrr']:.4f} "
            f"unranked {result['unranked']}"
        )
    best = max(
        results,
        key=lambda r: (
            r["mrr"],
            r["top1"],
            -r["params"]["top_k"],
            r["params"]["aggregate"] == "mean",
            -r["params"]["prior"],
        ),
    )
    print("dev choice:", best["params"], {k: best[k] for k in ("top1", "top3", "mrr")})
    if args.json:
        print(
            json.dumps(
                {
                    "best": best["params"],
                    "grid": [{k: v for k, v in r.items() if k != "ranks"} for r in results],
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
