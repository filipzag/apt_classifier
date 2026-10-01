"""Offline replay of the top-three promotion rule over recorded attribution artifacts.

Reads ``scripts/attribution_benchmark.py`` artifacts (rows carry the shortlist order ``top3``
and the gate window's per-channel discrimination ``topk``) and applies
``apt_classifier.top3.decide_promotion`` -- the function the engine runs -- without re-running
the pipeline. For each artifact it prints top-1 before and after, the rows won and lost, the
change in MRR, the promotions that were blocked by the contradiction guard, the slack (largest D
gap a correct leader faced, smallest gap a recovered truth needed) and, for promoted leaders,
the separation they would have against the naming bars (they are never named; this is the
danger margin). Rows recorded with the stage on are replayed from the order the rule saw.

    PYTHONPATH=. .venv/bin/python scripts/top3_replay.py data/audits/top3/baseline_*.json \\
        --margin 1.25 --tolerance 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.separation import (  # noqa: E402
    CandidateDiscrimination,
    DiscriminationResult,
    DiscriminationWeights,
)
from apt_classifier.top3 import decide_promotion, order_before_promotion  # noqa: E402

CHANNELS = ("technique", "behavior", "software", "negative", "soft")


def weights_for(artifact: dict[str, Any]) -> DiscriminationWeights:
    """The discrimination weights the artifact was scored with (recorded since the top-three
    instrumentation; older artifacts fall back to the current settings)."""
    recorded = (artifact.get("fingerprint_inputs") or {}).get("discrimination_weights")
    if recorded:
        return DiscriminationWeights(**{key: float(recorded[key]) for key in CHANNELS})
    from apt_classifier.config import Settings

    settings = Settings()
    return DiscriminationWeights(
        technique=float(settings.attribution_separation_technique_weight),
        behavior=float(settings.attribution_separation_behavior_weight),
        software=float(settings.attribution_separation_software_weight),
        negative=float(settings.attribution_separation_negative_weight),
        soft=float(settings.attribution_separation_soft_weight),
    )


def discrimination_for(row: dict[str, Any], weights: DiscriminationWeights) -> Any:
    scores = tuple(
        CandidateDiscrimination(
            group_id=str(item["group_id"]),
            rrf_score=float(item.get("rrf", 0.0) or 0.0),
            technique=float(item["technique"]),
            behavior=float(item["behavior"]),
            software=float(item["software"]),
            negative=float(item["negative"]),
            total=float(item["total"]),
            soft=float(item.get("soft", 0.0) or 0.0),
        )
        for item in row.get("topk") or []
    )
    return DiscriminationResult(scores=scores, top_k=len(scores), weights=weights)


def promoted_separation(row: dict[str, Any], group_id: str, floor: float) -> tuple[float, float]:
    """(separation, D) the promoted group has against the rest of the same gate window."""
    totals = {str(item["group_id"]): float(item["total"]) for item in row.get("topk") or []}
    own = totals.get(group_id, 0.0)
    other = max((value for gid, value in totals.items() if gid != group_id), default=0.0)
    denominator = max(own, other, floor)
    return ((own - other) / denominator if denominator > 0 else 0.0), own


def replay(
    artifact: dict[str, Any], *, margin: float, tolerance: float, floor: float, bar: float
) -> dict[str, Any]:
    weights = weights_for(artifact)
    known_rows = artifact.get("known") or []
    unknown_rows = artifact.get("unknown") or []
    out: dict[str, Any] = {
        "known": len(known_rows),
        "wins": [],
        "losses": [],
        "neutral_promotions": 0,
        "blocked": [],
        "correct_leader_max_gap": 0.0,
        "recovered_min_gap": None,
        "promoted_wrong_max_separation": None,
        "promoted_nameable_wrong": 0,
        "unknown_promotions": 0,
        "unknown_promoted_max_separation": None,
    }
    before_ranks: list[int | None] = []
    after_ranks: list[int | None] = []
    for row in known_rows:
        truth = row["group_id"]
        order = order_before_promotion(row.get("top3") or [], row.get("top3_promotion"))
        discrimination = discrimination_for(row, weights)
        rank = row.get("rank")
        if row.get("top3_promotion") and not row["top3_promotion"].get("blocked"):
            rank = row.get("rank_before_top3", rank)
        before_ranks.append(rank)
        totals = discrimination.by_group()
        if rank == 1 and order and order[0] in totals:
            gaps = [
                totals[gid].total - totals[order[0]].total for gid in order[1:3] if gid in totals
            ]
            out["correct_leader_max_gap"] = max([out["correct_leader_max_gap"], *gaps])
        promotion = decide_promotion(order, discrimination, margin=margin, tolerance=tolerance)
        label = f"{truth}:{row.get('report_id') or ''}".rstrip(":")
        if promotion is None:
            after_ranks.append(rank)
            continue
        if promotion.blocked:
            out["blocked"].append(
                f"{label} {promotion.promoted} over {promotion.previous_leader} "
                f"({promotion.advantage_channel} {promotion.leader_advantage:.2f})"
            )
            after_ranks.append(rank)
            continue
        new_rank = rank
        if truth == promotion.promoted:
            new_rank = 1
        elif rank is not None and rank < promotion.from_rank:
            new_rank = rank + 1
        after_ranks.append(new_rank)
        separation, total = promoted_separation(row, promotion.promoted, floor)
        if truth != promotion.promoted:
            previous = out["promoted_wrong_max_separation"]
            out["promoted_wrong_max_separation"] = (
                separation if previous is None else max(previous, separation)
            )
            if separation >= bar and total >= 3.0:
                out["promoted_nameable_wrong"] += 1
        if new_rank == 1 and rank != 1:
            out["wins"].append(f"{label} (from #{promotion.from_rank}, gap {promotion.gap:.2f})")
            previous_gap = out["recovered_min_gap"]
            out["recovered_min_gap"] = (
                promotion.gap if previous_gap is None else min(previous_gap, promotion.gap)
            )
        elif rank == 1 and new_rank != 1:
            out["losses"].append(f"{label} (lost to {promotion.promoted})")
        else:
            out["neutral_promotions"] += 1
    for row in unknown_rows:
        order = order_before_promotion(row.get("top3") or [], row.get("top3_promotion"))
        promotion = decide_promotion(
            order, discrimination_for(row, weights), margin=margin, tolerance=tolerance
        )
        if promotion is None or promotion.blocked:
            continue
        out["unknown_promotions"] += 1
        separation, _ = promoted_separation(row, promotion.promoted, floor)
        previous = out["unknown_promoted_max_separation"]
        out["unknown_promoted_max_separation"] = (
            separation if previous is None else max(previous, separation)
        )

    def top1(ranks: list[int | None]) -> int:
        return sum(1 for value in ranks if value == 1)

    def mrr(ranks: list[int | None]) -> float:
        return fmean(1.0 / value if value else 0.0 for value in ranks) if ranks else 0.0

    out["top1_before"] = top1(before_ranks)
    out["top1_after"] = top1(after_ranks)
    out["mrr_delta"] = round(mrr(after_ranks) - mrr(before_ranks), 4)
    in_top3 = [value for value in before_ranks if value and value <= 3]
    out["top1_given_top3_before"] = round(top1(before_ranks) / len(in_top3), 4) if in_top3 else None
    out["top1_given_top3_after"] = round(top1(after_ranks) / len(in_top3), 4) if in_top3 else None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--margin", type=float, nargs="+", default=[1.25])
    parser.add_argument("--tolerance", type=float, default=0.75)
    parser.add_argument("--floor", type=float, default=1.0, help="separation floor (settings)")
    parser.add_argument("--bar", type=float, default=0.5, help="naming separation bar")
    parser.add_argument("--json", action="store_true", help="print one JSON object per run")
    args = parser.parse_args(argv)
    for path in args.artifacts:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        for margin in args.margin:
            result = replay(
                artifact,
                margin=margin,
                tolerance=args.tolerance,
                floor=args.floor,
                bar=args.bar,
            )
            if args.json:
                print(json.dumps({"artifact": path.name, "margin": margin, **result}))
                continue
            print(f"== {path.name}  margin {margin}  tolerance {args.tolerance}")
            print(
                f"  top-1 {result['top1_before']} -> {result['top1_after']} of {result['known']}"
                f"  (+{len(result['wins'])} / -{len(result['losses'])}, "
                f"{result['neutral_promotions']} neutral), MRR {result['mrr_delta']:+.4f}"
            )
            print(
                f"  top-1 given top-3 {result['top1_given_top3_before']} -> "
                f"{result['top1_given_top3_after']}"
            )
            for key, label in (("wins", "win"), ("losses", "loss"), ("blocked", "blocked")):
                for item in result[key]:
                    print(f"  {label}: {item}")
            recovered = result["recovered_min_gap"]
            print(
                f"  slack: correct leaders faced a gap of at most "
                f"{result['correct_leader_max_gap']:.2f}; recovered truths needed at least "
                f"{'n/a' if recovered is None else f'{recovered:.2f}'}"
            )
            print(
                f"  wrong promoted leaders: max separation "
                f"{result['promoted_wrong_max_separation']}, would clear the naming bars: "
                f"{result['promoted_nameable_wrong']} (never named by design)"
            )
            if artifact.get("unknown"):
                print(
                    f"  removed actors: {result['unknown_promotions']} of "
                    f"{len(artifact['unknown'])} reordered, max separation "
                    f"{result['unknown_promoted_max_separation']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
