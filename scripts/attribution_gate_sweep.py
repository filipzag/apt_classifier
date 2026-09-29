"""Offline sweep of the naming gate over recorded attribution artifacts.

Reads one or more ``scripts/attribution_benchmark.py`` artifacts whose rows carry the raw
per-channel discrimination of the top-K (``topk``), recombines the channels over a grid of
weights and floors *without re-running the pipeline*, reports how well each score separates
correct from wrong leaders (rank-based AUC), and sweeps gate rules, printing only the rules that
name no wrong group on any known arm and at most ``--max-unknown-named`` removed actors, ordered
by the number of correct names. The guard band (distance of the highest-scoring wrong leader to
the boundary) and a leave-one-group-out check say how much slack a rule has.

    PYTHONPATH=. .venv/bin/python scripts/attribution_gate_sweep.py \\
        data/audits/attribution_heldout_current.json data/audits/attribution_cti_hal_current.json
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


def auc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def recombine(
    row: dict[str, Any],
    *,
    w_t: float,
    w_b: float,
    w_s: float,
    w_n: float,
    floor: float,
    k: int,
    w_soft: float = 1.0,
) -> tuple[float, float]:
    """(separation, leader total) from the recorded raw channels with new weights. Rows
    recorded before the contrastive ``soft`` channel existed replay with soft = 0."""
    candidates = list(row.get("topk") or [])[: max(k, 1)]
    if not candidates:
        return 0.0, 0.0
    totals = [
        max(
            0.0,
            w_t * float(c["technique"])
            + w_b * float(c["behavior"])
            + w_s * float(c["software"])
            + w_soft * float(c.get("soft", 0.0) or 0.0)
            - w_n * float(c["negative"]),
        )
        for c in candidates
    ]
    leader = totals[0]
    runner = max(totals[1:]) if len(totals) > 1 else 0.0
    denominator = max(leader, runner, floor)
    separation = (leader - runner) / denominator if denominator > 0 else 0.0
    return max(-1.0, min(1.0, separation)), leader


def load(paths: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    return [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--max-unknown-named", type=int, default=1)
    parser.add_argument("--top", type=int, default=15, help="rules to print")
    parser.add_argument("--weights-grid", default="0,0.5,1,2", help="values for w_b and w_s")
    parser.add_argument(
        "--soft-grid", default="1", help="values for the contrastive soft channel weight w_soft"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    sets = load(args.artifacts)
    known = {name: [r for r in d["known"] if r.get("rank") is not None] for name, d in sets}
    unknown = {name: list(d.get("unknown", [])) for name, d in sets}
    if any("topk" not in r for rows in known.values() for r in rows):
        parser.error("artifacts lack per-row 'topk' channels; rerun the benchmark first")

    # 1. AUC of the recorded scores
    report: dict[str, Any] = {"auc": {}, "rules": []}
    for name, rows in known.items():
        pos = [r for r in rows if r["rank"] == 1]
        neg = [r for r in rows if r["rank"] != 1]
        scores = {
            "margin": lambda r: float(r.get("margin") or 0.0),
            "maturity": lambda r: float(r.get("maturity") or 0.0),
            "separation": lambda r: float(r.get("separation") or 0.0),
            "leader_total": lambda r: float((r.get("topk") or [{}])[0].get("total", 0.0)),
            "leader_specificity": lambda r: float(r.get("leader_specificity") or 0.0),
        }
        report["auc"][name] = {
            key: auc([f(r) for r in pos], [f(r) for r in neg]) for key, f in scores.items()
        }
        print(f"== {name}: {len(rows)} known rows, {len(pos)} correct leaders")
        for key, value in report["auc"][name].items():
            print(f"   AUC {key:18s} {value if value is None else round(value, 3)}")

    # 2. rule sweep over recombined channels
    grid = [float(v) for v in args.weights_grid.split(",")]
    soft_grid = [float(v) for v in args.soft_grid.split(",")]
    results = []
    for w_b, w_s, w_soft, floor, k in itertools.product(
        grid, grid, soft_grid, (0.5, 1.0, 2.0), (3, 5, 10)
    ):
        for sep_bar, d_bar, margin_bar in itertools.product(
            (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
            (0.0, 0.05, 0.10, 0.15),
        ):
            per_set = {}
            clean = True
            for name, rows in known.items():
                named = correct = 0
                worst_wrong = -1.0
                for r in rows:
                    sep, leader = recombine(
                        r, w_t=1.0, w_b=w_b, w_s=w_s, w_n=0.3, floor=floor, k=k, w_soft=w_soft
                    )
                    margin = float(r.get("margin") or 0.0)
                    if sep >= sep_bar and leader >= d_bar and margin >= margin_bar:
                        named += 1
                        if r["rank"] == 1:
                            correct += 1
                        else:
                            clean = False
                    elif r["rank"] != 1:
                        worst_wrong = max(worst_wrong, sep if leader >= d_bar else -1.0)
                per_set[name] = {
                    "named": named,
                    "correct": correct,
                    "guard": round(sep_bar - worst_wrong, 3),
                }
            unknown_named = 0
            for rows in unknown.values():
                for r in rows:
                    sep, leader = recombine(
                        r, w_t=1.0, w_b=w_b, w_s=w_s, w_n=0.3, floor=floor, k=k, w_soft=w_soft
                    )
                    if (
                        sep >= sep_bar
                        and leader >= d_bar
                        and float(r.get("margin") or 0.0) >= margin_bar
                    ):
                        unknown_named += 1
            if clean and unknown_named <= args.max_unknown_named:
                results.append(
                    {
                        "w_b": w_b,
                        "w_s": w_s,
                        "w_soft": w_soft,
                        "floor": floor,
                        "k": k,
                        "sep": sep_bar,
                        "d": d_bar,
                        "margin": margin_bar,
                        "sets": per_set,
                        "unknown_named": unknown_named,
                        "correct_total": sum(v["correct"] for v in per_set.values()),
                        "min_guard": min(v["guard"] for v in per_set.values()),
                    }
                )
    results.sort(key=lambda item: (-item["correct_total"], -item["min_guard"]))
    report["rules"] = results[: args.top]
    print(
        f"\n== clean rules (no wrong name on any known arm, unknown named <= {args.max_unknown_named}): {len(results)}"
    )
    print(
        f"{'w_b':>4s} {'w_s':>4s} {'w_sf':>4s} {'flr':>4s} {'k':>2s} {'sep':>4s} {'d':>4s} {'mrg':>5s} | named(correct) per set | unk | guard"
    )
    for item in results[: args.top]:
        sets_text = " ".join(
            f"{name[:12]}={v['named']}({v['correct']})" for name, v in item["sets"].items()
        )
        print(
            f"{item['w_b']:4.1f} {item['w_s']:4.1f} {item['w_soft']:4.1f} {item['floor']:4.1f} {item['k']:2d} "
            f"{item['sep']:4.1f} {item['d']:4.1f} {item['margin']:5.2f} | {sets_text} | "
            f"{item['unknown_named']} | {item['min_guard']:.2f}"
        )
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
