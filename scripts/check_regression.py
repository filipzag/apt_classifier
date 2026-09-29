"""Fail when a fresh public held-out artifact regresses against a committed baseline.

Compares ``results[<mode>]['pooled']['metrics']`` (and per-dataset metrics for
information) between two artifacts written by ``scripts/benchmark_mapping.py``.
Exit codes: 0 no regression, 1 regression beyond tolerance, 2 artifacts not comparable
(different case sets) unless ``--allow-different-cases``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WATCHED = (
    "micro_f1",
    "hit_at_10",
    "micro_precision",
    "micro_recall",
    "hit_at_1",
    "parent_aware_f1",
)


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    mode: str = "hybrid",
    max_f1_drop: float = 0.02,
    max_hit10_drop: float = 0.03,
) -> dict[str, Any]:
    """Return a comparison report with ``regressions`` listing violated tolerances."""
    base_run = baseline.get("results", {}).get(mode)
    curr_run = current.get("results", {}).get(mode)
    if base_run is None or curr_run is None:
        raise KeyError(f"mode {mode!r} missing from one of the artifacts")
    same_cases = baseline.get("case_set_hash") == current.get("case_set_hash")
    report: dict[str, Any] = {
        "mode": mode,
        "same_case_set": same_cases,
        "baseline_fingerprint": base_run.get("pipeline_fingerprint"),
        "current_fingerprint": curr_run.get("pipeline_fingerprint"),
        "pooled": {},
        "datasets": {},
        "regressions": [],
    }
    base_pooled = base_run["pooled"]["metrics"]
    curr_pooled = curr_run["pooled"]["metrics"]
    for name in WATCHED:
        if name in base_pooled and name in curr_pooled:
            report["pooled"][name] = {
                "baseline": round(float(base_pooled[name]), 4),
                "current": round(float(curr_pooled[name]), 4),
                "delta": round(float(curr_pooled[name]) - float(base_pooled[name]), 4),
            }
    tolerances = {"micro_f1": max_f1_drop, "hit_at_10": max_hit10_drop}
    for name, tolerance in tolerances.items():
        entry = report["pooled"].get(name)
        if entry and entry["delta"] < -tolerance:
            report["regressions"].append(
                f"pooled {name} dropped {-entry['delta']:.4f} (> {tolerance})"
            )
    for dataset in sorted(set(base_run.get("datasets", {})) & set(curr_run.get("datasets", {}))):
        base_metrics = base_run["datasets"][dataset]["metrics"]
        curr_metrics = curr_run["datasets"][dataset]["metrics"]
        report["datasets"][dataset] = {
            name: {
                "baseline": round(float(base_metrics[name]), 4),
                "current": round(float(curr_metrics[name]), 4),
                "delta": round(float(curr_metrics[name]) - float(base_metrics[name]), 4),
            }
            for name in WATCHED
            if name in base_metrics and name in curr_metrics
        }
    return report


def _print(report: dict[str, Any]) -> None:
    print(f"mode={report['mode']} same_case_set={report['same_case_set']}")
    print(f"{'metric':20s}{'baseline':>10s}{'current':>10s}{'delta':>9s}")
    for name, entry in report["pooled"].items():
        print(f"{name:20s}{entry['baseline']:10.4f}{entry['current']:10.4f}{entry['delta']:+9.4f}")
    for dataset, metrics in report["datasets"].items():
        f1 = metrics.get("micro_f1", {})
        hit = metrics.get("hit_at_10", {})
        print(
            f"  {dataset:20s} micro_f1 {f1.get('baseline', 0):.3f}->{f1.get('current', 0):.3f} "
            f"({f1.get('delta', 0):+.3f})  hit@10 {hit.get('baseline', 0):.3f}->"
            f"{hit.get('current', 0):.3f} ({hit.get('delta', 0):+.3f})"
        )
    for item in report["regressions"]:
        print(f"REGRESSION: {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--max-f1-drop", type=float, default=0.02)
    parser.add_argument("--max-hit10-drop", type=float, default=0.03)
    parser.add_argument("--allow-different-cases", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    report = compare(
        baseline,
        current,
        mode=args.mode,
        max_f1_drop=args.max_f1_drop,
        max_hit10_drop=args.max_hit10_drop,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print(report)
    if not report["same_case_set"] and not args.allow_different_cases:
        print("case sets differ; pass --allow-different-cases to compare anyway", file=sys.stderr)
        return 2
    return 1 if report["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
