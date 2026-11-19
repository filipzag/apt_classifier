"""Offline replay of the naming probability floor over recorded attribution artifacts.

The floor (``ATTRIBUTION_HYPOTHESIS_MIN_PROBABILITY``, ``MaturityThresholds.
hypothesis_min_probability``) adds one condition to the single-group naming gate: the learned
combiner's probability for the leader must reach it. A leader it holds fails the single gate, so
the documented-cluster step decides as for any other single-gate failure (a cluster name when the
leader's documented twin clears the cluster bars, else the shortlist). Cluster names have their
own combined-probability bar and are untouched; the ranking never changes. A run made without the
floor records everything the rule needs (the leader probability, the cluster candidate, the
accepted techniques, the decisive-software factor), so the replay is exact; ``--on`` checks it row
for row against a run made with the floor.

``--choose`` applies the registered design rule (prereg
``attribution_probability_floor_2026-10-25.json``) to the given design artifacts: the smallest grid
floor that holds no correct single name and holds every bad single name (a known row's wrong leader,
any removed-actor name) whose probability lies below the lowest correct single name's; 0 when no
bad name lies below it.

    PYTHONPATH=. .venv/bin/python scripts/probability_floor_replay.py \\
        data/audits/top3/design_seed41.json data/audits/top3/design_seed53.json --choose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attribution import MaturityThresholds, _cluster_step  # noqa: E402

HYPOTHESIS = "known_group_hypothesis"
CLUSTER = "known_cluster_hypothesis"
SHORTLIST = "known_group_shortlist"
GRID = tuple(round(0.05 * step, 2) for step in range(1, 13))  # 0.05 .. 0.60


def recorded_thresholds(artifact: dict[str, Any], floor: float) -> MaturityThresholds:
    """The run's own gate thresholds with the floor set."""
    recorded = dict((artifact.get("fingerprint_inputs") or {}).get("thresholds") or {})
    known = set(MaturityThresholds.__dataclass_fields__)
    values = {key: value for key, value in recorded.items() if key in known}
    values["hypothesis_min_probability"] = float(floor)
    return MaturityThresholds(**values)


def replay_row(row: dict[str, Any], thresholds: MaturityThresholds) -> dict[str, Any]:
    """The stage, named members and whether the floor held the leader, for one row."""
    stage = row.get("stage")
    members = list(row.get("named_cluster") or [])
    probability = row.get("combiner_leader_probability")
    floor = float(thresholds.hypothesis_min_probability)
    if stage != HYPOTHESIS or floor <= 0 or probability is None or float(probability) >= floor:
        return {"stage": stage, "named_cluster": members, "held": False}
    decisive = row.get("decisive_software")
    step, candidate = _cluster_step(
        thresholds,
        {"cluster": row.get("cluster_candidate")},
        str(row.get("leading")),
        bool(row.get("leader_guard_fired")),
        technique_count=len(row.get("accepted_techniques") or []),
        decisive_factor=float((decisive or {}).get("factor", 1.0) or 1.0)
        if isinstance(decisive, dict)
        else 1.0,
    )
    if step == "cluster" and candidate is not None:
        return {"stage": CLUSTER, "named_cluster": list(candidate["members"]), "held": True}
    return {"stage": SHORTLIST, "named_cluster": [], "held": True}


def outcome(arm: str, row: dict[str, Any], result: dict[str, Any]) -> str:
    """correct / wrong / removed for a name, "" for no name."""
    if result["stage"] not in (HYPOTHESIS, CLUSTER):
        return ""
    if arm != "known":
        return "removed"
    if result["stage"] == CLUSTER:
        return "correct" if row["group_id"] in result["named_cluster"] else "wrong"
    return "correct" if row.get("leading") == row["group_id"] else "wrong"


def label(arm: str, row: dict[str, Any]) -> str:
    return f"{arm}:{row['group_id']}:{row.get('report_id') or ''}".rstrip(":")


def replay(artifact: dict[str, Any], floor: float) -> dict[str, Any]:
    """Counts of one artifact under one floor (``floor`` 0 = the run itself)."""
    thresholds = recorded_thresholds(artifact, floor)
    out: dict[str, Any] = {
        "floor": floor,
        "correct": 0,
        "wrong": [],
        "removed": [],
        "open_set": [],
        "held_correct": [],
        "held_bad": [],
        "cluster_fallback": [],
        "predictions": {},
    }
    arms = [
        ("known", artifact.get("known") or []),
        ("unknown", artifact.get("unknown") or []),
        ("open", artifact.get("open_set") or []),  # Elastic's open set: every name is a bad one
    ]
    for arm, rows in arms:
        for row in rows:
            result = replay_row(row, thresholds)
            name = label(arm, row)
            out["predictions"][name] = result
            kind = outcome(arm, row, result)
            leader = row.get("leading")
            probability = row.get("combiner_leader_probability")
            if kind == "correct":
                out["correct"] += 1
            elif kind == "wrong":
                out["wrong"].append(f"{name} as {leader} (p {probability})")
            elif kind == "removed" and arm == "open":
                out["open_set"].append(f"{name} as {leader} (p {probability})")
            elif kind == "removed":
                out["removed"].append(f"{name} as {leader} (p {probability})")
            if result["held"]:
                before = outcome(arm, row, {"stage": HYPOTHESIS, "named_cluster": []})
                bucket = "held_correct" if before == "correct" else "held_bad"
                out[bucket].append(f"{name} as {leader} (p {probability})")
                if result["stage"] == CLUSTER:
                    out["cluster_fallback"].append(
                        f"{name} as {' / '.join(result['named_cluster'])} ({kind})"
                    )
    return out


def single_names(artifacts: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Leader probabilities of the correct and of the bad single names in the runs."""
    correct: list[float] = []
    bad: list[float] = []
    for artifact in artifacts:
        for arm in ("known", "unknown"):
            for row in artifact.get(arm) or []:
                probability = row.get("combiner_leader_probability")
                if row.get("stage") != HYPOTHESIS or probability is None:
                    continue
                right = arm == "known" and row.get("leading") == row["group_id"]
                (correct if right else bad).append(float(probability))
    return correct, bad


def choose(artifacts: list[dict[str, Any]], grid: tuple[float, ...] = GRID) -> dict[str, Any]:
    """The registered design rule (see the module docstring)."""
    correct, bad = single_names(artifacts)
    lowest = min(correct) if correct else 1.0
    catchable = sorted(p for p in bad if p < lowest)
    result = {
        "lowest_correct": lowest,
        "bad": sorted(bad),
        "catchable": catchable,
        "floor": 0.0,
    }
    if not catchable:
        return result
    for floor in grid:
        if floor > catchable[-1] and floor <= lowest:
            result["floor"] = floor
            return result
    result["floor"] = None  # no grid point separates them: no floor is chosen
    return result


def compare(off: dict[str, Any], on: dict[str, Any], floor: float) -> list[str]:
    """Rows where the replay of ``off`` under ``floor`` disagrees with the run ``on``."""
    replayed = replay(off, floor)["predictions"]
    actual = replay(on, 0.0)["predictions"]
    mismatches = []
    for name, result in replayed.items():
        other = actual.get(name)
        if other is None or (result["stage"], result["named_cluster"]) != (
            other["stage"],
            other["named_cluster"],
        ):
            mismatches.append(f"{name}: replay {result['stage']} / run {other and other['stage']}")
    return mismatches


def summary(report: dict[str, Any]) -> str:
    return (
        f"floor {report['floor']:.2f}: correct {report['correct']} wrong {len(report['wrong'])} "
        f"removed {len(report['removed'])} open-set {len(report['open_set'])} | held "
        f"{len(report['held_correct'])} correct, {len(report['held_bad'])} bad; cluster fallback "
        f"{report['cluster_fallback']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--floors", default=",".join(str(f) for f in GRID))
    parser.add_argument("--choose", action="store_true", help="apply the registered design rule")
    parser.add_argument("--on", nargs="*", type=Path, default=[], help="runs made with the floor")
    parser.add_argument("--floor", type=float, default=None, help="the floor of the --on runs")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts]
    floors = [0.0, *(float(item) for item in args.floors.split(",") if item)]
    for path, artifact in zip(args.artifacts, artifacts, strict=True):
        print(path)
        for floor in floors:
            report = replay(artifact, floor)
            print("  " + summary(report))
            if args.details and floor > 0:
                for key in ("held_correct", "held_bad", "wrong", "removed", "open_set"):
                    if report[key]:
                        print(f"    {key}: {report[key]}")
    if args.choose:
        print("design rule:", json.dumps(choose(artifacts)))
    if args.on:
        if args.floor is None or len(args.on) != len(artifacts):
            parser.error("--on needs one run per artifact and --floor")
        failed = 0
        for path, off, on_path in zip(args.artifacts, artifacts, args.on, strict=True):
            mismatches = compare(off, json.loads(on_path.read_text(encoding="utf-8")), args.floor)
            print(
                f"{path} vs {on_path}: {len(mismatches)} mismatches", *mismatches[:10], sep="\n  "
            )
            failed += len(mismatches)
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
