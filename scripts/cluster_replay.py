"""Offline replay of documented-cluster naming over recorded attribution artifacts.

Reads a ``scripts/attribution_benchmark.py`` artifact run with ``ATTRIBUTION_CLUSTER_NAMING``
off, whose rows record the engine's cluster candidate (``cluster_candidate``: the leader's
documented twin in the gate window, the merged pair's separation and mass, the members' combined
combiner probability), and applies the stage rule ``attribution.assess_attribution`` uses when
the setting is on. The recorded candidate was selected under the run's
``ATTRIBUTION_CLUSTER_SCOPE``, so replay with the same ``--scope``: a narrower scope here only
drops candidates and cannot find the lighter twin the engine would have picked instead. It
reports the pre-registered counts of docs 21.3.15 (A1-A5): correct and
wrong cluster names, removed actors named through a cluster, single names upgraded or held back,
and removed-actor rows that clear two of the three cluster bars. With ``--on`` it compares the
replay with an artifact run with the setting on, row for row (A4).

    PYTHONPATH=. .venv/bin/python scripts/cluster_replay.py data/audits/x_off.json --on data/audits/x_on.json
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


def recorded_scope(artifact: dict[str, Any]) -> str:
    """The cluster scope an artifact's candidates were selected under; artifacts from before the
    setting recorded no scope and were selected under the first rule ("full")."""
    thresholds = (artifact.get("fingerprint_inputs") or {}).get("thresholds") or {}
    return str(thresholds.get("cluster_scope") or "full")


def thresholds_for(args: argparse.Namespace, scope: str | None = None) -> MaturityThresholds:
    return MaturityThresholds(
        gate_mode="separation",
        hypothesis_separation=args.separation,
        hypothesis_min_discriminative=args.min_discriminative,
        cluster_naming=True,
        cluster_min_probability=args.min_probability,
        cluster_min_discriminative=args.cluster_min_discriminative,
        cluster_scope=scope or args.scope or "full",
    )


def replay_row(row: dict[str, Any], thresholds: MaturityThresholds) -> dict[str, Any]:
    """The stage and named members the setting would give this off-run row."""
    stage = row.get("stage")
    leader = row.get("leading")
    out = {"stage": stage, "named_cluster": []}
    if stage not in (HYPOTHESIS, SHORTLIST) or not leader:
        return out
    candidate = row.get("cluster_candidate")
    step, used = _cluster_step(
        thresholds, {"cluster": candidate}, leader, bool(row.get("leader_guard_fired"))
    )
    if step == "cluster" and used is not None:
        return {"stage": CLUSTER, "named_cluster": list(used["members"])}
    if stage == HYPOTHESIS and used is not None and used.get("nested"):
        return {"stage": SHORTLIST, "named_cluster": []}
    return out


def bars_cleared(candidate: dict[str, Any] | None, thresholds: MaturityThresholds) -> int:
    if not candidate:
        return 0
    nested = bool(candidate.get("nested"))
    mass_bar = (
        thresholds.hypothesis_min_discriminative
        if nested
        else max(thresholds.hypothesis_min_discriminative, thresholds.cluster_min_discriminative)
    )
    probability = candidate.get("probability")
    return (
        int(float(candidate.get("separation", 0.0)) >= thresholds.hypothesis_separation)
        + int(float(candidate.get("discriminative", 0.0)) >= mass_bar)
        + int(probability is not None and float(probability) >= thresholds.cluster_min_probability)
    )


def replay(artifact: dict[str, Any], thresholds: MaturityThresholds) -> dict[str, Any]:
    out: dict[str, Any] = {
        "single_named_off": 0,
        "single_named_on": 0,
        "single_correct_on": 0,
        "removed_single_named": [],
        "cluster_correct": [],
        "cluster_wrong": [],
        "cluster_recovered": 0,
        "upgrades": [],
        "held_back": [],
        "removed_named_by_cluster": [],
        "removed_two_of_three": [],
        "missing_candidate_field": 0,
    }
    predictions: list[dict[str, Any]] = []
    for arm in ("known", "unknown"):
        for row in artifact.get(arm) or []:
            if "cluster_candidate" not in row:
                out["missing_candidate_field"] += 1
            truth = row["group_id"] if arm == "known" else None
            label = f"{arm}:{row['group_id']}:{row.get('report_id') or ''}".rstrip(":")
            result = replay_row(row, thresholds)
            predictions.append({"label": label, **result})
            if arm == "known" and row.get("stage") == HYPOTHESIS:
                out["single_named_off"] += 1
            if result["stage"] == HYPOTHESIS and arm == "known":
                out["single_named_on"] += 1
                out["single_correct_on"] += int(row.get("rank") == 1)
            elif result["stage"] == HYPOTHESIS:
                out["removed_single_named"].append(f"{label} as {row.get('leading')}")
            if result["stage"] == CLUSTER:
                members = result["named_cluster"]
                if arm == "unknown":
                    out["removed_named_by_cluster"].append(f"{label} as {' / '.join(members)}")
                elif truth in members:
                    out["cluster_correct"].append(f"{label} as {' / '.join(members)}")
                    out["cluster_recovered"] += int(row.get("rank") != 1)
                else:
                    out["cluster_wrong"].append(f"{label} as {' / '.join(members)}")
                if row.get("stage") == HYPOTHESIS:
                    out["upgrades"].append(label)
            if row.get("stage") == HYPOTHESIS and result["stage"] == SHORTLIST:
                out["held_back"].append(label)
            if arm == "unknown" and bars_cleared(row.get("cluster_candidate"), thresholds) >= 2:
                out["removed_two_of_three"].append(label)
    out["predictions"] = predictions
    return out


def compare(predictions: list[dict[str, Any]], on: dict[str, Any]) -> list[str]:
    rows = [
        (f"{arm}:{row['group_id']}:{row.get('report_id') or ''}".rstrip(":"), row)
        for arm in ("known", "unknown")
        for row in on.get(arm) or []
    ]
    mismatches = []
    for prediction, (label, row) in zip(predictions, rows, strict=False):
        if prediction["label"] != label:
            mismatches.append(f"row order differs at {label}")
            break
        if prediction["stage"] != row.get("stage") or list(prediction["named_cluster"]) != list(
            row.get("named_cluster") or []
        ):
            mismatches.append(
                f"{label}: replay {prediction['stage']} {prediction['named_cluster']} vs on "
                f"{row.get('stage')} {row.get('named_cluster')}"
            )
    if len(predictions) != len(rows):
        mismatches.append(f"row counts differ: {len(predictions)} vs {len(rows)}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", nargs="+", type=Path, help="artifacts run with clusters off")
    parser.add_argument("--on", nargs="*", type=Path, default=[], help="matching on-run artifacts")
    parser.add_argument("--separation", type=float, default=0.5)
    parser.add_argument("--min-discriminative", type=float, default=3.0)
    parser.add_argument("--cluster-min-discriminative", type=float, default=3.5)
    parser.add_argument("--min-probability", type=float, default=0.8)
    parser.add_argument(
        "--scope",
        choices=("documented", "full", "nested"),
        default=None,
        help="default: the scope each artifact recorded (candidates were selected under it)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.on and len(args.on) != len(args.off):
        parser.error("--on needs one artifact per off artifact")
    status = 0
    for index, path in enumerate(args.off):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        scope = args.scope or recorded_scope(artifact)
        if args.scope and args.scope != recorded_scope(artifact):
            print(
                f"note: {path.name} was recorded under scope {recorded_scope(artifact)!r}; a "
                f"replay under {args.scope!r} only drops candidates and cannot find the twin the "
                "engine would select under it",
                file=sys.stderr,
            )
        result = replay(artifact, thresholds_for(args, scope))
        mismatches = (
            compare(result["predictions"], json.loads(args.on[index].read_text(encoding="utf-8")))
            if args.on
            else None
        )
        summary = {k: v for k, v in result.items() if k != "predictions"}
        if mismatches is not None:
            summary["on_mismatches"] = mismatches
            status = status or int(bool(mismatches))
        if args.json:
            print(json.dumps({"artifact": path.name, "scope": scope, **summary}))
            continue
        print(f"== {path.name} (scope {scope})")
        print(
            f"  single names: {summary['single_named_off']} off -> {summary['single_named_on']} "
            f"on ({summary['single_correct_on']} correct); upgraded {len(summary['upgrades'])}, "
            f"held back {len(summary['held_back'])}"
        )
        print(
            f"  cluster names: {len(summary['cluster_correct'])} correct "
            f"({summary['cluster_recovered']} with a wrong leader), "
            f"{len(summary['cluster_wrong'])} wrong; removed actors named via a cluster: "
            f"{len(summary['removed_named_by_cluster'])}"
        )
        for key in (
            "cluster_correct",
            "cluster_wrong",
            "removed_named_by_cluster",
            "removed_single_named",
            "held_back",
        ):
            for item in summary[key]:
                print(f"    {key}: {item}")
        print(f"  removed actors clearing 2 of 3 cluster bars: {summary['removed_two_of_three']}")
        if summary["missing_candidate_field"]:
            print(f"  WARNING: {summary['missing_candidate_field']} rows lack cluster_candidate")
        if mismatches is not None:
            print(f"  on-run agreement: {mismatches if mismatches else 'exact'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
