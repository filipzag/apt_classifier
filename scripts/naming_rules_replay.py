"""Offline replay of the independent-inputs naming rules over recorded attribution artifacts.

Two rules (prereg ``attribution_independent_inputs_2026-10-27.json``, docs 10.6 and 21.3.27):

  technique list once  (``MaturityThresholds.technique_list_one_family``) the families in
                       ``attribution.TECHNIQUE_LIST_FAMILIES`` all read the accepted technique list
                       and count as one for the independent-family check; a leader below the
                       minimum goes to ``unresolved_activity_cluster`` before any other step, so it
                       gets no single name and no cluster name
  above null           (``hypothesis_above_null``) a single name needs the combiner's leader
                       probability above its null ("none of these") probability; a held leader
                       falls through to the documented-cluster step, as under the naming floor

The ranking never changes. A run made without the rules records the leader's classifier ranks
(``top3_detail[0]``), the leader and null probabilities and the cluster candidate, so the replay is
exact; ``--on`` checks it row for row against runs made with them. A name is correct when the
leader (or a named cluster member) is the row's group or one of its ``accept`` groups.

    PYTHONPATH=. .venv/bin/python scripts/naming_rules_replay.py <artifacts> [--details]
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

from apt_classifier.attribution import (  # noqa: E402
    MaturityThresholds,
    _cluster_step,
    supporting_families,
)
from apt_classifier.session import _CLASSIFIER_FAMILIES  # noqa: E402

HYPOTHESIS = "known_group_hypothesis"
CLUSTER = "known_cluster_hypothesis"
SHORTLIST = "known_group_shortlist"
UNRESOLVED = "unresolved_activity_cluster"
ARMS = {
    "run": (False, False),
    "technique_list": (True, False),
    "above_null": (False, True),
    "both": (True, True),
}


def recorded_thresholds(artifact: dict[str, Any], technique_list: bool, above_null: bool):
    """The run's own gate thresholds with the two rules set."""
    recorded = dict((artifact.get("fingerprint_inputs") or {}).get("thresholds") or {})
    known = set(MaturityThresholds.__dataclass_fields__)
    values = {key: value for key, value in recorded.items() if key in known}
    values["technique_list_one_family"] = technique_list
    values["hypothesis_above_null"] = above_null
    return MaturityThresholds(**values)


def leader_ranks(row: dict[str, Any]) -> dict[str, Any] | None:
    detail = row.get("top3_detail") or []
    if not detail or detail[0].get("group_id") != row.get("leading"):
        return None
    return dict(detail[0].get("classifier_ranks") or {})


def replay_row(row: dict[str, Any], thresholds: MaturityThresholds) -> dict[str, Any]:
    """The stage, named members and which rule held the row (None when unchanged)."""
    stage = row.get("stage")
    unchanged = {
        "stage": stage,
        "named_cluster": list(row.get("named_cluster") or []),
        "held": None,
    }
    if stage not in (HYPOTHESIS, CLUSTER, SHORTLIST):
        return unchanged
    if thresholds.technique_list_one_family:
        ranks = leader_ranks(row)
        if ranks is None:
            raise ValueError(f"{row.get('report_id') or row.get('group_id')}: no leader ranks")
        families = supporting_families(ranks, _CLASSIFIER_FAMILIES, technique_list_one_family=True)
        if len(families) < thresholds.min_independent_evidence_families:
            return {"stage": UNRESOLVED, "named_cluster": [], "held": "technique_list"}
    probability = row.get("combiner_leader_probability")
    null = row.get("combiner_null_probability")
    if (
        stage == HYPOTHESIS
        and thresholds.hypothesis_above_null
        and probability is not None
        and null is not None
        and float(null) >= float(probability)
    ):
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
            return {"stage": CLUSTER, "named_cluster": list(candidate["members"]), "held": "null"}
        return {"stage": SHORTLIST, "named_cluster": [], "held": "null"}
    return unchanged


def outcome(arm: str, row: dict[str, Any], result: dict[str, Any]) -> str:
    """correct / wrong / removed / open for a name, "" for no name."""
    if result["stage"] not in (HYPOTHESIS, CLUSTER):
        return ""
    if arm == "unknown":
        return "removed"
    truth = {row.get("group_id"), *(row.get("accept") or ())} - {None}
    if arm == "open_set" or not truth:
        return "open"
    named = result["named_cluster"] if result["stage"] == CLUSTER else [row.get("leading")]
    return "correct" if truth & set(named) else "wrong"


def label(arm: str, row: dict[str, Any]) -> str:
    return f"{arm}:{row.get('group_id')}:{row.get('report_id') or ''}".rstrip(":")


def replay(artifact: dict[str, Any], technique_list: bool, above_null: bool) -> dict[str, Any]:
    thresholds = recorded_thresholds(artifact, technique_list, above_null)
    out: dict[str, Any] = {
        "correct": 0,
        "wrong": [],
        "removed": [],
        "open": [],
        "held_correct": [],
        "held_bad": [],
        "cluster_fallback": [],
        "predictions": {},
    }
    for arm in ("known", "unknown", "open_set"):
        for row in artifact.get(arm) or []:
            result = replay_row(row, thresholds)
            name = label(arm, row)
            out["predictions"][name] = result
            kind = outcome(arm, row, result)
            leader = row.get("leading")
            text = f"{name} as {leader} (p {row.get('combiner_leader_probability')})"
            if kind == "correct":
                out["correct"] += 1
            elif kind:
                out[kind].append(text)
            if result["held"] is None:
                continue
            before = outcome(arm, row, {"stage": row.get("stage"), "named_cluster": []})
            if row.get("stage") == CLUSTER:
                before = outcome(
                    arm, row, {"stage": CLUSTER, "named_cluster": row.get("named_cluster") or []}
                )
            if before:
                bucket = "held_correct" if before == "correct" else "held_bad"
                out[bucket].append(f"{text} by {result['held']}")
            if result["stage"] == CLUSTER:
                out["cluster_fallback"].append(f"{name} as {result['named_cluster']} ({kind})")
    return out


def compare(off: dict[str, Any], on: dict[str, Any]) -> list[str]:
    """Rows where the replay of ``off`` with both rules disagrees with the run ``on``."""
    replayed = replay(off, True, True)["predictions"]
    actual = replay(on, False, False)["predictions"]
    mismatches = []
    for name, result in replayed.items():
        other = actual.get(name)
        if other is None or (result["stage"], result["named_cluster"]) != (
            other["stage"],
            other["named_cluster"],
        ):
            mismatches.append(f"{name}: replay {result['stage']} / run {other and other['stage']}")
    return mismatches


def summary(arm: str, report: dict[str, Any]) -> str:
    return (
        f"{arm:15s} correct {report['correct']} wrong {len(report['wrong'])} removed "
        f"{len(report['removed'])} open-set {len(report['open'])} | held "
        f"{len(report['held_correct'])} correct, {len(report['held_bad'])} bad; cluster fallback "
        f"{len(report['cluster_fallback'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--on", nargs="*", type=Path, default=[], help="runs made with the rules")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--output", type=Path, help="write the per-artifact counts as JSON")
    args = parser.parse_args(argv)
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts]
    written: dict[str, Any] = {}
    for path, artifact in zip(args.artifacts, artifacts, strict=True):
        print(path)
        written[str(path)] = {}
        for arm, (technique_list, above_null) in ARMS.items():
            report = replay(artifact, technique_list, above_null)
            print("  " + summary(arm, report))
            if args.details and arm != "run":
                for key in ("held_correct", "held_bad", "cluster_fallback"):
                    if report[key]:
                        print(f"    {key}: {report[key]}")
            written[str(path)][arm] = {k: v for k, v in report.items() if k != "predictions"}
    if args.output:
        args.output.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    if args.on:
        if len(args.on) != len(artifacts):
            parser.error("--on needs one run per artifact")
        failed = 0
        for path, off, on_path in zip(args.artifacts, artifacts, args.on, strict=True):
            mismatches = compare(off, json.loads(on_path.read_text(encoding="utf-8")))
            print(
                f"{path} vs {on_path}: {len(mismatches)} mismatches", *mismatches[:10], sep="\n  "
            )
            failed += len(mismatches)
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
