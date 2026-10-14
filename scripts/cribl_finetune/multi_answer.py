"""Up to three answers per sentence from stored family-head predictions (standard library only;
prereg ``cribl_yesno_2026-10-14.json``).

A sentence's answers are its most probable family plus the next ones (up to three in all) whose
probability reaches a threshold (``multi_family.families_named``), each with the family's most probable technique (the first of the
family in ``technique_top``, i.e. the highest P(technique | family)). The threshold is set per
model and split to give ``budget`` answers per sentence on average, from the model's own
probabilities and without labels (:func:`budget_threshold`), so models are compared with the same
number of answers. A sentence is a hit when any answer is a labelled family (or, exact, a labelled
technique).

  python multi_answer.py --budget 1.3 E2=E2_cti_hal.jsonl M1=M1_cti_hal.jsonl --baseline E2
  -> per model: answers per sentence, family and exact hits, multi-label coverage, extras that
     match a label; paired bootstrap 95% CIs against the baseline
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import bootstrap_mean_differences, read_jsonl
from multi_family import MAX_FAMILIES, families_named, micro_prf

AUDITS = Path(__file__).resolve().parents[2] / "data" / "audits" / "cribl_ft"
LIMIT = MAX_FAMILIES


def family_of(technique: str) -> str:
    return technique.split(".")[0]


def answers(row: Mapping[str, Any], threshold: float) -> list[tuple[str, str | None]]:
    """(family, its best technique) for the row's answers; none for an error row."""
    if row.get("status") != "ok":
        return []
    families = families_named(row, threshold)
    best: dict[str, str] = {}
    for technique, _ in row.get("technique_top") or ():
        best.setdefault(family_of(technique), technique)
    return [(family, best.get(family)) for family in families]


def budget_threshold(rows: Sequence[Mapping[str, Any]], budget: float, limit: int = LIMIT) -> float:
    """The threshold giving at most ``budget`` answers per sentence on average (labels unused)."""
    extra = math.floor(round((budget - 1.0) * len(rows), 6))
    values = sorted(
        (p for row in rows if row.get("status") == "ok" for _, p in row["top"][1:limit]),
        reverse=True,
    )
    if extra <= 0 or not values:
        return math.inf
    if extra >= len(values):
        return 0.0
    # the next value down is excluded, so ties at the cut cannot push past the budget
    return (
        math.nextafter(values[extra], math.inf)
        if values[extra - 1] == values[extra]
        else values[extra - 1]
    )


def recall(row: Mapping[str, Any], threshold: float) -> tuple[float, float]:
    """(share of the labelled families among the answers, share of the labelled techniques)."""
    found = answers(row, threshold)
    families = {family for family, _ in found}
    techniques = {technique for _, technique in found if technique}
    gold_families, gold = set(row["gold_families"]), set(row["gold_ids"])
    return (
        len(families & gold_families) / len(gold_families) if gold_families else 0.0,
        len(techniques & gold) / len(gold) if gold else 0.0,
    )


def hits(row: Mapping[str, Any], threshold: float) -> tuple[bool, bool]:
    """(a labelled family among the answers, a labelled technique among the answers)."""
    found = answers(row, threshold)
    return (
        any(family in row["gold_families"] for family, _ in found),
        any(technique in row["gold_ids"] for _, technique in found if technique),
    )


def multi_block(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    count = len(rows)
    found = [answers(row, threshold) for row in rows]
    sizes = [len(items) for items in found]
    family_hits = [hits(row, threshold)[0] for row in rows]
    exact_hits = [hits(row, threshold)[1] for row in rows]
    several = [i for i, row in enumerate(rows) if len(row["gold_families"]) > 1]
    single = [i for i, row in enumerate(rows) if len(row["gold_families"]) == 1]
    extras = [(rows[i], family) for i, items in enumerate(found) for family, _ in items[1:]]

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    return {
        "n": count,
        "threshold": None if math.isinf(threshold) else round(threshold, 6),
        "answers_per_sentence": mean(sizes),
        "answers": {str(k): sizes.count(k) for k in range(0, LIMIT + 1) if sizes.count(k)},
        "family_hit": mean(family_hits),
        "exact_hit": mean(exact_hits),
        "answers_single_label": mean([sizes[i] for i in single]),
        "answers_several_labels": mean([sizes[i] for i in several]),
        "several_labels_all_found": mean(
            [set(rows[i]["gold_families"]) <= {family for family, _ in found[i]} for i in several]
        ),
        "several_labels_n": len(several),
        "extras": len(extras),
        "extras_matching_a_label": sum(family in row["gold_families"] for row, family in extras),
        "family_micro": micro_prf(rows, threshold),
        "family_recall_per_case": mean([recall(row, threshold)[0] for row in rows]),
        "exact_recall_per_case": mean([recall(row, threshold)[1] for row in rows]),
    }


def compare(
    base: Sequence[Mapping[str, Any]],
    base_threshold: float,
    arm: Sequence[Mapping[str, Any]],
    arm_threshold: float,
) -> dict[str, Any]:
    """Arm minus base on the cases both scored (paired bootstrap): family and exact hits, and the
    per-case share of labelled families and techniques found (recall)."""
    by_id = {row["case_id"]: row for row in base}
    pairs = [(by_id[row["case_id"]], row) for row in arm if row["case_id"] in by_id]
    names = ("family_hit", "exact_hit", "family_recall", "exact_recall")
    columns: dict[str, list[float]] = {name: [] for name in names}
    for base_row, arm_row in pairs:
        before = (*hits(base_row, base_threshold), *recall(base_row, base_threshold))
        after = (*hits(arm_row, arm_threshold), *recall(arm_row, arm_threshold))
        for name, b, a in zip(names, before, after, strict=True):
            columns[name].append(float(a) - float(b))
    return {
        "n": len(pairs),
        "delta": {name: round(sum(values) / len(values), 6) for name, values in columns.items()},
        "ci95": bootstrap_mean_differences(columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="NAME=<predictions .jsonl> (relative: the audits)")
    parser.add_argument("--budget", type=float, default=1.3)
    parser.add_argument("--baseline", help="the NAME every other run is compared with")
    parser.add_argument(
        "--thresholds", nargs="*", type=float, default=[], help="also report these fixed thresholds"
    )
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    parser.add_argument("--dataset", help="score only the rows of this dataset (e.g. expert)")
    args = parser.parse_args()
    runs: dict[str, list[dict[str, Any]]] = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        rows = read_jsonl(Path(path) if Path(path).is_absolute() else AUDITS / path)
        runs[name] = [row for row in rows if not args.dataset or row["dataset"] == args.dataset]
    report: dict[str, Any] = {"budget": args.budget, "runs": {}, "vs": {}}
    cut = {name: budget_threshold(rows, args.budget) for name, rows in runs.items()}
    for name, rows in runs.items():
        report["runs"][name] = {
            "budget": multi_block(rows, cut[name]),
            "one_answer": multi_block(rows, math.inf),
            **{f"fixed_{t}": multi_block(rows, t) for t in args.thresholds},
        }
        block = report["runs"][name]["budget"]
        print(
            f"{name}: threshold {block['threshold']} answers {block['answers_per_sentence']} "
            f"{block['answers']} family hit {block['family_hit']} exact hit {block['exact_hit']} | "
            f"answers single/several labels {block['answers_single_label']}/"
            f"{block['answers_several_labels']} all labels found {block['several_labels_all_found']}"
            f" (n {block['several_labels_n']}) | extras matching a label "
            f"{block['extras_matching_a_label']}/{block['extras']} | one answer: family "
            f"{report['runs'][name]['one_answer']['family_hit']} exact "
            f"{report['runs'][name]['one_answer']['exact_hit']}"
        )
        for t in args.thresholds:
            fixed = report["runs"][name][f"fixed_{t}"]
            print(
                f"  fixed {t}: answers {fixed['answers_per_sentence']} {fixed['answers']} "
                f"family hit {fixed['family_hit']} exact hit {fixed['exact_hit']}"
            )
    if args.baseline:
        for name, rows in runs.items():
            if name != args.baseline:
                delta = compare(runs[args.baseline], cut[args.baseline], rows, cut[name])
                report["vs"][name] = delta
                print(
                    f"{name} - {args.baseline}: {delta['delta']} {delta['ci95']} (n {delta['n']})"
                )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
