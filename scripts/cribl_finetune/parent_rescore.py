"""Parent-level scoring of stored family-head predictions (standard library only; prereg
``cribl_recipe_base_2026-10-15.json``).

Prediction files (``family_head.py evaluate``) carry the gold ids, the family answer (``top``) and,
with a technique head, the exact answer (``technique_pick``). Three numbers per run, pooled and per
dataset, and paired bootstrap 95% intervals against a baseline:

- family top-1: the first family is a gold family;
- parent level: the answered technique's parent is a gold id's parent (the family answer when the
  run has no technique head) -- the headline of the recipe and base-model runs;
- exact: the answered technique is a gold id (strict; ``None`` without a technique head).

  python parent_rescore.py E1=E1_cti_hal.jsonl E2=E2_cti_hal.jsonl E3=E3_cti_hal.jsonl --baseline E2
  python parent_rescore.py ... --dataset expert --output <file>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import bootstrap_mean_differences, hits, read_jsonl

AUDITS = Path(__file__).resolve().parents[2] / "data" / "audits" / "cribl_ft"
MEASURES = ("family_top1", "parent_level", "exact")


def answered(row: Mapping[str, Any]) -> str | None:
    """The run's technique answer, or its family answer without a technique head."""
    if row.get("status") != "ok":
        return None
    return row.get("technique_pick") or row.get("pick")


def measures(row: Mapping[str, Any]) -> dict[str, float | None]:
    answer = answered(row)
    exact, parent = hits(answer, row["gold_ids"])
    family = row.get("status") == "ok" and row["top"][0][0] in row["gold_families"]
    return {
        "family_top1": float(family),
        "parent_level": float(parent),
        "exact": float(exact) if "technique_pick" in row or answer is None else None,
    }


def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [measures(row) for row in rows]
    out: dict[str, Any] = {"n": len(rows)}
    for name in MEASURES:
        values = [item[name] for item in scored if item[name] is not None]
        out[name] = (
            round(sum(values) / len(values), 6) if values and len(values) == len(rows) else None
        )
    out["errors"] = sum(1 for row in rows if row.get("status") != "ok")
    return out


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        datasets.setdefault(str(row.get("dataset") or "all"), []).append(row)
    return {
        "pooled": block(rows),
        "datasets": {name: block(subset) for name, subset in sorted(datasets.items())},
    }


def compare(base: Sequence[Mapping[str, Any]], arm: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Arm minus base on the cases both scored, with paired bootstrap 95% intervals."""
    by_id = {row["case_id"]: row for row in base}
    pairs = [(by_id[row["case_id"]], row) for row in arm if row["case_id"] in by_id]
    columns: dict[str, list[float]] = {name: [] for name in MEASURES}
    for before, after in pairs:
        b, a = measures(before), measures(after)
        for name in MEASURES:
            if b[name] is not None and a[name] is not None:
                columns[name].append(a[name] - b[name])
    columns = {name: values for name, values in columns.items() if len(values) == len(pairs)}
    return {
        "n": len(pairs),
        "delta": {name: round(sum(values) / len(values), 6) for name, values in columns.items()},
        "ci95": bootstrap_mean_differences(columns) if columns else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="NAME=<predictions .jsonl> (relative: the audits)")
    parser.add_argument("--baseline", help="the NAME every other run is compared with")
    parser.add_argument("--dataset", help="score only the rows of this dataset (e.g. expert)")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args()
    runs: dict[str, list[dict[str, Any]]] = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        rows = read_jsonl(Path(path) if Path(path).is_absolute() else AUDITS / path)
        runs[name] = [row for row in rows if not args.dataset or row["dataset"] == args.dataset]
    report: dict[str, Any] = {"dataset": args.dataset, "runs": {}, "vs": {}}
    for name, rows in runs.items():
        report["runs"][name] = summary(rows)
        pooled = report["runs"][name]["pooled"]
        print(
            f"{name}: n {pooled['n']} family top-1 {pooled['family_top1']} parent level "
            f"{pooled['parent_level']} exact {pooled['exact']} errors {pooled['errors']}"
        )
    if args.baseline:
        for name, rows in runs.items():
            if name != args.baseline:
                delta = compare(runs[args.baseline], rows)
                report["vs"][name] = delta
                print(
                    f"{name} - {args.baseline}: {delta['delta']} {delta['ci95']} (n {delta['n']})"
                )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
