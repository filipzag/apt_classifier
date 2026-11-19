"""Let the family head name more than one family (prereg ``cribl_family_head_2026-10-13.json``,
amendment 3; standard library only).

The first family is always the most probable one; another family is added when its probability is
at least ``threshold``, up to ``MAX_FAMILIES`` in all. Scored as family micro precision / recall /
F1 over the gold families (a gold id's parent; a family outside the catalogue is a miss). The
threshold is chosen on the dev predictions only (mean F1 over the dev files; ties: the higher
threshold) and applied once to the confirmation sets.

  tune:   python3 multi_family.py tune F1_step15521_dev.jsonl F1_step15521_annoctr_dev.jsonl
  apply:  python3 multi_family.py apply --threshold T F1_cti_hal.jsonl F1_test.jsonl ...

Files are read from ``data/audits/cribl_ft`` unless given as absolute paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import parent_of, read_jsonl  # noqa: E402

AUDITS = HERE.parents[1] / "data" / "audits" / "cribl_ft"
MAX_FAMILIES = 3
GRID = tuple(round(0.05 * step, 2) for step in range(1, 11))  # 0.05 … 0.50
TOP1_ONLY = 1.01  # a threshold no second family can reach


def families_named(
    row: Mapping[str, Any], threshold: float, max_families: int = MAX_FAMILIES
) -> list[str]:
    """The families a row names under ``threshold`` (its ``top`` list is sorted by probability)."""
    top = row.get("top") or []
    if not top:
        return []
    named = [top[0][0]]
    for family, probability in top[1:max_families]:
        if probability >= threshold:
            named.append(family)
    return named


def micro_prf(rows: Iterable[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = count = 0
    for row in rows:
        gold = {parent_of(item) for item in row["gold_ids"]}
        named = set(families_named(row, threshold))
        tp += len(named & gold)
        fp += len(named - gold)
        fn += len(gold - named)
        count += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "families_per_case": round((tp + fp) / count, 4) if count else None,
    }


def tune(files: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    curve = []
    for threshold in (TOP1_ONLY, *GRID):
        scores = {name: micro_prf(rows, threshold) for name, rows in files.items()}
        mean = sum(score["f1"] for score in scores.values()) / len(scores)
        curve.append({"threshold": threshold, "mean_f1": round(mean, 6), "sets": scores})
    best = max(curve, key=lambda item: (item["mean_f1"], item["threshold"]))
    return {"chosen_threshold": best["threshold"], "max_families": MAX_FAMILIES, "curve": curve}


def _path(name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else AUDITS / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("tune", "apply"))
    parser.add_argument("files", nargs="+")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = {name: read_jsonl(_path(name)) for name in args.files}
    if args.command == "tune":
        report = tune(files)
        for item in report["curve"]:
            print(
                f"threshold {item['threshold']:.2f}: mean F1 {item['mean_f1']:.4f} | "
                + " | ".join(
                    f"{name} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} "
                    f"({s['families_per_case']}/case)"
                    for name, s in item["sets"].items()
                )
            )
        print(f"chosen on dev: {report['chosen_threshold']}")
    else:
        if args.threshold is None:
            raise SystemExit("apply needs --threshold (the one chosen on dev)")
        report = {
            "threshold": args.threshold,
            "max_families": MAX_FAMILIES,
            "sets": {
                name: {
                    "top1_only": micro_prf(rows, TOP1_ONLY),
                    "multi": micro_prf(rows, args.threshold),
                }
                for name, rows in files.items()
            },
        }
        for name, item in report["sets"].items():
            one, multi = item["top1_only"], item["multi"]
            print(
                f"{name}: one family F1 {one['f1']:.3f} (P {one['precision']:.3f} R {one['recall']:.3f})"
                f" -> up to {MAX_FAMILIES} F1 {multi['f1']:.3f} (P {multi['precision']:.3f}"
                f" R {multi['recall']:.3f}, {multi['families_per_case']}/case)"
            )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
