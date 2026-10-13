"""Re-decide stored predictions with the never-abstain rule (prereg amendment 4; README.md here).

Reads ``<name>_<split>.jsonl`` files written by ``evaluate.py`` (or a training run's
``<arm>_step<N>_dev.jsonl``) and, without the model, picks the most probable technique of each
stored three-order distribution instead of ``none_of_these`` (:func:`common.forced_pick`). Writes
``<stem>_forced.json`` beside each input: the arm under the rule (3 orders and the first order),
the arm as registered, the pipeline's rank-1, Jev C1 as reported and under the same rule, the
paired deltas against each, and how well P(none_of_these) flags the cases whose gold technique
is not among the candidates (AUROC; a diagnostic). Standard library only.

Usage: python3 scripts/cribl_finetune/rescore.py --split cti_hal B1_cti_hal.jsonl B2_cti_hal.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    FLAT_K,
    forced_row,
    paired_delta,
    read_jsonl,
    reference_rows,
    score_rows,
)

REPO_ROOT = HERE.parents[1]
OUT = REPO_ROOT / "data" / "audits" / "cribl_ft"


def auroc(scored: list[tuple[float, bool]]) -> float | None:
    """Probability that a positive scores above a negative (ties count half)."""
    positives = sum(1 for _, label in scored if label)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(scored, key=lambda item: item[0])
    rank_sum, index = 0.0, 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        mean_rank = (index + end) / 2 + 1
        rank_sum += mean_rank * sum(1 for _, label in ordered[index : end + 1] if label)
        index = end + 1
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 6)


def rescore(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {case["case_id"]: case for case in cases}
    order = {
        case["case_id"]: [item["id"] for item in case["candidates"][:FLAT_K]] for case in cases
    }
    forced = [forced_row(row, order[row["case_id"]]) for row in rows]
    single = [
        forced_row(
            {**row, "first": (row.get("rotations") or [None])[0]}, order[row["case_id"]], "first"
        )
        for row in rows
    ]
    absent = [
        (row["p_none"], not by_id[row["case_id"]]["gold_in_candidates"])
        for row in forced
        if row.get("p_none") is not None
    ]
    pipeline = reference_rows(cases, "pipeline")
    report: dict[str, Any] = {
        "rule": "never_abstain (prereg amendment 4)",
        "arm": score_rows(forced),
        "arm_single_order": score_rows(single),
        "arm_as_registered": score_rows(rows),
        "pipeline_rank1": score_rows(pipeline),
        "vs_pipeline_rank1": paired_delta(pipeline, forced),
        "gold_in_candidates": round(
            sum(case["gold_in_candidates"] for case in cases) / max(len(cases), 1), 6
        ),
        "none_flag": {
            "auroc_gold_absent": auroc(absent),
            "mean_p_none": round(math.fsum(p for p, _ in absent) / len(absent), 6)
            if absent
            else None,
        },
    }
    jev = reference_rows(cases, "jev_c1")
    if jev:
        jev_forced = reference_rows(cases, "jev_c1_forced")
        report["jev_c1"] = score_rows(jev)
        report["jev_c1_forced"] = score_rows(jev_forced)
        report["vs_jev_c1"] = paired_delta(jev, forced)
        report["vs_jev_c1_forced"] = paired_delta(jev_forced, forced)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--split", required=True, choices=["dev", "cti_hal", "test", "annoctr_dev", "annoctr_test"]
    )
    parser.add_argument("predictions", nargs="+", help="prediction files (names in --dir or paths)")
    parser.add_argument("--dir", type=Path, default=OUT)
    args = parser.parse_args()
    cases = read_jsonl(args.dir / f"cases_{args.split}.jsonl")
    wanted = {case["case_id"] for case in cases}
    for name in args.predictions:
        path = Path(name) if Path(name).exists() else args.dir / name
        rows = read_jsonl(path)
        got = {row["case_id"] for row in rows}
        if got != wanted:
            raise SystemExit(
                f"{path.name}: {len(wanted - got)} cases missing, {len(got - wanted)} not in the "
                f"{args.split} case file; score the whole split first"
            )
        report = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "split": args.split,
            "predictions": path.name,
            **rescore(cases, rows),
        }
        target = path.with_name(f"{path.stem}_forced.json")
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        arm, base = report["arm"]["pooled"], report["pipeline_rank1"]["pooled"]
        line = (
            f"{path.stem} never-abstain: exact {arm['exact']} parent-aware {arm['parent_aware']} "
            f"ECE {arm['ece']} | as registered {report['arm_as_registered']['pooled']['exact']} "
            f"| pipeline rank-1 {base['exact']}"
        )
        if "jev_c1" in report:
            line += (
                f" | Jev C1 {report['jev_c1']['pooled']['exact']}"
                f" (never-abstain {report['jev_c1_forced']['pooled']['exact']})"
            )
        print(line)


if __name__ == "__main__":
    main()
