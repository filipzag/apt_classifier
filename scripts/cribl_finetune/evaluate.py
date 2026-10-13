"""Score a case split with cribl-decision (``.venv-cribl``; README.md in this folder).

Asks the Jev C1 question (flat choice over the case's top-10 candidates + none_of_these, three
option orders combined) through the lettered readout and writes per-case predictions
(``<out>/<name>_<split>.jsonl``, resumable) and metrics (``<out>/<name>_<split>.json``): exact,
parent-aware, no-pick, ECE, per dataset, and paired deltas against the pipeline's rank-1 and, where
stored, Jev C1 on the same cases.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import (
    C1_INSTRUCTIONS,
    FORMATS,
    ROTATIONS,
    c1_options,
    c1_state,
    paired_delta,
    pick_from,
    read_jsonl,
    reference_rows,
    score_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "audits" / "cribl_ft"


def predict_case(readout: Any, case: dict[str, Any], rotations: int) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "gold_ids": case["gold_ids"],
    }
    if not case["candidates"]:  # Jev C1 treats an empty list as an error: a miss
        return {**row, "status": "error", "error": "no candidates", "pick": None, "p_pick": None}
    options = c1_options(case["candidates"])
    order = [option["id"] for option in options]
    combined, per_rotation = readout.choice(
        c1_state(case["text"]), C1_INSTRUCTIONS, options, rotations
    )
    pick, p_pick = pick_from(combined, order)
    first_pick, first_p = pick_from(per_rotation[0], order)
    return {
        **row,
        "status": "ok" if pick else "no_pick",
        "pick": pick,
        "p_pick": None if pick is None else p_pick,
        "distribution": combined,
        "rotations": per_rotation,
        "single_order_pick": first_pick,
        "single_order_p": None if first_pick is None else first_p,
    }


def metrics(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    single = [
        {**row, "pick": row.get("single_order_pick"), "p_pick": row.get("single_order_p")}
        for row in rows
    ]
    pipeline = reference_rows(cases, "pipeline")
    jev = reference_rows(cases, "jev_c1")
    report: dict[str, Any] = {
        "arm": score_rows(rows),
        "arm_single_order": score_rows(single),
        "pipeline_rank1": score_rows(pipeline),
        "vs_pipeline_rank1": paired_delta(pipeline, rows),
        "gold_in_candidates": round(
            sum(case["gold_in_candidates"] for case in cases) / max(len(cases), 1), 6
        ),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
    }
    if jev:
        report["jev_c1"] = score_rows(jev)
        report["vs_jev_c1"] = paired_delta(jev, rows)
    return report


def run_split(
    readout: Any,
    cases: list[dict[str, Any]],
    predictions_path: Path,
    *,
    rotations: int = ROTATIONS,
    progress: int = 50,
) -> list[dict[str, Any]]:
    """Predict every case not yet in ``predictions_path`` (appending), then return all rows in
    case order."""
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        done = {row["case_id"]: row for row in read_jsonl(predictions_path)}
    started = time.monotonic()
    fresh = 0
    with predictions_path.open("a", encoding="utf-8") as handle:
        for case in cases:
            if case["case_id"] in done:
                continue
            row = predict_case(readout, case, rotations)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            done[case["case_id"]] = row
            fresh += 1
            if progress and fresh % progress == 0:
                rate = (time.monotonic() - started) / fresh
                left = sum(1 for item in cases if item["case_id"] not in done)
                print(
                    f"  {len(done)}/{len(cases)} cases, {rate:.2f} s/case, ~{left * rate / 60:.0f} min left",
                    flush=True,
                )
    return [done[case["case_id"]] for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--split", required=True, choices=["dev", "cti_hal", "test", "annoctr_dev", "annoctr_test"]
    )
    parser.add_argument("--adapter", default="cribl", help="cribl | none | <checkpoint dir>")
    parser.add_argument("--format", default="plain", choices=FORMATS)
    parser.add_argument("--name", required=True, help="arm name for the output files")
    parser.add_argument("--limit", type=int, help="first N cases only (format probe)")
    parser.add_argument("--rotations", type=int, default=ROTATIONS)
    parser.add_argument("--cases-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from decision_model import Readout, load_model, with_oom_retry

    cases = read_jsonl(args.cases_dir / f"cases_{args.split}.jsonl")
    if args.limit:
        cases = cases[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.split}" + (f"_first{args.limit}" if args.limit else "")
    model, tokenizer, info = load_model(args.adapter)
    readout = Readout(model, tokenizer, args.format)
    started = time.monotonic()
    rows = with_oom_retry(
        lambda: run_split(readout, cases, args.out / f"{stem}.jsonl", rotations=args.rotations),
        note=stem,
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "limit": args.limit,
        "format": args.format,
        "rotations": args.rotations,
        "model": info,
        "runtime_s": round(time.monotonic() - started, 1),
        **metrics(cases, rows),
    }
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arm, base = report["arm"]["pooled"], report["pipeline_rank1"]["pooled"]
    line = f"{stem}: exact {arm['exact']} parent-aware {arm['parent_aware']} no-pick {arm['no_pick']} ECE {arm['ece']} | pipeline rank-1 exact {base['exact']}"
    if "jev_c1" in report:
        line += f" | Jev C1 exact {report['jev_c1']['pooled']['exact']}"
    print(line)


if __name__ == "__main__":
    main()
