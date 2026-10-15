"""Refit the product's group network for extracted-technique inputs (project .venv; prereg
``cribl_group_mlp_refit_2026-10-16.json``).

The network (``apt_classifier.group_mlp``) is fitted on synthetic draws whose shape (3-20 true
techniques, 40-95% precision) was set by hand. Two changes, both through the fitter's own options:

1. ``measure``: how the reference technique model (E2) actually extracts, measured on the
   CTI-Rationale train and validation reports (labelled paragraphs; never a test set here): per
   report the extracted set's size, its precision at exact and parent level and what the false
   positives are (a parent or sub-technique of a labelled one, a sibling, or unrelated). Written to
   ``group_mlp_noise.json`` and turned into draw parameters (true-technique count, precision range).
2. ``fit``: a small grid over the real-case weight (the 34 vendor reports with the techniques E2
   names on them, ``real_cases``) and the measured draw parameters, each fit selected on the ATT&CK
   campaigns (techniques only, MRR), then the chosen model scored once on the CTI-HAL and Elastic
   inputs (``cases_group_*.jsonl``) against the current artifact, both with the software block empty.
   The model goes to a new file (``group_mlp_extracted.npz``); the product's artifact is untouched.

  PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/refit_group_mlp.py measure
  PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/refit_group_mlp.py fit [--weights 0,3,10,30]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from common import bootstrap_mean_differences, parent_of, read_jsonl, write_jsonl  # noqa: E402
from group_baseline import ranked_row  # noqa: E402

AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
NOISE = AUDITS / "group_mlp_noise.json"
MODEL_OUT = Path("/home/fzagar/apt_classifier/data/models/mlp/group_mlp_extracted.npz")
TEST_SPLITS = (
    "group_cti_hal_extracted",
    "group_cti_hal_gold",
    "group_cti_hal_accepted",
    "group_elastic",
)
DEV_SPLIT = "group_campaigns"
REAL_SPLIT = "group_vendor"
NAME = "MLP2"


# ----------------------------------------------------------------------------- measure
def report_sets(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[set[str], set[str]]]:
    """report id -> (labelled technique ids, extracted technique ids) over its paragraphs."""
    gold: dict[str, set[str]] = defaultdict(set)
    extracted: dict[str, set[str]] = defaultdict(set)
    picks = {
        row["case_id"]: row.get("technique_pick")
        for row in predictions
        if row.get("status") == "ok"
    }
    for case in cases:
        # case ids are "cti_rationale-<report id>-<chunk id>"; report ids may hold hyphens
        report = case["case_id"].rsplit("-", 1)[0].split("-", 1)[1]
        gold[report].update(case["gold_ids"])
        if picks.get(case["case_id"]):
            extracted[report].add(picks[case["case_id"]])
    return {report: (gold[report], extracted.get(report, set())) for report in gold}


def false_positive_kind(technique: str, gold: set[str]) -> str:
    """granularity: the parent or a sub-technique of a labelled id; sibling: another sub-technique
    of a labelled family; other: unrelated."""
    parents = {parent_of(g) for g in gold}
    if technique in parents or parent_of(technique) in gold:
        return "granularity"
    if parent_of(technique) in parents:
        return "sibling"
    return "other"


def measure_noise(sets: Mapping[str, tuple[set[str], set[str]]]) -> dict[str, Any]:
    sizes, precisions, parent_precisions, recalls, kinds = [], [], [], [], Counter()
    for gold, extracted in sets.values():
        if not gold or not extracted:
            continue
        sizes.append(len(extracted))
        right = {t for t in extracted if t in gold}
        parent_right = {
            t
            for t in extracted
            if t in gold or parent_of(t) in gold or t in {parent_of(g) for g in gold}
        }
        precisions.append(len(right) / len(extracted))
        parent_precisions.append(len(parent_right) / len(extracted))
        recalls.append(len(right) / len(gold))
        for t in extracted - right:
            kinds[false_positive_kind(t, gold)] += 1

    def quantiles(values: Sequence[float]) -> dict[str, float]:
        q = statistics.quantiles(values, n=10) if len(values) >= 10 else sorted(values)
        return {
            "min": round(min(values), 3),
            "q10": round(q[0], 3),
            "median": round(statistics.median(values), 3),
            "q90": round(q[-1], 3),
            "max": round(max(values), 3),
            "mean": round(statistics.mean(values), 3),
        }

    total = sum(kinds.values()) or 1
    return {
        "reports": len(sizes),
        "extracted_size": quantiles(sizes),
        "precision_exact": quantiles(precisions),
        "precision_parent": quantiles(parent_precisions),
        "recall_exact": quantiles(recalls),
        "false_positive_kinds": {k: round(v / total, 3) for k, v in kinds.items()},
    }


def draw_params(noise: Mapping[str, Any]) -> dict[str, Any]:
    """Draw parameters from the measured extraction: the true-technique count spans the extracted
    sizes scaled by the median precision; the precision range is the measured 10th-90th
    percentile band at parent level (the network sets a sub-technique's parent itself)."""
    size = noise["extracted_size"]
    precision = noise["precision_parent"]
    median_precision = max(float(precision["median"]), 0.05)
    return {
        "size_min": max(2, int(round(size["q10"] * median_precision))),
        "size_max": max(4, int(round(size["q90"] * median_precision))),
        "precision_min": round(max(0.1, float(precision["q10"])), 2),
        "precision_max": round(min(1.0, float(precision["q90"])), 2),
    }


def measure(args: argparse.Namespace) -> None:
    sets: dict[str, tuple[set[str], set[str]]] = {}
    for split in ("cti_rationale_train", "cti_rationale_validation"):
        sets.update(
            report_sets(
                read_jsonl(AUDITS / f"cases_{split}.jsonl"),
                read_jsonl(AUDITS / f"{args.extractor}_{split}.jsonl"),
            )
        )
    noise = measure_noise(sets)
    noise["extractor"] = args.extractor
    noise["source"] = (
        "CTI-Rationale train + validation reports (labelled paragraphs; not a test set here)"
    )
    noise["draw_params"] = draw_params(noise)
    NOISE.write_text(json.dumps(noise, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(noise, indent=2))


# ----------------------------------------------------------------------------- fit
def vectors(model: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    return np.stack([model.features.vector(row["techniques"], []) for row in rows])


def rank_rows(model: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    probabilities = model.predict(vectors(model, rows))
    return [
        ranked_row(row, dict(zip(model.group_ids, probabilities[i].tolist(), strict=True)))
        for i, row in enumerate(rows)
    ]


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ranks = [row["gold_rank"] or 10**6 for row in rows]
    return {
        "n": len(rows),
        "top1": round(sum(r == 1 for r in ranks) / len(ranks), 4),
        "top3": round(sum(r <= 3 for r in ranks) / len(ranks), 4),
        "top5": round(sum(r <= 5 for r in ranks) / len(ranks), 4),
        "mrr": round(sum(1 / r for r in ranks) / len(ranks), 4),
    }


def fit(args: argparse.Namespace) -> None:
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.group_mlp import fit_group_mlp, load_group_mlp

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    current, status = load_group_mlp(settings.group_mlp_path, kb, device="cpu")
    if current is None:
        raise SystemExit(f"current group network: {status}")
    noise = json.loads(NOISE.read_text(encoding="utf-8"))
    measured = noise["draw_params"]
    dev = read_jsonl(AUDITS / f"cases_{DEV_SPLIT}.jsonl")
    real_rows = read_jsonl(AUDITS / f"cases_{REAL_SPLIT}.jsonl")
    weights = [int(w) for w in args.weights.split(",") if w != ""]
    grid = [
        {"name": "current_draws", "params": {}},
        {"name": "measured_draws", "params": measured},
    ]
    trials: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    started = time.monotonic()
    for setting in grid:
        for weight in weights:
            params = {**setting["params"], "real_weight": weight, "seed": args.seed}
            label = f"{setting['name']}/real_weight={weight}"
            real = None
            if weight > 0:
                real_vectors = vectors(current, real_rows)
                real = (real_vectors, [row["gold_ids"][0] for row in real_rows])
            model = fit_group_mlp(kb, params=params, device=args.device, real_cases=real)
            dev_rows = rank_rows(model, dev)
            score = summary(dev_rows)
            trials.append({"label": label, "params": params, "dev": score})
            fitted[label] = model
            print(f"{label}: campaigns (techniques only) {score}", flush=True)
    baseline_dev = summary(rank_rows(current, dev))
    print(f"current artifact: campaigns (techniques only) {baseline_dev}")
    best = max(trials, key=lambda t: (t["dev"]["mrr"], -trials.index(t)))
    chosen = fitted[best["label"]]
    print(f"selected on dev: {best['label']} (MRR {best['dev']['mrr']})")

    report: dict[str, Any] = {
        "noise": noise,
        "grid": trials,
        "selected": best["label"],
        "current_dev": baseline_dev,
        "seconds": round(time.monotonic() - started, 1),
        "test": {},
    }
    for split in TEST_SPLITS:
        rows = read_jsonl(AUDITS / f"cases_{split}.jsonl")
        mine = rank_rows(chosen, rows)
        theirs = rank_rows(current, rows)
        write_jsonl(AUDITS / f"{NAME}_{split}.jsonl", mine)
        write_jsonl(AUDITS / f"MLP_{split}.jsonl", theirs)
        by_id = {row["case_id"]: row for row in theirs}
        deltas = {
            "top1": [
                float(m["gold_rank"] == 1) - float(by_id[m["case_id"]]["gold_rank"] == 1)
                for m in mine
            ],
            "top5": [
                float(bool(m["gold_rank"]) and m["gold_rank"] <= 5)
                - float(
                    bool(by_id[m["case_id"]]["gold_rank"]) and by_id[m["case_id"]]["gold_rank"] <= 5
                )
                for m in mine
            ],
        }
        report["test"][split] = {
            "refit": summary(mine),
            "current": summary(theirs),
            "delta": {k: round(sum(v) / len(v), 4) for k, v in deltas.items()},
            "ci95": bootstrap_mean_differences(deltas),
        }
        t = report["test"][split]
        print(
            f"{split}: refit top-1 {t['refit']['top1']} top-5 {t['refit']['top5']} MRR {t['refit']['mrr']} | "
            f"current top-1 {t['current']['top1']} top-5 {t['current']['top5']} | "
            f"delta top-1 {t['delta']['top1']} {t['ci95']['top1']}"
        )
    if args.save:
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        chosen.save(MODEL_OUT)
        report["artifact"] = str(MODEL_OUT)
        print(f"saved {MODEL_OUT}")
    (AUDITS / "group_mlp_refit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    m = commands.add_parser("measure")
    m.set_defaults(func=measure)
    m.add_argument("--extractor", default="E2")
    f = commands.add_parser("fit")
    f.set_defaults(func=fit)
    f.add_argument("--weights", default="0,3,10,30", help="real_weight values tried")
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--device", default=None)
    f.add_argument("--save", action="store_true", help=f"write the chosen model to {MODEL_OUT}")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
