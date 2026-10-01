"""Train/validation/test splits over the CTI-HAL reports for the PyTorch group network.

    PYTHONPATH=. .venv/bin/python scripts/cti_hal_split_eval.py [--folds 5] [--weights 3,10,30]

The 55 reports are assigned to ``--folds`` parts, stratified by group (seeded). Rotation ``k`` uses
part ``k`` as test, part ``k+1`` as validation and the rest as training, so every report is tested
exactly once by a model that never saw it; rotation 0 is also written as the fixed split
(``data/benchmarks/cti_hal/splits.json``, local data like the rest of that folder).

Input per report: the techniques and software the local pipeline accepted from the anonymized text
(``attribution_cti_hal_current.json``). Per rotation the network is trained on the synthetic ATT&CK
cases plus the training reports repeated ``real_weight`` times per draw; the weight is chosen on the
validation part (MRR + macro F1), then the test part is scored. Compared on the same test reports:

- ``synthetic_only``: the same network without real reports;
- ``synthetic_only_closed``: that network restricted to the groups in the training reports, the
  "answer is one of these groups" shortcut a real-report-trained model can learn for free;
- ``campaigns``: the 21 ATT&CK campaigns before/after, which shows whether training on six groups'
  reports costs accuracy on everyone else.

CTI-HAL has no licence: a model trained on it is local evaluation only and is never saved here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fit_group_mlp import CTI_HAL_RUN, evaluation_sets, ranked  # noqa: E402

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.group_mlp import classification_metrics, fit_group_mlp  # noqa: E402

SPLITS = REPO_ROOT / "data" / "benchmarks" / "cti_hal" / "splits.json"
AUDIT = REPO_ROOT / "data" / "audits" / "group_mlp_cti_hal_splits.json"


def assign_folds(report_groups: dict[str, str], folds: int, seed: int) -> dict[str, int]:
    """Report id -> part, stratified by group: each group's reports dealt round-robin."""
    rng = np.random.default_rng(seed)
    out: dict[str, int] = {}
    offset = 0
    for group in sorted(set(report_groups.values())):
        reports = sorted(r for r, g in report_groups.items() if g == group)
        rng.shuffle(reports)
        for i, report in enumerate(reports):
            out[report] = (offset + i) % folds
        offset += len(reports)  # the next group starts where this one stopped
    return out


def _score(metrics: dict[str, Any]) -> float:
    return float(metrics.get("mrr", 0.0)) + float(metrics.get("macro_f1", 0.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--weights", default="3,10,30", help="real_weight values tried")
    parser.add_argument("--members", type=int, default=1, help="networks per fit (speed)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=AUDIT)
    args = parser.parse_args(argv)
    kb = AttackKnowledgeBase.load(Settings().kb_path)
    rows = json.loads(CTI_HAL_RUN.read_text())["known"]
    folds = assign_folds({r["report_id"]: r["group_id"] for r in rows}, args.folds, args.seed)
    rotations = [{"test": k, "val": (k + 1) % args.folds} for k in range(args.folds)]

    def ids(rotation: dict[str, int], part: str) -> list[str]:
        if part == "train":
            skip = {rotation["test"], rotation["val"]}
            return sorted(r for r, f in folds.items() if f not in skip)
        return sorted(r for r, f in folds.items() if f == rotation[part])

    SPLITS.write_text(
        json.dumps(
            {
                "note": "CTI-HAL derived (no licence): local evaluation only",
                "seed": args.seed,
                "folds": dict(sorted(folds.items())),
                "rotations": rotations,
                "fixed_split": {part: ids(rotations[0], part) for part in ("train", "val", "test")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    params = {"members": args.members, "seed": args.seed}
    base = fit_group_mlp(kb, params=params, device=args.device)
    features = base.features
    by_id = {r["report_id"]: r for r in rows}

    def matrix(report_ids: list[str]) -> tuple[np.ndarray, list[str]]:
        return (
            np.stack(
                [
                    features.vector(by_id[r]["accepted_techniques"], by_id[r]["software_ids"])
                    for r in report_ids
                ]
            ),
            [by_id[r]["group_id"] for r in report_ids],
        )

    campaigns_x, campaigns_y = evaluation_sets(kb, base)["campaigns"]
    pooled: dict[str, tuple[list[str], list[list[str]]]] = {
        name: ([], []) for name in ("synthetic_only", "synthetic_only_closed", "with_real")
    }
    per_rotation = []
    campaign_after = []
    for k, rotation in enumerate(rotations):
        train_x, train_y = matrix(ids(rotation, "train"))
        val_x, val_y = matrix(ids(rotation, "val"))
        test_x, test_y = matrix(ids(rotation, "test"))
        tried = {}
        models = {}
        for weight in [int(w) for w in args.weights.split(",") if w]:
            model = fit_group_mlp(
                kb,
                params={**params, "real_weight": weight},
                device=args.device,
                real_cases=(train_x, train_y),
            )
            tried[weight] = classification_metrics(
                val_y, ranked(model.predict(val_x), model.group_ids)
            )
            models[weight] = model
        chosen = max(tried, key=lambda w: (_score(tried[w]), -w))
        model = models[chosen]
        base_probabilities = base.predict(test_x)
        closed = base_probabilities.copy()
        allowed = np.asarray([g in set(train_y) for g in base.group_ids])
        closed[:, ~allowed] = 0.0
        for name, probabilities, group_ids in (
            ("synthetic_only", base_probabilities, base.group_ids),
            ("synthetic_only_closed", closed, base.group_ids),
            ("with_real", model.predict(test_x), model.group_ids),
        ):
            pooled[name][0].extend(test_y)
            pooled[name][1].extend(ranked(probabilities, group_ids))
        after = classification_metrics(
            campaigns_y, ranked(model.predict(campaigns_x), model.group_ids)
        )
        campaign_after.append(after)
        per_rotation.append(
            {
                "rotation": k,
                "sizes": {"train": len(train_y), "val": len(val_y), "test": len(test_y)},
                "validation": {
                    str(w): {m: tried[w][m] for m in ("top1", "mrr", "macro_f1")} for w in tried
                },
                "chosen_real_weight": chosen,
                "campaigns_after": {m: after[m] for m in ("top1", "mrr", "macro_f1")},
            }
        )
        print(
            f"rotation {k}: sizes {per_rotation[-1]['sizes']} chose real_weight={chosen}",
            flush=True,
        )
    campaigns_before = classification_metrics(
        campaigns_y, ranked(base.predict(campaigns_x), base.group_ids)
    )
    test = {name: classification_metrics(*pooled[name]) for name in pooled}
    report = {
        "folds": args.folds,
        "members_per_fit": args.members,
        "cti_hal_run": json.loads(CTI_HAL_RUN.read_text()).get("generated_at"),
        "test_pooled": {
            n: {k: v for k, v in m.items() if k != "per_class"} for n, m in test.items()
        },
        "test_per_class_with_real": test["with_real"]["per_class"],
        "campaigns": {
            "before": {m: campaigns_before[m] for m in ("top1", "mrr", "macro_f1")},
            "after_mean": {
                m: round(float(np.mean([a[m] for a in campaign_after])), 4)
                for m in ("top1", "mrr", "macro_f1")
            },
        },
        "rotations": per_rotation,
        "note": "test pooled over rotations: every report tested once by a model that never saw it",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{'model':22} {'top1':>6} {'top5':>6} {'mrr':>6} {'macroF1':>8}  (55 test reports, pooled)"
    )
    for name, m in report["test_pooled"].items():
        print(
            f"{name:22} {m['top1']:>6.3f} {m['top5']:>6.3f} {m['mrr']:>6.3f} {m['macro_f1']:>8.3f}"
        )
    print(
        f"campaigns before {report['campaigns']['before']} after {report['campaigns']['after_mean']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
