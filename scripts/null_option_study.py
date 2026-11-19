"""Can the combiner's null option ("no documented group") hold confident removed-actor names?

Development study (docs 21.3.23), no artifact written. Group-level five-fold cross-validation on
the real-report combiner's training lists (the fold lists of the procedure combiner and the
development real-report lists at weight 10, l2 1, the adopted recipe), out of fold:

1. null variants: the null utility on the evidence-regime features (V0, as fitted), plus the best
   candidate's absolute fit (V1: matched share, distinctive share, exclusive software), plus the
   runner-up gap, support and discrimination channels (V2). Because the null option sits in the
   softmax, a better null lowers the leader probability and the naming floor holds the name; the
   trade is removed-actor "names" (unknown lists, leader p >= floor) against correct ones.
2. upper bound: among the lists V0 would name (leader p >= 0.5, correct leader or unknown list), a
   gradient-boosted classifier on the leader's and runner-up's feature rows, the regime and the
   probabilities, against a higher floor (leader p alone) at the same cost in correct names.

    PYTHONPATH=. .venv/bin/python scripts/null_option_study.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier import combiner as C  # noqa: E402

FOLD_LISTS = REPO_ROOT / "data/audits/procedure_match/training_lists_procedure_match.jsonl.gz"
REAL_LISTS = REPO_ROOT / "data/audits/real_combiner/training_lists_real.jsonl.gz"
REAL_WEIGHT = 10
L2 = 1.0
FOLDS = 5
BASE_REGIME = tuple(C.REGIME_FEATURES)
COLUMN = {name: i for i, name in enumerate(C.FEATURE_NAMES)}
FIT_EXTRAS = ("max_matched_share", "max_matched_distinctive", "max_exclusive_software")
VARIANTS: dict[str, tuple[str, ...]] = {
    "V0": (),
    "V1": FIT_EXTRAS,
    "V2": (
        *FIT_EXTRAS,
        "max_support",
        "second_rrf_rel",
        "max_disc_technique",
        "max_disc_behavior",
        "max_disc_software",
    ),
}


def list_extras(features: np.ndarray) -> dict[str, float]:
    """List-level absolute fit of the best candidate (maxima over the shortlist)."""
    if not len(features):
        return dict.fromkeys(VARIANTS["V2"], 0.0)

    def top(name: str) -> float:
        return float(features[:, COLUMN[name]].max())

    rrf = np.sort(features[:, COLUMN["rrf_rel"]])[::-1]
    return {
        "max_matched_share": top("matched_share"),
        "max_matched_distinctive": top("matched_distinctive_share"),
        "max_exclusive_software": top("exclusive_software"),
        "max_support": top("rrf_support") / 10.0,
        "second_rrf_rel": float(rrf[1]) if len(rrf) > 1 else 0.0,
        "max_disc_technique": top("disc_technique") / 5.0,
        "max_disc_behavior": top("disc_behavior") / 5.0,
        "max_disc_software": top("disc_software") / 5.0,
    }


def load() -> list[dict[str, Any]]:
    records = []
    for path in (FOLD_LISTS, REAL_LISTS):
        with gzip.open(path, "rt") as handle:
            records += [json.loads(line) for line in handle]
    records = [r for r in records if r.get("truth") is None or r["truth"] in r["group_ids"]]
    for record in records:
        lst = C.CombinerList.from_dict(record)
        record["_list"] = lst
        record["_extra"] = list_extras(lst.features)
    return records


def is_real(record: dict[str, Any]) -> bool:
    return str(record.get("source") or "").startswith("real:")


def with_extras(record: dict[str, Any], variant: str) -> C.CombinerList:
    lst = record["_list"]
    regime = {**lst.regime, **{k: record["_extra"][k] for k in VARIANTS[variant]}}
    return C.CombinerList(lst.group_ids, lst.features, regime, lst.truth)


def use_variant(variant: str) -> None:
    """The null row reads REGIME_FEATURES; widen it for this study only."""
    C.REGIME_FEATURES = BASE_REGIME + VARIANTS[variant]
    C.NULL_FEATURE_NAMES = ("null_bias", *C.REGIME_FEATURES)


def splits(records: list[dict[str, Any]]) -> dict[str, int]:
    groups = sorted({r["group_id"] for r in records})
    random.Random(0).shuffle(groups)
    return {g: i % FOLDS for i, g in enumerate(groups)}


def out_of_fold(records: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    use_variant(variant)
    split_of = splits(records)
    rows = []
    for split in range(FOLDS):
        train = [r for r in records if split_of[r["group_id"]] != split]
        lists = [with_extras(r, variant) for r in train if not is_real(r)]
        lists += [with_extras(r, variant) for r in train if is_real(r)] * REAL_WEIGHT
        model = C.AttributionCombiner.fit(
            lists, attack_version="study", pipeline_fingerprint="study", l2=L2
        )
        for record in records:
            if split_of[record["group_id"]] != split:
                continue
            lst = with_extras(record, variant)
            probabilities, null = model.probabilities(lst)
            if not probabilities:
                continue
            order = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
            rows.append(
                {
                    "record": record,
                    "split": split,
                    "real": is_real(record),
                    "variant": int(record["variant"]),
                    "unknown": record.get("truth") is None,
                    "correct": order[0] == record.get("truth"),
                    "order": order,
                    "probabilities": probabilities,
                    "null": float(null or 0.0),
                }
            )
    use_variant("V0")
    return rows


def leader_p(row: dict[str, Any]) -> float:
    return row["probabilities"][row["order"][0]]


def trade(rows: list[dict[str, Any]], real: bool, known_variants: set[int]) -> list[str]:
    known = [
        r for r in rows if r["real"] == real and not r["unknown"] and r["variant"] in known_variants
    ]
    unknown = [r for r in rows if r["real"] == real and r["unknown"]]
    lines = []
    for floor in (0.3, 0.4, 0.5, 0.6, 0.7):
        correct = sum(1 for r in known if r["correct"] and leader_p(r) >= floor)
        wrong = sum(1 for r in known if not r["correct"] and leader_p(r) >= floor)
        removed = sum(1 for r in unknown if leader_p(r) >= floor)
        lines.append(
            f"    floor {floor}: correct {correct}/{len(known)} wrong {wrong} "
            f"removed {removed}/{len(unknown)}"
        )
    return lines


def upper_bound(rows: list[dict[str, Any]]) -> list[str]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    X, y, real, split = [], [], [], []
    for r in rows:
        leader = r["order"][0]
        p = r["probabilities"]
        if p[leader] < 0.5 or (not r["unknown"] and not r["correct"]):
            continue
        lst = r["record"]["_list"]
        second = r["order"][1] if len(r["order"]) > 1 else leader
        X.append(
            np.concatenate(
                [
                    lst.features[lst.group_ids.index(leader)],
                    lst.features[lst.group_ids.index(second)],
                    [lst.regime.get(k, 0.0) for k in BASE_REGIME],
                    [p[leader], p[second], r["null"]],
                ]
            )
        )
        y.append(1 if r["unknown"] else 0)
        real.append(r["real"])
        split.append(r["split"])
    X, y, real, split = (np.asarray(v) for v in (X, y, real, split))
    score = np.zeros(len(y))
    for s in range(FOLDS):
        train, test = split != s, split == s
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced"
        )
        clf.fit(X[train], y[train])
        score[test] = clf.predict_proba(X[test])[:, 1]
    leader_p = X[:, -3]
    lines = []
    for name, mask in (("folds", ~real), ("real", real)):
        yy = y[mask]
        lines.append(
            f"  {name}: {int(yy.sum())} removed names, {int((yy == 0).sum())} correct names; "
            f"AUC classifier {roc_auc_score(yy, score[mask]):.3f}, "
            f"leader p alone {roc_auc_score(yy, -leader_p[mask]):.3f}"
        )
        for cost in (0.01, 0.02, 0.05, 0.10):
            held = []
            for s in (score[mask], -leader_p[mask]):
                threshold = np.quantile(s[yy == 0], 1 - cost)
                held.append(int((s[yy == 1] > threshold).sum()))
            lines.append(
                f"    at {cost:.0%} of correct names: classifier holds {held[0]}, "
                f"a higher floor holds {held[1]} of {int(yy.sum())}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args(argv)
    records = load()
    print(f"{len(records)} lists")
    base: list[dict[str, Any]] = []
    for variant in args.variants:
        rows = out_of_fold(records, variant)
        if variant == "V0":
            base = rows
        print(f"== {variant}\n  fold lists (clean):")
        print("\n".join(trade(rows, False, {0})))
        print("  development real-report lists:")
        print("\n".join(trade(rows, True, {0, 1})))
    if base:
        print("== upper bound (lists V0 names at p >= 0.5)")
        print("\n".join(upper_bound(base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
