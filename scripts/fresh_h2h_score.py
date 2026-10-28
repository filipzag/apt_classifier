"""Score the fresh-report head-to-head: the product, a frontier LLM and their registered combination.

Prereg ``fresh_reports_head_to_head_2026-10-28.json``. Per report the product artifact
(``attribution_benchmark.py --report-set fresh``) and the LLM artifact
(``fresh_llm_attribution.py``) at the same reading; only reports both arms answered are scored.

* first choice right (known reports): product leader / the LLM's first mappable candidate;
* top three right: the truth among the product's top three / the LLM's three candidates;
* names: the product's named stages (a cluster name is right when the truth is a member); the
  LLM's ``name`` decision on its first candidate. On the open set every name is wrong.
* combination (registered, no fitting): first choice = the LLM's first candidate when the
  product ranks it in its top five, else the product's leader; a name only when the product's
  leader and the LLM's first candidate are the same group and the LLM decides ``name``.

Paired bootstrap (10,000 report resamples, seed 7) for the first-choice differences and exact
McNemar tests; per-group macro first-choice as a secondary reading.

    PYTHONPATH=. .venv/bin/python scripts/fresh_h2h_score.py PRODUCT.json LLM.json [--output F]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

NAMED = ("known_group_hypothesis", "known_cluster_hypothesis")
SEED = 7
RESAMPLES = 10_000


def product_rows(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for arm in ("known", "open_set"):
        for row in artifact.get(arm) or []:
            top5 = [entry["group_id"] for entry in (row.get("topk") or [])][:5]
            named = row.get("stage") in NAMED
            members = (
                list(row.get("named_cluster") or [])
                if row.get("stage") == "known_cluster_hypothesis"
                else [row.get("leading")]
            )
            out[row["report_id"]] = {
                "arm": arm,
                "truth": row.get("group_id"),
                "accept": set(row.get("accept") or ()),
                "first": row.get("leading"),
                "top3": list(row.get("top3") or [])[:3],
                "top5": top5,
                "named": named,
                "named_members": members if named else [],
            }
    return out


def llm_rows(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in artifact.get("rows") or []:
        if row.get("error") or row.get("decision") is None:
            continue
        candidates = [group for group in row.get("candidates") or [] if group]
        first = candidates[0] if candidates else None
        named = row["decision"] == "name"
        out[row["report_id"]] = {
            "arm": row["arm"],
            "truth": row.get("group_id"),
            "first": first,
            "top3": candidates[:3],
            "named": named,
            # a name the model gives on a candidate ATT&CK does not have is a name (and wrong)
            "named_members": [first] if named else [],
            "decision": row["decision"],
        }
    return out


def combine(product: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    first = llm["first"] if llm["first"] in product["top5"] else product["first"]
    named = product["first"] is not None and product["first"] == llm["first"] and llm["named"]
    top3 = list(dict.fromkeys([first, *product["top3"]]))[:3]
    return {
        "arm": product["arm"],
        "truth": product["truth"],
        "first": first,
        "top3": top3,
        "named": named,
        "named_members": [first] if named else [],
    }


def outcome(row: dict[str, Any], accept: set[str]) -> dict[str, Any]:
    truth = ({row["truth"]} | accept) - {None}
    named_right = bool(row["named"] and truth & set(row["named_members"]))
    return {
        "first": bool(truth and row["first"] in truth),
        "top3": bool(truth & set(row["top3"])),
        "named": bool(row["named"]),
        "named_right": named_right,
        "named_wrong": bool(row["named"] and not named_right),
    }


def mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_first": 0, "only_second": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_first": only_a, "only_second": only_b, "p": round(p, 4)}


def bootstrap_diff(a: list[bool], b: list[bool]) -> list[float]:
    rng = random.Random(SEED)
    n = len(a)
    diffs = []
    for _ in range(RESAMPLES):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in sample) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * RESAMPLES)], 4), round(diffs[int(0.975 * RESAMPLES)], 4)]


def score(product_artifact: dict[str, Any], llm_artifact: dict[str, Any]) -> dict[str, Any]:
    product = product_rows(product_artifact)
    llm = llm_rows(llm_artifact)
    ids = sorted(set(product) & set(llm))
    arms: dict[str, dict[str, dict[str, Any]]] = {"product": {}, "llm": {}, "combination": {}}
    for report_id in ids:
        accept = product[report_id]["accept"]
        arms["product"][report_id] = outcome(product[report_id], accept)
        arms["llm"][report_id] = outcome(llm[report_id], accept)
        arms["combination"][report_id] = outcome(
            combine(product[report_id], llm[report_id]), accept
        )
    known = [i for i in ids if product[i]["arm"] == "known"]
    open_set = [i for i in ids if product[i]["arm"] == "open_set"]
    summary: dict[str, Any] = {
        "reports_scored": len(ids),
        "known": len(known),
        "open_set": len(open_set),
        "missing_llm": sorted(set(product) - set(llm)),
        "groups": len({product[i]["truth"] for i in known}),
        "arms": {},
    }
    for name, rows in arms.items():
        k = [rows[i] for i in known]
        o = [rows[i] for i in open_set]
        by_group: dict[str, list[bool]] = defaultdict(list)
        for i in known:
            by_group[product[i]["truth"]].append(rows[i]["first"])
        summary["arms"][name] = {
            "first_right": sum(r["first"] for r in k),
            "first_rate": round(sum(r["first"] for r in k) / max(1, len(k)), 4),
            "top3_right": sum(r["top3"] for r in k),
            "macro_first_rate": round(
                sum(sum(v) / len(v) for v in by_group.values()) / max(1, len(by_group)), 4
            ),
            "names_known": sum(r["named"] for r in k),
            "names_right": sum(r["named_right"] for r in k),
            "names_wrong_known": sum(r["named_wrong"] for r in k),
            "names_open_set": sum(r["named"] for r in o),
            "names_wrong_total": sum(r["named_wrong"] for r in k) + sum(r["named"] for r in o),
        }
    pairs = {}
    for a, b in (("llm", "product"), ("combination", "product"), ("combination", "llm")):
        first_a = [arms[a][i]["first"] for i in known]
        first_b = [arms[b][i]["first"] for i in known]
        pairs[f"{a}_vs_{b}"] = {
            "first_diff": round((sum(first_a) - sum(first_b)) / max(1, len(known)), 4),
            "first_diff_ci95": bootstrap_diff(first_a, first_b) if known else None,
            "mcnemar_first": mcnemar(first_a, first_b),
        }
    summary["pairs"] = pairs
    summary["per_report"] = {
        i: {
            "arm": product[i]["arm"],
            "truth": product[i]["truth"],
            "product": {"first": product[i]["first"], "named": product[i]["named"]},
            "llm": {"first": llm[i]["first"], "decision": llm[i]["decision"]},
        }
        for i in ids
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("product", type=Path)
    parser.add_argument("llm", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = score(
        json.loads(args.product.read_text(encoding="utf-8")),
        json.loads(args.llm.read_text(encoding="utf-8")),
    )
    print(
        f"{result['reports_scored']} reports ({result['known']} known, {result['groups']} groups; "
        f"{result['open_set']} open set); missing LLM answers: {len(result['missing_llm'])}"
    )
    for name, arm in result["arms"].items():
        print(
            f"  {name:12s} first {arm['first_right']:3d} ({arm['first_rate']:.3f}) macro "
            f"{arm['macro_first_rate']:.3f} top3 {arm['top3_right']:3d} | names {arm['names_known']}"
            f" right {arm['names_right']} wrong {arm['names_wrong_known']} | open-set names "
            f"{arm['names_open_set']}"
        )
    for name, pair in result["pairs"].items():
        print(
            f"  {name:24s} first diff {pair['first_diff']:+.3f} CI {pair['first_diff_ci95']} "
            f"McNemar p {pair['mcnemar_first']['p']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
