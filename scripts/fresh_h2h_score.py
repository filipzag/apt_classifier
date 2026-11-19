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

The confidence cut-off test (prereg ``llm_confidence_cutoff_2026-11-02.json``) adds:

* ``--matching name``: the LLM's candidates are matched by their name (ATT&CK group names and
  aliases, case and punctuation ignored); a name ATT&CK does not list is not an ATT&CK group, so
  naming it gives no ATT&CK name (counted as ``names_outside_attack``). The registered matching
  (``id``) takes the id first and the name second;
* ``--llm-min-probability P``: arm ``llm_confident`` names only when the model decides ``name`` and
  states a probability of at least P for its first candidate; arm ``agreed_or_confident`` names
  the model's first candidate when the model decides ``name`` and either states at least P or the
  product's leader is the same group (the registered combination's name). Exact (Clopper-Pearson)
  95% intervals for each arm's share of right names;
* ``--switch-probability S`` (with ``--llm-min-probability``): arm ``switch``, the model's answer
  where it is sure and the product's elsewhere: first choice = the model's first candidate when
  it states at least S for it, else the product's leader; names = ``llm_confident``'s.

    PYTHONPATH=. .venv/bin/python scripts/fresh_h2h_score.py PRODUCT.json LLM.json [--output F]
        [--matching name --llm-min-probability 0.99 --switch-probability 0.5]
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
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB = REPO_ROOT / "data" / "kb" / "attack_kb.json"
# a stated probability written as 0.99 is at least 0.99
PROBABILITY_TOLERANCE = 1e-9


def name_key(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def group_names(groups: dict[str, Any]) -> dict[str, str]:
    """Normalised ATT&CK group names and aliases -> group id (names of two groups are left out)."""
    index: dict[str, set[str]] = defaultdict(set)
    for group_id, group in groups.items():
        for name in (group.name, *group.aliases):
            if name and name_key(name):
                index[name_key(name)].add(group_id)
    return {key: next(iter(ids)) for key, ids in index.items() if len(ids) == 1}


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


def _probability(candidate: dict[str, Any] | None) -> float | None:
    value = (candidate or {}).get("probability")
    return float(value) if isinstance(value, int | float) else None


def llm_rows(
    artifact: dict[str, Any], names: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """The LLM's answers; ``names`` (``group_names``) matches each candidate by its name."""
    out = {}
    for row in artifact.get("rows") or []:
        if row.get("error") or row.get("decision") is None:
            continue
        raw = list(row.get("raw_candidates") or [])
        decision = row["decision"] == "name"
        outside = False
        if names is None:
            mapped = list(row.get("candidates") or [])
            candidates = [group for group in mapped if group]
            first = candidates[0] if candidates else None
            named = decision
            # a name the model gives on a candidate ATT&CK does not have is a name (and wrong)
            members = [first] if named else []
            position = next((i for i, group in enumerate(mapped) if group), None)
            probability = _probability(raw[position]) if position is not None and raw else None
        else:
            mapped = [names.get(name_key(str(c.get("group_name") or ""))) for c in raw]
            candidates = [group for group in mapped if group]
            first = candidates[0] if candidates else None
            named = decision and bool(mapped) and mapped[0] is not None
            outside = decision and bool(mapped) and mapped[0] is None
            members = [mapped[0]] if named else []
            probability = _probability(raw[0]) if raw else None
        out[row["report_id"]] = {
            "arm": row["arm"],
            "truth": row.get("group_id"),
            "first": first,
            "top3": candidates[:3],
            "named": named,
            "named_members": members,
            "outside": outside,
            "probability": probability,
            "decision": row["decision"],
        }
    return out


def confident(llm: dict[str, Any], min_probability: float) -> dict[str, Any]:
    """The LLM's answer, named only at a stated probability of at least ``min_probability``."""
    sure = llm["probability"] is not None and (
        llm["probability"] >= min_probability - PROBABILITY_TOLERANCE
    )
    named = llm["named"] and sure
    return {
        **llm,
        "named": named,
        "named_members": llm["named_members"] if named else [],
        "outside": llm["outside"] and sure,
    }


def agreed_or_confident(
    product: dict[str, Any], llm: dict[str, Any], min_probability: float
) -> dict[str, Any]:
    """The registered combination, plus a name wherever the LLM is confident on its own."""
    combined = combine(product, llm)
    sure = confident(llm, min_probability)
    named = combined["named"] or sure["named"]
    if not named:
        return combined
    group = llm["named_members"][0]
    return {
        **combined,
        "first": group,
        "top3": list(dict.fromkeys([group, *combined["top3"]]))[:3],
        "named": True,
        "named_members": [group],
    }


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


def switch(
    product: dict[str, Any], llm: dict[str, Any], min_probability: float, switch_probability: float
) -> dict[str, Any]:
    """The model's first candidate where it states at least ``switch_probability``, else the
    product's leader; names only where the model is confident (``confident``)."""
    sure = llm["first"] is not None and (
        llm["probability"] is not None
        and llm["probability"] >= switch_probability - PROBABILITY_TOLERANCE
    )
    first = llm["first"] if sure else product["first"]
    named = confident(llm, min_probability)
    return {
        "arm": product["arm"],
        "truth": product["truth"],
        "first": first,
        "top3": list(dict.fromkeys([first, *(llm["top3"] if sure else product["top3"])]))[:3],
        "named": named["named"],
        "named_members": named["named_members"],
        "outside": named["outside"],
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
        "outside": bool(row.get("outside")),
    }


def _binomial(trials: int, outcomes: range, p: float) -> float:
    return sum(math.comb(trials, i) * p**i * (1 - p) ** (trials - i) for i in outcomes)


def _bisect(function, *, rising: bool, target: float = 0.025) -> float:
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if (function(middle) < target) == rising:
            low = middle
        else:
            high = middle
    return round((low + high) / 2, 4)


def exact_interval(successes: int, trials: int) -> list[float] | None:
    """Clopper-Pearson 95% interval for a share."""
    if trials == 0:
        return None
    lower = (
        0.0
        if successes == 0
        else _bisect(lambda p: _binomial(trials, range(successes, trials + 1), p), rising=True)
    )
    upper = (
        1.0
        if successes == trials
        else _bisect(lambda p: _binomial(trials, range(successes + 1), p), rising=False)
    )
    return [lower, upper]


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


def score(
    product_artifact: dict[str, Any],
    llm_artifact: dict[str, Any],
    *,
    names: dict[str, str] | None = None,
    min_probability: float | None = None,
    switch_probability: float | None = None,
) -> dict[str, Any]:
    product = product_rows(product_artifact)
    llm = llm_rows(llm_artifact, names)
    ids = sorted(set(product) & set(llm))
    arms: dict[str, dict[str, dict[str, Any]]] = {"product": {}, "llm": {}, "combination": {}}
    if min_probability is not None:
        arms |= {"llm_confident": {}, "agreed_or_confident": {}}
        if switch_probability is not None:
            arms["switch"] = {}
    for report_id in ids:
        accept = product[report_id]["accept"]
        arms["product"][report_id] = outcome(product[report_id], accept)
        arms["llm"][report_id] = outcome(llm[report_id], accept)
        arms["combination"][report_id] = outcome(
            combine(product[report_id], llm[report_id]), accept
        )
        if min_probability is not None:
            arms["llm_confident"][report_id] = outcome(
                confident(llm[report_id], min_probability), accept
            )
            arms["agreed_or_confident"][report_id] = outcome(
                agreed_or_confident(product[report_id], llm[report_id], min_probability), accept
            )
            if switch_probability is not None:
                arms["switch"][report_id] = outcome(
                    switch(product[report_id], llm[report_id], min_probability, switch_probability),
                    accept,
                )
    known = [i for i in ids if product[i]["arm"] == "known"]
    open_set = [i for i in ids if product[i]["arm"] == "open_set"]
    summary: dict[str, Any] = {
        "reports_scored": len(ids),
        "known": len(known),
        "open_set": len(open_set),
        "missing_llm": sorted(set(product) - set(llm)),
        "groups": len({product[i]["truth"] for i in known}),
        "matching": "id" if names is None else "name",
        "llm_min_probability": min_probability,
        "switch_probability": switch_probability,
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
        given = sum(r["named"] for r in k) + sum(r["named"] for r in o)
        right = sum(r["named_right"] for r in k)
        summary["arms"][name] |= {
            "names_given": given,
            "names_right_share": round(right / given, 4) if given else None,
            "names_right_share_ci95": exact_interval(right, given),
            "names_outside_attack_known": sum(r["outside"] for r in k),
            "names_outside_attack_open_set": sum(r["outside"] for r in o),
        }
    pairs = {}
    compared = [("llm", "product"), ("combination", "product"), ("combination", "llm")]
    if "switch" in arms:
        compared += [("switch", "llm"), ("switch", "product")]
    for a, b in compared:
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
            "llm": {
                "first": llm[i]["first"],
                "decision": llm[i]["decision"],
                "named": llm[i]["named_members"][:1],
                "outside": llm[i]["outside"],
                "probability": llm[i]["probability"],
            },
        }
        for i in ids
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("product", type=Path)
    parser.add_argument("llm", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--matching",
        choices=("id", "name"),
        default="id",
        help="match the LLM's candidates by id first (registered 2026-10-28) or by name",
    )
    parser.add_argument("--llm-min-probability", type=float, help="adds the cut-off arms")
    parser.add_argument(
        "--switch-probability", type=float, help="adds the switch arm (needs the cut-off)"
    )
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    args = parser.parse_args(argv)
    names = None
    if args.matching == "name":
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from apt_classifier.attack.kb import AttackKnowledgeBase

        names = group_names(AttackKnowledgeBase.load(args.kb).groups)
    result = score(
        json.loads(args.product.read_text(encoding="utf-8")),
        json.loads(args.llm.read_text(encoding="utf-8")),
        names=names,
        min_probability=args.llm_min_probability,
        switch_probability=args.switch_probability if args.llm_min_probability else None,
    )
    print(
        f"{result['reports_scored']} reports ({result['known']} known, {result['groups']} groups; "
        f"{result['open_set']} open set); missing LLM answers: {len(result['missing_llm'])}"
    )
    for name, arm in result["arms"].items():
        print(
            f"  {name:20s} first {arm['first_right']:3d} ({arm['first_rate']:.3f}) macro "
            f"{arm['macro_first_rate']:.3f} top3 {arm['top3_right']:3d} | names {arm['names_known']}"
            f" right {arm['names_right']} wrong {arm['names_wrong_known']} | open-set names "
            f"{arm['names_open_set']} | right share {arm['names_right_share']} "
            f"{arm['names_right_share_ci95']} | outside ATT&CK "
            f"{arm['names_outside_attack_known']}/{arm['names_outside_attack_open_set']}"
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
