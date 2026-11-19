"""Development race: the core's ranking per report, scored against the saved product and model runs.

Every report reads only store posts available strictly before its date, never its own post or its
copies (``StoreMembers.hidden_for``), with the store pin of its date (older name lists for older
reports). Optionally the report's own publisher is hidden too.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vestige_core.decide import Proposal, decide

SAVED = Path("/home/fzagar/apt_classifier/data/benchmarks")


def right(actor: str | None, truth: str | None, accept: Iterable[str] = ()) -> bool:
    return actor is not None and truth is not None and (actor == truth or actor in set(accept))


# --------------------------------------------------------------------------- saved baselines
def product_rows(set_name: str, level: str) -> dict[str, dict[str, Any]] | None:
    path = SAVED / set_name / "saved_runs" / f"product_{level}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    out = {}
    for row in (data.get("known") or []) + (data.get("open_set") or []):
        named = None
        if row.get("stage") == "known_group_hypothesis":
            named = row.get("leading")
        cluster = list(row.get("named_cluster") or [])
        out[str(row["report_id"])] = {
            "first": row.get("leading"),
            "named": named,
            "cluster": cluster,
        }
    return out


def model_rows(set_name: str, level: str) -> dict[str, Proposal] | None:
    path = SAVED / set_name / "saved_runs" / f"llm_gpt56sol_{level}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    out = {}
    for row in data["rows"]:
        first = next((c for c in row.get("candidates") or [] if c), None)  # registered id matching
        out[str(row["report_id"])] = Proposal(first=first, decision=str(row.get("decision")))
    return out


# --------------------------------------------------------------------------- scoring
def score(
    rows: Sequence[Mapping[str, Any]],
    margin: float,
    product: Mapping[str, Mapping[str, Any]] | None = None,
    model: Mapping[str, Proposal] | None = None,
) -> dict[str, Any]:
    """Counts on known and open-set reports for the core, the core with the model, and the saved
    product and model runs on the same reports (rows missing from a baseline are left out of
    every arm, so the arms read the same reports)."""
    if product is not None:
        rows = [r for r in rows if r["report_id"] in product]
    if model is not None:
        rows = [r for r in rows if r["report_id"] in model]
    known = [r for r in rows if r["group_id"]]
    open_ = [r for r in rows if not r["group_id"]]

    def arm(first_of, named_of) -> dict[str, Any]:
        k_first = sum(right(first_of(r), r["group_id"], r["accept"]) for r in known)
        k_named = [r for r in known if named_of(r)]
        k_wrong = sum(not _named_right(named_of(r), r) for r in k_named)
        o_named = sum(bool(named_of(r)) for r in open_)
        return {
            "first": k_first,
            "named": len(k_named),
            "named_wrong": k_wrong,
            "open_named": o_named,
            "wrong_total": k_wrong + o_named,
        }

    def core_named(r):
        return decide(r["first"], r["margin"], margin)

    out: dict[str, Any] = {"known": len(known), "open_set": len(open_), "margin": margin}
    out["core"] = arm(lambda r: r["first"], core_named)
    if model is not None:
        out["core_with_model"] = arm(
            lambda r: r["first"],
            lambda r: decide(r["first"], r["margin"], margin, model[r["report_id"]]),
        )
        out["model"] = arm(
            lambda r: model[r["report_id"]].first,
            lambda r: model[r["report_id"]].first
            if model[r["report_id"]].decision == "name"
            else None,
        )
    if product is not None:
        out["product"] = arm(
            lambda r: product[r["report_id"]]["first"],
            lambda r: product[r["report_id"]]["named"]
            or (product[r["report_id"]]["cluster"] or None),
        )
        out["first_paired_vs_product"] = paired(
            known,
            lambda r: right(r["first"], r["group_id"], r["accept"]),
            lambda r: right(product[r["report_id"]]["first"], r["group_id"], r["accept"]),
        )
    if model is not None:
        out["first_paired_vs_model"] = paired(
            known,
            lambda r: right(r["first"], r["group_id"], r["accept"]),
            lambda r: right(model[r["report_id"]].first, r["group_id"], r["accept"]),
        )
    return out


def _named_right(named: Any, r: Mapping[str, Any]) -> bool:
    if isinstance(named, list):  # a product cluster: right when the truth is one of its members
        return any(right(n, r["group_id"], r["accept"]) for n in named)
    return right(named, r["group_id"], r["accept"])


def paired(rows: Sequence[Any], a, b, samples: int = 1000, seed: int = 7) -> dict[str, Any]:
    """Gains and losses of ``a`` over ``b``, an exact sign test and a bootstrap 95% interval of the
    difference in counts."""
    from math import comb

    wins = sum(bool(a(r)) and not b(r) for r in rows)
    losses = sum(bool(b(r)) and not a(r) for r in rows)
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, k) for k in range(max(wins, losses), n + 1)) / 2**n) if n else 1.0
    diffs = [int(bool(a(r))) - int(bool(b(r))) for r in rows]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) for _ in range(samples)) if diffs else [0]
    return {
        "gained": wins,
        "lost": losses,
        "sign_p": round(p, 4),
        "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
    }


def smallest_margin(rows: Sequence[Mapping[str, Any]], max_wrong_rate: float) -> float | None:
    """The smallest margin at which wrong names (known reports named wrongly plus open-set reports
    named at all) stay at or below ``max_wrong_rate`` of all reports. Wrong names only fall as the
    margin rises, so the candidates are the margins themselves."""
    margins = sorted({r["margin"] for r in rows if r["margin"] is not None})
    for m in [0.0, *margins]:
        wrong = sum(
            1
            for r in rows
            if (named := decide(r["first"], r["margin"], m)) is not None
            and not right(named, r["group_id"], r["accept"])
        )
        if wrong <= max_wrong_rate * len(rows):
            return m
    return None
