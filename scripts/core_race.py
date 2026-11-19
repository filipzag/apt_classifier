"""Lean core, development race: rank every report of a set, then score it.

    scripts/core_watch.sh scripts/core_py scripts/core_race.py rank --set fresh --level behavior \
        --corpus matched [--hide-publisher]
    scripts/core_py scripts/core_race.py score data/core/dev_fresh_behavior_matched.json --margin M

``--corpus store`` reads the store's own masking (store text exactly); ``matched`` masks the store
posts at the report's level (``vestige_core/corpus.py``). Rows go to ``data/core/dev_*.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vestige_core.util import DATA_DIR, require_memory  # noqa: E402

SETS = ("store_train", "fresh", "fresh2")


def rank(args: argparse.Namespace) -> int:
    require_memory()
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.knowledge_store.members import StoreMembers, members_path
    from apt_classifier.knowledge_store.pins import pin_for
    from apt_classifier.knowledge_store.sources import PUBLISHERS
    from apt_classifier.text_group import prominence_log_prior
    from vestige_core.corpus import load_rows
    from vestige_core.rank import TextRanker
    from vestige_core.sets import load_set

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    with gzip.open(members_path(settings), "rt", encoding="utf-8") as handle:
        names = tuple(json.load(handle).get("masking_names") or ())
    extra = names if args.level == "behavior" else ()
    reports = load_set(DATA_DIR / "benchmarks" / args.set, kb, args.level, extra_names=extra)
    groups = tuple(sorted(kb.groups))
    prior = None if args.no_prior else prominence_log_prior(kb, groups)
    out, rankers = [], {}
    start = time.time()
    for report in sorted(reports, key=lambda r: pin_for(r.as_of)):
        pin = pin_for(report.as_of)
        if pin not in rankers:
            rankers.clear()
            require_memory()
            members = StoreMembers.load(members_path(settings, pin), kb)
            level = "store" if args.corpus == "store" else args.level
            rows = load_rows(members, kb, pin, level, extra)
            rankers[pin] = (members, TextRanker(rows, members.posts, groups, prior))
        members, ranker = rankers[pin]
        hidden = set(members.hidden_for(report.store_keys, report.raw_statements))
        if args.hide_publisher:
            publisher = PUBLISHERS.get(report.source) or report.source
            hidden |= {
                k
                for k, p in members.posts.items()
                if (PUBLISHERS.get(str(p["source"])) or str(p["source"])) == publisher
            }
        ranking = ranker.rank(" ".join(report.statements), ranker.weights(report.as_of, hidden))
        truth = report.group_id
        out.append(
            {
                "report_id": report.report_id,
                "group_id": truth,
                "actor": report.actor,
                "accept": sorted(report.accept),
                "published": report.published,
                "pin": pin,
                "first": ranking.first,
                "margin": ranking.margin,
                "truth_rank": ranking.rank_of(truth) if truth else None,
                "top5": [[a, round(s, 6)] for a, s in ranking.order[:5]],
            }
        )
    name = f"dev_{args.set}_{args.level}_{args.corpus}{'_pubhidden' if args.hide_publisher else ''}"
    if args.no_prior:
        name += "_noprior"
    path = DATA_DIR / "core" / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "set": args.set,
                "level": args.level,
                "corpus": args.corpus,
                "hide_publisher": args.hide_publisher,
                "prior": not args.no_prior,
                "seconds": round(time.time() - start),
                "rows": out,
            },
            indent=1,
        )
    )
    known = [r for r in out if r["group_id"]]
    first = sum(r["first"] == r["group_id"] or r["first"] in r["accept"] for r in known)
    print(
        json.dumps(
            {
                "file": str(path),
                "known": len(known),
                "first": first,
                "open_set": len(out) - len(known),
            }
        )
    )
    return 0


def score(args: argparse.Namespace) -> int:
    from vestige_core.evaluate import model_rows, product_rows, smallest_margin
    from vestige_core.evaluate import score as score_rows

    data = json.loads(Path(args.file).read_text())
    rows = data["rows"]
    margin = args.margin
    if margin is None:
        margin = smallest_margin(rows, args.max_wrong_rate)
    result = score_rows(
        rows,
        margin if margin is not None else float("inf"),
        product_rows(data["set"], data["level"]),
        model_rows(data["set"], data["level"]),
    )
    print(json.dumps({"file": args.file, **result}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--set", choices=SETS, required=True)
    r.add_argument("--level", choices=["strict", "behavior"], required=True)
    r.add_argument("--corpus", choices=["store", "matched"], required=True)
    r.add_argument("--hide-publisher", action="store_true")
    r.add_argument("--no-prior", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("file")
    s.add_argument("--margin", type=float)
    s.add_argument("--max-wrong-rate", type=float, default=0.02)
    args = parser.parse_args()
    return rank(args) if args.command == "rank" else score(args)


if __name__ == "__main__":
    sys.exit(main())
