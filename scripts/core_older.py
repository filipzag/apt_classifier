"""Lean core on older dated report sets (development, descriptive): vendor and Elastic.

Same reading as ``scripts/core_race.py`` (store posts strictly before the report's date, own post
and copies hidden, the pin of its date, no prior, store masked at the report's level), for sets
whose files ``core_race.py`` does not know. ``--hide-publisher`` hides every store post from the
report's own web host. Scored against the product's saved runs on the current defaults (ATT&CK
20.0, procedure core). The frozen test code is imported, never changed.

    scripts/core_watch.sh scripts/core_py scripts/core_older.py rank --set vendor --level strict
    scripts/core_py scripts/core_older.py score data/core/older_vendor_strict.json PRODUCT.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vestige_core.util import DATA_DIR, require_memory  # noqa: E402

BENCH = DATA_DIR / "benchmarks"
FILES = {
    "vendor": [BENCH / "vendor" / "reports.jsonl"],
    "elastic": [BENCH / "elastic" / "reports.jsonl", BENCH / "elastic" / "reports_open_set.jsonl"],
}


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def rank(args: argparse.Namespace) -> int:
    require_memory()
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.knowledge_store.members import StoreMembers, members_path
    from apt_classifier.knowledge_store.pins import pin_for
    from vestige_core.corpus import load_rows
    from vestige_core.rank import TextRanker
    from vestige_core.sets import load_reports

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    with gzip.open(members_path(settings), "rt", encoding="utf-8") as handle:
        names = tuple(json.load(handle).get("masking_names") or ())
    extra = names if args.level == "behavior" else ()
    reports, urls = [], {}
    for path in FILES[args.set]:
        reports += load_reports(path, kb, args.level, extra_names=extra)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                urls[str(row["report_id"])] = str(row.get("url") or "")
    db = sqlite3.connect(f"file:{DATA_DIR / 'knowledge_store' / 'store.sqlite'}?mode=ro", uri=True)
    post_host = {k: _host(u or "") for k, u in db.execute("select key, url from posts")}
    groups = tuple(sorted(kb.groups))
    out, rankers = [], {}
    start = time.time()
    for report in sorted(reports, key=lambda r: pin_for(r.as_of)):
        pin = pin_for(report.as_of)
        if pin not in rankers:
            rankers.clear()
            require_memory()
            members = StoreMembers.load(members_path(settings, pin), kb)
            rows = load_rows(members, kb, pin, args.level, extra)
            rankers[pin] = (members, TextRanker(rows, members.posts, groups, None))
        members, ranker = rankers[pin]
        hidden = set(members.hidden_for(report.store_keys, report.raw_statements))
        if args.hide_publisher:
            host = _host(urls.get(report.report_id, ""))
            hidden |= {k for k in members.posts if host and post_host.get(k) == host}
        ranking = ranker.rank(" ".join(report.statements), ranker.weights(report.as_of, hidden))
        truth = report.group_id
        out.append(
            {
                "report_id": report.report_id,
                "group_id": truth,
                "accept": sorted(report.accept),
                "published": report.published,
                "pin": pin,
                "first": ranking.first,
                "margin": ranking.margin,
                "truth_rank": ranking.rank_of(truth) if truth else None,
                "top5": [[a, round(s, 6)] for a, s in ranking.order[:5]],
            }
        )
    name = f"older_{args.set}_{args.level}{'_pubhidden' if args.hide_publisher else ''}"
    path = DATA_DIR / "core" / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "set": args.set,
                "level": args.level,
                "hide_publisher": args.hide_publisher,
                "seconds": round(time.time() - start),
                "rows": out,
            },
            indent=1,
        )
    )
    known = [r for r in out if r["group_id"]]
    first = sum(r["first"] == r["group_id"] or r["first"] in r["accept"] for r in known)
    print(json.dumps({"file": str(path), "known": len(known), "first": first}))
    return 0


def _product(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    out = {}
    for row in (data.get("known") or []) + (data.get("open_set") or []):
        named = row.get("leading") if row.get("stage") == "known_group_hypothesis" else None
        out[str(row["report_id"])] = {
            "first": row.get("leading"),
            "named": named,
            "cluster": list(row.get("named_cluster") or []),
        }
    return out


def score(args: argparse.Namespace) -> int:
    from vestige_core.evaluate import score as score_rows

    data = json.loads(Path(args.file).read_text())
    result = score_rows(data["rows"], float("inf"), _product(args.product), None)
    # margin inf: the core names nothing without the model, so only first choices compare here
    print(json.dumps({"file": args.file, "product": args.product, **result}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--set", choices=sorted(FILES), required=True)
    r.add_argument("--level", choices=["names", "strict", "tradecraft", "behavior"], required=True)
    r.add_argument("--hide-publisher", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("file")
    s.add_argument("product")
    args = parser.parse_args()
    return rank(args) if args.command == "rank" else score(args)


if __name__ == "__main__":
    sys.exit(main())
