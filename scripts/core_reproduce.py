"""Step 3 of the lean core: reproduce store text alone on fresh2 exactly.

Reads the reference ``data/knowledge_store/lead_rule/fresh2_store_text.json`` (``store_text_lead.py``)
and checks, per report, reading and arm (whole store, own publisher hidden): the same first group,
the same top ten and margins within 1e-9. Exit 0 only when everything matches.

    scripts/core_py scripts/core_reproduce.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vestige_core.util import DATA_DIR, require_memory  # noqa: E402

TOLERANCE = 1e-9


def main() -> int:
    require_memory()
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.knowledge_store.members import StoreMembers, members_path
    from apt_classifier.knowledge_store.pins import pin_for
    from apt_classifier.knowledge_store.sources import PUBLISHERS
    from vestige_core.rank import from_store_members
    from vestige_core.sets import load_set

    reference = json.loads(
        (DATA_DIR / "knowledge_store/lead_rule/fresh2_store_text.json").read_text()
    )
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    with gzip.open(members_path(settings), "rt", encoding="utf-8") as handle:
        store_names = tuple(json.load(handle).get("masking_names") or ())
    rows_by_key = {(r["report_id"], r["reading"]): r for r in reference["rows"]}
    pins = {pin_for(r["as_of"]) for r in reference["rows"]}
    rankers, members = {}, {}
    for pin in sorted(pins):
        require_memory()
        members[pin] = StoreMembers.load(members_path(settings, pin), kb)
        rankers[pin] = from_store_members(members[pin], kb)
        print(f"pin {pin}: vocabulary {rankers[pin].vocabulary_size}", file=sys.stderr)
    checked = mismatches = 0
    worst = 0.0
    for reading in reference["readings"]:
        extra = store_names if reading == "behavior" else ()
        reports = load_set(DATA_DIR / "benchmarks/fresh2", kb, reading, extra_names=extra)
        for report in reports:
            ref = rows_by_key.get((report.report_id, reading))
            if ref is None:
                print(f"{report.report_id} {reading}: not in the reference", file=sys.stderr)
                mismatches += 1
                continue
            if len(report.statements) != ref["statements"]:
                print(
                    f"{report.report_id} {reading}: {len(report.statements)} statements, "
                    f"reference {ref['statements']}",
                    file=sys.stderr,
                )
                mismatches += 1
            pin = pin_for(report.as_of)
            m, ranker = members[pin], rankers[pin]
            hidden = m.hidden_for(report.store_keys, report.raw_statements)
            publisher = PUBLISHERS.get(report.source) or report.source
            theirs = {
                k
                for k, p in m.posts.items()
                if (PUBLISHERS.get(str(p["source"])) or str(p["source"])) == publisher
            }
            text = " ".join(report.statements)
            for arm, hide in (("whole_store", hidden), ("publisher_hidden", hidden | theirs)):
                got = ranker.rank(text, ranker.weights(report.as_of, hide))
                want = ref[arm]
                checked += 1
                top = [[a, s] for a, s in got.order[:10]]
                same_ids = [a for a, _ in top] == [a for a, _ in want["top"]]
                diff = max(
                    (abs(a[1] - b[1]) for a, b in zip(top, want["top"], strict=False)), default=0.0
                )
                if want["margin"] is not None and got.margin is not None:
                    diff = max(diff, abs(got.margin - want["margin"]))
                worst = max(worst, diff)
                if got.first != want["first"] or not same_ids or diff > TOLERANCE:
                    mismatches += 1
                    print(
                        f"MISMATCH {report.report_id} {reading} {arm}: first {got.first} vs "
                        f"{want['first']}, diff {diff:.3g}",
                        file=sys.stderr,
                    )
    print(json.dumps({"checked": checked, "mismatches": mismatches, "max_abs_diff": worst}))
    return 0 if mismatches == 0 and checked == len(reference["rows"]) * 2 else 1


if __name__ == "__main__":
    sys.exit(main())
