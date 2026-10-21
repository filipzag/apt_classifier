"""Recompute a development report set's ``hidden_texts`` on the current ATT&CK bundle.

The vendor and campaign sets were prepared on ATT&CK 19.2 (``prepare_vendor_reports.py``,
``prepare_campaign_reports.py`` on the jev-framing branch); an ATT&CK update can reword or add the
procedure examples that cite or copy a report. Each row keeps its old ``hidden_texts`` and gains
every procedure example of the current bundle that cites its ``source_name`` (vendor rows) or
shares a 10-token run with one of its statements (the rules of ``prepare_advisory_reports.py``).
The output goes next to the input as ``<stem>_attack<version>.jsonl``; the input is not changed.

    PYTHONPATH=. .venv/bin/python scripts/refresh_hidden_texts.py \\
        data/benchmarks/vendor/reports_hide_cited.jsonl data/benchmarks/campaigns/reports.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from scripts.prepare_advisory_reports import (  # noqa: E402
    hidden_procedures,
    procedure_examples,
    runs,
    tokens,
)


def refresh(path: Path, bundle: dict, version: str) -> Path:
    examples = procedure_examples(bundle)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    added = 0
    for row in rows:
        statement_runs = set().union(
            set(),
            *(runs(tokens(str(item.get("text") or ""))) for item in row.get("statements", [])),
        )
        names = [row["source_name"]] if row.get("source_name") else []
        found, _, _ = hidden_procedures(examples, names, statement_runs)
        before = set(row.get("hidden_texts") or ())
        row["hidden_texts"] = sorted(before | set(found))
        added += len(row["hidden_texts"]) - len(before)
    output = path.with_name(f"{path.stem}_attack{version}.jsonl")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"{path.name}: {len(rows)} reports, {added} hidden texts added -> {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument(
        "--bundle", type=Path, default=REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
    )
    args = parser.parse_args(argv)
    kb = AttackKnowledgeBase.load(Settings().kb_path)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    for path in args.paths:
        refresh(path, bundle, str(kb.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
