"""Parent-relabelled copies of the evaluation case files (standard library only; prereg
``cribl_recipe_base_2026-10-15.json``).

Every gold id becomes its parent technique (``T1059.001`` -> ``T1059``), de-duplicated in first-seen
order; text and candidates are untouched. On such a file an exact answer is right whenever its family
is, so the annotators' habit of writing the parent or a sub-technique no longer counts against the
model (on CTI-HAL 111 of E2's 131 right-family exact misses). Training files are not relabelled: the
sub-technique targets stay, and the technique head with them.

  python relabel_parents.py [--splits dev test ...]
  -> data/audits/cribl_ft/cases_<split>_parents.jsonl for every existing split (family_head.py
     evaluate --split <split>_parents); counts in cases_manifest.json["parents"]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from common import parent_of, read_jsonl, write_jsonl

AUDITS = Path(__file__).resolve().parents[2] / "data" / "audits" / "cribl_ft"
SPLITS = (
    "dev",
    "annoctr_dev",
    "test",
    "annoctr_test",
    "cti_hal",
    "cti_rationale_test",
    "cti_rationale_validation",
)


def parent_labels(gold_ids: Iterable[str]) -> list[str]:
    """The gold ids' parents, each once, in first-seen order."""
    return list(dict.fromkeys(parent_of(str(item)) for item in gold_ids))


def relabel(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rows with ``gold_ids`` replaced by their parents, and the counts of the change."""
    out, counts = [], {"cases": 0, "labels_before": 0, "labels_after": 0, "cases_changed": 0}
    for row in rows:
        parents = parent_labels(row["gold_ids"])
        counts["cases"] += 1
        counts["labels_before"] += len(row["gold_ids"])
        counts["labels_after"] += len(parents)
        counts["cases_changed"] += list(row["gold_ids"]) != parents
        out.append({**row, "gold_ids": parents})
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--cases-dir", type=Path, default=AUDITS)
    args = parser.parse_args()
    manifest_path = args.cases_dir / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    for split in args.splits:
        source = args.cases_dir / f"cases_{split}.jsonl"
        if not source.exists():
            print(f"{source.name}: missing, skipped")
            continue
        rows, counts = relabel(read_jsonl(source))
        target = args.cases_dir / f"cases_{split}_parents.jsonl"
        write_jsonl(target, rows)
        manifest.setdefault("parents", {})[split] = counts
        print(f"{target.name}: {counts}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
