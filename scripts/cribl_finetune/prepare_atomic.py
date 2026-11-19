"""Atomic Red Team tests as family-head training rows (project .venv; README.md here; prereg
``cribl_family_data_2026-10-14.json``).

Reads ``atomics/Indexes/index.yaml`` of redcanaryco/atomic-red-team (MIT licence; downloaded once to
``data/benchmarks/atomic/raw/``). Each atomic test (deduplicated by its GUID; the index lists a
technique under every tactic) becomes one row, "<name>. <description>", labelled with the test's
technique; commands, input arguments and ``#{...}`` placeholders are left out and technique ids are
removed from the text. Rows pass the cleaning rules (``clean_cases.broken_reason``) and the held-out
overlap filter of the ATT&CK procedure export.

  python prepare_atomic.py  -> data/audits/cribl_ft/cases_atomic_train.jsonl + manifest counts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

INDEX = REPO_ROOT / "data" / "benchmarks" / "atomic" / "raw" / "index.yaml"
MAX_CHARS = 1000
_PLACEHOLDER = re.compile(r"#\{[^}]*\}")
_TECHNIQUE_ID = re.compile(r"\bT1\d{3}(?:\.\d{3})?\w*")
_MARKUP = re.compile(r"[`*_]+|\[([^\]]*)\]\([^)]*\)")


def atomic_text(test: Mapping[str, Any]) -> str:
    """ "<name>. <description>" on one line, without placeholders, technique ids or markdown."""
    name = str(test.get("name") or "").strip().rstrip(".")
    description = str(test.get("description") or "").strip()
    text = f"{name}. {description}" if description else f"{name}."
    text = _PLACEHOLDER.sub("", text)
    text = _MARKUP.sub(lambda match: match.group(1) or "", text)
    text = _TECHNIQUE_ID.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS].rsplit(" ", 1)[0] if len(text) > MAX_CHARS else text


def atomic_tests(index: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """(technique id, test GUID, test) once per test, in index order."""
    found, seen = [], set()
    for techniques in index.values():
        for technique, entry in (techniques or {}).items():
            for test in (entry or {}).get("atomic_tests") or ():
                guid = str(test.get("auto_generated_guid") or f"{technique}:{test.get('name')}")
                if guid not in seen:
                    seen.add(guid)
                    found.append((str(technique), guid, test))
    return found


def main() -> None:
    import yaml
    from clean_cases import broken_reason
    from common import read_jsonl, write_jsonl
    from export_attack_procedures import HELD_OUT, drop_overlaps
    from export_cases import OUT, drop_near_duplicates
    from prepare_advisories import resolve_ids

    from apt_classifier.config import get_settings
    from scripts.prepare_elastic_reports import revoked_technique_map

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    settings = get_settings()
    catalog = json.loads((OUT / "catalog.json").read_text(encoding="utf-8"))["techniques"]
    families = {item["family"] for item in catalog}
    revoked = revoked_technique_map(settings.attack_bundle_path)
    index = yaml.safe_load(args.index.read_text(encoding="utf-8"))
    tests = atomic_tests(index)
    rows, dropped = [], Counter()
    for technique, guid, test in tests:
        gold = resolve_ids([technique], revoked, families)
        text = atomic_text(test)
        if not gold:
            dropped["no_listed_family"] += 1
        elif reason := broken_reason(text):
            dropped[reason] += 1
        else:
            rows.append({"text": text, "gold_ids": gold, "guid": guid})
    held_out = [
        row["text"] for split in HELD_OUT for row in read_jsonl(OUT / f"cases_{split}.jsonl")
    ]
    rows, near = drop_near_duplicates(rows, held_out)
    rows, overlaps = drop_overlaps(rows, held_out)
    out_rows = [
        {
            "case_id": f"atomic-{position:06d}",
            "dataset": "atomic",
            "split": "atomic_train",
            "text": row["text"],
            "gold_ids": row["gold_ids"],
            "candidates": [],
            "gold_in_candidates": False,
            "gold_family_in_candidates": False,
            "pipeline_rank1": None,
            "jev_c1": None,
            "source": f"atomic-red-team {row['guid']}",
        }
        for position, row in enumerate(rows, start=1)
    ]
    path = OUT / "cases_atomic_train.jsonl"
    write_jsonl(path, out_rows)
    info = {
        "source": "redcanaryco/atomic-red-team atomics/Indexes/index.yaml (MIT)",
        "tests": len(tests),
        "dropped": {**dict(sorted(dropped.items())), **near, **overlaps},
        "cases": len(out_rows),
        "families": len({item.split(".")[0] for row in out_rows for item in row["gold_ids"]}),
    }
    manifest_path = OUT / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("splits", {})["atomic_train"] = info
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{path}: {json.dumps(info)}")


if __name__ == "__main__":
    main()
