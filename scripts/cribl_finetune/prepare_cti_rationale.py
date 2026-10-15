"""CTI-Rationale report paragraphs as family-head training rows and a multi-label test set (project
.venv; README.md here; prereg ``cribl_cti_rationale_2026-10-15.json``).

CTI-Rationale (Hugging Face ``EbruResul/CTI-Rationale``, gated; annotations CC BY 4.0, paragraph text
the publishers') labels 1,394 paragraphs of 100 public reports from 8 publishers with ATT&CK v18.1
techniques: two analysts each, adjudicated gold rows. A paragraph's labels are the techniques of its
gold rows. The dataset's report split is kept: train reports become training rows, validation
reports a checkpoint-selection set (``cases_cti_rationale_validation.jsonl``; with
``--merge-validation`` they join the training rows instead, E3's data) and test reports a held-out
paragraph test set (about one in six labelled with two or more families).

  download (the owner accepts the gate and runs ``hf auth login`` first):
    hf download EbruResul/CTI-Rationale cti_rationale_gold_v2.jsonl report_split.csv LICENSE \\
      README.md --repo-type dataset --local-dir data/benchmarks/cti_rationale/raw
  build:  -> data/audits/cribl_ft/cases_cti_rationale_{train,test}.jsonl + cases_manifest.json counts

Leakage guards in ``build``: a report whose labelled paragraphs name a CTI-HAL group, one of its
ATT&CK aliases or its own software (``prepare_advisories.excluded_names``), or one listed in
``ATTRIBUTED`` (attributed to such a group without naming it), is dropped whole; a
paragraph overlapping a held-out text (public dev/test, CTI-HAL, AnnoCTR dev/test) or an Elastic or
vendor-set statement is dropped (``export_cases.drop_near_duplicates`` +
``export_attack_procedures.drop_overlaps``); a training paragraph repeating an existing training text
is dropped; a test paragraph overlapping any training text (the existing files and the new training
rows) is dropped. Training rows pass ``clean_cases.broken_reason``; test rows are not cleaned, as
for the other test sets. No CTI-HAL text is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

RAW = REPO_ROOT / "data" / "benchmarks" / "cti_rationale" / "raw"
GOLD = "cti_rationale_gold_v2.jsonl"
SPLIT_FILE = "report_split.csv"
TRAIN_REPORTS = ("train", "validation")
TRAINING_FILES = (
    "cases_train_clean.jsonl",
    "cases_annoctr_train_clean.jsonl",
    "cases_attack_train_clean.jsonl",
    "cases_advisories_train.jsonl",
    "cases_atomic_train.jsonl",
)
# reports their publishers attribute to a CTI-HAL group whose labelled paragraphs name neither the
# group nor ATT&CK-listed software of it (reviewed by hand on 2026-10-15, before any model result)
ATTRIBUTED = {
    "ar23-243a": "G0034: Infamous Chisel (CISA/NCSC attribute it to Sandworm, GRU unit 74455)",
    "russian-targeting-gov-business": "G0016: CEELOADER (Mandiant, UNC2452/APT29 activity)",
}


def paragraphs(gold_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per (report, paragraph) with its gold technique ids in first-seen order."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in gold_rows:
        record = found.setdefault(
            (str(row["report_id"]), str(row["chunk_id"])),
            {
                "report_id": str(row["report_id"]),
                "chunk_id": str(row["chunk_id"]),
                "vendor": row["vendor"],
                "text": " ".join(str(row["chunk_text"]).split()),
                "ids": [],
            },
        )
        technique = str(row["technique_id"]).strip()
        if technique and technique not in record["ids"]:
            record["ids"].append(technique)
    return list(found.values())


def statement_texts(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """The statement texts of benchmark report rows (Elastic, vendor set)."""
    return [item["text"] for row in rows for item in row["statements"] if item.get("text")]


def case(record: Mapping[str, Any], gold: Sequence[str], split: str) -> dict[str, Any]:
    return {
        "case_id": f"cti_rationale-{record['report_id']}-{record['chunk_id']}",
        "dataset": "cti_rationale",
        "split": split,
        "text": record["text"],
        "gold_ids": list(gold),
        "candidates": [],
        "gold_in_candidates": False,
        "gold_family_in_candidates": False,
        "pipeline_rank1": None,
        "jev_c1": None,
        "source": f"CTI-Rationale {record['vendor']} {record['report_id']}",
    }


def build(raw: Path = RAW, merge_validation: bool = False) -> None:
    from clean_cases import broken_reason
    from common import read_jsonl, write_jsonl
    from export_attack_procedures import HELD_OUT, drop_overlaps
    from export_cases import OUT, drop_near_duplicates
    from prepare_advisories import excluded_names, names_present, resolve_ids

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from scripts.prepare_elastic_reports import revoked_technique_map

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    catalog = json.loads((OUT / "catalog.json").read_text(encoding="utf-8"))["techniques"]
    families = {item["family"] for item in catalog}
    revoked = revoked_technique_map(settings.attack_bundle_path)
    manifest_cti_hal = REPO_ROOT / "data" / "benchmarks" / "cti_hal" / "MANIFEST.json"
    cti_hal_groups = sorted(set(json.loads(manifest_cti_hal.read_text())["group_ids"].values()))
    names = excluded_names(kb, cti_hal_groups)
    with (raw / SPLIT_FILE).open(encoding="utf-8") as handle:
        report_split = {row["report_id"]: row["split"] for row in csv.DictReader(handle)}
    records = paragraphs(read_jsonl(raw / GOLD))

    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_report[record["report_id"]].append(record)
    excluded = {}
    for report_id, items in sorted(by_report.items()):
        hits = sorted({hit for item in items for hit in names_present(item["text"], names)})
        if hits:
            excluded[report_id] = hits
        elif report_id in ATTRIBUTED:
            excluded[report_id] = [ATTRIBUTED[report_id]]

    train_reports = TRAIN_REPORTS if merge_validation else ("train",)
    parts: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    drops: dict[str, Counter] = {part: Counter() for part in parts}
    for record in records:
        split = report_split[record["report_id"]]
        part = (
            "train" if split in train_reports else "validation" if split == "validation" else "test"
        )
        gold = resolve_ids(record["ids"], revoked, families)
        if record["report_id"] in excluded:
            drops[part]["cti_hal_group_report"] += 1
        elif not gold:
            drops[part]["no_listed_family"] += 1
        elif part == "train" and (reason := broken_reason(record["text"])):
            drops[part][reason] += 1
        else:
            parts[part].append(case(record, gold, f"cti_rationale_{part}"))

    held_out = [
        row["text"] for split in HELD_OUT for row in read_jsonl(OUT / f"cases_{split}.jsonl")
    ]
    held_out += statement_texts(read_jsonl(REPO_ROOT / "data/benchmarks/elastic/reports.jsonl"))
    held_out += statement_texts(
        read_jsonl(REPO_ROOT / "data/benchmarks/elastic/reports_open_set.jsonl")
    )
    held_out += statement_texts(read_jsonl(REPO_ROOT / "data/benchmarks/vendor/reports.jsonl"))
    training = [row["text"] for name in TRAINING_FILES for row in read_jsonl(OUT / name)]

    overlaps: dict[str, dict[str, int]] = {}
    for part in parts:
        rows, near = drop_near_duplicates(parts[part], held_out)
        rows, contained = drop_overlaps(rows, held_out)
        overlaps[part] = {**near, **contained}
        parts[part] = rows
    parts["train"], repeats = drop_near_duplicates(parts["train"], training)
    overlaps["train"]["repeats_existing_training"] = repeats["held_out_duplicates"]
    # validation (a selection set) and test may not overlap any training text either
    seen = training + [row["text"] for row in parts["train"]]
    for part in ("validation", "test"):
        rows, near = drop_near_duplicates(parts[part], seen)
        parts[part], contained = drop_overlaps(rows, seen)
        overlaps[part]["overlaps_training"] = near["held_out_duplicates"] + sum(contained.values())

    def part_of(split: str) -> str:
        return (
            "train" if split in train_reports else "validation" if split == "validation" else "test"
        )

    manifest_path = OUT / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("splits", {}).pop("cti_rationale_validation", None)
    for part, rows in parts.items():
        path = OUT / f"cases_cti_rationale_{part}.jsonl"
        if part == "validation" and merge_validation:
            path.unlink(missing_ok=True)
            continue
        write_jsonl(path, rows)
        several = sum(len({item.split(".")[0] for item in row["gold_ids"]}) > 1 for row in rows)
        info = {
            "source": "huggingface.co/datasets/EbruResul/CTI-Rationale (gold v2; annotations "
            "CC BY 4.0, paragraph text the publishers')",
            "attack_version": kb.version,
            "validation_merged_into_train": merge_validation,
            "reports": sorted(r for r, s in report_split.items() if part_of(s) == part),
            "excluded_reports": {
                r: h for r, h in excluded.items() if part_of(report_split[r]) == part
            },
            "excluded_groups": cti_hal_groups,
            "drops": dict(sorted(drops[part].items())),
            "overlap_drops": overlaps[part],
            "cases": len(rows),
            "several_families": several,
            "families": len({item.split(".")[0] for row in rows for item in row["gold_ids"]}),
        }
        manifest.setdefault("splits", {})[f"cti_rationale_{part}"] = info
        print(
            f"{path.name}: {len(rows)} paragraphs, {several} with 2+ families, "
            f"{info['families']} families; drops {info['drops']}; overlaps {overlaps[part]}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"reports dropped whole (CTI-HAL group or software named): {len(excluded)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument(
        "--merge-validation",
        action="store_true",
        help="the validation reports join the training rows (E3's data) instead of forming "
        "cases_cti_rationale_validation.jsonl, a checkpoint-selection set",
    )
    args = parser.parse_args()
    build(args.raw, merge_validation=args.merge_validation)


if __name__ == "__main__":
    main()
