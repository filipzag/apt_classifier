"""The actor head's case files: one anonymised report sentence -> threat group (project .venv;
README.md here; prereg ``cribl_actor_head_2026-10-16.json``).

Training texts are ATT&CK's own per-group texts as ``apt_classifier.text_group.build_corpus``
anonymises them (group procedure examples and descriptions, single-group software and campaign
procedure examples; actor names -> "the adversary", operation names -> "the operation"), labelled
with the group(s) they document and kept when every group is in the group catalogue and there are
at most ``MAX_GROUPS`` of them. The report sets are the statements of the CTI-HAL, Elastic and
vendor reports with the same anonymisation, one row per statement labelled with the report's group
and carrying ``report_id`` for the per-report aggregation (``group_text_eval.py``).

Leakage: ATT&CK cites some of these reports, so training texts that are near copies of, contain,
or share a 10-token run with any report statement are dropped (``export_cases.drop_near_duplicates``,
``export_attack_procedures.drop_overlaps``).

  build -> group_catalog.json (from export_group_cases.py, required) and
    cases_actor_train.jsonl      training sentences (dataset actor_attack)
    cases_actor_vendor_dev.jsonl dev: up to DEV_PER_REPORT statements per vendor report
    cases_actor_vendor.jsonl     every vendor statement (report-level dev)
    cases_actor_cti_hal.jsonl    test (local only)
    cases_actor_elastic.jsonl    test (local only; actor names removed like CTI-HAL, not the
                                 Elastic benchmark's strict anonymiser)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

MAX_GROUPS = 3
DEV_PER_REPORT = 25
MIN_WORDS = 3
RAW_STIX = Path("/home/fzagar/apt_classifier/data/raw/enterprise-attack.json")


def anonymiser(kb: Any) -> Any:
    """``text -> text`` with every ATT&CK group name or alias replaced by "the adversary" and
    every campaign name by "the operation" (the text-group corpus's patterns)."""
    from apt_classifier.text_group import _campaign_pattern, actor_names, evidence_name_pattern

    actors = evidence_name_pattern(actor_names(kb), before_hyphen=True)
    campaigns = _campaign_pattern(kb)

    def clean(text: str) -> str:
        text = str(text or "")
        if actors is not None:
            text = actors.sub(lambda m: "the adversary" + (m.group("possessive") or ""), text)
        if campaigns is not None:
            text = campaigns.sub("the operation", text)
        text = re.sub(r"\b(the|a|an) the (adversary|operation)\b", r"the \2", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    return clean


def training_rows(records: Iterable[Mapping[str, Any]], listed: set[str]) -> list[dict[str, Any]]:
    """Corpus records as cases: the groups documented (1..MAX_GROUPS, all listed)."""
    rows = []
    for index, record in enumerate(records, start=1):
        groups = [g for g in record.get("groups", ()) if g in listed]
        if not groups or len(groups) > MAX_GROUPS or len(groups) != len(record.get("groups", ())):
            continue
        rows.append(
            {
                "case_id": f"actor_train-{index:06d}",
                "dataset": "actor_attack",
                "split": "actor_train",
                "text": record["text"],
                "gold_ids": groups,
                "candidates": [],
                "gold_in_candidates": False,
                "gold_family_in_candidates": False,
                "pipeline_rank1": None,
                "jev_c1": None,
                "source": f"ATT&CK {record.get('source')} {record.get('owner')}",
            }
        )
    return rows


def statement_rows(
    reports: Iterable[Mapping[str, Any]], split: str, clean: Any, listed: set[str]
) -> tuple[list[dict[str, Any]], Counter]:
    rows, drops = [], Counter()
    for report in reports:
        group_id = report.get("group_id")
        if group_id not in listed:
            drops["group_not_listed"] += 1
            continue
        kept = 0
        for number, statement in enumerate(report["statements"], start=1):
            text = clean(statement.get("text", ""))
            if len(text.split()) < MIN_WORDS:
                drops["short_statement"] += 1
                continue
            rows.append(
                {
                    "case_id": f"{split}-{report['report_id']}-{number:03d}",
                    "report_id": report["report_id"],
                    "dataset": split,
                    "split": split,
                    "text": text,
                    "gold_ids": [group_id],
                    "candidates": [],
                    "gold_in_candidates": False,
                    "gold_family_in_candidates": False,
                    "pipeline_rank1": None,
                    "jev_c1": None,
                }
            )
            kept += 1
        if not kept:
            drops["no_statements"] += 1
    return rows, drops


def dev_subsample(
    rows: Sequence[Mapping[str, Any]], per_report: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_report: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_report.setdefault(row["report_id"], []).append(row)
    out: list[dict[str, Any]] = []
    for report_id in sorted(by_report):
        items = list(by_report[report_id])
        rng.shuffle(items)
        out.extend(
            dict(item, split="actor_vendor_dev", dataset="actor_vendor_dev")
            for item in items[:per_report]
        )
    return out


def build(args: argparse.Namespace) -> None:
    from common import read_jsonl, write_jsonl
    from export_attack_procedures import drop_overlaps
    from export_cases import OUT, drop_near_duplicates

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.text_group import build_corpus

    kb = AttackKnowledgeBase.load(get_settings().kb_path)
    listed = {
        item["id"] for item in json.loads((OUT / "group_catalog.json").read_text())["techniques"]
    }
    clean = anonymiser(kb)
    bundle = json.loads(RAW_STIX.read_text(encoding="utf-8")) if RAW_STIX.is_file() else None
    records = build_corpus(kb, bundle)
    train = training_rows(records, listed)
    by_source = Counter(row["source"].split(" ")[1] for row in train)

    cti_hal = read_jsonl(REPO_ROOT / "data/benchmarks/cti_hal/reports.jsonl")
    elastic = read_jsonl(REPO_ROOT / "data/benchmarks/elastic/reports.jsonl")
    vendor = read_jsonl(REPO_ROOT / "data/benchmarks/vendor/reports.jsonl")
    files: dict[str, list[dict[str, Any]]] = {}
    drops: dict[str, Counter] = {}
    files["actor_cti_hal"], drops["cti_hal"] = statement_rows(
        cti_hal, "actor_cti_hal", clean, listed
    )
    files["actor_elastic"], drops["elastic"] = statement_rows(
        elastic, "actor_elastic", clean, listed
    )
    files["actor_vendor"], drops["vendor"] = statement_rows(vendor, "actor_vendor", clean, listed)
    files["actor_vendor_dev"] = dev_subsample(files["actor_vendor"], DEV_PER_REPORT, args.seed)

    held_out = [
        row["text"]
        for name in ("actor_cti_hal", "actor_elastic", "actor_vendor")
        for row in files[name]
    ]
    train, near = drop_near_duplicates(train, held_out)
    train, contained = drop_overlaps(train, held_out)
    files["actor_train"] = train

    manifest_path = OUT / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    info: dict[str, Any] = {
        "attack_version": kb.version,
        "software_bundle": bundle is not None,
        "train_by_source_before_filters": dict(by_source),
        "overlap_drops": {**near, **contained},
        "drops": {k: dict(v) for k, v in drops.items()},
        "anonymisation": "ATT&CK group names and aliases -> 'the adversary', campaign names -> 'the operation' (text_group patterns); software names kept",
    }
    for name, rows in files.items():
        write_jsonl(OUT / f"cases_{name}.jsonl", rows)
        info[name] = {
            "cases": len(rows),
            "groups": len({g for row in rows for g in row["gold_ids"]}),
            "reports": len({row.get("report_id") for row in rows if row.get("report_id")}),
            "several_groups": sum(len(row["gold_ids"]) > 1 for row in rows),
        }
        print(f"cases_{name}.jsonl: {info[name]}")
    manifest["actor"] = info
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"training by source: {dict(by_source)}; overlap drops {info['overlap_drops']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
