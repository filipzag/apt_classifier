"""Blind check of extra answers: is a second or third answer supported by the sentence, whether or
not it is the label (standard library only; prereg ``cribl_yesno_2026-10-14.json``).

  sheet: draw ``--per-model`` extra answers per run at the ``--budget`` threshold (seed 7), merge the
         same (sentence, technique) from both runs into one item, shuffle, and write
         ``blind_<split>_sheet.tsv`` (item, sentence, technique name; no model, no label) and
         ``blind_<split>_key.json`` (which runs gave each item, whether it is a labelled family)
  score: read the sheet with a filled ``supported`` column (yes / no) and the key; per run: the
         share of its drawn extra answers that are supported, split by label match

  python3 blind_check.py sheet --split cti_hal E2=E2_cti_hal.jsonl M1=M1_cti_hal.jsonl
  python3 blind_check.py score --split cti_hal

Files are read from and written to ``data/audits/cribl_ft`` (local: CTI-HAL text).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from typing import Any

from common import read_jsonl
from multi_answer import AUDITS, answers, budget_threshold

SEED = 7


def extras(rows: list[dict[str, Any]], threshold: float) -> list[tuple[str, str, bool]]:
    """(case id, technique, is a labelled family) for every second or third answer."""
    found = []
    for row in rows:
        for family, technique in answers(row, threshold)[1:]:
            found.append((row["case_id"], technique or family, family in row["gold_families"]))
    return found


def sheet(args: argparse.Namespace) -> None:
    labels = {
        item["id"]: item["label"]
        for item in json.loads((AUDITS / "catalog.json").read_text(encoding="utf-8"))["techniques"]
    }
    texts = {
        case["case_id"]: re.sub(r"\s+", " ", case["text"]).strip()
        for case in read_jsonl(AUDITS / f"cases_{args.split}.jsonl")
    }
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        rows = read_jsonl(AUDITS / path)
        drawn = extras(rows, budget_threshold(rows, args.budget))
        for case_id, technique, labelled in random.Random(SEED).sample(
            drawn, min(args.per_model, len(drawn))
        ):
            item = items.setdefault(
                (case_id, technique),
                {"case_id": case_id, "technique": technique, "labelled": labelled, "runs": []},
            )
            item["runs"].append(name)
    order = list(items.values())
    random.Random(SEED + 1).shuffle(order)
    with (AUDITS / f"blind_{args.split}_sheet.tsv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["item", "sentence", "technique", "supported"])
        for number, item in enumerate(order, start=1):
            item["item"] = number
            label = labels.get(item["technique"], item["technique"])
            writer.writerow([number, texts[item["case_id"]], label, ""])
    key = {"budget": args.budget, "per_model": args.per_model, "seed": SEED, "items": order}
    (AUDITS / f"blind_{args.split}_key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(f"{len(order)} items -> blind_{args.split}_sheet.tsv")


def score(args: argparse.Namespace) -> None:
    key = json.loads((AUDITS / f"blind_{args.split}_key.json").read_text(encoding="utf-8"))
    with (AUDITS / f"blind_{args.split}_sheet.tsv").open(encoding="utf-8", newline="") as handle:
        judged = {
            int(row["item"]): row["supported"].strip().lower()
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if any(value not in ("yes", "no") for value in judged.values()):
        raise SystemExit("every item needs supported = yes or no")
    report: dict[str, Any] = {}
    runs = sorted({run for item in key["items"] for run in item["runs"]})
    for run in runs:
        mine = [item for item in key["items"] if run in item["runs"]]

        def rate(subset: list[dict[str, Any]]) -> dict[str, Any]:
            yes = sum(judged[item["item"]] == "yes" for item in subset)
            return {
                "n": len(subset),
                "supported": yes,
                "share": round(yes / len(subset), 3) if subset else None,
            }

        report[run] = {
            "all": rate(mine),
            "labelled_family": rate([item for item in mine if item["labelled"]]),
            "not_labelled": rate([item for item in mine if not item["labelled"]]),
        }
        print(run, json.dumps(report[run]))
    (AUDITS / f"blind_{args.split}_result.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    make = commands.add_parser("sheet")
    make.set_defaults(func=sheet)
    make.add_argument("runs", nargs="+", help="NAME=<predictions .jsonl>")
    make.add_argument("--split", default="cti_hal")
    make.add_argument("--budget", type=float, default=1.3)
    make.add_argument("--per-model", type=int, default=50)
    check = commands.add_parser("score")
    check.set_defaults(func=score)
    check.add_argument("--split", default="cti_hal")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
