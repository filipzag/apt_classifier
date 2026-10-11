"""Smoke test and zero-shot format probe for cribl-decision (``.venv-cribl``).

1. Loads Cribl's adapter (fails unless every tensor loaded) and asks the model card's example
   ("A host made a network connection." -> network vs file) in each prompt format, both orders.
2. Scores the first ``--limit`` dev cases (C1 question, 3 orders) in each format with the adapter,
   and in the best format with the adapter switched off (plain Qwen3.5-4B-Base), and writes
   ``data/audits/cribl_ft/probe.json`` with the chosen format: the best dev exact (ties: the
   earlier format in ``FORMATS``). Dev only; CTI-HAL and test are never read here.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from common import FORMATS, read_jsonl
from decision_model import Readout, load_model
from evaluate import metrics, run_split

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
CARD_OPTIONS = [
    {"id": "network", "label": "network", "what": "network activity"},
    {"id": "file", "label": "file", "what": "file activity"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer, info = load_model("cribl")
    print(f"adapter check: {info['adapter_check']}")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": info,
        "card_example": {},
        "dev_first": args.limit,
        "formats": {},
    }
    for fmt in FORMATS:
        readout = Readout(model, tokenizer, fmt)
        both = []
        for options in (CARD_OPTIONS, CARD_OPTIONS[::-1]):
            probabilities = readout.probabilities(
                "A host made a network connection.", "Which event type is this?", options
            )
            both.append(
                {
                    option["id"]: round(p, 4)
                    for option, p in zip(options, probabilities, strict=True)
                }
            )
        report["card_example"][fmt] = both
        print(f"card example [{fmt}]: {both}")

    dev = read_jsonl(AUDITS / "cases_dev.jsonl")[: args.limit]
    for fmt in FORMATS:
        rows = run_split(
            Readout(model, tokenizer, fmt),
            dev,
            AUDITS / f"B0probe_{fmt}_dev_first{args.limit}.jsonl",
            progress=0,
        )
        pooled = metrics(dev, rows)["arm"]["pooled"]
        report["formats"][fmt] = pooled
        print(
            f"[{fmt}] dev first {args.limit}: exact {pooled['exact']} parent-aware {pooled['parent_aware']} no-pick {pooled['no_pick']} ECE {pooled['ece']}"
        )
    chosen = max(FORMATS, key=lambda fmt: (report["formats"][fmt]["exact"], -FORMATS.index(fmt)))
    report["chosen_format"] = chosen

    with model.disable_adapter():
        rows = run_split(
            Readout(model, tokenizer, chosen),
            dev,
            AUDITS / f"base_{chosen}_dev_first{args.limit}.jsonl",
            progress=0,
        )
    report["base_without_adapter"] = metrics(dev, rows)["arm"]["pooled"]
    report["references"] = {
        key: value
        for key, value in metrics(dev, rows).items()
        if key in ("pipeline_rank1", "jev_c1")
    }
    print(f"plain base [{chosen}]: exact {report['base_without_adapter']['exact']}")
    print(f"chosen format: {chosen}")
    (AUDITS / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
