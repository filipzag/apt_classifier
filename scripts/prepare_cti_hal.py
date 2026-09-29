"""Convert the CTI-HAL corpus (dessertlab/CTI-HAL) into local benchmark files.

CTI-HAL pairs human-annotated statements from real threat reports with ATT&CK techniques
and, through its folder layout, with the reporting target's threat group. The repository
publishes no licence, so the derived files are for local evaluation only and are written
under ``data/benchmarks/cti_hal/`` (not redistributed).

Outputs:
- ``cti_hal_test.jsonl``: one technique-mapping case per annotated statement (expected id =
  sub-technique when annotated, else technique), compatible with ``benchmark_mapping.py
  --root data/benchmarks/cti_hal --datasets cti_hal --allow-leaky`` (the split is named
  ``test`` for the loader; it is not leaky, the flag only bypasses the name check when the
  dataset is not in the default list).
- ``reports.jsonl``: one row per report with its ATT&CK group id and the statements, for the
  real-report attribution benchmark (``attribution_benchmark.py --cti-hal``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GROUP_IDS = {
    "apt29": "G0016",
    "carbanak": "G0008",
    "fin6": "G0037",
    "fin7": "G0046",
    "oilrig": "G0049",
    "sandworm": "G0034",
    "wizardspider": "G0102",
}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "benchmarks" / "cti_hal"


def convert(source: Path, output: Path, *, kb_techniques: set[str] | None = None) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    reports = []
    stats = {"reports": 0, "statements": 0, "unknown_group": 0, "unmapped_label": 0}
    for path in sorted((source / "data").glob("*/*.json")):
        folder = path.parent.name.lower()
        group_id = GROUP_IDS.get(folder)
        if group_id is None:
            stats["unknown_group"] += 1
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            continue
        report_id = f"cti_hal-{folder}-{path.stem}"
        statements = []
        for item in items:
            text = str(item.get("context") or "").strip()
            meta = item.get("metadata") or {}
            label = str(meta.get("sub_technique") or item.get("technique") or "").upper().strip()
            if not text or not label:
                continue
            if kb_techniques is not None and label not in kb_techniques:
                parent = label.split(".")[0]
                if parent in kb_techniques:
                    label = parent
                else:
                    stats["unmapped_label"] += 1
                    continue
            statements.append({"text": text, "technique_id": label})
        if not statements:
            continue
        stats["reports"] += 1
        # one case per distinct statement text (a statement can carry several labels)
        by_text: dict[str, set[str]] = {}
        for statement in statements:
            by_text.setdefault(statement["text"], set()).add(statement["technique_id"])
        for text, labels in by_text.items():
            digest = hashlib.sha256(f"{report_id}:{text}".encode()).hexdigest()[:10]
            cases.append(
                {
                    "case_id": f"cti_hal-test-{digest}",
                    "document_id": report_id,
                    "text": text,
                    "expected_technique_ids": sorted(labels),
                    "original_technique_ids": sorted(labels),
                    "label_quality": "public_dataset",
                    "source": "dessertlab/CTI-HAL:cti_hal",
                    "split": "test",
                    "metadata": {"group_id": group_id, "report": path.stem, "folder": folder},
                }
            )
            stats["statements"] += 1
        reports.append(
            {
                "report_id": report_id,
                "group_id": group_id,
                "folder": folder,
                "statements": [
                    {"text": text, "technique_ids": sorted(labels)}
                    for text, labels in by_text.items()
                ],
            }
        )
    (output / "cti_hal_test.jsonl").write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8"
    )
    (output / "reports.jsonl").write_text(
        "\n".join(json.dumps(report, ensure_ascii=False) for report in reports) + "\n",
        encoding="utf-8",
    )
    (output / "MANIFEST.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/dessertlab/CTI-HAL",
                "licence": "none published; local evaluation only, do not redistribute",
                "group_ids": GROUP_IDS,
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="path to a CTI-HAL clone")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kb", type=Path, default=REPO_ROOT / "data" / "kb" / "attack_kb.json")
    args = parser.parse_args(argv)
    kb_techniques = None
    if args.kb.is_file():
        kb_techniques = set(json.loads(args.kb.read_text(encoding="utf-8"))["techniques"])
    stats = convert(args.source, args.output, kb_techniques=kb_techniques)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
