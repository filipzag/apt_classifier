"""The other evaluation sets in the mapping benchmark's case format (docs 21.3.18, 2026-10-20).

Writes ``data/benchmarks/extra/<name>_test.jsonl`` (gitignored) for ``scripts/benchmark_mapping.py
--root data/benchmarks/extra --datasets <name>``:

- ``annoctr``: AnnoCTR test sentences (``cases_annoctr_test.jsonl``; parent-level labels)
- ``cti_rationale``: CTI-Rationale test (gated, local only)
- ``athena_ate``: AthenaBench ATE scenarios (non-commercial licence, evaluation only)
- ``ctibench_ate``: CTIBench ATE sentence rows (CC BY-NC-SA, evaluation only)

The cribl_ft case rows carry ``gold_ids``. The Elastic and vendor reports label techniques per
report, not per statement, so they have no statement-level mapping set (their attribution
benchmarks cover them).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = REPO_ROOT / "data" / "audits" / "cribl_ft"
OUT = REPO_ROOT / "data" / "benchmarks" / "extra"
SOURCES = {
    "annoctr": ("cases_annoctr_test.jsonl", "annoctr:annoctr"),
    "cti_rationale": ("cases_cti_rationale_test.jsonl", "cti_rationale:cti_rationale"),
    "athena_ate": ("cases_ext_athena_ate.jsonl", "athenabench:athena_ate"),
    "ctibench_ate": ("cases_ext_ctibench_ate.jsonl", "ctibench:ctibench_ate"),
}


def convert(name: str, path: Path, source: str) -> int:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        gold = [str(t) for t in case.get("gold_ids") or []]
        text = str(case.get("text") or "").strip()
        if not gold or not text:
            continue
        rows.append(
            {
                "case_id": f"{name}-test-{case['case_id']}",
                "document_id": str(case.get("report_id") or case.get("case_id")),
                "text": text,
                "expected_technique_ids": gold,
                "label_quality": "public_dataset",
                "source": source,
                "split": "test",
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return len(rows)


def main() -> None:
    for name, (filename, source) in SOURCES.items():
        print(name, convert(name, CASES / filename, source), "cases")


if __name__ == "__main__":
    main()
