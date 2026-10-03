"""Export the real-report test texts exactly as the attribution benchmarks read them.

Run with PYTHONPATH=<elastic branch archive> (see README.md). Writes test_sets.json (STUDY_DIR, or
the path given as the first argument): per set, per report, the observation texts (after the
benchmark's statement cap and anonymization level)."""

import json
import sys
from pathlib import Path

from harness import REPO, STUDY_DIR

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from scripts.attribution_benchmark import anonymize_report_folds, load_report_folds
from scripts.report_anonymizer import ReportAnonymizer

out = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY_DIR / "test_sets.json"
kb = AttackKnowledgeBase.load(Settings().kb_path)
B = REPO / "data" / "benchmarks"
sets = {}


def export(name, path, cap, level, open_path=None):
    dropper = ReportAnonymizer(kb, level) if level in ("strict", "tradecraft") else None
    loaded = load_report_folds(path, max_observations=cap, anonymizer=dropper)
    if open_path:
        loaded += load_report_folds(open_path, max_observations=cap, anonymizer=dropper)
    if level is not None:
        loaded, _, totals = anonymize_report_folds(loaded, kb, level=level)
    sets[name] = [
        {
            "report_id": f.report_id,
            "group_id": f.group_id,
            "accept": sorted(f.accept),
            "texts": list(f.observations),
        }
        for f in loaded
    ]
    print(name, len(loaded), "reports", sum(len(f.observations) for f in loaded), "texts")


for level, tag in (
    (None, "as_reported"),
    ("names", "names"),
    ("strict", "strict"),
    ("tradecraft", "tradecraft"),
):
    export(f"cti_hal_{tag}", B / "cti_hal/reports.jsonl", 30, level)
    export(
        f"elastic_{tag}",
        B / "elastic/reports.jsonl",
        None,
        level,
        B / "elastic/reports_open_set.jsonl",
    )
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sets))
