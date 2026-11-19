"""Drop broken training rows (standard library only; prereg ``cribl_family_data_2026-10-14.json``).

A training row is dropped when its text is not a sentence a model can learn a technique from:

- ``fragment``: fewer than ``MIN_CONTENT_WORDS`` content words (3+ letters, not a stop word), e.g.
  what is left of "<group> has utilized <tool> during <campaign> and <campaign>." once the names are
  removed ("uses DNS for C2." has two and stays);
- ``figure``: a figure caption ("*Figure 5: Encrypted files*"): a caption label first ("Figure 5:",
  "Fig.3.", "Figure 9") and fewer than ``CAPTION_CONTENT_WORDS`` content words ("Figure 4 shows a
  PDF that pretends to be ..." stays);
- ``table_row``: a markdown table row (starts with "|", 3+ "|");
- ``technique_id``: the text names an ATT&CK technique id ("T1020 Automated exfiltration"), which
  would teach the head to read the label;
- ``ioc_list``: ``IOC_HASHES``+ hashes with fewer than ``IOC_CONTENT_WORDS`` content words (a hash table).

The thresholds were set by reading 30 dropped training rows (dev and test unread), before any model
existed.

Only training files are cleaned; dev and test stay as they are, so results remain comparable.

  python3 clean_cases.py cases_train.jsonl cases_annoctr_train.jsonl ...
  -> <name>_clean.jsonl beside each input, per-rule counts in cases_manifest.json ("cleaning")
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import read_jsonl, write_jsonl  # noqa: E402

AUDITS = HERE.parents[1] / "data" / "audits" / "cribl_ft"
MIN_CONTENT_WORDS = 2
CAPTION_CONTENT_WORDS = 10
IOC_HASHES = 3
IOC_CONTENT_WORDS = 12
_STOP_WORDS = """
    the and for with has have had was were are been being its their they them this that
    these those from into onto via also can could may might will would but not all any each
    other such than then there which who whom whose what when where while during against
    about after before between through within without upon among across over under out off
    own more most some only both either neither just very does did done his her she him our
    your you one two
"""
STOP_WORDS = frozenset(_STOP_WORDS.split())
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
_HASH = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
_FIGURE = re.compile(r"^\W*(?:figure|fig\.)\s*\d+\W*(?:[:.\-–]|$)", re.I)  # a caption label
_TECHNIQUE_ID = re.compile(r"\bT1\d{3}(?:\.\d{3})?\b")  # T9000 is malware, not a technique


def content_words(text: str) -> int:
    """Words of 3+ letters that are not stop words (letter runs inside hashes do not count)."""
    return sum(1 for word in _WORD.findall(_HASH.sub(" ", text)) if word.lower() not in STOP_WORDS)


def broken_reason(text: str) -> str | None:
    """The rule a training text breaks, or ``None`` for a usable sentence."""
    stripped = text.strip()
    words = content_words(stripped)
    if _FIGURE.match(stripped) and words < CAPTION_CONTENT_WORDS:
        return "figure"
    if stripped.startswith("|") and stripped.count("|") >= 3:
        return "table_row"
    if _TECHNIQUE_ID.search(stripped):
        return "technique_id"
    if len(_HASH.findall(stripped)) >= IOC_HASHES and words < IOC_CONTENT_WORDS:
        return "ioc_list"
    if words < MIN_CONTENT_WORDS:
        return "fragment"
    return None


def clean_path(path: Path) -> Path:
    return path.with_name(path.stem + "_clean.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--cases-dir", type=Path, default=AUDITS)
    args = parser.parse_args()
    report = {}
    for name in args.files:
        path = args.cases_dir / name
        rows = read_jsonl(path)
        reasons = Counter()
        kept = []
        for row in rows:
            reason = broken_reason(row["text"])
            if reason:
                reasons[f"{row['dataset']}:{reason}"] += 1
            else:
                kept.append(row)
        write_jsonl(clean_path(path), kept)
        report[name] = {
            "rows": len(rows),
            "kept": len(kept),
            "dropped": dict(sorted(reasons.items())),
        }
        print(f"{name}: kept {len(kept)} of {len(rows)} {dict(sorted(reasons.items()))}")
    manifest_path = args.cases_dir / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("cleaning", {}).update(report)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
