"""Turn AnnoCTR's ATT&CK technique links into sentence cases (project .venv; README.md here).

AnnoCTR (Lange et al., LREC-COLING 2024; github.com/boschresearch/anno-ctr-lrec-coling-2024,
CC-BY-SA 4.0) annotates real vendor threat reports; ``AnnoCTR/linking_mitre_only/<split>.jsonl``
holds one record per linked mention with its sentence (``context_left + mention +
context_right``). Every sentence with at least one technique link becomes one case whose gold ids are
its linked techniques, remapped to the local ATT&CK release (revoked or deprecated ids follow
``AttackIdMap``; ids that cannot be mapped are recorded as unscorable). Markdown links in the text
keep their anchor text only. AnnoCTR's own document split (train/dev/test) is kept.

Writes ``<out>/annoctr_<split>.jsonl`` in the public benchmark format (``benchmark_mapping.py
--root <out> --datasets annoctr``) and ``<out>/MANIFEST.json``. Near duplicates of held-out texts are
removed later, when the case files are exported (``export_cases.py``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import get_settings  # noqa: E402
from apt_classifier.evaluation import AttackIdMap  # noqa: E402

SOURCE = REPO_ROOT / "data" / "benchmarks" / "annoctr" / "source"
OUT = REPO_ROOT / "data" / "benchmarks" / "annoctr" / "normalized"
SPLITS = ("train", "dev", "test")
TECHNIQUE_LINK = re.compile(r"attack\.mitre\.org/techniques/(T\d{4})(?:/(\d{3}))?")
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()\s]|\([^()\s]*\))*\)")


def clean_sentence(text: str) -> str:
    """Anchor text for markdown links, single spaces."""
    return " ".join(MARKDOWN_LINK.sub(r"\1", text).split())


def technique_id(link: str) -> str | None:
    match = TECHNIQUE_LINK.search(link or "")
    if match is None:
        return None
    return match.group(1) + (f".{match.group(2)}" if match.group(2) else "")


def sentence_cases(records: Iterable[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    """One case per (document, sentence) with a technique link, in first-seen order."""
    labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    classes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.get("entity_type") != "TECHNIQUE":
            continue
        identifier = technique_id(str(record.get("label_link") or ""))
        if identifier is None:
            continue
        sentence = clean_sentence(
            f"{record.get('context_left', '')}{record.get('mention', '')}{record.get('context_right', '')}"
        )
        if not sentence:
            continue
        key = (str(record["document"]), sentence)
        if identifier not in labels[key]:
            labels[key].append(identifier)
        classes[key][str(record.get("entity_class") or "")] += 1
    return [
        {
            "case_id": f"annoctr-{split}-{index:06d}",
            "document_id": document,
            "text": sentence,
            "original_technique_ids": ids,
            "split": split,
            "metadata": {"mention_classes": dict(classes[(document, sentence)])},
        }
        for index, ((document, sentence), ids) in enumerate(labels.items(), start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    id_map = AttackIdMap.from_bundle(settings.attack_bundle_path)
    commit = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "https://github.com/boschresearch/anno-ctr-lrec-coling-2024",
        "source_commit": commit,
        "licence": "CC-BY-SA-4.0 (AnnoCTR corpus); cite Lange et al., LREC-COLING 2024",
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "attack_version": kb.version,
        "splits": {},
    }
    for split in SPLITS:
        path = args.source / "AnnoCTR" / "linking_mitre_only" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        cases = sentence_cases(records, split)
        remapped = unscorable = 0
        rows = []
        for case in cases:
            remap = id_map.remap(case["original_technique_ids"], kb.techniques)
            remapped += len(remap.replacements)
            unscorable += len(remap.unscorable)
            rows.append(
                {
                    **case,
                    "expected_technique_ids": list(remap.mapped),
                    "label_replacements": remap.replacements,
                    "unscorable_technique_ids": list(remap.unscorable),
                    "label_quality": "annoctr_expert_link",
                    "source": "boschresearch/anno-ctr-lrec-coling-2024:annoctr",
                    "published_at": None,
                }
            )
        with (args.out / f"annoctr_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        scorable = [row for row in rows if row["expected_technique_ids"]]
        manifest["splits"][split] = {
            "records": len(records),
            "cases": len(rows),
            "scorable_cases": len(scorable),
            "documents": len({row["document_id"] for row in rows}),
            "techniques": len({t for row in scorable for t in row["expected_technique_ids"]}),
            "subtechnique_share": round(
                sum(any("." in t for t in row["expected_technique_ids"]) for row in scorable)
                / max(len(scorable), 1),
                4,
            ),
            "remapped_labels": remapped,
            "unscorable_labels": unscorable,
        }
        print(f"{split}: {manifest['splits'][split]}")
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
