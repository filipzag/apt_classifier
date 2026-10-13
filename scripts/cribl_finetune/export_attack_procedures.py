"""ATT&CK's own procedure examples as family-head training rows (project .venv; README.md here).

Every active ``uses`` relationship from a group, software or campaign to a listed technique (the C1
catalogue, ``export_cases.build_catalog``) whose description is non-empty becomes one sentence:
citations and markdown links stripped, one line. Identical texts are merged (their techniques
together). Rows that overlap any held-out text are dropped, the held-out texts being the cases of
public dev, public test, CTI-HAL, AnnoCTR dev and AnnoCTR test:

- exact or near duplicates (token Jaccard >= 0.9, ``export_cases.drop_near_duplicates``);
- containment: one text's tokens appear as a contiguous run in the other (the shorter has at least
  ``MIN_CONTAINED`` tokens), which catches a short CTI-HAL statement quoted inside a procedure;
- a shared run of ``SHARED_RUN`` tokens, which catches paraphrases that keep a long phrase.

Writes ``data/audits/cribl_ft/cases_attack_train.jsonl`` (dataset ``attack_procedures``, no
candidates: for ``family_head.py`` only) and records the counts in ``cases_manifest.json``. The
output holds ATT&CK text only; CTI-HAL is read locally for the filter and never written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from common import read_jsonl, write_jsonl  # noqa: E402
from export_cases import (  # noqa: E402
    OUT,
    _active,
    _attack_id,
    _load,
    build_catalog,
    drop_near_duplicates,
)

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.attack.parse import strip_attack_citations  # noqa: E402
from apt_classifier.config import get_settings  # noqa: E402

HELD_OUT = ("dev", "test", "cti_hal", "annoctr_dev", "annoctr_test")
SOURCES = ("intrusion-set", "malware", "tool", "campaign")
MIN_CONTAINED = 6
SHARED_RUN = 10
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def procedure_rows(bundle: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per distinct cleaned procedure text, with every technique it was written for."""
    objects = [item for item in bundle.get("objects") or () if isinstance(item, Mapping)]
    by_ref = {str(item.get("id")): item for item in objects}
    labels: dict[str, list[str]] = {}
    origins: dict[str, set[str]] = {}
    for item in objects:
        if item.get("type") != "relationship" or item.get("relationship_type") != "uses":
            continue
        if not _active(item) or not str(item.get("description") or "").strip():
            continue
        source = by_ref.get(str(item.get("source_ref")))
        target = by_ref.get(str(item.get("target_ref")))
        if not source or not target or source.get("type") not in SOURCES:
            continue
        if target.get("type") != "attack-pattern" or not _active(source) or not _active(target):
            continue
        technique = _attack_id(target)
        if technique not in catalog:
            continue
        text = " ".join(strip_attack_citations(str(item["description"])).split())
        if not text:
            continue
        if technique not in labels.setdefault(text, []):
            labels[text].append(technique)
        origins.setdefault(text, set()).add(str(source.get("type")))
    return [
        {
            "case_id": f"attack_procedures-{index:06d}",
            "dataset": "attack_procedures",
            "split": "attack_train",
            "text": text,
            "gold_ids": ids,
            "candidates": [],
            "gold_in_candidates": False,
            "gold_family_in_candidates": False,
            "pipeline_rank1": None,
            "jev_c1": None,
            "source_types": sorted(origins[text]),
        }
        for index, (text, ids) in enumerate(sorted(labels.items()), start=1)
    ]


def _runs(sequence: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(sequence[index : index + size]) for index in range(len(sequence) - size + 1)}


def drop_overlaps(
    rows: Sequence[dict[str, Any]], held_out_texts: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """``rows`` minus those that contain, are contained in, or share a long run with a held-out
    text (the near-duplicate pass runs first, in ``drop_near_duplicates``)."""
    held = [tokens(text) for text in held_out_texts]
    shared = set().union(*(_runs(seq, SHARED_RUN) for seq in held)) if held else set()
    contained = {
        tuple(seq) for seq in held if len(seq) >= MIN_CONTAINED
    }  # held-out texts that may sit inside a procedure
    lengths = sorted({len(seq) for seq in contained})
    kept, counts = [], {"held_out_contained": 0, "contained_in_held_out": 0, "shared_run": 0}
    held_joined = [" " + " ".join(seq) + " " for seq in held]
    for row in rows:
        seq = tokens(row["text"])
        runs = {
            tuple(seq[index : index + size])
            for size in lengths
            if size <= len(seq)
            for index in range(len(seq) - size + 1)
        }
        if runs & contained:
            counts["held_out_contained"] += 1
            continue
        if len(seq) >= MIN_CONTAINED:
            needle = " " + " ".join(seq) + " "
            if any(needle in text for text in held_joined):
                counts["contained_in_held_out"] += 1
                continue
        if _runs(seq, SHARED_RUN) & shared:
            counts["shared_run"] += 1
            continue
        kept.append(row)
    return kept, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    bundle = _load(settings.attack_bundle_path)
    catalog = build_catalog(bundle, kb)
    rows = procedure_rows(bundle, catalog)
    held_out = [
        row["text"] for split in HELD_OUT for row in read_jsonl(args.out / f"cases_{split}.jsonl")
    ]
    total = len(rows)
    rows, near = drop_near_duplicates(rows, held_out)
    rows, overlaps = drop_overlaps(rows, held_out)
    path = args.out / "cases_attack_train.jsonl"
    write_jsonl(path, rows)
    families = {item.split(".")[0] for row in rows for item in row["gold_ids"]}
    info = {
        "source": str(settings.attack_bundle_path),
        "attack_version": kb.version,
        "procedure_texts": total,
        "dropped": {**near, **overlaps},
        "cases": len(rows),
        "families": len(families),
        "held_out_splits": list(HELD_OUT),
        "rules": {
            "near_duplicate_jaccard": 0.9,
            "min_contained_tokens": MIN_CONTAINED,
            "shared_run_tokens": SHARED_RUN,
        },
    }
    manifest_path = args.out / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("splits", {})["attack_train"] = info
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{path}: {json.dumps(info)}")


if __name__ == "__main__":
    main()
