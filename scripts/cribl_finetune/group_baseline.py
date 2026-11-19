"""The existing group models on the group head's case files (project .venv; prereg
``cribl_group_head_2026-10-15.json``): the product's group network (``apt_classifier.group_mlp``,
techniques only, software block empty) and a Jaccard match to each group's documented technique
profile. Both write prediction files in the family head's format (``top``, ``pick``,
``gold_rank``), so ``parent_rescore.py`` compares them with the new model case by case.

  python group_baseline.py --splits group_cti_hal_extracted group_cti_hal_gold ...
  -> data/audits/cribl_ft/MLP_<split>.jsonl, JACCARD_<split>.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

TOP_KEPT = 20
DEFAULT_SPLITS = (
    "group_campaigns",
    "group_vendor",
    "group_cti_hal_gold",
    "group_cti_hal_extracted",
    "group_cti_hal_accepted",
    "group_elastic",
)


def ranked_row(row: Mapping[str, Any], scores: Mapping[str, float]) -> dict[str, Any]:
    """A prediction row from group -> score (higher is better; ties by group id)."""
    ranked = sorted(scores, key=lambda g: (-scores[g], g))
    position = {g: rank for rank, g in enumerate(ranked, start=1)}
    gold = [g for g in row["gold_ids"] if g in position]
    return {
        "case_id": row["case_id"],
        "dataset": row["dataset"],
        "gold_ids": row["gold_ids"],
        "gold_families": list(row["gold_ids"]),
        "status": "ok",
        "pick": ranked[0],
        "p_pick": float(scores[ranked[0]]),
        "top": [[g, round(float(scores[g]), 6)] for g in ranked[:TOP_KEPT]],
        "gold_rank": min((position[g] for g in gold), default=None),
        "gold_p": round(sum(float(scores[g]) for g in gold), 6),
    }


def jaccard_scores(
    techniques: Sequence[str], profiles: Mapping[str, frozenset[str]], parents: Mapping[str, str]
) -> dict[str, float]:
    """Jaccard between the case's techniques (with parents) and each group's documented profile."""
    mine = set(techniques) | {parents[t] for t in techniques if t in parents}
    return {
        g: len(mine & profile) / len(mine | profile) if mine | profile else 0.0
        for g, profile in profiles.items()
    }


def main() -> None:
    import numpy as np
    from common import read_jsonl, write_jsonl
    from export_cases import OUT

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.group_mlp import load_group_mlp
    from apt_classifier.profile_forest import group_profiles

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = parser.parse_args()
    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    model, status = load_group_mlp(
        settings.group_mlp_path, kb, device="cpu"
    )  # tiny; GPU may be busy
    if model is None:
        raise SystemExit(f"group network: {status}")
    print(f"group network: {status}; {len(model.group_ids)} groups")
    profiles = {g: p.techniques for g, p in group_profiles(kb).items()}
    parents = {
        t: technique.parent_id
        for t, technique in kb.techniques.items()
        if getattr(technique, "parent_id", None)
    }
    for split in args.splits:
        rows = read_jsonl(OUT / f"cases_{split}.jsonl")
        vectors = np.stack([model.features.vector(row["techniques"], []) for row in rows])
        probabilities = model.predict(vectors)
        mlp_rows = [
            ranked_row(row, dict(zip(model.group_ids, probabilities[i].tolist(), strict=True)))
            for i, row in enumerate(rows)
        ]
        jaccard_rows = [
            ranked_row(row, jaccard_scores(row["techniques"], profiles, parents)) for row in rows
        ]
        for name, out in (("MLP", mlp_rows), ("JACCARD", jaccard_rows)):
            write_jsonl(OUT / f"{name}_{split}.jsonl", out)
            top1 = sum(r["gold_rank"] == 1 for r in out) / len(out)
            top5 = sum(bool(r["gold_rank"]) and r["gold_rank"] <= 5 for r in out) / len(out)
            print(f"{name}_{split}: n {len(out)} top-1 {top1:.3f} top-5 {top5:.3f}")


if __name__ == "__main__":
    main()
