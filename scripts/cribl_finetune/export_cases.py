"""Write the cribl-decision experiment's case files (project .venv; README.md in this folder).

Each split becomes ``<out>/cases_<split>.jsonl`` with rows ``{case_id, dataset, split, text,
gold_ids, candidates [{id, label, what}], gold_in_candidates, pipeline_rank1, jev_c1}``:

- ``dev`` (public tram/expert dev, 659) and ``cti_hal`` (968 statements): text, gold and the exact
  candidate lists Jev C1 was asked about, with Jev's stored answer, from the Jev framing artifacts.
- ``test`` (public tram/expert/derived_procedures test): the pipeline's top-10 from a public
  held-out artifact (no Jev answer exists there).
- ``train`` (public train splits): the top-10 from ``benchmark_mapping.py --split train`` run with
  ``TECHNIQUE_EXEMPLARS_ENABLED=false`` (the exemplar index holds the train sentences, so with it
  the answer would leak into the candidates); exact and near duplicates of dev, test and CTI-HAL
  texts are dropped.

Options use the Jev C1 rules (``common.py``): catalogue techniques only, deduplicated, in rank
order, the first 10; label ``Name (Txxxx)``; ``what`` = the description cut at 400 characters.
CTI-HAL text has no published licence: the output stays under the gitignored ``data/audits``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from common import FLAT_K, parent_of, technique_option, write_jsonl  # noqa: E402

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.attack.parse import strip_attack_citations  # noqa: E402
from apt_classifier.config import get_settings  # noqa: E402
from scripts.train_technique_reranker import _near_duplicate, _normalized_tokens  # noqa: E402

JEV_FRAMING = Path(
    "/home/fzagar/apt_classifier/.claude/worktrees/jev-framing-harness-ae6b44/data/audits/jev/framing"
)
MAIN_AUDITS = Path("/home/fzagar/apt_classifier/data/audits")
OUT = REPO_ROOT / "data" / "audits" / "cribl_ft"
NEAR_DUPLICATE = 0.9


def clean_text(text: Any) -> str:
    """ATT&CK text without citation markers or markdown link syntax, on one line."""
    return " ".join(strip_attack_citations(str(text or "")).split())


def _attack_id(item: Mapping[str, Any]) -> str | None:
    for reference in item.get("external_references") or ():
        if isinstance(reference, Mapping) and reference.get("source_name") == "mitre-attack":
            return str(reference.get("external_id") or "") or None
    return None


def _active(item: Mapping[str, Any]) -> bool:
    return not item.get("revoked") and not item.get("x_mitre_deprecated")


def build_catalog(bundle: Mapping[str, Any], kb: Any) -> dict[str, dict[str, str]]:
    """Technique id -> {name, description} for the techniques the Jev judge lists
    (``TacticCatalog._build`` at f9b452c): active parents under an active tactic, and their
    active sub-techniques."""
    tactics: set[str] = set()
    inactive: set[str] = set()
    for item in bundle.get("objects") or ():
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "attack-pattern":
            attack_id = _attack_id(item)
            if attack_id and not _active(item):
                inactive.add(attack_id)
        elif item.get("type") == "x-mitre-tactic" and _active(item):
            shortname = str(item.get("x_mitre_shortname") or "").strip()
            if shortname:
                tactics.add(shortname)
    parents: set[str] = set()
    children: dict[str, set[str]] = {}
    records: dict[str, Any] = {}
    for key, technique in kb.techniques.items():
        technique_id = str(getattr(technique, "id", "") or key)
        if technique_id in inactive:
            continue
        records[technique_id] = technique
        if getattr(technique, "is_subtechnique", False) or "." in technique_id:
            parent = str(getattr(technique, "parent_id", "") or technique_id.split(".")[0])
            children.setdefault(parent, set()).add(technique_id)
        elif any(shortname in tactics for shortname in getattr(technique, "tactics", ()) or ()):
            parents.add(technique_id)
    listed = [*sorted(parents)]
    for parent in sorted(parents):
        listed += sorted(children.get(parent, ()))
    return {
        technique_id: {
            "name": clean_text(getattr(records[technique_id], "name", "")) or technique_id,
            "description": clean_text(getattr(records[technique_id], "description", "")),
        }
        for technique_id in listed
    }


def known_candidates(catalog: Mapping[str, Any], ranked: Iterable[Any]) -> list[str]:
    cleaned = (str(item).strip() for item in ranked or ())
    return list(dict.fromkeys(item for item in cleaned if item in catalog))[:FLAT_K]


def case_row(
    case: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, str]],
    split: str,
    candidate_ids: Sequence[str],
    jev: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gold = [str(item) for item in case.get("expected_technique_ids") or ()]
    ranked = [str(item) for item in case.get("ranked_technique_ids") or ()]
    row = {
        "case_id": str(case["case_id"]),
        "dataset": str(case.get("dataset") or ""),
        "split": split,
        "text": str(case.get("text") or ""),
        "gold_ids": gold,
        "candidates": [
            technique_option(item, catalog[item]["name"], catalog[item]["description"])
            for item in candidate_ids
        ],
        "gold_in_candidates": bool(set(gold) & set(candidate_ids)),
        "gold_family_in_candidates": bool(
            ({*gold, *(parent_of(item) for item in gold)})
            & {*candidate_ids, *(parent_of(item) for item in candidate_ids)}
        ),
        "pipeline_rank1": ranked[0] if ranked else None,
        "jev_c1": None,
    }
    if jev is not None:
        path = jev.get("path") or {}
        row["jev_c1"] = {
            "status": jev.get("status"),
            "pick": jev.get("pick"),
            "p_final": jev.get("p_final"),
            "distribution": path.get("final_distribution"),
            "model": jev.get("model"),
        }
    return row


def _cases(artifact: Mapping[str, Any], key: str = "hybrid") -> list[dict[str, Any]]:
    return list(artifact["results"][key]["cases"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def jev_split(
    artifact: Mapping[str, Any], catalog: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], int]:
    """Rows from a Jev framing artifact; counts the cases whose stored C1 candidate list differs
    from this catalogue's filter of the ranked list (expected 0: same catalogue rules)."""
    rows, mismatches = [], 0
    for case in _cases(artifact):
        record = (case.get("jev_judge_variants") or {}).get("C1")
        offered = [str(item) for item in ((record or {}).get("path") or {}).get("candidates") or ()]
        mine = known_candidates(catalog, case.get("ranked_technique_ids"))
        if record is not None and offered != mine:
            mismatches += 1
        rows.append(case_row(case, catalog, split, offered or mine, record))
    return rows, mismatches


def drop_near_duplicates(
    rows: Sequence[dict[str, Any]], held_out_texts: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Train rows minus exact or near (token Jaccard >= 0.9) duplicates of any held-out text,
    and minus repeats within train (first kept)."""
    held = [_normalized_tokens(text) for text in held_out_texts]
    held_exact = set(held)
    kept: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    counts = {"held_out_duplicates": 0, "train_repeats": 0}
    for row in rows:
        tokens = _normalized_tokens(row["text"])
        if tokens in held_exact or any(
            _near_duplicate(tokens, other, NEAR_DUPLICATE) for other in held
        ):
            counts["held_out_duplicates"] += 1
            continue
        if tokens in seen:
            counts["train_repeats"] += 1
            continue
        seen.add(tokens)
        kept.append(row)
    return kept, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--jev-framing", type=Path, default=JEV_FRAMING)
    parser.add_argument(
        "--public-test",
        type=Path,
        default=MAIN_AUDITS / "refresh_2026-10-07" / "public_heldout_current.json",
    )
    parser.add_argument("--train", type=Path, default=OUT / "train_candidates.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    catalog = build_catalog(_load(settings.attack_bundle_path), kb)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "attack_version": kb.version,
        "catalog_size": len(catalog),
        "splits": {},
    }

    sources = {
        "dev": args.jev_framing / "s1_dev_c1.json",
        "cti_hal": args.jev_framing / "s3_cti_hal_statements.json",
        "test": args.public_test,
        "train": args.train,
    }
    held_out_texts: list[str] = []
    for split in ("dev", "cti_hal", "test", "train"):
        artifact = _load(sources[split])
        info: dict[str, Any] = {
            "source": str(sources[split]),
            "source_sha256": _sha256(sources[split]),
            "source_split": artifact.get("split"),
            "source_case_set_hash": artifact.get("case_set_hash"),
            "source_attack_version": artifact.get("attack_version"),
            "exemplar_status": artifact.get("exemplar_status"),
            "acceptance_model_status": artifact.get("acceptance_model_status"),
        }
        if split in ("dev", "cti_hal"):
            rows, mismatches = jev_split(artifact, catalog, split)
            info["candidate_mismatches_vs_jev"] = mismatches
        else:
            rows = [
                case_row(
                    case, catalog, split, known_candidates(catalog, case["ranked_technique_ids"])
                )
                for case in _cases(artifact)
            ]
        if split == "train":
            if artifact.get("split") != "train":
                raise SystemExit(f"{sources[split]} is not a train-split artifact")
            rows, info["dedup"] = drop_near_duplicates(rows, held_out_texts)
        else:
            held_out_texts += [row["text"] for row in rows]
        info["cases"] = write_jsonl(args.out / f"cases_{split}.jsonl", rows)
        info["gold_in_candidates"] = round(
            sum(row["gold_in_candidates"] for row in rows) / max(len(rows), 1), 6
        )
        info["datasets"] = {
            name: sum(1 for row in rows if row["dataset"] == name)
            for name in sorted({row["dataset"] for row in rows})
        }
        manifest["splits"][split] = info
        print(
            f"{split}: {info['cases']} cases, gold in candidates {info['gold_in_candidates']:.3f}"
            + (f", dedup {info['dedup']}" if "dedup" in info else "")
            + (
                f", candidate mismatches vs Jev {info['candidate_mismatches_vs_jev']}"
                if "candidate_mismatches_vs_jev" in info
                else ""
            )
        )
    (args.out / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
