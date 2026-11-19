"""The group head's catalogue and case files: technique lists -> threat group (project .venv;
README.md here; prereg ``cribl_group_head_2026-10-15.json``).

A case's text is the list of ATT&CK techniques extracted from one report, "id name" entries joined
by "; " and nothing else: no prose, no actor or tool names. The catalogue lists every ATT&CK group
with at least ``MIN_TECHNIQUES`` documented techniques as its own family (``id == family``), so
``family_head.py train --task group --catalog group_catalog.json`` trains a group head with the
family-head code unchanged.

  build -> data/audits/cribl_ft/group_catalog.json and
    cases_group_train.jsonl        synthetic draws from each group's documented techniques, after
                                   ``group_mlp.training_cases`` without the software and context
                                   parts: 3-20 true techniques, false positives by technique
                                   popularity at precision 0.4-0.95, a sub-technique replaced by
                                   its parent with p 0.2, order shuffled (dataset group_synthetic)
    cases_group_campaigns.jsonl    dev: ATT&CK campaigns attributed to one listed group, 3+ techniques
    cases_group_vendor.jsonl       dev: the vendor reports, the techniques the reference model names
                                   on their statements (they carry no technique labels)
    cases_group_cti_hal_gold.jsonl       test: CTI-HAL reports, the labelled technique ids
    cases_group_cti_hal_extracted.jsonl  test (headline input): the techniques the reference
                                   technique model names on the report's statements
    cases_group_cti_hal_accepted.jsonl   test: the techniques the pipeline accepted (software dropped)
    cases_group_elastic.jsonl      test: the Elastic reports with labelled techniques
Every row also carries ``techniques`` (the ids) for the baselines (``group_baseline.py``). The
CTI-HAL and Elastic files are local only. ATT&CK's group profiles cite some CTI-HAL and vendor
reports, so gold-input numbers flatter every model alike; counts go to cases_manifest.json["group"].
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

MIN_TECHNIQUES = 5
DRAW = {
    "samples_per_group": 300,
    "size_min": 3,
    "size_max": 20,
    "precision_min": 0.40,
    "precision_max": 0.95,
    "parent_rate": 0.20,
    "seed": 7,
}
MIN_CAMPAIGN_TECHNIQUES = 3
# the attribution benchmark's CTI-HAL run lives in the main checkout's audits (export_cases.MAIN_AUDITS)
ATTRIBUTION_RUN = next(
    (
        path
        for path in (
            REPO_ROOT / "data" / "audits" / "attribution_cti_hal_current.json",
            Path("/home/fzagar/apt_classifier/data/audits/attribution_cti_hal_current.json"),
        )
        if path.is_file()
    ),
    REPO_ROOT / "data" / "audits" / "attribution_cti_hal_current.json",
)


def technique_labels(kb: Any) -> dict[str, str]:
    """technique id -> "id Parent name: name" (sub-techniques) or "id name"."""
    out = {}
    for technique_id, technique in kb.techniques.items():
        name = str(getattr(technique, "name", "") or technique_id)
        parent_id = getattr(technique, "parent_id", None)
        parent = kb.techniques.get(parent_id) if parent_id else None
        if parent is not None:
            name = f"{getattr(parent, 'name', parent_id)}: {name}"
        out[technique_id] = f"{technique_id} {name}"
    return out


def case_text(techniques: Iterable[str], labels: Mapping[str, str]) -> str:
    """The technique list as the model reads it (order as given; callers shuffle or sort)."""
    return "; ".join(labels.get(t, t) for t in techniques)


def group_catalog(kb: Any) -> list[dict[str, str]]:
    """One catalogue item per group with ``MIN_TECHNIQUES`` documented techniques, its own family."""
    from common import cut_description

    items = []
    for group_id, group in sorted(kb.groups.items()):
        techniques = set(kb.group_techniques.get(group_id, ()) or group.technique_ids)
        if len(techniques) < MIN_TECHNIQUES:
            continue
        items.append(
            {
                "id": group_id,
                "family": group_id,
                "label": f"{group.name} ({group_id})",
                "what": cut_description(group.description),
            }
        )
    return items


def draw_cases(
    profiles: Mapping[str, Sequence[str]],
    parents: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
) -> list[tuple[str, list[str]]]:
    """Synthetic (group, technique ids) draws: ``size_min``-``size_max`` of the group's own
    techniques, false positives drawn by how many groups document a technique at a precision of
    ``precision_min``-``precision_max``, each sub-technique replaced by its parent with
    ``parent_rate``, order shuffled. Seeded, so the file is reproducible."""
    p = {**DRAW, **(params or {})}
    rng = random.Random(int(p["seed"]))
    popularity = Counter(t for techniques in profiles.values() for t in set(techniques))
    universe = sorted(popularity)
    out: list[tuple[str, list[str]]] = []
    for group_id in sorted(profiles):
        own = sorted(set(profiles[group_id]))
        others = [t for t in universe if t not in set(own)]
        weights = [popularity[t] for t in others]
        for _ in range(int(p["samples_per_group"])):
            size = rng.randint(int(p["size_min"]), int(p["size_max"]))
            kept = rng.sample(own, min(size, len(own)))
            precision = rng.uniform(float(p["precision_min"]), float(p["precision_max"]))
            false_positives = round(len(kept) * (1 - precision) / precision)
            if false_positives and others:
                kept += rng.choices(others, weights=weights, k=false_positives)
            kept = [
                parents.get(t, t) if "." in t and rng.random() < float(p["parent_rate"]) else t
                for t in kept
            ]
            kept = list(dict.fromkeys(kept))
            rng.shuffle(kept)
            out.append((group_id, kept))
    return out


def case(
    case_id: str,
    split: str,
    dataset: str,
    group_id: str,
    techniques: Sequence[str],
    labels: Mapping[str, str],
    source: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "dataset": dataset,
        "split": split,
        "text": case_text(techniques, labels),
        "techniques": list(techniques),
        "gold_ids": [group_id],
        "candidates": [],
        "gold_in_candidates": False,
        "gold_family_in_candidates": False,
        "pipeline_rank1": None,
        "jev_c1": None,
        "source": source,
    }


def report_rows(
    reports: Iterable[Mapping[str, Any]],
    split: str,
    labels: Mapping[str, str],
    listed: set[str],
    techniques_of: Any,
    source: str,
) -> tuple[list[dict[str, Any]], Counter]:
    rows, drops = [], Counter()
    for report in reports:
        group_id = report.get("group_id")
        techniques = sorted(set(techniques_of(report)))
        if not group_id or group_id not in listed:
            drops["group_not_listed"] += 1
        elif not techniques:
            drops["no_techniques"] += 1
        else:
            rows.append(
                case(
                    f"{split}-{report['report_id']}",
                    split,
                    split,
                    group_id,
                    techniques,
                    labels,
                    source,
                )
            )
    return rows, drops


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extracted_per_report(
    reports: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """report id -> the technique answers of the technique model on the report's statements
    (statement rows joined to reports by their text; every prediction must find its report)."""
    owner: dict[str, str] = {}
    for report in reports:
        for statement in report["statements"]:
            owner.setdefault(_key(statement["text"]), report["report_id"])
    text_of = {row["case_id"]: row["text"] for row in cases}
    out: dict[str, list[str]] = {}
    missing = 0
    for row in predictions:
        if row.get("status") != "ok":
            continue
        report_id = owner.get(_key(text_of.get(row["case_id"], "")))
        if report_id is None:
            missing += 1
            continue
        out.setdefault(report_id, []).append(row["technique_pick"])
    if missing:
        raise SystemExit(f"{missing} predictions matched no report statement")
    return {report_id: sorted(set(picks)) for report_id, picks in out.items()}


def build(args: argparse.Namespace) -> None:
    from common import read_jsonl, write_jsonl
    from export_cases import OUT
    from prepare_advisories import excluded_names

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings

    kb = AttackKnowledgeBase.load(get_settings().kb_path)
    labels = technique_labels(kb)
    parents = {
        technique_id: technique.parent_id
        for technique_id, technique in kb.techniques.items()
        if getattr(technique, "parent_id", None)
    }
    catalog = group_catalog(kb)
    listed = {item["id"] for item in catalog}
    (OUT / "group_catalog.json").write_text(
        json.dumps({"attack_version": kb.version, "techniques": catalog}, indent=2) + "\n",
        encoding="utf-8",
    )
    profiles = {
        g: sorted(
            t
            for t in (kb.group_techniques.get(g, ()) or kb.groups[g].technique_ids)
            if t in kb.techniques
        )
        for g in listed
    }
    manifest_info: dict[str, Any] = {
        "attack_version": kb.version,
        "groups": len(catalog),
        "min_techniques": MIN_TECHNIQUES,
        "draw": DRAW,
        "leakage_note": "ATT&CK's group profiles cite some CTI-HAL and vendor reports; gold-input "
        "numbers flatter every model alike. Extracted and accepted inputs come from the "
        "technique stage, not the labels.",
    }
    files: dict[str, list[dict[str, Any]]] = {}

    draws = draw_cases(profiles, parents, {**DRAW, "seed": args.seed})
    files["group_train"] = [
        case(
            f"group_train-{i:06d}",
            "group_train",
            "group_synthetic",
            g,
            t,
            labels,
            "ATT&CK group profile, synthetic draw",
        )
        for i, (g, t) in enumerate(draws, start=1)
    ]

    campaigns = [
        {
            "report_id": cid,
            "group_id": c.group_ids[0],
            "techniques": [t for t in c.technique_ids if t in kb.techniques],
        }
        for cid, c in sorted(kb.campaigns.items())
        if len(c.group_ids) == 1
    ]
    files["group_campaigns"], drops_c = report_rows(
        [c for c in campaigns if len(c["techniques"]) >= MIN_CAMPAIGN_TECHNIQUES],
        "group_campaigns",
        labels,
        listed,
        lambda r: r["techniques"],
        "ATT&CK campaign",
    )
    # the vendor reports carry no technique labels: their dev input is what the technique model
    # names on their statements (cases_vendor_statements.jsonl -> <extractor>_vendor_statements.jsonl)
    vendor = read_jsonl(REPO_ROOT / "data/benchmarks/vendor/reports.jsonl")
    vendor_statements = OUT / "cases_vendor_statements.jsonl"
    if not vendor_statements.exists():
        write_jsonl(
            vendor_statements,
            [
                {
                    "case_id": f"vendor-{report['report_id']}-{number:03d}",
                    "dataset": "vendor",
                    "split": "vendor_statements",
                    "text": statement["text"],
                    "gold_ids": [],
                    "candidates": [],
                }
                for report in vendor
                for number, statement in enumerate(report["statements"], start=1)
                if statement.get("text", "").strip()
            ],
        )
    vendor_predictions = OUT / f"{args.extractor}_vendor_statements.jsonl"
    vendor_extracted = (
        extracted_per_report(vendor, read_jsonl(vendor_statements), read_jsonl(vendor_predictions))
        if vendor_predictions.exists()
        else {}
    )
    files["group_vendor"], drops_v = report_rows(
        vendor,
        "group_vendor",
        labels,
        listed,
        lambda r: [t for t in vendor_extracted.get(r["report_id"], ()) if t in kb.techniques],
        f"vendor report, techniques named by {args.extractor}",
    )
    cti_hal = read_jsonl(REPO_ROOT / "data/benchmarks/cti_hal/reports.jsonl")
    files["group_cti_hal_gold"], drops_g = report_rows(
        cti_hal,
        "group_cti_hal_gold",
        labels,
        listed,
        lambda r: [t for s in r["statements"] for t in s["technique_ids"] if t in kb.techniques],
        "CTI-HAL report, labelled techniques (local only)",
    )
    extracted = extracted_per_report(
        cti_hal,
        read_jsonl(OUT / "cases_cti_hal.jsonl"),
        read_jsonl(OUT / f"{args.extractor}_cti_hal.jsonl"),
    )
    files["group_cti_hal_extracted"], drops_e = report_rows(
        cti_hal,
        "group_cti_hal_extracted",
        labels,
        listed,
        lambda r: [t for t in extracted.get(r["report_id"], ()) if t in kb.techniques],
        f"CTI-HAL report, techniques named by {args.extractor} (local only)",
    )
    run = json.loads(ATTRIBUTION_RUN.read_text(encoding="utf-8"))["known"]
    files["group_cti_hal_accepted"], drops_a = report_rows(
        run,
        "group_cti_hal_accepted",
        labels,
        listed,
        lambda r: [t for t in r["accepted_techniques"] if t in kb.techniques],
        "CTI-HAL report, techniques the pipeline accepted (local only)",
    )
    elastic = read_jsonl(REPO_ROOT / "data/benchmarks/elastic/reports.jsonl")
    files["group_elastic"], drops_el = report_rows(
        elastic,
        "group_elastic",
        labels,
        listed,
        lambda r: [t for t in r["technique_ids"] if t in kb.techniques],
        "Elastic report, labelled techniques (local only)",
    )

    # every text is made of technique labels only, so one pass over the distinct labels in use
    # finds any actor name; one combined pattern keeps it fast
    names = excluded_names(kb, listed)
    pattern = re.compile(
        "(?<![\\w-])(?:"
        + "|".join(re.escape(name) for group_names in names.values() for name in group_names)
        + ")(?![\\w-])",
        re.I,
    )
    used_labels = {
        labels.get(t, t) for rows in files.values() for row in rows for t in row["techniques"]
    }
    flagged = sorted(label for label in used_labels if pattern.search(label))
    leaks = len(flagged)
    manifest_info["labels_with_actor_names"] = flagged[:20]
    manifest_path = OUT / "cases_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest_info["drops"] = {
        "campaigns": dict(drops_c),
        "vendor": dict(drops_v),
        "cti_hal_gold": dict(drops_g),
        "cti_hal_extracted": dict(drops_e),
        "cti_hal_accepted": dict(drops_a),
        "elastic": dict(drops_el),
    }
    manifest_info["actor_names_in_texts"] = leaks
    for name, rows in files.items():
        write_jsonl(OUT / f"cases_{name}.jsonl", rows)
        sizes = [len(row["techniques"]) for row in rows]
        manifest_info[name] = {
            "cases": len(rows),
            "groups": len({row["gold_ids"][0] for row in rows}),
            "techniques_per_case_median": sorted(sizes)[len(sizes) // 2] if sizes else None,
        }
        print(f"cases_{name}.jsonl: {manifest_info[name]}")
    manifest["group"] = manifest_info
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"group_catalog.json: {len(catalog)} groups; actor names in case texts: {leaks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--seed", type=int, default=DRAW["seed"])
    parser.add_argument(
        "--extractor",
        default="E2",
        help="the technique model whose CTI-HAL predictions give the extracted input",
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
