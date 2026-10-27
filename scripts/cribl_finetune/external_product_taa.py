"""The full attribution pipeline on CTIBench CTI-TAA and AthenaBench TAA: evaluation only (prereg
``product_external_taa_2026-10-27.json``). Both benchmarks are non-commercial; nothing is trained
or tuned on them and the converted reports stay under data/benchmarks/ (gitignored).

  prepare  -> data/benchmarks/{cti_bench,athenabench}/product_taa_reports.jsonl
           the texts external_bench.py prepared (anonymised, split), one report per text, gold =
           the ATT&CK groups alias-connected to the answer (exact name first, others accept;
           none = open set)
  (run)    scripts/attribution_benchmark.py --cti-hal <reports> --anonymize-actors
           --report-observations 0 --output <artifact>, one artifact per benchmark and arm
  score    <artifact> ... -> the benchmark's rule on the top-ranked group (always answers) and on
           the product's own names (held = no answer)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
BENCHMARKS = REPO_ROOT / "data" / "benchmarks"
SETS = {
    "ctibench_taa": BENCHMARKS / "cti_bench" / "product_taa_reports.jsonl",
    "athena_taa": BENCHMARKS / "athenabench" / "product_taa_reports.jsonl",
}
NAMED_STAGES = ("known_group_hypothesis", "known_cluster_hypothesis")


def gold_groups(
    answer: str, kb: Any, aliases: Mapping[str, set[str]]
) -> tuple[list[str], list[str]]:
    """ATT&CK groups whose own name or alias is the answer, and the others alias-connected to it
    (the benchmark's alias walk can link distinct groups, e.g. APT28 and APT29)."""
    from external_bench import connected

    wanted = answer.strip().lower()
    exact, linked = [], []
    for gid, group in sorted(kb.groups.items()):
        names = [n.strip().lower() for n in (group.name, *group.aliases) if n and n.strip()]
        if wanted in names:
            exact.append(gid)
        elif any(connected(answer, n, aliases) for n in names):
            linked.append(gid)
    return exact, linked


def prepare(_: argparse.Namespace) -> None:
    from common import read_jsonl, write_jsonl
    from external_bench import alias_graph

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings

    kb = AttackKnowledgeBase.load(get_settings().kb_path)
    aliases, _ = alias_graph(kb)
    for name, path in SETS.items():
        by_report: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for case in read_jsonl(AUDITS / f"cases_ext_{name}.jsonl"):
            by_report[case["report_id"]].append(case)
        rows = []
        for report_id, items in sorted(by_report.items()):
            answer = items[0]["gold_actor"]
            exact, linked = gold_groups(answer, kb, aliases)
            groups = exact + linked
            rows.append(
                {
                    "report_id": report_id,
                    "group_id": groups[0] if groups else None,
                    "accept": groups[1:],
                    "gold_actor": answer,
                    "exact": exact,
                    "technique_ids": [],
                    "statements": [{"text": i["text"], "technique_ids": []} for i in items],
                }
            )
        write_jsonl(path, rows)
        mapped = sum(1 for r in rows if r["group_id"])
        print(f"{path}: {len(rows)} reports, {mapped} map to an ATT&CK group")


def _ranked(row: Mapping[str, Any]) -> list[str]:
    ranked = [str(item["group_id"]) for item in row.get("topk") or () if item.get("group_id")]
    leader = row.get("leading")
    if leader and (not ranked or ranked[0] != leader):
        ranked = [str(leader), *[g for g in ranked if g != leader]]
    return ranked


def score_artifact(path: Path, kb: Any, aliases: Any, related: Any) -> dict[str, Any]:
    from external_bench import connected, taa_block, taa_score

    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = list(artifact.get("known") or []) + list(artifact.get("open_set") or [])
    name = next(
        n
        for n in SETS
        if any(
            str(r.get("report_id", "")).startswith(
                "ctib_taa" if n == "ctibench_taa" else "athena_taa"
            )
            for r in rows
        )
    )
    gold = {
        r["report_id"]: r
        for r in (json.loads(line) for line in SETS[name].read_text().splitlines() if line)
    }
    label = {gid: g.name for gid, g in kb.groups.items()}
    top1, named_scores, details = [], [], []
    hits = {3: 0, 5: 0}
    mapped: list[int] = []
    names_given = names_correct = strict = 0
    for row in sorted(rows, key=lambda r: r["report_id"]):
        answer = gold[row["report_id"]]["gold_actor"]
        exact = set(gold[row["report_id"]]["exact"])
        ranked = _ranked(row)
        strict += bool(ranked) and ranked[0] in exact
        lead = label.get(ranked[0], "") if ranked else ""
        top1.append(taa_score(answer, lead, aliases, related))
        for k in hits:
            hits[k] += any(connected(answer, label.get(g, ""), aliases) for g in ranked[:k])
        stage = str(row.get("stage") or "")
        named: list[str] = []
        if stage == "known_group_hypothesis" and ranked:
            named = [ranked[0]]
        elif stage == "known_cluster_hypothesis":
            named = [str(g) for g in row.get("named_cluster") or ()]
        if named:
            names_given += 1
            best = max(
                (taa_score(answer, label.get(g, ""), aliases, related) for g in named),
                key=lambda s: s["combined"],
            )
            names_correct += int(best["correct"])
        else:
            best = {"correct": 0.0, "plausible": 0.0, "combined": 0.0}
        named_scores.append(best)
        mapped.append(int(row.get("group_id") is not None))
        details.append(
            {
                "report": row["report_id"],
                "answer": answer,
                "top1": lead,
                "top3": [label.get(g, g) for g in ranked[:3]],
                "stage": stage,
                "named": [label.get(g, g) for g in named],
                "leader_probability": row.get("combiner_leader_probability"),
                "top1_score": top1[-1]["combined"],
            }
        )
    n = len(rows) or 1
    on_mapped = [s for s, m in zip(top1, mapped, strict=True) if m]
    return {
        "artifact": str(path),
        "benchmark": name,
        "reports": len(rows),
        "answers_mapping_to_an_attack_group": sum(mapped),
        "top_ranked": taa_block(top1),
        "top1_exact_group": round(strict / n, 4),
        "top3_correct": round(hits[3] / n, 4),
        "top5_correct": round(hits[5] / n, 4),
        "top_ranked_on_mapped": taa_block(on_mapped),
        "named": {
            **taa_block(named_scores),
            "names_given": names_given,
            "names_correct": names_correct,
        },
        "details": details,
    }


def score(args: argparse.Namespace) -> None:
    from external_bench import alias_graph

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings

    kb = AttackKnowledgeBase.load(get_settings().kb_path)
    aliases, related = alias_graph(kb)
    report = [score_artifact(Path(p), kb, aliases, related) for p in args.artifacts]
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in report:
        print(
            Path(item["artifact"]).name,
            {k: v for k, v in item.items() if k not in ("details", "artifact")},
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    commands.add_parser("prepare").set_defaults(func=prepare)
    s = commands.add_parser("score")
    s.set_defaults(func=score)
    s.add_argument("artifacts", nargs="+")
    s.add_argument("--output")
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
