"""CTIBench and AthenaBench on the fine-tuned heads: evaluation only (prereg
``cribl_external_bench_2026-10-19.json``). Both benchmarks are non-commercial (CTIBench CC BY-NC-SA
4.0; AthenaBench's academic research licence); nothing here is trained on them or redistributed.
Raw files: data/benchmarks/cti_bench/raw and data/benchmarks/athenabench/raw (local).

  prepare  (project .venv) -> data/audits/cribl_ft/cases_ext_*.jsonl
    ctibench_ate   60 ATT&CK software descriptions split into sentences (the product's splitter);
                   gold = the parent technique ids of the row (6.6 per text)
    athena_ate     500 GPT-written scenarios, whole; gold = the answer's parent technique
    ctibench_taa   50 anonymised reports split into sentences; gold actor name from the key file
    athena_taa     100 bullet profiles split into bullets; gold actor name
  predict  (.venv-cribl) --technique <E2 dir> --actor <T1 dir>
    -> E2 predictions on the ATE rows (technique head), T1 on the TAA rows (160 groups kept)
  score    (project .venv) -> data/audits/cribl_ft/external_bench.json
    ATE: CTIBench F1 (micro and per-text mean) at parent level over the union of the sentences'
         top-1 techniques, also with up to three per sentence (multi_answer, 1.3 per sentence);
         Enterprise rows and all rows (the catalogue is Enterprise only). AthenaBench accuracy at
         parent level of the top-1 technique (the benchmark's rule), plus top-3/top-5.
    TAA: per report, the mean of each group's probability over the sentences (the actor head) and
         the product's TF-IDF text-group model on the whole text; the predicted ATT&CK group's name
         scored with AthenaBench's rule: correct when alias-connected to the answer (their alias list
         plus ATT&CK's), plausible when connected through their related-group list, combined
         = 1 / 0.5 / 0. Coverage: how many answers map to any ATT&CK group at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
CTIB = Path("/home/fzagar/apt_classifier/data/benchmarks/cti_bench/raw")
ATHENA = Path("/home/fzagar/apt_classifier/data/benchmarks/athenabench/raw")
TECHNIQUE_ID = re.compile(r"T\d{4}(?:\.\d{3})?")
ATE_BUDGET = 1.3


def parent_of(technique: str) -> str:
    return technique.split(".")[0]


def row(case_id: str, split: str, text: str, gold: Sequence[str], **extra: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "dataset": split,
        "split": split,
        "text": text,
        "gold_ids": list(gold),
        "candidates": [],
        "gold_in_candidates": False,
        "gold_family_in_candidates": False,
        "pipeline_rank1": None,
        "jev_c1": None,
        **extra,
    }


def bullets(text: str) -> list[str]:
    parts = re.split(r"(?:^|\s)-\s+", str(text or "").strip())
    return [p.strip() for p in parts if len(p.split()) >= 3]  # a bullet of 3+ words


# ----------------------------------------------------------------------------- prepare
def prepare(args: argparse.Namespace) -> None:
    from common import write_jsonl
    from export_actor_cases import anonymiser

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.evidence.local_extract import local_observations

    kb = AttackKnowledgeBase.load(get_settings().kb_path)
    clean = anonymiser(kb)
    files: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("ctibench_ate", "athena_ate", "ctibench_taa", "athena_taa")
    }

    with (CTIB / "cti-ate.tsv").open(encoding="utf-8") as handle:
        for number, item in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
            gold = sorted({parent_of(t) for t in TECHNIQUE_ID.findall(item["GT"])})
            sentences = [o.text for o in local_observations(item["Description"])] or [
                item["Description"]
            ]
            for k, sentence in enumerate(sentences, start=1):
                files["ctibench_ate"].append(
                    row(
                        f"ctib_ate-{number:03d}-{k:02d}",
                        "ctibench_ate",
                        sentence,
                        gold,
                        report_id=f"ctib_ate-{number:03d}",
                        platform=item["Platform"],
                    )
                )
    for number, line in enumerate(
        (ATHENA / "athena-cti-ate.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        files["athena_ate"].append(
            row(
                f"athena_ate-{number:03d}",
                "athena_ate",
                item["scenario"],
                [parent_of(item["answer"])],
                answer=item["answer"],
            )
        )
    with (CTIB / "cti-taa-responses.tsv").open(encoding="utf-8") as handle:
        key = [r["GT"].strip() for r in csv.DictReader(handle, delimiter="\t")]
    with (CTIB / "cti-taa.tsv").open(encoding="utf-8") as handle:
        reports = list(csv.DictReader(handle, delimiter="\t"))
    if len(key) != len(reports):
        raise SystemExit(f"CTIBench TAA: {len(reports)} reports, {len(key)} answers")
    for number, (item, answer) in enumerate(zip(reports, key, strict=True), start=1):
        text = clean(item["Text"].replace("[PLACEHOLDER]", "the adversary"))
        sentences = [o.text for o in local_observations(text)] or [text]
        for k, sentence in enumerate(sentences, start=1):
            files["ctibench_taa"].append(
                row(
                    f"ctib_taa-{number:03d}-{k:03d}",
                    "ctibench_taa",
                    sentence,
                    [],
                    report_id=f"ctib_taa-{number:03d}",
                    gold_actor=answer,
                )
            )
    for number, line in enumerate(
        (ATHENA / "athena-cti-taa.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        for k, bullet in enumerate(bullets(clean(item["description"])), start=1):
            files["athena_taa"].append(
                row(
                    f"athena_taa-{number:03d}-{k:03d}",
                    "athena_taa",
                    bullet,
                    [],
                    report_id=f"athena_taa-{number:03d}",
                    gold_actor=item["answer"],
                )
            )
    for name, rows in files.items():
        write_jsonl(AUDITS / f"cases_ext_{name}.jsonl", rows)
        reports_n = len({r.get("report_id", r["case_id"]) for r in rows})
        print(f"cases_ext_{name}.jsonl: {len(rows)} rows, {reports_n} texts")


# ----------------------------------------------------------------------------- predict
def predict(args: argparse.Namespace) -> None:
    import family_head
    from common import read_jsonl
    from decision_model import with_oom_retry
    from family_head import load_checkpoint, load_techniques, run_split

    def run(
        encoder: Any, head: Any, families: Any, cases: Any, path: Path, techniques: Any
    ) -> None:
        with_oom_retry(
            lambda: run_split(encoder, head, families, cases, path, 500, args.batch, techniques),
            note=path.name,
        )
        print(f"{path.name}: {len(cases)} rows")

    _, encoder, head, info = load_checkpoint(args.technique)
    techniques = load_techniques(args.technique, encoder.device)
    for name in ("ctibench_ate", "athena_ate"):
        cases = read_jsonl(AUDITS / f"cases_ext_{name}.jsonl")
        run(
            encoder,
            head,
            info["families"],
            cases,
            AUDITS / f"{args.technique_name}_ext_{name}.jsonl",
            techniques,
        )
    family_head.TOP_KEPT = 160
    _, encoder, head, info = load_checkpoint(args.actor)
    for name in ("ctibench_taa", "athena_taa"):
        cases = read_jsonl(AUDITS / f"cases_ext_{name}.jsonl")
        run(
            encoder,
            head,
            info["families"],
            cases,
            AUDITS / f"{args.actor_name}_ext_{name}.jsonl",
            None,
        )


# ----------------------------------------------------------------------------- score
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ate_sets(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, tuple[set[str], set[str]]]:
    """text id -> (gold parents, predicted parents from the sentences' answers)."""
    from multi_answer import answers

    gold: dict[str, set[str]] = {}
    predicted: dict[str, set[str]] = defaultdict(set)
    report_of = {c["case_id"]: c["report_id"] for c in cases}
    for c in cases:
        gold[c["report_id"]] = set(c["gold_ids"])
    for p in predictions:
        if p.get("status") != "ok":
            continue
        for family, technique in answers(p, threshold):
            predicted[report_of[p["case_id"]]].add(parent_of(technique or family))
    return {t: (gold[t], predicted.get(t, set())) for t in gold}


def ate_scores(sets: Mapping[str, tuple[set[str], set[str]]]) -> dict[str, float]:
    tp = fp = fn = 0
    per_text = []
    for g, p in sets.values():
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        per_text.append(prf(len(g & p), len(p - g), len(g - p))[2])
    mp, mr, mf = prf(tp, fp, fn)
    return {
        "texts": len(sets),
        "micro_precision": round(mp, 4),
        "micro_recall": round(mr, 4),
        "micro_f1": round(mf, 4),
        "mean_f1": round(sum(per_text) / len(per_text), 4) if per_text else 0.0,
        "predicted_per_text": round(sum(len(p) for _, p in sets.values()) / len(sets), 2),
    }


def alias_graph(kb: Any) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """AthenaBench's alias and related-group lists, both directions, plus ATT&CK's own names and
    aliases (so a group's ATT&CK name connects to the benchmark's name for it)."""
    aliases: dict[str, set[str]] = defaultdict(set)
    related: dict[str, set[str]] = defaultdict(set)
    with (ATHENA / "taa" / "aliases.csv").open(encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            a, b = item["ThreatActor"].strip().lower(), item["Alias"].strip().lower()
            if a and b:
                aliases[a].add(b)
                aliases[b].add(a)
    with (ATHENA / "taa" / "related_groups.csv").open(encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            a, b = item["ThreatActor"].strip().lower(), item["RelatedGroup"].strip().lower()
            if a and b:
                related[a].add(b)
                related[b].add(a)
    for group in kb.groups.values():
        names = [n.strip().lower() for n in (group.name, *group.aliases) if n and n.strip()]
        for n in names:
            for m in names:
                if n != m:
                    aliases[n].add(m)
    return aliases, related


def connected(
    a: str, b: str, graph: Mapping[str, set[str]], extra: Mapping[str, set[str]] | None = None
) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    seen, queue = {a}, deque([a])
    while queue:
        cur = queue.popleft()
        for nxt in set(graph.get(cur, ())) | set((extra or {}).get(cur, ())):
            if nxt == b:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def taa_score(answer: str, predicted_name: str, aliases: Any, related: Any) -> dict[str, float]:
    if connected(answer, predicted_name, aliases):
        return {"correct": 1.0, "plausible": 1.0, "combined": 1.0}
    if connected(answer, predicted_name, aliases, related):
        return {"correct": 0.0, "plausible": 1.0, "combined": 0.5}
    return {"correct": 0.0, "plausible": 0.0, "combined": 0.0}


def taa_block(scored: Sequence[Mapping[str, float]]) -> dict[str, float]:
    n = len(scored) or 1
    return {
        "reports": len(scored),
        "correct": round(sum(s["correct"] for s in scored) / n, 4),
        "plausible": round(sum(s["plausible"] for s in scored) / n, 4),
        "combined": round(sum(s["combined"] for s in scored) / n, 4),
    }


def score(args: argparse.Namespace) -> None:
    from common import read_jsonl
    from multi_answer import budget_threshold

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import get_settings
    from apt_classifier.text_group import TextGroupModel

    settings = get_settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    report: dict[str, Any] = {
        "technique_model": args.technique_name,
        "actor_model": args.actor_name,
        "ate": {},
        "taa": {},
    }

    # ATE, CTIBench
    cases = read_jsonl(AUDITS / "cases_ext_ctibench_ate.jsonl")
    predictions = read_jsonl(AUDITS / f"{args.technique_name}_ext_ctibench_ate.jsonl")
    import math

    platform_of = {c["report_id"]: c["platform"] for c in cases}
    for label, threshold in (
        ("one_per_sentence", math.inf),
        (f"up_to_3_at_{ATE_BUDGET}", budget_threshold(predictions, ATE_BUDGET)),
    ):
        sets = ate_sets(cases, predictions, threshold)
        enterprise = {t: v for t, v in sets.items() if platform_of[t] == "Enterprise"}
        report["ate"][f"ctibench_{label}"] = {
            "all": ate_scores(sets),
            "enterprise": ate_scores(enterprise),
        }
    # ATE, AthenaBench
    cases = read_jsonl(AUDITS / "cases_ext_athena_ate.jsonl")
    predictions = {
        p["case_id"]: p for p in read_jsonl(AUDITS / f"{args.technique_name}_ext_athena_ate.jsonl")
    }
    hits = Counter()
    for c in cases:
        p = predictions.get(c["case_id"])
        if not p or p.get("status") != "ok":
            continue
        gold = c["gold_ids"][0]
        ranked_parents = []
        for technique, _ in p["technique_top"]:
            parent = parent_of(technique)
            if parent not in ranked_parents:
                ranked_parents.append(parent)
        hits["top1"] += parent_of(p["technique_pick"]) == gold
        hits["top3"] += gold in ranked_parents[:3]
        hits["top5"] += gold in ranked_parents[:5]
        hits["family_top1"] += p["pick"] == gold
    n = len(cases)
    report["ate"]["athena"] = {
        "scenarios": n,
        **{k: round(v / n, 4) for k, v in hits.items()},
        "errors": sum(1 for c in cases if predictions.get(c["case_id"], {}).get("status") != "ok"),
    }

    # TAA
    aliases, related = alias_graph(kb)
    tfidf = TextGroupModel.load(settings.text_group_path, kb)
    groups = [
        c["id"] for c in json.loads((AUDITS / "group_catalog.json").read_text())["techniques"]
    ]
    attack_names = {
        n.strip().lower() for g in kb.groups.values() for n in (g.name, *g.aliases) if n
    }
    for name in ("ctibench_taa", "athena_taa"):
        cases = read_jsonl(AUDITS / f"cases_ext_{name}.jsonl")
        predictions = read_jsonl(AUDITS / f"{args.actor_name}_ext_{name}.jsonl")
        by_report: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for c in cases:
            by_report[c["report_id"]].append(c)
        sums: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(groups, 0.0))
        counts: dict[str, int] = defaultdict(int)
        report_of = {c["case_id"]: c["report_id"] for c in cases}
        for p in predictions:
            if p.get("status") != "ok":
                continue
            r = report_of[p["case_id"]]
            counts[r] += 1
            for g, prob in p["top"]:
                if g in sums[r]:
                    sums[r][g] += float(prob)
        head_scores, tfidf_scores, coverable = [], [], 0
        details = []
        for r, items in sorted(by_report.items()):
            answer = items[0]["gold_actor"]
            coverable += any(connected(answer, n, aliases) for n in attack_names)
            head_pick = max(sums[r], key=sums[r].get) if counts.get(r) else None
            text = " ".join(i["text"] for i in items)
            tf = tfidf.scores(text) if tfidf is not None else {}
            tf_pick = max(tf, key=tf.get) if tf else None
            head_name = kb.groups[head_pick].name if head_pick else ""
            tf_name = kb.groups[tf_pick].name if tf_pick else ""
            head_scores.append(taa_score(answer, head_name, aliases, related))
            tfidf_scores.append(taa_score(answer, tf_name, aliases, related))
            details.append(
                {
                    "report": r,
                    "answer": answer,
                    "head": head_name,
                    "tfidf": tf_name,
                    "head_score": head_scores[-1]["combined"],
                    "tfidf_score": tfidf_scores[-1]["combined"],
                }
            )
        report["taa"][name] = {
            "reports": len(by_report),
            "answers_mapping_to_an_attack_group": coverable,
            "actor_head": taa_block(head_scores),
            "tfidf_text_group": taa_block(tfidf_scores),
            "details": details,
        }
    (AUDITS / "external_bench.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for k, v in report["ate"].items():
        print("ATE", k, json.dumps(v))
    for k, v in report["taa"].items():
        print("TAA", k, {kk: vv for kk, vv in v.items() if kk != "details"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    commands.add_parser("prepare").set_defaults(func=prepare)
    p = commands.add_parser("predict")
    p.set_defaults(func=predict)
    p.add_argument("--technique", type=Path, required=True)
    p.add_argument("--technique-name", default="E2")
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--actor-name", default="T1")
    p.add_argument("--batch", type=int, default=16)
    s = commands.add_parser("score")
    s.set_defaults(func=score)
    s.add_argument("--technique-name", default="E2")
    s.add_argument("--actor-name", default="T1")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
