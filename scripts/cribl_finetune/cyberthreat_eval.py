"""CyberThreat-Eval TTP extraction (xse/CyberThreat-Eval, MIT) through the product pipeline
(prereg ``docs/preregistration/cribl_cyberthreat_eval_2026-10-20.json``; evaluation only).

  prepare  raw/data/100-days-articles.json -> data/benchmarks/cyberthreat_eval/reports.jsonl
           (markdown stripped, sentences split as the product's no-LLM mode splits a pasted report)
  run      --name <arm>: every article as one ledger (one observation per sentence) through
           AttributionEngine.map_techniques in local mode; accepted techniques per article ->
           data/audits/cribl_ft/default/cte_<arm>.jsonl (the technique head follows the settings)
  score    --arms off head: the authors' micro precision/recall (exact ids; an id outside their
           TTP mapping is a false positive) and the same at parent level, paired article
           bootstrap head minus off -> data/audits/cribl_ft/default/cte_scores.json

Project ``.venv``; ChromaDB must be up; the head service for the head arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = REPO_ROOT / "data" / "benchmarks" / "cyberthreat_eval" / "raw"
REPORTS = REPO_ROOT / "data" / "benchmarks" / "cyberthreat_eval" / "reports.jsonl"
OUT = REPO_ROOT / "data" / "audits" / "cribl_ft" / "default"
SOURCE = "cyberthreat_eval"
BOOTSTRAP = 1000
SEED = 7
_ID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def parent_of(technique_id: str) -> str:
    return technique_id.split(".", 1)[0]


# ----- prepare -----


def clean_markdown(text: str) -> str:
    """Prose lines only: headings, list markers, emphasis, code and links stripped."""
    lines = []
    for raw in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)  # headings keep their text
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)  # list markers
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # links -> text
        line = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", line)
        line = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", line)
        line = re.sub(r"_{2}([^_]+)_{2}", r"\1", line)
        line = line.replace("|", " ")
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> None:
    from apt_classifier.evidence.local_extract import local_observations

    articles = json.loads((RAW / "data" / "100-days-articles.json").read_text(encoding="utf-8"))
    mapping = ttp_mapping()
    rows = []
    dropped_labels = 0
    for index, article in enumerate(articles):
        ids: list[str] = []
        for item in article.get("ttps") or []:
            match = _ID.search(str(item))
            if match:
                ids.append(match.group(0))
            else:
                dropped_labels += 1
        statements = [
            obs.text
            for obs in local_observations(
                clean_markdown(article.get("content", "")),
                source=SOURCE,
                source_name=str(article.get("title", ""))[:120],
            )
        ]
        rows.append(
            {
                "report_id": f"cte-{index:03d}",
                "title": article.get("title"),
                "source": article.get("source"),
                "statements": statements,
                "ttps": sorted(dict.fromkeys(ids)),
                "ttps_in_mapping": sorted(i for i in dict.fromkeys(ids) if i in mapping),
            }
        )
    REPORTS.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    statements = sum(len(r["statements"]) for r in rows)
    labels = sum(len(r["ttps"]) for r in rows)
    print(
        f"{len(rows)} articles, {statements} statements, {labels} labels "
        f"({sum(len(r['ttps_in_mapping']) for r in rows)} in the TTP mapping, "
        f"{dropped_labels} unparsable), {len(mapping)} mapping ids -> {REPORTS}"
    )


def ttp_mapping() -> dict[str, str]:
    """The authors' id -> name mapping (column 1 and 2 of TTP_Mapping.csv)."""
    mapping: dict[str, str] = {}
    with (RAW / "data" / "TTP_Mapping.csv").open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) >= 3 and _ID.fullmatch(fields[1].strip()):
                mapping[fields[1].strip()] = fields[2].strip()
    return mapping


# ----- run -----


def load_reports() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in REPORTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args: argparse.Namespace) -> int:
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.embeddings import SentenceTransformerEmbedder
    from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
    from apt_classifier.session import AttributionEngine
    from apt_classifier.vectorstore import ChromaVectorStore

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    store = ChromaVectorStore(settings, SentenceTransformerEmbedder.from_settings(settings))
    engine = AttributionEngine(settings, kb, store, None)  # type: ignore[arg-type]
    head_status = str(getattr(engine, "technique_head_status", "disabled"))
    print(f"technique head: {head_status}; acceptance model: {engine.acceptance_model_status}")
    if args.require_technique_head and engine.technique_head is None:
        print(f"refusing to run: the technique head is {head_status}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cte_{args.name}.jsonl"
    started = time.monotonic()
    with path.open("w", encoding="utf-8") as handle:
        reports = load_reports()
        for position, report in enumerate(reports, 1):
            ledger = EvidenceLedger(
                observations=[
                    Observation(
                        text=text,
                        provenance=Provenance(source=SOURCE, source_name=report["report_id"]),
                    )
                    for text in report["statements"]
                ]
            )
            retrieval, _ = engine.map_techniques(ledger, mode="local")
            diagnostics = retrieval.diagnostics or {}
            handle.write(
                json.dumps(
                    {
                        "report_id": report["report_id"],
                        "accepted": sorted(retrieval.observed_techniques),
                        "gold": report["ttps"],
                        "statements": len(report["statements"]),
                        "candidates": int(diagnostics.get("candidate_count", 0)),
                        "technique_head": diagnostics.get("technique_head"),
                        "technique_head_errors": diagnostics.get("technique_head_errors"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if position % 20 == 0:
                print(
                    f"  {position}/{len(reports)} articles, "
                    f"{(time.monotonic() - started) / position:.1f} s/article",
                    flush=True,
                )
    errors = int(getattr(engine.technique_head, "errors", 0) or 0) if engine.technique_head else 0
    print(f"wrote {path} in {time.monotonic() - started:.0f}s; head requests failed: {errors}")
    if args.require_technique_head and errors:
        print("refusing: the technique head failed requests during the run", file=sys.stderr)
        return 2
    return 0


# ----- score -----


def counts(predicted: set[str], gold: set[str], mapping: set[str]) -> tuple[int, int, int]:
    """The authors' rule: an id outside the TTP mapping is a false positive."""
    valid = {p for p in predicted if p in mapping}
    invalid = len(predicted) - len(valid)
    tp = len(valid & gold)
    return tp, len(valid - gold) + invalid, len(gold - valid)


def micro(triples: list[tuple[int, int, int]]) -> dict[str, float]:
    tp = sum(t[0] for t in triples)
    fp = sum(t[1] for t in triples)
    fn = sum(t[2] for t in triples)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def arm_triples(rows: list[dict[str, Any]], mapping: set[str]) -> dict[str, list]:
    exact, parent = [], []
    parents = {parent_of(i) for i in mapping} | mapping
    for row in rows:
        predicted, gold = set(row["accepted"]), set(row["gold"])
        exact.append(counts(predicted, gold, mapping))
        parent.append(
            counts({parent_of(p) for p in predicted}, {parent_of(g) for g in gold}, parents)
        )
    return {"exact": exact, "parent": parent}


def paired(base: list, other: list) -> dict[str, Any]:
    n = len(base)
    rng = random.Random(SEED)
    deltas: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(BOOTSTRAP):
        sample = [rng.randrange(n) for _ in range(n)]
        a = micro([base[i] for i in sample])
        b = micro([other[i] for i in sample])
        for key in deltas:
            deltas[key].append(b[key] - a[key])
    out = {}
    for key, values in deltas.items():
        values.sort()
        out[key] = {
            "estimate": round(micro(other)[key] - micro(base)[key], 4),
            "ci95": [
                round(values[int(0.025 * (BOOTSTRAP - 1))], 4),
                round(values[int(0.975 * (BOOTSTRAP - 1))], 4),
            ],
        }
    return out


def score(args: argparse.Namespace) -> None:
    mapping = set(ttp_mapping())
    arms: dict[str, dict[str, Any]] = {}
    triples: dict[str, dict[str, list]] = {}
    for name in args.arms:
        path = OUT / f"cte_{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda r: r["report_id"])
        triples[name] = arm_triples(rows, mapping)
        arms[name] = {
            "articles": len(rows),
            "accepted_per_article": round(sum(len(r["accepted"]) for r in rows) / len(rows), 2),
            "gold_per_article": round(sum(len(r["gold"]) for r in rows) / len(rows), 2),
            "exact": micro(triples[name]["exact"]),
            "parent": micro(triples[name]["parent"]),
            "head_errors": sum(int(r.get("technique_head_errors") or 0) for r in rows),
            "head_status": sorted({str(r.get("technique_head")) for r in rows}),
        }
    report: dict[str, Any] = {"arms": arms, "paired": {}}
    base = args.arms[0]
    for name in args.arms[1:]:
        report["paired"][f"{name}_vs_{base}"] = {
            level: paired(triples[base][level], triples[name][level])
            for level in ("exact", "parent")
        }
    out = OUT / "cte_scores.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, block in arms.items():
        print(
            f"{name:6s} exact P {block['exact']['precision']} R {block['exact']['recall']} "
            f"F1 {block['exact']['f1']} | parent P {block['parent']['precision']} "
            f"R {block['parent']['recall']} F1 {block['parent']['f1']} | accepted/article "
            f"{block['accepted_per_article']} (gold {block['gold_per_article']})"
        )
    for key, block in report["paired"].items():
        for level, deltas in block.items():
            print(
                f"{key} {level}: "
                + ", ".join(f"{k} {v['estimate']:+.4f} {v['ci95']}" for k, v in deltas.items())
            )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(required=True)
    prep = commands.add_parser("prepare")
    prep.set_defaults(func=prepare)
    runner = commands.add_parser("run")
    runner.add_argument("--name", required=True)
    runner.add_argument("--require-technique-head", action="store_true")
    runner.set_defaults(func=run)
    scorer = commands.add_parser("score")
    scorer.add_argument("--arms", nargs="+", default=["off", "head"])
    scorer.set_defaults(func=score)
    args = parser.parse_args()
    result = args.func(args)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
