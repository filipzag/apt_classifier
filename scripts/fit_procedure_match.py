"""Choose the procedure member's parameters on development data only (the vendor set and the
ATT&CK campaigns; prereg ``docs/preregistration/attribution_procedure_match_2026-10-21.json``).

The member reads what the pipeline accepted, so each development report is first run through
the engine once (``--extract``: retrieval, the technique head, acceptance; GPU, a few minutes)
and its accepted techniques with their supporting statements and the software it names are
cached (``--inputs``, local). The registered grid is then scored on the CPU with the member alone,
each report against a corpus that hides the campaign's near-copies (``hidden_texts``) as the
benchmarks do, and the variant with the best MRR (ties: top-1, then the simplest) is the dev
choice to put into the ``PROCEDURE_MATCH_*`` defaults. Nothing else is written.

    PYTHONPATH=. .venv/bin/python scripts/fit_procedure_match.py --extract   # once
    PYTHONPATH=. .venv/bin/python scripts/fit_procedure_match.py             # the grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.procedure_match import ProcedureMatchModel  # noqa: E402
from apt_classifier.text_group import (  # noqa: E402
    TextGroupModel,
    _campaign_pattern,
    actor_names,
    evidence_name_pattern,
)

GRID = {
    "min_similarity": (0.15, 0.3),
    "family": ("exact", "parent"),
    "software_route": (False, True),
    "seen_bonus": (0.0, 1.0),
    "link_weight": (0.0, 1.0),
    "aggregate": ("mean", "top5"),
}
MIN_WORDS = 3
DEFAULT_REPORTS = (
    REPO_ROOT / "data" / "benchmarks" / "vendor" / "reports.jsonl",
    REPO_ROOT / "data" / "benchmarks" / "campaigns" / "reports.jsonl",
)
DEFAULT_INPUTS = REPO_ROOT / "data" / "audits" / "procedure_match" / "dev_inputs.json"


def load_reports(paths: list[Path], kb: Any) -> list[dict[str, Any]]:
    actors = evidence_name_pattern(actor_names(kb), before_hyphen=True)
    campaigns = _campaign_pattern(kb)
    reports = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            statements = []
            for item in row.get("statements") or []:
                text = str(item.get("text") or "")
                if actors is not None:
                    text = actors.sub(
                        lambda m: "the adversary" + (m.group("possessive") or ""), text
                    )
                if campaigns is not None:
                    text = campaigns.sub("the operation", text)
                text = " ".join(text.split())
                if len(text.split()) >= MIN_WORDS:
                    statements.append(text)
            if statements and row.get("group_id"):
                reports.append(
                    {
                        "report_id": row["report_id"],
                        "group_id": row["group_id"],
                        "statements": statements,
                        "hidden_texts": list(row.get("hidden_texts") or []),
                        "source": path.parent.name,
                    }
                )
    return reports


def extract(reports: list[dict[str, Any]], output: Path) -> None:
    """Run each report through the engine once and cache the member's inputs."""
    from apt_classifier.evidence import EvidenceLedger, Observation, Provenance
    from apt_classifier.llm import OpenRouterClient
    from apt_classifier.session import AttributionEngine
    from apt_classifier.vectorstore import ChromaVectorStore

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    store = ChromaVectorStore(settings=settings)
    client = OpenRouterClient(settings) if settings.openrouter_api_key else None
    engine = AttributionEngine(settings, kb, store, client)  # type: ignore[arg-type]
    rows: dict[str, Any] = {}
    for i, report in enumerate(reports, 1):
        ledger = EvidenceLedger(
            observations=[
                Observation(text=t, provenance=Provenance(source="fit_procedure_match"))
                for t in report["statements"]
            ]
        )
        state = engine.analyze(ledger)
        context = state.context
        rows[report["report_id"]] = {
            "group_id": report["group_id"],
            "source": report["source"],
            "hidden_texts": report["hidden_texts"],
            "technique_statements": {
                t: list(s) for t, s in dict(getattr(context, "technique_statements", {})).items()
            },
            "software_ids": sorted(getattr(context, "software_ids", set()) or ()),
        }
        print(
            f"  {i}/{len(reports)} {report['report_id'][:48]}: "
            f"{len(rows[report['report_id']]['technique_statements'])} techniques, "
            f"{len(rows[report['report_id']]['software_ids'])} software",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"attack_version": kb.version, "reports": rows}, indent=1))
    print(f"wrote {output}")


def evaluate(
    base: TextGroupModel, kb: Any, params: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    model = ProcedureMatchModel(base, kb, params)
    ranks: list[int | None] = []
    for row in inputs.values():
        scorer = model
        if row.get("hidden_texts"):
            scorer = model.without(kb, group_id=row["group_id"], texts=row["hidden_texts"])
        scores, _ = scorer.scores(row["technique_statements"], set(row["software_ids"]))
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        rank = next((i for i, (g, _) in enumerate(ordered, 1) if g == row["group_id"]), None)
        ranks.append(rank)
    n = len(ranks)
    return {
        "reports": n,
        "top1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "top3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
        "top5": round(sum(1 for r in ranks if r and r <= 5) / n, 4),
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 4),
        "unranked": sum(1 for r in ranks if r is None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        default=list(DEFAULT_REPORTS),
        help="development reports (vendor set + ATT&CK campaigns); never a test set",
    )
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--extract", action="store_true", help="run the engine and cache inputs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    if args.extract:
        extract(load_reports(args.reports, kb), args.inputs)
        return 0
    if not args.inputs.is_file():
        parser.error(f"{args.inputs} missing: run with --extract first")
    payload = json.loads(args.inputs.read_text(encoding="utf-8"))
    if str(payload.get("attack_version")) != str(kb.version):
        parser.error(
            f"inputs were extracted on ATT&CK {payload.get('attack_version')}, kb {kb.version}"
        )
    inputs = payload["reports"]
    base = TextGroupModel.load(settings.text_group_path, kb)
    if base is None:
        parser.error(f"text member artifact not found: {settings.text_group_path}")
    print(f"{len(inputs)} reports; corpus {len(base.records)} sentences; ATT&CK {kb.version}")
    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), values, strict=True))
        result = evaluate(base, kb, params, inputs)
        results.append({"params": params, **result})
        print(
            f"sim {params['min_similarity']:.2f} {params['family']:6s} software "
            f"{int(params['software_route'])} bonus {params['seen_bonus']:.0f} link "
            f"{params['link_weight']:.0f} {params['aggregate']:5s}: top-1 {result['top1']:.3f} "
            f"top-3 {result['top3']:.3f} top-5 {result['top5']:.3f} MRR {result['mrr']:.4f} "
            f"unranked {result['unranked']}",
            flush=True,
        )
    best = max(
        results,
        key=lambda r: (
            r["mrr"],
            r["top1"],
            r["params"]["family"] == "exact",
            not r["params"]["software_route"],
            -r["params"]["seen_bonus"],
            -r["params"]["link_weight"],
            r["params"]["aggregate"] == "mean",
            -r["params"]["min_similarity"],
        ),
    )
    print("dev choice:", best["params"], {k: best[k] for k in ("top1", "top3", "mrr")})
    if args.json:
        print(json.dumps({"best": best["params"], "grid": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
