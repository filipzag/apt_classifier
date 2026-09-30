"""Compose ``data/audits/verification_results_current.json`` from benchmark artifacts.

The composed file is the single place that summarizes the local benchmarks. It never
contains hand-typed numbers: every value is read from an artifact written by
``scripts/benchmark_mapping.py``, ``scripts/curated_retrieval_benchmark.py`` or
``scripts/evidenceforge_eval.py``, and each source is recorded with its path, mtime and
``artifact_hash`` so staleness is visible.

Use ``--run`` to regenerate the public and curated artifacts first, ``--evidenceforge-run
BUNDLE SCENARIO OUTPUT`` (repeatable) to regenerate EvidenceForge artifacts, and
``--gates`` to record pytest / ruff / docker compose / index-count gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apt_classifier.evaluation import write_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDITS = REPO_ROOT / "data" / "audits"
PYTHON = sys.executable

SCENARIO_STATUS = {
    "branch-office-example": "development: structured signals and gating were tuned on it",
    "heldout-manufacturing": "development: used to fix narration and acceptance defects",
    "heldout-clinic": "held-out: never used for tuning",
}
SUMMARY_METRICS = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "parent_aware_f1",
    "hit_at_1",
    "hit_at_3",
    "hit_at_10",
    "mrr",
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _source_record(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    }
    if payload is None:
        record["status"] = "missing"
        return record
    if path.is_file():
        record["mtime"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    record["artifact_hash"] = payload.get("artifact_hash")
    record["generated_at"] = payload.get("generated_at")
    return record


def _metric_summary(block: dict[str, Any]) -> dict[str, Any]:
    """Summarize a metrics block that may carry bootstrap intervals."""
    metrics = block.get("metrics", block)
    intervals = block.get("bootstrap_confidence_intervals", {})
    summary: dict[str, Any] = {"case_count": metrics.get("case_count")}
    for name in SUMMARY_METRICS:
        if name not in metrics:
            continue
        entry: dict[str, Any] = {"estimate": round(float(metrics[name]), 4)}
        if name in intervals:
            entry["ci95"] = [
                round(float(intervals[name]["lower"]), 4),
                round(float(intervals[name]["upper"]), 4),
            ]
        summary[name] = entry
    if "zero_label_false_positive_rate" in metrics:
        summary["zero_label_false_positive_rate"] = round(
            float(metrics["zero_label_false_positive_rate"]), 4
        )
    return summary


def summarize_public(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    out: dict[str, Any] = {
        "split": payload.get("split"),
        "datasets": payload.get("datasets"),
        "leaky_datasets": payload.get("leaky_datasets", []),
        "case_count": payload.get("case_count"),
        "case_set_hash": payload.get("case_set_hash"),
        "case_diagnostics": payload.get("case_diagnostics"),
        "runs": {},
    }
    for key, run in results.items():
        out["runs"][key] = {
            "mode": run.get("mode"),
            "cues_enabled": run.get("cues_enabled"),
            "pipeline_fingerprint": run.get("pipeline_fingerprint"),
            "pooled": run.get("pooled", {}).get("summary")
            or _metric_summary(run.get("pooled", {})),
            "datasets": {
                name: block.get("summary") or _metric_summary(block)
                for name, block in run.get("datasets", {}).items()
            },
        }
    return out


def summarize_curated(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", payload)
    out: dict[str, Any] = {"pipeline_fingerprint": payload.get("pipeline_fingerprint"), "runs": {}}
    for key, run in results.items():
        if not isinstance(run, dict) or "positive_cases" not in run:
            continue
        positives = run["positive_cases"]
        accepted = run.get("accepted_set", {})
        out["runs"][key] = {
            "mode": run.get("mode"),
            "cues_enabled": run.get("cues_enabled", True),
            "top1": f"{run['top1']}/{positives}",
            "top3": f"{run['top3']}/{positives}",
            "top10": f"{run['top10']}/{positives}",
            "accepted_set": {
                name: accepted.get(name)
                for name in ("tp", "fp", "fn", "precision", "recall", "micro_f1")
            },
            "hard_negative_accepted_fp_cases": (
                f"{accepted.get('accepted_fp_cases', accepted.get('hard_negative_fp_cases'))}"
                f"/{run.get('hard_negative_cases')}"
            ),
            "hard_negative_no_candidate_abstentions": run.get("abstain_count"),
        }
    return out


def summarize_evidenceforge(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    scenario_path = Path(str(metadata.get("scenario_path", "")))
    scenario = scenario_path.parent.name or "unknown"
    out: dict[str, Any] = {
        "scenario": scenario,
        "status": SCENARIO_STATUS.get(scenario, "unknown provenance"),
        "generation_seed": metadata.get("generation_seed"),
        "pipeline": (metadata.get("pipeline") or {}).get("pipeline"),
        "verify": (metadata.get("pipeline") or {}).get("verify"),
        "pipeline_fingerprint": metadata.get("pipeline_fingerprint"),
        "modes": {},
    }
    for mode, block in payload.get("results", {}).items():
        summary = _metric_summary(block)
        coverage = block.get("coverage", {})
        summary["coverage"] = coverage.get("coverage")
        summary["unsupported_reasons"] = coverage.get("unsupported_reasons")
        out["modes"][mode] = summary
    return out


def _run(command: list[str], *, timeout: int = 3600) -> dict[str, Any]:
    started = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-3:]
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "tail": tail,
        "started": started.isoformat(),
        "seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
    }


def run_gates(*, include_index: bool) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "pytest": _run([PYTHON, "-m", "pytest", "-q"]),
        "ruff_check": _run(
            [PYTHON, "-m", "ruff", "check", "apt_classifier", "app", "tests", "scripts"]
        ),
        "ruff_format": _run(
            [PYTHON, "-m", "ruff", "format", "--check", "apt_classifier", "app", "tests", "scripts"]
        ),
        "docker_compose_config": _run(["docker", "compose", "config", "--quiet"]),
    }
    if include_index:
        try:
            from apt_classifier.attack.kb import AttackKnowledgeBase
            from apt_classifier.config import Settings
            from apt_classifier.embeddings import SentenceTransformerEmbedder
            from apt_classifier.vectorstore import TECHNIQUE_SCHEMA_VERSION, ChromaVectorStore

            settings = Settings()
            kb = AttackKnowledgeBase.load(settings.kb_path)
            store = ChromaVectorStore(settings, SentenceTransformerEmbedder.from_settings(settings))
            counts = store.counts()
            gates["attack_index"] = {
                "attack_version": kb.version,
                "schema_version": TECHNIQUE_SCHEMA_VERSION,
                **{
                    name: getattr(counts, name)
                    for name in dir(counts)
                    if not name.startswith("_") and isinstance(getattr(counts, name), int)
                },
            }
        except Exception as error:  # pragma: no cover - environment dependent
            gates["attack_index"] = {"error": f"{type(error).__name__}: {error}"}
    return gates


def summarize_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    keys = (
        "known_count",
        "unknown_count",
        "top1",
        "cluster_top1",
        "top1_ci95",
        "top5",
        "mrr",
        "median_rank",
        "mean_technique_recall",
        "unknown_recall",
        "false_named_attribution_rate",
        "named_coverage",
        "named_risk",
        "named_risk_cluster_aware",
        "stage_distribution",
        "deterministic_top1",
        "deterministic_top5",
        "deterministic_mrr",
        "named_correct",
        "separation_auc",
        "margin_auc",
        "llm",
    )
    return {
        "parameters": payload.get("parameters"),
        "pipeline_fingerprint": payload.get("pipeline_fingerprint"),
        "leak_control": payload.get("leak_control"),
        **{key: metrics.get(key) for key in keys},
    }


def compose(
    *,
    public: dict[str, Any] | None,
    curated: dict[str, Any] | None,
    evidenceforge: dict[Path, dict[str, Any] | None],
    public_path: Path,
    curated_path: Path,
    gates: dict[str, Any] | None,
    attribution: dict[str, Any] | None = None,
    attribution_path: Path | None = None,
    cti_hal_mapping: dict[str, Any] | None = None,
    cti_hal_mapping_path: Path | None = None,
    cti_hal_attribution: dict[str, Any] | None = None,
    cti_hal_attribution_path: Path | None = None,
    cti_hal_attribution_production: dict[str, Any] | None = None,
    cti_hal_attribution_production_path: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "All numbers below are read from the listed source artifacts. The public held-out "
            "benchmark (tram/expert/derived_procedures test splits) is the primary gate; the "
            "curated set is a development diagnostic; only the heldout-clinic EvidenceForge "
            "bundle is untouched by tuning. Attribution is measured by the text-in, group-out "
            "self-consistency benchmark (scripts/attribution_benchmark.py), not by field data."
        ),
        "sources": {
            "public_heldout": _source_record(public_path, public),
            "curated": _source_record(curated_path, curated),
            "evidenceforge": [
                _source_record(path, payload) for path, payload in evidenceforge.items()
            ],
            "attribution": _source_record(attribution_path, attribution)
            if attribution_path is not None
            else None,
            "cti_hal_mapping": _source_record(cti_hal_mapping_path, cti_hal_mapping)
            if cti_hal_mapping_path is not None
            else None,
            "cti_hal_attribution_production": _source_record(
                cti_hal_attribution_production_path, cti_hal_attribution_production
            )
            if cti_hal_attribution_production_path is not None
            and cti_hal_attribution_production is not None
            else None,
            "cti_hal_attribution": _source_record(cti_hal_attribution_path, cti_hal_attribution)
            if cti_hal_attribution_path is not None
            else None,
        },
        "gates": gates,
        "public_heldout": summarize_public(public) if public else None,
        "curated": summarize_curated(curated) if curated else None,
        "evidenceforge": [
            summarize_evidenceforge(payload)
            for payload in evidenceforge.values()
            if payload is not None
        ],
        "attribution_text_heldout": summarize_attribution(attribution) if attribution else None,
        "cti_hal_mapping": summarize_public(cti_hal_mapping) if cti_hal_mapping else None,
        "cti_hal_attribution": summarize_attribution(cti_hal_attribution)
        if cti_hal_attribution
        else None,
        # Production pipeline (technique verifier) on a credit-bounded sample of the CTI-HAL
        # reports; never the adoption gate. The committed artifact predates the removal of the
        # LLM zero-shot vote (2026-09-30), so it still includes that vote.
        "cti_hal_attribution_production": summarize_attribution(cti_hal_attribution_production)
        if cti_hal_attribution_production
        else None,
        "superseded": "data/audits/superseded/README.md",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=AUDITS / "public_heldout_current.json")
    parser.add_argument("--curated", type=Path, default=AUDITS / "curated_hybrid_current.json")
    parser.add_argument(
        "--evidenceforge",
        type=Path,
        nargs="*",
        default=[
            AUDITS / "evidenceforge_local_leakfree.json",
            AUDITS / "evidenceforge_heldout_local.json",
            AUDITS / "evidenceforge_clinic_local.json",
            AUDITS / "evidenceforge_clinic_production.json",
        ],
    )
    parser.add_argument(
        "--attribution", type=Path, default=AUDITS / "attribution_heldout_current.json"
    )
    parser.add_argument(
        "--cti-hal-mapping", type=Path, default=AUDITS / "cti_hal_mapping_current.json"
    )
    parser.add_argument(
        "--cti-hal-attribution", type=Path, default=AUDITS / "attribution_cti_hal_current.json"
    )
    parser.add_argument(
        "--cti-hal-attribution-production",
        type=Path,
        default=AUDITS / "attribution_cti_hal_production.json",
    )
    parser.add_argument("--output", type=Path, default=AUDITS / "verification_results_current.json")
    parser.add_argument("--run", action="store_true", help="regenerate public + curated first")
    parser.add_argument(
        "--evidenceforge-run",
        nargs=3,
        metavar=("BUNDLE_DIR", "SCENARIO", "OUTPUT"),
        action="append",
        default=[],
        help="regenerate an EvidenceForge local artifact before composing",
    )
    parser.add_argument("--gates", action="store_true", help="run pytest/ruff/compose/index gates")
    parser.add_argument("--no-index-gate", action="store_true")
    args = parser.parse_args(argv)

    runs: dict[str, Any] = {}
    if args.run:
        runs["public"] = _run(
            [
                PYTHON,
                str(REPO_ROOT / "scripts" / "benchmark_mapping.py"),
                "--ablate-cues",
                "--quiet",
                "--output",
                str(args.public),
            ]
        )
        runs["curated"] = _run(
            [
                PYTHON,
                str(REPO_ROOT / "scripts" / "curated_retrieval_benchmark.py"),
                "--modes",
                "legacy_dense",
                "hybrid",
                "lexical",
                "--ablate-cues",
                "--output",
                str(args.curated),
            ]
        )
    for bundle, scenario, output in args.evidenceforge_run:
        runs[f"evidenceforge:{Path(scenario).parent.name}"] = _run(
            [
                PYTHON,
                str(REPO_ROOT / "scripts" / "evidenceforge_eval.py"),
                "--evidenceforge-dir",
                bundle,
                "--scenario",
                scenario,
                "--pipeline",
                "local",
                "--mode",
                "all",
                "--output-local",
                output,
            ]
        )
        if Path(output) not in args.evidenceforge:
            args.evidenceforge.append(Path(output))
    gates = run_gates(include_index=not args.no_index_gate) if args.gates else None
    report = compose(
        public=_load(args.public),
        curated=_load(args.curated),
        evidenceforge={path: _load(path) for path in args.evidenceforge},
        public_path=args.public,
        curated_path=args.curated,
        gates=gates,
        attribution=_load(args.attribution),
        attribution_path=args.attribution,
        cti_hal_mapping=_load(args.cti_hal_mapping),
        cti_hal_mapping_path=args.cti_hal_mapping,
        cti_hal_attribution=_load(args.cti_hal_attribution),
        cti_hal_attribution_path=args.cti_hal_attribution,
        cti_hal_attribution_production=_load(args.cti_hal_attribution_production),
        cti_hal_attribution_production_path=args.cti_hal_attribution_production,
    )
    if runs:
        report["benchmark_runs"] = runs
    destination = write_artifact(args.output, report)
    print(f"wrote {destination}")
    failed = [name for name, item in runs.items() if item["returncode"] != 0]
    if gates:
        failed.extend(
            name
            for name, item in gates.items()
            if isinstance(item, dict) and item.get("returncode")
        )
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
