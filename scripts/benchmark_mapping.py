"""Held-out public technique-mapping benchmark.

Runs prepared public cases (``tumeteor/mitre-ttp-mapping`` via
``scripts/prepare_public_benchmarks.py``) through the local retrieval pipeline and
reports accepted-set and ranking metrics per dataset and pooled, with case-level
bootstrap confidence intervals and per-case rows for error analysis.

Held-out policy
---------------
- ``procedures`` is refused by default: about 22% of its texts are byte-identical to the
  ATT&CK procedure sentences embedded in the technique index. Pass ``--allow-leaky`` for a
  diagnostic run; the artifact records ``leaky_datasets``.
- Every case whose normalized text equals an indexed procedure sentence is dropped and
  counted under ``dropped_indexed_text`` regardless of dataset.
- Sampling is stratified per dataset with a fixed seed; it never takes the first N rows.

The pipeline fingerprint covers the ATT&CK version, embedding model, retrieval settings
and the source of the gating/cue modules, so code changes are visible even when settings
are unchanged. The case set has its own hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apt_classifier.acceptance import load_acceptance_model
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.parse import strip_attack_citations
from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evaluation import (
    TechniqueEvaluationCase,
    TechniquePrediction,
    bootstrap_technique_metrics,
    evaluate_technique_predictions,
    pipeline_fingerprint,
    source_fingerprint,
    write_artifact,
)
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.exemplars import exemplar_kwargs, load_exemplar_index
from apt_classifier.reranker import LocalReranker, RerankerConfig
from apt_classifier.retrieve import (
    LEXICAL_DEFAULT_FIELDS,
    LexicalTechniqueIndex,
    retrieve_techniques,
)
from apt_classifier.vectorstore import (
    ChromaVectorStore,
    normalize_procedure_text,
    procedure_sentence_texts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "benchmarks" / "public" / "normalized"
DEFAULT_DATASETS = ("tram", "expert", "derived_procedures")
LEAKY_DATASETS = frozenset({"procedures"})
FINGERPRINTED_SOURCES = (
    REPO_ROOT / "apt_classifier" / "retrieve.py",
    REPO_ROOT / "apt_classifier" / "technique_signals.py",
    REPO_ROOT / "apt_classifier" / "presets.py",
    REPO_ROOT / "apt_classifier" / "vectorstore.py",
    REPO_ROOT / "apt_classifier" / "acceptance.py",
    REPO_ROOT / "apt_classifier" / "exemplars.py",
    REPO_ROOT / "apt_classifier" / "platforms.py",
)
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


def dataset_of(case: TechniqueEvaluationCase) -> str:
    if ":" in case.source:
        return case.source.rsplit(":", 1)[-1]
    return case.case_id.rsplit("-", 2)[0] if "-" in case.case_id else case.source


def load_cases(
    root: Path,
    *,
    datasets: set[str] | tuple[str, ...],
    split: str,
    sample: int | None = None,
    seed: int = 7,
    indexed_texts: frozenset[str] = frozenset(),
    allow_leaky: bool = False,
) -> tuple[list[TechniqueEvaluationCase], dict[str, Any]]:
    """Load prepared cases with per-dataset stratified sampling and leak filtering."""
    refused = sorted(name for name in datasets if name in LEAKY_DATASETS and not allow_leaky)
    diagnostics: dict[str, Any] = {
        "refused_leaky_datasets": refused,
        "leaky_datasets_included": sorted(
            name for name in datasets if name in LEAKY_DATASETS and allow_leaky
        ),
        "dropped_indexed_text": {},
        "dropped_unscorable": {},
        "loaded": {},
        "sample_per_dataset": sample,
        "seed": seed,
    }
    cases: list[TechniqueEvaluationCase] = []
    for dataset in sorted(set(datasets) - set(refused)):
        path = root / f"{dataset}_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing prepared benchmark: {path}")
        rows: list[TechniqueEvaluationCase] = []
        dropped_leak = dropped_unscorable = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            original = tuple(raw.get("original_technique_ids", ()))
            mapped = tuple(raw.get("expected_technique_ids", ()))
            if original and not mapped:
                dropped_unscorable += 1
                continue
            case = TechniqueEvaluationCase.from_dict(raw)
            if indexed_texts and normalize_procedure_text(strip_attack_citations(case.text)) in (
                indexed_texts
            ):
                dropped_leak += 1
                continue
            rows.append(case)
        if sample is not None and len(rows) > sample:
            rows = random.Random(f"{seed}:{dataset}").sample(rows, sample)
            rows.sort(key=lambda item: item.case_id)
        diagnostics["dropped_indexed_text"][dataset] = dropped_leak
        diagnostics["dropped_unscorable"][dataset] = dropped_unscorable
        diagnostics["loaded"][dataset] = len(rows)
        cases.extend(rows)
    return cases, diagnostics


def case_set_hash(cases: list[TechniqueEvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.case_id):
        digest.update(case.case_id.encode())
        digest.update(b"\0")
        digest.update(case.text.encode())
        digest.update(b"\0")
        digest.update(",".join(case.expected_technique_ids).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_row(candidate: Any) -> dict[str, Any]:
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    reranker = candidate.evidence_for("reranker")
    return {
        "technique_id": candidate.technique_id,
        "fused_rank": candidate.fused_rank,
        "fused_score": round(float(candidate.fused_score), 6),
        "state": str(candidate.state),
        "state_reason": candidate.state_reason,
        "dense_score": round(float(dense.score), 4) if dense and dense.score is not None else None,
        "dense_field": dense.field if dense else None,
        "dense_rank": dense.rank if dense else None,
        "lexical_rank": lexical.rank if lexical else None,
        "lexical_score": (
            round(float(lexical.score), 4) if lexical and lexical.score is not None else None
        ),
        "reranker_score": (
            round(float(reranker.score), 4) if reranker and reranker.score is not None else None
        ),
        "structured_signal": candidate.evidence_for("structured_signal") is not None,
        "explicit_cue": candidate.evidence_for("explicit_cue") is not None,
        "hierarchy_parent": candidate.evidence_for("hierarchy_parent") is not None,
    }


def summarize(metrics: dict[str, Any], intervals: dict[str, dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"case_count": metrics.get("case_count", 0)}
    for name in SUMMARY_METRICS:
        if name not in metrics:
            continue
        entry: dict[str, Any] = {"estimate": round(float(metrics[name]), 4)}
        if name in intervals:
            entry["ci95"] = [round(intervals[name]["lower"], 4), round(intervals[name]["upper"], 4)]
        summary[name] = entry
    return summary


def run(
    cases: list[TechniqueEvaluationCase],
    *,
    mode: str,
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    lexical_index: LexicalTechniqueIndex,
    cues_enabled: bool = True,
    acceptance_model: Any = None,
    reranker: Any = None,
    exemplar_index: Any = None,
    bootstrap_samples: int = 300,
    bootstrap_seed: int = 7,
    max_candidate_rows: int = 12,
) -> dict[str, Any]:
    started = time.time()
    predictions: list[TechniquePrediction] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        ledger = EvidenceLedger(
            observations=[
                Observation(
                    text=case.text,
                    provenance=Provenance(source="public_benchmark", source_name=case.source),
                )
            ]
        )
        result = retrieve_techniques(
            ledger,
            store,
            kb,
            threshold=settings.similarity_threshold,
            top_k=settings.retrieval_top_k,
            retrieval_mode=mode,
            candidate_pool=settings.technique_candidate_pool,
            parent_policy=settings.technique_parent_policy,
            lexical_index=lexical_index,
            acceptance_threshold=settings.technique_acceptance_threshold,
            relative_margin=settings.technique_relative_margin,
            cues_enabled=cues_enabled,
            acceptance_model=acceptance_model,
            reranker=reranker,
            segment_acceptance=bool(getattr(settings, "technique_segment_acceptance", False)),
            hierarchy_parent_mode=str(getattr(settings, "technique_hierarchy_parent_mode", "off")),
            include_synthetic=bool(getattr(settings, "technique_include_synthetic", False)),
            rank_by_acceptance=bool(getattr(settings, "technique_rank_by_acceptance", False)),
            corroboration_accept_observations=int(
                getattr(settings, "technique_corroboration_accept_observations", 0)
            ),
            corroboration_accept_min_probability=float(
                getattr(settings, "technique_corroboration_accept_min_probability", 0.2)
            ),
            platform_from_text=bool(getattr(settings, "technique_platform_from_text", False)),
            **exemplar_kwargs(settings, exemplar_index),
        )
        candidates = sorted(result.candidates, key=lambda item: item.fused_rank)
        ranked = tuple(dict.fromkeys(candidate.technique_id for candidate in candidates))
        accepted = tuple(result.observed_techniques)
        predictions.append(TechniquePrediction.from_ranked(case.case_id, ranked, accepted=accepted))
        expected = set(case.expected_technique_ids)
        rows.append(
            {
                "case_id": case.case_id,
                "dataset": dataset_of(case),
                "text": case.text,
                "expected_technique_ids": list(case.expected_technique_ids),
                "ranked_technique_ids": list(ranked[:10]),
                "accepted_technique_ids": sorted(accepted),
                "expected_in_pool": bool(expected & set(ranked)),
                "expected_accepted": bool(expected & set(accepted)),
                "candidates": [candidate_row(item) for item in candidates[:max_candidate_rows]],
            }
        )
    by_dataset: dict[str, list[TechniqueEvaluationCase]] = {}
    for case in cases:
        by_dataset.setdefault(dataset_of(case), []).append(case)
    prediction_by_case = {prediction.case_id: prediction for prediction in predictions}
    datasets: dict[str, Any] = {}
    for dataset, subset in sorted(by_dataset.items()):
        subset_predictions = [prediction_by_case[case.case_id] for case in subset]
        metrics = evaluate_technique_predictions(subset, subset_predictions, kb=kb)
        intervals = bootstrap_technique_metrics(
            subset, subset_predictions, kb=kb, samples=bootstrap_samples, seed=bootstrap_seed
        )
        datasets[dataset] = {
            "summary": summarize(metrics, intervals),
            "metrics": metrics,
            "bootstrap_confidence_intervals": intervals,
        }
    pooled_metrics = evaluate_technique_predictions(cases, predictions, kb=kb)
    pooled_intervals = bootstrap_technique_metrics(
        cases, predictions, kb=kb, samples=bootstrap_samples, seed=bootstrap_seed
    )
    fingerprint_payload = {
        "attack_version": kb.version,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "mode": mode,
        "cues_enabled": cues_enabled,
        "acceptance_model": (
            {
                "threshold": acceptance_model.threshold,
                "pipeline_fingerprint": acceptance_model.metadata.get("pipeline_fingerprint"),
            }
            if acceptance_model is not None
            else None
        ),
        "reranker": (
            {
                "model": getattr(reranker, "model_name", None) or str(reranker.config.model_name),
                "validated": bool(getattr(reranker, "validated", False)),
                "thresholds": getattr(reranker, "calibrated_thresholds", None),
            }
            if reranker is not None
            else None
        ),
        "settings": retrieval_settings(settings),
        "exemplar_index": (
            {
                "fingerprint": exemplar_index.fingerprint,
                "count": len(exemplar_index),
                "k": int(getattr(settings, "technique_exemplar_k", 64)),
                "tau": float(getattr(settings, "technique_exemplar_tau", 0.05)),
                "duplicate_similarity": float(
                    getattr(settings, "technique_exemplar_duplicate_similarity", 0.95)
                ),
                "new_candidates": int(getattr(settings, "technique_exemplar_new_candidates", 0)),
            }
            if exemplar_index is not None
            else None
        ),
        "source_fingerprint": source_fingerprint(FINGERPRINTED_SOURCES),
    }
    return {
        "mode": mode,
        "cues_enabled": cues_enabled,
        "acceptance_model_active": acceptance_model is not None,
        "reranker_active": reranker is not None,
        "pipeline_fingerprint": pipeline_fingerprint(fingerprint_payload),
        "fingerprint_inputs": fingerprint_payload,
        "runtime_seconds": round(time.time() - started, 1),
        "pooled": {
            "summary": summarize(pooled_metrics, pooled_intervals),
            "metrics": pooled_metrics,
            "bootstrap_confidence_intervals": pooled_intervals,
        },
        "datasets": datasets,
        "cases": rows,
    }


def build_reranker(settings: Settings, kb: AttackKnowledgeBase) -> LocalReranker | None:
    """Construct the reranker exactly as ``AttributionEngine`` does; ``None`` when disabled/unloadable."""
    if not bool(getattr(settings, "reranker_enabled", False)):
        return None
    reranker = LocalReranker(
        RerankerConfig(
            enabled=True,
            model_name=str(getattr(settings, "reranker_model", "")),
            model_revision=getattr(settings, "reranker_model_revision", None),
            max_candidates=int(getattr(settings, "reranker_max_candidates", 20)),
            attack_version=str(getattr(kb, "version", "")) or None,
            expected_base_model=getattr(settings, "reranker_expected_base_model", None),
            expected_base_model_revision=getattr(
                settings, "reranker_expected_base_model_revision", None
            ),
            expected_artifact_fingerprint=getattr(
                settings, "reranker_expected_artifact_fingerprint", None
            ),
        )
    )
    if not reranker.available:
        print(f"reranker unavailable: {reranker.load_error}", file=sys.stderr)
        return None
    return reranker


def retrieval_settings(settings: Settings) -> dict[str, Any]:
    return {
        "similarity_threshold": settings.similarity_threshold,
        "retrieval_top_k": settings.retrieval_top_k,
        "candidate_pool": settings.technique_candidate_pool,
        "acceptance_threshold": settings.technique_acceptance_threshold,
        "relative_margin": settings.technique_relative_margin,
        "parent_policy": settings.technique_parent_policy,
        "preset": getattr(settings, "technique_mapping_preset", None),
        "include_synthetic": getattr(settings, "technique_include_synthetic", False),
        "segment_acceptance": getattr(settings, "technique_segment_acceptance", False),
        "hierarchy_parent_mode": getattr(settings, "technique_hierarchy_parent_mode", "off"),
        "rank_by_acceptance": getattr(settings, "technique_rank_by_acceptance", False),
        "platform_from_text": getattr(settings, "technique_platform_from_text", False),
    }


def _print_summary(label: str, block: dict[str, Any]) -> None:
    summary = block["summary"]
    parts = [f"n={summary['case_count']}"]
    for name in ("micro_precision", "micro_recall", "micro_f1", "hit_at_1", "hit_at_10"):
        entry = summary.get(name)
        if entry is None:
            continue
        text = f"{name}={entry['estimate']:.3f}"
        if "ci95" in entry:
            text += f" [{entry['ci95'][0]:.2f},{entry['ci95'][1]:.2f}]"
        parts.append(text)
    print(f"  {label:22s} " + "  ".join(parts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--modes", nargs="+", default=["hybrid"])
    parser.add_argument("--sample", type=int, help="stratified sample size per dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-leaky", action="store_true")
    parser.add_argument("--disable-cues", action="store_true", help="run without regex cues only")
    parser.add_argument("--ablate-cues", action="store_true", help="run with and without cues")
    parser.add_argument(
        "--no-acceptance-model", action="store_true", help="force the rule-based gate"
    )
    parser.add_argument(
        "--ablate-acceptance-model",
        action="store_true",
        help="run with the configured acceptance model and with the rule-based gate",
    )
    parser.add_argument(
        "--ablate-reranker",
        action="store_true",
        help="when a reranker is configured, run with and without it",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--max-candidate-rows", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    indexed_texts = procedure_sentence_texts(kb)
    cases, case_diagnostics = load_cases(
        args.root,
        datasets=set(args.datasets),
        split=args.split,
        sample=args.sample,
        seed=args.seed,
        indexed_texts=indexed_texts,
        allow_leaky=args.allow_leaky,
    )
    if not cases:
        parser.error("no cases loaded")
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings, embedder)
    include_synthetic = bool(getattr(settings, "technique_include_synthetic", False))
    lexical_index = LexicalTechniqueIndex.from_kb(
        kb,
        fields=tuple(
            field for field in LEXICAL_DEFAULT_FIELDS if include_synthetic or field != "synthetic"
        ),
    )
    cue_settings = (
        (True, False) if args.ablate_cues else ((False,) if args.disable_cues else (True,))
    )
    exemplar_index, exemplar_status = load_exemplar_index(settings)
    model, model_status = (
        (None, "disabled by --no-acceptance-model")
        if args.no_acceptance_model
        else load_acceptance_model(settings, kb, exemplar_index)
    )
    model_settings = (model, None) if args.ablate_acceptance_model and model else (model,)
    reranker = build_reranker(settings, kb)
    if bool(getattr(settings, "reranker_enabled", False)) and reranker is None:
        parser.error("RERANKER_ENABLED is set but the reranker failed to load (see stderr)")
    reranker_settings = (reranker, None) if args.ablate_reranker and reranker else (reranker,)
    if not args.quiet:
        print(f"exemplars: {exemplar_status}")
        print(f"acceptance model: {model_status}")
        print(f"reranker: {'active' if reranker is not None else 'off'}")
    results: dict[str, Any] = {}
    for mode in args.modes:
        for cues_enabled in cue_settings:
            for acceptance_model in model_settings:
                for active_reranker in reranker_settings:
                    key = mode if cues_enabled else f"{mode}_no_cues"
                    if acceptance_model is None and model is not None:
                        key += "_no_model"
                    if active_reranker is None and reranker is not None:
                        key += "_no_reranker"
                    results[key] = run(
                        cases,
                        mode=mode,
                        settings=settings,
                        kb=kb,
                        store=store,
                        lexical_index=lexical_index,
                        cues_enabled=cues_enabled,
                        acceptance_model=acceptance_model,
                        reranker=active_reranker,
                        exemplar_index=exemplar_index,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.seed,
                        max_candidate_rows=args.max_candidate_rows,
                    )
                    if not args.quiet:
                        print(f"\n=== {key} ({results[key]['runtime_seconds']}s) ===")
                        for dataset, block in results[key]["datasets"].items():
                            _print_summary(dataset, block)
                        _print_summary("pooled", results[key]["pooled"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "public_heldout",
        "split": args.split,
        "datasets": sorted(set(args.datasets) - set(case_diagnostics["refused_leaky_datasets"])),
        "leaky_datasets": case_diagnostics["leaky_datasets_included"],
        "case_count": len(cases),
        "case_set_hash": case_set_hash(cases),
        "case_diagnostics": case_diagnostics,
        "attack_version": kb.version,
        "acceptance_model_status": model_status,
        "exemplar_status": exemplar_status,
        "results": results,
    }
    if args.output:
        destination = write_artifact(args.output, report)
        if not args.quiet:
            print(f"\nwrote {destination}")
    elif args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
