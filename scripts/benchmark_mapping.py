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

Candidate rows carry the calibrated acceptance probability (unrounded), the exemplar features
and the runtime guards (benign, sparse, cue-supported), so ``scripts/compare_runs.py`` can
re-threshold a stored run offline and check it against the run's own accepted sets.
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
from apt_classifier.exemplars import EVIDENCE_SOURCE as EXEMPLAR_EVIDENCE_SOURCE
from apt_classifier.exemplars import FEATURE_NAMES as EXEMPLAR_FEATURE_NAMES
from apt_classifier.exemplars import load_exemplar_index
from apt_classifier.reranker import LocalReranker, RerankerConfig
from apt_classifier.retrieve import (
    ACCEPTANCE_EVIDENCE_SOURCE,
    LexicalTechniqueIndex,
    acceptance_views,
    candidate_gate_flags,
    lexical_fields,
    retrieval_contract_for,
    retrieval_kwargs,
    retrieve_techniques,
)
from apt_classifier.technique_head import EVIDENCE_SOURCE as TECHNIQUE_HEAD_EVIDENCE_SOURCE
from apt_classifier.technique_head import FEATURE_NAMES as TECHNIQUE_HEAD_FEATURE_NAMES
from apt_classifier.technique_head import load_technique_head
from apt_classifier.vectorstore import (
    ChromaVectorStore,
    normalize_procedure_text,
    procedure_sentence_texts,
)
from scripts.run_tooling import (
    acceptance_identity,
    add_exemplar_backend_arguments,
    exemplar_backend_problem,
    exemplar_identity,
    exemplar_setting_overrides,
    exemplar_signature,
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
    REPO_ROOT / "apt_classifier" / "exemplar_store.py",
    REPO_ROOT / "apt_classifier" / "platforms.py",
    REPO_ROOT / "apt_classifier" / "technique_head.py",
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


def _exemplar_columns(evidence: Any) -> dict[str, Any]:
    """The exemplar features as the acceptance model saw them; ``None`` without exemplar
    evidence (index off), so a missing index is not mistaken for a zero vote."""
    metadata = dict(getattr(evidence, "metadata", {}) or {}) if evidence is not None else {}
    values = metadata.get("features") if isinstance(metadata.get("features"), dict) else None
    columns: dict[str, Any] = {
        name: (round(float(values[name]), 6) if values and name in values else None)
        for name in EXEMPLAR_FEATURE_NAMES
        if name != "exemplar_only"
    }
    columns["exemplar_only"] = bool(metadata.get("exemplar_only", False))
    return columns


def _technique_head_columns(evidence: Any) -> dict[str, Any]:
    """The technique head's features as the acceptance model saw them; ``None`` without head
    evidence (the head off or its request failed)."""
    metadata = dict(getattr(evidence, "metadata", {}) or {}) if evidence is not None else {}
    values = metadata.get("features") if isinstance(metadata.get("features"), dict) else None
    columns: dict[str, Any] = {
        name: (round(float(values[name]), 6) if values and name in values else None)
        for name in TECHNIQUE_HEAD_FEATURE_NAMES
        if name != "th_only"
    }
    columns["th_only"] = bool(metadata.get("head_only", False))
    return columns


def candidate_row(candidate: Any, flags: dict[str, bool] | None = None) -> dict[str, Any]:
    dense = candidate.evidence_for("dense")
    lexical = candidate.evidence_for("lexical")
    reranker = candidate.evidence_for("reranker")
    acceptance = candidate.evidence_for(ACCEPTANCE_EVIDENCE_SOURCE)
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
        "parent_technique_id": getattr(candidate, "parent_technique_id", None),
        # unrounded, so re-thresholding at the run's own threshold reproduces its decisions
        "acceptance_probability": (
            float(acceptance.score)
            if acceptance is not None and acceptance.score is not None
            else None
        ),
        **_exemplar_columns(candidate.evidence_for(EXEMPLAR_EVIDENCE_SOURCE)),
        **_technique_head_columns(candidate.evidence_for(TECHNIQUE_HEAD_EVIDENCE_SOURCE)),
        **(flags or {"benign": None, "sparse": None, "cue_supported": None}),
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
    technique_head: Any = None,
) -> dict[str, Any]:
    started = time.time()
    head_errors_before = int(getattr(technique_head, "errors", 0) or 0)
    predictions: list[TechniquePrediction] = []
    rows: list[dict[str, Any]] = []
    # why retrieve_techniques refused the model in a case (its retrieval contract, or the
    # exemplar index it needs), with the case count: those cases ran the rule-based gate
    disabled_reasons: dict[str, int] = {}
    for case in cases:
        ledger = EvidenceLedger(
            observations=[
                Observation(
                    text=case.text,
                    provenance=Provenance(source="public_benchmark", source_name=case.source),
                )
            ]
        )
        # the serving arguments (retrieval_kwargs), with this run's mode, cues and model
        result = retrieve_techniques(
            ledger,
            store,
            kb,
            **{
                **retrieval_kwargs(
                    settings,
                    exemplar_index=exemplar_index,
                    lexical_index=lexical_index,
                    technique_head=technique_head,
                ),
                "retrieval_mode": mode,
                "cues_enabled": cues_enabled,
                "acceptance_model": acceptance_model,
                "reranker": reranker,
            },
        )
        disabled = result.diagnostics.get("acceptance_model_disabled")
        if acceptance_model is not None and disabled:
            disabled_reasons[str(disabled)] = disabled_reasons.get(str(disabled), 0) + 1
        candidates = sorted(result.candidates, key=lambda item: item.fused_rank)
        # the guards per judged view (the whole case, or its sentence with segment acceptance)
        flags = {
            id(original): candidate_gate_flags(
                original, str(view.text), threshold=settings.similarity_threshold
            )
            for view, _, group in acceptance_views(
                ledger.observations[0],
                list(result.candidates),
                segment_acceptance=bool(getattr(settings, "technique_segment_acceptance", False)),
            )
            for original in group
        }
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
                "candidates": [
                    candidate_row(item, flags.get(id(item)))
                    for item in candidates[:max_candidate_rows]
                ],
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
    disabled_cases = sum(disabled_reasons.values())
    # a model refused in every case did not run: the pipeline is the rule-based one
    model_scored = acceptance_model is not None and (
        disabled_cases == 0 or disabled_cases < len(cases)
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
                **({"disabled_cases": disabled_cases} if disabled_cases else {}),
            }
            if model_scored
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
                "vote_norm": str(getattr(settings, "technique_exemplar_vote_norm", "none")),
                "segment_votes": str(getattr(settings, "technique_exemplar_segment_votes", "off")),
                # backend, search and collection (a Chroma copy is a different input)
                "store": exemplar_signature(exemplar_index),
            }
            if exemplar_index is not None
            else None
        ),
        "technique_head": (
            {
                "fingerprint": technique_head.fingerprint,
                "top_families": technique_head.config.top_families,
                "new_candidates": technique_head.config.new_candidates,
            }
            if technique_head is not None
            else None
        ),
        "source_fingerprint": source_fingerprint(FINGERPRINTED_SOURCES),
    }
    head_errors = int(getattr(technique_head, "errors", 0) or 0) - head_errors_before
    return {
        "mode": mode,
        "cues_enabled": cues_enabled,
        **({"technique_head_errors": head_errors} if technique_head is not None else {}),
        # True only when the model scored every case (not refused per call in any)
        "acceptance_model_active": acceptance_model is not None and not disabled_cases,
        **(
            {
                "acceptance_model_disabled": {
                    "cases": disabled_cases,
                    "reasons": dict(sorted(disabled_reasons.items())),
                }
            }
            if disabled_cases
            else {}
        ),
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
        "include_detection": getattr(settings, "technique_include_detection", False),
        "rrf_k": getattr(settings, "technique_rrf_k", 60),
        "dense_relative_margin": getattr(settings, "technique_dense_relative_margin", 0.12),
        "dense_overfetch": getattr(settings, "technique_dense_overfetch", 0),
        "technique_collection": getattr(settings, "technique_collection_name", "attack_techniques"),
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
    parser.add_argument(
        "--require-technique-head",
        action="store_true",
        help="exit 2 unless the technique head service (TECHNIQUE_HEAD_ENABLED) scored every "
        "case of every run",
    )
    add_exemplar_backend_arguments(parser)
    args = parser.parse_args(argv)

    settings = Settings(**exemplar_setting_overrides(args))
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
    lexical_index = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(settings))
    cue_settings = (
        (True, False) if args.ablate_cues else ((False,) if args.disable_cues else (True,))
    )
    exemplar_index, exemplar_status = load_exemplar_index(settings, client=store.client)
    problem = exemplar_backend_problem(exemplar_index, args.require_exemplar_backend)
    if problem:
        parser.error(f"{problem} ({exemplar_status})")
    technique_head, head_status = load_technique_head(settings, kb)
    if args.require_technique_head and technique_head is None:
        parser.error(f"--require-technique-head: the technique head is {head_status}")
    model, model_status = (
        (None, "disabled by --no-acceptance-model")
        if args.no_acceptance_model
        # refused unless the settings meet the model's retrieval contract; a --modes value
        # the contract does not allow disables it per run inside retrieve_techniques, and that
        # run's block records it (acceptance_model_active false, acceptance_model_disabled)
        else load_acceptance_model(
            settings,
            kb,
            exemplar_index,
            contract=retrieval_contract_for(
                settings, exemplar_index, store=store, technique_head=technique_head
            ),
        )
    )
    model_settings = (model, None) if args.ablate_acceptance_model and model else (model,)
    reranker = build_reranker(settings, kb)
    if bool(getattr(settings, "reranker_enabled", False)) and reranker is None:
        parser.error("RERANKER_ENABLED is set but the reranker failed to load (see stderr)")
    reranker_settings = (reranker, None) if args.ablate_reranker and reranker else (reranker,)
    if not args.quiet:
        print(f"exemplars: {exemplar_status}")
        print(f"technique head: {head_status}")
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
                        technique_head=technique_head,
                    )
                    refused = results[key].get("acceptance_model_disabled")
                    if refused:
                        print(
                            f"{key}: acceptance model refused in {refused['cases']} of "
                            f"{len(cases)} cases (rule-based gate): "
                            + "; ".join(refused["reasons"]),
                            file=sys.stderr,
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
        # with case_diagnostics.seed, what an offline re-threshold needs to redo the intervals
        "bootstrap_samples": args.bootstrap_samples,
        "attack_version": kb.version,
        "acceptance_model_status": model_status,
        "exemplar_status": exemplar_status,
        "technique_head_status": head_status,
        # the artifacts actually loaded (path, threshold, file hash, required exemplar index)
        "acceptance_model": acceptance_identity(settings, model, model_status),
        "exemplars": exemplar_identity(settings, exemplar_index, exemplar_status),
        "technique_head": technique_head.identity()
        if technique_head is not None
        else {"status": head_status},
        "results": results,
    }
    if args.output:
        destination = write_artifact(args.output, report)
        if not args.quiet:
            print(f"\nwrote {destination}")
    elif args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    problem = exemplar_backend_problem(exemplar_index, args.require_exemplar_backend)
    if problem:
        # the artifact is written (its exemplar header carries the counters) but the run fails
        print(problem, file=sys.stderr)
        return 2
    if args.require_technique_head and technique_head is not None and technique_head.errors:
        print(
            f"--require-technique-head: the service failed {technique_head.errors} requests "
            f"({technique_head.last_error})",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
