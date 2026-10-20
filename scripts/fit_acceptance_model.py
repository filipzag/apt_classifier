"""Fit the calibrated technique-acceptance model on the public *dev* splits.

Runs the local retrieval pipeline (no verifier) over the dev splits, extracts one feature
row per candidate, fits :class:`apt_classifier.acceptance.AcceptanceModel`, chooses the
acceptance threshold at a precision-constrained operating point on dev, and writes the
artifact to ``data/models/acceptance/acceptance_model.json``. Test splits are never read
here; evaluate with ``scripts/benchmark_mapping.py`` afterwards.

Retrieval runs with exactly the serving arguments (``apt_classifier.retrieve.retrieval_kwargs``
and ``lexical_fields``), and the artifact stores the resulting retrieval contract: the runtime
refuses the model whenever a setting its features depend on differs. (Models fitted before
2026-10-02 retrieved with the synthetic sentences on and without platform-from-text scoping,
while serving has the synthetic sentences off and the scoping on. They store no contract and are
held only to the knobs new with contracts, ``acceptance.legacy_contract``.)

Threshold (``--threshold-selection``):

- ``oof_runtime`` (default): the weights are fitted on all of dev; the threshold is chosen on
  pooled out-of-fold probabilities (``--cv-folds`` grouped folds stratified by dataset, near-copy
  cases at cosine >= ``--group-similarity`` kept in one fold; ``scripts/retrieval_cache.py``)
  and scored with the runtime gate (benign/sparse guards, cue-supported candidates, hierarchy
  parents; ``acceptance.choose_threshold_runtime``). The in-sample runtime threshold is recorded
  beside it and a gap above 0.03 is flagged.
- ``in_sample``: the earlier procedure, the in-sample probabilities scored on ``p`` alone.

``--dump-rows PATH`` writes the feature rows, labels, guards, folds and probabilities as ``.npz``
for the tuning harness's fit check.

Exemplar library (``--exemplar-backend``, default ``TECHNIQUE_EXEMPLAR_BACKEND``): with
``chroma`` the model is fitted on the Chroma library, whose vectors Chroma re-normalised on
insert, so it has its own fingerprint. The artifact records that fingerprint
(``metadata.exemplar_fingerprint``, ``retrieval_contract.exemplar_fingerprint``), the backend
(``metadata.exemplar_backend``, ``retrieval_contract.exemplar_backend``) and the npz the library
was copied from (``metadata.exemplars.source_fingerprint``); the runtime refuses it on any other
library, the npz included. Pass ``--require-exemplar-backend chroma`` so a fallback to the npz
stops the fit instead.

An existing ``--output`` (or ``--dump-rows``) is replaced only with ``--overwrite``, and a path
that resolves outside the repository (a worktree's ``data/models`` symlink) only with
``--allow-shared-output``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.acceptance import (
    FEATURE_NAMES,
    AcceptanceModel,
    candidate_features,
    choose_threshold,
    choose_threshold_runtime,
    compute_technique_priors,
    feature_vector,
    prior_features,
)
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evaluation import (
    TechniqueEvaluationCase,
    pipeline_fingerprint,
    source_fingerprint,
)
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance
from apt_classifier.exemplars import exemplar_kwargs, load_exemplar_index
from apt_classifier.retrieve import (
    LexicalTechniqueIndex,
    acceptance_views,
    candidate_gate_flags,
    context_gate_flags,
    lexical_fields,
    retrieval_contract_for,
    retrieval_kwargs,
    retrieve_techniques,
    runtime_gate_masks,
    suppressed_parent_flags,
)
from apt_classifier.technique_head import load_technique_head
from apt_classifier.vectorstore import ChromaVectorStore, procedure_sentence_texts
from scripts.benchmark_mapping import DEFAULT_ROOT, FINGERPRINTED_SOURCES, dataset_of, load_cases
from scripts.retrieval_cache import (
    DEFAULT_CV_SEED,
    DEFAULT_FOLDS,
    DEFAULT_GROUP_SIMILARITY,
    case_folds,
    fold_summary,
    out_of_fold_probabilities,
)
from scripts.run_tooling import (
    add_exemplar_backend_arguments,
    add_output_guard_arguments,
    enforce_output_guard,
    exemplar_backend_problem,
    exemplar_setting_overrides,
    exemplar_signature,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def exemplar_library_backend(index: Any) -> str | None:
    """``npz`` or ``chroma``: where the library the model is fitted on was read from."""
    return str(getattr(index, "backend", "npz") or "npz") if index is not None else None


def exemplar_fit_metadata(settings: Any, exemplar_index: Any) -> dict[str, Any]:
    """What the artifact records about the exemplar library it was fitted on. The runtime
    refuses the model unless that library is loaded (``exemplar_fingerprint``: the npz's, or
    the Chroma library's own, recomputed from the vectors Chroma returns); the backend is
    recorded here and in the retrieval contract, the npz a Chroma library was copied from in
    ``exemplars.source_fingerprint``."""
    if exemplar_index is None:
        return {"exemplar_fingerprint": None, "exemplar_backend": None, "exemplars": None}
    backend = exemplar_library_backend(exemplar_index)
    return {
        "exemplar_fingerprint": exemplar_index.fingerprint,
        "exemplar_backend": backend,
        "exemplars": {
            "fingerprint": exemplar_index.fingerprint,
            "backend": backend,
            "source_fingerprint": getattr(
                exemplar_index, "source_fingerprint", exemplar_index.fingerprint
            ),
            "collection": getattr(exemplar_index, "collection_name", None),
            "count": len(exemplar_index),
            **{
                k: v
                for k, v in exemplar_kwargs(settings, exemplar_index).items()
                if k != "exemplar_index"
            },
            "store": exemplar_signature(exemplar_index),
        },
    }


DEFAULT_OUTPUT = REPO_ROOT / "data" / "models" / "acceptance" / "acceptance_model.json"
# the single-answer cap's bars for a second acceptance (None: no cap; 1.0: strictly one)
CAP_BARS: tuple[float | None, ...] = (None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# the head-first rule's bars on the head's joint probability (None: no rule)
HEAD_FIRST_BARS: tuple[float | None, ...] = (None, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# an out-of-fold threshold this far from the in-sample runtime one is flagged in the metadata
THRESHOLD_GAP_FLAG = 0.03


@dataclass
class FitRows:
    """One feature row per judged candidate, with what the runtime gate needs to score it."""

    rows: list[dict[str, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    case_ids: list[str] = field(default_factory=list)
    technique_ids: list[str] = field(default_factory=list)
    # candidate_gate_flags plus parent_suppressed (suppressed_parent_flags)
    flags: list[dict[str, bool]] = field(default_factory=list)
    expected_counts: dict[str, int] = field(default_factory=dict)

    def masks(self) -> tuple[list[bool], list[bool]]:
        """``(blocked, forced)`` for ``choose_threshold_runtime``."""
        return runtime_gate_masks(self.flags)


def extract_rows(
    cases,
    *,
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    lexical_index: LexicalTechniqueIndex,
    priors: dict[str, list[float]] | None = None,
    exemplar_index: Any = None,
    technique_head: Any = None,
) -> FitRows:
    # the serving arguments, so the rows are the candidates the runtime later judges
    kwargs = {
        **retrieval_kwargs(
            settings,
            exemplar_index=exemplar_index,
            lexical_index=lexical_index,
            technique_head=technique_head,
        ),
        "acceptance_model": None,
    }
    output = FitRows()
    for case in cases:
        ledger = EvidenceLedger(
            observations=[
                Observation(
                    text=case.text,
                    provenance=Provenance(source="public_benchmark", source_name=case.source),
                )
            ]
        )
        result = retrieve_techniques(ledger, store, kb, **kwargs)
        expected = set(case.expected_technique_ids)
        output.expected_counts[case.case_id] = len(expected)
        candidates = list(result.candidates)
        suppressed = {
            id(candidate): flag
            for candidate, flag in zip(candidates, suppressed_parent_flags(candidates), strict=True)
        }
        # Same per-sentence views the runtime judges, so features match inference exactly.
        for view, views, group in acceptance_views(
            ledger.observations[0],
            candidates,
            segment_acceptance=bool(kwargs["segment_acceptance"]),
        ):
            context = context_gate_flags(str(view.text))
            for candidate, original in zip(views, group, strict=True):
                row = candidate_features(
                    candidate,
                    views,
                    benign_context=context["benign"],
                    sparse_context=context["sparse"],
                )
                if priors is not None:
                    row.update(prior_features(priors, candidate.technique_id))
                flags = candidate_gate_flags(
                    original, str(view.text), threshold=kwargs["threshold"], context=context
                )
                flags["parent_suppressed"] = suppressed[id(original)]
                output.rows.append(row)
                output.labels.append(int(candidate.technique_id in expected))
                output.case_ids.append(case.case_id)
                output.technique_ids.append(candidate.technique_id)
                output.flags.append(flags)
    return output


@dataclass
class ThresholdSelection:
    operating_point: dict[str, Any]
    # everything recorded in the artifact's metadata.threshold_selection
    selection: dict[str, Any]
    out_of_fold: np.ndarray | None = None
    folds: dict[str, int] | None = None
    groups: dict[str, int] | None = None


def select_threshold(
    fitted: FitRows,
    probabilities: list[float],
    *,
    cases: list[Any],
    datasets: dict[str, str],
    embedder: Any,
    method: str,
    attack_version: str,
    l2: float,
    min_precision: float,
    cv_folds: int = DEFAULT_FOLDS,
    cv_seed: int = DEFAULT_CV_SEED,
    group_similarity: float = DEFAULT_GROUP_SIMILARITY,
    hierarchy_parent_mode: str = "off",
    feature_schema_version: int = 3,
    cap_bars: Sequence[float | None] = (None,),
    head_first_bars: Sequence[float | None] = (None,),
) -> ThresholdSelection:
    """The acceptance threshold by ``method`` (``oof_runtime`` or ``in_sample``), with both
    in-sample operating points recorded beside it; ``probabilities`` are the in-sample ones of
    the model fitted on all rows."""
    rows, labels, case_ids = fitted.rows, fitted.labels, fitted.case_ids
    blocked, forced = fitted.masks()
    runtime_scoring = {
        "blocked": blocked,
        "forced": forced,
        "technique_ids": fitted.technique_ids,
        "min_precision": min_precision,
    }
    in_sample_p_only = choose_threshold(
        probabilities, labels, case_ids, fitted.expected_counts, min_precision=min_precision
    )
    in_sample_runtime = choose_threshold_runtime(
        probabilities, labels, case_ids, fitted.expected_counts, **runtime_scoring
    )
    selection: dict[str, Any] = {
        "method": method,
        "in_sample_probability_only": in_sample_p_only,
        "in_sample_runtime": in_sample_runtime,
        # the runtime rule emulates the gate without the hierarchy-consistency pass
        "runtime_rule_exact": hierarchy_parent_mode == "off",
    }
    if hierarchy_parent_mode != "off":
        print(
            f"warning: hierarchy_parent_mode {hierarchy_parent_mode!r} is not emulated by the "
            "runtime threshold rule",
            file=sys.stderr,
        )
    if method == "in_sample":
        return ThresholdSelection(in_sample_p_only, selection)
    if method != "oof_runtime":
        raise ValueError(f"unknown threshold selection {method!r}")
    # near-copies are found from the case texts (dev has no document ids)
    vectors = embedder.embed([case.text for case in cases])
    folds, groups = case_folds(
        [case.case_id for case in cases],
        [datasets[case.case_id] for case in cases],
        vectors,
        n_splits=cv_folds,
        seed=cv_seed,
        min_similarity=group_similarity,
    )
    out_of_fold = out_of_fold_probabilities(
        rows,
        labels,
        [folds[case_id] for case_id in case_ids],
        attack_version=attack_version,
        l2=l2,
        feature_schema_version=feature_schema_version,
    )
    # the single-answer cap: every bar on the grid (None = no cap) scored with the runtime
    # rule on the out-of-fold probabilities; the best F1 at the precision floor wins
    cap_grid: list[dict[str, Any]] = []
    operating_point: dict[str, Any] | None = None
    head_probabilities = [float(row.get("th_technique_p", 0.0) or 0.0) for row in rows]
    for bar in cap_bars:
        for head_bar in head_first_bars:
            point = choose_threshold_runtime(
                out_of_fold.tolist(),
                labels,
                case_ids,
                fitted.expected_counts,
                second_min_probability=bar,
                head_probabilities=head_probabilities if head_bar is not None else None,
                head_first_min_probability=head_bar,
                **runtime_scoring,
            )
            cap_grid.append(point)
            if operating_point is None or (
                point["precision_floor_met"],
                point["micro_f1"],
            ) > (operating_point["precision_floor_met"], operating_point["micro_f1"]):
                operating_point = point
    assert operating_point is not None
    gap = abs(float(operating_point["threshold"]) - float(in_sample_runtime["threshold"]))
    selection.update(
        {
            "out_of_fold_runtime": operating_point,
            **({"cap_grid": cap_grid} if len(cap_grid) > 1 else {}),
            "cv": {
                "folds": cv_folds,
                "seed": cv_seed,
                "strata": "dataset",
                "group_similarity": group_similarity,
                **fold_summary(folds, groups, datasets),
            },
            "in_sample_gap": gap,
            "in_sample_gap_flagged": gap > THRESHOLD_GAP_FLAG,
        }
    )
    if gap > THRESHOLD_GAP_FLAG:
        print(
            f"warning: out-of-fold threshold {operating_point['threshold']:.2f} is {gap:.2f} "
            f"from the in-sample runtime threshold {in_sample_runtime['threshold']:.2f}",
            file=sys.stderr,
        )
    return ThresholdSelection(operating_point, selection, out_of_fold, folds, groups)


def dump_rows(
    path: Path,
    fitted: FitRows,
    *,
    datasets: dict[str, str],
    in_sample: list[float],
    out_of_fold: np.ndarray | None,
    folds: dict[str, int] | None,
    groups: dict[str, int] | None,
    summary: dict[str, Any],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> Path:
    """The fit's rows as ``.npz`` (no pickles): features in ``feature_names`` order (the fitted
    model's schema), labels, guards, per-row case/technique/dataset/fold/group and both
    probabilities."""
    blocked, forced = fitted.masks()
    flag_names = ("benign", "sparse", "hierarchy_parent", "cue_supported", "parent_suppressed")
    payload: dict[str, Any] = {
        "feature_names": np.asarray(list(feature_names)),
        "features": np.vstack([feature_vector(row, feature_names) for row in fitted.rows])
        if fitted.rows
        else np.zeros((0, len(feature_names))),
        "labels": np.asarray(fitted.labels, dtype=np.int8),
        "case_ids": np.asarray(fitted.case_ids),
        "technique_ids": np.asarray(fitted.technique_ids),
        "datasets": np.asarray([datasets.get(case_id, "") for case_id in fitted.case_ids]),
        "blocked": np.asarray(blocked, dtype=bool),
        "forced": np.asarray(forced, dtype=bool),
        "p_in_sample": np.asarray(in_sample, dtype=np.float64),
        "p_out_of_fold": out_of_fold
        if out_of_fold is not None
        else np.full(len(fitted.rows), np.nan),
        "folds": np.asarray(
            [folds[case_id] if folds else -1 for case_id in fitted.case_ids], dtype=np.int64
        ),
        "groups": np.asarray(
            [groups[case_id] if groups else -1 for case_id in fitted.case_ids], dtype=np.int64
        ),
        "expected_case_ids": np.asarray(list(fitted.expected_counts)),
        "expected_counts": np.asarray(list(fitted.expected_counts.values()), dtype=np.int64),
        "summary": np.asarray(json.dumps(summary, sort_keys=True, default=str)),
        **{
            name: np.asarray([bool(row.get(name)) for row in fitted.flags], dtype=bool)
            for name in flag_names
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # a file handle, so numpy cannot append ".npz" to the path the output guard checked
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["tram", "expert", "derived_procedures"])
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--sample", type=int, help="stratified sample per dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument(
        "--threshold-selection",
        choices=("in_sample", "oof_runtime"),
        default="oof_runtime",
        help="oof_runtime: pooled out-of-fold probabilities scored with the runtime gate "
        "(default); in_sample: the in-sample probabilities scored on p alone (the procedure "
        "of models fitted before 2026-10-02)",
    )
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument(
        "--group-similarity",
        type=float,
        default=DEFAULT_GROUP_SIMILARITY,
        help="cases whose texts reach this cosine share a fold (union-find)",
    )
    parser.add_argument(
        "--dump-rows",
        type=Path,
        help="also write the feature rows, guards, folds and probabilities to this .npz",
    )
    parser.add_argument(
        "--no-exemplars",
        action="store_true",
        help="fit without the exemplar index (schema-3 features stay at their empty values)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--priors-split",
        default="train",
        choices=["train", "none"],
        help="split whose label counts feed the per-technique prior features (never dev/test); "
        "'none' fits without priors (all prior features zero)",
    )
    parser.add_argument(
        "--extra-jsonl",
        type=Path,
        nargs="*",
        default=[],
        help="additional case files (TechniqueEvaluationCase rows) to fit on, e.g. the CTI-HAL "
        "statements; they become development data and must not be quoted as held-out",
    )
    parser.add_argument(
        "--require-technique-head",
        action="store_true",
        help="exit 2 unless the technique head service scored every case (TECHNIQUE_HEAD_ENABLED)",
    )
    parser.add_argument(
        "--single-answer-cap",
        action="store_true",
        help="choose a single-answer cap on dev: at most one probability-only acceptance per "
        "observation, a second above a bar from {0.5..1.0}; 'no cap' competes on the same grid",
    )
    parser.add_argument(
        "--head-first",
        action="store_true",
        help="choose a head-first bar on dev: a candidate the technique head gives at least that "
        "joint probability is accepted outright; 'no rule' competes on the same grid",
    )
    parser.add_argument(
        "--feature-schema",
        choices=("auto", "4", "5"),
        default="auto",
        help="acceptance feature schema: auto = 4 with the technique head active, else 3; "
        "5 adds the head's second-answer features (needs the head)",
    )
    add_output_guard_arguments(parser)
    add_exemplar_backend_arguments(parser)
    args = parser.parse_args(argv)
    enforce_output_guard(parser, args, [args.output, args.dump_rows], repo_root=REPO_ROOT)

    settings = Settings(**exemplar_setting_overrides(args))
    kb = AttackKnowledgeBase.load(settings.kb_path)
    cases, diagnostics = load_cases(
        args.root,
        datasets=set(args.datasets),
        split=args.split,
        sample=args.sample,
        seed=args.seed,
        indexed_texts=procedure_sentence_texts(kb),
    )
    extra_count = 0
    for path in args.extra_jsonl:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(TechniqueEvaluationCase.from_dict(json.loads(line)))
                extra_count += 1
    diagnostics["extra_jsonl"] = {str(path): True for path in args.extra_jsonl}
    diagnostics["extra_cases"] = extra_count
    if not cases:
        parser.error("no cases loaded")
    priors: dict[str, list[float]] | None = None
    priors_info: dict[str, Any] = {"split": args.priors_split}
    if args.priors_split != "none":
        prior_cases, _ = load_cases(
            args.root,
            datasets=set(args.datasets),
            split=args.priors_split,
            indexed_texts=procedure_sentence_texts(kb),
        )
        label_counts = Counter(
            technique_id for case in prior_cases for technique_id in case.expected_technique_ids
        )
        priors = compute_technique_priors(kb, label_counts)
        priors_info.update(
            {
                "case_count": len(prior_cases),
                "labelled_techniques": len(label_counts),
                "datasets": sorted(set(args.datasets)),
            }
        )
        print(
            f"priors from {len(prior_cases)} {args.priors_split} cases: "
            f"{len(label_counts)} labelled techniques"
        )
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings, embedder)
    # the serving lexical fields (no synthetic sentences unless TECHNIQUE_INCLUDE_SYNTHETIC)
    lexical_index = LexicalTechniqueIndex.from_kb(kb, fields=lexical_fields(settings))
    exemplar_index, exemplar_status = load_exemplar_index(settings, client=store.client)
    if args.no_exemplars:
        exemplar_index, exemplar_status = None, "disabled by --no-exemplars"
    elif exemplar_index is None:
        parser.error(
            f"exemplar index unavailable ({exemplar_status}); run make exemplars or pass --no-exemplars"
        )
    if exemplar_index is not None and args.split != "dev":
        parser.error("the exemplar index is built from the train split: fit on --split dev only")
    if not args.no_exemplars:
        problem = exemplar_backend_problem(exemplar_index, args.require_exemplar_backend)
        if problem:
            parser.error(f"{problem} ({exemplar_status})")
    print(f"exemplars: {exemplar_status}")
    # the technique head (off by default): a fit with it active reads schema 4 and binds the
    # artifact to the served checkpoint
    technique_head, head_status = load_technique_head(settings, kb)
    if args.require_technique_head and technique_head is None:
        parser.error(f"--require-technique-head: the technique head is {head_status}")
    print(f"technique head: {head_status}")
    if exemplar_index is not None:
        print(
            f"exemplar library: backend {exemplar_library_backend(exemplar_index)}, fingerprint "
            f"{exemplar_index.fingerprint}"
            + (
                f" (copied from npz {exemplar_index.source_fingerprint})"
                if getattr(exemplar_index, "source_fingerprint", None)
                else ""
            )
        )
    started = time.time()
    fitted = extract_rows(
        cases,
        settings=settings,
        kb=kb,
        store=store,
        lexical_index=lexical_index,
        priors=priors,
        exemplar_index=exemplar_index,
        technique_head=technique_head,
    )
    if technique_head is not None and technique_head.errors:
        print(
            f"refusing to fit: the technique head failed {technique_head.errors} requests "
            f"({technique_head.last_error})",
            file=sys.stderr,
        )
        return 2
    rows, labels = fitted.rows, fitted.labels
    print(
        f"extracted {len(rows)} candidate rows from {len(cases)} cases in {time.time() - started:.0f}s"
    )
    # what the runtime must meet for this model to run (stored in the artifact)
    contract = retrieval_contract_for(
        settings, exemplar_index, store=store, technique_head=technique_head
    )
    schema = 4 if technique_head is not None else 3
    if args.feature_schema != "auto":
        if technique_head is None:
            parser.error(f"--feature-schema {args.feature_schema} needs the technique head")
        schema = int(args.feature_schema)
    model = AcceptanceModel.fit(
        rows,
        labels,
        attack_version=kb.version,
        l2=args.l2,
        technique_priors=priors or {},
        contract=contract,
        feature_schema_version=schema,
    )
    probabilities = [model.probability(row) for row in rows]
    datasets = {case.case_id: dataset_of(case) for case in cases}
    selected = select_threshold(
        fitted,
        probabilities,
        cases=cases,
        datasets=datasets,
        embedder=store.embedder,
        method=args.threshold_selection,
        attack_version=kb.version,
        l2=args.l2,
        min_precision=args.min_precision,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
        group_similarity=args.group_similarity,
        hierarchy_parent_mode=str(getattr(settings, "technique_hierarchy_parent_mode", "off")),
        feature_schema_version=schema,
        cap_bars=CAP_BARS if args.single_answer_cap else (None,),
        head_first_bars=HEAD_FIRST_BARS if args.head_first else (None,),
    )
    operating_point, selection = selected.operating_point, selected.selection
    model.threshold = float(operating_point["threshold"])
    chosen_bar = operating_point.get("second_min_probability")
    gate = {
        "single_answer": chosen_bar is not None,
        "second_min_probability": chosen_bar,
        "head_first_min_probability": operating_point.get("head_first_min_probability"),
        "chosen_on": "public dev, out-of-fold",
    }
    gate_chosen = args.single_answer_cap or args.head_first
    model.metadata = {
        **({"gate": gate} if gate_chosen else {}),
        **exemplar_fit_metadata(settings, exemplar_index),
        # the head the schema-4 features came from; the runtime refuses the model on any other
        "technique_head_fingerprint": technique_head.fingerprint if technique_head else None,
        "technique_head": technique_head.identity() if technique_head else None,
        "split": args.split,
        "datasets": sorted(set(args.datasets)),
        "case_count": len(cases),
        "row_count": len(rows),
        "positive_rows": int(sum(labels)),
        "case_diagnostics": diagnostics,
        "l2": args.l2,
        "min_precision": args.min_precision,
        "priors": priors_info,
        "dev_operating_point": operating_point,
        "threshold_selection": selection,
        "retrieval_settings": {
            "segment_acceptance": getattr(settings, "technique_segment_acceptance", False),
            "hierarchy_parent_mode": getattr(settings, "technique_hierarchy_parent_mode", None),
            "similarity_threshold": settings.similarity_threshold,
            "retrieval_top_k": settings.retrieval_top_k,
            "candidate_pool": settings.technique_candidate_pool,
            "acceptance_threshold": settings.technique_acceptance_threshold,
            "relative_margin": settings.technique_relative_margin,
            "retrieval_mode": settings.technique_retrieval_mode,
        },
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "source_fingerprint": source_fingerprint(FINGERPRINTED_SOURCES),
        "pipeline_fingerprint": pipeline_fingerprint(
            {
                "attack_version": kb.version,
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_model_revision,
                "source_fingerprint": source_fingerprint(FINGERPRINTED_SOURCES),
            }
        ),
    }
    if not args.no_exemplars:
        problem = exemplar_backend_problem(exemplar_index, args.require_exemplar_backend)
        if problem:
            # refuse the artifact: a model whose rows partly came from a fallback is not the
            # run that was asked for
            print(f"refusing to write {args.output}: {problem}", file=sys.stderr)
            return 2
    if args.dump_rows:
        written = dump_rows(
            args.dump_rows,
            fitted,
            datasets=datasets,
            in_sample=probabilities,
            out_of_fold=selected.out_of_fold,
            folds=selected.folds,
            groups=selected.groups,
            summary={
                "threshold": model.threshold,
                "threshold_selection": selection,
                "retrieval_contract": contract,
            },
            feature_names=model.feature_names,
        )
        print(f"wrote {written}")
    destination = model.save(args.output)
    weights: dict[str, Any] = dict(
        zip(model.to_dict()["feature_names"], model.weights.tolist(), strict=True)
    )
    print("dev operating point:", operating_point)
    if gate_chosen:
        print("gate record:", gate)
        for point in selection.get("cap_grid", []):
            print(
                f"  cap bar {point['second_min_probability']} head-first bar "
                f"{point['head_first_min_probability']}: threshold {point['threshold']:.2f} "
                f"P {point['precision']:.3f} R {point['recall']:.3f} F1 {point['micro_f1']:.4f}"
            )
    print("standardized weights (top by |w|):")
    for name, value in sorted(weights.items(), key=lambda item: -abs(item[1]))[:12]:
        print(f"  {name:26s} {value:+.3f}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
