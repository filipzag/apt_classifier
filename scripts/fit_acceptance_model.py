"""Fit the calibrated technique-acceptance model on the public *dev* splits.

Runs the local retrieval pipeline (no verifier) over the dev splits, extracts one feature
row per candidate, fits :class:`apt_classifier.acceptance.AcceptanceModel`, chooses the
acceptance threshold at a precision-constrained operating point on dev, and writes the
artifact to ``data/models/acceptance/acceptance_model.json``. Test splits are never read
here; evaluate with ``scripts/benchmark_mapping.py`` afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from apt_classifier.acceptance import (
    AcceptanceModel,
    candidate_features,
    choose_threshold,
    compute_technique_priors,
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
    _BENIGN_INDICATORS,
    MIN_SIMILARITY_ONLY_WORDS,
    LexicalTechniqueIndex,
    _distinct_word_count,
    acceptance_views,
    retrieve_techniques,
)
from apt_classifier.vectorstore import ChromaVectorStore, procedure_sentence_texts
from scripts.benchmark_mapping import DEFAULT_ROOT, FINGERPRINTED_SOURCES, load_cases

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "models" / "acceptance" / "acceptance_model.json"


def extract_rows(
    cases,
    *,
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    lexical_index: LexicalTechniqueIndex,
    priors: dict[str, list[float]] | None = None,
    exemplar_index: Any = None,
) -> tuple[list[dict[str, float]], list[int], list[str], dict[str, int]]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    case_ids: list[str] = []
    expected_counts: dict[str, int] = {}
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
            retrieval_mode=settings.technique_retrieval_mode,
            candidate_pool=settings.technique_candidate_pool,
            parent_policy=settings.technique_parent_policy,
            lexical_index=lexical_index,
            acceptance_threshold=settings.technique_acceptance_threshold,
            relative_margin=settings.technique_relative_margin,
            acceptance_model=None,
            **exemplar_kwargs(settings, exemplar_index),
        )
        expected = set(case.expected_technique_ids)
        expected_counts[case.case_id] = len(expected)
        # Same per-sentence views the runtime judges, so features match inference exactly.
        for view, views, _ in acceptance_views(
            ledger.observations[0],
            list(result.candidates),
            segment_acceptance=bool(getattr(settings, "technique_segment_acceptance", False)),
        ):
            benign = _BENIGN_INDICATORS.search(str(view.text)) is not None
            sparse = _distinct_word_count(str(view.text)) < MIN_SIMILARITY_ONLY_WORDS
            for candidate in views:
                row = candidate_features(
                    candidate, views, benign_context=benign, sparse_context=sparse
                )
                if priors is not None:
                    row.update(prior_features(priors, candidate.technique_id))
                rows.append(row)
                labels.append(int(candidate.technique_id in expected))
                case_ids.append(case.case_id)
    return rows, labels, case_ids, expected_counts


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
    args = parser.parse_args(argv)

    settings = Settings()
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
    lexical_index = LexicalTechniqueIndex.from_kb(kb)
    exemplar_index, exemplar_status = load_exemplar_index(settings)
    if args.no_exemplars:
        exemplar_index, exemplar_status = None, "disabled by --no-exemplars"
    elif exemplar_index is None:
        parser.error(
            f"exemplar index unavailable ({exemplar_status}); run make exemplars or pass --no-exemplars"
        )
    if exemplar_index is not None and args.split != "dev":
        parser.error("the exemplar index is built from the train split: fit on --split dev only")
    print(f"exemplars: {exemplar_status}")
    started = time.time()
    rows, labels, case_ids, expected_counts = extract_rows(
        cases,
        settings=settings,
        kb=kb,
        store=store,
        lexical_index=lexical_index,
        priors=priors,
        exemplar_index=exemplar_index,
    )
    print(
        f"extracted {len(rows)} candidate rows from {len(cases)} cases in {time.time() - started:.0f}s"
    )
    model = AcceptanceModel.fit(
        rows, labels, attack_version=kb.version, l2=args.l2, technique_priors=priors or {}
    )
    probabilities = [model.probability(row) for row in rows]
    operating_point = choose_threshold(
        probabilities, labels, case_ids, expected_counts, min_precision=args.min_precision
    )
    model.threshold = float(operating_point["threshold"])
    model.metadata = {
        # the runtime refuses this model unless the same exemplar index is loaded
        "exemplar_fingerprint": exemplar_index.fingerprint if exemplar_index is not None else None,
        "exemplars": (
            {
                "fingerprint": exemplar_index.fingerprint,
                "count": len(exemplar_index),
                **{
                    k: v
                    for k, v in exemplar_kwargs(settings, exemplar_index).items()
                    if k != "exemplar_index"
                },
            }
            if exemplar_index is not None
            else None
        ),
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
    destination = model.save(args.output)
    weights: dict[str, Any] = dict(
        zip(model.to_dict()["feature_names"], model.weights.tolist(), strict=True)
    )
    print("dev operating point:", operating_point)
    print("standardized weights (top by |w|):")
    for name, value in sorted(weights.items(), key=lambda item: -abs(item[1]))[:12]:
        print(f"  {name:26s} {value:+.3f}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
