"""Fit the learned attribution combiner on ATT&CK-derived folds (never on CTI-HAL).

Training lists come from the same fold machinery as ``scripts/attribution_benchmark.py`` but
with *training seeds* (the benchmark's seed 7 stays evaluation-only) and, per fold, randomised
evidence regimes that mimic real reports: a subset of the accepted techniques, injected
false-positive techniques drawn from an empirical pool, dropped software, thinned behaviour
matches. An unknown-actor arm (the group removed from the knowledge base) trains the "no
candidate" option. Group-level K-fold cross-validation reports out-of-fold ranking metrics
against the reciprocal-rank-fusion order of the same lists before the final refit.

    PYTHONPATH=. .venv/bin/python scripts/fit_attribution_combiner.py --seeds 11 23 37 --variants 5 --unknown-all
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.combiner import (  # noqa: E402
    FEATURE_NAMES,
    AttributionCombiner,
    CombinerList,
    combiner_inputs,
)
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.embeddings import SentenceTransformerEmbedder  # noqa: E402
from apt_classifier.group_mlp import precompute_draws, refit_without_groups  # noqa: E402
from apt_classifier.separation import discriminate, technique_document_counts  # noqa: E402
from apt_classifier.session import AttributionEngine  # noqa: E402
from apt_classifier.vectorstore import ChromaVectorStore  # noqa: E402


def _benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "attribution_benchmark", REPO_ROOT / "scripts" / "attribution_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- lists


def _basis(engine: AttributionEngine, fusion: Any, context: Any, top_n: int) -> Any:
    settings = engine.settings
    return discriminate(
        engine.kb,
        fusion,
        context,
        engine.discrimination_weights,
        top_k=top_n,
        floor=float(getattr(settings, "attribution_separation_floor", 1.0)),
        tool_weight_cap=float(getattr(settings, "software_tool_weight_cap", 1.5)),
        behavior_threshold=float(settings.behavior_similarity_threshold),
        soft_techniques=getattr(context, "contrast_techniques", None) or None,
    )


def list_from_state(
    engine: AttributionEngine,
    results: Any,
    fusion: Any,
    context: Any,
    *,
    top_n: int,
    truth: str | None,
) -> CombinerList:
    fused_weights, _ = engine._decisive_evidence_weights(context, dict(engine.default_weights()))  # noqa: SLF001
    basis = _basis(engine, fusion, context, top_n)
    return combiner_inputs(
        results,
        fusion,
        basis,
        context,
        engine.kb,
        top_n=top_n,
        weights=fused_weights,
        truth=truth,
        tool_weight_cap=float(getattr(engine.settings, "software_tool_weight_cap", 1.5)),
    )


def _documented(kb: AttackKnowledgeBase, group_id: str) -> set[str]:
    techniques = set(getattr(kb.groups.get(group_id), "technique_ids", ()) or ())
    parents = {
        getattr(kb.techniques.get(tid), "parent_id", None)
        for tid in techniques
        if tid in kb.techniques
    }
    return techniques | {p for p in parents if p}


@dataclass(frozen=True)
class Regime:
    """Randomisation bands; the defaults approximate real reports (technique recall ~0.3,
    precision ~0.5, software missing in ~30% of reports)."""

    keep_low: float = 0.3
    keep_high: float = 0.8
    precision_low: float = 0.3
    precision_high: float = 0.8
    drop_software: float = 0.3
    thin_behavior: float = 0.3
    generic_pool: bool = False  # weight injected false positives by how many groups document them


DEFAULT_REGIME = Regime()


def perturb(
    context: Any,
    kb: AttackKnowledgeBase,
    truth: str,
    pool: dict[str, list[float]],
    rng: random.Random,
    regime: Regime = DEFAULT_REGIME,
) -> Any:
    """A copy of the context under a randomised evidence regime."""
    ctx = copy.copy(context)
    observed = dict(context.observed_techniques)
    technique_observations = {k: tuple(v) for k, v in context.technique_observations.items()}
    all_observation_ids = sorted({oid for ids in technique_observations.values() for oid in ids})
    # keep a fraction of the accepted techniques (at least two)
    keep_fraction = rng.uniform(regime.keep_low, regime.keep_high)
    keys = sorted(observed)
    rng.shuffle(keys)
    keep = max(2, round(keep_fraction * len(keys))) if len(keys) >= 2 else len(keys)
    kept = keys[:keep]
    observed = {tid: observed[tid] for tid in kept}
    technique_observations = {tid: technique_observations.get(tid, ()) for tid in kept}
    # inject false positives to a target precision
    precision = rng.uniform(regime.precision_low, regime.precision_high)
    wanted = int(round(len(observed) * (1.0 - precision) / max(precision, 0.05)))
    documented = _documented(kb, truth)
    candidates = [tid for tid in pool if tid not in observed and tid not in documented]
    if regime.generic_pool:
        doc_counts = technique_document_counts(kb)
        weights = [len(pool[tid]) * (1 + doc_counts.get(tid, 0)) for tid in candidates]
    else:
        weights = [len(pool[tid]) for tid in candidates]
    injected = 0
    attempts = 0
    while candidates and injected < wanted and attempts < wanted * 4:
        attempts += 1
        tid = rng.choices(candidates, weights=weights, k=1)[0]
        if tid in observed:
            continue
        observed[tid] = float(rng.choice(pool[tid]))
        anchor = rng.choice(all_observation_ids) if all_observation_ids else f"obs-fp-{injected}"
        technique_observations[tid] = (anchor,)
        injected += 1
    ctx.observed_techniques = observed
    ctx.technique_observations = technique_observations
    # drop software sometimes
    if rng.random() < regime.drop_software:
        ctx.software_ids = set()
        ctx.decisive_software = {}
    else:
        ctx.software_ids = set(context.software_ids)
    # thin behaviour matches sometimes
    matches = tuple(getattr(context, "specific_behavior_matches", ()) or ())
    if matches and rng.random() < regime.thin_behavior:
        kept_matches = tuple(m for m in matches if rng.random() < 0.5)
        scores: dict[str, float] = {}
        signals: dict[str, tuple[str, ...]] = {}
        for match in kept_matches:
            scores[match.group_id] = max(scores.get(match.group_id, 0.0), float(match.score))
            signals.setdefault(match.group_id, ())
            signals[match.group_id] = (*signals[match.group_id], match.technique_id)
        ctx.specific_behavior_matches = kept_matches
        ctx.specific_behavior_scores = scores
        ctx.specific_behavior_signals = signals
    ctx.soft_techniques = {}
    ctx.soft_technique_observations = {}
    ctx.contrast_techniques = {}
    ctx.contrast_technique_observations = {}
    return ctx


def lists_leak_control(records: Iterable[dict[str, Any]]) -> str:
    """The leak control the training lists were generated under, read from the records
    themselves (a reused cache keeps its own): lists written before 2026-10-03 carry no marker
    and hid fold evidence from the dense store only."""
    markers = {str(r.get("leak_control") or "dense (unrecorded)") for r in records}
    return markers.pop() if len(markers) == 1 else "mixed: " + ", ".join(sorted(markers))


def generate_lists(
    engine: AttributionEngine,
    bench: Any,
    *,
    seeds: Iterable[int],
    variants: int,
    top_n: int,
    fold_args: dict[str, Any],
    limit: int | None,
    log: Any,
    regime: Regime = DEFAULT_REGIME,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clean and randomised lists for every training seed; returns (lists, stats)."""
    kb = engine.kb
    base_store = engine.store
    base_lexical = getattr(engine, "lexical_index", None)
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"seeds": {}, "pool_techniques": 0}
    pool: dict[str, list[float]] = {}
    try:
        for seed in seeds:
            folds = bench.build_folds(kb, seed=seed, limit=limit, **fold_args)
            log(f"seed {seed}: {len(folds)} folds")
            started = time.time()
            clean_states = []
            for index, fold in enumerate(folds, 1):
                engine.store = bench.FoldStore(
                    base_store,
                    group_id=fold.group_id,
                    held_out_techniques=fold.held_out_techniques,
                    held_out_texts=fold.held_out_texts,
                )
                engine.lexical_index = base_lexical
                bench.hide_lexical(engine, fold.held_out_texts)
                state = engine.analyze(bench._ledger(fold))  # noqa: SLF001
                clean_states.append((fold, state))
                documented = _documented(kb, fold.group_id)
                for tid, confidence in state.context.observed_techniques.items():
                    if tid not in documented:
                        pool.setdefault(tid, []).append(float(confidence))
                if index % 25 == 0:
                    log(f"  {index}/{len(folds)} analysed ({time.time() - started:.0f} s)")
            stats["pool_techniques"] = len(pool)
            rng = random.Random(seed * 1000 + 17)
            clean = randomised = skipped = 0
            for fold, state in clean_states:
                context = state.context
                lst = list_from_state(
                    engine,
                    state.classifier_results,
                    state.deterministic_fusion or state.fusion,
                    context,
                    top_n=top_n,
                    truth=fold.group_id,
                )
                if lst.truth_index is None:
                    skipped += 1
                else:
                    records.append(
                        {
                            "seed": seed,
                            "group_id": fold.group_id,
                            "variant": 0,
                            "leak_control": bench.LEAK_CONTROL,
                            **lst.to_dict(),
                        }
                    )
                    clean += 1
                for variant in range(1, variants + 1):
                    ctx = perturb(context, kb, fold.group_id, pool, rng, regime)
                    results, _, fusion, _ = engine._score(ctx, dict(state.weights))  # noqa: SLF001
                    lst = list_from_state(
                        engine, results, fusion, ctx, top_n=top_n, truth=fold.group_id
                    )
                    if lst.truth_index is None:
                        skipped += 1
                        continue
                    records.append(
                        {
                            "seed": seed,
                            "group_id": fold.group_id,
                            "variant": variant,
                            "leak_control": bench.LEAK_CONTROL,
                            **lst.to_dict(),
                        }
                    )
                    randomised += 1
            stats["seeds"][str(seed)] = {
                "folds": len(folds),
                "clean": clean,
                "randomised": randomised,
                "skipped_truth_outside_top_n": skipped,
                "seconds": round(time.time() - started, 1),
            }
            log(f"seed {seed}: {clean} clean + {randomised} randomised lists, {skipped} skipped")
    finally:
        engine.store = base_store
        engine.lexical_index = base_lexical
    return records, stats


def generate_unknown_lists(
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    bench: Any,
    *,
    seed: int,
    top_n: int,
    fold_args: dict[str, Any],
    limit: int | None,
    log: Any,
) -> list[dict[str, Any]]:
    folds = bench.build_folds(kb, seed=seed, limit=limit, **fold_args)
    records = []
    started = time.time()
    draws = None
    for index, fold in enumerate(folds, 1):
        reduced = copy.deepcopy(kb)
        del reduced.groups[fold.group_id]
        for campaign in reduced.campaigns.values():
            if fold.group_id in getattr(campaign, "group_ids", []):
                campaign.group_ids = [g for g in campaign.group_ids if g != fold.group_id]
        for software in reduced.software.values():
            if fold.group_id in getattr(software, "group_ids", []):
                software.group_ids = [g for g in software.group_ids if g != fold.group_id]
        reduced.recompute_statistics()
        proxy = bench.FoldStore(
            store,
            group_id=fold.group_id,
            held_out_techniques=fold.held_out_techniques,
            held_out_texts=fold.held_out_texts,
            hide_group_entirely=True,
        )
        engine = AttributionEngine(settings, reduced, proxy, None)  # type: ignore[arg-type]
        bench.hide_lexical(engine, fold.held_out_texts)
        if bench._forest_votes(engine):  # noqa: SLF001
            # the shipped forest knows the removed actor; the null option must be learned from
            # a model that never saw it (same parameters and seed, GPU when available)
            engine.set_profile_forest(bench._refit_forest_without_actor(engine, reduced))  # noqa: SLF001
        if getattr(engine, "group_mlp", None) is not None:
            # same rule for the group network; for speed the full knowledge base's draws are
            # reused without the removed actor's cases (training lists only, never a gate)
            if draws is None:
                draws = precompute_draws(kb, engine.group_mlp.metadata.get("params"))
            engine.set_group_mlp(refit_without_groups(engine.group_mlp, reduced, draws))
        state = engine.analyze(bench._ledger(fold))  # noqa: SLF001
        fusion = state.deterministic_fusion or state.fusion
        if not fusion.entries:
            continue
        lst = list_from_state(
            engine, state.classifier_results, fusion, state.context, top_n=top_n, truth=None
        )
        records.append(
            {
                "seed": seed,
                "group_id": fold.group_id,
                "variant": -1,
                "leak_control": bench.LEAK_CONTROL,
                **lst.to_dict(),
            }
        )
        if index % 25 == 0:
            log(f"  unknown {index}/{len(folds)} ({time.time() - started:.0f} s)")
    return records


# ----------------------------------------------------------------------------- evaluation


def _rank_metrics(lists: list[CombinerList], model: AttributionCombiner | None) -> dict[str, float]:
    """top-1 / top-5 / MRR of the combiner order (model) or the RRF order (None)."""
    hits1 = hits5 = 0
    reciprocal = 0.0
    scored = 0
    for lst in lists:
        index = lst.truth_index
        if index is None:
            continue
        scored += 1
        if model is None:
            rank = index + 1
        else:
            probabilities, _ = model.probabilities(lst)
            ordered = sorted(
                lst.group_ids, key=lambda g: (-probabilities.get(g, 0.0), lst.group_ids.index(g))
            )
            rank = ordered.index(lst.truth) + 1  # type: ignore[arg-type]
        hits1 += rank == 1
        hits5 += rank <= 5
        reciprocal += 1.0 / rank
    if not scored:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "count": 0}
    return {
        "top1": hits1 / scored,
        "top5": hits5 / scored,
        "mrr": reciprocal / scored,
        "count": scored,
    }


def _calibration(lists: list[CombinerList], model: AttributionCombiner) -> dict[str, float]:
    """Brier score and rank AUC of P(leader) for 'the combiner's leader is the truth'."""
    pairs = []
    for lst in lists:
        if lst.truth_index is None:
            continue
        probabilities, _ = model.probabilities(lst)
        leader = max(probabilities, key=probabilities.get)
        pairs.append((probabilities[leader], 1.0 if leader == lst.truth else 0.0))
    if not pairs:
        return {"brier": 0.0, "auc": 0.0}
    brier = float(np.mean([(p - y) ** 2 for p, y in pairs]))
    positives = [p for p, y in pairs if y > 0]
    negatives = [p for p, y in pairs if y == 0]
    if not positives or not negatives:
        return {"brier": brier, "auc": 0.0}
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return {"brier": brier, "auc": wins / (len(positives) * len(negatives))}


def cross_validate(
    records: list[dict[str, Any]],
    *,
    folds: int,
    l2_grid: list[float],
    null_option: bool,
    seed: int,
    log: Any,
) -> dict[str, Any]:
    """Group-level K-fold: every list of a group (clean, randomised, unknown) sits in one split."""
    groups = sorted({r["group_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    split_of = {g: i % folds for i, g in enumerate(groups)}
    report: dict[str, Any] = {"folds": folds, "groups": len(groups), "grid": {}}
    best_l2 = l2_grid[0]
    best_mrr = -1.0
    for l2 in l2_grid:
        metrics = _cv_metrics(records, split_of, folds, l2, null_option)
        report["grid"][str(l2)] = metrics
        log(
            f"l2={l2}: clean top1 {metrics['combiner']['clean']['top1']:.3f} "
            f"(rrf {metrics['rrf']['clean']['top1']:.3f}), randomised top1 "
            f"{metrics['combiner']['randomised']['top1']:.3f} "
            f"(rrf {metrics['rrf']['randomised']['top1']:.3f}), mixed mrr "
            f"{metrics['combiner']['mixed']['mrr']:.3f} (rrf {metrics['rrf']['mixed']['mrr']:.3f}), "
            f"leader auc {metrics['calibration']['auc']:.3f}"
        )
        if metrics["combiner"]["mixed"]["mrr"] > best_mrr:
            best_mrr = metrics["combiner"]["mixed"]["mrr"]
            best_l2 = l2
    report["chosen_l2"] = best_l2
    return report


def _cv_metrics(
    records: list[dict[str, Any]],
    split_of: dict[str, int],
    folds: int,
    l2: float,
    null_option: bool,
) -> dict[str, Any]:
    combiner_lists: dict[str, list[tuple[CombinerList, AttributionCombiner]]] = {
        "clean": [],
        "randomised": [],
    }
    null_pairs: list[tuple[float, float]] = []
    for split in range(folds):
        train = [CombinerList.from_dict(r) for r in records if split_of[r["group_id"]] != split]
        model = AttributionCombiner.fit(
            train, attack_version="cv", pipeline_fingerprint="cv", l2=l2, null_option=null_option
        )
        for record in records:
            if split_of[record["group_id"]] != split:
                continue
            lst = CombinerList.from_dict(record)
            if record["variant"] == -1:
                if null_option:
                    null_pairs.append((float(model.probabilities(lst)[1] or 0.0), 1.0))
                continue
            bucket = "clean" if record["variant"] == 0 else "randomised"
            combiner_lists[bucket].append((lst, model))
            if null_option:
                null_pairs.append((float(model.probabilities(lst)[1] or 0.0), 0.0))

    def scored(items: list[tuple[CombinerList, AttributionCombiner]]) -> dict[str, float]:
        hits1 = hits5 = 0
        reciprocal = 0.0
        pairs = []
        for lst, model in items:
            probabilities, _ = model.probabilities(lst)
            ordered = sorted(
                lst.group_ids, key=lambda g: (-probabilities.get(g, 0.0), lst.group_ids.index(g))
            )
            rank = ordered.index(lst.truth) + 1  # type: ignore[arg-type]
            hits1 += rank == 1
            hits5 += rank <= 5
            reciprocal += 1.0 / rank
            leader = ordered[0]
            pairs.append((probabilities[leader], 1.0 if leader == lst.truth else 0.0))
        n = len(items) or 1
        return {
            "top1": hits1 / n,
            "top5": hits5 / n,
            "mrr": reciprocal / n,
            "count": len(items),
            "_pairs": pairs,
        }

    out: dict[str, Any] = {"combiner": {}, "rrf": {}, "l2": l2}
    all_pairs: list[tuple[float, float]] = []
    mixed_items: list[tuple[CombinerList, AttributionCombiner]] = []
    for bucket, items in combiner_lists.items():
        result = scored(items)
        all_pairs += result.pop("_pairs")
        out["combiner"][bucket] = result
        out["rrf"][bucket] = _rank_metrics([lst for lst, _ in items], None)
        mixed_items += items
    mixed = scored(mixed_items)
    mixed.pop("_pairs")
    out["combiner"]["mixed"] = mixed
    out["rrf"]["mixed"] = _rank_metrics([lst for lst, _ in mixed_items], None)
    positives = [p for p, y in all_pairs if y > 0]
    negatives = [p for p, y in all_pairs if y == 0]
    brier = float(np.mean([(p - y) ** 2 for p, y in all_pairs])) if all_pairs else 0.0
    auc = (
        sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in positives for q in negatives)
        / (len(positives) * len(negatives))
        if positives and negatives
        else 0.0
    )
    out["calibration"] = {
        "brier": brier,
        "auc": auc,
        "leader_correct_share": (len(positives) / len(all_pairs)) if all_pairs else 0.0,
    }
    if null_pairs:
        pos = [p for p, y in null_pairs if y > 0]
        neg = [p for p, y in null_pairs if y == 0]
        out["null_option"] = {
            "unknown_lists": len(pos),
            "known_lists": len(neg),
            "mean_null_probability_unknown": float(np.mean(pos)) if pos else 0.0,
            "mean_null_probability_known": float(np.mean(neg)) if neg else 0.0,
            "auc": (
                sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
                / (len(pos) * len(neg))
                if pos and neg
                else 0.0
            ),
        }
    return out


# ----------------------------------------------------------------------------- main


# fold seeds kept for evaluation and design (docs 21.3.13-21.3.15): never fitted on
EVALUATION_SEEDS = frozenset({7, 41, 43, 47, 53, 59, 61})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 23, 37],
        help="training fold seeds (7 is evaluation-only)",
    )
    parser.add_argument("--variants", type=int, default=5, help="randomised regimes per fold")
    parser.add_argument(
        "--unknown-all", action="store_true", help="unknown-actor lists for every fold"
    )
    parser.add_argument(
        "--unknown", type=int, default=0, help="unknown-actor lists (first N folds)"
    )
    parser.add_argument("--coverage", type=float, default=0.4)
    parser.add_argument("--min-behaviors", type=int, default=5)
    parser.add_argument("--min-techniques", type=int, default=5)
    parser.add_argument("--max-observations", type=int, default=12)
    parser.add_argument("--limit", type=int, help="folds per seed (smoke runs)")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--l2-grid", default="0.3,1,3")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-null-option", action="store_true")
    parser.add_argument(
        "--lists-cache",
        type=Path,
        default=REPO_ROOT / "data" / "models" / "combiner" / "training_lists.jsonl.gz",
    )
    parser.add_argument(
        "--reuse-cache", action="store_true", help="skip generation and fit from the cache"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "models" / "combiner" / "attribution_combiner.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "data" / "audits" / "attribution_combiner_cv.json",
    )
    parser.add_argument("--keep-band", default="0.3,0.8", help="share of accepted techniques kept")
    parser.add_argument(
        "--precision-band", default="0.3,0.8", help="target precision after injection"
    )
    parser.add_argument("--drop-software", type=float, default=0.3)
    parser.add_argument("--thin-behavior", type=float, default=0.3)
    parser.add_argument(
        "--generic-pool",
        action="store_true",
        help="inject generic (widely documented) false positives preferentially",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--lists-only",
        action="store_true",
        help="generate and cache the lists, do not fit (allowed on evaluation seeds, for reports)",
    )
    parser.add_argument(
        "--overwrite-production",
        action="store_true",
        help="allow writing the default (shipped) artifact, lists cache and report paths",
    )
    args = parser.parse_args(argv)
    evaluation = sorted(set(args.seeds) & EVALUATION_SEEDS)
    if evaluation and not args.lists_only:
        parser.error(
            f"seeds {evaluation} are evaluation seeds; fit on 11/23/37 (or use --lists-only)"
        )
    defaults = {
        "--output": (args.output, parser.get_default("output")),
        "--lists-cache": (args.lists_cache, parser.get_default("lists_cache")),
        "--report": (args.report, parser.get_default("report")),
    }
    shipped = [
        flag
        for flag, (value, default) in defaults.items()
        if Path(value).resolve() == Path(default).resolve()
        and not (args.lists_only and flag != "--lists-cache")
    ]
    if shipped and not args.overwrite_production:
        parser.error(
            f"{', '.join(shipped)} would overwrite the shipped files; pass other paths or "
            "--overwrite-production"
        )

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    bench = _benchmark_module()
    overrides = bench.pipeline_overrides("local", args.set)
    # features must come from the plain RRF order: no discriminative blend, no combiner
    overrides.update({"attribution_rerank_blend": 0.0, "attribution_combiner_enabled": False})
    settings = Settings(**overrides)
    kb = AttackKnowledgeBase.load(settings.kb_path)
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings, embedder)
    engine = AttributionEngine(settings, kb, store, None)  # type: ignore[arg-type]
    fold_args = {
        "coverage": args.coverage,
        "min_behaviors": args.min_behaviors,
        "min_techniques": args.min_techniques,
        "max_observations": args.max_observations,
    }
    null_option = not args.no_null_option
    keep_low, keep_high = (float(v) for v in args.keep_band.split(","))
    precision_low, precision_high = (float(v) for v in args.precision_band.split(","))
    regime = Regime(
        keep_low=keep_low,
        keep_high=keep_high,
        precision_low=precision_low,
        precision_high=precision_high,
        drop_software=args.drop_software,
        thin_behavior=args.thin_behavior,
        generic_pool=args.generic_pool,
    )
    stats: dict[str, Any] = {"regime": regime.__dict__}
    if args.reuse_cache and args.lists_cache.is_file():
        with gzip.open(args.lists_cache, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        log(f"loaded {len(records)} lists from {args.lists_cache}")
    else:
        records, stats = generate_lists(
            engine,
            bench,
            seeds=args.seeds,
            variants=args.variants,
            top_n=args.top_n,
            fold_args=fold_args,
            limit=args.limit,
            log=log,
            regime=regime,
        )
        stats["regime"] = regime.__dict__
        if null_option and (args.unknown_all or args.unknown):
            unknown = generate_unknown_lists(
                settings,
                kb,
                store,
                bench,
                seed=args.seeds[0],
                top_n=args.top_n,
                fold_args=fold_args,
                limit=None if args.unknown_all else args.unknown,
                log=log,
            )
            records += unknown
            stats["unknown_lists"] = len(unknown)
        args.lists_cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.lists_cache, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        log(f"wrote {len(records)} lists to {args.lists_cache}")
    if args.lists_only:
        log("lists only: no fit")
        return 0
    if not records:
        print("no training lists", file=sys.stderr)
        return 1
    l2_grid = [float(v) for v in args.l2_grid.split(",")]
    cv = cross_validate(
        records, folds=args.cv_folds, l2_grid=l2_grid, null_option=null_option, seed=97, log=log
    )
    chosen = float(cv["chosen_l2"])
    lists = [CombinerList.from_dict(r) for r in records]
    model = AttributionCombiner.fit(
        lists,
        attack_version=str(getattr(kb, "version", "") or ""),
        pipeline_fingerprint=engine.combiner_fingerprint,
        l2=chosen,
        null_option=null_option,
        top_n=args.top_n,
        metadata={
            "seeds": list(args.seeds),
            # training lists hide each fold from the dense store and the BM25 index
            "lists_leak_control": lists_leak_control(records),
            "variants": args.variants,
            "fold_args": fold_args,
            "generation": stats,
            "cross_validation": cv,
            "records": len(records),
            "feature_count": len(FEATURE_NAMES),
        },
    )
    path = model.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "generated_at": model.metadata["fitted_at"],
                "artifact": str(path),
                "pipeline_fingerprint": engine.combiner_fingerprint,
                "records": len(records),
                "generation": stats,
                "cross_validation": cv,
                "weights": {
                    name: round(float(w), 4)
                    for name, w in zip(FEATURE_NAMES, model.weights, strict=True)
                },
                "null_weights": [
                    round(float(w), 4)
                    for w in (model.null_weights if model.null_weights is not None else [])
                ],
                "top_features": sorted(
                    (
                        (name, round(float(w), 4))
                        for name, w in zip(FEATURE_NAMES, model.weights, strict=True)
                    ),
                    key=lambda item: -abs(item[1]),
                )[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    counts = Counter(
        "unknown" if r["variant"] == -1 else "clean" if r["variant"] == 0 else "randomised"
        for r in records
    )
    log(f"fitted l2={chosen}: {dict(counts)} lists -> {path}; report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
