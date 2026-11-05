"""Fit the learned attribution combiner on ATT&CK-derived folds (never on CTI-HAL).

Training lists come from the same fold machinery as ``scripts/attribution_benchmark.py`` but
with *training seeds* (the benchmark's seed 7 stays evaluation-only) and, per fold, randomised
evidence regimes that mimic real reports: a subset of the accepted techniques, injected
false-positive techniques drawn from an empirical pool, dropped software, thinned behaviour
matches. An unknown-actor arm (the group removed from the knowledge base) trains the "no
candidate" option. Group-level K-fold cross-validation reports out-of-fold ranking metrics
against the reciprocal-rank-fusion order of the same lists before the final refit.

    PYTHONPATH=. .venv/bin/python scripts/fit_attribution_combiner.py --seeds 11 23 37 --variants 5 --unknown-all

Outputs (``--output``, ``--report`` and a ``--lists-cache`` that will be written) replace an
existing file only with ``--overwrite``, and are written outside the repository (a worktree's
``data/models`` symlink) only with ``--allow-shared-output``. The default (shipped) paths
also need ``--overwrite-production``, and evaluation fold seeds are refused unless
``--lists-only`` (lists for reports, no fit).
"""

from __future__ import annotations

import argparse
import copy
import gc
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
from apt_classifier.behavior_text import behavior_text_votes  # noqa: E402
from apt_classifier.combiner import (  # noqa: E402
    FEATURE_NAMES,
    AttributionCombiner,
    CombinerList,
    combiner_inputs,
)
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.embeddings import SentenceTransformerEmbedder  # noqa: E402
from apt_classifier.group_mlp import precompute_draws, refit_without_groups  # noqa: E402
from apt_classifier.knowledge_store.members import (  # noqa: E402
    citation_keys,
    store_members_votes,
)
from apt_classifier.procedure_match import procedure_match_votes  # noqa: E402
from apt_classifier.separation import discriminate, technique_document_counts  # noqa: E402
from apt_classifier.session import AttributionEngine  # noqa: E402
from apt_classifier.text_group import text_group_votes  # noqa: E402
from apt_classifier.vectorstore import ChromaVectorStore  # noqa: E402
from scripts.run_tooling import (  # noqa: E402
    add_exemplar_backend_arguments,
    add_output_guard_arguments,
    enforce_output_guard,
    exemplar_backend_problem,
    exemplar_setting_overrides,
    exemplar_signature,
)


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
    resume: dict[str, Any] | None = None,
    checkpoint: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clean and randomised lists for every training seed; returns (lists, stats).

    ``checkpoint`` (when given) is called after every seed with ``{"records", "pool", "stats"}``;
    ``resume`` is such a checkpoint: its seeds are not generated again and the false-positive
    pool carries over, so a resumed run writes the lists an uninterrupted one would."""
    kb = engine.kb
    base_store = engine.store
    base_text = getattr(engine, "text_group", None)
    base_text_status = getattr(engine, "text_group_status", None)
    base_behavior = getattr(engine, "behavior_text", None)
    base_behavior_status = getattr(engine, "behavior_text_status", None)
    base_procedure = getattr(engine, "procedure_match", None)
    base_procedure_status = getattr(engine, "procedure_match_status", None)
    citations = (
        citation_keys(engine.settings.attack_bundle_path)
        if getattr(engine, "store_members", None) is not None
        else None
    )

    def hide_text(fold: Any) -> None:
        # the store members read the whole store without the posts ATT&CK cites for the fold
        bench.store_case(engine, fold, citations=citations)
        # the text model's group documents hide the fold like the store does (cheap refit)
        if base_text is not None:
            engine.set_text_group(
                base_text.without(
                    kb,
                    group_id=fold.group_id,
                    techniques=fold.held_out_techniques,
                    texts=fold.held_out_texts,
                )
            )
        if base_behavior is not None:
            engine.set_behavior_text(
                base_behavior.without(
                    kb,
                    group_id=fold.group_id,
                    techniques=fold.held_out_techniques,
                    texts=fold.held_out_texts,
                )
            )
        if base_procedure is not None:
            engine.set_procedure_match(
                base_procedure.without(
                    kb,
                    group_id=fold.group_id,
                    techniques=fold.held_out_techniques,
                    texts=fold.held_out_texts,
                )
            )

    base_lexical = getattr(engine, "lexical_index", None)
    records: list[dict[str, Any]] = list((resume or {}).get("records") or [])
    stats: dict[str, Any] = dict((resume or {}).get("stats") or {"seeds": {}, "pool_techniques": 0})
    pool: dict[str, list[float]] = dict((resume or {}).get("pool") or {})
    try:
        for seed in seeds:
            if str(seed) in stats["seeds"]:
                log(f"seed {seed}: resumed from the checkpoint")
                continue
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
                hide_text(fold)
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
                # the variants re-score this fold's context: same hidden text model
                hide_text(fold)
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
            if checkpoint is not None:
                checkpoint({"records": records, "pool": pool, "stats": stats})
    finally:
        engine.store = base_store
        if base_text is not None:
            engine.set_text_group(base_text, base_text_status)
        if base_behavior is not None:
            engine.set_behavior_text(base_behavior, base_behavior_status)
        if base_procedure is not None:
            engine.set_procedure_match(base_procedure, base_procedure_status)
        engine.lexical_index = base_lexical
        if getattr(engine, "store_members", None) is not None:
            engine.clear_store_case()
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
    resume: dict[str, Any] | None = None,
    checkpoint: Any = None,
) -> list[dict[str, Any]]:
    """One removed-actor list per fold of ``seed``. ``checkpoint`` (when given) is called every 25
    folds with ``{"records", "processed"}``; ``resume`` is such a checkpoint (its folds skipped)."""
    folds = bench.build_folds(kb, seed=seed, limit=limit, **fold_args)
    records = list((resume or {}).get("records") or [])
    processed = int((resume or {}).get("processed") or 0)
    if processed:
        log(f"  unknown: resumed after {processed}/{len(folds)} folds")
    started = time.time()
    draws = None
    citations = None
    for index, fold in enumerate(folds, 1):
        if index <= processed:
            continue
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
        if getattr(engine, "text_group", None) is not None:
            engine.set_text_group(
                engine.text_group.without(
                    reduced, group_id=fold.group_id, texts=fold.held_out_texts, remove_group=True
                )
            )
        if getattr(engine, "behavior_text", None) is not None:
            engine.set_behavior_text(
                engine.behavior_text.without(
                    reduced, group_id=fold.group_id, texts=fold.held_out_texts, remove_group=True
                )
            )
        if getattr(engine, "procedure_match", None) is not None:
            engine.set_procedure_match(
                engine.procedure_match.without(
                    reduced, group_id=fold.group_id, texts=fold.held_out_texts, remove_group=True
                )
            )
        if getattr(engine, "store_members", None) is not None:
            if citations is None:
                citations = citation_keys(settings.attack_bundle_path)
            # the whole store without the actor and the posts ATT&CK cites for the fold
            bench.store_case(engine, fold, citations=citations, remove_actor=True)
        state = engine.analyze(bench._ledger(fold))  # noqa: SLF001
        fusion = state.deterministic_fusion or state.fusion
        if fusion.entries:
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
        # one engine (and knowledge-base copy) per fold: release it before the next one is built
        del engine, state, reduced, proxy
        gc.collect()
        if index % 25 == 0:
            log(f"  unknown {index}/{len(folds)} ({time.time() - started:.0f} s)")
            if checkpoint is not None:
                checkpoint({"records": records, "processed": index})
    return records


@dataclass(frozen=True)
class RealCondition:
    """How a development report is read for its training list: the anonymization level and the
    statement cap (``None`` = every statement). The defaults mirror the two test conditions:
    CTI-HAL's headline (names, the first 30 statements) and Elastic's (strict, all)."""

    level: str
    cap: int | None

    @classmethod
    def parse(cls, value: str) -> RealCondition:
        level, _, cap = value.partition(":")
        return cls(level=level, cap=None if cap in ("", "all") else int(cap))

    @property
    def label(self) -> str:
        return f"{self.level}:{self.cap if self.cap is not None else 'all'}"


DEFAULT_REAL_CONDITIONS = (RealCondition("names", 30), RealCondition("strict", None))


def load_real_folds(
    bench: Any, kb: AttackKnowledgeBase, path: Path, condition: RealCondition
) -> list[Any]:
    """A development report set read under one condition (as the benchmark reads it)."""
    from scripts.report_anonymizer import ReportAnonymizer

    dropper = (
        ReportAnonymizer(kb, condition.level)
        if condition.level in ("strict", "tradecraft")
        else None
    )
    folds = bench.load_report_folds(path, max_observations=condition.cap, anonymizer=dropper, kb=kb)
    folds = [fold for fold in folds if fold.group_id is not None and fold.group_id in kb.groups]
    folds, _, _ = bench.anonymize_report_folds(folds, kb, level=condition.level)
    return folds


def _hide_members(engine: AttributionEngine, kb: Any, fold: Any, *, remove_group: bool) -> None:
    """The text-based members without the report's cited / copied texts (and, for the unknown
    arm, without the actor)."""
    texts = frozenset(fold.hidden_raw)
    for attribute, setter in (
        ("text_group", engine.set_text_group),
        ("behavior_text", engine.set_behavior_text),
        ("procedure_match", engine.set_procedure_match),
    ):
        member = getattr(engine, attribute, None)
        if member is not None:
            setter(
                member.without(kb, group_id=fold.group_id, texts=texts, remove_group=remove_group)
            )


def _known_real_lists(
    engine: AttributionEngine,
    kb: AttackKnowledgeBase,
    bench: Any,
    sets: dict[str, Path],
    conditions: list[RealCondition],
    top_n: int,
    log: Any,
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    skip: frozenset[tuple[str, str]] = frozenset(),
    on_done: Any = None,
) -> None:
    base_store = engine.store
    base_lexical = getattr(engine, "lexical_index", None)
    saved = {
        name: (getattr(engine, name, None), getattr(engine, f"{name}_status", None))
        for name in ("text_group", "behavior_text", "procedure_match")
    }
    setters = {
        "text_group": engine.set_text_group,
        "behavior_text": engine.set_behavior_text,
        "procedure_match": engine.set_procedure_match,
    }
    try:
        for set_name, path in sets.items():
            for variant, condition in enumerate(conditions):
                if (set_name, condition.label) in skip:
                    continue
                folds = load_real_folds(bench, kb, path, condition)
                started = time.time()
                outside = 0
                hidden = {"technique": 0, "behavior": 0, "profile": 0, "lexical": 0}
                for fold in folds:
                    for name, (member, status) in saved.items():
                        if member is not None:
                            setters[name](member, status)
                    _hide_members(engine, kb, fold, remove_group=False)
                    engine.lexical_index = base_lexical
                    hidden["lexical"] += bench.hide_lexical(engine, fold.hidden_texts) or 0
                    proxy = bench.FoldStore(
                        base_store,
                        group_id=fold.group_id,
                        held_out_techniques=frozenset(),
                        held_out_texts=fold.hidden_texts,
                        any_group_texts=fold.hidden_texts,
                    )
                    engine.store = proxy
                    # a dated report's store members read the store before its date; an
                    # undated one (campaigns, advisories) reads none of it
                    bench.store_case(engine, fold)
                    with bench.hidden_campaign(engine.kb, fold.hidden_campaign):
                        state = engine.analyze(bench._ledger(fold))  # noqa: SLF001
                        lst = list_from_state(
                            engine,
                            state.classifier_results,
                            state.deterministic_fusion or state.fusion,
                            state.context,
                            top_n=top_n,
                            truth=fold.group_id,
                        )
                    outside += lst.truth_index is None
                    for key, value in proxy.dropped.items():
                        hidden[key] += value
                    records.append(
                        {
                            "seed": None,
                            "group_id": fold.group_id,
                            "variant": variant,
                            "source": f"real:{set_name}",
                            "condition": condition.label,
                            "report_id": fold.report_id,
                            "leak_control": "dense+lexical+members (cited/copied texts)",
                            **lst.to_dict(),
                        }
                    )
                stats["sets"].setdefault(set_name, {})[condition.label] = {
                    "reports": len(folds),
                    "truth_outside_top_n": outside,
                    "hidden_hits": hidden,
                    "seconds": round(time.time() - started, 1),
                }
                log(
                    f"real {set_name} {condition.label}: {len(folds)} lists, truth outside the "
                    f"top {top_n} in {outside}; hidden hits {hidden}"
                )
                if on_done is not None:
                    on_done(records)
    finally:
        engine.store = base_store
        engine.lexical_index = base_lexical
        for name, (member, status) in saved.items():
            if member is not None:
                setters[name](member, status)
        if getattr(engine, "store_members", None) is not None:
            engine.clear_store_case()


def generate_real_lists(
    engine: AttributionEngine | None,
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    bench: Any,
    *,
    sets: dict[str, Path],
    conditions: Iterable[RealCondition],
    unknown: bool,
    top_n: int,
    log: Any,
    known: bool = True,
    skip_known: frozenset[tuple[str, str]] = frozenset(),
    skip_unknown: frozenset[str] = frozenset(),
    on_done: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One list per development report and condition (truth = its group; skipped without
    ``known``), plus, with ``unknown``, one removed-actor list per report under the first
    condition (truth = None).
    Each report hides the ATT&CK texts that cite or copy it and its own statements (the
    benchmark's development-set rule); a campaign report also hides its campaign. ``on_done``
    is called with the lists so far after every set and condition (a checkpoint); the
    ``skip_*`` pairs and sets were done by an earlier run."""
    conditions = list(conditions)
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"sets": {}, "conditions": [c.label for c in conditions]}
    if known:
        assert engine is not None, "the known lists need the engine"
        _known_real_lists(
            engine, kb, bench, sets, conditions, top_n, log, records, stats, skip_known, on_done
        )
    if unknown:
        import gc

        import torch

        def release() -> None:
            # each removed-actor engine refits the forest on the GPU next to the technique head
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        release()
        unknown_count = 0
        for set_name, path in sets.items():
            if set_name in skip_unknown:
                continue
            folds = load_real_folds(bench, kb, path, conditions[0])
            started = time.time()
            for fold in folds:
                release()
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
                    held_out_techniques=frozenset(),
                    held_out_texts=fold.hidden_texts,
                    hide_group_entirely=True,
                    any_group_texts=fold.hidden_texts,
                )
                reduced_engine = AttributionEngine(settings, reduced, proxy, None)  # type: ignore[arg-type]
                bench.hide_lexical(reduced_engine, fold.hidden_texts)
                if bench._forest_votes(reduced_engine):  # noqa: SLF001
                    reduced_engine.set_profile_forest(
                        bench._refit_forest_without_actor(reduced_engine, reduced)  # noqa: SLF001
                    )
                if getattr(reduced_engine, "group_mlp", None) is not None:
                    reduced_engine.set_group_mlp(
                        refit_without_groups(reduced_engine.group_mlp, reduced)
                    )
                _hide_members(reduced_engine, reduced, fold, remove_group=True)
                bench.store_case(reduced_engine, fold, remove_actor=True)
                with bench.hidden_campaign(reduced, fold.hidden_campaign):
                    state = reduced_engine.analyze(bench._ledger(fold))  # noqa: SLF001
                    fusion = state.deterministic_fusion or state.fusion
                    if not fusion.entries:
                        continue
                    lst = list_from_state(
                        reduced_engine,
                        state.classifier_results,
                        fusion,
                        state.context,
                        top_n=top_n,
                        truth=None,
                    )
                unknown_count += 1
                records.append(
                    {
                        "seed": None,
                        "group_id": fold.group_id,
                        "variant": -1,
                        "source": f"real:{set_name}",
                        "condition": conditions[0].label,
                        "report_id": fold.report_id,
                        "leak_control": "dense+lexical+members (cited/copied texts), actor removed",
                        **lst.to_dict(),
                    }
                )
                del reduced_engine, reduced, proxy, state
            log(f"real {set_name} unknown: {len(folds)} reports ({time.time() - started:.0f} s)")
            if on_done is not None:
                on_done(records)
        stats["unknown_lists"] = unknown_count
    return records, stats


def is_real(record: dict[str, Any]) -> bool:
    return str(record.get("source") or "").startswith("real:")


def trainable(record: dict[str, Any]) -> bool:
    """A list the fit can use: the unknown arm, or a known list whose truth is in it."""
    return record.get("truth") is None or record["truth"] in record.get("group_ids", [])


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
    real_weights: Iterable[int] = (0,),
    fold_top1_margin: float = 0.01,
) -> dict[str, Any]:
    """Group-level K-fold: every list of a group (clean, randomised, unknown, real) sits in one
    split. Without real-report lists the l2 with the best mixed fold MRR is chosen (as before).
    With them, each (real weight, l2) is scored out of fold; the reference is the fold clean
    top-1 of weight 0 at its best-mixed-MRR l2, and among the points whose fold clean top-1 is
    within ``fold_top1_margin`` of it the best real-report MRR wins (ties: the smaller weight,
    then the larger l2)."""
    usable = [r for r in records if trainable(r)]
    has_real = any(is_real(r) for r in usable)
    weights = sorted({int(w) for w in real_weights}) if has_real else [0]
    groups = sorted({r["group_id"] for r in usable})
    rng = random.Random(seed)
    rng.shuffle(groups)
    split_of = {g: i % folds for i, g in enumerate(groups)}
    report: dict[str, Any] = {"folds": folds, "groups": len(groups), "grid": {}}
    entries: list[tuple[int, float, dict[str, Any]]] = []
    for weight in weights:
        for l2 in l2_grid:
            metrics = _cv_metrics(usable, split_of, folds, l2, null_option, real_weight=weight)
            key = f"w{weight}_l2{l2}" if has_real else str(l2)
            report["grid"][key] = metrics
            entries.append((weight, l2, metrics))
            real = metrics["combiner"].get("real")
            log(
                f"w={weight} l2={l2}: clean top1 {metrics['combiner']['clean']['top1']:.3f} "
                f"(rrf {metrics['rrf']['clean']['top1']:.3f}), randomised top1 "
                f"{metrics['combiner']['randomised']['top1']:.3f}, mixed mrr "
                f"{metrics['combiner']['mixed']['mrr']:.3f} (rrf {metrics['rrf']['mixed']['mrr']:.3f})"
                + (
                    f", real top1 {real['top1']:.3f} mrr {real['mrr']:.3f} "
                    f"(rrf {metrics['rrf']['real']['top1']:.3f}/{metrics['rrf']['real']['mrr']:.3f})"
                    if real
                    else ""
                )
                + f", leader auc {metrics['calibration']['auc']:.3f}"
            )
    base = max((e for e in entries if e[0] == 0), key=lambda e: e[2]["combiner"]["mixed"]["mrr"])
    if not has_real:
        report["chosen_l2"] = base[1]
        report["chosen_real_weight"] = 0
        return report
    reference = base[2]["combiner"]["clean"]["top1"]
    eligible = [
        e for e in entries if e[2]["combiner"]["clean"]["top1"] >= reference - fold_top1_margin
    ]
    chosen = max(eligible, key=lambda e: (e[2]["combiner"]["real"]["mrr"], -e[0], e[1]))
    report["selection"] = {
        "rule": "best real-report MRR among points within the fold clean top-1 margin of weight 0",
        "reference": {"real_weight": 0, "l2": base[1], "clean_top1": reference},
        "margin": fold_top1_margin,
        "eligible": [f"w{e[0]}_l2{e[1]}" for e in eligible],
    }
    report["chosen_l2"] = chosen[1]
    report["chosen_real_weight"] = chosen[0]
    return report


def _cv_metrics(
    records: list[dict[str, Any]],
    split_of: dict[str, int],
    folds: int,
    l2: float,
    null_option: bool,
    *,
    real_weight: int = 0,
) -> dict[str, Any]:
    buckets: dict[str, list[tuple[CombinerList, AttributionCombiner]]] = {
        "clean": [],
        "randomised": [],
    }
    real_buckets: dict[str, list[tuple[CombinerList, AttributionCombiner]]] = {}
    null_pairs: list[tuple[float, float]] = []
    real_null_pairs: list[tuple[float, float]] = []
    for split in range(folds):
        train = [
            CombinerList.from_dict(r)
            for r in records
            if split_of[r["group_id"]] != split and not is_real(r)
        ]
        if real_weight:
            real_train = [
                CombinerList.from_dict(r)
                for r in records
                if split_of[r["group_id"]] != split and is_real(r)
            ]
            train += real_train * real_weight
        model = AttributionCombiner.fit(
            train, attack_version="cv", pipeline_fingerprint="cv", l2=l2, null_option=null_option
        )
        for record in records:
            if split_of[record["group_id"]] != split:
                continue
            lst = CombinerList.from_dict(record)
            real = is_real(record)
            if record["variant"] == -1:
                if null_option:
                    pair = (float(model.probabilities(lst)[1] or 0.0), 1.0)
                    (real_null_pairs if real else null_pairs).append(pair)
                continue
            if real:
                for key in (
                    "real",
                    f"real {record.get('condition')}",
                    f"real {str(record['source']).removeprefix('real:')}",
                ):
                    real_buckets.setdefault(key, []).append((lst, model))
                if null_option:
                    real_null_pairs.append((float(model.probabilities(lst)[1] or 0.0), 0.0))
                continue
            bucket = "clean" if record["variant"] == 0 else "randomised"
            buckets[bucket].append((lst, model))
            if null_option:
                null_pairs.append((float(model.probabilities(lst)[1] or 0.0), 0.0))

    def scored(items: list[tuple[CombinerList, AttributionCombiner]]) -> dict[str, Any]:
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

    def auc(pairs: list[tuple[float, float]]) -> float:
        pos = [p for p, y in pairs if y > 0]
        neg = [p for p, y in pairs if y == 0]
        if not pos or not neg:
            return 0.0
        return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (
            len(pos) * len(neg)
        )

    out: dict[str, Any] = {"combiner": {}, "rrf": {}, "l2": l2, "real_weight": real_weight}
    all_pairs: list[tuple[float, float]] = []
    mixed_items: list[tuple[CombinerList, AttributionCombiner]] = []
    for bucket, items in buckets.items():
        result = scored(items)
        all_pairs += result.pop("_pairs")
        out["combiner"][bucket] = result
        out["rrf"][bucket] = _rank_metrics([lst for lst, _ in items], None)
        mixed_items += items
    mixed = scored(mixed_items)
    mixed.pop("_pairs")
    out["combiner"]["mixed"] = mixed
    out["rrf"]["mixed"] = _rank_metrics([lst for lst, _ in mixed_items], None)
    real_pairs: list[tuple[float, float]] = []
    for bucket, items in sorted(real_buckets.items()):
        result = scored(items)
        pairs = result.pop("_pairs")
        if bucket == "real":
            real_pairs = pairs
        out["combiner"][bucket] = result
        out["rrf"][bucket] = _rank_metrics([lst for lst, _ in items], None)
    brier = float(np.mean([(p - y) ** 2 for p, y in all_pairs])) if all_pairs else 0.0
    out["calibration"] = {
        "brier": brier,
        "auc": auc(all_pairs),
        "leader_correct_share": (
            (sum(1 for _, y in all_pairs if y > 0) / len(all_pairs)) if all_pairs else 0.0
        ),
    }
    if real_pairs:
        out["calibration_real"] = {
            "brier": float(np.mean([(p - y) ** 2 for p, y in real_pairs])),
            "auc": auc(real_pairs),
            "leader_correct_share": sum(1 for _, y in real_pairs if y > 0) / len(real_pairs),
        }
    for name, pairs in (("null_option", null_pairs), ("null_option_real", real_null_pairs)):
        if not pairs:
            continue
        pos = [p for p, y in pairs if y > 0]
        neg = [p for p, y in pairs if y == 0]
        out[name] = {
            "unknown_lists": len(pos),
            "known_lists": len(neg),
            "mean_null_probability_unknown": float(np.mean(pos)) if pos else 0.0,
            "mean_null_probability_known": float(np.mean(neg)) if neg else 0.0,
            "auc": auc(pairs),
        }
    return out


# ----------------------------------------------------------------------------- main


# fold seeds kept for evaluation and design (docs 21.3.13-21.3.15, the member gates of 21.3.19
# and 21.3.20, the real-report combiner gate, the probability floor gate, the procedure-core
# gate): never fitted on
EVALUATION_SEEDS = frozenset(
    {7, 41, 43, 47, 53, 59, 61, 67, 73, 79, 83, 89, 101, 103, 107, 109, 113, 127}
)
# report sets that are tests: never a source of training lists
TEST_REPORT_SETS = ("cti_hal", "elastic")


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
        default=REPO_ROOT / "data" / "models" / "combiner" / "training_lists_fsum.jsonl.gz",
    )
    parser.add_argument(
        "--reuse-cache", action="store_true", help="skip generation and fit from the cache"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "models" / "combiner" / "attribution_combiner_fsum.json",
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
    parser.add_argument(
        "--real-reports",
        nargs="+",
        default=[],
        metavar="SET=PATH",
        help="development report sets (vendor, campaigns, advisories) whose lists join the fold "
        "lists; never CTI-HAL or Elastic",
    )
    parser.add_argument(
        "--real-conditions",
        default=",".join(c.label for c in DEFAULT_REAL_CONDITIONS),
        help="level:cap per real-report reading (cap 'all' = every statement)",
    )
    parser.add_argument(
        "--real-unknown", action="store_true", help="removed-actor lists for the real reports"
    )
    parser.add_argument("--real-cache", type=Path, help="gzip JSONL of the real-report lists")
    parser.add_argument(
        "--reuse-real-cache", action="store_true", help="read the real-report lists from the cache"
    )
    parser.add_argument(
        "--real-weights",
        default="0,3,10,30",
        help="cross-validated copies of each real-report list in the fit",
    )
    parser.add_argument(
        "--fold-top1-margin",
        type=float,
        default=0.01,
        help="fold clean top-1 a real-report weight may lose against weight 0",
    )
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
    add_output_guard_arguments(parser)
    add_exemplar_backend_arguments(parser)
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
    real_sets: dict[str, Path] = {}
    for item in args.real_reports:
        name, _, value = item.partition("=")
        if not value:
            parser.error(f"--real-reports takes SET=PATH, got {item!r}")
        if name in TEST_REPORT_SETS or any(t in value for t in ("cti_hal", "elastic")):
            parser.error(f"{item}: CTI-HAL and Elastic are test sets, never training lists")
        real_sets[name] = Path(value)
    if real_sets and args.real_cache is None:
        parser.error("--real-reports needs --real-cache")
    real_conditions = [RealCondition.parse(v) for v in args.real_conditions.split(",") if v]
    # a reused cache is only read; every other output is written at the end of an hour-long run
    # (--lists-only writes the cache alone: no artifact, no report)
    writes_cache = not (args.reuse_cache and args.lists_cache.is_file())
    writes_real_cache = bool(real_sets) and not (
        args.reuse_real_cache and args.real_cache is not None and args.real_cache.is_file()
    )
    enforce_output_guard(
        parser,
        args,
        [
            None if args.lists_only else args.output,
            None if args.lists_only else args.report,
            args.lists_cache if writes_cache else None,
            args.real_cache if writes_real_cache else None,
        ],
        repo_root=REPO_ROOT,
    )

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    bench = _benchmark_module()
    overrides = bench.pipeline_overrides("local", args.set)
    # features must come from the plain RRF order: no discriminative blend, no combiner
    overrides.update({"attribution_rerank_blend": 0.0, "attribution_combiner_enabled": False})
    overrides.update(exemplar_setting_overrides(args))
    settings = Settings(**overrides)
    kb = AttackKnowledgeBase.load(settings.kb_path)
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings, embedder)
    engine = AttributionEngine(settings, kb, store, None)  # type: ignore[arg-type]
    problem = exemplar_backend_problem(engine.exemplar_index, args.require_exemplar_backend)
    if problem:
        parser.error(f"{problem} ({engine.exemplar_status})")
    if text_group_votes(settings) and engine.text_group is None:
        raise SystemExit(
            f"refusing to fit: TEXT_GROUP_WEIGHT > 0 but the text model is "
            f"{engine.text_group_status!r} (build it with make fit-text-group)"
        )
    if behavior_text_votes(settings) and engine.behavior_text is None:
        raise SystemExit(
            f"refusing to fit: BEHAVIOR_TEXT_WEIGHT > 0 but the behaviour-text member is "
            f"{engine.behavior_text_status!r} (it reads TEXT_GROUP_PATH)"
        )
    if store_members_votes(settings) and engine.store_members is None:
        raise SystemExit(
            f"refusing to fit: a store member's weight is > 0 but its artifact is "
            f"{engine.store_members_status!r} (scripts/knowledge_store.py members)"
        )
    if procedure_match_votes(settings) and engine.procedure_match is None:
        raise SystemExit(
            f"refusing to fit: PROCEDURE_MATCH_WEIGHT > 0 but the procedure member is "
            f"{engine.procedure_match_status!r} (it reads TEXT_GROUP_PATH)"
        )
    stamp = args.lists_cache.with_name(args.lists_cache.name + ".pipeline")
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
    # what the artifact records about the engine, kept so the engine can be released before the
    # memory-heavy steps (the removed-actor engines, the fold lists, the cross-validation)
    fingerprint = engine.combiner_fingerprint
    exemplar_status = engine.exemplar_status
    exemplar_store = exemplar_signature(engine.exemplar_index)
    exemplar_problem = exemplar_backend_problem(
        engine.exemplar_index, args.require_exemplar_backend
    )
    generate_folds = not (args.reuse_cache and args.lists_cache.is_file())

    def release_engine() -> None:
        nonlocal engine
        if engine is not None and not generate_folds:
            import gc

            engine = None
            gc.collect()

    records: list[dict[str, Any]] = []
    real_records: list[dict[str, Any]] = []
    if real_sets:
        real_stamp = args.real_cache.with_name(args.real_cache.name + ".pipeline")

        def write_real(records_out: list[dict[str, Any]]) -> None:
            args.real_cache.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(args.real_cache, "wt", encoding="utf-8") as handle:
                for record in records_out:
                    handle.write(json.dumps(record) + "\n")
            real_stamp.write_text(fingerprint + "\n")
            log(f"wrote {len(records_out)} real-report lists to {args.real_cache}")

        existing: list[dict[str, Any]] = []
        if not writes_real_cache:
            cached_stamp = real_stamp.read_text().strip() if real_stamp.is_file() else None
            if cached_stamp != fingerprint:
                raise SystemExit(
                    f"refusing --reuse-real-cache: {args.real_cache} was generated by pipeline "
                    f"{cached_stamp}, this run is {fingerprint}"
                )
            with gzip.open(args.real_cache, "rt", encoding="utf-8") as handle:
                existing = [json.loads(line) for line in handle if line.strip()]
            log(f"loaded {len(existing)} real-report lists from {args.real_cache}")
        # a cache is resumed: the sets and conditions it holds are not generated again
        done_known = frozenset(
            (str(r["source"]).removeprefix("real:"), str(r.get("condition")))
            for r in existing
            if r["variant"] != -1
        )
        done_unknown = frozenset(
            str(r["source"]).removeprefix("real:") for r in existing if r["variant"] == -1
        )
        missing_known = [
            (name, c.label)
            for name in real_sets
            for c in real_conditions
            if (name, c.label) not in done_known
        ]
        if missing_known:
            # checkpointed after every set and condition, before the removed-actor arm
            new_known, real_stats = generate_real_lists(
                engine,
                settings,
                kb,
                store,
                bench,
                sets=real_sets,
                conditions=real_conditions,
                unknown=False,
                top_n=args.top_n,
                log=log,
                skip_known=done_known,
                on_done=lambda so_far: write_real(existing + so_far),
            )
            stats["real"] = real_stats
            existing = existing + new_known
        missing_unknown = [name for name in real_sets if name not in done_unknown]
        if args.real_unknown and null_option and missing_unknown:
            release_engine()
            new_unknown, unknown_stats = generate_real_lists(
                engine,
                settings,
                kb,
                store,
                bench,
                sets=real_sets,
                conditions=real_conditions,
                unknown=True,
                top_n=args.top_n,
                log=log,
                known=False,
                skip_unknown=done_unknown,
                on_done=lambda so_far: write_real(existing + so_far),
            )
            stats.setdefault("real", {})["unknown_lists"] = unknown_stats.get("unknown_lists", 0)
            existing = existing + new_unknown
        real_records = existing
        stats["real_counts"] = dict(
            Counter(
                f"{r['source']} {r.get('condition')} "
                + ("unknown" if r["variant"] == -1 else "in list" if trainable(r) else "outside")
                for r in real_records
            )
        )
    if args.reuse_cache and args.lists_cache.is_file():
        cached_stamp = stamp.read_text().strip() if stamp.is_file() else None
        if cached_stamp is not None and cached_stamp != fingerprint:
            raise SystemExit(
                f"refusing --reuse-cache: {args.lists_cache} was generated by pipeline "
                f"{cached_stamp}, this run is {fingerprint}"
            )
        with gzip.open(args.lists_cache, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        log(f"loaded {len(records)} lists from {args.lists_cache}")
    else:
        # the fold lists are checkpointed after every seed and every 25 removed-actor folds (a
        # memory watchdog may stop a long run); a checkpoint of another pipeline is ignored
        partial = args.lists_cache.with_name(args.lists_cache.name + ".partial")
        saved: dict[str, Any] = {}
        if partial.is_file():
            with gzip.open(partial, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved.get("pipeline") != fingerprint:
                log(f"ignoring {partial}: another pipeline")
                saved = {}
            else:
                log(f"resuming from {partial}")

        def save(**update: Any) -> None:
            saved.update(update, pipeline=fingerprint)
            partial.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial.with_name(partial.name + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(saved, handle)
            temporary.replace(partial)

        records, fold_stats = generate_lists(
            engine,
            bench,
            seeds=args.seeds,
            variants=args.variants,
            top_n=args.top_n,
            fold_args=fold_args,
            limit=args.limit,
            log=log,
            regime=regime,
            resume=saved.get("folds"),
            checkpoint=lambda state: save(folds=state),
        )
        stats.update(fold_stats)
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
                resume=saved.get("unknown"),
                checkpoint=lambda state: save(unknown=state),
            )
            records += unknown
            stats["unknown_lists"] = len(unknown)
        args.lists_cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.lists_cache, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        stamp.write_text(fingerprint + "\n")
        partial.unlink(missing_ok=True)
        log(f"wrote {len(records)} lists to {args.lists_cache}")
    if args.lists_only:
        log("lists only: no fit")
        return 0
    release_engine()
    records = records + real_records
    if not records:
        print("no training lists", file=sys.stderr)
        return 1
    l2_grid = [float(v) for v in args.l2_grid.split(",")]
    cv = cross_validate(
        records,
        folds=args.cv_folds,
        l2_grid=l2_grid,
        null_option=null_option,
        seed=97,
        log=log,
        real_weights=[int(v) for v in args.real_weights.split(",") if v != ""],
        fold_top1_margin=args.fold_top1_margin,
    )
    chosen = float(cv["chosen_l2"])
    real_weight = int(cv.get("chosen_real_weight") or 0)
    lists = [CombinerList.from_dict(r) for r in records if not is_real(r) and trainable(r)]
    lists += [
        CombinerList.from_dict(r) for r in records if is_real(r) and trainable(r)
    ] * real_weight
    model = AttributionCombiner.fit(
        lists,
        attack_version=str(getattr(kb, "version", "") or ""),
        pipeline_fingerprint=fingerprint,
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
            "real_reports": {name: str(path) for name, path in real_sets.items()},
            "real_conditions": [c.label for c in real_conditions],
            "real_weight": real_weight,
            "feature_count": len(FEATURE_NAMES),
            # the exemplar library the lists' technique mapping ran with
            "exemplar_status": exemplar_status,
            "exemplar_store": exemplar_store,
        },
    )
    if exemplar_problem:
        print(f"refusing to write {args.output}: {exemplar_problem}", file=sys.stderr)
        return 2
    path = model.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "generated_at": model.metadata["fitted_at"],
                "artifact": str(path),
                "pipeline_fingerprint": fingerprint,
                "records": len(records),
                "real_weight": real_weight,
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
        (
            f"real {r.get('condition')} " + ("unknown" if r["variant"] == -1 else "known")
            if is_real(r)
            else (
                "unknown" if r["variant"] == -1 else "clean" if r["variant"] == 0 else "randomised"
            )
        )
        for r in records
    )
    log(
        f"fitted l2={chosen}, real weight {real_weight}: {dict(counts)} lists -> {path}; "
        f"report {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
