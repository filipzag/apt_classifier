"""Text-in, group-out attribution benchmark over the real pipeline.

For every ATT&CK group with enough documented procedure sentences, a fraction of its
(technique, sentence) behaviours is held out, anonymized (actor names -> "the adversary")
and fed as free-text observations through ``AttributionEngine.analyze``. ``--pipeline local``
(default) is deterministic: verifier off, no OpenRouter calls. ``--pipeline production``
runs the shipped pipeline: the configured technique verifier on the ambiguous candidates,
with call counts, errors and cost recorded per fold. The ensemble itself is deterministic in
both (the LLM zero-shot vote was removed on 2026-09-30). The benchmark then asks whether the
originating group is recovered by the fused ranking (``rank``: what the analyst sees;
``deterministic_rank``: what the stage decision reads; equal since the vote was removed, both
kept so older artifacts compare) and which attribution stage is reached.

Leak control per fold (``FoldStore`` and ``fold_lexical_index``):
- technique-index hits whose text equals a held-out sentence are dropped, from the dense index
  (it embeds the same anonymized sentences), and the BM25 lexical index is rebuilt per fold
  without them (it holds them as ``procedure`` documents; before 2026-10-03 only the dense index
  was filtered, so a held-out sentence was the top BM25 hit for about 99.7% of fold
  observations). Not hidden: the fold group's one-word variants of a held sentence filed under
  other techniques (about 1.4% of queries have one in the BM25 top 3), and, when
  ``TECHNIQUE_INCLUDE_SYNTHETIC`` is on, synthetic paraphrases generated from procedure examples;
- the held-out group's actor-behaviour documents for the held-out techniques are dropped;
- in the *unknown-actor* arm the group is removed from the knowledge base and every one of
  its documents is hidden, so the correct answer is "no named group".

What it does not control: group technique lists, technique-profile centroids and
descriptions still include the held-out techniques; that is the documented ATT&CK answer
key every classifier scores against, so this remains a self-consistency check, not
field attribution accuracy. It replaces the technique-ID version in
``scripts/ensemble_benchmark.py`` as the attribution gate because it exercises retrieval,
acceptance, eligibility, family weights and the stage decision as the product does.

Real-report mode (``--cti-hal``) feeds each report's statements unchanged. About half of the
CTI-HAL reports name their own group in them, which name-sensitive signals (group-description
similarity, actor-behaviour retrieval) can match without reading any tradecraft;
``--anonymize-actors`` applies the folds' replacement to the statements first and records what it
removed, including the software matches lost where a group alias is also a software name
(Carbanak is both G0008 and S0030).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.related import related_groups
from apt_classifier.attribution import AttributionStage
from apt_classifier.classifiers.profile_forest import PROFILE_FOREST_NAME, SOURCE_COUNTRY_NAME
from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evaluation import (
    AttributionEvaluationCase,
    AttributionPrediction,
    bootstrap_interval,
    evaluate_attribution_predictions,
    pipeline_fingerprint,
    source_fingerprint,
    write_artifact,
)
from apt_classifier.evidence.iocs import match_software_aliases
from apt_classifier.evidence.models import EvidenceLedger, EvidenceStatus, Observation, Provenance
from apt_classifier.group_mlp import refit_without_groups
from apt_classifier.llm import OpenRouterClient
from apt_classifier.overlap_cluster import SAME_ACTIVITY_LINKS, link_kind
from apt_classifier.profile_forest import fit_profile_forest
from apt_classifier.session import AttributionEngine
from apt_classifier.vectorstore import (
    ChromaVectorStore,
    VectorHit,
    _actor_name_pattern,
    _sanitize_procedure,
    normalize_procedure_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _evidence_name_pattern(names: tuple[str, ...]) -> re.Pattern[str] | None:
    """The actor-name pattern for *evidence* text (fold statements, real-report statements).

    Like ``vectorstore._actor_name_pattern``, but a name that ends a sentence ("the backdoor known
    as Carbanak.") is replaced too; a dot followed by a word character still protects file and
    family names (Patchwork.dll, Silence.Downloader). The indexed procedure sentences keep the
    index's pattern: changing it there would need a re-index and a combiner refit.
    """
    ordered = sorted(
        (name for name in names if name), key=lambda item: (-len(item), item.casefold())
    )
    if not ordered:
        return None
    return re.compile(
        r"(?<![\w.-])(?:" + "|".join(re.escape(name) for name in ordered) + r")"
        r"(?P<possessive>['’]s)?(?![\w-]|\.\w)",
        re.IGNORECASE,
    )


FINGERPRINTED_SOURCES = (
    REPO_ROOT / "apt_classifier" / "fusion.py",
    REPO_ROOT / "apt_classifier" / "attribution.py",
    REPO_ROOT / "apt_classifier" / "session.py",
    REPO_ROOT / "apt_classifier" / "retrieve.py",
    REPO_ROOT / "apt_classifier" / "platforms.py",
    REPO_ROOT / "apt_classifier" / "classifiers" / "base.py",
    REPO_ROOT / "apt_classifier" / "classifiers" / "naive_bayes.py",
    REPO_ROOT / "apt_classifier" / "classifiers" / "software.py",
    REPO_ROOT / "apt_classifier" / "classifiers" / "strategic_context.py",
    REPO_ROOT / "apt_classifier" / "combiner.py",
    REPO_ROOT / "apt_classifier" / "separation.py",
    REPO_ROOT / "apt_classifier" / "origin.py",
    REPO_ROOT / "apt_classifier" / "profile_forest.py",
    REPO_ROOT / "apt_classifier" / "torch_forest.py",
    REPO_ROOT / "apt_classifier" / "classifiers" / "profile_forest.py",
    REPO_ROOT / "apt_classifier" / "group_mlp.py",
    REPO_ROOT / "apt_classifier" / "overlap_cluster.py",
    REPO_ROOT / "apt_classifier" / "attack" / "related.py",
    REPO_ROOT / "apt_classifier" / "attack" / "same_activity.py",
    REPO_ROOT / "apt_classifier" / "software_questions.py",
)


@dataclass(frozen=True)
class Fold:
    group_id: str
    held_out_techniques: frozenset[str]
    observations: tuple[str, ...]
    held_out_texts: frozenset[str]
    report_id: str = ""


class FoldStore:
    """Vector-store proxy that hides one group's own evidence for a fold."""

    def __init__(
        self,
        store: Any,
        *,
        group_id: str,
        held_out_techniques: frozenset[str],
        held_out_texts: frozenset[str],
        hide_group_entirely: bool = False,
    ) -> None:
        self._store = store
        self.group_id = group_id
        self.held_out_techniques = held_out_techniques
        self.held_out_texts = held_out_texts
        self.hide_group_entirely = hide_group_entirely
        self.dropped = {"technique": 0, "behavior": 0, "profile": 0}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def query_techniques(
        self, texts: list[str], *, n_results: int = 10, fields: tuple[str, ...] | None = None
    ) -> list[list[VectorHit]]:
        extra = len(self.held_out_texts)
        raw = self._store.query_techniques(texts, n_results=n_results + extra, fields=fields)
        output: list[list[VectorHit]] = []
        for hits in raw:
            kept = []
            for hit in hits:
                if normalize_procedure_text(hit.document) in self.held_out_texts:
                    self.dropped["technique"] += 1
                    continue
                kept.append(hit)
            output.append(kept[:n_results])
        return output

    def _hides_behavior(self, hit: VectorHit) -> bool:
        if str(hit.metadata.get("group_id") or "") != self.group_id:
            return False
        if self.hide_group_entirely:
            return True
        technique_id = str(hit.metadata.get("technique_id") or "")
        return technique_id in self.held_out_techniques or (
            normalize_procedure_text(hit.document) in self.held_out_texts
        )

    def query_group_behaviors(
        self, texts: list[str], *, n_results: int = 20
    ) -> list[list[VectorHit]]:
        raw = self._store.query_group_behaviors(texts, n_results=n_results * 2)
        output: list[list[VectorHit]] = []
        for hits in raw:
            kept = []
            for hit in hits:
                if self._hides_behavior(hit):
                    self.dropped["behavior"] += 1
                    continue
                kept.append(hit)
            output.append(kept[:n_results])
        return output

    def query_group_profile(
        self, texts: list[str], *, document_type: str, n_results: int = 20
    ) -> list[VectorHit]:
        hits = self._store.query_group_profile(
            texts, document_type=document_type, n_results=n_results + 1
        )
        if not self.hide_group_entirely:
            return hits[:n_results]
        kept = []
        for hit in hits:
            if str(hit.metadata.get("group_id") or "") == self.group_id:
                self.dropped["profile"] += 1
                continue
            kept.append(hit)
        return kept[:n_results]


LEAK_CONTROL = "dense+lexical"  # recorded in fold artifacts; before 2026-10-03 only "dense"


def fold_lexical_index(index: Any, held_out_texts: frozenset[str]) -> tuple[Any, int]:
    """The BM25 index rebuilt without the documents whose normalized text is held out, and how
    many were removed. Document frequencies and the average length are recomputed, so it equals
    an index built without those sentences (a hit filter would keep their corpus statistics);
    about 0.07 s on the real knowledge base. A sentence can be several documents (a technique
    and its sub-technique, several groups): every one goes."""
    kept = tuple(
        document
        for document in index.documents
        if normalize_procedure_text(document.text) not in held_out_texts
    )
    removed = len(index.documents) - len(kept)
    if not removed:
        return index, 0
    frequencies: Counter[str] = Counter()
    for document in kept:
        frequencies.update(set(document.tokens))
    average = sum(len(d.tokens) for d in kept) / len(kept) if kept else 1.0
    return type(index)(kept, dict(frequencies), average), removed


def hide_lexical(engine: Any, held_out_texts: frozenset[str]) -> int | None:
    """Swap ``engine.lexical_index`` for the fold's filtered index; returns the number of hidden
    documents (``None`` when nothing is hidden). Fails closed when there is something to hide but
    no index while retrieval is lexical or hybrid: retrieval would then build its own,
    unfiltered index. The caller restores the original index afterwards."""
    if not held_out_texts:
        return None
    index = getattr(engine, "lexical_index", None)
    if index is None:
        mode = str(getattr(getattr(engine, "settings", None), "technique_retrieval_mode", "hybrid"))
        if mode in ("hybrid", "lexical"):
            raise RuntimeError(
                f"fold leak control: retrieval mode {mode!r} but the engine has no lexical index "
                "to filter (retrieval would build an unfiltered one)"
            )
        return None
    engine.lexical_index, removed = fold_lexical_index(index, held_out_texts)
    return removed


def actor_names(kb: AttackKnowledgeBase) -> tuple[str, ...]:
    """Every ATT&CK group name and alias, longest first, as the replacement pattern needs."""
    return tuple(
        sorted(
            {
                name.strip()
                for group in kb.groups.values()
                for name in (group.name, *group.aliases)
                if name and name.strip()
            },
            key=lambda item: (-len(item), item.casefold()),
        )
    )


def build_folds(
    kb: AttackKnowledgeBase,
    *,
    coverage: float,
    seed: int,
    min_behaviors: int,
    min_techniques: int,
    max_observations: int,
    limit: int | None = None,
) -> list[Fold]:
    names = actor_names(kb)
    pattern = _evidence_name_pattern(names)
    index_pattern = _actor_name_pattern(names)
    rng = random.Random(seed)
    folds: list[Fold] = []
    for group_id in sorted(kb.groups):
        group = kb.groups[group_id]
        pairs = [
            (technique_id, sentence)
            for technique_id, sentences in sorted(group.technique_behaviors.items())
            if technique_id in kb.techniques
            for sentence in sentences
        ]
        if len(pairs) < min_behaviors or len(group.technique_ids) < min_techniques:
            continue
        rng.shuffle(pairs)
        count = max(1, round(coverage * len(pairs)))
        held = pairs[:count][:max_observations]
        texts = tuple(_sanitize_procedure(sentence, names, pattern=pattern) for _, sentence in held)
        texts = tuple(text for text in texts if len(text.split()) >= 3)
        if not texts:
            continue
        # the technique index holds the same sentences anonymized with the index's pattern; hide
        # both forms so a sentence-end name the evidence pattern removes cannot leak back in
        hidden = {normalize_procedure_text(text) for text in texts} | {
            normalize_procedure_text(_sanitize_procedure(sentence, names, pattern=index_pattern))
            for _, sentence in held
        }
        folds.append(
            Fold(
                group_id=group_id,
                held_out_techniques=frozenset(technique_id for technique_id, _ in held),
                observations=texts,
                held_out_texts=frozenset(hidden),
            )
        )
        if limit is not None and len(folds) >= limit:
            break
    return folds


def load_report_folds(path: Path, *, max_observations: int) -> list[Fold]:
    """Real-report folds from ``scripts/prepare_cti_hal.py`` output (one fold per report).

    Observations are the report's human-annotated statements; the expected group is the
    report's subject. Nothing is hidden from the store: these sentences were never indexed,
    and the group's own documented behaviour is legitimate evidence for a real report.
    """
    folds: list[Fold] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts = tuple(
            str(item["text"]).strip()
            for item in row.get("statements", [])
            if len(str(item.get("text", "")).split()) >= 3
        )[:max_observations]
        if not texts:
            continue
        techniques = frozenset(
            technique_id
            for item in row.get("statements", [])
            for technique_id in item.get("technique_ids", [])
        )
        folds.append(
            Fold(
                group_id=str(row["group_id"]),
                held_out_techniques=techniques,
                observations=texts,
                held_out_texts=frozenset(),
                report_id=str(row.get("report_id") or ""),
            )
        )
    return folds


def anonymize_report_folds(
    folds: list[Fold], kb: AttackKnowledgeBase
) -> tuple[list[Fold], list[dict[str, Any]], dict[str, Any]]:
    """Replace every ATT&CK group name and alias in the report statements with "the adversary".

    The replacement is the held-out folds' (``_sanitize_procedure`` over all names), applied
    after the statements were selected, so both arms read the same statements; a statement
    without a name stays byte-identical. Some aliases are also software names (Carbanak is
    G0008 and S0030), so the software matches the replacement removes are counted with the
    engine's own matcher over the joined narrative, as the classifier context reads it.
    Returns the folds, one record per fold, and the totals.
    """
    names = actor_names(kb)
    pattern = _evidence_name_pattern(names)
    owners: dict[str, set[str]] = {}
    for group in kb.groups.values():
        for name in (group.name, *group.aliases):
            if name and name.strip():
                owners.setdefault(name.strip().casefold(), set()).add(group.id)
    anonymized: list[Fold] = []
    records: list[dict[str, Any]] = []
    for fold in folds:
        texts: list[str] = []
        replaced: Counter[str] = Counter()
        changed = 0
        for text in fold.observations:
            found = list(pattern.finditer(text)) if pattern is not None else []
            if not found:
                texts.append(text)
                continue
            changed += 1
            replaced.update(
                match.group(0).removesuffix(match.group("possessive") or "").casefold()
                for match in found
            )
            texts.append(_sanitize_procedure(text, names, pattern=pattern))
        removed: Counter[str] = Counter()
        lost: set[str] = set()
        if changed:
            before = Counter(
                match.software_id
                for match in match_software_aliases(" ".join(fold.observations), kb)
                if match.software_id
            )
            after = Counter(
                match.software_id
                for match in match_software_aliases(" ".join(texts), kb)
                if match.software_id
            )
            removed, lost = before - after, set(before) - set(after)
        records.append(
            {
                "statements_changed": changed,
                "replacements": sum(replaced.values()),
                "names": dict(sorted(replaced.items())),
                "own_group_named": any(fold.group_id in owners.get(name, ()) for name in replaced),
                "software_matches_removed": sum(removed.values()),
                "software_ids_removed": sorted(lost),
            }
        )
        anonymized.append(replace(fold, observations=tuple(texts)))
    names_replaced: Counter[str] = Counter()
    software_reports: Counter[str] = Counter()
    for record in records:
        names_replaced.update(record["names"])
        software_reports.update(record["software_ids_removed"])
    totals = {
        "replacement": "the adversary",
        "reports": len(folds),
        "reports_changed": sum(1 for record in records if record["statements_changed"]),
        "reports_naming_own_group": sum(1 for record in records if record["own_group_named"]),
        "statements": sum(len(fold.observations) for fold in folds),
        "statements_changed": sum(record["statements_changed"] for record in records),
        "replacements": sum(names_replaced.values()),
        "names": dict(names_replaced.most_common()),
        "software_matches_removed": sum(record["software_matches_removed"] for record in records),
        # software id -> number of reports in which it is no longer detected
        "software_ids_removed": dict(sorted(software_reports.items())),
    }
    return anonymized, records, totals


def sample_folds(folds: list[Fold], count: int, seed: int) -> list[Fold]:
    """Seeded sample of ``count`` folds, round-robin over groups so no group dominates."""
    if count >= len(folds):
        return list(folds)
    rng = random.Random(seed)
    by_group: dict[str, list[Fold]] = {}
    for fold in folds:
        by_group.setdefault(fold.group_id, []).append(fold)
    for members in by_group.values():
        rng.shuffle(members)
    chosen: list[Fold] = []
    while len(chosen) < count:
        for group_id in sorted(by_group):
            if by_group[group_id] and len(chosen) < count:
                chosen.append(by_group[group_id].pop())
    return chosen


def _ledger(fold: Fold) -> EvidenceLedger:
    return EvidenceLedger(
        observations=[
            Observation(text=text, provenance=Provenance(source="attribution_benchmark"))
            for text in fold.observations
        ]
    )


def _llm_accounting(state: Any, client: OpenRouterClient | None, cost_before: float) -> dict:
    diagnostics = dict(getattr(state.retrieval, "diagnostics", None) or {})
    return {
        "verifier_calls": int(diagnostics.get("verifier_calls", 0) or 0),
        "verifier_errors": int(diagnostics.get("verifier_errors", 0) or 0),
        "verifier_cost_usd": round(float(diagnostics.get("verifier_cost_usd", 0.0) or 0.0), 6),
        "cost_usd": round(
            (client.total_usage.cost_usd - cost_before) if client is not None else 0.0, 6
        ),
    }


class LLMUnavailableError(RuntimeError):
    """Every LLM call of the first production fold failed: refuse to produce an artifact."""


def _all_llm_calls_failed(accounting: dict[str, Any]) -> bool:
    calls = int(accounting.get("verifier_calls", 0) or 0)
    errors = int(accounting.get("verifier_errors", 0) or 0)
    return calls > 0 and errors == calls


def _separation_fields(state: Any, truth: str | None, decision: Any) -> dict[str, Any]:
    """Per-row separation diagnostics: the decision's separation, the leader's evidence
    specificity, the best-supported alternative, and the raw per-channel discrimination of the
    top-K so channel weights, floors and bars can be replayed offline
    (``scripts/attribution_gate_sweep.py``)."""
    discrimination = getattr(state.context, "discrimination", None)
    candidates = discrimination.to_dict()["candidates"] if discrimination is not None else []
    ordered = sorted(candidates, key=lambda item: -float(item["total"]))
    truth_rank = next(
        (index for index, item in enumerate(ordered, 1) if truth and item["group_id"] == truth),
        None,
    )
    return {
        "separation": float(getattr(decision, "separation", 0.0) or 0.0) if decision else None,
        "leader_specificity": (
            float((decision.separation_factors or {}).get("leader_specificity", 0.0))
            if decision
            else None
        ),
        "runner_up": getattr(decision, "runner_up_group_id", None) if decision else None,
        "runner_up_related": bool(getattr(decision, "runner_up_related", False))
        if decision
        else False,
        "maturity_factors": dict(getattr(decision, "maturity_factors", {}) or {})
        if decision
        else {},
        "discriminative_truth_rank": truth_rank,
        "topk": candidates,
    }


def _combiner_fields(state: Any, truth: str | None) -> dict[str, Any]:
    """Learned-combiner diagnostics per row: the truth's rank in the RRF order the combiner
    started from, the probability it assigned to its leader, and its status. For offline
    replays also every combiner probability (the RRF top N), the probabilities of the gate
    window's members, the RRF order to place 60 (the head from ``rrf_ranks``; past the head the
    deterministic order is the RRF order), and whether the truth reached the gate window."""
    info = getattr(state.context, "combiner", None)
    discrimination = getattr(state.context, "discrimination", None)
    window = [item.group_id for item in discrimination.scores] if discrimination is not None else []
    window_fields = {"truth_in_window": (truth in window) if truth else None}
    if not isinstance(info, dict):
        return {
            "rrf_rank": None,
            "combiner_leader_probability": None,
            "combiner_status": None,
            **window_fields,
        }
    ranks = info.get("rrf_ranks") or {}
    probabilities = info.get("probabilities") or {}
    deterministic = state.deterministic_fusion or state.fusion
    head = [gid for gid, _ in sorted(ranks.items(), key=lambda item: item[1])]
    tail = [entry.group_id for entry in deterministic.entries[len(head) : 60]]
    return {
        "rrf_rank": ranks.get(truth) if truth else None,
        "combiner_leader_probability": info.get("leader_probability"),
        "combiner_null_probability": info.get("null_probability"),
        "combiner_status": info.get("status"),
        "combiner_probabilities": dict(probabilities),
        "window_probabilities": {gid: probabilities.get(gid, 0.0) for gid in window},
        "rrf_top60": (head + tail)[:60],
        **window_fields,
    }


def _top3_fields(state: Any, ranked: list[str]) -> dict[str, Any]:
    """Top-three diagnostics and exact-replay inputs per row (docs 10.10): for the first three
    places their combiner probability, RRF rank and classifier ranks; the accepted techniques
    with their confidences, the actor-behaviour matches of the gate window (ids and scores only,
    never text), decisive software and negative answers, so ``discriminate`` can be replayed
    offline; and whether the leader-consistency guard fired."""
    context = state.context
    discrimination = getattr(context, "discrimination", None)
    window = (
        {item.group_id for item in discrimination.scores} if discrimination is not None else set()
    )
    info = context.combiner if isinstance(getattr(context, "combiner", None), dict) else {}
    probabilities = info.get("probabilities") or {}
    rrf_ranks = info.get("rrf_ranks") or {}
    deterministic = state.deterministic_fusion or state.fusion
    entries = deterministic.by_group()
    return {
        "top3_detail": [
            {
                "group_id": group_id,
                "combiner_probability": probabilities.get(group_id),
                "rrf_rank": rrf_ranks.get(group_id),
                "classifier_ranks": dict(entries[group_id].classifier_ranks)
                if group_id in entries
                else {},
            }
            for group_id in ranked[:3]
        ],
        "accepted_confidences": {
            technique_id: round(float(confidence), 4)
            for technique_id, confidence in sorted(context.observed_techniques.items())
        },
        "window_behavior_matches": sorted(
            [
                str(match.group_id),
                str(match.observation_id),
                str(match.technique_id),
                round(float(match.score), 4),
            ]
            for match in getattr(context, "specific_behavior_matches", ()) or ()
            if str(getattr(match, "group_id", "")) in window
        ),
        "decisive_software": dict(getattr(context, "decisive_software", {}) or {}),
        "negative_techniques": sorted(getattr(context, "negative_techniques", ()) or ()),
        "leader_guard_fired": bool(
            discrimination is not None
            and deterministic.entries
            and discrimination.leader_id != deterministic.entries[0].group_id
        ),
    }


def _combiner_artifact(settings: Any, engine: Any) -> dict[str, Any] | None:
    """Which combiner artifact scored the run (path, content hash, top N, pipeline fingerprint),
    so a later replay can tell artifacts fitted on different lists apart."""
    model = getattr(engine, "combiner", None)
    path = getattr(settings, "attribution_combiner_path", None)
    if model is None or not path:
        return None
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]
    except OSError:
        digest = None
    return {
        "path": str(path),
        "sha256": digest,
        "top_n": int(getattr(model, "top_n", 0) or 0),
        "pipeline_fingerprint": getattr(model, "pipeline_fingerprint", None),
        # the fold evidence its training lists hid (combiners fitted before 2026-10-03 record
        # none: dense store only, the BM25 index was not filtered)
        "lists_leak_control": (getattr(model, "metadata", None) or {}).get(
            "lists_leak_control", "dense (unrecorded)"
        ),
    }


def _cluster_fields(state: Any, decision: Any) -> dict[str, Any]:
    """The named documented-overlap cluster (if any) and the engine's cluster candidate for the
    leader (recorded whether or not cluster naming is on, so the gate can be replayed)."""
    info = (
        state.context.combiner if isinstance(getattr(state.context, "combiner", None), dict) else {}
    )
    return {
        "named_cluster": list(getattr(decision, "named_cluster", ()) or ()) if decision else [],
        "cluster_separation": getattr(decision, "cluster_separation", None) if decision else None,
        "cluster_discriminative": (
            getattr(decision, "cluster_discriminative", None) if decision else None
        ),
        "cluster_probability": getattr(decision, "cluster_probability", None) if decision else None,
        "cluster_candidate": info.get("cluster"),
    }


def _rank_of(fusion: Any, group_id: str) -> int | None:
    ranked = [entry.group_id for entry in fusion.entries]
    return ranked.index(group_id) + 1 if group_id in ranked else None


# ----- software interview: enumerated simulation (docs 21.3.15 C) -----
SOFTWARE_INTERVIEW_Q = (0.1, 0.3, 1.0)
SOFTWARE_INTERVIEW_FP = (0.0, 0.05)
_STAGE_CODES = {
    "insufficient_evidence": "ins",
    "unresolved_activity_cluster": "unr",
    "known_group_shortlist": "sho",
    "known_group_hypothesis": "hyp",
    "known_cluster_hypothesis": "clu",
}


def _named_members(decision: Any) -> list[str]:
    """The named set of a decision: the group, the cluster's members, or nothing."""
    if decision is None:
        return []
    stage = decision.stage.value
    if stage == AttributionStage.KNOWN_CLUSTER_HYPOTHESIS.value:
        return [str(gid) for gid in decision.named_cluster]
    if stage == AttributionStage.KNOWN_GROUP_HYPOTHESIS.value and decision.leading_group_id:
        return [str(decision.leading_group_id)]
    return []


def enumerate_software_interview(
    engine: Any, state: Any, truth_software: set[str]
) -> dict[str, Any]:
    """Every answer path of the software questions on an analysed case.

    The questions are the engine's own (``AttributionEngine.software_question``: asked while the
    case is not named, at most INTERVIEW_SOFTWARE_MAX_QUESTIONS). Each is answered "found" and, on
    a second branch, "not found" through ``answer_question``, which re-scores the case exactly as
    an analyst answer does; "not checked" records nothing, so it equals not asking and is no
    branch (at most 2^3 leaves). Every node is kept (path, stage, named set, separation) so leaf
    probabilities under any answer model are computed afterwards. ``truth_software`` is the true
    group's full ATT&CK software list (the removed actor's own list for removed-actor rows); only
    which asked items it contains is stored. The state is restored to its analysed form."""
    answers = state.ledger.question_answers
    root = len(answers)
    baseline = _named_members(state.attribution_decision)
    nodes: list[dict[str, Any]] = []
    truth_asked: set[str] = set()

    def visit(path: tuple[str, ...]) -> None:
        decision = state.attribution_decision
        question = engine.software_question(state)
        nodes.append(
            {
                "path": "|".join(path),
                "stage": _STAGE_CODES.get(decision.stage.value, decision.stage.value)
                if decision is not None
                else None,
                "named": _named_members(decision),
                "separation": round(float(decision.separation), 3)
                if decision is not None
                else None,
                "leaf": question is None,
            }
        )
        if question is None:
            return
        if question.software_id in truth_software:
            truth_asked.add(str(question.software_id))
        depth = len(answers)
        for found in (True, False):
            del answers[depth:]
            state.current_question = question
            engine.answer_question(
                state, EvidenceStatus.POSITIVE if found else EvidenceStatus.NEGATIVE
            )
            step = f"{question.software_id}:{question.target_rank}:{'F' if found else 'N'}"
            visit((*path, step))

    try:
        visit(())
    finally:
        del answers[root:]
        state.current_question = None
        if len(nodes) > 1:
            engine.rescore(state)
    return {
        "baseline_named": baseline,
        "truth_documents": sorted(truth_asked),
        "nodes": nodes,
    }


def _path_steps(path: str) -> list[tuple[str, str]]:
    """[(software id, "F" | "N"), ...] of an encoded node path."""
    return [(step.split(":")[0], step.split(":")[-1]) for step in path.split("|")] if path else []


def _path_probability(path: str, truth: set[str], q: float, fp: float) -> float:
    """Answer model: "found" with probability q when the true group's ATT&CK entry lists the
    malware, with probability fp (a wrong "found") otherwise; "not found" in every other case."""
    probability = 1.0
    for software_id, answer in _path_steps(path):
        found = q if software_id in truth else fp
        probability *= found if answer == "F" else 1.0 - found
    return probability


def _software_expectations(
    known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]], q: float, fp: float
) -> dict[str, float]:
    keys = (
        "questions",
        "found_answers",
        "named",
        "correct",
        "gained",
        "wrong_added",
        "lost",
        "removed_questions",
        "removed_found_answers",
        "removed_named",
        "removed_named_added",
    )
    totals = dict.fromkeys(keys, 0.0)
    for arm, rows in (("known", known_rows), ("unknown", unknown_rows)):
        for row in rows:
            simulation = row.get("software_interview")
            if not simulation:
                continue
            truth = set(simulation["truth_documents"])
            base = set(simulation["baseline_named"])
            base_correct = arm == "known" and row["group_id"] in base
            base_wrong = arm == "known" and bool(base) and not base_correct
            for node in simulation["nodes"]:
                if not node["leaf"]:
                    continue
                probability = _path_probability(node["path"], truth, q, fp)
                if probability <= 0:
                    continue
                steps = _path_steps(node["path"])
                found = sum(1 for _, answer in steps if answer == "F")
                named = set(node["named"])
                if arm == "unknown":
                    totals["removed_questions"] += probability * len(steps)
                    totals["removed_found_answers"] += probability * found
                    totals["removed_named"] += probability * bool(named)
                    totals["removed_named_added"] += probability * (bool(named) and not base)
                    continue
                correct = row["group_id"] in named
                wrong = bool(named) and not correct
                totals["questions"] += probability * len(steps)
                totals["found_answers"] += probability * found
                totals["named"] += probability * bool(named)
                totals["correct"] += probability * correct
                totals["gained"] += probability * (correct and not base_correct)
                totals["wrong_added"] += probability * (wrong and not base_wrong)
                totals["lost"] += probability * (base_correct and not correct)
    result = {key: round(value, 4) for key, value in totals.items()}
    result["hit_rate"] = (
        round(totals["found_answers"] / totals["questions"], 4) if totals["questions"] else None
    )
    return result


def summarize_software_interview(
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    *,
    cluster_naming: bool,
) -> dict[str, Any]:
    """Expected values of the enumerated software interview and the pre-registered criteria
    C1-C8 (docs 21.3.15). "Gained", "wrong added" and "lost" compare each leaf with the row's own
    baseline (no netting across rows); every expectation is exact (all leaves enumerated)."""
    expected = {
        f"q={q},fp={fp}": _software_expectations(known_rows, unknown_rows, q, fp)
        for q in SOFTWARE_INTERVIEW_Q
        for fp in SOFTWARE_INTERVIEW_FP
    }
    possible_wrong = 0  # leaves with positive probability at fp = 0 (any 0 < q < 1)
    possible_removed = 0
    lost_leaves = 0  # any leaf (fp > 0 makes every leaf possible)
    all_not_found_changed = 0
    all_not_found_named = 0
    baseline_named = 0
    monotonicity_violations = 0
    rows_asked = questions = leaves = 0
    for arm, rows in (("known", known_rows), ("unknown", unknown_rows)):
        for row in rows:
            simulation = row.get("software_interview")
            if not simulation:
                continue
            truth = set(simulation["truth_documents"])
            base = set(simulation["baseline_named"])
            baseline_named += bool(base)
            base_correct = arm == "known" and row["group_id"] in base
            base_wrong = arm == "known" and bool(base) and not base_correct
            by_path = {node["path"]: node for node in simulation["nodes"]}
            rows_asked += not by_path.get("", {"leaf": True})["leaf"]
            for node in simulation["nodes"]:
                named = set(node["named"])
                steps = _path_steps(node["path"])
                if steps and steps[-1][1] == "N":
                    parent = by_path.get("|".join(node["path"].split("|")[:-1]))
                    parent_named = set(parent["named"]) if parent is not None else set()
                    if named and named != parent_named:
                        monotonicity_violations += 1
                if not node["leaf"]:
                    questions += 1
                    continue
                leaves += 1
                if all(answer == "N" for _, answer in steps):
                    all_not_found_changed += named != base
                    all_not_found_named += bool(named)
                feasible = all(answer == "N" or sid in truth for sid, answer in steps)
                if arm == "known":
                    correct = row["group_id"] in named
                    if feasible and bool(named) and not correct and not base_wrong:
                        possible_wrong += 1
                    if base_correct and not correct:
                        lost_leaves += 1
                elif feasible and named and not base:
                    possible_removed += 1
    zero = {q: expected[f"q={q},fp=0.0"] for q in SOFTWARE_INTERVIEW_Q}
    five = {q: expected[f"q={q},fp=0.05"] for q in SOFTWARE_INTERVIEW_Q}
    c1 = {
        "wrong_added": {str(q): zero[q]["wrong_added"] for q in SOFTWARE_INTERVIEW_Q},
        "removed_named": {str(q): zero[q]["removed_named"] for q in SOFTWARE_INTERVIEW_Q},
        "possible_wrong_leaves": possible_wrong,
        "possible_removed_leaves": possible_removed,
    }
    c1["pass"] = possible_wrong == 0 and possible_removed == 0
    c5 = {
        str(q): round(five[q]["wrong_added"] + five[q]["removed_named_added"], 4)
        for q in SOFTWARE_INTERVIEW_Q
    }
    criteria: dict[str, Any] = {
        "C1": c1,
        "C2": {
            "gained_q0.3": zero[0.3]["gained"],
            "gained_q1.0": zero[1.0]["gained"],
            "pass": zero[0.3]["gained"] >= 8 and zero[1.0]["gained"] >= 30,
        },
        "C3": {
            "lost_leaves": lost_leaves,
            "lost": {key: value["lost"] for key, value in expected.items()},
            "pass": lost_leaves == 0,
        },
        "C4": {
            "baseline_named": baseline_named,
            "all_not_found_named": all_not_found_named,
            "rows_changed": all_not_found_changed,
            "pass": all_not_found_changed == 0,
        },
        "C5": {
            "wrong_plus_removed_fp0.05": c5,
            "pass": all(value <= 1.0 for value in c5.values()),
        },
        "C6": {
            "pass": None,
            "note": "compare the expected names with scripts/software_interview_replay.py",
        },
        "C7": {"violations": monotonicity_violations, "pass": monotonicity_violations == 0},
    }
    criteria["C8"] = (
        {
            "C1": c1["pass"],
            "C5": criteria["C5"]["pass"],
            "pass": c1["pass"] and criteria["C5"]["pass"],
        }
        if cluster_naming
        else {"pass": None, "note": "run with --set attribution_cluster_naming=true"}
    )
    return {
        "mode": "enumerate",
        "answer_model": {
            "found_if_truth_documents": "q",
            "found_otherwise": "fp (wrong found)",
            "not_checked": "same as not asking (no branch)",
            "q": list(SOFTWARE_INTERVIEW_Q),
            "fp": list(SOFTWARE_INTERVIEW_FP),
        },
        "rows_asked": rows_asked,
        "questions": questions,
        "leaves": leaves,
        "expected": expected,
        "criteria": criteria,
    }


def run_known(
    engine: AttributionEngine,
    folds: list[Fold],
    *,
    hide_evidence: bool = True,
    client: OpenRouterClient | None = None,
    software_interview: bool = False,
) -> tuple[list[dict[str, Any]], dict]:
    rows: list[dict[str, Any]] = []
    base_store = engine.store
    base_lexical = getattr(engine, "lexical_index", None)
    related = related_groups(engine.kb)
    dropped_total = {"technique": 0, "behavior": 0, "profile": 0, "lexical": 0}
    try:
        for fold in folds:
            engine.lexical_index = base_lexical
            hidden_lexical = hide_lexical(
                engine, fold.held_out_texts if hide_evidence else frozenset()
            )
            proxy = FoldStore(
                base_store,
                group_id=fold.group_id,
                held_out_techniques=fold.held_out_techniques if hide_evidence else frozenset(),
                held_out_texts=fold.held_out_texts if hide_evidence else frozenset(),
            )
            engine.store = proxy
            cost_before = client.total_usage.cost_usd if client is not None else 0.0
            state = engine.analyze(_ledger(fold))
            # ``fusion`` is what the analyst sees (includes the LLM vote in production);
            # ``deterministic_fusion`` is the ranking before the vote and drives the stage.
            fusion = state.fusion
            deterministic = state.deterministic_fusion or state.fusion
            ranked = [entry.group_id for entry in fusion.entries]
            decision = state.attribution_decision
            observed = set(state.context.observed_techniques)
            accounting = _llm_accounting(state, client, cost_before)
            truth_ranks = {
                result.classifier: (
                    result.by_group()[fold.group_id].rank
                    if fold.group_id in result.by_group()
                    else None
                )
                for result in state.classifier_results
                if result.applicable
            }
            if client is not None and _all_llm_calls_failed(accounting):
                label = fold.report_id or fold.group_id
                if not rows:
                    raise LLMUnavailableError(
                        f"all {accounting['verifier_calls']} verifier calls failed on the first "
                        f"fold ({label}); check the OpenRouter key/credit before running the "
                        "production pipeline"
                    )
                # Credit or the API went away mid-run: keep the folds that were measured with
                # the LLM and stop, rather than silently appending local-only folds.
                dropped_total["llm_unavailable_at"] = label  # type: ignore[assignment]
                print(f"stopping: all LLM calls failed at {label}", file=sys.stderr)
                break
            rows.append(
                {
                    "group_id": fold.group_id,
                    "report_id": fold.report_id,
                    "observations": len(fold.observations),
                    "held_out_techniques": sorted(fold.held_out_techniques),
                    "accepted_techniques": sorted(observed),
                    "technique_recall": (
                        len(observed & fold.held_out_techniques) / len(fold.held_out_techniques)
                        if fold.held_out_techniques
                        else 0.0
                    ),
                    "rank": ranked.index(fold.group_id) + 1 if fold.group_id in ranked else None,
                    "deterministic_rank": _rank_of(deterministic, fold.group_id),
                    **_combiner_fields(state, fold.group_id),
                    "llm": accounting,
                    "classifier_truth_ranks": truth_ranks,
                    "software_ids": sorted(state.context.software_ids),
                    **_separation_fields(state, fold.group_id, decision),
                    "soft_techniques": len(getattr(state.context, "soft_techniques", {}) or {}),
                    "technique_recall_soft": (
                        len(
                            (observed | set(getattr(state.context, "soft_techniques", {}) or {}))
                            & fold.held_out_techniques
                        )
                        / len(fold.held_out_techniques)
                        if fold.held_out_techniques
                        else 0.0
                    ),
                    "eligible_groups": (
                        len(state.context.eligible_group_ids)
                        if state.context.eligible_group_ids is not None
                        else None
                    ),
                    "leader_related_groups": sorted(
                        (getattr(decision, "related_leading_groups", {}) or {}).keys()
                    )
                    if decision
                    else [],
                    "leader_related_to_truth": bool(
                        ranked
                        and ranked[0] != fold.group_id
                        and ranked[0] in related.get(fold.group_id, {})
                    ),
                    # a group ATT&CK documents as overlapping with the truth is in the top three
                    "top3_related_to_truth": any(
                        gid != fold.group_id and gid in related.get(fold.group_id, {})
                        for gid in ranked[:3]
                    ),
                    # the same, counting only links that document the same activity (docs 10.11)
                    "leader_same_activity_as_truth": bool(
                        ranked
                        and ranked[0] != fold.group_id
                        and link_kind(engine.kb, fold.group_id, ranked[0]) in SAME_ACTIVITY_LINKS
                    ),
                    "top3_same_activity_as_truth": any(
                        gid != fold.group_id
                        and link_kind(engine.kb, fold.group_id, gid) in SAME_ACTIVITY_LINKS
                        for gid in ranked[:3]
                    ),
                    "candidates": len(ranked),
                    "top3": ranked[:3],
                    "stage": decision.stage.value if decision else None,
                    "margin": decision.margin if decision else None,
                    "maturity": decision.maturity_score if decision else None,
                    "leading": decision.leading_group_id if decision else None,
                    **_cluster_fields(state, decision),
                    **_top3_fields(state, ranked),
                }
            )
            if software_interview:
                group = engine.kb.groups.get(fold.group_id)
                rows[-1]["software_interview"] = enumerate_software_interview(
                    engine, state, set(getattr(group, "software_ids", ()) or ())
                )
            for key, value in proxy.dropped.items():
                dropped_total[key] += value
            rows[-1]["lexical_hidden"] = hidden_lexical or 0
            dropped_total["lexical"] += rows[-1]["lexical_hidden"]
    finally:
        engine.store = base_store
        engine.lexical_index = base_lexical
    return rows, dropped_total


def _forest_votes(engine: Any) -> bool:
    weights = engine.default_weights()
    return getattr(engine, "profile_forest", None) is not None and (
        weights.get(PROFILE_FOREST_NAME, 0.0) > 0 or weights.get(SOURCE_COUNTRY_NAME, 0.0) > 0
    )


def _refit_forest_without_actor(engine: Any, reduced_kb: Any) -> Any:
    """The shipped forest was fitted on the full knowledge base, so it knows the removed actor's
    profile; masking its output would still leak that knowledge into the other votes. Refit both
    forests on the reduced knowledge base with the same parameters and seed (GPU when present)."""
    model = engine.profile_forest
    # a group forest without a vote cannot move the ranking: keep it and refit the country forest
    silent = engine.default_weights().get(PROFILE_FOREST_NAME, 0.0) <= 0
    return fit_profile_forest(
        reduced_kb,
        params=dict(model.metadata.get("params") or {}),
        device=model.device,
        reuse_group_forest=model if silent else None,
    )


def run_unknown(
    settings: Settings,
    kb: AttackKnowledgeBase,
    store: ChromaVectorStore,
    folds: list[Fold],
    *,
    client: OpenRouterClient | None = None,
    software_interview: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        reduced = copy.deepcopy(kb)
        del reduced.groups[fold.group_id]
        for campaign in reduced.campaigns.values():
            if fold.group_id in getattr(campaign, "group_ids", []):
                campaign.group_ids = [g for g in campaign.group_ids if g != fold.group_id]
        for software in reduced.software.values():
            if fold.group_id in getattr(software, "group_ids", []):
                software.group_ids = [g for g in software.group_ids if g != fold.group_id]
        reduced.recompute_statistics()
        proxy = FoldStore(
            store,
            group_id=fold.group_id,
            held_out_techniques=fold.held_out_techniques,
            held_out_texts=fold.held_out_texts,
            hide_group_entirely=True,
        )
        engine = AttributionEngine(settings, reduced, proxy, client)  # type: ignore[arg-type]
        # the reduced knowledge base's index lacks the actor's own sentences; identical
        # sentences of other records are hidden too, as the dense store does
        hidden_lexical = hide_lexical(engine, fold.held_out_texts)
        forest = "off"
        if _forest_votes(engine):
            engine.set_profile_forest(_refit_forest_without_actor(engine, reduced))
            forest = "refit without the actor"
        group_mlp = "off"
        if getattr(engine, "group_mlp", None) is not None:
            # exact refit on the reduced knowledge base (fresh draws), never a masked output
            engine.set_group_mlp(refit_without_groups(engine.group_mlp, reduced))
            group_mlp = "refit without the actor"
        cost_before = client.total_usage.cost_usd if client is not None else 0.0
        state = engine.analyze(_ledger(fold))
        fusion = state.fusion
        decision = state.attribution_decision
        rows.append(
            {
                "group_id": fold.group_id,
                "observations": len(fold.observations),
                "profile_forest": forest,
                "group_mlp": group_mlp,
                "lexical_hidden": hidden_lexical or 0,
                "llm": _llm_accounting(state, client, cost_before),
                **_separation_fields(state, None, decision),
                **_combiner_fields(state, None),
                "stage": decision.stage.value if decision else None,
                "leading": decision.leading_group_id if decision else None,
                "margin": decision.margin if decision else None,
                "maturity": decision.maturity_score if decision else None,
                "top3": [entry.group_id for entry in fusion.entries[:3]],
                "software_ids": sorted(state.context.software_ids),
                **_cluster_fields(state, decision),
                **_top3_fields(state, [entry.group_id for entry in fusion.entries]),
            }
        )
        if software_interview:
            # the removed actor's own ATT&CK list answers; the engine sees the reduced base
            rows[-1]["software_interview"] = enumerate_software_interview(
                engine, state, set(kb.groups[fold.group_id].software_ids)
            )
    return rows


def _case_id(prefix: str, index: int, row: dict[str, Any]) -> str:
    # Real-report mode yields several folds per group, so ids must not collapse by group.
    return f"{prefix}:{index}:{row['group_id']}:{row.get('report_id', '')}"


def summarize(
    known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    named = AttributionStage.KNOWN_GROUP_HYPOTHESIS.value
    cluster = AttributionStage.KNOWN_CLUSTER_HYPOTHESIS.value
    named_stages = {named, cluster}
    cases = [
        AttributionEvaluationCase(
            case_id=_case_id("known", index, row),
            document_id=row.get("report_id") or row["group_id"],
            expected_group_id=row["group_id"],
        )
        for index, row in enumerate(known_rows)
    ] + [
        AttributionEvaluationCase(
            case_id=_case_id("unknown", index, row),
            document_id=row.get("report_id") or row["group_id"],
            expected_group_id=None,
        )
        for index, row in enumerate(unknown_rows)
    ]
    predictions = [
        AttributionPrediction(
            _case_id("known", index, row),
            tuple(row["top3"]),
            row["stage"] in named_stages,
            tuple(row.get("named_cluster") or ()),
        )
        for index, row in enumerate(known_rows)
    ] + [
        AttributionPrediction(
            _case_id("unknown", index, row),
            tuple(row["top3"]),
            row["stage"] in named_stages,
            tuple(row.get("named_cluster") or ()),
        )
        for index, row in enumerate(unknown_rows)
    ]
    # ranks beyond the stored top-3 are still needed for top-5 / MRR: recompute from rows.
    ranks = [row["rank"] for row in known_rows]
    metrics = evaluate_attribution_predictions(cases, predictions)
    metrics["top1"] = fmean(1.0 if rank == 1 else 0.0 for rank in ranks) if ranks else 0.0
    metrics["cluster_top1"] = (
        fmean(
            1.0 if row["rank"] == 1 or row.get("leader_related_to_truth") else 0.0
            for row in known_rows
        )
        if known_rows
        else 0.0
    )
    if known_rows and all("leader_same_activity_as_truth" in row for row in known_rows):
        # cluster-aware with same-activity links only (cluster_top1 counts any ATT&CK relation,
        # shared rare malware included)
        metrics["same_activity_top1"] = fmean(
            1.0 if row["rank"] == 1 or row.get("leader_same_activity_as_truth") else 0.0
            for row in known_rows
        )
    named_rows = [row for row in known_rows if row["stage"] == named]
    metrics["named_risk_cluster_aware"] = (
        fmean(
            0.0 if row["rank"] == 1 or row.get("leader_related_to_truth") else 1.0
            for row in named_rows
        )
        if named_rows
        else 0.0
    )
    metrics["top5"] = fmean(1.0 if rank and rank <= 5 else 0.0 for rank in ranks) if ranks else 0.0
    metrics["mrr"] = fmean(1.0 / rank if rank else 0.0 for rank in ranks) if ranks else 0.0
    _, low, high = bootstrap_interval(
        ranks, lambda values: fmean(1.0 if r == 1 else 0.0 for r in values)
    )
    metrics["top1_ci95"] = [round(low, 4), round(high, 4)]
    metrics["median_rank"] = (
        sorted(r if r else 10**6 for r in ranks)[len(ranks) // 2] if ranks else None
    )
    metrics["mean_technique_recall"] = (
        fmean(row["technique_recall"] for row in known_rows) if known_rows else 0.0
    )
    metrics["named_correct"] = sum(1 for row in named_rows if row["rank"] == 1)
    if known_rows and all(row.get("separation") is not None for row in known_rows):
        metrics["separation_auc"] = _auc(
            [float(row["separation"]) for row in known_rows if row["rank"] == 1],
            [float(row["separation"]) for row in known_rows if row["rank"] != 1],
        )
        metrics["margin_auc"] = _auc(
            [float(row["margin"] or 0.0) for row in known_rows if row["rank"] == 1],
            [float(row["margin"] or 0.0) for row in known_rows if row["rank"] != 1],
        )
    with_software = [row for row in known_rows if row.get("software_ids")]
    without_software = [row for row in known_rows if not row.get("software_ids")]
    metrics["top1_with_software"] = (
        fmean(1.0 if row["rank"] == 1 else 0.0 for row in with_software) if with_software else None
    )
    metrics["top1_without_software"] = (
        fmean(1.0 if row["rank"] == 1 else 0.0 for row in without_software)
        if without_software
        else None
    )
    metrics["software_present_count"] = len(with_software)
    probabilities = [
        row for row in known_rows if row.get("combiner_leader_probability") is not None
    ]
    if probabilities:
        metrics["leader_probability_auc"] = _auc(
            [
                float(row["combiner_leader_probability"])
                for row in probabilities
                if row["rank"] == 1
            ],
            [
                float(row["combiner_leader_probability"])
                for row in probabilities
                if row["rank"] != 1
            ],
        )
        metrics["rrf_top1"] = fmean(
            1.0 if row.get("rrf_rank") == 1 else 0.0 for row in probabilities
        )
    if known_rows and all("deterministic_rank" in row for row in known_rows):
        det = [row["deterministic_rank"] for row in known_rows]
        metrics["deterministic_top1"] = fmean(1.0 if r == 1 else 0.0 for r in det)
        metrics["deterministic_top5"] = fmean(1.0 if r and r <= 5 else 0.0 for r in det)
        metrics["deterministic_mrr"] = fmean(1.0 / r if r else 0.0 for r in det)
    clusters = [row for row in known_rows if row["stage"] == cluster]
    unknown_clusters = sum(1 for row in unknown_rows if row["stage"] == cluster)
    if clusters or unknown_clusters or any("named_cluster" in row for row in known_rows):
        # a cluster name is correct when the truth is a member (named_risk above counts it so);
        # named_correct keeps its single-group meaning
        correct = [row for row in clusters if row["group_id"] in (row.get("named_cluster") or ())]
        metrics["named_cluster_correct"] = len(correct)
        metrics["named_cluster_wrong"] = len(clusters) - len(correct) + unknown_clusters
        metrics["named_cluster_recovered"] = sum(1 for row in correct if row["rank"] != 1)
        metrics["cluster_coverage"] = (
            (len(clusters) + unknown_clusters) / (len(known_rows) + len(unknown_rows))
            if known_rows or unknown_rows
            else 0.0
        )
        singles = [row for row in known_rows if row["stage"] == named]
        single_unknown = sum(1 for row in unknown_rows if row["stage"] == named)
        single_total = len(singles) + single_unknown
        metrics["named_risk_single"] = (
            (sum(1 for row in singles if row["rank"] != 1) + single_unknown) / single_total
            if single_total
            else 0.0
        )
    if known_rows:
        metrics["top3"] = fmean(
            1.0 if row["rank"] and row["rank"] <= 3 else 0.0 for row in known_rows
        )
        if all("top3_related_to_truth" in row for row in known_rows):
            # strict top-3 or a documented-overlap group in it (always reported beside top3)
            metrics["cluster_top3"] = fmean(
                1.0
                if (row["rank"] and row["rank"] <= 3) or row.get("top3_related_to_truth")
                else 0.0
                for row in known_rows
            )
        if all("top3_same_activity_as_truth" in row for row in known_rows):
            metrics["same_activity_top3"] = fmean(
                1.0
                if (row["rank"] and row["rank"] <= 3) or row.get("top3_same_activity_as_truth")
                else 0.0
                for row in known_rows
            )
        in_top3 = [row for row in known_rows if row["rank"] and row["rank"] <= 3]
        in_top2 = [row for row in known_rows if row["rank"] in (1, 2)]
        metrics["top1_given_top3"] = (
            fmean(1.0 if row["rank"] == 1 else 0.0 for row in in_top3) if in_top3 else None
        )
        metrics["pairwise_12"] = (
            fmean(1.0 if row["rank"] == 1 else 0.0 for row in in_top2) if in_top2 else None
        )
    if any("leader_guard_fired" in row for row in known_rows + unknown_rows):
        metrics["leader_guard_fired"] = sum(
            1 for row in known_rows + unknown_rows if row.get("leader_guard_fired")
        )
    all_rows = known_rows + unknown_rows
    if all_rows and any("llm" in row for row in all_rows):
        llm_rows = [row["llm"] for row in all_rows if "llm" in row]
        metrics["llm"] = {
            "verifier_calls": sum(item["verifier_calls"] for item in llm_rows),
            "verifier_errors": sum(item["verifier_errors"] for item in llm_rows),
            "cost_usd": round(sum(item["cost_usd"] for item in llm_rows), 4),
        }
    stages: dict[str, dict[str, int]] = {"known": {}, "unknown": {}}
    for row in known_rows:
        stages["known"][row["stage"]] = stages["known"].get(row["stage"], 0) + 1
    for row in unknown_rows:
        stages["unknown"][row["stage"]] = stages["unknown"].get(row["stage"], 0) + 1
    metrics["stage_distribution"] = stages
    return metrics


def _inactive_combiner_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Rows scored without the learned combiner although it is enabled: the engine falls back
    to the RRF order silently when the artifact fails closed (e.g. a fingerprint mismatch).
    Rows without a fusion (no candidates) carry no status and are not counted."""
    return [
        f"{row['group_id']}:{row.get('report_id') or ''}".rstrip(":")
        for row in rows
        if row.get("combiner_status") not in (None, "active")
    ]


def _auc(positives: list[float], negatives: list[float]) -> float | None:
    """Rank-based AUC of a score for separating two groups (0.5 = uninformative)."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def default_output(pipeline: str, *, real_reports: bool, anonymize_actors: bool) -> Path:
    """Artifact path when ``--output`` is not given, named as the Makefile targets name it.

    Every mode has its own file, so a run never replaces another benchmark's artifact: a
    real-report run used to overwrite ``attribution_heldout_*`` (the ATT&CK folds, which the
    verification collector reads as the attribution gate), and a run on the statements as
    reported must not replace the anonymized CTI-HAL headline.
    """
    production = pipeline == "production"
    if not real_reports:
        name = "attribution_heldout_" + ("production" if production else "current")
    elif anonymize_actors:
        name = "attribution_cti_hal_" + ("production" if production else "current")
    else:
        name = "attribution_cti_hal_as_reported" + ("_production" if production else "")
    return REPO_ROOT / "data" / "audits" / f"{name}.json"


def pipeline_overrides(pipeline: str, set_args: list[str]) -> dict[str, Any]:
    """Settings overrides for a pipeline; explicit ``--set KEY=VALUE`` items win.

    ``local`` never touches OpenRouter (``local_only`` preset, no verifier); ``production``
    keeps the configured preset and verifier. CyberRE is analyst-triggered in the
    product and is excluded from both.
    """
    if pipeline == "local":
        # ``AttributionEngine`` applies the mapping preset on top of the settings, and the
        # ``balanced`` preset forces the verifier to ``ambiguous``; ``local_only`` is the
        # same preset with the verifier off, so this is the only override that sticks.
        overrides: dict[str, Any] = {
            "cyberre_weight": 0.0,
            "technique_mapping_preset": "local_only",
        }
    elif pipeline == "production":
        overrides = {"cyberre_weight": 0.0}
    else:
        raise ValueError(f"unknown pipeline {pipeline!r}")
    for item in set_args:
        key, _, value = item.partition("=")
        if not key or not value:
            raise ValueError(f"invalid --set {item!r}; expected KEY=VALUE")
        overrides[key] = _coerce(value)
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-behaviors", type=int, default=5)
    parser.add_argument("--min-techniques", type=int, default=5)
    parser.add_argument("--max-observations", type=int, default=12)
    parser.add_argument("--unknown", type=int, default=25, help="unknown-actor folds to run")
    parser.add_argument("--limit", type=int, help="known folds to run (default all eligible)")
    parser.add_argument(
        "--sample",
        type=int,
        help="real-report mode: run a seeded random sample of this many reports "
        "(spread across groups) instead of the first --limit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="artifact path (default: data/audits/attribution_heldout_{current,production}.json; "
        "with --cti-hal, attribution_cti_hal_{current,production}.json when anonymized, else "
        "attribution_cti_hal_as_reported[_production].json)",
    )
    parser.add_argument(
        "--pipeline",
        choices=("local", "production"),
        default="local",
        help="local: verifier off, no OpenRouter calls (default); "
        "production: configured technique verifier, costs recorded",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--cti-hal",
        type=Path,
        help="reports.jsonl from scripts/prepare_cti_hal.py: run real-report attribution "
        "instead of the held-out ATT&CK folds (no unknown-actor arm, no evidence hiding)",
    )
    parser.add_argument(
        "--anonymize-actors",
        action="store_true",
        help="real-report mode: replace every ATT&CK group name and alias in the statements with "
        '"the adversary" before analysis, as the held-out folds do (default: statements as reported)',
    )
    parser.add_argument(
        "--allow-inactive-combiner",
        action="store_true",
        help="write the artifact even when the learned combiner is enabled but did not load "
        "(the engine then falls back to the RRF order); default: refuse",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a Settings field for this run (e.g. --set technique_family_weight=2.0)",
    )
    parser.add_argument(
        "--simulate-software-interview",
        choices=("enumerate",),
        help="after each row, walk every found / not-found path of the software questions "
        "through the real pipeline and report expected values for q in {0.1, 0.3, 1.0} and a "
        "wrong-found rate in {0, 0.05} plus the criteria C1-C8 (docs 21.3.15); turns "
        "INTERVIEW_SOFTWARE_QUESTIONS on unless --set says otherwise. Default: off",
    )
    args = parser.parse_args(argv)
    if args.anonymize_actors and args.cti_hal is None:
        parser.error("--anonymize-actors needs --cti-hal; the ATT&CK folds are always anonymized")
    if args.output is None:
        args.output = default_output(
            args.pipeline,
            real_reports=args.cti_hal is not None,
            anonymize_actors=args.anonymize_actors,
        )

    try:
        overrides = pipeline_overrides(args.pipeline, args.set)
    except ValueError as error:
        parser.error(str(error))
    simulate = args.simulate_software_interview is not None
    if simulate:
        overrides.setdefault("interview_software_questions", True)
    settings = Settings(**overrides)
    production = args.pipeline == "production"
    client = OpenRouterClient(settings) if production else None
    kb = AttackKnowledgeBase.load(settings.kb_path)
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    store = ChromaVectorStore(settings, embedder)
    real_reports = args.cti_hal is not None
    actor_records: list[dict[str, Any]] = []
    actor_totals: dict[str, Any] | None = None
    if real_reports:
        folds = load_report_folds(args.cti_hal, max_observations=max(args.max_observations, 30))
        if args.sample is not None:
            folds = sample_folds(folds, args.sample, args.seed)
        elif args.limit is not None:
            folds = folds[: args.limit]
        if args.anonymize_actors:
            folds, actor_records, actor_totals = anonymize_report_folds(folds, kb)
    else:
        folds = build_folds(
            kb,
            coverage=args.coverage,
            seed=args.seed,
            min_behaviors=args.min_behaviors,
            min_techniques=args.min_techniques,
            max_observations=args.max_observations,
            limit=args.limit,
        )
    if not folds:
        parser.error("no eligible groups")
    engine = AttributionEngine(settings, kb, store, client)  # type: ignore[arg-type]
    combiner_expected = bool(getattr(settings, "attribution_combiner_enabled", False))
    if (
        combiner_expected
        and engine.combiner_status != "active"
        and not args.allow_inactive_combiner
    ):
        print(
            f"refusing to run: the learned combiner is enabled but {engine.combiner_status!r}; "
            "a fingerprinted source or weight changed (refit with make fit-combiner), or pass "
            "--allow-inactive-combiner to measure the RRF fallback deliberately",
            file=sys.stderr,
        )
        return 3
    started = time.time()
    try:
        known_rows, dropped = run_known(
            engine,
            folds,
            hide_evidence=not real_reports,
            client=client,
            software_interview=simulate,
        )
    except LLMUnavailableError as error:
        print(f"aborted: {error}", file=sys.stderr)
        if client is not None:
            client.close()
        return 2
    known_seconds = time.time() - started
    # rows follow the folds in order (a truncated production run keeps a prefix)
    for row, record in zip(known_rows, actor_records, strict=False):
        row["actor_name_replacement"] = record
    unknown_folds = (
        []
        if real_reports
        else random.Random(args.seed).sample(folds, min(args.unknown, len(folds)))
    )
    started = time.time()
    unknown_rows = (
        run_unknown(settings, kb, store, unknown_folds, client=client, software_interview=simulate)
        if unknown_folds
        else []
    )
    if client is not None:
        client.close()
    unknown_seconds = time.time() - started
    metrics = summarize(known_rows, unknown_rows)
    fingerprint_inputs = {
        "attack_version": kb.version,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_model_revision,
        "weights": engine.default_weights(),
        "family_weights": engine.family_weights(),
        "rrf_k": settings.rrf_k,
        "thresholds": engine.attribution_thresholds.__dict__,
        "acceptance_model": engine.acceptance_model_status,
        "combiner": engine.combiner_status,
        "combiner_artifact": _combiner_artifact(settings, engine),
        "profile_forest": {
            "status": engine.profile_forest_status,
            "fingerprint": getattr(engine.profile_forest, "fingerprint", None),
            "votes": _forest_votes(engine),
        },
        # the channel weights the gate window was scored with (offline replays read them)
        "discrimination_weights": dict(vars(engine.discrimination_weights)),
        # only when the network votes, so a default run keeps its fingerprint inputs
        **(
            {
                "group_mlp": {
                    "status": engine.group_mlp_status,
                    "fingerprint": engine.group_mlp.fingerprint,
                }
            }
            if getattr(engine, "group_mlp", None) is not None
            else {}
        ),
        "source_fingerprint": source_fingerprint(FINGERPRINTED_SOURCES),
        "platform_from_text": bool(getattr(settings, "technique_platform_from_text", False)),
        "pipeline": args.pipeline,
        "verifier_mode": str(getattr(settings, "technique_verifier_mode", "ambiguous")),
        "verifier_model": (
            str(getattr(settings, "technique_verifier_model", settings.extract_model))
            if production
            else None
        ),
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "attribution_cti_hal_reports" if real_reports else "attribution_text_heldout",
        "source": str(args.cti_hal)
        if real_reports
        else "ATT&CK procedure sentences (held out per group)",
        "attack_version": kb.version,
        "parameters": {
            "pipeline": args.pipeline,
            "settings_overrides": {k: v for k, v in overrides.items()},
            "coverage": args.coverage,
            "seed": args.seed,
            "min_behaviors": args.min_behaviors,
            "min_techniques": args.min_techniques,
            "max_observations": args.max_observations,
            "known_folds": len(known_rows),
            "unknown_folds": len(unknown_rows),
            "sample": args.sample,
            "limit": args.limit,
            "truncated_at": dropped.get("llm_unavailable_at"),
            # fold evidence hidden from the dense store and the BM25 index; real reports hide nothing
            "leak_control": (
                "none (real reports)"
                if real_reports
                else LEAK_CONTROL
                + (
                    " (synthetic paraphrases not hidden)"
                    if getattr(settings, "technique_include_synthetic", False)
                    else ""
                )
            ),
            # the ATT&CK folds are always anonymized; real reports only with --anonymize-actors
            "actor_names": (
                "as_reported" if real_reports and not args.anonymize_actors else "anonymized"
            ),
        },
        "pipeline_fingerprint": pipeline_fingerprint(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "leak_control": {
            "dropped_hits": dropped,
            "dropped_hits_unknown": {
                "lexical": sum(int(row.get("lexical_hidden") or 0) for row in unknown_rows)
            },
            **({"actor_name_replacement": actor_totals} if actor_totals is not None else {}),
        },
        "runtime_seconds": {"known": round(known_seconds, 1), "unknown": round(unknown_seconds, 1)},
        "metrics": metrics,
        "known": known_rows,
        "unknown": unknown_rows,
    }
    if simulate:
        report["software_interview"] = summarize_software_interview(
            known_rows,
            unknown_rows,
            cluster_naming=bool(getattr(settings, "attribution_cluster_naming", False)),
        )
    inactive = _inactive_combiner_rows(known_rows + unknown_rows)
    if combiner_expected and inactive and not args.allow_inactive_combiner:
        print(
            f"refusing to write {args.output}: the learned combiner is enabled but was inactive "
            f"on {len(inactive)} rows (first {inactive[0]}); pass --allow-inactive-combiner to "
            "record the RRF fallback deliberately",
            file=sys.stderr,
        )
        return 3
    destination = write_artifact(args.output, report)
    if not args.quiet:
        print(
            f"known folds {len(known_rows)} ({known_seconds:.0f}s), unknown folds {len(unknown_rows)} ({unknown_seconds:.0f}s)"
        )
        for key in (
            "top1",
            "cluster_top1",
            "top5",
            "mrr",
            "median_rank",
            "mean_technique_recall",
            "unknown_recall",
            "false_named_attribution_rate",
            "named_coverage",
            "named_risk",
            "named_correct",
            "separation_auc",
            "margin_auc",
        ):
            if key in metrics:
                print(f"  {key:30s} {metrics[key]}")
        print("  top1 ci95", metrics["top1_ci95"])
        print("  stages:", metrics["stage_distribution"])
        if "deterministic_top1" in metrics:
            print(
                f"  deterministic (before LLM vote): top1 {metrics['deterministic_top1']:.3f} "
                f"top5 {metrics['deterministic_top5']:.3f} mrr {metrics['deterministic_mrr']:.3f}"
            )
        if "llm" in metrics:
            print("  llm:", metrics["llm"])
        print("  leak-control dropped hits:", dropped)
        if simulate:
            simulation = report["software_interview"]
            print(
                f"  software interview: {simulation['rows_asked']} rows asked, "
                f"{simulation['questions']} questions, {simulation['leaves']} leaves"
            )
            for key in ("q=0.1,fp=0.0", "q=0.3,fp=0.0", "q=1.0,fp=0.0", "q=0.3,fp=0.05"):
                item = simulation["expected"][key]
                print(
                    f"    {key:14s} gained {item['gained']:.2f} wrong added "
                    f"{item['wrong_added']:.2f} lost {item['lost']:.2f} removed named "
                    f"{item['removed_named']:.2f} hit rate {item['hit_rate']}"
                )
            print(
                "    criteria: "
                + " ".join(
                    f"{name}={value.get('pass')}" for name, value in simulation["criteria"].items()
                )
            )
        if actor_totals is not None:
            print(
                f"  actor names anonymized: {actor_totals['reports_changed']}/"
                f"{actor_totals['reports']} reports ({actor_totals['reports_naming_own_group']} "
                f"name their own group), {actor_totals['statements_changed']} statements, "
                f"{actor_totals['replacements']} names; software matches removed "
                f"{actor_totals['software_matches_removed']} {actor_totals['software_ids_removed']}"
            )
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
