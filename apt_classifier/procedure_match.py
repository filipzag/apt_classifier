"""Technique-conditioned procedure matching with ATT&CK's software links (docs 9.13, 21.3.20;
prereg ``attribution_procedure_match_2026-10-21``).

Groups differ in *how* they do a technique, not in which one. ATT&CK documents that "how" as
procedure sentences tied to a technique: the group's own (``Group.technique_behaviors``), the
procedure examples of the malware and tools it uses (the ``uses`` relationships of the raw bundle),
and the procedures of its campaigns. The text member's corpus (:func:`text_group.build_corpus`)
already holds all three, anonymised and labelled with the technique id and the groups credited.

This member reads the case **per accepted technique**: for each technique the pipeline accepted
and the statements that support it, it compares those statements only with the corpus sentences
documented under that technique (``family=parent``: also its parent and its sub-techniques), and
votes for a group through three routes:

1. *group route*: the group's (or its campaign's) own procedure sentence under the technique,
   vote = cosine;
2. *software route*: a procedure sentence of a malware or tool under the technique, vote = cosine
   x the software's rarity (the software member's rule, normalised to [0, 1], tools capped) x
   ``1 + seen_bonus`` when the report names that software, for every group ATT&CK links the
   software to;
3. *link route* (no text): the report names a software and ATT&CK documents that software for
   the accepted technique, vote = ``link_weight`` x rarity for every group that uses it.

A group's vote for a technique is the best of the routes; its case score is the mean of its votes
over the accepted techniques (``aggregate=mean``, absent = 0) or over its five best techniques
(``top5``): the share of the report's techniques the group has done *this way*. The score is
z-scored across the groups, plus an optional prominence prior, and reported as a softmax over the
groups reached (the convention of the text and behaviour members). The member joins the ensemble
only with ``PROCEDURE_MATCH_WEIGHT`` > 0, never counts as an independent evidence family (it
re-reads evidence the technique, software and behaviour families already score), and the
benchmarks hide a fold's held-out sentences and the removed actor from it through
:meth:`ProcedureMatchModel.without` (the text member's refit, re-indexed).
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.behavior_text import sentence_rows, transform_queries
from apt_classifier.classifiers.software import COMMODITY_TOOL_WEIGHT_CAP, software_rarity_weights
from apt_classifier.text_group import TextGroupModel

PROCEDURE_MATCH_NAME = "procedure_match"
AGGREGATES = ("mean", "top5")
FAMILIES = ("exact", "parent")
DEFAULT_PARAMS: dict[str, Any] = {
    # a corpus sentence votes only above this cosine
    "min_similarity": 0.15,
    # "exact": sentences documented under the accepted technique only; "parent": also under its
    # parent and its sub-techniques (never its siblings, which are what separates groups)
    "family": "parent",
    # software procedure sentences under the technique vote for the software's user groups
    "software_route": True,
    # multiplier (1 + seen_bonus) on the software route when the report names that software
    "seen_bonus": 1.0,
    # the text-free link: named software documented for the accepted technique -> its users
    "link_weight": 1.0,
    # "mean": votes averaged over every accepted technique; "top5": over the five best
    "aggregate": "top5",
    # weight of the log-prominence prior on the z-scored case score (0: none)
    "prior": 0.0,
}
MAX_DETAIL = 90


class ProcedureMatchModel:
    """The technique-indexed sentence index over a :class:`TextGroupModel`'s records."""

    def __init__(
        self, base: TextGroupModel, kb: Any, params: Mapping[str, Any] | None = None
    ) -> None:
        self.base = base
        self.kb = kb
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        if self.params["aggregate"] not in AGGREGATES:
            raise ValueError(f"unknown aggregate {self.params['aggregate']!r}")
        if self.params["family"] not in FAMILIES:
            raise ValueError(f"unknown family rule {self.params['family']!r}")
        self.group_ids = base.group_ids
        self.log_prior = base.log_prior
        self._group_index = {g: i for i, g in enumerate(self.group_ids)}
        self._idf, self._rows_t = sentence_rows(base)
        techniques = getattr(kb, "techniques", {}) or {}
        self._parent = {
            tid: str(getattr(t, "parent_id", "") or "") for tid, t in techniques.items()
        }
        children: dict[str, list[str]] = {}
        for tid, parent in self._parent.items():
            if parent:
                children.setdefault(parent, []).append(tid)
        self._children = {p: tuple(sorted(c)) for p, c in children.items()}
        by_technique: dict[str, list[int]] = {}
        for i, record in enumerate(base.records):
            tid = record.get("technique_id")
            if tid and record["groups"]:
                by_technique.setdefault(str(tid), []).append(i)
        self._by_technique = {t: np.asarray(v, dtype=np.int64) for t, v in by_technique.items()}
        self._record_groups = [tuple(r["groups"]) for r in base.records]
        self._record_source = [str(r["source"]) for r in base.records]
        self._record_owner = [str(r["owner"]) for r in base.records]
        self._record_text = [str(r["text"]) for r in base.records]
        software = getattr(kb, "software", {}) or {}
        self._software_groups = {
            sid: tuple(sorted(getattr(s, "group_ids", ()) or ())) for sid, s in software.items()
        }
        self._software_techniques = {
            sid: frozenset(getattr(s, "technique_ids", ()) or ()) for sid, s in software.items()
        }
        self._rarity_cache: dict[str, float] = {}
        self._without_cache: dict[Any, ProcedureMatchModel] = {}

    # ----- identity -----
    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.base.fingerprint.encode())
        digest.update(json.dumps(dict(sorted(self.params.items())), sort_keys=True).encode())
        digest.update(str(getattr(self.kb, "version", "") or "").encode())
        return digest.hexdigest()[:24]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.base.metadata

    @property
    def runtime(self) -> dict[str, str]:
        return self.base.runtime

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        return self.base.compatible_with(kb)

    # ----- rules -----
    def family(self, technique_id: str) -> tuple[str, ...]:
        """The technique ids whose sentences may answer for ``technique_id``."""
        if self.params["family"] == "exact":
            return (technique_id,)
        ids = [technique_id]
        parent = self._parent.get(technique_id, "")
        if parent:
            ids.append(parent)
        ids.extend(self._children.get(technique_id, ()))
        return tuple(dict.fromkeys(ids))

    def rarity(self, software_id: str) -> float:
        """The software member's rarity weight, normalised to [0, 1] (tools capped)."""
        cached = self._rarity_cache.get(software_id)
        if cached is not None:
            return cached
        weights = software_rarity_weights(
            self.kb, {software_id}, tool_weight_cap=COMMODITY_TOOL_WEIGHT_CAP
        )
        groups = len(getattr(self.kb, "groups", {}) or {}) or 1
        top = math.log((groups + 1) / 2.0) + 1.0
        value = min(1.0, float(weights.get(software_id, 0.0)) / top) if top > 0 else 0.0
        self._rarity_cache[software_id] = value
        return value

    # ----- scoring -----
    def votes(
        self,
        technique_statements: Mapping[str, Sequence[str]],
        software_ids: Iterable[str] = (),
    ) -> tuple[np.ndarray, list[str], dict[str, tuple[str, float, str]]]:
        """``(votes, techniques, details)``: ``votes`` is (techniques, groups), each entry the
        best route vote of that group for that accepted technique; ``details`` names, per group,
        the technique, vote and matched sentence behind its best vote."""
        techniques = sorted(str(t) for t in technique_statements if str(t))
        seen = {str(s) for s in software_ids if str(s)}
        output = np.zeros((len(techniques), len(self.group_ids)), dtype=np.float64)
        details: dict[str, tuple[str, float, str]] = {}
        floor = float(self.params["min_similarity"])
        bonus = float(self.params["seen_bonus"])
        software_route = bool(self.params["software_route"])
        link = float(self.params["link_weight"])

        def credit(row: int, groups: Iterable[str], vote: float, tid: str, note: str) -> None:
            for g in groups:
                gi = self._group_index.get(g)
                if gi is None or vote <= output[row, gi]:
                    continue
                output[row, gi] = vote
                details[g] = (tid, vote, note)

        for row, tid in enumerate(techniques):
            texts = [str(s) for s in technique_statements[tid] if str(s).strip()]
            family = self.family(tid)
            members = [self._by_technique[f] for f in family if f in self._by_technique]
            if texts and members:
                records = np.unique(np.concatenate(members))
                query = transform_queries(self.base, self._idf, texts)
                similarity = (query @ self._rows_t[:, records]).toarray()
                best_per_record = similarity.max(axis=0)
                for column, record in enumerate(records):
                    best = float(best_per_record[column])
                    if best <= floor:
                        continue
                    source = self._record_source[record]
                    owner = self._record_owner[record]
                    if source == "software":
                        if not software_route:
                            continue
                        weight = self.rarity(owner) * (1.0 + bonus if owner in seen else 1.0)
                        vote = best * weight
                        if vote <= 0:
                            continue
                        note = (
                            f"{owner}{' (named)' if owner in seen else ''}: "
                            f"{self._record_text[record][:MAX_DETAIL]}"
                        )
                    else:
                        vote = best
                        note = f"{source}: {self._record_text[record][:MAX_DETAIL]}"
                    credit(row, self._record_groups[record], vote, tid, note)
            if link > 0 and seen:
                family_set = set(family)
                for sid in sorted(seen):
                    if not (self._software_techniques.get(sid, frozenset()) & family_set):
                        continue
                    vote = link * self.rarity(sid)
                    if vote <= 0:
                        continue
                    credit(
                        row,
                        self._software_groups.get(sid, ()),
                        vote,
                        tid,
                        f"{sid} (named) is documented for {tid}",
                    )
        return output, techniques, details

    def score_vector(
        self,
        technique_statements: Mapping[str, Sequence[str]],
        software_ids: Iterable[str] = (),
    ) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[str, float, str]]]:
        """``(support, score, details)`` over ``group_ids``: support = the aggregated vote (0 for
        a group no technique reached), score = its z-score across the groups plus the prior."""
        empty = np.zeros(len(self.group_ids))
        votes, _, details = self.votes(technique_statements, software_ids)
        if not votes.size:
            return empty, empty, details
        if self.params["aggregate"] == "top5":
            top = -np.sort(-votes, axis=0)[: min(5, votes.shape[0])]
            support = top.mean(axis=0)
        else:
            support = votes.mean(axis=0)
        if not np.any(support > 0):
            return empty, empty, details
        z = (support - support.mean()) / (support.std() + 1e-9)
        return support, z + float(self.params["prior"]) * self.log_prior, details

    def scores(
        self,
        technique_statements: Mapping[str, Sequence[str]],
        software_ids: Iterable[str] = (),
    ) -> tuple[dict[str, float], dict[str, tuple[str, float, str]]]:
        """Per group at least one technique reached: the softmax of the score over those groups
        (positive), and the details behind each group's best vote."""
        support, score, details = self.score_vector(technique_statements, software_ids)
        reached = support > 0
        if not reached.any():
            return {}, details
        kept = score[reached]
        weights = np.exp(kept - kept.max())
        weights /= weights.sum()
        groups = [g for g, s in zip(self.group_ids, reached, strict=True) if s]
        return {g: float(w) for g, w in zip(groups, weights, strict=True)}, details

    # ----- fold hiding -----
    def without(
        self,
        kb: Any,
        *,
        group_id: str | None = None,
        techniques: Iterable[str] = (),
        texts: Iterable[str] = (),
        remove_group: bool = False,
    ) -> ProcedureMatchModel:
        """The index over the text member's refit without a fold's evidence (same rules)."""
        held, hidden = frozenset(techniques), frozenset(texts)
        key = (group_id, held, hidden, remove_group, id(kb))
        cached = self._without_cache.get(key)
        if cached is not None:
            return cached
        model = ProcedureMatchModel(
            self.base.without(
                kb, group_id=group_id, techniques=held, texts=hidden, remove_group=remove_group
            ),
            kb,
            self.params,
        )
        if len(self._without_cache) >= 4:
            self._without_cache.pop(next(iter(self._without_cache)))
        self._without_cache[key] = model
        return model


# ---------------------------------------------------------------------------- pipeline member
def procedure_match_votes(settings: Any) -> bool:
    """The member joins the ensemble only with a positive weight (at 0 nothing is loaded and
    the fusion and the combiner fingerprint are unchanged)."""
    return bool(getattr(settings, "procedure_match_enabled", False)) and (
        float(getattr(settings, "procedure_match_weight", 0.0) or 0.0) > 0
    )


def procedure_match_params(settings: Any) -> dict[str, Any]:
    return {
        "min_similarity": float(
            getattr(settings, "procedure_match_min_similarity", DEFAULT_PARAMS["min_similarity"])
        ),
        "family": str(getattr(settings, "procedure_match_family", DEFAULT_PARAMS["family"])),
        "software_route": bool(
            getattr(settings, "procedure_match_software_route", DEFAULT_PARAMS["software_route"])
        ),
        "seen_bonus": float(
            getattr(settings, "procedure_match_seen_bonus", DEFAULT_PARAMS["seen_bonus"])
        ),
        "link_weight": float(
            getattr(settings, "procedure_match_link_weight", DEFAULT_PARAMS["link_weight"])
        ),
        "aggregate": str(
            getattr(settings, "procedure_match_aggregate", DEFAULT_PARAMS["aggregate"])
        ),
        "prior": float(getattr(settings, "procedure_match_prior", DEFAULT_PARAMS["prior"])),
    }


def load_procedure_match_for(
    settings: Any, kb: Any, base: TextGroupModel | None = None
) -> tuple[ProcedureMatchModel | None, str]:
    """The member for the engine, fail-closed: ``(model, status)``. ``base`` is an already
    loaded text member (the same artifact); otherwise ``TEXT_GROUP_PATH`` is read."""
    if not procedure_match_votes(settings):
        return None, "disabled (weight 0)"
    if base is None:
        path = getattr(settings, "text_group_path", "")
        try:
            base = TextGroupModel.load(path, kb)
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
            OSError,
            EOFError,
            zlib.error,
        ) as error:
            return None, f"invalid artifact: {error}"
        if base is None:
            return None, f"artifact not found: {path} (run make fit-text-group)"
        ok, reason = base.compatible_with(kb)
        if not ok:
            return None, f"incompatible artifact: {reason}"
    try:
        return ProcedureMatchModel(base, kb, procedure_match_params(settings)), "active"
    except ValueError as error:
        return None, f"invalid settings: {error}"


def procedure_match_bound_fingerprint(base: str, model: ProcedureMatchModel) -> str:
    """The combiner pipeline fingerprint when the member votes (as the text member binds it)."""
    digest = hashlib.sha256()
    digest.update(base.encode())
    digest.update(model.fingerprint.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:24]


class ProcedureMatchClassifier:
    """Ensemble member: the model's score for each eligible group at least one accepted
    technique reached. It cannot make a group eligible."""

    name = PROCEDURE_MATCH_NAME

    def __init__(self, model: ProcedureMatchModel | None = None) -> None:
        self.model = model

    def classify(self, context: Any) -> Any:
        from apt_classifier.classifiers.base import build_result, group_ids

        statements = {
            str(t): tuple(str(s) for s in texts)
            for t, texts in dict(getattr(context, "technique_statements", {}) or {}).items()
            if t and texts
        }
        if self.model is None or not statements:
            return build_result(self.name, context, {}, applicable=False)
        software_ids = {str(s) for s in (getattr(context, "software_ids", set()) or set())}
        allowed = set(group_ids(context))
        scores, details = self.model.scores(statements, software_ids)
        scores = {g: s for g, s in scores.items() if g in allowed}
        signals = {}
        for g, s in scores.items():
            parts = [f"procedure_match:p={s:.3f}"]
            if g in details:
                tid, vote, note = details[g]
                parts.append(f"{tid} {vote:.2f} {note}")
            signals[g] = tuple(parts)
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)
