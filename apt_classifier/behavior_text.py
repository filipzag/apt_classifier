"""Statement-level behaviour matching: each report statement against ATT&CK's per-group procedure
sentences (docs 9.15, 21.3.19; prereg ``attribution_behavior_text_2026-10-20``).

The free-text member (:mod:`apt_classifier.text_group`) scores the whole case narrative against
one TF-IDF document per group. This member keeps that corpus (the anonymised group procedure
examples, the groups' tools' procedure examples, the campaign procedures and the group
descriptions, :func:`apt_classifier.text_group.build_corpus`) but scores **sentence by sentence**:
every statement of the case is compared with every corpus sentence (TF-IDF 1-2 grams, IDF over
sentences), the ``top_k`` nearest sentences vote for their groups with their cosine (one vote per
group per statement, the best), and a group's case score is the mean of its votes over the
statements. A group therefore wins by matching many of the report's statements with procedure
text of its own, not by the size of its corpus; the per-case z-score and the optional prominence
prior follow the text member. The ensemble member (:class:`BehaviorTextClassifier`) is a learned
profile that never counts as an independent evidence family; the benchmarks hide a fold's
held-out sentences from it through :meth:`BehaviorTextModel.without`, exactly as for the text
member. It reads the text member's artifact (``TEXT_GROUP_PATH``) and joins the ensemble only
with ``BEHAVIOR_TEXT_WEIGHT`` > 0.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.text_group import TextGroupModel

BEHAVIOR_TEXT_NAME = "behavior_text"
AGGREGATES = ("mean", "top5")
DEFAULT_PARAMS: dict[str, Any] = {
    # nearest corpus sentences per statement that may vote (dev choice 2026-10-21: 10)
    "top_k": 10,
    # a statement's vote counts only above this cosine
    "min_similarity": 0.0,
    # "mean": the group's votes averaged over every statement (absent = 0); "top5": over its
    # five best statements only (dev choice 2026-10-21)
    "aggregate": "top5",
    # weight of the log-prominence prior on the z-scored case score (0: none)
    "prior": 0.0,
}


class BehaviorTextModel:
    """The sentence index over a :class:`TextGroupModel`'s records."""

    def __init__(self, base: TextGroupModel, params: Mapping[str, Any] | None = None) -> None:
        from scipy import sparse

        self.base = base
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        if self.params["aggregate"] not in AGGREGATES:
            raise ValueError(f"unknown aggregate {self.params['aggregate']!r}")
        self.group_ids = base.group_ids
        self.log_prior = base.log_prior
        counts = base._counts.tocsr()
        n_rows, vocabulary = counts.shape
        df = np.bincount(counts.indices, minlength=vocabulary)
        self._idf = (np.log((1.0 + n_rows) / (1.0 + df)) + 1.0).astype(np.float32)
        rows = counts.astype(np.float32).copy()
        rows.data = np.log1p(rows.data)
        rows = base._normalize(rows @ sparse.diags(self._idf), norm="l2")
        self._rows_t = rows.T.tocsc()
        index = {g: i for i, g in enumerate(self.group_ids)}
        weight = float(base.params.get("software_weight", 1.0))
        r_idx, g_idx, vals = [], [], []
        for i, record in enumerate(base.records):
            w = weight if record["source"] == "software" else 1.0
            if w <= 0:
                continue
            for g in record["groups"]:
                if g in index:
                    r_idx.append(i)
                    g_idx.append(index[g])
                    vals.append(w)
        self._membership = sparse.csr_matrix(
            (np.asarray(vals, np.float32), (r_idx, g_idx)), shape=(n_rows, len(self.group_ids))
        )
        self._without_cache: dict[Any, BehaviorTextModel] = {}

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.base.fingerprint.encode())
        digest.update(json.dumps(dict(sorted(self.params.items())), sort_keys=True).encode())
        return digest.hexdigest()[:24]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.base.metadata

    @property
    def runtime(self) -> dict[str, str]:
        return self.base.runtime

    def votes(self, statements: Sequence[str]) -> np.ndarray:
        """(statements, groups): per statement each group's best cosine among the statement's
        ``top_k`` nearest corpus sentences (0 when none of them is the group's)."""
        from scipy import sparse

        groups = len(self.group_ids)
        texts = [str(s) for s in statements if str(s).strip()]
        if not texts:
            return np.zeros((0, groups))
        query = self.base._vectorizer.transform(texts).tocsr().astype(np.float32)
        query.data = np.log1p(query.data)
        query = self.base._normalize(query @ sparse.diags(self._idf), norm="l2")
        similarity = (query @ self._rows_t).tocsr()
        top_k = max(int(self.params["top_k"]), 1)
        floor = float(self.params["min_similarity"])
        output = np.zeros((len(texts), groups), dtype=np.float64)
        for s in range(len(texts)):
            start, end = similarity.indptr[s], similarity.indptr[s + 1]
            if start == end:
                continue
            columns = similarity.indices[start:end]
            values = similarity.data[start:end]
            if len(values) > top_k:
                keep = np.argpartition(-values, top_k - 1)[:top_k]
                columns, values = columns[keep], values[keep]
            for record, value in zip(columns, values, strict=True):
                if value <= floor:
                    continue
                member = self._membership[record]
                for g, w in zip(member.indices, member.data, strict=True):
                    vote = float(value) * float(w)
                    if vote > output[s, g]:
                        output[s, g] = vote
        return output

    def score_vector(self, statements: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(support, score)`` over ``group_ids``: support = the aggregated vote (0 for a group
        no statement reached), score = its z-score across the groups plus the prior."""
        empty = np.zeros(len(self.group_ids))
        votes = self.votes(statements)
        if not votes.size:
            return empty, empty
        if self.params["aggregate"] == "top5":
            top = -np.sort(-votes, axis=0)[: min(5, votes.shape[0])]
            support = top.mean(axis=0)
        else:
            support = votes.mean(axis=0)
        if not np.any(support > 0):
            return empty, empty
        z = (support - support.mean()) / (support.std() + 1e-9)
        return support, z + float(self.params["prior"]) * self.log_prior

    def scores(self, statements: Sequence[str]) -> dict[str, float]:
        """Per group that at least one statement reached: the softmax of the score over those
        groups (positive; the convention of the text member)."""
        support, score = self.score_vector(statements)
        reached = support > 0
        if not reached.any():
            return {}
        kept = score[reached]
        weights = np.exp(kept - kept.max())
        weights /= weights.sum()
        groups = [g for g, s in zip(self.group_ids, reached, strict=True) if s]
        return {g: float(w) for g, w in zip(groups, weights, strict=True)}

    def without(
        self,
        kb: Any,
        *,
        group_id: str | None = None,
        techniques: Iterable[str] = (),
        texts: Iterable[str] = (),
        remove_group: bool = False,
    ) -> BehaviorTextModel:
        """The index over the text member's refit without a fold's evidence (same rules)."""
        held, hidden = frozenset(techniques), frozenset(texts)
        key = (group_id, held, hidden, remove_group, id(kb))
        cached = self._without_cache.get(key)
        if cached is not None:
            return cached
        model = BehaviorTextModel(
            self.base.without(
                kb, group_id=group_id, techniques=held, texts=hidden, remove_group=remove_group
            ),
            self.params,
        )
        if len(self._without_cache) >= 4:
            self._without_cache.pop(next(iter(self._without_cache)))
        self._without_cache[key] = model
        return model

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        return self.base.compatible_with(kb)


# ---------------------------------------------------------------------------- pipeline member
def behavior_text_votes(settings: Any) -> bool:
    """The member joins the ensemble only with a positive weight (at 0 nothing is loaded and
    the fusion and the combiner fingerprint are unchanged)."""
    return bool(getattr(settings, "behavior_text_enabled", False)) and (
        float(getattr(settings, "behavior_text_weight", 0.0) or 0.0) > 0
    )


def behavior_text_params(settings: Any) -> dict[str, Any]:
    return {
        "top_k": int(getattr(settings, "behavior_text_top_k", DEFAULT_PARAMS["top_k"])),
        "min_similarity": float(
            getattr(settings, "behavior_text_min_similarity", DEFAULT_PARAMS["min_similarity"])
        ),
        "aggregate": str(getattr(settings, "behavior_text_aggregate", DEFAULT_PARAMS["aggregate"])),
        "prior": float(getattr(settings, "behavior_text_prior", DEFAULT_PARAMS["prior"])),
    }


def load_behavior_text_for(
    settings: Any, kb: Any, base: TextGroupModel | None = None
) -> tuple[BehaviorTextModel | None, str]:
    """The member for the engine, fail-closed: ``(model, status)``. ``base`` is an already
    loaded text member (the same artifact); otherwise ``TEXT_GROUP_PATH`` is read."""
    if not behavior_text_votes(settings):
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
        return BehaviorTextModel(base, behavior_text_params(settings)), "active"
    except ValueError as error:
        return None, f"invalid settings: {error}"


def behavior_text_bound_fingerprint(base: str, model: BehaviorTextModel) -> str:
    """The combiner pipeline fingerprint when the member votes (as the text member binds it)."""
    digest = hashlib.sha256()
    digest.update(base.encode())
    digest.update(model.fingerprint.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:24]


class BehaviorTextClassifier:
    """Ensemble member: the behaviour model's score for each eligible group at least one
    statement reached. It cannot make a group eligible."""

    name = BEHAVIOR_TEXT_NAME

    def __init__(self, model: BehaviorTextModel | None = None) -> None:
        self.model = model

    def classify(self, context: Any) -> Any:
        from apt_classifier.classifiers.base import build_result, group_ids

        statements = tuple(getattr(context, "statements", ()) or ())
        if not statements:
            from apt_classifier.attack.parse import split_sentences

            statements = tuple(
                s
                for s in split_sentences(str(getattr(context, "narrative", "") or ""))
                if s.strip()
            )
        if self.model is None or not statements:
            return build_result(self.name, context, {}, applicable=False)
        allowed = set(group_ids(context))
        scores = {g: s for g, s in self.model.scores(statements).items() if g in allowed}
        signals = {g: (f"behavior_text:p={s:.3f}",) for g, s in scores.items()}
        return build_result(self.name, context, scores, applicable=bool(scores), signals=signals)
