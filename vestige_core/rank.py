"""The one ranker: store text's TF-IDF, computed incrementally and exactly.

Store text (``knowledge_store.members.StoreMembers._text_slice`` + ``text_group.TextGroupModel``)
rebuilds every actor document for every case. Each quantity it needs is a sum over posts, so the
counts are summed per post once and a case is a few sparse products:

- documents = Lᵀ · diag(w) · PostCounts, with ``w`` the posts the case may read;
- a term counts only where the readable statements hold it at least ``min_df`` times
  (wᵀ · PostDF ≥ min_df), as the slice re-zeroes it;
- then the same IDF over actor documents, log1p and L2, cosine with the report, z-score, the
  prominence prior and a softmax over the actors sharing a term.

Integer counts sum exactly in float32, so the result equals the slice's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from apt_classifier.text_group import _EXTRA_STOP, DEFAULT_PARAMS, _preprocess, prominence_log_prior


@dataclass(frozen=True)
class Ranking:
    order: tuple[tuple[str, float], ...]  # actor, softmax score; falling, ties by id

    @property
    def first(self) -> str | None:
        return self.order[0][0] if self.order else None

    @property
    def margin(self) -> float | None:
        return self.order[0][1] - self.order[1][1] if len(self.order) > 1 else None

    def rank_of(self, actor: str) -> int | None:
        for i, (a, _) in enumerate(self.order, 1):
            if a == actor:
                return i
        return None


class TextRanker:
    """Actor documents from labelled posts; ``rank`` reads only the posts a case may read."""

    def __init__(
        self,
        rows: Sequence[tuple[str, str, str]],  # (post key, actor, statement text)
        posts: Mapping[str, Mapping[str, Any]],  # key -> {"source", "available_from"}
        actors: Sequence[str],  # every actor that may be scored (the KB's groups, maybe more)
        log_prior: np.ndarray | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        vectorizer: Any = None,
    ) -> None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.actors = tuple(actors)
        self.prior = (
            np.zeros(len(self.actors)) if log_prior is None else np.asarray(log_prior, np.float64)
        )
        if self.prior.shape != (len(self.actors),):
            raise ValueError("prior does not match the actor list")
        actor_index = {a: i for i, a in enumerate(self.actors)}
        if vectorizer is None:
            vectorizer = CountVectorizer(
                preprocessor=_preprocess,
                analyzer="word",
                ngram_range=(1, int(self.params["ngram_max"])),
                stop_words=sorted(ENGLISH_STOP_WORDS | set(_EXTRA_STOP)),
                min_df=int(self.params["min_df"]),
                dtype=np.float32,
            )
            counts = vectorizer.fit_transform([r[2] for r in rows]).tocsr()
        else:
            counts = vectorizer.transform([r[2] for r in rows]).tocsr()
        # the vocabulary is fitted on every row (as store text's full model); then only the
        # rows of scorable actors count
        kept = [i for i, r in enumerate(rows) if r[1] in actor_index]
        rows = [rows[i] for i in kept]
        counts = counts[kept]
        self.vectorizer = vectorizer
        # one aggregate row per (post, actor): a post has one label, but keep it general
        units = sorted({(r[0], r[1]) for r in rows})
        unit_index = {u: i for i, u in enumerate(units)}
        member = sparse.csr_matrix(
            (
                np.ones(len(rows), np.float32),
                ([unit_index[(r[0], r[1])] for r in rows], np.arange(len(rows))),
            ),
            shape=(len(units), len(rows)),
        )
        self.unit_counts = (member @ counts).tocsr()
        present = counts.copy()
        present.data = np.ones_like(present.data)
        self.unit_df = (member @ present).tocsr()
        self.unit_posts = [u[0] for u in units]
        self.unit_actor = np.asarray([actor_index[u[1]] for u in units])
        self.posts = {k: dict(v) for k, v in posts.items()}
        self._unit_date = np.asarray(
            [str(self.posts.get(k, {}).get("available_from") or "9999") for k in self.unit_posts]
        )

    @property
    def vocabulary_size(self) -> int:
        return int(self.unit_counts.shape[1])

    def weights(
        self,
        as_of: str | None,
        hidden: Iterable[str] = (),
        *,
        allowed_actors: Iterable[str] | None = None,
        posts: Iterable[str] | None = None,
    ) -> np.ndarray:
        """1 for each unit the case may read: dated strictly before ``as_of``, not hidden, of an
        allowed actor, and (if given) among ``posts``."""
        keep = np.ones(len(self.unit_posts), dtype=bool)
        if as_of is not None:
            keep &= self._unit_date < as_of
        hidden = set(hidden)
        if hidden:
            keep &= np.asarray([k not in hidden for k in self.unit_posts])
        if posts is not None:
            chosen = set(posts)
            keep &= np.asarray([k in chosen for k in self.unit_posts])
        if allowed_actors is not None:
            index = {a: i for i, a in enumerate(self.actors)}
            allowed = np.zeros(len(self.actors), dtype=bool)
            allowed[[index[a] for a in allowed_actors if a in index]] = True
            keep &= allowed[self.unit_actor]
        return keep.astype(np.float32)

    def rank(self, text: str, w: np.ndarray) -> Ranking:
        if not (text or "").strip() or not w.any():
            return Ranking(())
        diag = sparse.diags(w)
        df = np.asarray((sparse.csr_matrix(w) @ self.unit_df).todense()).ravel()
        keep_terms = (df >= int(self.params["min_df"])).astype(np.float32)
        label = sparse.csr_matrix(
            (np.ones(len(self.unit_actor), np.float32), (self.unit_actor, np.arange(len(w)))),
            shape=(len(self.actors), len(w)),
        )
        documents = (label @ diag @ self.unit_counts @ sparse.diags(keep_terms)).tocsr()
        documents.eliminate_zeros()
        n_actors, vocab = documents.shape
        doc_df = np.bincount(documents.indices, minlength=vocab)
        idf = (np.log((1.0 + n_actors) / (1.0 + doc_df)) + 1.0).astype(np.float32)
        documents.data = np.log1p(documents.data)
        documents = _l2(documents @ sparse.diags(idf))
        row = self.vectorizer.transform([text]).tocsr()
        if row.nnz == 0:
            return Ranking(())
        row.data = np.log1p(row.data)
        row = _l2(row @ sparse.diags(idf))
        cosine = np.asarray((row @ documents.T.tocsc()).todense(), dtype=np.float64)[0]
        if not np.any(cosine > 0):
            return Ranking(())
        z = (cosine - cosine.mean()) / (cosine.std() + 1e-9)
        score = z + float(self.params["prior"]) * self.prior
        support = cosine > 0
        kept = score[support]
        p = np.exp(kept - kept.max())
        p /= p.sum()
        actors = [a for a, s in zip(self.actors, support, strict=True) if s]
        order = sorted(
            ((a, float(v)) for a, v in zip(actors, p, strict=True)), key=lambda x: (-x[1], x[0])
        )
        return Ranking(tuple(order))


def _l2(matrix: Any) -> Any:
    from sklearn.preprocessing import normalize

    return normalize(matrix, norm="l2")


def from_store_members(members: Any, kb: Any, *, prior: bool = True) -> TextRanker:
    """A ranker over a store-members artifact's labelled text (``store_members.json.gz``), scoring
    the knowledge base's groups as store text does."""
    groups = tuple(sorted(getattr(kb, "groups", {}) or {}))
    rows = [(r["key"], r["group"], r["text"]) for r in members.text_rows]
    log_prior = prominence_log_prior(kb, groups) if prior else None
    return TextRanker(rows, members.posts, groups, log_prior)
