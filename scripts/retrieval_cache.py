"""Cross-validation helpers shared by the acceptance fit and the dev-only tuning harness.

``scripts/fit_acceptance_model.py --threshold-selection oof_runtime`` and the harness
(``scripts/tune_retrieval.py``, docs 21.3.17) choose the acceptance threshold on out-of-fold
probabilities. The folds must keep near-copies of a text together, or the held-out fold scores
a case whose twin trained the model: the public dev cases have no document ids
(``document_id == case_id``), so near-duplicates are found from the texts themselves.

- :func:`near_duplicate_groups`: union-find over cases whose embedded texts have a cosine of at
  least ``min_similarity`` (0.90), one group label per case.
- :func:`stratified_group_folds`: the greedy assignment of scikit-learn's
  ``StratifiedGroupKFold(shuffle=True)`` (groups never split, the per-stratum shares kept as even
  as possible across folds), reimplemented here because scikit-learn is not a dependency.
- :func:`out_of_fold_probabilities`: one :class:`AcceptanceModel` per fold, each scoring the
  rows of its held-out fold.

Everything here is deterministic for a given seed and reads nothing but its arguments.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from apt_classifier.acceptance import AcceptanceModel

DEFAULT_FOLDS = 5
DEFAULT_CV_SEED = 7
DEFAULT_GROUP_SIMILARITY = 0.90


def near_duplicate_groups(
    vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_similarity: float = DEFAULT_GROUP_SIMILARITY,
    block: int = 1024,
) -> list[int]:
    """Group label per row: rows whose cosine is at least ``min_similarity`` share a label,
    transitively (union-find). Labels are 0, 1, ... in order of each group's first row."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    count = len(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    parent = list(range(count))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    for start in range(0, count, max(1, int(block))):
        similarities = unit[start : start + block] @ unit.T
        rows, columns = np.nonzero(similarities >= float(min_similarity))
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            left, right = find(start + row), find(column)
            if left != right:
                # the smaller index becomes the root, so labels do not depend on pair order
                parent[max(left, right)] = min(left, right)
    labels: dict[int, int] = {}
    return [labels.setdefault(find(item), len(labels)) for item in range(count)]


def stratified_group_folds(
    strata: Sequence[Hashable],
    groups: Sequence[Hashable],
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_CV_SEED,
) -> list[int]:
    """Fold (0..n_splits-1) per item; items of one group always share a fold.

    Groups are visited in a seeded random order, stably re-sorted by how unevenly their items
    spread over the strata (most uneven first), and each goes to the fold that keeps the
    per-stratum fold shares most even (mean over strata of the standard deviation across folds;
    ties go to the fold with fewer items, then the lower index)."""
    if len(strata) != len(groups):
        raise ValueError("strata and groups are misaligned")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    stratum_index = {value: index for index, value in enumerate(sorted(set(strata), key=str))}
    group_index = {value: index for index, value in enumerate(sorted(set(groups), key=str))}
    if len(group_index) < n_splits:
        raise ValueError(f"{len(group_index)} groups cannot fill {n_splits} folds")
    counts = np.zeros((len(group_index), len(stratum_index)), dtype=np.float64)
    for stratum, group in zip(strata, groups, strict=True):
        counts[group_index[group], stratum_index[stratum]] += 1.0
    totals = counts.sum(axis=0)
    order = np.random.default_rng(int(seed)).permutation(len(group_index))
    order = order[np.argsort(-np.std(counts[order], axis=1), kind="stable")]
    fold_counts = np.zeros((int(n_splits), len(stratum_index)), dtype=np.float64)
    group_fold = np.zeros(len(group_index), dtype=np.int64)
    for group in order.tolist():
        best_fold, best_evaluation, best_size = 0, np.inf, np.inf
        for fold in range(int(n_splits)):
            fold_counts[fold] += counts[group]
            evaluation = float(np.mean(np.std(fold_counts / totals, axis=0)))
            fold_counts[fold] -= counts[group]
            size = float(fold_counts[fold].sum())
            tied = bool(np.isclose(evaluation, best_evaluation))
            if (evaluation < best_evaluation and not tied) or (tied and size < best_size):
                best_fold, best_evaluation, best_size = fold, evaluation, size
        fold_counts[best_fold] += counts[group]
        group_fold[group] = best_fold
    return [int(group_fold[group_index[group]]) for group in groups]


def case_folds(
    case_ids: Sequence[str],
    strata: Sequence[Hashable],
    vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_CV_SEED,
    min_similarity: float = DEFAULT_GROUP_SIMILARITY,
) -> tuple[dict[str, int], dict[str, int]]:
    """``(fold per case id, near-duplicate group per case id)`` for cases with one text vector
    each, stratified by ``strata`` (the dataset)."""
    if not (len(case_ids) == len(strata) == len(vectors)):
        raise ValueError("case ids, strata and vectors are misaligned")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")
    groups = near_duplicate_groups(vectors, min_similarity=min_similarity)
    folds = stratified_group_folds(strata, groups, n_splits=n_splits, seed=seed)
    return (
        dict(zip(case_ids, folds, strict=True)),
        dict(zip(case_ids, groups, strict=True)),
    )


def out_of_fold_probabilities(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    row_folds: Sequence[int],
    *,
    attack_version: str,
    l2: float = 1.0,
) -> np.ndarray:
    """Acceptance probability of every row from a model fitted on the other folds' rows (the
    same :meth:`AcceptanceModel.fit` as the final model; prior features are already in the
    rows)."""
    folds = np.asarray(row_folds, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    if not (len(rows) == len(truth) == len(folds)):
        raise ValueError("rows, labels and folds are misaligned")
    output = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in sorted(set(folds.tolist())):
        held_out = folds == fold
        train = np.flatnonzero(~held_out)
        if not len(train):
            raise ValueError("a fold holds every row; nothing is left to fit on")
        model = AcceptanceModel.fit(
            [rows[index] for index in train.tolist()],
            truth[train].tolist(),
            attack_version=attack_version,
            l2=l2,
        )
        for index in np.flatnonzero(held_out).tolist():
            output[index] = model.probability(dict(rows[index]))
    return output


def fold_summary(
    case_folds_by_id: Mapping[str, int], groups_by_id: Mapping[str, int], strata: Mapping[str, Any]
) -> dict[str, Any]:
    """Sizes recorded with a fit: cases and strata per fold, group count and largest group."""
    sizes: dict[int, int] = {}
    per_stratum: dict[int, dict[str, int]] = {}
    for case_id, fold in case_folds_by_id.items():
        sizes[fold] = sizes.get(fold, 0) + 1
        bucket = per_stratum.setdefault(fold, {})
        stratum = str(strata.get(case_id))
        bucket[stratum] = bucket.get(stratum, 0) + 1
    group_sizes: dict[int, int] = {}
    for group in groups_by_id.values():
        group_sizes[group] = group_sizes.get(group, 0) + 1
    return {
        "cases_per_fold": [sizes.get(fold, 0) for fold in sorted(sizes)],
        "strata_per_fold": [per_stratum.get(fold, {}) for fold in sorted(sizes)],
        "groups": len(group_sizes),
        "largest_group": max(group_sizes.values(), default=0),
        "cases_in_multi_case_groups": sum(size for size in group_sizes.values() if size > 1),
    }
