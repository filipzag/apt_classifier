"""Grouped, stratified cross-validation for the acceptance threshold (scripts/retrieval_cache.py)."""

from __future__ import annotations

import numpy as np
import pytest

from apt_classifier.acceptance import FEATURE_NAMES, AcceptanceModel
from scripts.retrieval_cache import (
    case_folds,
    fold_summary,
    near_duplicate_groups,
    out_of_fold_probabilities,
    stratified_group_folds,
)


def test_near_duplicate_groups_are_transitive_and_respect_the_threshold():
    vectors = [
        [1.0, 0.0, 0.0],
        [0.95, 0.31, 0.0],  # cos 0.95 to row 0
        [0.81, 0.59, 0.0],  # cos 0.95 to row 1, only 0.81 to row 0: joined through row 1
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],  # same direction as row 3 (length does not matter)
        [0.0, 1.0, 0.0],
    ]
    groups = near_duplicate_groups(vectors, min_similarity=0.90)
    assert groups == [0, 0, 0, 1, 1, 2]
    assert near_duplicate_groups(vectors, min_similarity=0.999) == [0, 1, 2, 3, 3, 4]
    # block size does not change the result
    assert near_duplicate_groups(vectors, min_similarity=0.90, block=2) == groups
    with pytest.raises(ValueError):
        near_duplicate_groups([1.0, 2.0])


def _strata_and_groups(seed: int = 0):
    rng = np.random.default_rng(seed)
    strata, groups = [], []
    for group in range(120):
        size = int(rng.integers(1, 4))
        stratum = ("tram", "expert", "derived")[group % 3]
        strata.extend([stratum] * size)
        groups.extend([group] * size)
    return strata, groups


def test_stratified_group_folds_keep_groups_whole_and_balance_strata():
    strata, groups = _strata_and_groups()
    folds = stratified_group_folds(strata, groups, n_splits=5, seed=7)
    assert set(folds) == set(range(5))
    by_group: dict[int, set[int]] = {}
    for group, fold in zip(groups, folds, strict=True):
        by_group.setdefault(group, set()).add(fold)
    assert all(len(assigned) == 1 for assigned in by_group.values())
    for stratum in ("tram", "expert", "derived"):
        counts = [
            sum(1 for s, f in zip(strata, folds, strict=True) if s == stratum and f == fold)
            for fold in range(5)
        ]
        assert max(counts) - min(counts) <= 3, (stratum, counts)
    # deterministic per seed, different across seeds
    assert stratified_group_folds(strata, groups, n_splits=5, seed=7) == folds
    assert stratified_group_folds(strata, groups, n_splits=5, seed=11) != folds
    with pytest.raises(ValueError):
        stratified_group_folds(["a", "b"], [0, 1], n_splits=5)
    with pytest.raises(ValueError):
        stratified_group_folds(["a"], [0, 1])


def test_case_folds_put_near_copies_in_one_fold():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(40, 8))
    vectors = np.vstack([base, base[:5] + 1e-4])  # five near-copies of the first five cases
    case_ids = [f"c{i}" for i in range(len(vectors))]
    strata = ["tram" if i % 2 else "expert" for i in range(len(vectors))]
    folds, groups = case_folds(case_ids, strata, vectors, n_splits=4, seed=7)
    for original in range(5):
        copy = 40 + original
        assert groups[f"c{original}"] == groups[f"c{copy}"]
        assert folds[f"c{original}"] == folds[f"c{copy}"]
    summary = fold_summary(folds, groups, dict(zip(case_ids, strata, strict=True)))
    assert sum(summary["cases_per_fold"]) == len(case_ids)
    assert summary["groups"] == 40 and summary["largest_group"] == 2
    assert summary["cases_in_multi_case_groups"] == 10
    with pytest.raises(ValueError, match="unique"):
        case_folds(["a", "a"], ["x", "x"], [[1.0], [0.5]], n_splits=2)


def test_out_of_fold_probabilities_never_score_a_row_with_its_own_fold():
    rng = np.random.default_rng(5)
    rows, labels = [], []
    for _ in range(200):
        dense = float(rng.uniform(0.3, 0.95))
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        row.update({"dense_score": dense, "dense_present": 1.0})
        rows.append(row)
        labels.append(int(dense + rng.normal(0, 0.05) > 0.7))
    row_folds = [index % 4 for index in range(len(rows))]
    oof = out_of_fold_probabilities(rows, labels, row_folds, attack_version="t", l2=1.0)
    assert not np.isnan(oof).any()
    held_out = [index for index, fold in enumerate(row_folds) if fold == 2]
    trained = [index for index, fold in enumerate(row_folds) if fold != 2]
    model = AcceptanceModel.fit(
        [rows[i] for i in trained], [labels[i] for i in trained], attack_version="t"
    )
    assert oof[held_out] == pytest.approx([model.probability(rows[i]) for i in held_out])
    in_sample = AcceptanceModel.fit(rows, labels, attack_version="t")
    assert oof[0] != in_sample.probability(rows[0])
    with pytest.raises(ValueError):
        out_of_fold_probabilities(rows, labels, row_folds[:-1], attack_version="t")
