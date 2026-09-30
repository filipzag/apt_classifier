"""PyTorch random forest on binary features (apt_classifier/torch_forest.py)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from apt_classifier.torch_forest import (  # noqa: E402
    ForestArrays,
    ForestPredictor,
    fit_forest,
    resolve_device,
)


def _xor(rows=600, noise=12, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.uniform(size=(rows, noise)) < 0.5).astype(np.uint8)
    return x, (x[:, 0] ^ x[:, 1]).astype(np.int64)


def test_learns_xor_and_probabilities_sum_to_one():
    x, y = _xor()
    arrays = fit_forest(x, y, 2, n_estimators=20, seed=1, device="cpu")
    probs = ForestPredictor(arrays, "cpu").predict_proba(x)
    assert (probs.argmax(1) == y).mean() > 0.99
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
    assert arrays.n_trees == 20 and arrays.max_depth >= 2


def test_training_is_deterministic_per_seed_and_arrays_round_trip(tmp_path):
    x, y = _xor()
    first = fit_forest(x, y, 2, n_estimators=8, seed=3, device="cpu")
    second = fit_forest(x, y, 2, n_estimators=8, seed=3, device="cpu")
    other = fit_forest(x, y, 2, n_estimators=8, seed=4, device="cpu")
    for name in ("feature", "left", "right", "leaf_class", "leaf_prob"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.feature, other.feature) or first.n_nodes != other.n_nodes
    np.savez(tmp_path / "f.npz", **first.to_npz_dict("g_"))
    with np.load(tmp_path / "f.npz", allow_pickle=False) as data:
        loaded = ForestArrays.from_npz(data, "g_")
    assert np.array_equal(
        ForestPredictor(loaded, "cpu").predict_proba(x),
        ForestPredictor(first, "cpu").predict_proba(x),
    )


def test_corrupted_arrays_are_rejected():
    x, y = _xor(rows=200)
    arrays = fit_forest(x, y, 2, n_estimators=2, seed=1, device="cpu")
    internal = int(np.flatnonzero(arrays.feature >= 0)[0])
    broken_left = arrays.left.copy()
    broken_left[internal] = arrays.n_nodes + 5
    with pytest.raises(ValueError):
        ForestArrays(**{**arrays.__dict__, "left": broken_left}).validate()
    broken_class = arrays.leaf_class.copy()
    broken_class[0] = 9
    with pytest.raises(ValueError):
        ForestArrays(**{**arrays.__dict__, "leaf_class": broken_class}).validate()
    with pytest.raises(ValueError):
        ForestPredictor(arrays, "cpu").predict_proba(np.zeros((1, arrays.n_features + 1)))


def test_min_samples_leaf_and_class_weight_shape_the_trees():
    rng = np.random.default_rng(5)
    x = (rng.uniform(size=(400, 10)) < 0.3).astype(np.uint8)
    y = np.where(x[:, 0] & x[:, 1], 1, 0).astype(np.int64)  # minority class
    coarse = fit_forest(x, y, 2, n_estimators=4, min_samples_leaf=50, seed=1, device="cpu")
    fine = fit_forest(x, y, 2, n_estimators=4, min_samples_leaf=1, seed=1, device="cpu")
    assert coarse.n_nodes < fine.n_nodes
    plain = ForestPredictor(
        fit_forest(x, y, 2, n_estimators=10, min_samples_leaf=40, seed=2, device="cpu"), "cpu"
    )
    balanced = ForestPredictor(
        fit_forest(
            x,
            y,
            2,
            n_estimators=10,
            min_samples_leaf=40,
            class_weight="balanced",
            seed=2,
            device="cpu",
        ),
        "cpu",
    )
    assert balanced.predict_proba(x)[:, 1].mean() > plain.predict_proba(x)[:, 1].mean()
    single = fit_forest(x, np.zeros(400, dtype=np.int64), 1, n_estimators=3, device="cpu")
    assert single.n_nodes == 3 and np.allclose(ForestPredictor(single, "cpu").predict_proba(x), 1.0)


def test_device_resolution_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_training_and_prediction_match_cpu_inference():
    x, y = _xor()
    arrays = fit_forest(x, y, 2, n_estimators=10, seed=1, device="cuda")
    gpu = ForestPredictor(arrays, "cuda").predict_proba(x)
    cpu = ForestPredictor(arrays, "cpu").predict_proba(x)
    assert np.array_equal(gpu, cpu)
    assert (gpu.argmax(1) == y).mean() > 0.99
