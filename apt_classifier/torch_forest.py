"""Random forest on binary features, trained and evaluated with PyTorch (GPU when available).

scikit-learn's forest runs on the CPU only and stores a dense class-probability row at every
node, which for a 171-class problem needs gigabytes. This implementation grows a batch of trees
level by level on the configured device:

- every feature is 0/1, so a split is "feature present" versus "absent";
- bootstrap multiplicities and (optional) class weights are integer counts, so every histogram
  is an integer scatter-add and training is deterministic for a seed on a given device;
- like scikit-learn, the ``max_features`` candidates of a node are drawn only among features that
  vary inside the node, so sparse data still finds informative splits;
- splits minimise Gini impurity subject to ``min_samples_leaf`` (bootstrap counts);
- leaves keep only their non-zero class fractions, and inference accumulates them in fixed point,
  so probabilities are identical on CPU and GPU and ties never depend on float summation order.

The exported arrays (:class:`ForestArrays`) are plain integer/float arrays saved without pickle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

FIXED_POINT = 1 << 24  # leaf fractions are summed as integers of 2**-24
CLASS_WEIGHT_SCALE = 100  # class weights become integer multiples of 1/100


def resolve_device(device: str | None = None) -> str:
    """``"auto"``/``None`` -> ``"cuda"`` when a GPU is usable, else ``"cpu"``."""
    import torch

    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


@dataclass(frozen=True)
class ForestArrays:
    """Exported trees. Node ``i`` is internal when ``feature[i] >= 0``: samples with feature
    value 0 go to ``left[i]``, value 1 to ``right[i]``. A leaf's class fractions are
    ``leaf_class/leaf_prob[leaf_start[i] : leaf_start[i] + leaf_count[i]]``."""

    roots: np.ndarray  # int32 (trees,)
    feature: np.ndarray  # int32 (nodes,), -1 for leaves
    left: np.ndarray  # int32 (nodes,)
    right: np.ndarray  # int32 (nodes,)
    leaf_start: np.ndarray  # int32 (nodes,)
    leaf_count: np.ndarray  # int32 (nodes,)
    leaf_class: np.ndarray  # int32 (entries,)
    leaf_prob: np.ndarray  # float32 (entries,)
    n_classes: int
    n_features: int
    max_depth: int

    @property
    def n_trees(self) -> int:
        return int(self.roots.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.feature.shape[0])

    def to_npz_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}roots": self.roots,
            f"{prefix}feature": self.feature,
            f"{prefix}left": self.left,
            f"{prefix}right": self.right,
            f"{prefix}leaf_start": self.leaf_start,
            f"{prefix}leaf_count": self.leaf_count,
            f"{prefix}leaf_class": self.leaf_class,
            f"{prefix}leaf_prob": self.leaf_prob,
            f"{prefix}shape": np.asarray(
                [self.n_classes, self.n_features, self.max_depth], dtype=np.int64
            ),
        }

    @classmethod
    def from_npz(cls, data: Any, prefix: str) -> ForestArrays:
        n_classes, n_features, max_depth = (int(v) for v in data[f"{prefix}shape"])
        arrays = cls(
            roots=np.asarray(data[f"{prefix}roots"], dtype=np.int32),
            feature=np.asarray(data[f"{prefix}feature"], dtype=np.int32),
            left=np.asarray(data[f"{prefix}left"], dtype=np.int32),
            right=np.asarray(data[f"{prefix}right"], dtype=np.int32),
            leaf_start=np.asarray(data[f"{prefix}leaf_start"], dtype=np.int32),
            leaf_count=np.asarray(data[f"{prefix}leaf_count"], dtype=np.int32),
            leaf_class=np.asarray(data[f"{prefix}leaf_class"], dtype=np.int32),
            leaf_prob=np.asarray(data[f"{prefix}leaf_prob"], dtype=np.float32),
            n_classes=n_classes,
            n_features=n_features,
            max_depth=max_depth,
        )
        arrays.validate()
        return arrays

    def validate(self) -> None:
        nodes = self.n_nodes
        if not (self.left.shape == self.right.shape == self.feature.shape == self.leaf_start.shape):
            raise ValueError("forest arrays have inconsistent node counts")
        internal = self.feature >= 0
        if internal.any() and (
            int(self.feature[internal].max()) >= self.n_features
            or int(self.left[internal].min()) < 0
            or int(self.right[internal].min()) < 0
            or int(self.left[internal].max()) >= nodes
            or int(self.right[internal].max()) >= nodes
        ):
            raise ValueError("forest arrays reference features or nodes out of range")
        if self.roots.size and (int(self.roots.min()) < 0 or int(self.roots.max()) >= nodes):
            raise ValueError("forest roots out of range")
        leaves = ~internal
        if leaves.any():
            ends = self.leaf_start[leaves].astype(np.int64) + self.leaf_count[leaves]
            if int(self.leaf_start[leaves].min()) < 0 or int(ends.max()) > self.leaf_class.shape[0]:
                raise ValueError("forest leaf ranges out of bounds")
        if self.leaf_class.size and int(self.leaf_class.max()) >= self.n_classes:
            raise ValueError("forest leaf classes out of range")


class ForestPredictor:
    """Device-resident copy of :class:`ForestArrays` for batched inference."""

    def __init__(self, arrays: ForestArrays, device: str | None = None) -> None:
        import torch

        self.arrays = arrays
        self.device = resolve_device(device)
        dev = torch.device(self.device)
        self._roots = torch.as_tensor(arrays.roots, dtype=torch.int64, device=dev)
        self._feature = torch.as_tensor(arrays.feature, dtype=torch.int64, device=dev)
        self._left = torch.as_tensor(arrays.left, dtype=torch.int64, device=dev)
        self._right = torch.as_tensor(arrays.right, dtype=torch.int64, device=dev)
        self._leaf_start = torch.as_tensor(arrays.leaf_start, dtype=torch.int64, device=dev)
        self._leaf_count = torch.as_tensor(arrays.leaf_count, dtype=torch.int64, device=dev)
        self._leaf_class = torch.as_tensor(arrays.leaf_class, dtype=torch.int64, device=dev)
        fixed = np.rint(arrays.leaf_prob.astype(np.float64) * FIXED_POINT).astype(np.int64)
        self._leaf_fixed = torch.as_tensor(fixed, dtype=torch.int64, device=dev)

    def predict_proba(self, features: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
        """Class probabilities (samples x classes) as float64, averaged over the trees."""
        import torch

        matrix = np.asarray(features)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.arrays.n_features:
            raise ValueError(f"expected {self.arrays.n_features} features, got {matrix.shape[1]}")
        output = np.zeros((matrix.shape[0], self.arrays.n_classes), dtype=np.float64)
        dev = torch.device(self.device)
        trees = self.arrays.n_trees
        with torch.no_grad():
            for start in range(0, matrix.shape[0], batch_size):
                block = torch.as_tensor(
                    (matrix[start : start + batch_size] > 0.5).astype(np.uint8), device=dev
                )
                rows = block.shape[0]
                sample = torch.arange(rows, device=dev).repeat_interleave(trees)
                node = self._roots.repeat(rows)
                for _ in range(self.arrays.max_depth + 1):
                    feature = self._feature[node]
                    internal = feature >= 0
                    if not bool(internal.any()):
                        break
                    live = internal.nonzero(as_tuple=True)[0]
                    value = block[sample[live], feature[live]]
                    current = node[live]
                    node[live] = torch.where(value > 0, self._right[current], self._left[current])
                counts = self._leaf_count[node]
                owner = torch.arange(node.shape[0], device=dev).repeat_interleave(counts)
                first = torch.cumsum(counts, 0) - counts
                offset = torch.arange(owner.shape[0], device=dev) - first[owner]
                entry = self._leaf_start[node][owner] + offset
                accumulator = torch.zeros(
                    rows * self.arrays.n_classes, dtype=torch.int64, device=dev
                )
                accumulator.index_add_(
                    0,
                    sample[owner] * self.arrays.n_classes + self._leaf_class[entry],
                    self._leaf_fixed[entry],
                )
                output[start : start + rows] = accumulator.view(
                    rows, self.arrays.n_classes
                ).cpu().numpy().astype(np.float64) / (FIXED_POINT * trees)
        return output


def _max_features(value: Any, n_features: int) -> int:
    if value in (None, "sqrt"):
        return max(1, int(math.sqrt(n_features)))
    if value == "log2":
        return max(1, int(math.log2(n_features)))
    if isinstance(value, float) and 0 < value <= 1:
        return max(1, int(value * n_features))
    return max(1, min(int(value), n_features))


def fit_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    n_estimators: int = 200,
    max_features: Any = "sqrt",
    min_samples_leaf: int = 1,
    max_depth: int = 128,
    class_weight: str | None = None,
    seed: int = 7,
    device: str | None = None,
    tree_batch: int = 32,
    node_batch: int = 8192,
    sample_batch: int = 1_000_000,
) -> ForestArrays:
    """Grow ``n_estimators`` bootstrap trees on binary ``features`` (samples x features)."""
    import torch

    dev = torch.device(resolve_device(device))
    x_np = (np.asarray(features) > 0.5).astype(np.uint8)
    y_np = np.asarray(labels, dtype=np.int64)
    n_samples, n_features = x_np.shape
    if n_samples == 0 or y_np.shape[0] != n_samples:
        raise ValueError("features and labels must be non-empty and aligned")
    if int(y_np.min()) < 0 or int(y_np.max()) >= n_classes:
        raise ValueError("labels must be class indices in [0, n_classes)")
    k = min(_max_features(max_features, n_features), n_features)
    X = torch.as_tensor(x_np, device=dev)
    y = torch.as_tensor(y_np, device=dev)
    # compressed rows of the present features: a padded matrix would be as wide as the densest
    # sample and dominate GPU memory
    rows_nz, cols_nz = np.nonzero(x_np)
    row_len = torch.as_tensor(np.bincount(rows_nz, minlength=n_samples), device=dev)
    row_ptr = torch.cumsum(row_len, 0) - row_len
    col_idx = torch.as_tensor(cols_nz.astype(np.int64), device=dev)
    if class_weight == "balanced":
        counts = np.bincount(y_np, minlength=n_classes).astype(np.float64)
        weights = np.where(counts > 0, n_samples / (n_classes * np.maximum(counts, 1)), 0.0)
        class_scale = torch.as_tensor(
            np.maximum(1, np.rint(weights * CLASS_WEIGHT_SCALE)).astype(np.int64), device=dev
        )
    elif class_weight is None:
        class_scale = torch.ones(n_classes, dtype=torch.int64, device=dev)
    else:
        raise ValueError(f"unsupported class_weight {class_weight!r}")
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))
    min_leaf = max(1, int(min_samples_leaf))

    parts: dict[str, list[np.ndarray]] = {
        "feature": [],
        "left": [],
        "right": [],
        "leaf_start": [],
        "leaf_count": [],
        "leaf_class": [],
        "leaf_prob": [],
    }
    roots: list[np.ndarray] = []
    node_base = 0
    entry_base = 0
    deepest = 0
    for first_tree in range(0, n_estimators, tree_batch):
        trees = min(tree_batch, n_estimators - first_tree)
        draws = torch.randint(0, n_samples, (trees, n_samples), generator=generator, device=dev)
        boot = torch.zeros((trees, n_samples), dtype=torch.int64, device=dev)
        boot.scatter_add_(1, draws, torch.ones_like(draws))
        node_of, sample_of = boot.nonzero(as_tuple=True)  # node ids 0..trees-1 are the roots
        count_of = boot[node_of, sample_of]
        weight_of = count_of * class_scale[y[sample_of]]
        del draws, boot
        capacity = max(1024, trees * 64)
        feature_np = np.full(capacity, -1, dtype=np.int32)
        left_np = np.full(capacity, -1, dtype=np.int32)
        right_np = np.full(capacity, -1, dtype=np.int32)
        leaf_start_np = np.zeros(capacity, dtype=np.int32)
        leaf_count_np = np.zeros(capacity, dtype=np.int32)
        next_id = trees
        active = torch.arange(trees, device=dev)
        depth = 0
        while active.numel():
            n_active = int(active.numel())
            first_active = int(active[0])  # active ids are one contiguous block
            pos = node_of - first_active
            totals = torch.zeros(n_active * n_classes, dtype=torch.int64, device=dev)
            totals.index_add_(0, pos * n_classes + y[sample_of], weight_of)
            totals = totals.view(n_active, n_classes)
            boot_total = torch.zeros(n_active, dtype=torch.int64, device=dev)
            boot_total.index_add_(0, pos, count_of)
            splittable = ((totals > 0).sum(1) > 1) & (boot_total >= 2 * min_leaf)
            if depth >= max_depth:
                splittable &= False
            best_feature = torch.full((n_active,), -1, dtype=torch.int64, device=dev)
            split_rows = splittable.nonzero(as_tuple=True)[0]
            # node batches bounded by node count and by the sample entries they hold, so memory
            # stays flat at the root level where a few nodes hold every sample
            entries = torch.bincount(pos, minlength=n_active)[split_rows].cpu().numpy()
            if entries.size:
                before = np.cumsum(entries) - entries
                batch_of = np.maximum(
                    before // max(1, sample_batch), np.arange(entries.size) // max(1, node_batch)
                )
                cuts = np.flatnonzero(np.diff(batch_of)) + 1
                edges = np.concatenate([[0], cuts, [entries.size]])
                bounds = list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))
            else:
                bounds = []
            for lo, hi in bounds:
                rows = split_rows[lo:hi]
                n_rows = int(rows.numel())
                local = torch.full((n_active,), -1, dtype=torch.int64, device=dev)
                local[rows] = torch.arange(n_rows, device=dev)
                sample_local = local[pos]
                in_chunk = sample_local >= 0
                s_local = sample_local[in_chunk]
                s_sample = sample_of[in_chunk]
                s_count = count_of[in_chunk]
                s_weight = weight_of[in_chunk]
                lengths = row_len[s_sample]
                owner = torch.repeat_interleave(torch.arange(s_sample.numel(), device=dev), lengths)
                starts = torch.cumsum(lengths, 0) - lengths
                offset = torch.arange(owner.numel(), device=dev) - starts[owner]
                columns = col_idx[row_ptr[s_sample][owner] + offset]
                presence = torch.zeros(n_rows * n_features, dtype=torch.int32, device=dev)
                presence.index_add_(
                    0, s_local[owner] * n_features + columns, s_count[owner].to(torch.int32)
                )
                presence = presence.view(n_rows, n_features)
                del lengths, owner, starts, offset, columns
                node_boot = boot_total[rows]
                varies = (presence > 0) & (presence < node_boot[:, None])
                scores = torch.rand((n_rows, n_features), generator=generator, device=dev)
                scores = torch.where(varies, scores, torch.full_like(scores, -1.0))
                top_scores, cand = scores.topk(k, dim=1)
                cand_valid = top_scores >= 0
                present_s, present_j = (X[s_sample[:, None], cand[s_local]] > 0).nonzero(
                    as_tuple=True
                )
                slots = s_local[present_s] * k + present_j
                left_hist = torch.zeros(n_rows * k * n_classes, dtype=torch.int32, device=dev)
                left_hist.index_add_(
                    0,
                    slots * n_classes + y[s_sample[present_s]],
                    s_weight[present_s].to(torch.int32),
                )
                left_boot = torch.zeros(n_rows * k, dtype=torch.int32, device=dev)
                left_boot.index_add_(0, slots, s_count[present_s].to(torch.int32))
                del present_s, present_j, slots
                # float32 (consumer GPUs run float64 at a small fraction of float32 speed); counts
                # are exact far beyond what occurs, and the gain threshold below is relative
                lh = left_hist.view(n_rows, k, n_classes).to(torch.float32)
                parent = totals[rows].to(torch.float32)
                rh = parent[:, None, :] - lh
                n_left = lh.sum(2)
                n_right = rh.sum(2)
                q = lh.pow(2).sum(2) / n_left.clamp(min=1e-12) + rh.pow(2).sum(2) / n_right.clamp(
                    min=1e-12
                )
                parent_q = parent.pow(2).sum(1) / parent.sum(1).clamp(min=1e-12)
                left_boot = left_boot.view(n_rows, k)
                ok = (
                    cand_valid
                    & (left_boot >= min_leaf)
                    & ((node_boot[:, None] - left_boot) >= min_leaf)
                    & (n_left > 0)
                    & (n_right > 0)
                )
                gain = torch.where(ok, q - parent_q[:, None], torch.full_like(q, -1.0))
                best_gain, best_j = gain.max(dim=1)
                best_feature[rows] = torch.where(
                    best_gain > 1e-6 * parent.sum(1),
                    cand.gather(1, best_j[:, None]).squeeze(1),
                    torch.full_like(best_j, -1),
                )
            best_np = best_feature.cpu().numpy()
            active_np = np.arange(first_active, first_active + n_active)
            split_idx = np.flatnonzero(best_np >= 0)
            leaf_idx = np.flatnonzero(best_np < 0)
            if leaf_idx.size:
                histograms = totals[torch.as_tensor(leaf_idx, device=dev)].cpu().numpy()
                leaf_rows, leaf_classes = np.nonzero(histograms)
                per_leaf = np.bincount(leaf_rows, minlength=leaf_idx.size)
                starts = entry_base + np.cumsum(per_leaf) - per_leaf
                leaf_nodes = active_np[leaf_idx]
                leaf_start_np[leaf_nodes] = starts
                leaf_count_np[leaf_nodes] = per_leaf
                sums = histograms.sum(1).astype(np.float64)
                parts["leaf_class"].append(leaf_classes.astype(np.int32))
                parts["leaf_prob"].append(
                    (histograms[leaf_rows, leaf_classes] / sums[leaf_rows]).astype(np.float32)
                )
                entry_base += int(leaf_classes.size)
            if not split_idx.size:
                break
            n_split = int(split_idx.size)
            needed = next_id + 2 * n_split
            if needed > feature_np.size:
                grow = max(needed, feature_np.size * 2) - feature_np.size
                feature_np = np.concatenate([feature_np, np.full(grow, -1, np.int32)])
                left_np = np.concatenate([left_np, np.full(grow, -1, np.int32)])
                right_np = np.concatenate([right_np, np.full(grow, -1, np.int32)])
                leaf_start_np = np.concatenate([leaf_start_np, np.zeros(grow, np.int32)])
                leaf_count_np = np.concatenate([leaf_count_np, np.zeros(grow, np.int32)])
            parents = active_np[split_idx]
            children = next_id + 2 * np.arange(n_split)
            feature_np[parents] = best_np[split_idx]
            left_np[parents] = children
            right_np[parents] = children + 1
            # route the samples of split nodes to their children; samples in leaves are done
            row_child = torch.full((n_active,), -1, dtype=torch.int64, device=dev)
            row_child[torch.as_tensor(split_idx, device=dev)] = torch.as_tensor(
                children, device=dev
            )
            keep = row_child[pos] >= 0
            node_of, sample_of = node_of[keep], sample_of[keep]
            count_of, weight_of = count_of[keep], weight_of[keep]
            kept_pos = pos[keep]
            goes_right = X[sample_of, best_feature[kept_pos]] > 0
            node_of = row_child[kept_pos] + goes_right.to(torch.int64)
            active = torch.arange(next_id, needed, device=dev)
            next_id = needed
            depth += 1
            deepest = max(deepest, depth)
        internal = feature_np[:next_id] >= 0
        left_chunk = left_np[:next_id].copy()
        right_chunk = right_np[:next_id].copy()
        left_chunk[internal] += node_base
        right_chunk[internal] += node_base
        parts["feature"].append(feature_np[:next_id].copy())
        parts["left"].append(left_chunk)
        parts["right"].append(right_chunk)
        parts["leaf_start"].append(leaf_start_np[:next_id].copy())
        parts["leaf_count"].append(leaf_count_np[:next_id].copy())
        roots.append(node_base + np.arange(trees, dtype=np.int32))
        node_base += next_id
    arrays = ForestArrays(
        roots=np.concatenate(roots).astype(np.int32),
        feature=np.concatenate(parts["feature"]),
        left=np.concatenate(parts["left"]),
        right=np.concatenate(parts["right"]),
        leaf_start=np.concatenate(parts["leaf_start"]),
        leaf_count=np.concatenate(parts["leaf_count"]),
        leaf_class=np.concatenate(parts["leaf_class"])
        if parts["leaf_class"]
        else np.zeros(0, np.int32),
        leaf_prob=np.concatenate(parts["leaf_prob"])
        if parts["leaf_prob"]
        else np.zeros(0, np.float32),
        n_classes=int(n_classes),
        n_features=int(n_features),
        max_depth=int(deepest),
    )
    arrays.validate()
    return arrays
