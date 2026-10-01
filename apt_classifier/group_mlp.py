"""Deep neural network (PyTorch MLP) that predicts the threat group from pre-extracted techniques.

The input (``MLPFeatures``) is the ``profile_forest.FeatureSpace`` 0/1 vector (one entry per ATT&CK
technique, a sub-technique also setting its parent, plus the analyst's motivation and target
geography with an "unknown" flag each) followed by one entry per ATT&CK software the evidence names.
The output is a softmax over the documented ATT&CK groups; ``members`` networks trained with
different seeds are averaged (logits) to cut seed variance.

Like the forest, the network is fitted from the ATT&CK knowledge base alone, never from benchmark
or case data. Training cases are synthetic partial observations of a report about a group:

- ``size_min``-``size_max`` of the group's documented techniques, independent of profile size;
- with probability ``software_rate``, the techniques of 1-``max_software`` of the group's
  documented software (each technique kept with ``software_keep``). Real reports describe the
  malware in use: on the CTI-HAL reports about 60% of the labelled techniques are in the group's
  own ATT&CK list and about 85% in that list plus its software's techniques. Each tool drawn is
  also *named* with ``software_named`` (reports name the malware they analyse);
- with ``software_false_rate``, one software the group does not document is named, drawn by how
  many groups use it (shared tools such as Mimikatz, or a wrong alias match);
- false positives drawn by how many groups document a technique (precision
  ``precision_min``-``precision_max``), as the mapping pipeline delivers;
- motivation and one target geography each given ``context_rate`` of the time.

Cases are redrawn every ``resample_every`` epochs so the network never sees the same set twice.
Training and inference run on the GPU when one is available (CPU fallback). The artifact is an
``.npz`` without pickle: the weights, the feature space, the class list and a JSON manifest pinned
to the ATT&CK version and integrity-checked by a fingerprint, loaded fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apt_classifier.profile_forest import (
    MOTIVATIONS,
    TARGETS,
    FeatureSpace,
    canonical_context,
    group_profiles,
)
from apt_classifier.torch_forest import resolve_device

SCHEMA_VERSION = 2
KIND = "group_mlp"
DEFAULT_PARAMS: dict[str, Any] = {
    # synthetic cases
    "samples_per_group": 400,
    "size_min": 3,
    "size_max": 20,
    "software_rate": 0.7,
    "max_software": 5,
    "software_keep": 0.5,
    "software_named": 0.6,
    "software_false_rate": 0.15,
    "precision_min": 0.40,
    "precision_max": 0.95,
    "context_rate": 0.40,
    "context_noise": 0.05,
    "min_group_techniques": 2,
    # network
    "hidden": [1024, 512],
    "dropout": 0.3,
    "input_dropout": 0.15,
    "members": 5,
    "real_weight": 10,  # repeats of each real case per draw (only with ``real_cases``)
    # optimisation
    "epochs": 30,
    "resample_every": 3,
    "batch_size": 512,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    # prediction: logits + prior_power * log(prominence + 1), a knowledge-base base rate
    "prior_power": 2.0,
    "seed": 7,
}


# ---------------------------------------------------------------------------- synthetic cases
def _parents(kb: Any, techniques: Iterable[str]) -> set[str]:
    out = set(techniques)
    for technique in list(out):
        item = (getattr(kb, "techniques", {}) or {}).get(technique)
        parent = getattr(item, "parent_id", None) if item is not None else None
        if parent:
            out.add(parent)
    return out


def group_tools(kb: Any) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    """Per group, ``(software id, technique set with parents)`` for every documented software."""
    software_techniques = getattr(kb, "software_techniques", {}) or {}
    return {
        group_id: [
            (software_id, tuple(sorted(_parents(kb, software_techniques.get(software_id, ())))))
            for software_id in sorted(software_ids)
        ]
        for group_id, software_ids in (getattr(kb, "group_software", {}) or {}).items()
    }


@dataclass(frozen=True)
class MLPFeatures:
    """``FeatureSpace`` block followed by one 0/1 entry per ATT&CK software."""

    space: FeatureSpace
    software: tuple[str, ...]

    @classmethod
    def from_kb(cls, kb: Any) -> MLPFeatures:
        return cls(
            space=FeatureSpace.from_kb(kb),
            software=tuple(sorted((getattr(kb, "software", {}) or {}).keys())),
        )

    @property
    def size(self) -> int:
        return self.space.size + len(self.software)

    def vector(
        self,
        techniques: Iterable[str],
        software: Iterable[str] = (),
        motivations: Iterable[str] = (),
        targets: Iterable[str] = (),
    ) -> np.ndarray:
        block = np.zeros(len(self.software), dtype=np.uint8)
        index = {software_id: i for i, software_id in enumerate(self.software)}
        block[[index[s] for s in software if s in index]] = 1
        return np.concatenate([self.space.vector(techniques, motivations, targets), block])

    def to_dict(self) -> dict[str, Any]:
        return {"space": self.space.to_dict(), "software": list(self.software)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MLPFeatures:
        return cls(
            space=FeatureSpace.from_dict(payload["space"]),
            software=tuple(str(s) for s in payload["software"]),
        )


def prominence_log_prior(kb: Any, group_ids: Sequence[str]) -> np.ndarray:
    """Centred ``log(prominence + 1)`` per group: ATT&CK references, aliases, campaigns and
    software. Better-documented groups are the ones reports are written about more often."""
    prominence = getattr(kb, "group_prominence", {}) or {}
    values = np.log(np.asarray([float(prominence.get(g, 0)) + 1.0 for g in group_ids]))
    return (values - values.mean()).astype(np.float32)


def training_cases(
    kb: Any, features: MLPFeatures, params: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Synthetic report-like observations of every group (seeded): ``(features, group ids)``.

    The number of true techniques per case is drawn from ``size_min``-``size_max`` (capped by
    what the group documents), not as a share of the profile: a share would make the count of
    active features reveal how well documented the group is, which real reports do not."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(int(p["seed"]))
    space = features.space
    profiles = group_profiles(kb)
    tools_of = group_tools(kb)
    software_index = {s: space.size + i for i, s in enumerate(features.software)}
    software_popularity = np.zeros(len(features.software), dtype=np.float64)
    for tools in tools_of.values():
        for software_id, _ in tools:
            if software_id in software_index:
                software_popularity[software_index[software_id] - space.size] += 1
    index = {technique: position for position, technique in enumerate(space.techniques)}
    n_techniques = len(space.techniques)
    parent_of = np.full(n_techniques, -1, dtype=np.int64)
    for technique, parent in space.parents.items():
        if technique in index and parent in index:
            parent_of[index[technique]] = index[parent]
    popularity = np.zeros(n_techniques, dtype=np.float64)
    for profile in profiles.values():
        for technique in profile.techniques:
            if technique in index:
                popularity[index[technique]] += 1
    motivation_base = n_techniques
    target_base = motivation_base + len(MOTIVATIONS) + 1
    eligible = [
        g
        for g in sorted(profiles)
        if sum(1 for t in profiles[g].techniques if t in index) >= int(p["min_group_techniques"])
    ]
    if not eligible:
        raise ValueError("no group has enough documented techniques to train on")
    per_group = int(p["samples_per_group"])
    matrix = np.zeros((len(eligible) * per_group, features.size), dtype=np.uint8)
    groups: list[str] = []
    row = 0
    for group_id in eligible:
        profile = profiles[group_id]
        own = np.asarray(sorted(index[t] for t in profile.techniques if t in index))
        tools = [
            (software_index[s], np.asarray([index[t] for t in ts if t in index], dtype=np.int64))
            for s, ts in tools_of.get(group_id, ())
            if s in software_index
        ]
        other = software_popularity.copy()
        other[[column - space.size for column, _ in tools]] = 0.0
        other_cumulative = np.cumsum(other)
        weights = popularity.copy()
        weights[own] = 0.0
        cumulative = np.cumsum(weights)
        motivations = [MOTIVATIONS.index(m) for m in sorted(profile.motivations)]
        targets = [TARGETS.index(t) for t in sorted(profile.targets)]
        for _ in range(per_group):
            size = int(rng.integers(int(p["size_min"]), int(p["size_max"]) + 1))
            pool = own
            if tools and rng.uniform() < p["software_rate"]:
                count = int(rng.integers(1, min(int(p["max_software"]), len(tools)) + 1))
                chosen = [tools[i] for i in rng.choice(len(tools), size=count, replace=False)]
                pool = np.union1d(own, np.concatenate([ts for _, ts in chosen]))
                named = [column for column, _ in chosen if rng.uniform() < p["software_named"]]
                matrix[row, named] = 1
            if other_cumulative[-1] > 0 and rng.uniform() < p["software_false_rate"]:
                draw = np.searchsorted(
                    other_cumulative, rng.uniform(0, other_cumulative[-1]), "right"
                )
                matrix[row, space.size + min(int(draw), len(features.software) - 1)] = 1
            kept = rng.choice(pool, size=min(size, pool.size), replace=False)
            precision = rng.uniform(p["precision_min"], p["precision_max"])
            false_positives = int(round(kept.size * (1 - precision) / precision))
            if false_positives and cumulative[-1] > 0:
                draws = rng.uniform(0, cumulative[-1], size=false_positives)
                kept = np.concatenate([kept, np.searchsorted(cumulative, draws, side="right")])
            kept = kept[kept < n_techniques]
            matrix[row, kept] = 1
            parents = parent_of[kept]
            matrix[row, parents[parents >= 0]] = 1
            given = []
            if motivations and rng.uniform() < p["context_rate"]:
                given = motivations
                if rng.uniform() < p["context_noise"]:
                    given = [int(rng.integers(len(MOTIVATIONS)))]
            matrix[row, [motivation_base + m for m in given] or [target_base - 1]] = 1
            given = []
            if targets and rng.uniform() < p["context_rate"]:
                given = [targets[int(rng.integers(len(targets)))]]
                if rng.uniform() < p["context_noise"]:
                    given = [int(rng.integers(len(TARGETS)))]
            matrix[row, [target_base + t for t in given] or [space.size - 1]] = 1
            groups.append(group_id)
            row += 1
    return matrix, groups


# ---------------------------------------------------------------------------- network
def _build_network(
    n_features: int, n_classes: int, hidden: Sequence[int], dropout: float, input_dropout: float
) -> Any:
    import torch.nn as nn

    # dropout layers are always present (p=0 at inference) so the state-dict keys never shift
    layers: list[Any] = [nn.Dropout(input_dropout)]
    width = n_features
    for size in hidden:
        layers += [
            nn.Linear(width, int(size)),
            nn.LayerNorm(int(size)),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        width = int(size)
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


class GroupMLP:
    """The fitted network with its feature space and group list; predictions run on ``device``."""

    def __init__(
        self,
        *,
        features: MLPFeatures,
        group_ids: tuple[str, ...],
        weights: Mapping[str, np.ndarray],
        hidden: Sequence[int],
        log_prior: np.ndarray | None = None,
        prior_power: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.features = features
        self.space = features.space
        self.group_ids = tuple(group_ids)
        self.weights = {
            name: np.asarray(value, dtype=np.float32) for name, value in weights.items()
        }
        self.hidden = tuple(int(h) for h in hidden)
        self.log_prior = (
            np.zeros(len(self.group_ids), np.float32)
            if log_prior is None
            else np.asarray(log_prior, dtype=np.float32)
        )
        if self.log_prior.shape != (len(self.group_ids),):
            raise ValueError("prior does not match the group list")
        self.prior_power = float(prior_power)
        self.metadata = dict(metadata or {})
        self.device = resolve_device(device)
        self.members = sorted({name.split(".", 1)[0] for name in self.weights})
        self._networks: list[Any] | None = None

    def _nets(self) -> list[Any]:
        if self._networks is None:
            import torch

            networks = []
            for member in self.members:
                network = _build_network(
                    self.features.size, len(self.group_ids), self.hidden, 0.0, 0.0
                )
                prefix = f"{member}."
                network.load_state_dict(
                    {
                        name[len(prefix) :]: torch.from_numpy(value)
                        for name, value in self.weights.items()
                        if name.startswith(prefix)
                    }
                )
                networks.append(network.to(self.device).eval())
            self._networks = networks
        return self._networks

    def logits(self, vectors: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Member-averaged network logits (no prior), one row per input vector."""
        import torch

        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[1] != self.features.size:
            raise ValueError(f"expected {self.features.size} features, got {matrix.shape[1]}")
        out = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], batch_size):
                chunk = torch.from_numpy(matrix[start : start + batch_size]).to(self.device)
                out.append(torch.stack([net(chunk) for net in self._nets()]).mean(0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.group_ids)), np.float32)

    def predict(self, vectors: np.ndarray, prior_power: float | None = None) -> np.ndarray:
        """Group probabilities (softmax of logits plus the weighted prior), one row per vector."""
        power = self.prior_power if prior_power is None else float(prior_power)
        scores = self.logits(vectors) + power * self.log_prior
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def group_probabilities(self, vector: np.ndarray) -> dict[str, float]:
        return {
            group_id: float(value)
            for group_id, value in zip(self.group_ids, self.predict(vector)[0], strict=True)
        }

    def context_vector(self, context: Any) -> np.ndarray:
        """Feature row for a classifier context: accepted techniques, named software, context."""
        motivations, targets = canonical_context(getattr(context, "strategic_context", None))
        return self.features.vector(
            getattr(context, "observed_techniques", {}) or {},
            getattr(context, "software_ids", ()) or (),
            motivations,
            targets,
        )

    def predict_context(self, context: Any) -> dict[str, float]:
        """Group probabilities for a classifier context."""
        return self.group_probabilities(self.context_vector(context))

    # ------------------------------------------------------------------ persistence
    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(self.weights):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(self.weights[name]).tobytes())
        digest.update(np.ascontiguousarray(self.log_prior).tobytes())
        digest.update(
            json.dumps(
                {
                    "features": self.features.to_dict(),
                    "groups": list(self.group_ids),
                    "hidden": list(self.hidden),
                    "prior_power": self.prior_power,
                    "schema": SCHEMA_VERSION,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()[:24]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "features": self.features.to_dict(),
            "group_ids": list(self.group_ids),
            "hidden": list(self.hidden),
            "weight_names": sorted(self.weights),
            "prior_power": self.prior_power,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }
        arrays = {f"w{i}": self.weights[name] for i, name in enumerate(sorted(self.weights))}
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
            log_prior=self.log_prior,
            **arrays,
        )
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> GroupMLP | None:
        source = Path(path)
        if not source.is_file():
            return None
        with np.load(source, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            if manifest.get("kind") != KIND:
                raise ValueError(f"not a group MLP artifact: {manifest.get('kind')!r}")
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema {manifest.get('schema_version')!r}")
            weights = {name: data[f"w{i}"] for i, name in enumerate(manifest["weight_names"])}
            log_prior = np.asarray(data["log_prior"])
        model = cls(
            features=MLPFeatures.from_dict(manifest["features"]),
            group_ids=tuple(manifest["group_ids"]),
            weights=weights,
            hidden=manifest["hidden"],
            log_prior=log_prior,
            prior_power=float(manifest.get("prior_power", 0.0)),
            metadata=manifest.get("metadata") or {},
            device=device,
        )
        if manifest.get("fingerprint") != model.fingerprint:
            raise ValueError("fingerprint mismatch (artifact modified or corrupted)")
        return model

    def compatible_with(self, kb: Any) -> tuple[bool, str]:
        version = str(getattr(kb, "version", "") or "")
        fitted = str(self.metadata.get("attack_version") or "")
        if fitted and version and fitted != version:
            return False, f"fitted on ATT&CK {fitted}, runtime has {version} (refit: make fit-mlp)"
        return True, "ok"


def _draw(
    kb: Any,
    features: MLPFeatures,
    p: Mapping[str, Any],
    seed: int,
    real_cases: tuple[np.ndarray, Sequence[str]] | None,
) -> tuple[np.ndarray, list[str]]:
    """One synthetic draw, plus the real cases repeated ``real_weight`` times when given."""
    matrix, groups = training_cases(kb, features, {**p, "seed": seed})
    repeat = int(p["real_weight"])
    if real_cases is not None and repeat > 0 and len(real_cases[1]):
        matrix = np.concatenate([matrix, np.tile(real_cases[0].astype(np.uint8), (repeat, 1))])
        groups = groups + list(real_cases[1]) * repeat
    return matrix, groups


def _fit_member(
    kb: Any,
    features: MLPFeatures,
    p: Mapping[str, Any],
    seed: int,
    device: str,
    log: Any,
    real_cases: tuple[np.ndarray, Sequence[str]] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, float]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    started = time.time()
    matrix, groups = _draw(kb, features, p, seed, real_cases)
    cases_seconds = time.time() - started
    group_ids = tuple(sorted(set(groups)))
    group_index = {group_id: position for position, group_id in enumerate(group_ids)}
    network = _build_network(
        features.size, len(group_ids), p["hidden"], float(p["dropout"]), float(p["input_dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
    )
    epochs = int(p["epochs"])
    batch = int(p["batch_size"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(p["lr"]), total_steps=epochs * -(-matrix.shape[0] // batch)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = y = None
    train_started = time.time()
    for epoch in range(epochs):
        if epoch and int(p["resample_every"]) and epoch % int(p["resample_every"]) == 0:
            t0 = time.time()
            matrix, groups = _draw(kb, features, p, seed + epoch, real_cases)
            cases_seconds += time.time() - t0
            x = None
        if x is None:
            x = torch.from_numpy(matrix).to(device, dtype=torch.float32)
            y = torch.as_tensor([group_index[g] for g in groups], device=device)
        network.train()
        order = torch.randperm(x.shape[0], generator=generator).to(device)
        total = 0.0
        for start in range(0, x.shape[0], batch):
            chosen = order[start : start + batch]
            loss = F.cross_entropy(
                network(x[chosen]), y[chosen], label_smoothing=float(p["label_smoothing"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
            total += float(loss.detach()) * chosen.numel()
        if log is not None:
            log(f"seed {seed} epoch {epoch + 1}/{epochs} loss {total / x.shape[0]:.4f}")
    weights = {k: v.detach().cpu().numpy() for k, v in network.state_dict().items()}
    seconds = {"cases": cases_seconds, "training": time.time() - train_started}
    return weights, group_ids, seconds


def fit_group_mlp(
    kb: Any,
    *,
    params: Mapping[str, Any] | None = None,
    device: str | None = None,
    log: Any = None,
    real_cases: tuple[np.ndarray, Sequence[str]] | None = None,
) -> GroupMLP:
    """Fit ``members`` networks on synthetic cases from ``kb`` (GPU when available).

    ``real_cases`` (feature rows, true groups) are labelled real reports added to every draw
    ``real_weight`` times. The default model uses none; a model fitted with them is derived from
    that data (CTI-HAL: local evaluation only, never shipped) and the metadata says so."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    resolved = resolve_device(device)
    features = MLPFeatures.from_kb(kb)
    weights: dict[str, np.ndarray] = {}
    group_ids: tuple[str, ...] = ()
    seconds = {"cases": 0.0, "training": 0.0}
    for member in range(max(1, int(p["members"]))):
        member_weights, group_ids, spent = _fit_member(
            kb, features, p, int(p["seed"]) + 1000 * member, resolved, log, real_cases
        )
        weights.update({f"m{member}.{k}": v for k, v in member_weights.items()})
        seconds = {k: seconds[k] + spent[k] for k in seconds}
    return GroupMLP(
        features=features,
        group_ids=group_ids,
        weights=weights,
        hidden=p["hidden"],
        log_prior=prominence_log_prior(kb, group_ids),
        prior_power=float(p["prior_power"]),
        metadata={
            "attack_version": str(getattr(kb, "version", "") or ""),
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "params": p,
            "device": resolved,
            "real_cases": 0 if real_cases is None else len(real_cases[1]),
            "seconds": {k: round(v, 1) for k, v in seconds.items()},
        },
        device=resolved,
    )


# ---------------------------------------------------------------------------- metrics
def classification_metrics(truth: Sequence[str], ranked: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Top-1/top-5/MRR plus macro and micro F1 of the top-1 prediction.

    Macro F1 averages per-class F1 over the classes present in ``truth``; a prediction of any
    other group counts as a false negative for the true class (and a false positive only for a
    predicted class that is itself one of the true classes)."""
    if not truth:
        return {"cases": 0}
    top1 = [row[0] if row else None for row in ranked]
    labels = sorted(set(truth))
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(truth, top1, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, top1, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, top1, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    ranks = [
        (list(row).index(t) + 1) if t in row else None for t, row in zip(truth, ranked, strict=True)
    ]
    n = len(truth)
    return {
        "cases": n,
        "top1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "top5": round(sum(1 for r in ranks if r is not None and r <= 5) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class.values()])), 4),
        # with one label per case, micro F1 over the true classes equals top-1 accuracy
        "per_class": per_class,
    }


def load_group_mlp(
    path: str | Path, kb: Any, device: str | None = None
) -> tuple[GroupMLP | None, str]:
    """Load the artifact fail-closed: ``(model, status)``."""
    try:
        model = GroupMLP.load(path, device=device)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        return None, f"invalid artifact: {error}"
    if model is None:
        return None, f"artifact not found: {path} (run make fit-mlp)"
    ok, reason = model.compatible_with(kb)
    if not ok:
        return None, f"incompatible artifact: {reason}"
    return model, f"active ({model.device})"
