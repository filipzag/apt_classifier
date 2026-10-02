"""Pre-registered selection rule of the dev-only retrieval tuning harness (docs 21.3.15).

``scripts/tune_retrieval.py`` evaluates variants; this module decides between them. Everything
here works on evaluation summaries and per-case counts, so the rule is testable without any
retrieval:

- :data:`KNOBS`: the settings a variant may change, their ``Settings`` field and environment
  name, their type and how a value's distance from the reference is measured (tie-break);
- :func:`load_prereg`: ``docs/preregistration/chroma_exemplars_2026-10-02.json`` read and
  validated against the registry; :func:`prereg_git_problems` refuses a file that is not
  committed or has uncommitted changes and returns every commit that touched it (the oldest is
  the registration), and :func:`amendment_problems` refuses a later commit the file does not
  declare in its ``amendments`` list (each names the commit it amends and why), within
  ``retry.allowed``. So an amendment after a result is always visible in the file and in every
  artifact (which records the history). What git cannot show is a rewritten history (``commit
  --amend`` or a rebase of the registration commit itself); artifacts record the commit they
  ran on, and ``replicate`` refuses a commit other than the sweep's;
- :func:`paired_cluster_bootstrap`: the paired bootstrap of the micro-F1 difference, resampling
  near-duplicate case groups (not cases), averaged over the CV seeds;
- :func:`bar_check`, :func:`tie_break_pick`, :func:`greedy_path`: plan section 6, family by
  family;
- :func:`replicate_step_back`: fresh-seed replication against the reference, stepping back along
  the greedy path until a configuration replicates (worst case the reference itself);
- :func:`env_assignments` / :func:`write_env`: the selected settings, and the pre-registered
  fixed settings pinned beside them, as ``VAR=value`` lines for ``env $(cat T1.env)``, and
  :func:`fit_arguments` for what is not a setting (``--l2``).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_SCHEMA = "tune_retrieval_prereg/1"
# fixed.exemplar_library.fingerprint until it is filled from the sync report: the Chroma library's
# fingerprint is known only once scripts/sync_exemplars_to_chroma.py --verify-only has recomputed
# it from the vectors Chroma returns (tune_retrieval.py prereg-fill writes it; sweep, replicate
# and copy-check refuse a placeholder)
EXEMPLAR_LIBRARY_PLACEHOLDER = "<library_fingerprint of the sync --verify-only report>"
EXEMPLAR_BACKENDS = ("npz", "chroma")


def is_library_fingerprint(value: Any) -> bool:
    """A library fingerprint is 24 lower-case hex digits (``ExemplarIndex.compute_fingerprint``);
    anything else (the placeholder included) is not one."""
    text = str(value or "")
    return len(text) == 24 and all(character in "0123456789abcdef" for character in text)


# comparisons against a bar value tolerate float noise of this size (a delta of exactly 0.010
# computed as 0.00999999999 still passes "at least 0.010")
EPSILON = 1e-12


@dataclass(frozen=True)
class Knob:
    """One tunable value. ``setting`` is the ``Settings`` field (``None`` for a harness or fit
    value), ``env`` its environment variable, ``distance`` how the tie-break measures it."""

    name: str
    kind: str  # int | float | bool | choice
    setting: str | None
    env: str | None
    distance: str  # log | relative | categorical
    choices: tuple[str, ...] = ()

    def cast(self, value: Any) -> Any:
        if self.kind == "int":
            if isinstance(value, bool) or float(value) != int(float(value)):
                raise ValueError(f"{self.name}: {value!r} is not an integer")
            return int(float(value))
        if self.kind == "float":
            if isinstance(value, bool):
                raise ValueError(f"{self.name}: {value!r} is not a number")
            return float(value)
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name}: {value!r} is not true/false")
            return value
        text = str(value)
        if self.choices and text not in self.choices:
            raise ValueError(f"{self.name}: {text!r} is not one of {', '.join(self.choices)}")
        return text


KNOBS: dict[str, Knob] = {
    knob.name: knob
    for knob in (
        Knob("exemplar_k", "int", "technique_exemplar_k", "TECHNIQUE_EXEMPLAR_K", "log"),
        Knob("exemplar_tau", "float", "technique_exemplar_tau", "TECHNIQUE_EXEMPLAR_TAU", "log"),
        Knob(
            "exemplar_new_candidates",
            "int",
            "technique_exemplar_new_candidates",
            "TECHNIQUE_EXEMPLAR_NEW_CANDIDATES",
            "relative",
        ),
        Knob(
            "exemplar_vote_norm",
            "choice",
            "technique_exemplar_vote_norm",
            "TECHNIQUE_EXEMPLAR_VOTE_NORM",
            "categorical",
            ("none", "label_split"),
        ),
        Knob(
            "exemplar_segment_votes",
            "choice",
            "technique_exemplar_segment_votes",
            "TECHNIQUE_EXEMPLAR_SEGMENT_VOTES",
            "categorical",
            ("off", "max", "mean"),
        ),
        Knob(
            "dense_relative_margin",
            "float",
            "technique_dense_relative_margin",
            "TECHNIQUE_DENSE_RELATIVE_MARGIN",
            "log",
        ),
        Knob(
            "dense_overfetch",
            "int",
            "technique_dense_overfetch",
            "TECHNIQUE_DENSE_OVERFETCH",
            "log",
        ),
        Knob(
            "candidate_pool", "int", "technique_candidate_pool", "TECHNIQUE_CANDIDATE_POOL", "log"
        ),
        Knob(
            "include_synthetic",
            "bool",
            "technique_include_synthetic",
            "TECHNIQUE_INCLUDE_SYNTHETIC",
            "categorical",
        ),
        Knob(
            "platform_from_text",
            "bool",
            "technique_platform_from_text",
            "TECHNIQUE_PLATFORM_FROM_TEXT",
            "categorical",
        ),
        # where the dense hits come from: Chroma's ANN index (serving), brute force (what a
        # Phase 5b collection copy with full recall serves) or a configured copy's ANN index
        Knob("dense_search", "choice", None, None, "categorical", ("ann", "exact", "copy")),
        # the acceptance fit's L2 penalty (fit_acceptance_model.py --l2), not a setting
        Knob("l2", "float", None, None, "log"),
    )
}
# knobs that change the candidate rows (l2 only changes the fit)
ROW_KNOBS = tuple(name for name in KNOBS if name != "l2")
# retrieve_techniques' default dense depth is max(3 * top_k, 4 * pool); top_k is fixed at 30
FIXED_TOP_K = 30


def full_config(reference: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict:
    """``reference`` with ``overrides`` applied, every value cast by its knob."""
    config = {name: KNOBS[name].cast(value) for name, value in reference.items()}
    for name, value in dict(overrides or {}).items():
        if name not in KNOBS:
            raise ValueError(f"unknown knob {name!r}")
        config[name] = KNOBS[name].cast(value)
    missing = sorted(set(KNOBS) - set(config))
    if missing:
        raise ValueError(f"configuration lacks {', '.join(missing)}")
    return config


def effective_value(name: str, config: Mapping[str, Any]) -> Any:
    """The value a knob acts with (``dense_overfetch`` 0 is the formula's depth)."""
    value = config[name]
    if name == "dense_overfetch" and int(value) <= 0:
        return max(3 * FIXED_TOP_K, 4 * int(config.get("candidate_pool", 12)))
    return value


def changed_knobs(config: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    return sorted(name for name in KNOBS if config.get(name) != reference.get(name))


def distance_to_reference(config: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
    """How far a configuration is from the reference: per changed knob ``|ln(v/d)|`` (log
    knobs, effective values), ``|v - d| / max(d, 1)`` (relative) or 1 (categorical)."""
    total = 0.0
    for name in changed_knobs(config, reference):
        knob = KNOBS[name]
        value, default = effective_value(name, config), effective_value(name, reference)
        if knob.distance == "log" and float(value) > 0 and float(default) > 0:
            total += abs(math.log(float(value) / float(default)))
        elif knob.distance in ("log", "relative"):
            total += abs(float(value) - float(default)) / max(abs(float(default)), 1.0)
        else:
            total += 1.0
    return total


# ----- pre-registration -----
@dataclass(frozen=True)
class Variant:
    id: str
    family: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Family:
    id: str
    group: str
    description: str
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Bar:
    min_delta_f1: float = 0.010
    ci_lower_above: float = 0.0
    min_precision: float = 0.50
    dataset_max_drop: dict[str, float] = field(
        default_factory=lambda: {"tram": 0.02, "derived_procedures": 0.02}
    )
    flag_dataset: str = "expert"
    flag_drop_above: float = 0.05
    min_delta_f1_no_cues: float = 0.0
    min_delta_hit_at_10: float = -0.01
    max_generic_share_rise: float = 0.02
    max_accepted_per_case_rise: float = 0.25


@dataclass(frozen=True)
class Replication:
    seeds: tuple[int, ...] = (41, 53)
    min_delta_f1_per_seed: float = 0.005
    pooled_ci_lower_above: float = 0.0


@dataclass(frozen=True)
class Prereg:
    path: Path
    sha256: str
    payload: dict[str, Any]
    reference_id: str
    reference: dict[str, Any]
    families: tuple[Family, ...]
    bar: Bar
    tie_window: float
    selection_seeds: tuple[int, ...]
    replication: Replication
    bootstrap_samples: int
    bootstrap_seed: int
    folds: int
    group_similarity: float
    nested_inner_folds: int
    min_precision: float
    p40_min_precision: float
    generic_min_groups: int
    fixed_settings: dict[str, Any]
    decomposition: tuple[dict[str, Any], ...]
    # the case set the selection runs on (dev_only.datasets, dev_only.cases)
    dev_datasets: tuple[str, ...]
    dev_cases: int
    # fixed.exemplar_library: fingerprint, count and split of the only library allowed, and
    # (when the pre-registration names them) its backend and the npz it was copied from
    exemplar_library: dict[str, Any]
    # declared amendments (each {"amends": <commit>, "reason": ...}) and how many are allowed
    amendments: tuple[dict[str, Any], ...]
    retry_allowed: int
    # fixed.exemplar_library.fingerprint is still the placeholder (not a library fingerprint)
    exemplar_library_placeholder: bool = False

    @property
    def variant_count(self) -> int:
        return sum(len(family.variants) for family in self.families)


class PreregError(ValueError):
    pass


def _number(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise PreregError(f"pre-registration lacks {'.'.join(keys)}")
        value = value[key]
    return value


def _registered_tolerance(value: Any, fixed_settings: Mapping[str, Any]) -> float:
    """``fixed.exemplar_library.tolerance``: a finite number >= 0 that ``fixed.settings`` pins
    as ``technique_exemplar_tolerance`` (so every selection step checks it and T1.env sets it;
    a registered number nothing enforces is refused)."""
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as error:
        raise PreregError(f"fixed.exemplar_library.tolerance {value!r} is not a number") from error
    if isinstance(value, bool) or not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise PreregError(f"fixed.exemplar_library.tolerance {value!r} is not a finite number >= 0")
    pinned = fixed_settings.get("technique_exemplar_tolerance")
    if pinned is None:
        raise PreregError(
            "fixed.settings must pin technique_exemplar_tolerance to fixed.exemplar_library."
            f"tolerance ({tolerance:g}), or nothing enforces it"
        )
    try:
        same = float(pinned) == tolerance and not isinstance(pinned, bool)
    except (TypeError, ValueError):
        same = False
    if not same:
        raise PreregError(
            f"fixed.settings.technique_exemplar_tolerance {pinned!r} != fixed.exemplar_library."
            f"tolerance {tolerance:g}"
        )
    return tolerance


def load_prereg(path: str | Path) -> Prereg:
    """Read and validate the pre-registration (raises :class:`PreregError`)."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreregError(f"{source}: not JSON ({error})") from error
    if payload.get("schema") != PREREG_SCHEMA:
        raise PreregError(f"{source}: schema {payload.get('schema')!r} != {PREREG_SCHEMA!r}")
    declared = payload.get("knobs") or {}
    unknown = sorted(set(declared) - set(KNOBS))
    if unknown:
        raise PreregError(f"knobs the harness does not know: {', '.join(unknown)}")
    for name, spec in declared.items():
        if isinstance(spec, Mapping) and spec.get("env") != KNOBS[name].env:
            raise PreregError(f"knob {name}: env {spec.get('env')!r} != {KNOBS[name].env!r}")
    try:
        reference = full_config(_number(payload, "reference", "config"))
    except ValueError as error:
        raise PreregError(f"reference: {error}") from error
    families: list[Family] = []
    seen: set[str] = set()
    for entry in _number(payload, "families"):
        variants = []
        for item in entry.get("variants") or ():
            variant_id = str(item["id"])
            if variant_id in seen:
                raise PreregError(f"variant id {variant_id} is used twice")
            seen.add(variant_id)
            overrides = dict(item.get("set") or {})
            if not overrides:
                raise PreregError(f"variant {variant_id} changes nothing")
            try:
                full_config(reference, overrides)
            except ValueError as error:
                raise PreregError(f"variant {variant_id}: {error}") from error
            if full_config(reference, overrides) == reference:
                raise PreregError(f"variant {variant_id} equals the reference")
            variants.append(Variant(variant_id, str(entry["id"]), overrides))
        if not variants:
            raise PreregError(f"family {entry.get('id')} has no variants")
        families.append(
            Family(
                str(entry["id"]),
                str(entry.get("group") or entry["id"]),
                str(entry.get("description") or ""),
                tuple(variants),
            )
        )
    declared_count = int(_number(payload, "variant_count"))
    counted = sum(len(family.variants) for family in families)
    if declared_count != counted:
        raise PreregError(f"variant_count {declared_count} but {counted} variants are listed")
    bar_spec = _number(payload, "selection", "bar")
    expert = bar_spec.get("flag_only") or {}
    bar = Bar(
        min_delta_f1=float(_number(bar_spec, "min_delta_f1")),
        ci_lower_above=float(_number(bar_spec, "bootstrap", "ci_lower_above")),
        min_precision=float(_number(bar_spec, "min_precision")),
        dataset_max_drop={
            str(key): float(value)
            for key, value in dict(_number(bar_spec, "dataset_max_f1_drop")).items()
        },
        flag_dataset=str(expert.get("dataset", "expert")),
        flag_drop_above=float(expert.get("flag_f1_drop_above", 0.05)),
        min_delta_f1_no_cues=float(_number(bar_spec, "min_delta_f1_no_cues")),
        min_delta_hit_at_10=float(_number(bar_spec, "min_delta_hit_at_10")),
        max_generic_share_rise=float(_number(bar_spec, "max_generic_share_rise")),
        max_accepted_per_case_rise=float(_number(bar_spec, "max_accepted_per_case_rise_relative")),
    )
    replication_spec = _number(payload, "replication")
    cv = _number(payload, "cv")
    dev_datasets = _number(payload, "dev_only", "datasets")
    if not isinstance(dev_datasets, list) or not dev_datasets:
        raise PreregError("dev_only.datasets must be a non-empty list")
    library = dict(_number(payload, "fixed", "exemplar_library"))
    for key in ("fingerprint", "count", "split"):
        if library.get(key) in (None, ""):
            raise PreregError(f"pre-registration lacks fixed.exemplar_library.{key}")
    fixed_settings = dict(_number(payload, "fixed", "settings"))
    exemplar_library: dict[str, Any] = {
        "fingerprint": str(library["fingerprint"]),
        "count": int(library["count"]),
        "split": str(library["split"]),
    }
    if library.get("backend") not in (None, ""):
        if str(library["backend"]) not in EXEMPLAR_BACKENDS:
            raise PreregError(
                f"fixed.exemplar_library.backend {library['backend']!r} is not one of "
                f"{', '.join(EXEMPLAR_BACKENDS)}"
            )
        exemplar_library["backend"] = str(library["backend"])
    if library.get("source_fingerprint") not in (None, ""):
        if not is_library_fingerprint(library["source_fingerprint"]):
            raise PreregError("fixed.exemplar_library.source_fingerprint is not a fingerprint")
        exemplar_library["source_fingerprint"] = str(library["source_fingerprint"])
    if library.get("tolerance") not in (None, ""):
        exemplar_library["tolerance"] = _registered_tolerance(library["tolerance"], fixed_settings)
    amendments = payload.get("amendments") or []
    if not isinstance(amendments, list):
        raise PreregError("amendments must be a list")
    for number, entry in enumerate(amendments):
        if (
            not isinstance(entry, Mapping)
            or not str(entry.get("amends") or "").strip()
            or not str(entry.get("reason") or "").strip()
        ):
            raise PreregError(f"amendment {number + 1} needs 'amends' (a commit) and 'reason'")
    retry = payload.get("retry") or {}
    return Prereg(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
        reference_id=str(_number(payload, "reference", "id")),
        reference=reference,
        families=tuple(families),
        bar=bar,
        tie_window=float(_number(payload, "selection", "tie_break", "f1_window")),
        selection_seeds=tuple(int(seed) for seed in _number(cv, "selection_seeds")),
        replication=Replication(
            seeds=tuple(int(seed) for seed in _number(replication_spec, "seeds")),
            min_delta_f1_per_seed=float(_number(replication_spec, "min_delta_f1_per_seed")),
            pooled_ci_lower_above=float(
                _number(replication_spec, "pooled_bootstrap_ci_lower_above")
            ),
        ),
        bootstrap_samples=int(_number(bar_spec, "bootstrap", "samples")),
        bootstrap_seed=int(_number(bar_spec, "bootstrap", "seed")),
        folds=int(_number(cv, "folds")),
        group_similarity=float(_number(cv, "group_similarity")),
        nested_inner_folds=int(_number(cv, "nested_inner_folds")),
        min_precision=float(_number(payload, "operating_points", "T1", "min_precision")),
        p40_min_precision=float(_number(payload, "operating_points", "T1-P40", "min_precision")),
        generic_min_groups=int(_number(bar_spec, "generic_min_groups")),
        fixed_settings=fixed_settings,
        decomposition=tuple(dict(item) for item in _number(payload, "e0", "decomposition")),
        dev_datasets=tuple(str(item) for item in dev_datasets),
        dev_cases=int(_number(payload, "dev_only", "cases")),
        exemplar_library=exemplar_library,
        amendments=tuple(dict(entry) for entry in amendments),
        retry_allowed=int(retry.get("allowed", 0) or 0) if isinstance(retry, Mapping) else 0,
        exemplar_library_placeholder=not is_library_fingerprint(library["fingerprint"]),
    )


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None


def prereg_git_problems(
    path: str | Path, *, repo_root: Path = REPO_ROOT
) -> tuple[list[str], dict[str, Any]]:
    """Why the pre-registration may not be used (empty when it is committed and clean), and its
    git identity: repository-relative path, the last commit touching it (``commit``), the first
    (``registered_commit``) and every one, oldest first (``history``). Whether the later
    commits are declared amendments is :func:`amendment_problems`' question."""
    root = Path(repo_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return [f"{path} does not exist"], {}
    if not resolved.is_relative_to(root):
        return [f"{path} is outside the repository {root}"], {}
    relative = str(resolved.relative_to(root))
    problems: list[str] = []
    tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
    if tracked is None:
        return [f"git is unavailable; cannot check {relative}"], {"path": relative}
    if tracked.returncode != 0:
        problems.append(f"{relative} is not committed (untracked); commit it before the sweep")
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", relative)
    if status is None or status.returncode != 0:
        problems.append(f"git status failed for {relative}")
    elif status.stdout.strip():
        problems.append(
            f"{relative} has uncommitted changes ({status.stdout.decode().strip()}); "
            "the pre-registration must be the committed version"
        )
    log = _git(root, "log", "--format=%H", "--", relative)
    newest_first = log.stdout.decode().split() if log is not None and log.returncode == 0 else []
    if not newest_first and not problems:
        problems.append(f"{relative} has no commit")
    history = list(reversed(newest_first))
    return problems, {
        "path": relative,
        "commit": history[-1] if history else None,
        "registered_commit": history[0] if history else None,
        "history": history,
    }


def amendment_problems(
    history: Sequence[str], amendments: Sequence[Mapping[str, Any]], *, allowed: int
) -> list[str]:
    """Commits after the registration commit that the file does not declare. ``history`` is
    oldest first; the file's ``amendments`` must name, in order, every commit before the current
    one (``amends``: the full hash or a prefix of at least 7 characters), each with a reason,
    and there may be no more of them than ``retry.allowed``."""
    earlier = list(history[:-1])
    problems: list[str] = []
    if len(amendments) != len(earlier):
        problems.append(
            f"the pre-registration was committed {len(history)} times (registered in "
            f"{history[0][:12] if history else '?'}) but declares {len(amendments)} amendment(s); "
            "every commit after the registration must be declared in 'amendments' "
            "({'amends': <the commit it replaces>, 'reason': ...}) so a change after a result "
            "stays visible"
        )
    for number, (commit, entry) in enumerate(zip(earlier, amendments, strict=False)):
        named = str(entry.get("amends") or "").strip().lower()
        if len(named) < 7 or not commit.lower().startswith(named):
            problems.append(
                f"amendment {number + 1} names {named or '(nothing)'}, but the commit it "
                f"replaces is {commit}"
            )
    if len(amendments) > int(allowed):
        problems.append(
            f"{len(amendments)} amendment(s) but retry.allowed is {int(allowed)} for the whole "
            "programme"
        )
    return problems


# ----- per-case counts and the paired cluster bootstrap -----
@dataclass
class CaseCounts:
    """Accepted-set counts per CV seed and case: ``tp``/``fp`` of shape (seeds, cases) and the
    expected label count per case (labels never retrieved included)."""

    tp: np.ndarray
    fp: np.ndarray
    expected: np.ndarray

    def __post_init__(self) -> None:
        self.tp = np.atleast_2d(np.asarray(self.tp, dtype=np.int64))
        self.fp = np.atleast_2d(np.asarray(self.fp, dtype=np.int64))
        self.expected = np.asarray(self.expected, dtype=np.int64)
        if self.tp.shape != self.fp.shape or self.tp.shape[1] != len(self.expected):
            raise ValueError("case counts are misaligned")


def micro_f1(tp: np.ndarray, fp: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Micro-F1 ``2TP / (TP + FP + expected)`` (= 2PR/(P+R)), 0 without a true positive;
    reduces the last axis."""
    tp_sum = np.asarray(tp, dtype=np.float64).sum(axis=-1)
    denominator = (
        tp_sum
        + np.asarray(fp, dtype=np.float64).sum(axis=-1)
        + np.asarray(expected, dtype=np.float64).sum(axis=-1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where((tp_sum > 0) & (denominator > 0), 2.0 * tp_sum / denominator, 0.0)
    return value


def paired_cluster_bootstrap(
    base: CaseCounts,
    current: CaseCounts,
    clusters: Sequence[int],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap of ``F1(current) - F1(base)`` (mean over the CV seeds) resampling whole
    near-duplicate groups with replacement; both arms see the same resamples. The interval uses
    the sorted-estimate convention of ``apt_classifier.evaluation.bootstrap_interval``."""
    if base.tp.shape != current.tp.shape:
        raise ValueError("the two arms were not scored on the same seeds and cases")
    if not np.array_equal(base.expected, current.expected):
        raise ValueError("the two arms have different expected labels")
    labels = np.asarray(clusters)
    if len(labels) != base.tp.shape[1]:
        raise ValueError("one cluster label per case is needed")
    _, cluster_index = np.unique(labels, return_inverse=True)
    groups = int(cluster_index.max()) + 1 if len(cluster_index) else 0
    seeds = base.tp.shape[0]

    def per_cluster(values: np.ndarray) -> np.ndarray:
        output = np.zeros((values.shape[0], groups), dtype=np.float64)
        for row in range(values.shape[0]):
            np.add.at(output[row], cluster_index, values[row])
        return output

    expected = per_cluster(base.expected[None, :])[0]
    arms = {
        name: (per_cluster(counts.tp), per_cluster(counts.fp))
        for name, counts in (("base", base), ("current", current))
    }
    point = float(
        np.mean(
            micro_f1(current.tp, current.fp, current.expected[None, :])
            - micro_f1(base.tp, base.fp, base.expected[None, :])
        )
    )
    rng = np.random.default_rng(int(seed))
    weights = np.zeros((int(samples), groups), dtype=np.float64)
    draws = rng.integers(0, groups, size=(int(samples), groups))
    np.add.at(weights, (np.repeat(np.arange(int(samples)), groups), draws.ravel()), 1.0)
    expected_sum = weights @ expected  # (samples,)

    def f1(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        tp_sum = weights @ tp.T  # (samples, seeds)
        denominator = tp_sum + weights @ fp.T + expected_sum[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tp_sum > 0, 2.0 * tp_sum / denominator, 0.0)

    deltas = np.mean(f1(*arms["current"]) - f1(*arms["base"]), axis=1)
    estimates = np.sort(deltas)
    tail = (1.0 - confidence) / 2.0
    lower = float(estimates[min(int(tail * samples), samples - 1)])
    upper = float(estimates[min(int((1.0 - tail) * samples), samples - 1)])
    return {
        "delta_f1": point,
        "ci95": [lower, upper],
        "samples": int(samples),
        "seed": int(seed),
        "clusters": groups,
        "seeds_averaged": seeds,
        "share_positive": float(np.mean(deltas > 0)),
    }


# ----- the bar, the tie-break and the greedy path -----
def _check(value: float | None, passed: bool, required: str) -> dict[str, Any]:
    return {"value": value, "required": required, "passed": bool(passed)}


def bar_check(
    candidate: Mapping[str, Any],
    best: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    bar: Bar,
) -> dict[str, Any]:
    """Plan section 6 criteria 1-7 for ``candidate`` against the current ``best`` (summaries as
    :func:`scripts.tune_retrieval.summarize` writes them)."""
    checks: dict[str, Any] = {}
    delta = float(candidate["f1"]) - float(best["f1"])
    checks["delta_f1"] = _check(
        delta, delta >= bar.min_delta_f1 - EPSILON, f">= {bar.min_delta_f1:+.3f}"
    )
    lower = float(bootstrap["ci95"][0])
    checks["bootstrap_ci_lower"] = _check(
        lower, lower > bar.ci_lower_above, f"> {bar.ci_lower_above:.3f}"
    )
    precision = float(candidate["precision"])
    checks["precision_floor"] = _check(
        precision,
        precision >= bar.min_precision - EPSILON and bool(candidate.get("precision_floor_met")),
        f">= {bar.min_precision:.2f} (and met on every seed)",
    )
    for dataset, max_drop in sorted(bar.dataset_max_drop.items()):
        if dataset not in candidate.get("dataset_f1", {}) or dataset not in best.get(
            "dataset_f1", {}
        ):
            # criterion 4 cannot be evaluated: never a silent pass
            checks[f"{dataset}_f1_drop"] = _check(
                None, False, f">= {-max_drop:+.3f} (no {dataset} cases were scored)"
            )
            continue
        change = float(candidate["dataset_f1"][dataset]) - float(best["dataset_f1"][dataset])
        checks[f"{dataset}_f1_drop"] = _check(
            change, change >= -max_drop - EPSILON, f">= {-max_drop:+.3f}"
        )
    no_cues = float(candidate["f1_no_cues"]) - float(best["f1_no_cues"])
    checks["delta_f1_no_cues"] = _check(
        no_cues,
        no_cues >= bar.min_delta_f1_no_cues - EPSILON,
        f">= {bar.min_delta_f1_no_cues:+.3f}",
    )
    hit = float(candidate["hit_at_10"]) - float(best["hit_at_10"])
    checks["delta_hit_at_10"] = _check(
        hit, hit >= bar.min_delta_hit_at_10 - EPSILON, f">= {bar.min_delta_hit_at_10:+.3f}"
    )
    generic = float(candidate["generic_share"]) - float(best["generic_share"])
    checks["generic_share_rise"] = _check(
        generic,
        generic <= bar.max_generic_share_rise + EPSILON,
        f"<= {bar.max_generic_share_rise:+.3f}",
    )
    base_accepted = float(best["accepted_per_case"])
    ratio = float(candidate["accepted_per_case"]) / base_accepted if base_accepted else math.inf
    checks["accepted_per_case_rise"] = _check(
        ratio - 1.0,
        float(candidate["accepted_per_case"])
        <= base_accepted * (1.0 + bar.max_accepted_per_case_rise) + EPSILON,
        f"<= {bar.max_accepted_per_case_rise:+.0%}",
    )
    flags: list[str] = []
    flagged = bar.flag_dataset
    if flagged in candidate.get("dataset_f1", {}) and flagged in best.get("dataset_f1", {}):
        drop = float(best["dataset_f1"][flagged]) - float(candidate["dataset_f1"][flagged])
        if drop > bar.flag_drop_above + EPSILON:
            flags.append(
                f"{flagged} F1 drops by {drop:.3f} (> {bar.flag_drop_above:.2f}; never selects)"
            )
    return {
        "passed": all(entry["passed"] for entry in checks.values()),
        "checks": checks,
        "flags": flags,
    }


@dataclass
class Candidate:
    """One evaluated configuration: its summary (means over seeds, :func:`bar_check` keys) and
    per-case counts for the bootstrap."""

    id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    counts: CaseCounts
    extra: dict[str, Any] = field(default_factory=dict)


def tie_break_pick(
    candidates: Sequence[Candidate], reference: Mapping[str, Any], *, window: float
) -> Candidate:
    """The family's best: within ``window`` F1 of the top, fewer changed knobs, then closer to
    the reference, then higher precision, then the pre-registered order."""
    if not candidates:
        raise ValueError("no candidates to choose from")
    top = max(float(candidate.summary["f1"]) for candidate in candidates)
    near = [
        (position, candidate)
        for position, candidate in enumerate(candidates)
        if float(candidate.summary["f1"]) >= top - window - EPSILON
    ]
    return min(
        near,
        key=lambda item: (
            len(changed_knobs(item[1].config, reference)),
            distance_to_reference(item[1].config, reference),
            -float(item[1].summary["precision"]),
            item[0],
        ),
    )[1]


def greedy_path(
    families: Sequence[Family],
    evaluate: Callable[[Variant, dict[str, Any]], Candidate],
    reference: Candidate,
    *,
    bar: Bar,
    window: float,
    clusters: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Family-wise greedy selection: every variant of a family is applied on top of the current
    best, the family's best (tie-break) is tested against the current best with the bar, and
    becomes the current best when it passes. Returns the path (reference first) and one report
    per family."""
    current = reference
    path = [reference]
    reports: list[dict[str, Any]] = []
    for family in families:
        tried: list[Candidate] = []
        skipped: list[str] = []
        for variant in family.variants:
            config = full_config(current.config, variant.overrides)
            if config == current.config:
                skipped.append(variant.id)  # already the current best (an earlier adoption)
                continue
            tried.append(evaluate(variant, config))
        report: dict[str, Any] = {
            "family": family.id,
            "group": family.group,
            "built_on": current.id,
            "evaluated": [candidate.id for candidate in tried],
            "identical_to_current": skipped,
        }
        if tried:
            chosen = tie_break_pick(tried, reference.config, window=window)
            bootstrap = paired_cluster_bootstrap(
                current.counts,
                chosen.counts,
                clusters,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            verdict = bar_check(chosen.summary, current.summary, bootstrap, bar)
            report.update(
                {
                    "chosen": chosen.id,
                    "bootstrap": bootstrap,
                    "bar": verdict,
                    "adopted": verdict["passed"],
                }
            )
            if verdict["passed"]:
                current = chosen
                path.append(chosen)
        else:
            report.update({"chosen": None, "adopted": False})
        reports.append(report)
    return path, reports


def replicate_step_back(
    path: Sequence[Candidate],
    reference: Candidate,
    *,
    rule: Replication,
    clusters: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[Candidate, list[dict[str, Any]]]:
    """Fresh-seed replication. ``path`` holds the greedy path's configurations evaluated on the
    replication seeds (reference first). From the finalist backwards, the first configuration
    whose F1 beats the reference by ``min_delta_f1_per_seed`` on every seed, with the pooled
    paired cluster bootstrap's lower bound above ``pooled_ci_lower_above``, is selected; the
    reference when none does."""
    steps: list[dict[str, Any]] = []
    for candidate in reversed(list(path[1:])):
        per_seed = [
            float(mine["f1"]) - float(theirs["f1"])
            for mine, theirs in zip(
                candidate.summary["per_seed"], reference.summary["per_seed"], strict=True
            )
        ]
        bootstrap = paired_cluster_bootstrap(
            reference.counts,
            candidate.counts,
            clusters,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        seeds_ok = all(delta >= rule.min_delta_f1_per_seed - EPSILON for delta in per_seed)
        ci_ok = float(bootstrap["ci95"][0]) > rule.pooled_ci_lower_above
        steps.append(
            {
                "id": candidate.id,
                "delta_f1_per_seed": per_seed,
                "bootstrap": bootstrap,
                "seeds_passed": seeds_ok,
                "pooled_ci_passed": ci_ok,
                "replicates": seeds_ok and ci_ok,
            }
        )
        if seeds_ok and ci_ok:
            return candidate, steps
    return reference, steps


# ----- T1.env -----
def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if not text or any(character.isspace() for character in text) or "=" in text:
        raise ValueError(f"{text!r} cannot be written as one env token")
    return text


EXEMPLAR_LIBRARY_SETTINGS = (
    "technique_exemplar_backend",
    "technique_exemplar_collection",
    "technique_exemplar_library_fingerprint",
)


def env_assignments(
    config: Mapping[str, Any],
    *,
    technique_collection: str | None = None,
    fixed: Mapping[str, Any] | None = None,
    exemplar_library: Mapping[str, Any] | None = None,
) -> list[str]:
    """``VAR=value`` for every setting knob (explicit even at the default), then every
    pre-registered fixed setting (``fixed``: ``Settings`` field -> value; its variable is the
    upper-cased field name), so a run under ``env $(cat T1.env)`` cannot inherit another value
    from the shell or a ``.env`` file, then the exemplar library the selection ran on
    (``exemplar_library``: its ``backend``, for ``chroma`` its ``collection``, and its
    ``fingerprint`` as ``TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT``, required for ``chroma``: the
    loader then refuses any other vectors, npz or not), plus ``TECHNIQUE_COLLECTION_NAME`` when
    the dense hits must come from a collection copy."""
    lines = [
        f"{KNOBS[name].env}={_env_value(KNOBS[name].cast(config[name]))}"
        for name in KNOBS
        if KNOBS[name].env
    ]
    knob_settings = {knob.setting for knob in KNOBS.values() if knob.setting}
    for name, value in sorted(dict(fixed or {}).items()):
        if name in knob_settings:
            raise ValueError(f"{name} is a tuning knob; it cannot also be a fixed setting")
        if name == "technique_collection_name":
            raise ValueError("technique_collection_name is not a fixed setting")
        if name in EXEMPLAR_LIBRARY_SETTINGS and exemplar_library:
            raise ValueError(f"{name} is pinned from the exemplar library, not a fixed setting")
        lines.append(f"{name.upper()}={_env_value(value)}")
    if exemplar_library:
        backend = str(exemplar_library.get("backend") or "")
        if backend not in EXEMPLAR_BACKENDS:
            raise ValueError(f"unknown exemplar backend {backend!r}")
        lines.append(f"TECHNIQUE_EXEMPLAR_BACKEND={backend}")
        if backend == "chroma":
            collection = str(exemplar_library.get("collection") or "")
            if not collection:
                raise ValueError("a Chroma exemplar library needs its collection name")
            lines.append(f"TECHNIQUE_EXEMPLAR_COLLECTION={_env_value(collection)}")
        fingerprint = str(exemplar_library.get("fingerprint") or "")
        if fingerprint and not is_library_fingerprint(fingerprint):
            raise ValueError(f"exemplar library fingerprint {fingerprint!r} is not a fingerprint")
        if backend == "chroma" and not fingerprint:
            raise ValueError(
                "a Chroma exemplar library needs its fingerprint (the loader checks it only "
                "when pinned)"
            )
        if fingerprint:
            lines.append(f"TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT={fingerprint}")
    if str(config.get("dense_search", "ann")) != "ann":
        if not technique_collection:
            raise ValueError(
                "the selected configuration searches the dense index exactly; it runs only "
                "against a Phase 5b collection copy: build and cache the copy (cache "
                "--technique-collection-copy), then copy-check re-runs the replication on the "
                "copy's hits and writes T1.env"
            )
        lines.append(f"TECHNIQUE_COLLECTION_NAME={_env_value(technique_collection)}")
    return lines


def fit_arguments(config: Mapping[str, Any]) -> list[str]:
    """Arguments ``scripts/fit_acceptance_model.py`` needs beyond the settings."""
    return ["--l2", repr(float(config["l2"]))]


def fill_exemplar_library(path: str | Path, report: Mapping[str, Any]) -> str:
    """Write the Chroma library fingerprint of a passing ``scripts/sync_exemplars_to_chroma.py
    --verify-only`` ``report`` into the placeholder ``fixed.exemplar_library.fingerprint`` of
    the pre-registration at ``path`` (only that string changes; the file stays byte-identical
    otherwise) and return it. Raises :class:`PreregError` unless the file still holds the
    placeholder there, the report passed, and its library has the pre-registered backend,
    source npz, count, split and collection and was verified within the pre-registered
    ``tolerance``: compared with the npz, at a tolerance no looser than the registered one,
    with every norm and component deviation inside it."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    prereg = load_prereg(source)
    if not prereg.exemplar_library_placeholder:
        raise PreregError(
            f"fixed.exemplar_library.fingerprint is {prereg.exemplar_library['fingerprint']!r}, "
            "not the placeholder: nothing to fill (a registered value changes only by amendment)"
        )
    library = report.get("library") if isinstance(report, Mapping) else None
    if not isinstance(library, Mapping) or report.get("ok") is not True:
        raise PreregError("the sync report did not pass (ok) or holds no library block")
    fingerprint = str(library.get("fingerprint") or "")
    if not is_library_fingerprint(fingerprint) or report.get("library_fingerprint") != fingerprint:
        raise PreregError("the sync report names no library fingerprint")
    expected = prereg.exemplar_library
    problems = []
    if expected.get("backend") and library.get("backend") != expected["backend"]:
        problems.append(
            f"backend {library.get('backend')!r}, pre-registered {expected['backend']!r}"
        )
    if expected.get("source_fingerprint") and (
        library.get("source_fingerprint") != expected["source_fingerprint"]
    ):
        problems.append(
            f"copied from {library.get('source_fingerprint')!r}, pre-registered "
            f"{expected['source_fingerprint']!r}"
        )
    if int(library.get("count") or -1) != int(expected["count"]):
        problems.append(f"{library.get('count')} sentences, pre-registered {expected['count']}")
    if library.get("split") is not None and library.get("split") != expected["split"]:
        problems.append(f"split {library.get('split')!r}, pre-registered {expected['split']!r}")
    if expected.get("tolerance") is not None:
        problems.extend(_tolerance_problems(report, library, float(expected["tolerance"])))
    registered = dict((prereg.payload.get("fixed") or {}).get("exemplar_library") or {})
    collection = registered.get("collection")
    if collection and library.get("collection") != collection:
        problems.append(f"collection {library.get('collection')!r}, pre-registered {collection!r}")
    if problems:
        raise PreregError(
            "the sync report's library is not the pre-registered one: " + "; ".join(problems)
        )
    quoted = json.dumps(EXEMPLAR_LIBRARY_PLACEHOLDER)
    if text.count(quoted) != 1:
        raise PreregError(f"the placeholder {quoted} must appear exactly once in {source}")
    filled = text.replace(quoted, json.dumps(fingerprint))
    payload = json.loads(filled)
    if payload["fixed"]["exemplar_library"]["fingerprint"] != fingerprint:
        raise PreregError("the placeholder is not at fixed.exemplar_library.fingerprint")
    source.write_text(filled, encoding="utf-8")
    return fingerprint


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tolerance_problems(
    report: Mapping[str, Any], library: Mapping[str, Any], registered: float
) -> list[str]:
    """How a sync report's verification falls short of the pre-registered tolerance."""
    problems = []
    for where, value in (
        ("report", report.get("tolerance")),
        ("library", library.get("tolerance")),
    ):
        number = _finite(value)
        if number is None or not number <= registered:
            problems.append(
                f"verified at tolerance {value!r} ({where}), pre-registered {registered:g}"
            )
    if library.get("compared_with_source") is not True:
        problems.append("the library was not compared with the npz it was copied from")
    for key in ("max_abs_diff", "max_norm_deviation"):
        number = _finite(library.get(key))
        if number is None or not number <= registered:
            problems.append(
                f"{key} {library.get(key)!r} is beyond the pre-registered tolerance {registered:g}"
            )
    return problems


def write_env(path: str | Path, lines: Sequence[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
