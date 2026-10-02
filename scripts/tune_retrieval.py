"""Dev-only retrieval tuning harness (docs 21.3.15; plan sections 4-6).

Every subcommand is hard-wired to the public dev split of tram, expert and derived_procedures;
paths naming CTI-HAL, EvidenceForge, the Elastic posts, the clinic bundle or a test/train split
are refused. Nothing here writes to ChromaDB or under ``data/models``.

``cache``        Stage A, the only GPU/Chroma step: embeds the dev texts, their retrieval segments
                 and the exemplar query texts once; reads ``attack_techniques`` read-only (ANN
                 hits with the runtime field filter at ``--dense-ann-depths``, the exact top
                 ``--exact-depth`` by brute force over the collection's embeddings, optionally a
                 ``--technique-collection-copy``) and the BM25 top ``--lexical-depth`` per
                 segment, into the new directory ``--output``.
``parity``       P1 (blocking, ``--live``): harness rows at the serving settings, replaying ANN at
                 depth 90, equal the real ``retrieve_techniques`` rows with the real
                 ``ChromaVectorStore`` and the memo embedder, bit for bit, cues on and off.
                 P1b (diagnostic): the old fit settings reproduce the shipped model's rows and dev
                 operating point; the shipped weights on serving rows agree with
                 ``--dev-artifact`` (R09). P2 (``--live``): the votes of the harness's
                 exemplar library equal those of the Chroma store the runtime loads
                 (``ChromaExemplarStore.query_vectors``) bit for bit: the same library (Chroma
                 is the source; a Chroma library is read with exact search). P3: the emulated
                 runtime decision and ranking equal ``finalize_retrieval_candidates`` /
                 ``rank_candidates_by_acceptance`` at ``--p3-thresholds``, cues on and off.
``ann-recall``   ANN against exact dense search at each cached depth: recall per segment and the
                 share of dev cases whose dense set or candidate pool changes.
``sweep``        E0 (report only) and the pre-registered families read from ``--prereg`` (refused
                 unless committed and clean in git, every later commit a declared amendment):
                 grouped CV seeds, an out-of-fold logistic per fold (``AcceptanceModel.fit``),
                 the threshold from ``choose_threshold_runtime`` on pooled out-of-fold p, the
                 section-6 bar with a paired cluster bootstrap, greedy family-wise selection,
                 and E4 (nested-CV F1 of the finalist). Refused unless ``--parity`` is a passing
                 P1-P3 artifact of the same cache, base settings and harness code; the base
                 settings equal ``fixed.settings``; the cache holds exactly ``dev_only`` (its
                 datasets and case count); and the exemplar library is ``fixed.exemplar_library``
                 (its fingerprint, backend ``chroma``, source npz and tolerance, with the
                 measured deviations inside it; a placeholder fingerprint is refused).
``replicate``    The greedy path on fresh seeds, stepping back until a configuration replicates;
                 writes ``--write-env`` (T1.env: ``VAR=value`` lines, the fixed settings pinned,
                 ``TECHNIQUE_EXEMPLAR_TOLERANCE`` among them, and the library's backend,
                 collection and ``TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT``) and ``T1.fit_args``. Refused unless the prereg commit, cache, base settings,
                 exemplar library and harness code are the sweep's.
``copy-check``   Only when replicate selected exact dense search (E2a): after Phase 5b, on a cache
                 holding the collection copy's hits, the replication step-back again with every
                 exact configuration served from the copy; its selection is T1 (T1.env names
                 the copy). Needs a passing ``parity --dense-source copy`` artifact.
``rethreshold``  T1-P40: the same weights at the P >= 0.40 point of the fit's pooled out-of-fold
                 probabilities (``fit_acceptance_model.py --dump-rows``); refused unless the
                 dump belongs to the model (threshold, retrieval contract, row counts), and
                 nothing is written when the dump does not reproduce the model's threshold.
``check-fit``    The fit's dumped rows, weights, threshold and retrieval contract against the
                 harness under the artifact's base settings. Rows match when technique, label
                 and guards are equal and every feature is within ``CHECK_FIT_FEATURE_ATOL``
                 (the fit embeds each case alone, the cache in padded batches: the low bits of
                 the vectors differ); the accepted sets per case must agree too.
``report``       A short text summary of the artifacts above.
``prereg-fill``  Writes the Chroma library fingerprint of a passing
                 ``scripts/sync_exemplars_to_chroma.py --verify-only`` report into the
                 pre-registration's placeholder ``fixed.exemplar_library.fingerprint`` (refused
                 unless the report's library is the pre-registered backend, source npz, count and
                 collection, verified against the npz at no looser than the pre-registered
                 tolerance with every deviation inside it); commit the file before the sweep.

    PYTHONPATH=. .venv/bin/python scripts/tune_retrieval.py cache --output OUT/cache/dev
    PYTHONPATH=. .venv/bin/python scripts/tune_retrieval.py sweep --cache OUT/cache/dev \\
        --prereg docs/preregistration/chroma_exemplars_2026-10-02.json --seeds 7 11 13 \\
        --bootstrap 1000 --parity OUT/runs/R21_parity.json --output OUT/runs/R23_dev_sweep.json

Exit codes: 0 done (and every check passed), 1 a check failed, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.acceptance import (  # noqa: E402
    FEATURE_NAMES,
    AcceptanceModel,
    choose_threshold,
    choose_threshold_runtime,
    contract_mismatches,
)
from apt_classifier.evaluation import source_fingerprint, write_artifact  # noqa: E402
from apt_classifier.exemplars import parent_map as exemplar_parent_map  # noqa: E402
from apt_classifier.retrieve import (  # noqa: E402
    LexicalTechniqueIndex,
    finalize_retrieval_candidates,
    lexical_fields,
    retrieval_contract_for,
    retrieval_kwargs,
    retrieve_techniques,
)
from apt_classifier.vectorstore import canonical_json  # noqa: E402
from scripts import tuning_harness, tuning_selection  # noqa: E402
from scripts.benchmark_mapping import DEFAULT_ROOT, FINGERPRINTED_SOURCES  # noqa: E402
from scripts.retrieval_cache import case_folds, stratified_group_folds  # noqa: E402
from scripts.run_tooling import (  # noqa: E402
    add_exemplar_backend_arguments,
    exemplar_backend_problem,
    exemplar_setting_overrides,
    exemplar_signature,
    file_sha256,
    git_state,
)
from scripts.tuning_harness import (  # noqa: E402
    DEV_DATASETS,
    CacheError,
    CacheMiss,
    DevCache,
    DevOnlyError,
    RowSet,
    accepted_rows,
    ann_segment_recall,
    build_cache,
    build_rowset,
    candidate_rows,
    case_rankings,
    case_texts,
    check_dev_artifact,
    depth_cap_share,
    exemplar_identity,
    fold_leak_check,
    generic_techniques,
    load_dev_cases,
    output_problems,
    require_dev_path,
    retrieval_arguments,
    rowset_differences,
    score_rows,
    threshold_curve,
)
from scripts.tuning_selection import (  # noqa: E402
    KNOBS,
    Candidate,
    CaseCounts,
    Prereg,
    PreregError,
    amendment_problems,
    env_assignments,
    fill_exemplar_library,
    fit_arguments,
    full_config,
    greedy_path,
    is_library_fingerprint,
    load_prereg,
    prereg_git_problems,
    replicate_step_back,
    write_env,
)

DEFAULT_PREREG = REPO_ROOT / "docs" / "preregistration" / "chroma_exemplars_2026-10-02.json"
# B0r: the serving settings (the pre-registration's reference; parity and ann-recall use it
# before the pre-registration is committed, and a test pins the two equal)
REFERENCE_CONFIG: dict[str, Any] = {
    "exemplar_k": 64,
    "exemplar_tau": 0.05,
    "exemplar_new_candidates": 3,
    "exemplar_vote_norm": "none",
    "exemplar_segment_votes": "off",
    "dense_relative_margin": 0.12,
    "dense_overfetch": 0,
    "candidate_pool": 12,
    "include_synthetic": False,
    "platform_from_text": True,
    "dense_search": "ann",
    "l2": 1.0,
}
# the old fit settings (synthetic sentences on, no platform-from-text scoping): P1b, E0
OLD_FIT_OVERRIDES = {"include_synthetic": True, "platform_from_text": False}
DEFAULT_P3_THRESHOLDS = (0.2, 0.35, 0.5)
P2_VOTE_SETTINGS = (
    {"k": 64, "tau": 0.05, "vote_norm": "none"},
    {"k": 256, "tau": 0.03, "vote_norm": "none"},
    {"k": 64, "tau": 0.05, "vote_norm": "label_split"},
)
P1B_ROW_TOLERANCE = 0.005
P1B_OPERATING_POINT_TOLERANCE = 0.01
P1B_AGREEMENT = 0.995
CHECK_FIT_ROW_SHARE = 0.995
CHECK_FIT_WEIGHT_COSINE = 0.999
CHECK_FIT_THRESHOLD = 0.02
# check-fit and copy-check compare features within this absolute tolerance, not bit for bit:
# the fit embeds each case's texts alone, the cache embedded them in padded batches of 64, and
# BERT's output moves in the low bits with the batch shape (about 1.5e-7 per vector component,
# 4e-7 per similarity; the exemplar vote softmax at tau 0.03 multiplies that by about 33).
# Labels, guards and technique ids are compared exactly, and so are the accepted sets per case.
CHECK_FIT_FEATURE_ATOL = 1e-4
CHECK_FIT_CASE_SHARE = 0.995
MEMORY_ROWSETS = 8
# the code the harness rows depend on (rows-cache key), and all the code a selection depends on
ROW_CODE_KEYS = ("sources", "harness")
# the pre-registered fixed settings, recorded in every artifact and compared between steps (a
# test pins this list to the pre-registration's fixed.settings)
RECORDED_SETTINGS = (
    "embedding_model",
    "embedding_model_revision",
    "reranker_enabled",
    "retrieval_top_k",
    "similarity_threshold",
    "technique_acceptance_threshold",
    "technique_corroboration_accept_min_probability",
    "technique_corroboration_accept_observations",
    "technique_corroboration_bonus",
    "technique_exemplar_duplicate_similarity",
    "technique_exemplar_tolerance",
    "technique_exemplars_enabled",
    "technique_hierarchy_parent_mode",
    "technique_include_detection",
    "technique_parent_policy",
    "technique_rank_by_acceptance",
    "technique_relative_margin",
    "technique_retrieval_mode",
    "technique_rrf_k",
    "technique_segment_acceptance",
)


class Refused(RuntimeError):
    """The harness refuses the request (exit 2)."""


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return write_artifact(path, _jsonable(dict(payload)))


def _guard_outputs(paths: Sequence[Path | str | None], *, overwrite: bool) -> None:
    for path in paths:
        if path is not None:
            require_dev_path(path, "output")
    problems = output_problems(paths, overwrite=overwrite)
    if problems:
        raise Refused("refusing to write: " + "; ".join(problems))


def _read_json(path: Path) -> dict[str, Any]:
    require_dev_path(path, "input")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Refused(f"{path} is not a JSON object")
    return payload


# ----- settings, Chroma client and exemplar library -----
def base_settings(args: argparse.Namespace, **extra: Any) -> Any:
    from apt_classifier.config import Settings

    return Settings(**exemplar_setting_overrides(args), **extra)


def _chroma_client(settings: Any) -> Any:
    import chromadb

    return chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port, ssl=settings.chroma_ssl
    )


class ClientHolder:
    """The Chroma client, created only when something needs it (the npz library does not)."""

    def __init__(self, settings: Any, client: Any = None) -> None:
        self.settings = settings
        self._client = client

    def get(self) -> Any:
        if self._client is None:
            self._client = _chroma_client(self.settings)
        return self._client


def library_settings(settings: Any) -> Any:
    """The settings the harness loads its exemplar library with: a Chroma library is searched
    exactly (the votes of ``hnsw`` search are bit-identical on the same library, P2 checks it,
    and exact search needs no Chroma query per exemplar query)."""
    backend = str(getattr(settings, "technique_exemplar_backend", "npz") or "npz").lower()
    if backend == "npz" or getattr(settings, "technique_exemplar_search", "exact") == "exact":
        return settings
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update={"technique_exemplar_search": "exact"})
    return SimpleNamespace(**{**vars(settings), "technique_exemplar_search": "exact"})


def load_exemplars(settings: Any, clients: ClientHolder, required: str | None = None) -> Any:
    from apt_classifier.exemplars import load_exemplar_index

    backend = str(getattr(settings, "technique_exemplar_backend", "npz") or "npz").lower()
    client = clients.get() if backend != "npz" else None
    index, status = load_exemplar_index(library_settings(settings), client=client)
    if index is None:
        raise Refused(f"the exemplar library did not load ({status})")
    problem = exemplar_backend_problem(index, required)
    if problem:
        raise Refused(f"{problem} ({status})")
    return index


def library_env(index: Any) -> dict[str, Any]:
    """What T1.env pins about the exemplar library the selection ran on: its backend and, for a
    Chroma library, its collection and fingerprint (``TECHNIQUE_EXEMPLAR_LIBRARY_FINGERPRINT``:
    the loader then refuses any other vectors, with or without the npz; the fitted model runs
    only on that library)."""
    backend = str(getattr(index, "backend", "npz") or "npz")
    entry: dict[str, Any] = {"backend": backend}
    if backend == "chroma":
        entry["collection"] = getattr(index, "collection_name", None)
        entry["fingerprint"] = getattr(index, "fingerprint", None)
    return entry


def fixed_setting_problems(settings: Any, fixed: Mapping[str, Any]) -> list[str]:
    """Pre-registered fixed settings the run's base settings depart from."""
    problems = []
    for name, expected in fixed.items():
        actual = getattr(settings, name, None)
        if isinstance(expected, bool):
            same = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int | float):
            same = actual is not None and float(actual) == float(expected)
        else:
            same = str(actual) == str(expected)
        if not same:
            problems.append(f"{name} is {actual!r}, pre-registered fixed at {expected!r}")
    return problems


def settings_for(base: Any, config: Mapping[str, Any]) -> Any:
    update = {
        KNOBS[name].setting: KNOBS[name].cast(value)
        for name, value in config.items()
        if KNOBS[name].setting
    }
    if hasattr(base, "model_copy"):
        return base.model_copy(update=update)
    copy = SimpleNamespace(**vars(base))
    for key, value in update.items():
        setattr(copy, key, value)
    return copy


# ----- what every step records and the next step compares -----
def harness_code() -> dict[str, str | None]:
    """Hashes of the code a result depends on: the fingerprinted retrieval sources and the
    three harness files (``sources`` and ``harness`` also key the rows cache)."""
    return {
        "sources": source_fingerprint(FINGERPRINTED_SOURCES),
        "harness": file_sha256(Path(tuning_harness.__file__)),
        "tune_retrieval": file_sha256(Path(__file__)),
        "selection": file_sha256(Path(tuning_selection.__file__)),
    }


def base_settings_record(settings: Any) -> dict[str, Any]:
    """The base settings a run's rows depend on besides the knobs: every ``retrieval_kwargs``
    value with the knobs pinned to the reference (so only non-knob settings remain), the
    lexical fields, and the pre-registered fixed settings (:data:`RECORDED_SETTINGS`)."""
    pinned = settings_for(settings, full_config(REFERENCE_CONFIG))
    kwargs = retrieval_kwargs(pinned, exemplar_index=None, lexical_index=None)
    return _jsonable(
        {
            "retrieval_kwargs": {
                key: value
                for key, value in sorted(kwargs.items())
                if key not in ("exemplar_index", "lexical_index")
            },
            "lexical_fields": list(lexical_fields(pinned)),
            "settings": {name: getattr(settings, name, None) for name in RECORDED_SETTINGS},
        }
    )


def record_problems(mine: Mapping[str, Any], theirs: Any, what: str) -> list[str]:
    """How this run's base settings depart from those ``what`` recorded."""
    if not isinstance(theirs, Mapping):
        return [f"{what} records no base settings (made by an older harness; re-run it)"]
    problems = []
    for section in ("retrieval_kwargs", "settings"):
        here, there = dict(mine.get(section) or {}), dict(theirs.get(section) or {})
        for name in sorted(set(here) | set(there)):
            if here.get(name) != there.get(name):
                problems.append(f"{name} is {here.get(name)!r} here, {there.get(name)!r} in {what}")
    if list(mine.get("lexical_fields") or ()) != list(theirs.get("lexical_fields") or ()):
        problems.append(
            f"lexical fields {mine.get('lexical_fields')} here, {theirs.get('lexical_fields')} "
            f"in {what}"
        )
    return problems


def code_problems(
    mine: Mapping[str, Any], theirs: Any, what: str, keys: Sequence[str] = ()
) -> list[str]:
    if not isinstance(theirs, Mapping):
        return [f"{what} records no harness code hashes (made by an older harness; re-run it)"]
    names = tuple(keys) or tuple(sorted(set(mine) | set(theirs)))
    return [
        f"{name} code differs from {what} ({str(mine.get(name))[:12]} here, "
        f"{str(theirs.get(name))[:12]} there)"
        for name in names
        if mine.get(name) != theirs.get(name)
    ]


def scope_problems(harness: Harness, prereg: Prereg) -> list[str]:
    """The cache must hold exactly the pre-registered case set (``dev_only``)."""
    datasets = sorted({case.dataset for case in harness.cache.cases})
    problems = []
    if datasets != sorted(prereg.dev_datasets):
        problems.append(
            f"the cache holds the datasets {datasets}, the pre-registration "
            f"{sorted(prereg.dev_datasets)} (rebuild it with cache and the default --datasets)"
        )
    recorded = sorted((harness.cache.manifest.get("dev_only") or {}).get("datasets") or ())
    if recorded != datasets:
        problems.append(f"the cache manifest names the datasets {recorded}, its cases {datasets}")
    if len(harness.cache.cases) != prereg.dev_cases:
        problems.append(
            f"the cache holds {len(harness.cache.cases)} cases, the pre-registration "
            f"{prereg.dev_cases}"
        )
    return problems


def library_problems(index: Any, expected: Mapping[str, Any]) -> list[str]:
    """The exemplar library must be ``fixed.exemplar_library`` (fingerprint, count, split and,
    when pre-registered, backend, the npz a Chroma library was copied from and the tolerance it
    was verified at, with its measured deviations inside it)."""
    fingerprint = str(getattr(index, "fingerprint", "") or "")
    count = len(index) if index is not None else 0
    split = (getattr(index, "metadata", None) or {}).get("split")
    backend = str(getattr(index, "backend", "npz") or "npz") if index is not None else None
    problems = []
    if expected.get("backend") and backend != expected["backend"]:
        problems.append(
            f"the exemplar library loaded from {backend!r}, pre-registered "
            f"{expected['backend']!r} (the tuning rows read the votes of that library)"
        )
    if expected.get("source_fingerprint"):
        source = str(getattr(index, "source_fingerprint", None) or fingerprint)
        if source != expected["source_fingerprint"]:
            problems.append(
                f"the exemplar library was copied from {source!r}, pre-registered "
                f"{expected['source_fingerprint']!r}"
            )
    if not is_library_fingerprint(expected["fingerprint"]):
        problems.append(
            f"fixed.exemplar_library.fingerprint is the placeholder {expected['fingerprint']!r}: "
            "fill it from the sync --verify-only report (tune_retrieval.py prereg-fill) and "
            "commit the pre-registration before the sweep"
        )
    elif fingerprint != str(expected["fingerprint"]):
        problems.append(
            f"the exemplar library is {fingerprint!r}, pre-registered {expected['fingerprint']!r}"
        )
    if count != int(expected["count"]):
        problems.append(
            f"the exemplar library holds {count} sentences, pre-registered {expected['count']}"
        )
    if expected.get("tolerance") is not None and backend == "chroma":
        registered = float(expected["tolerance"])
        loaded = getattr(index, "tolerance", None)
        if loaded is None or float(loaded) != registered:
            problems.append(
                f"the Chroma library was verified at tolerance {loaded!r}, pre-registered "
                f"{registered:g}"
            )
        verification = dict(getattr(index, "verification", None) or {})
        for key in ("max_norm_deviation", "max_abs_diff"):
            value = verification.get(key)
            if key == "max_abs_diff" and not verification.get("compared_with_source"):
                continue  # no npz present: the fingerprint above pins the vectors exactly
            if value is None or not float(value) <= registered:
                problems.append(
                    f"the Chroma library's {key} {value!r} is beyond the pre-registered "
                    f"tolerance {registered:g}"
                )
    if split != expected["split"]:
        problems.append(
            f"the exemplar library is from split {split!r}, pre-registered {expected['split']!r}"
        )
    return problems


def selection_problems(harness: Harness, prereg: Prereg) -> list[str]:
    """What a selection step (sweep, replicate, copy-check) refuses: base settings other than
    ``fixed.settings``, a case set other than ``dev_only``, a library other than
    ``fixed.exemplar_library``."""
    return [
        *(
            f"base settings depart from the pre-registration: {problem}"
            for problem in fixed_setting_problems(harness.settings, prereg.fixed_settings)
        ),
        *scope_problems(harness, prereg),
        *library_problems(harness.exemplar_index, prereg.exemplar_library),
    ]


def parity_problems(
    payload: Mapping[str, Any], harness: Harness, prereg: Prereg, *, dense_search: str
) -> list[str]:
    """A selection step runs only on rows P1 showed equal to serving: ``payload`` must be a
    passing parity run (P1-P3, P1 live) of this cache at the reference, under these base
    settings and this row code."""
    if payload.get("kind") != "tune_retrieval_parity":
        return ["--parity is not a parity artifact"]
    problems = []
    blocking = payload.get("blocking") or {}
    for name in ("P1", "P2", "P3"):
        if blocking.get(name) is not True:
            problems.append(f"parity {name} did not pass")
    if payload.get("passed") is not True:
        problems.append("the parity run did not pass")
    p1 = payload.get("P1") or {}
    for key in ("cues_on", "cues_off"):
        if (p1.get(key) or {}).get("differing_cases") != 0:
            problems.append(f"P1 {key} was not run or found differing cases")
    if (payload.get("cache") or {}).get("key") != harness.cache.key:
        problems.append("the parity run used another cache")
    try:
        config = full_config(payload.get("config") or {})
    except ValueError as error:
        problems.append(f"the parity configuration is unreadable ({error})")
    else:
        expected = full_config(prereg.reference, {"dense_search": dense_search})
        if config != expected:
            problems.append(f"parity ran at {config}, not at the reference {expected}")
    problems += record_problems(
        base_settings_record(harness.settings), payload.get("base_settings"), "the parity run"
    )
    problems += code_problems(harness.code, payload.get("code"), "the parity run", ROW_CODE_KEYS)
    return problems


def _parity_reference(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "artifact_hash": payload.get("artifact_hash"),
        "generated_at": payload.get("generated_at"),
    }


# ----- the harness context -----
@dataclass
class SeedResult:
    seed: int
    operating_point: dict[str, Any]
    on: tuning_harness.Score
    off: tuning_harness.Score | None
    probabilities: np.ndarray


class Harness:
    """Rows per configuration (memory and disk cached), folds per seed and evaluations."""

    def __init__(
        self,
        cache: DevCache,
        settings: Any,
        kb: Any,
        exemplar_index: Any,
        *,
        folds: int = 5,
        group_similarity: float = 0.90,
        min_precision: float = 0.5,
        generic_min_groups: int = 30,
        use_rows_cache: bool = True,
        log: Callable[[str], None] = _log,
    ) -> None:
        self.cache = cache
        self.settings = settings
        self.kb = kb
        self.exemplar_index = exemplar_index
        self.n_folds = int(folds)
        self.group_similarity = float(group_similarity)
        self.min_precision = float(min_precision)
        self.generic_min_groups = int(generic_min_groups)
        self.generic = generic_techniques(kb, self.generic_min_groups)
        self.use_rows_cache = use_rows_cache
        self.log = log
        self.case_ids = [case.case_id for case in cache.cases]
        self.datasets = [case.dataset for case in cache.cases]
        self.rank_by_acceptance = bool(getattr(settings, "technique_rank_by_acceptance", True))
        self.attack_version = str(getattr(kb, "version", "") or "")
        self._vectors: np.ndarray | None = None
        self._folds: dict[int, np.ndarray] = {}
        self._groups: np.ndarray | None = None
        self._rows: OrderedDict[str, RowSet] = OrderedDict()
        self.code = harness_code()
        self._code = {key: self.code[key] for key in ROW_CODE_KEYS}
        self.rows_built = 0

    # rows
    def settings_for(self, config: Mapping[str, Any]) -> Any:
        return settings_for(self.settings, config)

    def rows_key(self, config: Mapping[str, Any], cues: bool) -> str:
        settings = self.settings_for(config)
        kwargs = {
            key: value
            for key, value in retrieval_kwargs(
                settings, exemplar_index=None, lexical_index=None
            ).items()
            if key not in ("exemplar_index", "lexical_index")
        }
        material = {
            "format": tuning_harness.ROWS_FORMAT_VERSION,
            "cache": self.cache.key,
            "kwargs": kwargs,
            "lexical_fields": list(lexical_fields(settings)),
            "dense_search": str(config["dense_search"]),
            "cues": bool(cues),
            "exemplars": exemplar_identity(self.exemplar_index)["fingerprint"],
            "generic_min_groups": self.generic_min_groups,
            "code": self._code,
        }
        return hashlib.sha256(canonical_json(material).encode()).hexdigest()[:32]

    def rows(self, config: Mapping[str, Any], cues: bool, *, label: str = "") -> RowSet:
        config = full_config(config)
        key = self.rows_key(config, cues)
        if key in self._rows:
            self._rows.move_to_end(key)
            return self._rows[key]
        path = self.cache.path / "rows" / f"{key}.npz"
        rows: RowSet | None = None
        if self.use_rows_cache and path.is_file():
            rows = RowSet.load(path)
            if rows.meta.get("rows_key") != key or rows.case_ids != tuple(self.case_ids):
                rows = None
        if rows is None:
            started = time.time()
            settings = self.settings_for(config)
            rows = build_rowset(
                self.cache.cases,
                kb=self.kb,
                settings=settings,
                store=self.cache.technique_store(str(config["dense_search"])),
                lexical_index=self.cache.lexical_index(lexical_fields(settings)),
                exemplar_index=self.exemplar_index,
                cues_enabled=cues,
                generic=self.generic,
                meta={"rows_key": key, "config": dict(config), "label": label},
            )
            self.rows_built += 1
            self.log(
                f"rows {label or key[:8]} cues {'on' if cues else 'off'}: {len(rows)} rows, "
                f"{int(rows.labels.sum())} positive, {time.time() - started:.0f}s"
            )
            if self.use_rows_cache:
                rows.save(path)
        self._rows[key] = rows
        while len(self._rows) > MEMORY_ROWSETS:
            self._rows.popitem(last=False)
        return rows

    # folds
    def _case_vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self.cache.case_vectors()
        return self._vectors

    def folds(self, seed: int) -> np.ndarray:
        if seed not in self._folds:
            by_case, groups = case_folds(
                self.case_ids,
                self.datasets,
                self._case_vectors(),
                n_splits=self.n_folds,
                seed=int(seed),
                min_similarity=self.group_similarity,
            )
            self._folds[seed] = np.asarray([by_case[case] for case in self.case_ids])
            if self._groups is None:
                self._groups = np.asarray([groups[case] for case in self.case_ids])
        return self._folds[seed]

    @property
    def groups(self) -> np.ndarray:
        """Near-duplicate group per case (the bootstrap's clusters; seed independent)."""
        if self._groups is None:
            self.folds(7)
        if self._groups is None:
            raise RuntimeError("no case groups")
        return self._groups

    # evaluation
    def evaluate(
        self,
        candidate_id: str,
        config: Mapping[str, Any],
        seeds: Sequence[int],
        *,
        fit_config: Mapping[str, Any] | None = None,
        rule: str = "runtime",
        with_off: bool = True,
        min_precision: float | None = None,
        curve_seed: int | None = None,
    ) -> Candidate:
        config = full_config(config)
        serve_on = self.rows(config, True, label=candidate_id)
        serve_off = self.rows(config, False, label=candidate_id) if with_off else None
        fit_rows = (
            self.rows(full_config(fit_config), True, label=f"{candidate_id}/fit")
            if fit_config is not None
            else serve_on
        )
        floor = self.min_precision if min_precision is None else float(min_precision)
        results: list[SeedResult] = []
        curve = None
        for seed in seeds:
            result = evaluate_seed(
                fit_rows,
                serve_on,
                serve_off,
                self.folds(seed),
                seed=int(seed),
                l2=float(config["l2"]),
                attack_version=self.attack_version,
                min_precision=floor,
                rule=rule,
                rank_by_acceptance=self.rank_by_acceptance,
            )
            results.append(result)
            if curve_seed is not None and int(seed) == int(curve_seed):
                curve = threshold_curve(serve_on, result.probabilities)
        counts = CaseCounts(
            tp=np.vstack([result.on.tp for result in results]),
            fp=np.vstack([result.on.fp for result in results]),
            expected=serve_on.expected_array,
        )
        return Candidate(
            candidate_id,
            dict(config),
            summarize(results),
            counts,
            extra={"curve": curve, "rows": len(serve_on), "positives": int(serve_on.labels.sum())},
        )


def fold_probabilities(
    fit_rows: RowSet,
    case_fold: np.ndarray,
    *,
    l2: float,
    attack_version: str,
    score: Sequence[RowSet] = (),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Out-of-fold probabilities: one ``AcceptanceModel.fit`` per fold on the other folds' rows
    of ``fit_rows`` (as ``retrieval_cache.out_of_fold_probabilities``), scoring its held-out
    rows and the held-out rows of every ``score`` set (the cues-off rows, or serving rows when
    the fit used other settings)."""
    fit_dicts = fit_rows.feature_dicts
    labels = fit_rows.labels.astype(np.int64)
    row_fold = case_fold[fit_rows.case_index]
    fit_output = np.full(len(fit_rows), np.nan)
    outputs = [np.full(len(rows), np.nan) for rows in score]
    score_folds = [case_fold[rows.case_index] for rows in score]
    folds = sorted(
        set(row_fold.tolist()) | {fold for item in score_folds for fold in item.tolist()}
    )
    for fold in folds:
        train = np.flatnonzero(row_fold != fold)
        if not len(train):
            raise ValueError("a fold holds every row; nothing is left to fit on")
        model = AcceptanceModel.fit(
            [fit_dicts[index] for index in train.tolist()],
            labels[train].tolist(),
            attack_version=attack_version,
            l2=l2,
        )
        for index in np.flatnonzero(row_fold == fold).tolist():
            fit_output[index] = model.probability(fit_dicts[index])
        for rows, folds_of_rows, output in zip(score, score_folds, outputs, strict=True):
            dicts = rows.feature_dicts
            for index in np.flatnonzero(folds_of_rows == fold).tolist():
                output[index] = model.probability(dicts[index])
    return fit_output, outputs


def choose_operating_point(
    rows: RowSet, probabilities: np.ndarray, *, rule: str, min_precision: float
) -> dict[str, Any]:
    if rule == "runtime":
        blocked, forced = rows.gate_masks()
        return choose_threshold_runtime(
            probabilities,
            rows.labels,
            rows.row_case_ids,
            rows.expected_counts,
            blocked=blocked,
            forced=forced,
            technique_ids=rows.technique_ids.tolist(),
            min_precision=min_precision,
        )
    if rule == "p_only":
        point = choose_threshold(
            probabilities,
            rows.labels,
            rows.row_case_ids,
            rows.expected_counts,
            min_precision=min_precision,
        )
        point["rule"] = "p_only"
        return point
    raise ValueError(f"unknown threshold rule {rule!r}")


def evaluate_seed(
    fit_rows: RowSet,
    serve_on: RowSet,
    serve_off: RowSet | None,
    case_fold: np.ndarray,
    *,
    seed: int,
    l2: float,
    attack_version: str,
    min_precision: float,
    rule: str = "runtime",
    rank_by_acceptance: bool = True,
) -> SeedResult:
    """One CV seed: fold models on ``fit_rows``, the threshold by ``rule`` on its pooled
    out-of-fold p, and the runtime decision on the serving rows (cues on and off)."""
    extra = [rows for rows in (serve_on, serve_off) if rows is not None and rows is not fit_rows]
    p_fit, scored = fold_probabilities(
        fit_rows, case_fold, l2=l2, attack_version=attack_version, score=extra
    )
    by_id = {id(rows): p for rows, p in zip(extra, scored, strict=True)}
    by_id[id(fit_rows)] = p_fit
    point = choose_operating_point(fit_rows, p_fit, rule=rule, min_precision=min_precision)
    threshold = float(point["threshold"])
    p_on = by_id[id(serve_on)]
    on = score_rows(serve_on, p_on, threshold, rank_by_acceptance=rank_by_acceptance)
    off = (
        score_rows(
            serve_off, by_id[id(serve_off)], threshold, rank_by_acceptance=rank_by_acceptance
        )
        if serve_off is not None
        else None
    )
    return SeedResult(int(seed), point, on, off, p_on)


SUMMARY_KEYS = (
    "precision",
    "recall",
    "f1",
    "hit_at_1",
    "hit_at_3",
    "hit_at_10",
    "mrr",
    "accepted_per_case",
    "generic_share",
    "pool_label_recall",
    "pool_case_share",
)


def summarize(results: Sequence[SeedResult]) -> dict[str, Any]:
    """Means over seeds of the metrics the bar reads, with the per-seed values."""
    per_seed: list[dict[str, Any]] = []
    for result in results:
        metrics = result.on.metrics
        entry: dict[str, Any] = {
            "seed": result.seed,
            "threshold": float(result.operating_point["threshold"]),
            "precision_floor_met": bool(result.operating_point.get("precision_floor_met")),
            "fit_operating_point": {
                key: result.operating_point.get(key)
                for key in ("precision", "recall", "micro_f1", "accepted", "rule")
            },
            **{key: metrics[key] for key in SUMMARY_KEYS},
            "dataset_f1": {name: block["f1"] for name, block in metrics["datasets"].items()},
            "dataset_precision": {
                name: block["precision"] for name, block in metrics["datasets"].items()
            },
            "dataset_recall": {
                name: block["recall"] for name, block in metrics["datasets"].items()
            },
        }
        if result.off is not None:
            entry.update(
                {
                    "f1_no_cues": result.off.metrics["f1"],
                    "precision_no_cues": result.off.metrics["precision"],
                    "recall_no_cues": result.off.metrics["recall"],
                    "hit_at_10_no_cues": result.off.metrics["hit_at_10"],
                }
            )
        per_seed.append(entry)
    summary: dict[str, Any] = {
        key: float(np.mean([entry[key] for entry in per_seed])) for key in SUMMARY_KEYS
    }
    for key in ("f1_no_cues", "precision_no_cues", "recall_no_cues", "hit_at_10_no_cues"):
        if all(key in entry for entry in per_seed):
            summary[key] = float(np.mean([entry[key] for entry in per_seed]))
    for key in ("dataset_f1", "dataset_precision", "dataset_recall"):
        names = sorted({name for entry in per_seed for name in entry[key]})
        summary[key] = {
            name: float(np.mean([entry[key][name] for entry in per_seed])) for name in names
        }
    summary["threshold_mean"] = float(np.mean([entry["threshold"] for entry in per_seed]))
    summary["precision_floor_met"] = all(entry["precision_floor_met"] for entry in per_seed)
    summary["seeds"] = [entry["seed"] for entry in per_seed]
    summary["per_seed"] = per_seed
    return summary


def candidate_record(candidate: Candidate, reference: Mapping[str, Any]) -> dict[str, Any]:
    from scripts.tuning_selection import changed_knobs, distance_to_reference

    return {
        "id": candidate.id,
        "config": candidate.config,
        "changed_knobs": changed_knobs(candidate.config, reference),
        "distance_to_reference": distance_to_reference(candidate.config, reference),
        "summary": candidate.summary,
        **{key: value for key, value in candidate.extra.items() if value is not None},
    }


def shipped_library(model: Any, index: Any) -> dict[str, Any]:
    """Which library a model was fitted on against the one the rows are built from (the shipped
    model was fitted on the npz; Chroma's library has its own fingerprint)."""
    fitted = getattr(model, "exemplar_fingerprint", None)
    rows = getattr(index, "fingerprint", None) if index is not None else None
    return {"fitted_on": fitted, "rows_from": rows, "same_library": fitted == rows}


def shipped_probabilities(model: Any, rows: RowSet) -> np.ndarray:
    """The shipped model's p per row, as the runtime computes it (its priors added)."""
    return np.asarray(
        [
            model.probability({**row, **model.prior_features(str(technique_id))})
            for row, technique_id in zip(rows.feature_dicts, rows.technique_ids, strict=True)
        ],
        dtype=np.float64,
    )


def accepted_sets(rows: RowSet, probabilities: np.ndarray, threshold: float) -> list[set[str]]:
    accepted = accepted_rows(rows, probabilities, threshold)
    output: list[set[str]] = [set() for _ in range(rows.case_count)]
    for row in np.flatnonzero(accepted).tolist():
        output[int(rows.case_index[row])].add(str(rows.technique_ids[row]))
    return output


# ----- shared setup -----
def open_harness(
    args: argparse.Namespace,
    *,
    cache_path: Path,
    prereg: Prereg | None = None,
    client: Any = None,
    use_rows_cache: bool = True,
    settings: Any = None,
    kb: Any = None,
) -> tuple[Harness, ClientHolder]:
    from apt_classifier.attack.kb import AttackKnowledgeBase

    cache = DevCache.load(cache_path)
    settings = settings if settings is not None else base_settings(args)
    kb = kb if kb is not None else AttackKnowledgeBase.load(settings.kb_path)
    clients = ClientHolder(settings, client)
    exemplar_index = load_exemplars(
        settings, clients, getattr(args, "require_exemplar_backend", None)
    )
    stale = cache.staleness_problems(settings=settings, exemplar_source=exemplar_index)
    if stale:
        raise Refused("the cache is stale: " + "; ".join(stale) + " (rebuild it with: cache)")
    harness = Harness(
        cache,
        settings,
        kb,
        exemplar_index,
        folds=prereg.folds if prereg else 5,
        group_similarity=prereg.group_similarity if prereg else 0.90,
        min_precision=prereg.min_precision if prereg else 0.5,
        generic_min_groups=prereg.generic_min_groups if prereg else 30,
        use_rows_cache=use_rows_cache,
    )
    return harness, clients


def run_header(harness: Harness | None = None, **extra: Any) -> dict[str, Any]:
    header: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "dev_only": {"split": "dev", "allowed_datasets": list(DEV_DATASETS)},
        "llm": {"calls": 0, "cost_usd": 0.0},
    }
    if harness is not None:
        # what this run actually covered (the cache's cases), not what the harness allows
        header["dev_only"]["datasets"] = sorted({case.dataset for case in harness.cache.cases})
        header["cache"] = {
            "path": str(harness.cache.path.resolve()),
            "key": harness.cache.key,
            "cases": harness.cache.material["case_count"],
            "built_at_commit": harness.cache.material.get("git_commit"),
        }
        header["exemplars"] = _jsonable(exemplar_signature(harness.exemplar_index))
        header["base_settings"] = base_settings_record(harness.settings)
        header["code"] = dict(harness.code)
    header.update(extra)
    return header


# ----- cache -----
def cmd_cache(
    args: argparse.Namespace, *, client: Any = None, embedder_factory: Any = None, **_: Any
) -> int:
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.embeddings import SentenceTransformerEmbedder

    require_dev_path(args.root, "--root")
    _guard_outputs([args.output], overwrite=True)  # build_cache refuses a non-empty directory
    extra = {"technique_exemplar_collection": args.collection} if args.collection else {}
    settings = base_settings(args, **extra)
    kb = AttackKnowledgeBase.load(settings.kb_path)
    cases, diagnostics = load_dev_cases(Path(args.root), args.datasets, kb)
    clients = ClientHolder(settings, client)
    exemplar_index = load_exemplars(settings, clients, args.require_exemplar_backend)
    if args.collection and getattr(exemplar_index, "collection_name", None) != args.collection:
        raise Refused(
            f"--collection {args.collection}: the exemplar library loaded from "
            f"{exemplar_signature(exemplar_index)}"
        )
    device = None if args.device == "auto" else args.device
    factory = embedder_factory or (
        lambda: SentenceTransformerEmbedder.from_settings(settings, device=device)
    )
    started = time.time()
    cache = build_cache(
        Path(args.output),
        settings=settings,
        kb=kb,
        cases=cases,
        client=clients.get(),
        embedder_factory=factory,
        exemplar_source=exemplar_index,
        ann_depths=tuple(args.dense_ann_depths),
        synthetic_ann_depths=tuple(args.synthetic_ann_depths),
        exact_depth=args.exact_depth,
        lexical_depth=args.lexical_depth,
        copy_collection=args.technique_collection_copy,
        notes={"case_diagnostics": diagnostics, "device": args.device},
        log=_log,
    )
    _log(
        f"cache {cache.path}: {len(cache.cases)} cases, {cache.manifest['segments']} segments, "
        f"key {cache.key[:12]} ({time.time() - started:.0f}s; dropped {diagnostics['dropped_indexed_text']})"
    )
    return 0


# ----- parity -----
class _StandInModel:
    """A model exposing only ``.probability`` and ``.threshold`` (no contract, no priors)."""

    def __init__(self, model: AcceptanceModel, threshold: float) -> None:
        self._model = model
        self.threshold = float(threshold)

    def probability(self, features: dict[str, float]) -> float:
        return self._model.probability(features)


def p3_check(
    harness: Harness,
    config: Mapping[str, Any],
    *,
    cues: bool,
    thresholds: Sequence[float],
    limit: int = 20,
) -> dict[str, Any]:
    """P3: the emulated decision (``accepted_rows``) and ranking (``case_rankings``) against the
    real ``finalize_retrieval_candidates`` with a stand-in model, per case and threshold."""
    rows = harness.rows(config, cues, label="P3")
    model = AcceptanceModel.fit(
        rows.feature_dicts,
        rows.labels.tolist(),
        attack_version=harness.attack_version,
        l2=float(config["l2"]),
    )
    probabilities = np.asarray([model.probability(row) for row in rows.feature_dicts])
    settings = harness.settings_for(config)
    kwargs = retrieval_arguments(
        settings,
        exemplar_index=harness.exemplar_index,
        lexical_index=harness.cache.lexical_index(lexical_fields(settings)),
        cues_enabled=cues,
    )
    store = harness.cache.technique_store(str(config["dense_search"]))
    rank = bool(kwargs["rank_by_acceptance"])
    emulated_accepted = {t: accepted_sets(rows, probabilities, t) for t in thresholds}
    emulated_ranking = case_rankings(rows, probabilities, rank_by_acceptance=rank)
    mismatches: list[dict[str, Any]] = []
    decisions = 0
    for position, case in enumerate(harness.cache.cases):
        ledger = case.ledger()
        retrieval = retrieve_techniques(ledger, store, harness.kb, **{**kwargs, "finalize": False})
        rows_here = rows.case_rows(position)
        extracted = candidate_rows(
            ledger.observations[0],
            retrieval.candidates,
            segment_acceptance=bool(kwargs["segment_acceptance"]),
            threshold=float(kwargs["threshold"]),
        )
        if [original.technique_id for original, _, _ in extracted] != rows.technique_ids[
            rows_here
        ].tolist():
            mismatches.append({"case_id": case.case_id, "what": "candidates"})
            continue
        for threshold in thresholds:
            final = finalize_retrieval_candidates(
                retrieval,
                ledger,
                harness.kb,
                retrieval_mode=kwargs["retrieval_mode"],
                threshold=kwargs["threshold"],
                acceptance_threshold=kwargs["acceptance_threshold"],
                relative_margin=kwargs["relative_margin"],
                reranker=None,
                acceptance_model=_StandInModel(model, threshold),
                corroboration_bonus=kwargs["corroboration_bonus"],
                segment_acceptance=kwargs["segment_acceptance"],
                hierarchy_parent_mode=kwargs["hierarchy_parent_mode"],
                rank_by_acceptance=kwargs["rank_by_acceptance"],
                corroboration_accept_observations=kwargs["corroboration_accept_observations"],
                corroboration_accept_min_probability=kwargs["corroboration_accept_min_probability"],
            )
            decisions += 1
            real = set(final.observed_techniques)
            ranked = [
                candidate.technique_id
                for candidate in sorted(final.candidates, key=lambda item: item.fused_rank)
            ]
            what = []
            if real != emulated_accepted[threshold][position]:
                what.append("accepted")
            if ranked != emulated_ranking[position]:
                what.append("ranking")
            if what:
                mismatches.append(
                    {
                        "case_id": case.case_id,
                        "threshold": threshold,
                        "what": what,
                        "real": sorted(real),
                        "emulated": sorted(emulated_accepted[threshold][position]),
                    }
                )
    return {
        "cues": cues,
        "thresholds": list(thresholds),
        "cases": len(harness.cache.cases),
        "decisions": decisions,
        "mismatches": len(mismatches),
        "examples": mismatches[:limit],
        "passed": not mismatches,
    }


def p2_check(harness: Harness, settings: Any, client: Any, *, limit: int = 20) -> dict[str, Any]:
    """P2: the votes the harness rows read (its exemplar library) against the Chroma store the
    runtime loads with these settings (``ChromaExemplarStore.query_vectors``, the configured
    search), bit for bit, in the runtime's query batches. Chroma is the source: a harness on the
    npz fails P2 whenever Chroma re-normalised a vector (another library, another fingerprint)."""
    from apt_classifier.exemplar_store import load_chroma_exemplar_store
    from scripts.exemplar_backend_equivalence import vote_differences

    reference = harness.exemplar_index
    if reference is None:
        return {"passed": False, "reason": "the harness has no exemplar library"}
    store, reason = load_chroma_exemplar_store(settings, client)
    if store is None:
        return {"passed": False, "reason": f"Chroma exemplar store unavailable: {reason}"}
    library = {
        "harness": exemplar_signature(reference),
        "runtime_fingerprint": store.fingerprint,
        "same_library": store.fingerprint == getattr(reference, "fingerprint", None),
    }
    parents = exemplar_parent_map(harness.kb)
    embedder = harness.cache.embedder
    queries = differences = fallbacks = misses = 0
    examples: list[dict[str, Any]] = []
    for vote_settings in P2_VOTE_SETTINGS:
        arguments = {
            "parent_of": parents,
            "k": vote_settings["k"],
            "tau": vote_settings["tau"],
            "duplicate_similarity": float(
                getattr(settings, "technique_exemplar_duplicate_similarity", 0.95)
            ),
            "vote_norm": vote_settings["vote_norm"],
        }
        for case in harness.cache.cases:
            texts = case_texts(case)
            batches = [list(texts.exemplar_queries[:1])]
            if len(texts.exemplar_queries) > 1:
                batches.append(list(texts.exemplar_queries[1:]))
            for batch in batches:
                vectors = embedder.embed(batch)
                left = reference.query_vectors(vectors, **arguments)
                right = store.query_vectors(vectors, **arguments)
                stats = store.last_query_stats
                fallbacks += int(stats.get("search_fallbacks", 0))
                misses += int(stats.get("recall_misses", 0))
                for position, (a, b) in enumerate(zip(left, right, strict=True)):
                    queries += 1
                    fields = vote_differences(a, b)
                    if fields:
                        differences += 1
                        if len(examples) < limit:
                            examples.append(
                                {
                                    "case_id": case.case_id,
                                    "query": position,
                                    "settings": vote_settings,
                                    "fields": fields,
                                }
                            )
    return {
        "store": store.signature(),
        "library": library,
        "queries": queries,
        "differences": differences,
        "search_fallbacks": fallbacks,
        "recall_misses": misses,
        "examples": examples,
        "passed": library["same_library"] and differences == 0 and fallbacks == 0,
    }


def p1b_check(harness: Harness, *, shipped_path: Path, dev_artifact: Path | None) -> dict[str, Any]:
    """P1b (diagnostic): the old fit settings against the shipped model's recorded rows and
    operating point; the shipped weights on serving rows against a dev run's accepted sets."""
    shipped = AcceptanceModel.load(shipped_path)
    if shipped is None:
        return {"ran": False, "reason": f"no shipped model at {shipped_path}"}
    metadata = dict(shipped.metadata or {})
    report: dict[str, Any] = {
        "ran": True,
        "shipped_model": str(shipped_path),
        "threshold": shipped.threshold,
        # the shipped weights are applied to rows of the harness's library: on the Chroma
        # library (another fingerprint) that is a diagnostic the runtime would refuse
        "library": shipped_library(shipped, harness.exemplar_index),
    }
    old = harness.rows(full_config(REFERENCE_CONFIG, OLD_FIT_OVERRIDES), True, label="P1b-old")
    rows_expected = metadata.get("row_count")
    positives_expected = metadata.get("positive_rows")

    def within(value: float, target: Any, tolerance: float) -> bool | None:
        if target in (None, 0):
            return None
        return abs(float(value) - float(target)) <= tolerance * float(target)

    probabilities = shipped_probabilities(shipped, old)
    point = choose_threshold(
        probabilities,
        old.labels,
        old.row_case_ids,
        old.expected_counts,
        min_precision=float(metadata.get("min_precision", 0.5)),
        grid=[float(shipped.threshold)],
    )
    recorded = dict(metadata.get("dev_operating_point") or {})
    report["old_fit_rows"] = {
        "rows": len(old),
        "rows_expected": rows_expected,
        "rows_within": within(len(old), rows_expected, P1B_ROW_TOLERANCE),
        "positives": int(old.labels.sum()),
        "positives_expected": positives_expected,
        "positives_within": within(int(old.labels.sum()), positives_expected, P1B_ROW_TOLERANCE),
        "in_sample_at_threshold": {
            "precision": point["precision"],
            "recall": point["recall"],
            "recorded_precision": recorded.get("precision"),
            "recorded_recall": recorded.get("recall"),
            "within": (
                abs(point["precision"] - float(recorded["precision"]))
                <= P1B_OPERATING_POINT_TOLERANCE
                and abs(point["recall"] - float(recorded["recall"]))
                <= P1B_OPERATING_POINT_TOLERANCE
            )
            if "precision" in recorded and "recall" in recorded
            else None,
        },
    }
    if dev_artifact is not None:
        payload = _read_json(dev_artifact)
        check_dev_artifact(payload, dev_artifact)
        agreement: dict[str, Any] = {
            "artifact": str(dev_artifact),
            "acceptance_model_status": payload.get("acceptance_model_status"),
        }
        for key, cues in (("hybrid", True), ("hybrid_no_cues", False)):
            block = (payload.get("results") or {}).get(key)
            if not block:
                continue
            serving = harness.rows(REFERENCE_CONFIG, cues, label="B0r")
            emulated = accepted_sets(
                serving, shipped_probabilities(shipped, serving), shipped.threshold
            )
            by_case = {
                row["case_id"]: set(row.get("accepted_technique_ids") or ())
                for row in block.get("cases") or ()
            }
            compared = same = 0
            differing: list[str] = []
            for position, case_id in enumerate(serving.case_ids):
                if case_id not in by_case:
                    continue
                compared += 1
                if by_case[case_id] == emulated[position]:
                    same += 1
                elif len(differing) < 20:
                    differing.append(case_id)
            share = same / compared if compared else None
            agreement[key] = {
                "compared": compared,
                "missing_from_artifact": len(serving.case_ids) - compared,
                "agreeing": same,
                "share": share,
                "passed": share is not None and share >= P1B_AGREEMENT,
                "examples": differing,
            }
        report["shipped_on_serving_rows"] = agreement
    return report


def cmd_parity(args: argparse.Namespace, *, client: Any = None, **_: Any) -> int:
    from apt_classifier.vectorstore import ChromaVectorStore

    _guard_outputs([args.output], overwrite=args.overwrite)
    harness, clients = open_harness(
        args, cache_path=Path(args.cache), client=client, use_rows_cache=False
    )
    config = full_config(REFERENCE_CONFIG, {"dense_search": args.dense_source})
    settings = harness.settings_for(config)
    report: dict[str, Any] = run_header(harness, kind="tune_retrieval_parity")
    report["config"] = config
    blocking: dict[str, bool] = {}
    if args.live:
        if (
            args.dense_source == "copy"
            and settings.technique_collection_name != harness.cache.copy_name
        ):
            raise Refused(
                f"--dense-source copy needs TECHNIQUE_COLLECTION_NAME={harness.cache.copy_name} "
                "so the live store reads the same copy"
            )
        p1: dict[str, Any] = {}
        for cues in (True, False):
            cached = harness.rows(config, cues, label="P1-harness")
            started = time.time()
            live = build_rowset(
                harness.cache.cases,
                kb=harness.kb,
                settings=settings,
                store=ChromaVectorStore(settings, harness.cache.embedder, client=clients.get()),
                lexical_index=LexicalTechniqueIndex.from_kb(
                    harness.kb, fields=lexical_fields(settings)
                ),
                exemplar_index=harness.exemplar_index,
                cues_enabled=cues,
                generic=harness.generic,
            )
            difference = rowset_differences(cached, live)
            difference["live_seconds"] = round(time.time() - started, 1)
            difference["passed"] = difference["differing_cases"] == 0
            p1["cues_on" if cues else "cues_off"] = difference
        p1["passed"] = all(block["passed"] for block in p1.values() if isinstance(block, dict))
        report["P1"] = p1
        blocking["P1"] = p1["passed"]
        report["P2"] = p2_check(harness, harness.settings, clients.get())
        blocking["P2"] = bool(report["P2"]["passed"])
    else:
        report["P1"] = {"ran": False, "reason": "P1 compares against live Chroma: pass --live"}
        report["P2"] = {"ran": False, "reason": "pass --live"}
        blocking["P1"] = False
    shipped = Path(args.shipped_model or harness.settings.acceptance_model_path)
    report["P1b"] = p1b_check(harness, shipped_path=shipped, dev_artifact=args.dev_artifact)
    p3 = {
        ("cues_on" if cues else "cues_off"): p3_check(
            harness, config, cues=cues, thresholds=tuple(args.p3_thresholds)
        )
        for cues in (True, False)
    }
    p3["passed"] = all(block["passed"] for block in p3.values() if isinstance(block, dict))
    report["P3"] = p3
    blocking["P3"] = p3["passed"]
    report["blocking"] = blocking
    report["passed"] = all(blocking.values())
    if args.output:
        write_json(Path(args.output), report)
    print(format_parity(report))
    return 0 if report["passed"] else 1


def format_parity(report: Mapping[str, Any]) -> str:
    lines = ["parity:"]
    for name in ("P1", "P2", "P3"):
        block = report.get(name) or {}
        if block.get("ran") is False:
            lines.append(f"  {name}: not run ({block.get('reason')})")
            continue
        parts = []
        for key in ("cues_on", "cues_off"):
            if key in block:
                entry = block[key]
                count = entry.get("differing_cases", entry.get("mismatches"))
                parts.append(f"{key} {count} differing")
        if name == "P2":
            parts.append(f"{block.get('differences')} of {block.get('queries')} votes differ")
        lines.append(f"  {name}: {'pass' if block.get('passed') else 'FAIL'} ({', '.join(parts)})")
    p1b = report.get("P1b") or {}
    if p1b.get("ran"):
        old = p1b["old_fit_rows"]
        lines.append(
            f"  P1b: old-fit rows {old['rows']} (expected {old['rows_expected']}), positives "
            f"{old['positives']} (expected {old['positives_expected']}), in-sample P/R "
            f"{old['in_sample_at_threshold']['precision']:.4f}/"
            f"{old['in_sample_at_threshold']['recall']:.4f}"
        )
        for key, entry in (p1b.get("shipped_on_serving_rows") or {}).items():
            if isinstance(entry, dict) and "share" in entry:
                lines.append(f"  P1b {key}: R09 agreement {entry['share']}")
    lines.append(f"  overall: {'pass' if report.get('passed') else 'FAIL'}")
    return "\n".join(lines)


# ----- ann-recall -----
def ann_recall_report(harness: Harness) -> dict[str, Any]:
    cache = harness.cache
    depths = cache.ann_depths()
    formula = max(3 * 30, 4 * int(REFERENCE_CONFIG["candidate_pool"]))
    report: dict[str, Any] = {
        "segment_recall": [ann_segment_recall(cache, depth) for depth in depths],
        "depth_cap": depth_cap_share(cache, depths),
        "cases": [],
    }
    if "copy" in cache.dense_sources():
        report["copy_segment_recall"] = [
            ann_segment_recall(cache, depth, source="copy")
            for depth in cache.ann_depths(source="copy")
        ]
    for depth in depths:
        overfetch = 0 if depth == formula else depth
        ann_config = full_config(REFERENCE_CONFIG, {"dense_overfetch": overfetch})
        exact_config = full_config(ann_config, {"dense_search": "exact"})
        ann = harness.rows(ann_config, True, label=f"ann{depth}")
        exact = harness.rows(exact_config, True, label=f"exact{depth}")
        dense_differs = pool_differs = order_differs = 0
        for case in range(ann.case_count):
            a, b = ann.case_rows(case), exact.case_rows(case)
            dense_a = set(ann.technique_ids[a][ann.dense_present[a]].tolist())
            dense_b = set(exact.technique_ids[b][exact.dense_present[b]].tolist())
            dense_differs += int(dense_a != dense_b)
            pool_a = ann.technique_ids[a][np.argsort(ann.fused_rank[a], kind="stable")].tolist()
            pool_b = exact.technique_ids[b][np.argsort(exact.fused_rank[b], kind="stable")].tolist()
            pool_differs += int(set(pool_a) != set(pool_b))
            order_differs += int(pool_a != pool_b)
        count = max(ann.case_count, 1)
        report["cases"].append(
            {
                "depth": depth,
                "dense_overfetch": overfetch,
                "cases": ann.case_count,
                "dense_set_differs_share": dense_differs / count,
                "pool_set_differs_share": pool_differs / count,
                "pool_order_differs_share": order_differs / count,
            }
        )
    return report


def cmd_ann_recall(args: argparse.Namespace, *, client: Any = None, **_: Any) -> int:
    _guard_outputs([args.output], overwrite=args.overwrite)
    harness, _ = open_harness(args, cache_path=Path(args.cache), client=client)
    report = {**run_header(harness, kind="tune_retrieval_ann_recall"), **ann_recall_report(harness)}
    if args.output:
        write_json(Path(args.output), report)
    for entry in report["cases"]:
        print(
            f"depth {entry['depth']}: dense set differs in {entry['dense_set_differs_share']:.1%} "
            f"of cases, pool in {entry['pool_set_differs_share']:.1%}"
        )
    return 0


# ----- sweep -----
def checked_prereg(path: Path, *, repo_root: Path = REPO_ROOT) -> tuple[Prereg, dict[str, Any]]:
    """The pre-registration, refused unless its exemplar library fingerprint is filled in (not
    the placeholder), it is committed and clean in git and every commit after the registration
    commit is a declared amendment (within ``retry.allowed``)."""
    try:
        prereg = load_prereg(path)
    except (OSError, PreregError) as error:
        raise Refused(f"pre-registration {path}: {error}") from error
    if prereg.exemplar_library_placeholder:
        raise Refused(
            f"pre-registration {path}: fixed.exemplar_library.fingerprint is the placeholder "
            f"{prereg.exemplar_library['fingerprint']!r}; fill it from the sync --verify-only "
            "report (tune_retrieval.py prereg-fill) and commit the file first"
        )
    problems, identity = prereg_git_problems(path, repo_root=repo_root)
    if not problems:
        problems = amendment_problems(
            identity.get("history") or [], prereg.amendments, allowed=prereg.retry_allowed
        )
    if problems:
        raise Refused("pre-registration refused: " + "; ".join(problems))
    identity.update(
        {
            "sha256": prereg.sha256,
            "variants": prereg.variant_count,
            "amendments": [dict(entry) for entry in prereg.amendments],
        }
    )
    return prereg, identity


def e0_block(harness: Harness, prereg: Prereg, reference: Candidate, shipped_path: Path) -> dict:
    """E0 (no selection): B0-dev, the decomposition, ANN vs exact at the serving depth and the
    label-free fold-leak check."""
    seeds = prereg.selection_seeds
    block: dict[str, Any] = {"B0r": candidate_record(reference, prereg.reference)}
    shipped = AcceptanceModel.load(shipped_path) if shipped_path.is_file() else None
    if shipped is not None:
        b0_dev: dict[str, Any] = {
            "model": str(shipped_path),
            "threshold": shipped.threshold,
            "library": shipped_library(shipped, harness.exemplar_index),
        }
        for cues in (True, False):
            rows = harness.rows(prereg.reference, cues, label=prereg.reference_id)
            score = score_rows(
                rows,
                shipped_probabilities(shipped, rows),
                shipped.threshold,
                rank_by_acceptance=harness.rank_by_acceptance,
            )
            b0_dev["cues_on" if cues else "cues_off"] = score.metrics
        block["B0-dev"] = b0_dev
    else:
        block["B0-dev"] = {"ran": False, "reason": f"no model at {shipped_path}"}
    decomposition = []
    for item in prereg.decomposition:
        candidate = harness.evaluate(
            str(item["id"]),
            prereg.reference,
            seeds,
            fit_config=full_config(prereg.reference, item.get("fit_overrides") or {}),
            rule=str(item.get("threshold_rule", "runtime")),
        )
        decomposition.append(
            {
                "id": candidate.id,
                "fit_overrides": item.get("fit_overrides") or {},
                "threshold_rule": item.get("threshold_rule", "runtime"),
                "summary": candidate.summary,
                "delta_f1_vs_B0r": candidate.summary["f1"] - reference.summary["f1"],
            }
        )
    block["decomposition"] = decomposition
    ann = harness.rows(prereg.reference, True, label=prereg.reference_id)
    exact = harness.rows(
        full_config(prereg.reference, {"dense_search": "exact"}), True, label="exact"
    )
    dense_differs = pool_differs = 0
    for case in range(ann.case_count):
        a, b = ann.case_rows(case), exact.case_rows(case)
        dense_differs += int(
            set(ann.technique_ids[a][ann.dense_present[a]].tolist())
            != set(exact.technique_ids[b][exact.dense_present[b]].tolist())
        )
        pool_differs += int(
            set(ann.technique_ids[a].tolist()) != set(exact.technique_ids[b].tolist())
        )
    block["ann_vs_exact_serving_depth"] = {
        "cases": ann.case_count,
        "dense_set_differs_share": dense_differs / max(ann.case_count, 1),
        "pool_differs_share": pool_differs / max(ann.case_count, 1),
        "segment_recall": ann_segment_recall(harness.cache, min(harness.cache.ann_depths())),
    }
    library = harness.exemplar_index
    block["fold_leak"] = fold_leak_check(
        np.asarray(library.embeddings),
        list(library.sources),
        harness.cache.collection_embeddings(),
        harness.cache.manifest["collection_fields"],
    )
    return block


def nested_cv(
    harness: Harness, config: Mapping[str, Any], seeds: Sequence[int], *, inner_folds: int
) -> dict[str, Any]:
    """E4: for each outer fold the threshold comes from inner out-of-fold p on the training
    folds only; pooled F1 of the outer folds at their own thresholds."""
    rows = harness.rows(config, True, label="nested")
    groups = harness.groups
    per_seed = []
    for seed in seeds:
        outer = harness.folds(seed)
        probabilities = np.full(len(rows), np.nan)
        thresholds = np.full(len(rows), np.nan)
        fold_thresholds = {}
        for fold in sorted(set(outer.tolist())):
            train_cases = np.flatnonzero(outer != fold)
            held_cases = np.flatnonzero(outer == fold)
            train, _ = rows.subset(train_cases)
            inner = np.asarray(
                stratified_group_folds(
                    [harness.datasets[case] for case in train_cases.tolist()],
                    [int(groups[case]) for case in train_cases.tolist()],
                    n_splits=int(inner_folds),
                    seed=int(seed),
                )
            )
            p_inner, _ = fold_probabilities(
                train, inner, l2=float(config["l2"]), attack_version=harness.attack_version
            )
            point = choose_operating_point(
                train, p_inner, rule="runtime", min_precision=harness.min_precision
            )
            fold_thresholds[int(fold)] = float(point["threshold"])
            model = AcceptanceModel.fit(
                train.feature_dicts,
                train.labels.tolist(),
                attack_version=harness.attack_version,
                l2=float(config["l2"]),
            )
            held_mask = np.isin(rows.case_index, held_cases)
            for index in np.flatnonzero(held_mask).tolist():
                probabilities[index] = model.probability(rows.feature_dicts[index])
                thresholds[index] = point["threshold"]
        score = score_rows(
            rows, probabilities, thresholds, rank_by_acceptance=harness.rank_by_acceptance
        )
        per_seed.append(
            {
                "seed": int(seed),
                "fold_thresholds": fold_thresholds,
                "precision": score.metrics["precision"],
                "recall": score.metrics["recall"],
                "f1": score.metrics["f1"],
            }
        )
    return {
        "inner_folds": int(inner_folds),
        "per_seed": per_seed,
        "f1": float(np.mean([entry["f1"] for entry in per_seed])),
        "precision": float(np.mean([entry["precision"] for entry in per_seed])),
    }


def run_sweep(harness: Harness, prereg: Prereg, *, shipped_path: Path) -> dict[str, Any]:
    seeds = prereg.selection_seeds
    started = time.time()
    reference = harness.evaluate(prereg.reference_id, prereg.reference, seeds, curve_seed=seeds[0])
    e0 = e0_block(harness, prereg, reference, shipped_path)
    evaluated: dict[str, dict[str, Any]] = {}

    def evaluate(variant: Any, config: dict[str, Any]) -> Candidate:
        candidate = harness.evaluate(variant.id, config, seeds, curve_seed=seeds[0])
        evaluated[variant.id] = candidate_record(candidate, prereg.reference)
        return candidate

    path, reports = greedy_path(
        prereg.families,
        evaluate,
        reference,
        bar=prereg.bar,
        window=prereg.tie_window,
        clusters=harness.groups,
        bootstrap_samples=prereg.bootstrap_samples,
        bootstrap_seed=prereg.bootstrap_seed,
    )
    for report in reports:
        report["variants"] = [evaluated[name] for name in report["evaluated"]]
    finalist = path[-1]
    nested = {
        "finalist": nested_cv(
            harness, finalist.config, seeds, inner_folds=prereg.nested_inner_folds
        ),
    }
    if finalist is not reference:
        nested[prereg.reference_id] = nested_cv(
            harness, reference.config, seeds, inner_folds=prereg.nested_inner_folds
        )
    return {
        "e0": e0,
        "families": reports,
        "path": [
            {"step": step, "id": candidate.id, "config": candidate.config}
            for step, candidate in enumerate(path)
        ],
        "finalist": {
            **candidate_record(finalist, prereg.reference),
            "delta_f1_vs_reference": finalist.summary["f1"] - reference.summary["f1"],
            "requires_collection_copy": str(finalist.config["dense_search"]) != "ann",
            "e4_nested_cv": nested,
        },
        "counts": {
            "pre_registered_variants": prereg.variant_count,
            "evaluated_variants": len(evaluated),
            "identical_to_current_skipped": sum(
                len(report["identical_to_current"]) for report in reports
            ),
            "reference": 1,
            "e0_decomposition_configs": len(prereg.decomposition),
            "families": len(prereg.families),
            "adopted_steps": len(path) - 1,
            "row_sets_built": harness.rows_built,
        },
        "runtime_seconds": round(time.time() - started, 1),
    }


def cmd_sweep(
    args: argparse.Namespace, *, client: Any = None, repo_root: Path = REPO_ROOT, **_: Any
) -> int:
    _guard_outputs([args.output], overwrite=args.overwrite)
    prereg, identity = checked_prereg(Path(args.prereg), repo_root=repo_root)
    if tuple(args.seeds) != prereg.selection_seeds:
        raise Refused(
            f"--seeds {args.seeds} differ from the pre-registered {prereg.selection_seeds}"
        )
    if int(args.bootstrap) != prereg.bootstrap_samples:
        raise Refused(
            f"--bootstrap {args.bootstrap} differs from the pre-registered {prereg.bootstrap_samples}"
        )
    if args.parity is None:
        raise Refused(
            "--parity is required: the passing P1-P3 artifact (R21) of this cache; P1 must pass "
            "before any sweep"
        )
    parity = _read_json(Path(args.parity))
    harness, _ = open_harness(
        args,
        cache_path=Path(args.cache),
        prereg=prereg,
        client=client,
        use_rows_cache=not args.no_rows_cache,
    )
    problems = [
        *selection_problems(harness, prereg),
        *parity_problems(parity, harness, prereg, dense_search="ann"),
    ]
    if problems:
        raise Refused("sweep refused: " + "; ".join(problems))
    shipped = Path(args.shipped_model or harness.settings.acceptance_model_path)
    result = run_sweep(harness, prereg, shipped_path=shipped)
    payload = {
        **run_header(harness, kind="tune_retrieval_sweep"),
        "parity": _parity_reference(Path(args.parity), parity),
        "prereg": {**identity, "path": str(Path(args.prereg).resolve())},
        "seeds": list(prereg.selection_seeds),
        "bootstrap": {"samples": prereg.bootstrap_samples, "seed": prereg.bootstrap_seed},
        "reference_id": prereg.reference_id,
        **result,
    }
    write_json(Path(args.output), payload)
    print(format_sweep(payload))
    return 0


def format_sweep(payload: Mapping[str, Any]) -> str:
    reference = payload["e0"]["B0r"]["summary"]
    lines = [
        f"sweep: B0r F1 {reference['f1']:.4f} (P {reference['precision']:.3f}, "
        f"R {reference['recall']:.3f}) over seeds {payload['seeds']}"
    ]
    for report in payload["families"]:
        bar = report.get("bar") or {}
        delta = ((bar.get("checks") or {}).get("delta_f1") or {}).get("value")
        lines.append(
            f"  {report['family']}: best {report.get('chosen')} "
            + (f"dF1 {delta:+.4f} " if delta is not None else "")
            + ("ADOPTED" if report.get("adopted") else "not adopted")
        )
    finalist = payload["finalist"]
    lines.append(
        f"finalist {finalist['id']}: F1 {finalist['summary']['f1']:.4f} "
        f"(dF1 {finalist['delta_f1_vs_reference']:+.4f}); "
        f"{payload['counts']['evaluated_variants']} of {payload['counts']['pre_registered_variants']} "
        "variants evaluated"
    )
    return "\n".join(lines)


# ----- replicate -----
def same_run_problems(harness: Harness, earlier: Mapping[str, Any], what: str) -> list[str]:
    """A later step must run under the earlier step's base settings, exemplar library (as
    loaded: backend and search included) and harness code."""
    return [
        *record_problems(
            base_settings_record(harness.settings), earlier.get("base_settings"), what
        ),
        *(
            [
                f"the exemplar library loaded as {_jsonable(exemplar_signature(harness.exemplar_index))}, "
                f"{what} used {earlier.get('exemplars')}"
            ]
            if _jsonable(exemplar_signature(harness.exemplar_index)) != earlier.get("exemplars")
            else []
        ),
        *code_problems(harness.code, earlier.get("code"), what),
    ]


def t1_operating_points(harness: Harness, prereg: Prereg, selected: Candidate) -> dict[str, Any]:
    """T1 and T1-P40 thresholds: pooled out-of-fold p at the final-threshold seed (D8, D11)."""
    final_seed = int(prereg.payload.get("cv", {}).get("final_threshold_seed", 7))
    rows = harness.rows(selected.config, True, label=selected.id)
    p_oof, _ = fold_probabilities(
        rows,
        harness.folds(final_seed),
        l2=float(selected.config["l2"]),
        attack_version=harness.attack_version,
    )
    points: dict[str, Any] = {"seed": final_seed}
    for name, floor in (("T1", prereg.min_precision), ("T1-P40", prereg.p40_min_precision)):
        points[name] = choose_operating_point(rows, p_oof, rule="runtime", min_precision=floor)
    return points


def cmd_replicate(
    args: argparse.Namespace, *, client: Any = None, repo_root: Path = REPO_ROOT, **_: Any
) -> int:
    sweep_path = Path(args.sweep)
    sweep = _read_json(sweep_path)
    if sweep.get("kind") != "tune_retrieval_sweep":
        raise Refused(f"{sweep_path} is not a sweep artifact")
    env_path = Path(args.write_env) if args.write_env else None
    fit_args_path = env_path.with_suffix(".fit_args") if env_path is not None else None
    _guard_outputs([args.output, env_path, fit_args_path], overwrite=args.overwrite)
    prereg, identity = checked_prereg(Path(sweep["prereg"]["path"]), repo_root=repo_root)
    if identity["sha256"] != sweep["prereg"]["sha256"]:
        raise Refused("the pre-registration changed since the sweep")
    for key in ("commit", "registered_commit"):
        if identity.get(key) != sweep["prereg"].get(key):
            raise Refused(
                f"the pre-registration's {key.replace('_', ' ')} is {identity.get(key)}, the "
                f"sweep ran on {sweep['prereg'].get(key)}"
            )
    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != prereg.replication.seeds:
        raise Refused(
            f"--seeds {list(seeds)} differ from the pre-registered {prereg.replication.seeds}"
        )
    cache_path = Path(args.cache or sweep["cache"]["path"])
    harness, _ = open_harness(args, cache_path=cache_path, prereg=prereg, client=client)
    if harness.cache.key != sweep["cache"]["key"]:
        raise Refused("the cache differs from the one the sweep used")
    problems = [
        *selection_problems(harness, prereg),
        *same_run_problems(harness, sweep, "the sweep"),
    ]
    if problems:
        raise Refused("replicate refused: " + "; ".join(problems))
    path_entries = sweep["path"]
    evaluated = [
        harness.evaluate(entry["id"], entry["config"], seeds, with_off=False)
        for entry in path_entries
    ]
    reference = evaluated[0]
    selected, steps = replicate_step_back(
        evaluated,
        reference,
        rule=prereg.replication,
        clusters=harness.groups,
        bootstrap_samples=prereg.bootstrap_samples,
        bootstrap_seed=prereg.bootstrap_seed,
    )
    operating_points = t1_operating_points(harness, prereg, selected)
    env_lines: list[str] | None = None
    env_problem = None
    try:
        env_lines = env_assignments(
            selected.config,
            fixed=prereg.fixed_settings,
            exemplar_library=library_env(harness.exemplar_index),
        )
    except ValueError as error:
        env_problem = str(error)
    payload = {
        **run_header(harness, kind="tune_retrieval_replicate"),
        "sweep": {"path": str(sweep_path), "artifact_hash": sweep.get("artifact_hash")},
        "prereg": {**identity, "path": sweep["prereg"]["path"]},
        "seeds": list(seeds),
        "rule": {
            "min_delta_f1_per_seed": prereg.replication.min_delta_f1_per_seed,
            "pooled_ci_lower_above": prereg.replication.pooled_ci_lower_above,
            "bootstrap": {"samples": prereg.bootstrap_samples, "seed": prereg.bootstrap_seed},
        },
        "path": [candidate_record(candidate, prereg.reference) for candidate in evaluated],
        "steps": steps,
        "selected": {
            "id": selected.id,
            "config": selected.config,
            "is_reference": selected is reference,
            "env": env_lines,
            "env_problem": env_problem,
            "fit_args": fit_arguments(selected.config),
            "requires_collection_copy": str(selected.config["dense_search"]) != "ann",
            "operating_points": operating_points,
        },
        "counts": {
            **sweep.get("counts", {}),
            "path_length": len(path_entries),
            "replication_evaluations": len(evaluated),
        },
        "t1_env": str(env_path) if env_path is not None and env_lines is not None else None,
    }
    write_json(Path(args.output), payload)
    if env_path is not None and env_lines is not None:
        write_env(env_path, env_lines)
        write_env(fit_args_path, [" ".join(fit_arguments(selected.config))])
        _log(f"wrote {env_path} and {fit_args_path}")
    print(
        f"replicate: selected {selected.id} "
        + ("(the reference: nothing replicated)" if selected is reference else "")
    )
    if env_problem:
        print(f"T1.env not written: {env_problem}", file=sys.stderr)
        return 1
    return 0


# ----- copy-check (Phase 5b, only after exact dense search was selected) -----
def served_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """What serves a configuration: exact dense search runs only from a collection copy."""
    config = full_config(config)
    if str(config["dense_search"]) == "exact":
        config["dense_search"] = "copy"
    return config


COPY_CHECK_EXCLUDED_MATERIAL = ("collection_copy", "lists", "git_commit")


def same_inputs_problems(copy_cache: DevCache, sweep_cache: DevCache) -> list[str]:
    """The copy cache must hold the sweep cache's inputs (cases, texts, embeddings, the
    technique collection, the library) and a collection copy."""
    if copy_cache.copy_name is None:
        return [
            f"{copy_cache.path} holds no collection copy (build it with cache "
            "--technique-collection-copy)"
        ]
    mine, theirs = copy_cache.material, sweep_cache.material
    return [
        f"the copy cache's {key} differs from the sweep cache's"
        for key in sorted(set(mine) | set(theirs))
        if key not in COPY_CHECK_EXCLUDED_MATERIAL and mine.get(key) != theirs.get(key)
    ]


def rows_agreement(left: RowSet, right: RowSet, *, atol: float = CHECK_FIT_FEATURE_ATOL) -> dict:
    """Per case: equal technique ids, labels and guards, and features within ``atol``."""
    within = bitwise = 0
    examples = []
    for case in range(left.case_count):
        a, b = left.case_rows(case), right.case_rows(case)
        same_rows = (
            len(a) == len(b)
            and np.array_equal(left.technique_ids[a], right.technique_ids[b])
            and np.array_equal(left.labels[a], right.labels[b])
            and all(np.array_equal(left.flags[n][a], right.flags[n][b]) for n in left.flags)
        )
        close = same_rows and bool(np.all(np.abs(left.features[a] - right.features[b]) <= atol))
        within += int(close)
        bitwise += int(
            same_rows
            and np.array_equal(left.features[a].view(np.uint64), right.features[b].view(np.uint64))
        )
        if not close and len(examples) < 20:
            examples.append(left.case_ids[case])
    count = max(left.case_count, 1)
    return {
        "cases": left.case_count,
        "within_tolerance_share": within / count,
        "bitwise_share": bitwise / count,
        "feature_atol": atol,
        "outside_tolerance_examples": examples,
    }


def cmd_copy_check(
    args: argparse.Namespace, *, client: Any = None, repo_root: Path = REPO_ROOT, **_: Any
) -> int:
    """E2a served from the copy: the replication step-back again on the copy's hits."""
    replicate_path = Path(args.replicate)
    replicate = _read_json(replicate_path)
    if replicate.get("kind") != "tune_retrieval_replicate":
        raise Refused(f"{replicate_path} is not a replicate artifact")
    if str((replicate.get("selected") or {}).get("config", {}).get("dense_search")) != "exact":
        raise Refused(
            "the replicate step did not select exact dense search: there is nothing to serve "
            "from a collection copy"
        )
    env_path = Path(args.write_env) if args.write_env else None
    fit_args_path = env_path.with_suffix(".fit_args") if env_path is not None else None
    _guard_outputs([args.output, env_path, fit_args_path], overwrite=args.overwrite)
    prereg, identity = checked_prereg(Path(replicate["prereg"]["path"]), repo_root=repo_root)
    for key in ("sha256", "commit", "registered_commit"):
        if identity.get(key) != replicate["prereg"].get(key):
            raise Refused(f"the pre-registration's {key} changed since the replicate step")
    if args.parity is None:
        raise Refused(
            "--parity is required: a passing parity --dense-source copy run of this cache"
        )
    parity = _read_json(Path(args.parity))
    sweep_cache = DevCache.load(Path(replicate["cache"]["path"]))
    if sweep_cache.key != replicate["cache"]["key"]:
        raise Refused("the sweep's cache changed since the replicate step")
    harness, _ = open_harness(args, cache_path=Path(args.cache), prereg=prereg, client=client)
    problems = [
        *same_inputs_problems(harness.cache, sweep_cache),
        *selection_problems(harness, prereg),
        *same_run_problems(harness, replicate, "the replicate step"),
        *parity_problems(parity, harness, prereg, dense_search="copy"),
    ]
    if problems:
        raise Refused("copy-check refused: " + "; ".join(problems))
    seeds = tuple(int(seed) for seed in prereg.replication.seeds)
    entries = replicate["path"]
    served = [
        harness.evaluate(entry["id"], served_config(entry["config"]), seeds, with_off=False)
        for entry in entries
    ]
    reference = served[0]
    selected, steps = replicate_step_back(
        served,
        reference,
        rule=prereg.replication,
        clusters=harness.groups,
        bootstrap_samples=prereg.bootstrap_samples,
        bootstrap_seed=prereg.bootstrap_seed,
    )
    replicated_id = str(replicate["selected"]["id"])
    same = selected.id == replicated_id
    exact_config = full_config(replicate["selected"]["config"])
    agreement: dict[str, Any] = {}
    for cues in (True, False):
        exact_rows = harness.rows(exact_config, cues, label=f"{replicated_id}/exact")
        copy_rows = harness.rows(served_config(exact_config), cues, label=f"{replicated_id}/copy")
        agreement["cues_on" if cues else "cues_off"] = {
            **rowset_differences(exact_rows, copy_rows),
            **rows_agreement(exact_rows, copy_rows),
        }
    exact_entry = next((item for item in entries if item["id"] == replicated_id), {})
    copy_entry = next((item for item in served if item.id == replicated_id), None)
    agreement["f1_per_seed"] = {
        "exact": [s["f1"] for s in (exact_entry.get("summary") or {}).get("per_seed", [])],
        "copy": [s["f1"] for s in copy_entry.summary["per_seed"]] if copy_entry else [],
    }
    operating_points = t1_operating_points(harness, prereg, selected)
    copy_name = str(harness.cache.copy_name)
    env_lines = env_assignments(
        selected.config,
        technique_collection=copy_name,
        fixed=prereg.fixed_settings,
        exemplar_library=library_env(harness.exemplar_index),
    )
    payload = {
        **run_header(harness, kind="tune_retrieval_copy_check"),
        "replicate": {
            "path": str(replicate_path),
            "artifact_hash": replicate.get("artifact_hash"),
            "selected": replicated_id,
        },
        "sweep_cache": {"path": str(sweep_cache.path.resolve()), "key": sweep_cache.key},
        "parity": _parity_reference(Path(args.parity), parity),
        "prereg": {**identity, "path": replicate["prereg"]["path"]},
        "seeds": list(seeds),
        "path": [candidate_record(candidate, prereg.reference) for candidate in served],
        "steps": steps,
        "same_as_replicate": same,
        "selected": {
            "id": selected.id,
            "config": selected.config,
            "is_reference": selected is reference,
            "env": env_lines,
            "fit_args": fit_arguments(selected.config),
            "collection_copy": copy_name if selected.config["dense_search"] == "copy" else None,
            "operating_points": operating_points,
            "replicate_operating_points": replicate["selected"].get("operating_points"),
        },
        "exact_vs_copy": agreement,
        "t1_env": str(env_path) if env_path is not None else None,
    }
    write_json(Path(args.output), payload)
    if env_path is not None:
        write_env(env_path, env_lines)
        write_env(fit_args_path, [" ".join(fit_arguments(selected.config))])
        _log(f"wrote {env_path} and {fit_args_path}")
    print(
        f"copy-check: served from {copy_name}, selected {selected.id}"
        + (
            " (as the replicate step)"
            if same
            else f" (the replicate step selected {replicated_id})"
        )
        + f"; exact vs copy rows within tolerance on "
        f"{agreement['cues_on']['within_tolerance_share']:.1%} of cases"
    )
    return 0


# ----- rethreshold -----
def _dump_arrays(path: Path) -> dict[str, np.ndarray]:
    require_dev_path(path, "--rows")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    datasets = {str(item) for item in arrays.get("datasets", np.zeros(0)).tolist()}
    unknown = sorted(datasets - set(DEV_DATASETS))
    if unknown:
        raise DevOnlyError(f"{path}: rows from {', '.join(unknown)}; dev datasets only")
    return arrays


def _dev_model(path: Path) -> AcceptanceModel:
    require_dev_path(path, "--model")
    model = AcceptanceModel.load(path)
    if model is None:
        raise Refused(f"no acceptance model at {path}")
    if (model.metadata or {}).get("split") != "dev":
        raise DevOnlyError(f"{path}: fitted on split {(model.metadata or {}).get('split')!r}")
    return model


def _dump_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
    if "summary" not in arrays:
        return None
    try:
        value = json.loads(str(arrays["summary"]))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def dump_model_problems(
    model: AcceptanceModel, arrays: Mapping[str, np.ndarray], *, rows_sha256: str | None = None
) -> list[str]:
    """Why the dumped rows are not the ones ``model``'s fit wrote: the dump's summary records
    the fit's threshold and retrieval contract (``fit_acceptance_model.py --dump-rows``), and
    the model records its row and positive counts (and the dump's sha256 when the fit records
    it, ``metadata.dump_rows.sha256``)."""
    problems = []
    recorded = (dict(model.metadata or {}).get("dump_rows") or {}).get("sha256")
    if recorded and rows_sha256 and recorded != rows_sha256:
        problems.append("the model records another dump (dump_rows.sha256 differs)")
    names = arrays.get("feature_names")
    if names is None or tuple(str(item) for item in names.tolist()) != FEATURE_NAMES:
        problems.append("the dump's feature names are not this schema's")
    summary = _dump_summary(arrays)
    if summary is None:
        return [*problems, "the dump records no fit summary; it cannot be tied to the model"]
    if "threshold" not in summary:
        problems.append("the dump records no threshold")
    elif abs(float(summary["threshold"]) - float(model.threshold)) > 1e-12:
        problems.append(
            f"the dump was written by a fit with threshold {float(summary['threshold']):.4f}, "
            f"the model's is {float(model.threshold):.4f}"
        )
    contract = summary.get("retrieval_contract")
    if not isinstance(contract, Mapping):
        problems.append("the dump records no retrieval contract")
    elif model.contract is None or model.contract_origin != "artifact":
        problems.append("the model stores no retrieval contract")
    elif canonical_json(dict(contract)) != canonical_json(dict(model.contract)):
        problems.append("the dump's retrieval contract differs from the model's")
    metadata = dict(model.metadata or {})
    labels = np.asarray(arrays["labels"])
    for key, value in (("row_count", len(labels)), ("positive_rows", int(labels.sum()))):
        if metadata.get(key) is None:
            problems.append(f"the model records no {key}")
        elif int(metadata[key]) != int(value):
            problems.append(f"the model records {key} {metadata[key]}, the dump holds {value}")
    return problems


def cmd_rethreshold(args: argparse.Namespace, **_: Any) -> int:
    require_dev_path(args.rows, "--rows")
    require_dev_path(args.model, "--model")
    _guard_outputs([args.output], overwrite=args.overwrite)
    model = _dev_model(Path(args.model))
    arrays = _dump_arrays(Path(args.rows))
    problems = dump_model_problems(model, arrays, rows_sha256=file_sha256(args.rows))
    if problems:
        raise Refused("the rows do not belong to the model: " + "; ".join(problems))
    probabilities = np.asarray(arrays["p_out_of_fold"], dtype=np.float64)
    if not len(probabilities) or np.isnan(probabilities).any():
        raise Refused(
            "the dump holds no out-of-fold probabilities (refit with --threshold-selection "
            "oof_runtime)"
        )
    expected = dict(
        zip(
            [str(item) for item in arrays["expected_case_ids"].tolist()],
            [int(item) for item in arrays["expected_counts"].tolist()],
            strict=True,
        )
    )
    arguments = {
        "blocked": arrays["blocked"].astype(bool),
        "forced": arrays["forced"].astype(bool),
        "technique_ids": [str(item) for item in arrays["technique_ids"].tolist()],
    }
    case_ids = [str(item) for item in arrays["case_ids"].tolist()]
    labels = arrays["labels"].astype(np.int64)
    point = choose_threshold_runtime(
        probabilities, labels, case_ids, expected, min_precision=args.min_precision, **arguments
    )
    own_floor = float((model.metadata or {}).get("min_precision", 0.5))
    own = choose_threshold_runtime(
        probabilities, labels, case_ids, expected, min_precision=own_floor, **arguments
    )
    clone = AcceptanceModel.from_dict(model.to_dict())
    clone.threshold = float(point["threshold"])
    clone.metadata = {
        **dict(model.metadata or {}),
        "dev_operating_point": point,
        "rethreshold": {
            "min_precision": args.min_precision,
            "source_model": str(args.model),
            "source_sha256": file_sha256(args.model),
            "source_threshold": model.threshold,
            "rows": str(args.rows),
            "rows_sha256": file_sha256(args.rows),
            "self_check": {
                "min_precision": own_floor,
                "threshold": own["threshold"],
                "matches_source": abs(float(own["threshold"]) - float(model.threshold)) < 1e-9,
            },
        },
    }
    same = all(
        np.array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))
        for a, b in (
            (clone.weights, model.weights),
            (clone.means, model.means),
            (clone.scales, model.scales),
            ([clone.intercept], [model.intercept]),
        )
    )
    if not same:
        raise Refused("the re-thresholded model's weights differ from the source (refusing)")
    matches = bool(clone.metadata["rethreshold"]["self_check"]["matches_source"])
    summary = (
        f"rethreshold: P >= {args.min_precision:.2f} -> threshold {point['threshold']:.2f} "
        f"(P {point['precision']:.3f}, R {point['recall']:.3f}); source threshold "
        f"{model.threshold:.2f}, self-check {'ok' if matches else 'differs'}"
    )
    if not matches:
        # the dump's probabilities do not reproduce the model's own threshold at its own floor:
        # a threshold chosen on them is not T1's operating point moved to P >= floor
        print(
            f"{summary}\nself-check failed: the rows give {float(own['threshold']):.2f} at "
            f"P >= {own_floor:.2f}, the model has {float(model.threshold):.2f}; nothing written",
            file=sys.stderr,
        )
        return 1
    clone.save(args.output)
    print(summary)
    return 0


# ----- check-fit -----
def _selected_config(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if payload.get("kind") in ("tune_retrieval_replicate", "tune_retrieval_copy_check"):
        return str(payload["selected"]["id"]), dict(payload["selected"]["config"])
    if payload.get("kind") == "tune_retrieval_sweep":
        return str(payload["finalist"]["id"]), dict(payload["finalist"]["config"])
    raise Refused("--sweep must be a sweep, replicate or copy-check artifact")


def _artifact_prereg(payload: Mapping[str, Any]) -> Prereg:
    """The pre-registration a selection artifact ran under, unchanged since (its sha256)."""
    info = payload.get("prereg") or {}
    if not info.get("path"):
        raise Refused("the selection artifact names no pre-registration")
    try:
        prereg = load_prereg(info["path"])
    except (OSError, PreregError) as error:
        raise Refused(f"pre-registration {info['path']}: {error}") from error
    if prereg.sha256 != info.get("sha256"):
        raise Refused("the pre-registration changed since the selection artifact was written")
    return prereg


def _accepted_by_case(
    case_ids: Sequence[str],
    technique_ids: Sequence[str],
    probabilities: np.ndarray,
    blocked: np.ndarray,
    forced: np.ndarray,
    threshold: float,
) -> dict[str, set[str]]:
    accepted = np.asarray(forced, dtype=bool) | (
        ~np.asarray(blocked, dtype=bool) & (np.asarray(probabilities) >= float(threshold))
    )
    output: dict[str, set[str]] = {}
    for row in np.flatnonzero(accepted).tolist():
        output.setdefault(str(case_ids[row]), set()).add(str(technique_ids[row]))
    return output


def cmd_check_fit(args: argparse.Namespace, *, client: Any = None, **_: Any) -> int:
    _guard_outputs([args.output], overwrite=args.overwrite)
    payload = _read_json(Path(args.sweep))
    selected_id, config = _selected_config(payload)
    config = full_config(config)
    if str(config["dense_search"]) == "exact":
        raise Refused(
            "exact dense search is served only from a collection copy, which is what the fit "
            "read: run copy-check after Phase 5b and pass its artifact as --sweep"
        )
    prereg = _artifact_prereg(payload)
    cache_path = Path(args.cache or payload["cache"]["path"])
    harness, _ = open_harness(args, cache_path=cache_path, prereg=prereg, client=client)
    if harness.cache.key != (payload.get("cache") or {}).get("key"):
        raise Refused("the cache differs from the one the selection artifact used")
    problems = [
        *record_problems(
            base_settings_record(harness.settings),
            payload.get("base_settings"),
            "the selection artifact",
        ),
        *code_problems(harness.code, payload.get("code"), "the selection artifact", ROW_CODE_KEYS),
    ]
    if problems:
        raise Refused("check-fit refused: " + "; ".join(problems))
    model = _dev_model(Path(args.model))
    arrays = _dump_arrays(Path(args.rows))
    link = dump_model_problems(model, arrays, rows_sha256=file_sha256(args.rows))
    if link:
        raise Refused("the rows do not belong to the model: " + "; ".join(link))
    rows = harness.rows(config, True, label=selected_id)

    def keyed(case_ids: Sequence[str], technique_ids: Sequence[str]) -> list[tuple[str, str, int]]:
        seen: dict[tuple[str, str], int] = {}
        output = []
        for case_id, technique_id in zip(case_ids, technique_ids, strict=True):
            pair = (str(case_id), str(technique_id))
            seen[pair] = seen.get(pair, 0) + 1
            output.append((*pair, seen[pair]))
        return output

    dump_keys = keyed(arrays["case_ids"].tolist(), arrays["technique_ids"].tolist())
    harness_keys = keyed(rows.row_case_ids, rows.technique_ids.tolist())
    harness_index = {key: position for position, key in enumerate(harness_keys)}
    features = np.ascontiguousarray(arrays["features"], dtype=np.float64)
    blocked, forced = rows.gate_masks()
    equal = bitwise = 0
    largest_gap = 0.0
    examples = []
    for position, key in enumerate(dump_keys):
        other = harness_index.get(key)
        if other is None:
            if len(examples) < 20:
                examples.append({"row": key, "why": "not in the harness rows"})
            continue
        guards = (
            int(arrays["labels"][position]) == int(rows.labels[other])
            and bool(arrays["blocked"][position]) == bool(blocked[other])
            and bool(arrays["forced"][position]) == bool(forced[other])
        )
        feature_gap = float(np.max(np.abs(features[position] - rows.features[other])))
        close = guards and feature_gap <= CHECK_FIT_FEATURE_ATOL  # a NaN gap is never close
        if guards and math.isfinite(feature_gap):
            largest_gap = max(largest_gap, feature_gap)
        equal += int(close)
        bitwise += int(
            guards
            and np.array_equal(
                features[position].view(np.uint64), rows.features[other].view(np.uint64)
            )
        )
        if not close and len(examples) < 20:
            why = f"features differ by {feature_gap:.3g}" if guards else "label or guards differ"
            examples.append({"row": key, "why": why})
    denominator = max(len(dump_keys), len(harness_keys), 1)
    share = equal / denominator
    # the accepted set per case under the model, from either side's rows
    dump_matrix = np.ascontiguousarray(features, dtype=np.float64)
    dump_p = np.asarray(
        [
            model.probability(tuning_harness.RowView(dump_matrix, row))
            for row in range(len(dump_matrix))
        ],
        dtype=np.float64,
    )
    harness_p = np.asarray([model.probability(row) for row in rows.feature_dicts], dtype=np.float64)
    dump_sets = _accepted_by_case(
        arrays["case_ids"].tolist(),
        arrays["technique_ids"].tolist(),
        dump_p,
        arrays["blocked"],
        arrays["forced"],
        model.threshold,
    )
    harness_sets = _accepted_by_case(
        rows.row_case_ids, rows.technique_ids.tolist(), harness_p, blocked, forced, model.threshold
    )
    agreeing = sum(
        int(dump_sets.get(case_id, set()) == harness_sets.get(case_id, set()))
        for case_id in rows.case_ids
    )
    case_share = agreeing / max(rows.case_count, 1)
    settings = harness.settings_for(config)
    harness_contract = retrieval_contract_for(
        settings,
        harness.exemplar_index,
        store=harness.cache.technique_store(str(config["dense_search"])),
    )
    contract_issues = (
        contract_mismatches(model.contract, harness_contract, exemplars=model.requires_exemplars)
        if model.contract is not None
        else ["the model stores no retrieval contract"]
    )
    refit = AcceptanceModel.fit(
        rows.feature_dicts,
        rows.labels.tolist(),
        attack_version=harness.attack_version,
        l2=float(config["l2"]),
    )
    a, b = np.asarray(model.weights, dtype=np.float64), np.asarray(refit.weights, dtype=np.float64)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if a.any() and b.any() else 0.0
    final_seed = int(((payload.get("selected") or {}).get("operating_points") or {}).get("seed", 7))
    p_oof, _ = fold_probabilities(
        rows,
        harness.folds(final_seed),
        l2=float(config["l2"]),
        attack_version=harness.attack_version,
    )
    point = choose_operating_point(rows, p_oof, rule="runtime", min_precision=harness.min_precision)
    gap = abs(float(model.threshold) - float(point["threshold"]))
    checks = {
        "rows_equal_share": {
            "value": share,
            "required": (
                f">= {CHECK_FIT_ROW_SHARE} (technique, label and guards equal, features within "
                f"{CHECK_FIT_FEATURE_ATOL:g})"
            ),
            "passed": share >= CHECK_FIT_ROW_SHARE,
        },
        "accepted_sets_share": {
            "value": case_share,
            "required": f">= {CHECK_FIT_CASE_SHARE} of cases (the model at its threshold)",
            "passed": case_share >= CHECK_FIT_CASE_SHARE,
        },
        "retrieval_contract": {
            "value": contract_issues,
            "required": "the model's contract met by the harness configuration",
            "passed": not contract_issues,
        },
        "weight_cosine": {
            "value": cosine,
            "required": f">= {CHECK_FIT_WEIGHT_COSINE}",
            "passed": cosine >= CHECK_FIT_WEIGHT_COSINE,
        },
        "threshold_gap": {
            "value": gap,
            "required": f"<= {CHECK_FIT_THRESHOLD}",
            "passed": gap <= CHECK_FIT_THRESHOLD + 1e-12,
        },
        "no_priors": {
            "value": len(model.technique_priors),
            "required": "0 (--priors-split none)",
            "passed": not model.technique_priors,
        },
    }
    report = {
        **run_header(harness, kind="tune_retrieval_check_fit"),
        "selected": {"id": selected_id, "config": config},
        "model": {
            "path": str(args.model),
            "threshold": model.threshold,
            "sha256": file_sha256(args.model),
        },
        "rows": {
            "dump": len(dump_keys),
            "harness": len(harness_keys),
            "equal": equal,
            "bitwise_equal": bitwise,
            "bitwise_share": bitwise / denominator,
            "largest_feature_gap": largest_gap,
            "feature_atol": CHECK_FIT_FEATURE_ATOL,
            "examples": examples,
        },
        "cases": {"compared": rows.case_count, "accepted_sets_agreeing": agreeing},
        "harness_threshold": point,
        "checks": checks,
        "passed": all(entry["passed"] for entry in checks.values()),
    }
    if args.output:
        write_json(Path(args.output), report)
    for name, entry in checks.items():
        print(
            f"{name}: {entry['value']} ({entry['required']}) {'ok' if entry['passed'] else 'FAIL'}"
        )
    return 0 if report["passed"] else 1


# ----- report -----
# ----- prereg-fill -----
def cmd_prereg_fill(args: argparse.Namespace, **_: Any) -> int:
    """Write the Chroma library fingerprint of a passing sync --verify-only report into the
    pre-registration's placeholder (``fixed.exemplar_library.fingerprint``); the file must then
    be committed (the registration commit) before the sweep."""
    try:
        report = json.loads(Path(args.sync_report).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {args.sync_report}: {error}") from error
    fingerprint = fill_exemplar_library(Path(args.prereg), report)
    _log(
        f"fixed.exemplar_library.fingerprint = {fingerprint} written to {args.prereg}; commit "
        "the pre-registration before the sweep"
    )
    print(fingerprint)
    return 0


def cmd_report(args: argparse.Namespace, **_: Any) -> int:
    _guard_outputs([args.output], overwrite=args.overwrite)
    summary: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    lines: list[str] = []
    if args.parity:
        payload = _read_json(Path(args.parity))
        lines.append(format_parity(payload))
        summary["parity"] = {"passed": payload.get("passed"), "blocking": payload.get("blocking")}
    if args.ann_recall:
        payload = _read_json(Path(args.ann_recall))
        for entry in payload.get("cases", []):
            lines.append(
                f"ann-recall depth {entry['depth']}: dense set differs "
                f"{entry['dense_set_differs_share']:.1%}, pool {entry['pool_set_differs_share']:.1%}"
            )
        summary["ann_recall"] = payload.get("cases")
    if args.sweep:
        payload = _read_json(Path(args.sweep))
        lines.append(format_sweep(payload))
        summary["sweep"] = {
            "finalist": payload["finalist"]["id"],
            "delta_f1": payload["finalist"]["delta_f1_vs_reference"],
            "counts": payload["counts"],
            "prereg_commit": payload["prereg"].get("commit"),
        }
    if args.replicate:
        payload = _read_json(Path(args.replicate))
        selected = payload["selected"]
        lines.append(
            f"replicate: selected {selected['id']}"
            + (" (reference)" if selected.get("is_reference") else "")
            + f"; T1 threshold {selected['operating_points']['T1']['threshold']:.2f}, "
            f"T1-P40 {selected['operating_points']['T1-P40']['threshold']:.2f}"
        )
        for step in payload.get("steps", []):
            lines.append(
                f"  {step['id']}: dF1 per seed {[f'{value:+.4f}' for value in step['delta_f1_per_seed']]}, "
                f"CI low {step['bootstrap']['ci95'][0]:+.4f} -> "
                + ("replicates" if step["replicates"] else "does not replicate")
            )
        summary["replicate"] = {"selected": selected["id"], "env": selected.get("env")}
    if args.copy_check:
        payload = _read_json(Path(args.copy_check))
        selected = payload["selected"]
        agreement = (payload.get("exact_vs_copy") or {}).get("cues_on") or {}
        lines.append(
            f"copy-check: selected {selected['id']}"
            + (" (as replicate)" if payload.get("same_as_replicate") else " (replicate differed)")
            + f"; exact vs copy rows within tolerance on "
            f"{float(agreement.get('within_tolerance_share', 0.0)):.1%} of cases"
        )
        summary["copy_check"] = {
            "selected": selected["id"],
            "same_as_replicate": payload.get("same_as_replicate"),
            "env": selected.get("env"),
        }
    print("\n".join(lines) if lines else "nothing to report (pass --parity/--sweep/...)")
    if args.output:
        write_json(Path(args.output), summary)
    return 0


# ----- CLI -----
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, *, cache: bool = True, output: bool = True) -> None:
        if cache:
            sub.add_argument("--cache", type=Path, required=True, help="the dev cache directory")
        if output:
            sub.add_argument("--output", type=Path, help="JSON artifact to write")
            sub.add_argument("--overwrite", action="store_true", help="replace an existing output")
        add_exemplar_backend_arguments(sub)

    cache = commands.add_parser("cache", help="Stage A: embed and read Chroma once")
    cache.add_argument("--output", type=Path, required=True, help="new cache directory")
    cache.add_argument(
        "--collection", help="the Chroma exemplar collection the library must load from"
    )
    cache.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    cache.add_argument("--datasets", nargs="+", default=list(DEV_DATASETS))
    cache.add_argument("--dense-ann-depths", type=int, nargs="+", default=[90, 150, 300])
    cache.add_argument(
        "--synthetic-ann-depths",
        type=int,
        nargs="*",
        default=[90],
        help="ANN depths with the synthetic sentences in the filter (P1b and E0 need 90)",
    )
    cache.add_argument("--exact-depth", type=int, default=400)
    cache.add_argument("--lexical-depth", type=int, default=400)
    cache.add_argument(
        "--technique-collection-copy", help="also cache this collection copy's ANN hits"
    )
    cache.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    add_exemplar_backend_arguments(cache)

    parity = commands.add_parser("parity", help="P1-P3 (P1 and P2 need --live)")
    common(parity)
    parity.add_argument("--live", action="store_true", help="compare against live Chroma (P1, P2)")
    parity.add_argument(
        "--dev-artifact", type=Path, help="a benchmark_mapping.py --split dev run (R09)"
    )
    parity.add_argument("--dense-source", choices=("ann", "copy"), default="ann")
    parity.add_argument("--shipped-model", type=Path, help="default: ACCEPTANCE_MODEL_PATH")
    parity.add_argument(
        "--p3-thresholds", type=float, nargs="+", default=list(DEFAULT_P3_THRESHOLDS)
    )

    ann = commands.add_parser("ann-recall", help="ANN against exact dense search")
    common(ann)

    sweep = commands.add_parser("sweep", help="E0-E4 under the pre-registered rule")
    common(sweep)
    sweep.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    sweep.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    sweep.add_argument("--bootstrap", type=int, default=1000)
    sweep.add_argument(
        "--shipped-model", type=Path, help="B0-dev model (default: ACCEPTANCE_MODEL_PATH)"
    )
    sweep.add_argument("--no-rows-cache", action="store_true", help="rebuild every row set")
    sweep.add_argument(
        "--parity",
        type=Path,
        help="the passing parity --live artifact of this cache (R21); required",
    )

    replicate = commands.add_parser("replicate", help="fresh seeds, step back, write T1.env")
    replicate.add_argument("--sweep", type=Path, required=True)
    replicate.add_argument("--seeds", type=int, nargs="+", default=[41, 53])
    replicate.add_argument("--output", type=Path, required=True)
    replicate.add_argument(
        "--write-env", type=Path, help="T1.env (T1.fit_args is written beside it)"
    )
    replicate.add_argument("--cache", type=Path, help="default: the sweep's cache")
    replicate.add_argument("--overwrite", action="store_true")
    add_exemplar_backend_arguments(replicate)

    copy_check = commands.add_parser(
        "copy-check", help="exact search selected: replicate again on the collection copy's hits"
    )
    copy_check.add_argument("--replicate", type=Path, required=True, help="R24 replicate artifact")
    copy_check.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="a cache built with cache --technique-collection-copy (the sweep's inputs otherwise)",
    )
    copy_check.add_argument(
        "--parity", type=Path, help="the passing parity --live --dense-source copy artifact"
    )
    copy_check.add_argument("--output", type=Path, required=True)
    copy_check.add_argument(
        "--write-env", type=Path, help="T1.env (T1.fit_args is written beside it)"
    )
    copy_check.add_argument("--overwrite", action="store_true")
    add_exemplar_backend_arguments(copy_check)

    rethreshold = commands.add_parser("rethreshold", help="T1-P40 from the fit's dumped rows")
    rethreshold.add_argument("--model", type=Path, required=True)
    rethreshold.add_argument("--rows", type=Path, required=True)
    rethreshold.add_argument("--min-precision", type=float, default=0.40)
    rethreshold.add_argument("--output", type=Path, required=True)
    rethreshold.add_argument("--overwrite", action="store_true")

    check = commands.add_parser("check-fit", help="the fit's rows against the harness")
    check.add_argument("--rows", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    check.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="the replicate artifact (the copy-check artifact when exact search was selected)",
    )
    check.add_argument("--cache", type=Path, help="default: the artifact's cache")
    check.add_argument("--output", type=Path)
    check.add_argument("--overwrite", action="store_true")
    add_exemplar_backend_arguments(check)

    fill = commands.add_parser(
        "prereg-fill",
        help="write the Chroma library fingerprint of a sync --verify-only report into the "
        "pre-registration's placeholder",
    )
    fill.add_argument("--sync-report", type=Path, required=True)
    fill.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)

    report = commands.add_parser("report", help="summarise harness artifacts")
    report.add_argument("--parity", type=Path)
    report.add_argument("--ann-recall", type=Path)
    report.add_argument("--sweep", type=Path)
    report.add_argument("--replicate", type=Path)
    report.add_argument("--copy-check", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--overwrite", action="store_true")
    return parser


COMMANDS: dict[str, Callable[..., int]] = {
    "cache": cmd_cache,
    "parity": cmd_parity,
    "ann-recall": cmd_ann_recall,
    "sweep": cmd_sweep,
    "replicate": cmd_replicate,
    "copy-check": cmd_copy_check,
    "rethreshold": cmd_rethreshold,
    "check-fit": cmd_check_fit,
    "report": cmd_report,
    "prereg-fill": cmd_prereg_fill,
}


def main(argv: list[str] | None = None, **injected: Any) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args, **injected)
    except (Refused, DevOnlyError, CacheError, CacheMiss, PreregError) as error:
        print(f"tune_retrieval {args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
