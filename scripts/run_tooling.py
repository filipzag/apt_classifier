"""Shared helpers for the benchmark and fit scripts: output guards and run identity.

Output guard
------------
In a git worktree ``data/models`` is a symlink to the main checkout's shared models, so a fit
or rebuild with its default output silently replaces the artifacts the app and every other
session use. :func:`enforce_output_guard` refuses any output whose *resolved* path lies outside
the repository root unless ``--allow-shared-output``, and refuses to replace an existing file
unless ``--overwrite``. In the main checkout the default paths resolve inside the repository, so
only ``--overwrite`` is needed there (the Makefile targets pass it).

Run identity
------------
:func:`git_state`, :func:`acceptance_identity` and :func:`exemplar_identity` describe what
produced an artifact (commit, uncommitted diff, the acceptance model and exemplar index actually
loaded), so two runs can be told apart without reading the code that made them.

:func:`technique_micro_metrics` is shared by ``attribution_benchmark.py`` (artifact metrics) and
``compare_runs.py`` (the same numbers recomputed from older artifacts' rows).

Exemplar backend
----------------
:func:`add_exemplar_backend_arguments` gives a script ``--exemplar-backend``,
``--exemplar-search`` (overrides of ``TECHNIQUE_EXEMPLAR_BACKEND`` / ``_SEARCH`` for the run, see
:func:`exemplar_setting_overrides`) and ``--require-exemplar-backend X``: the run stops with exit
2 when the library did not load from ``X`` (:func:`exemplar_backend_problem` right after loading)
or when a Chroma exemplar query fell back to the local search (checked again at the end). Recall
misses of an HNSW search are reported in the artifact, never refused: the votes stay exact.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def add_output_guard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite", action="store_true", help="replace output files that already exist"
    )
    parser.add_argument(
        "--allow-shared-output",
        action="store_true",
        help="allow outputs that resolve outside this repository (e.g. a worktree's data/models, "
        "which is a symlink to the main checkout's shared models)",
    )


def guard_output_paths(
    paths: Iterable[Path | str | None],
    *,
    repo_root: Path = REPO_ROOT,
    allow_shared: bool = False,
    overwrite: bool = False,
) -> list[str]:
    """Reasons the given output paths must not be written (empty when all are allowed)."""
    root = Path(repo_root).resolve()
    problems: list[str] = []
    for path in paths:
        if path is None:
            continue
        resolved = Path(path).resolve()
        if not allow_shared and not resolved.is_relative_to(root):
            problems.append(
                f"{path} resolves to {resolved}, outside the repository {root}; pass "
                "--allow-shared-output to write there"
            )
        if not overwrite and resolved.exists():
            problems.append(f"{path} already exists; pass --overwrite to replace it")
    return problems


def enforce_output_guard(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    paths: Iterable[Path | str | None],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """``parser.error`` (exit 2) before any work when an output path is refused."""
    problems = guard_output_paths(
        paths,
        repo_root=repo_root,
        allow_shared=bool(getattr(args, "allow_shared_output", False)),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    if problems:
        parser.error("refusing to write: " + "; ".join(problems))


def _git(root: Path, *arguments: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def git_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, dirty flag and SHA-256 of the uncommitted diff of tracked files.

    Uses ``git -C`` rather than reading ``.git/HEAD``: in a worktree ``.git`` is a file, and a
    parent-directory search would report the main checkout's commit instead."""
    commit = _git(root, "rev-parse", "HEAD")
    diff = _git(root, "diff", "HEAD", "--binary")
    return {
        "commit": commit.decode().strip() if commit else None,
        "dirty": bool(diff) if diff is not None else None,
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
    }


def file_sha256(path: Path | str | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path | str | None, repo_root: Path = REPO_ROOT) -> str | None:
    """A path as recorded in artifacts: relative to the repository when inside it (not following
    symlinks, so the same configured path reads the same in every checkout), else absolute."""
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        return str(candidate.relative_to(repo_root))
    except ValueError:
        return str(candidate)


def acceptance_identity(
    settings: Any, model: Any, status: str, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Which acceptance model a run used: the artifact actually loaded (primary or fallback),
    its threshold, the exemplar index it requires and the file hash."""
    primary = getattr(settings, "acceptance_model_path", None)
    fallback = getattr(settings, "acceptance_model_fallback_path", None)
    loaded = None
    if model is not None:
        loaded = fallback if str(status).startswith("fallback") else primary
    threshold = getattr(model, "threshold", None) if model is not None else None
    return {
        "status": status,
        "path": repo_relative(loaded, repo_root),
        "configured_path": repo_relative(primary, repo_root),
        "fallback_path": repo_relative(fallback, repo_root),
        "threshold": float(threshold) if threshold is not None else None,
        "exemplar_fingerprint": getattr(model, "exemplar_fingerprint", None)
        if model is not None
        else None,
        "sha256": file_sha256(loaded),
    }


def acceptance_fingerprint_inputs(identity: dict[str, Any]) -> dict[str, Any] | None:
    """The stable part of :func:`acceptance_identity` for a pipeline fingerprint (no status
    text, which can carry machine-specific paths)."""
    if identity.get("path") is None:
        return None
    return {
        key: identity.get(key) for key in ("path", "threshold", "exemplar_fingerprint", "sha256")
    }


def exemplar_signature(index: Any) -> dict[str, Any] | None:
    """The loaded library's ``signature()`` (backend, search, collection, fingerprint)."""
    signature = getattr(index, "signature", None) if index is not None else None
    return dict(signature()) if callable(signature) else None


def exemplar_query_totals() -> dict[str, int]:
    """Chroma exemplar queries, recall misses and search fallbacks so far in this process."""
    from apt_classifier.exemplar_store import query_totals

    return query_totals()


def exemplar_identity(
    settings: Any, index: Any, status: str, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    signature = exemplar_signature(index)
    identity = {
        "status": status,
        "path": repo_relative(getattr(settings, "technique_exemplar_index_path", None), repo_root),
        "fingerprint": getattr(index, "fingerprint", None) if index is not None else None,
        "count": len(index) if index is not None else None,
        "backend": (signature or {}).get("backend") if index is not None else None,
        "signature": signature,
    }
    if (signature or {}).get("backend") == "chroma":
        # counters at the time the identity is taken (the end of the run in the benchmarks)
        identity["query_totals"] = exemplar_query_totals()
    return identity


def add_exemplar_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exemplar-backend",
        choices=("npz", "chroma", "auto"),
        help="exemplar library source for this run (default: TECHNIQUE_EXEMPLAR_BACKEND)",
    )
    parser.add_argument(
        "--exemplar-search",
        choices=("exact", "hnsw"),
        help="Chroma exemplar search for this run (default: TECHNIQUE_EXEMPLAR_SEARCH)",
    )
    parser.add_argument(
        "--require-exemplar-backend",
        choices=("npz", "chroma"),
        help="exit 2 unless the exemplar library loaded from this backend and no Chroma "
        "exemplar query fell back to the local search (recall misses are only reported)",
    )


def exemplar_setting_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Settings fields set by ``--exemplar-backend`` / ``--exemplar-search``."""
    overrides: dict[str, Any] = {}
    if getattr(args, "exemplar_backend", None):
        overrides["technique_exemplar_backend"] = args.exemplar_backend
    if getattr(args, "exemplar_search", None):
        overrides["technique_exemplar_search"] = args.exemplar_search
    return overrides


def exemplar_backend_problem(index: Any, required: str | None) -> str | None:
    """Why ``--require-exemplar-backend`` stops the run, ``None`` when it may go on."""
    if not required:
        return None
    if index is None:
        return f"--require-exemplar-backend {required}: no exemplar library was loaded"
    backend = str(getattr(index, "backend", "npz") or "npz")
    if backend != required:
        return f"--require-exemplar-backend {required}: the library loaded from {backend}"
    fallbacks = exemplar_query_totals().get("search_fallbacks", 0) if backend == "chroma" else 0
    if fallbacks:
        return (
            f"--require-exemplar-backend {required}: {fallbacks} exemplar queries fell back "
            "from Chroma to the local search"
        )
    return None


def technique_micro_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled technique precision/recall of the accepted sets against the gold labels, and the
    median accepted count per report (the combiner's input regime). Precision and recall pool
    the reports with gold labels only; the median counts every report. Empty for rows without
    accepted sets (older artifacts, unknown-actor rows)."""
    if not rows or not all("accepted_techniques" in row for row in rows):
        return {}
    gold_rows = [row for row in rows if row.get("held_out_techniques")]
    true_positive = sum(
        len(set(row["accepted_techniques"]) & set(row["held_out_techniques"])) for row in gold_rows
    )
    accepted = sum(len(row["accepted_techniques"]) for row in gold_rows)
    gold = sum(len(row["held_out_techniques"]) for row in gold_rows)
    return {
        "technique_micro_precision": true_positive / accepted if accepted else None,
        "technique_micro_recall": true_positive / gold if gold else None,
        "technique_micro_counts": {
            "true_positive": true_positive,
            "accepted": accepted,
            "gold": gold,
            "reports": len(gold_rows),
        },
        "median_accepted_techniques": float(
            median(len(row["accepted_techniques"]) for row in rows)
        ),
    }
