"""Append-only run ledger: one JSON line per benchmark or fit run.

    PYTHONPATH=. .venv/bin/python scripts/run_ledger.py append --ledger $OUT/ledger.jsonl \\
        --run-id R01 --stage B0 --gate G3 --command "scripts/benchmark_mapping.py ..." \\
        --artifact $OUT/runs/R01_B0_public_test.json \\
        --snapshot-before $OUT/chroma/R01_before.json --snapshot-after $OUT/chroma/R01_after.json \\
        --verdict recorded

Each line records what produced the artifacts and what they claim: the commit and a SHA-256 of
the uncommitted diff, the command and the configuration environment (secret-looking variables
only as set/empty), per artifact its file hash, embedded ``artifact_hash``, pipeline and source
fingerprints, acceptance model and exemplar identity, combiner status, pipeline and LLM spend,
and the Chroma snapshot hashes before and after with their diff verdict. Lines are never
rewritten; ``scripts/compare_runs.py summary`` composes them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_tooling import file_sha256, git_state, repo_relative  # noqa: E402

ENV_PREFIXES = (
    "ACCEPTANCE_",
    "ATTRIBUTION_",
    "CHROMA_",
    "EMBEDDING_",
    "GROUP_MLP",
    "OPENROUTER_",
    "PROFILE_",
    "RERANKER_",
    "TECHNIQUE_",
)
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _redact(name: str, value: str) -> str:
    if any(marker in name.upper() for marker in SECRET_MARKERS):
        return "<set>" if value else "<empty>"
    return value


def capture_env(
    environ: Mapping[str, str], explicit: Iterable[str] = (), prefixes: Iterable[str] = ENV_PREFIXES
) -> dict[str, str]:
    """Configuration variables from the environment plus explicit ``KEY=VALUE`` overrides;
    secret-looking names keep only whether they were set."""
    captured = {
        name: _redact(name, value)
        for name, value in environ.items()
        if name.upper().startswith(tuple(prefixes))
    }
    for item in explicit:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        captured[name] = _redact(name, value)
    return dict(sorted(captured.items()))


def _fingerprints(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(pipeline fingerprints, source fingerprints)`` by where they sit in the artifact:
    top level (attribution), ``metadata`` (EvidenceForge) or per result mode (mapping)."""
    pipeline: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    if payload.get("pipeline_fingerprint"):
        pipeline["artifact"] = payload["pipeline_fingerprint"]
    source = (payload.get("fingerprint_inputs") or {}).get("source_fingerprint")
    if source:
        sources["artifact"] = source
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if metadata.get("pipeline_fingerprint"):
        pipeline["metadata"] = metadata["pipeline_fingerprint"]
    for mode, block in sorted((payload.get("results") or {}).items()):
        if not isinstance(block, dict):
            continue
        if block.get("pipeline_fingerprint"):
            pipeline[mode] = block["pipeline_fingerprint"]
        source = (block.get("fingerprint_inputs") or {}).get("source_fingerprint")
        if source:
            sources[mode] = source
    return pipeline, sources


def _rethresholded(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    return {key: block.get(key) for key in ("threshold", "source_artifact", "source_artifact_hash")}


def describe_artifact(path: Path) -> dict[str, Any]:
    """What one artifact file says about itself (never its rows)."""
    entry: dict[str, Any] = {
        "path": repo_relative(Path(path).absolute()),
        "sha256": file_sha256(path),
    }
    if entry["sha256"] is None:
        entry["missing"] = True
        return entry
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return entry  # a model or cache file: the hash is the identity
    if not isinstance(payload, dict):
        return entry
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    pipeline, sources = _fingerprints(payload)
    inputs = payload.get("fingerprint_inputs") or {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    llm = metrics.get("llm") or {}
    entry.update(
        {
            "artifact_hash": payload.get("artifact_hash"),
            "generated_at": payload.get("generated_at") or metadata.get("generated_at"),
            "pipeline_fingerprints": pipeline,
            "source_fingerprints": sources,
            "acceptance_model": payload.get("acceptance_model") or metadata.get("acceptance_model"),
            # an offline re-threshold (compare_runs.py mapping --rethreshold) of another run
            "rethresholded": _rethresholded(payload.get("rethresholded")),
            # mapping modes whose cases ran without the loaded model (refused per call)
            "acceptance_model_refused": {
                mode: block["acceptance_model_disabled"].get("cases")
                for mode, block in sorted((payload.get("results") or {}).items())
                if isinstance(block, dict)
                and isinstance(block.get("acceptance_model_disabled"), dict)
            }
            or None,
            "exemplars": payload.get("exemplars") or metadata.get("exemplars"),
            "combiner": inputs.get("combiner"),
            "pipeline": (payload.get("parameters") or {}).get("pipeline")
            or (metadata.get("pipeline") or {}).get("pipeline"),
            "verify": (metadata.get("pipeline") or {}).get("verify"),
            "llm": llm or None,
        }
    )
    return entry


def describe_snapshots(before: Path | None, after: Path | None) -> dict[str, Any] | None:
    if before is None and after is None:
        return None
    loaded = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in (("before", before), ("after", after))
        if path is not None
    }
    out: dict[str, Any] = {
        f"{label}_hash": snapshot.get("snapshot_hash") for label, snapshot in loaded.items()
    }
    if len(loaded) == 2:
        from scripts.chroma_snapshot import diff_snapshots

        diff = diff_snapshots(loaded["before"], loaded["after"])
        out.update(
            {
                "clean": diff["clean"],
                "changed": sorted(diff["changed"]),
                "removed": diff["removed"],
                "added": diff["added"],
            }
        )
    return out


def build_entry(args: argparse.Namespace, environ: Mapping[str, str]) -> dict[str, Any]:
    git = git_state(REPO_ROOT)
    artifacts = [describe_artifact(path) for path in args.artifact or []]
    reported_costs = [
        float((item.get("llm") or {}).get("cost_usd") or 0.0)
        for item in artifacts
        if item.get("llm")
    ]
    cost = (
        args.llm_cost_usd
        if args.llm_cost_usd is not None
        else (round(sum(reported_costs), 4) if reported_costs else None)
    )
    return {
        "run_id": args.run_id,
        "stage": args.stage,
        "gate": args.gate,
        "recorded_at": datetime.now(UTC).isoformat(),
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_diff_sha256": git["diff_sha256"],
        "command": args.command,
        "argv": shlex.split(args.command) if args.command else None,
        "env": capture_env(environ, args.env or ()),
        "artifacts": artifacts,
        "chroma": describe_snapshots(args.snapshot_before, args.snapshot_after),
        "llm_cost_usd": cost,
        "verdict": args.verdict,
        "notes": args.notes,
    }


def append_entry(ledger: Path, entry: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command_name", required=True)
    append = commands.add_parser("append", help="append one run")
    append.add_argument("--ledger", type=Path, required=True)
    append.add_argument("--run-id", required=True)
    append.add_argument("--stage")
    append.add_argument("--gate")
    append.add_argument("--command", help="the command line that was run")
    append.add_argument("--artifact", type=Path, action="append", help="artifact written by it")
    append.add_argument("--env", action="append", metavar="KEY=VALUE", help="extra settings")
    append.add_argument("--snapshot-before", type=Path)
    append.add_argument("--snapshot-after", type=Path)
    append.add_argument("--llm-cost-usd", type=float, help="default: from the artifacts")
    append.add_argument("--started-at")
    append.add_argument("--ended-at")
    append.add_argument("--verdict")
    append.add_argument("--notes")
    args = parser.parse_args(argv)
    try:
        entry = build_entry(args, os.environ if environ is None else environ)
    except ValueError as error:
        parser.error(str(error))
    append_entry(args.ledger, entry)
    print(json.dumps({"run_id": entry["run_id"], "ledger": str(args.ledger)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
