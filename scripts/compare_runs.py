"""Compare benchmark artifacts offline: identity checks, paired bootstraps, re-thresholding.

Reads only artifact files (no Chroma, no model, no LLM). Subcommands:

``mapping BASE [CURRENT]`` (``scripts/benchmark_mapping.py`` artifacts)
    ``--expect-identical``  exit 1 unless every mode has the same cases, ranked and accepted
                            lists, candidate rows (fields both artifacts carry) and metrics;
    ``--paired-bootstrap N --seed S``  paired case bootstrap of the micro P/R/F1 and hit@10
                            differences CURRENT - BASE;
    ``--pr-curve``          precision/recall/F1 per threshold from the stored per-candidate
                            acceptance probabilities and runtime guards, with a self-check that
                            the run's own threshold reproduces its accepted sets;
    ``--rethreshold T --output PATH``  one artifact re-thresholded at T, written in the
                            benchmark's own format (``check_regression.py`` reads it); refused
                            unless the self-check reproduces every accepted set.
``attribution BASE CURRENT`` (``scripts/attribution_benchmark.py`` artifacts)
    ``--expect-identical``, ``--names`` (who is named, wrong names), ``--technique-pr`` (accepted
    technique micro precision/recall, median accepted count) and ``--combiner-shift`` (the
    pre-registered combiner refit trigger).
``evidenceforge BASE CURRENT`` (``scripts/evidenceforge_eval.py`` artifacts)
    per-mode metric deltas, red herrings accepted, ``--expect-identical``.
``summary --ledger LEDGER`` composes the run ledger (``scripts/run_ledger.py``).

Exit codes: 0 fine, 1 an ``--expect-identical`` difference, 2 not comparable or refused.
Re-thresholding is exact only where the stored rows hold everything the gate reads: a single
observation per case, no validated reranker, hierarchy-parent mode off, and every candidate the
gate accepted within the stored candidate rows; the self-check enforces the last one.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_tooling import technique_micro_metrics  # noqa: E402

NAMED_STAGE = "known_group_hypothesis"
ACCEPTED = "auto_accepted"
REVIEW = "needs_review"
VISIBLE_STATES = frozenset({ACCEPTED, REVIEW})
# the ranking metrics depend on the full ranked list, of which artifacts store the top ten
RANKING_METRICS = ("hit_at_1", "hit_at_3", "hit_at_10", "mrr", "ndcg_at_10")
# Artifacts written before the maximum-matching fix (up to and including the 2026-10-02 runs,
# e.g. R01-R37 of the Chroma exemplar programme) computed parent-aware metrics with a
# first-fit pairing over Python sets, so identical case rows could score differently under
# another PYTHONHASHSEED. Differences there are reported, never an identity failure; case
# rows and accepted sets compare exactly, and they determine the parent-aware counts.
ORDER_DEPENDENT_METRICS = frozenset(
    {"parent_aware_precision", "parent_aware_recall", "parent_aware_f1"}
)
DEFAULT_CURVE = tuple(round(value, 2) for value in np.arange(0.05, 0.96, 0.05))
LOW_PRECISION = 0.30  # lower edge of the combiner's training precision band


class NotComparableError(ValueError):
    """Artifacts that cannot be compared (missing modes, different case sets, no rows)."""


# ----------------------------------------------------------------------------- helpers


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NotComparableError(f"{path} is not a JSON object")
    return payload


def _same(left: Any, right: Any) -> bool:
    """Deep equality where NaN equals NaN (JSON round-trips NaN metrics)."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def compare_records(
    base: dict[str, Any], current: dict[str, Any], *, ignore: Iterable[str] = ()
) -> tuple[list[str], list[str], list[str]]:
    """``(differing keys, keys only in base, keys only in current)``; values compared deeply on
    the keys both carry, so fields added by newer code are reported, not counted as changes."""
    skip = set(ignore)
    shared = (base.keys() & current.keys()) - skip
    differing = sorted(key for key in shared if not _same(base[key], current[key]))
    return (
        differing,
        sorted(base.keys() - current.keys() - skip),
        sorted(current.keys() - base.keys() - skip),
    )


def _upper(values: Iterable[Any]) -> set[str]:
    return {str(item).strip().upper() for item in values or () if str(item).strip()}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ----------------------------------------------------------------------------- mapping


def _modes(base: dict[str, Any], current: dict[str, Any] | None, wanted: Sequence[str] | None):
    base_modes = set(base.get("results") or {})
    current_modes = set((current or {}).get("results") or {}) if current is not None else base_modes
    if wanted:
        missing = [mode for mode in wanted if mode not in base_modes or mode not in current_modes]
        if missing:
            raise NotComparableError(f"mode(s) {missing} missing from an artifact")
        return list(wanted)
    shared = sorted(base_modes & current_modes)
    if not shared:
        raise NotComparableError("the artifacts share no result mode")
    return shared


def compare_mapping_identity(
    base: dict[str, Any], current: dict[str, Any], *, modes: Sequence[str], max_examples: int = 20
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "same_case_set": base.get("case_set_hash") == current.get("case_set_hash"),
        "modes": {},
    }
    identical = report["same_case_set"]
    for mode in modes:
        left, right = base["results"][mode], current["results"][mode]
        left_cases = {row["case_id"]: row for row in left.get("cases", [])}
        right_cases = {row["case_id"]: row for row in right.get("cases", [])}
        examples: list[dict[str, Any]] = []
        differing_cases = 0
        fields_only: dict[str, set[str]] = {"base": set(), "current": set()}
        for case_id in sorted(left_cases.keys() | right_cases.keys()):
            a, b = left_cases.get(case_id), right_cases.get(case_id)
            if a is None or b is None:
                differing_cases += 1
                if len(examples) < max_examples:
                    examples.append(
                        {"case_id": case_id, "missing_in": "base" if a is None else "current"}
                    )
                continue
            fields, _, _ = compare_records(a, b, ignore=("candidates",))
            rows_a, rows_b = a.get("candidates") or [], b.get("candidates") or []
            if len(rows_a) != len(rows_b):
                fields.append("candidates.length")
            else:
                for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b, strict=True)):
                    keys, only_a, only_b = compare_records(row_a, row_b)
                    fields_only["base"].update(only_a)
                    fields_only["current"].update(only_b)
                    fields.extend(f"candidates[{index}].{key}" for key in keys)
            if fields:
                differing_cases += 1
                if len(examples) < max_examples:
                    examples.append({"case_id": case_id, "fields": fields[:12]})
        metric_blocks = {"pooled": (left.get("pooled") or {}, right.get("pooled") or {})}
        for dataset in sorted(set(left.get("datasets") or {}) | set(right.get("datasets") or {})):
            metric_blocks[f"datasets.{dataset}"] = (
                (left.get("datasets") or {}).get(dataset) or {},
                (right.get("datasets") or {}).get(dataset) or {},
            )
        metric_differences: dict[str, list[str]] = {}
        order_dependent: dict[str, list[str]] = {}
        for label, (block_a, block_b) in metric_blocks.items():
            for part in ("metrics", "bootstrap_confidence_intervals"):
                keys, only_a, only_b = compare_records(
                    block_a.get(part) or {}, block_b.get(part) or {}
                )
                loose = [key for key in keys if key in ORDER_DEPENDENT_METRICS]
                keys = [key for key in keys if key not in ORDER_DEPENDENT_METRICS]
                if loose:
                    order_dependent[f"{label}.{part}"] = loose
                if keys or (not block_a) != (not block_b):
                    metric_differences[f"{label}.{part}"] = keys or ["<missing>"]
        mode_identical = differing_cases == 0 and not metric_differences
        identical = identical and mode_identical
        report["modes"][mode] = {
            "identical": mode_identical,
            "cases": len(left_cases),
            "differing_cases": differing_cases,
            "examples": examples,
            "metric_differences": metric_differences,
            "order_dependent_metric_differences": order_dependent,
            "candidate_fields_only_in_base": sorted(fields_only["base"]),
            "candidate_fields_only_in_current": sorted(fields_only["current"]),
            "pipeline_fingerprint_equal": left.get("pipeline_fingerprint")
            == right.get("pipeline_fingerprint"),
            # whether the acceptance model scored every case of each run
            "acceptance_model_active": {
                "base": left.get("acceptance_model_active"),
                "current": right.get("acceptance_model_active"),
            },
        }
    report["identical"] = identical
    return report


def _case_counts(row: dict[str, Any], accepted: Iterable[str] | None = None) -> tuple[int, ...]:
    """``(positive, tp, fp, fn, hit@10)`` for one case, as ``evaluate_technique_predictions``
    counts them (zero-label cases add nothing to the micro counts)."""
    expected = _upper(row.get("expected_technique_ids"))
    if not expected:
        return (0, 0, 0, 0, 0)
    chosen = _upper(row.get("accepted_technique_ids") if accepted is None else accepted)
    ranked = [str(item).upper() for item in row.get("ranked_technique_ids") or []][:10]
    return (
        1,
        len(expected & chosen),
        len(chosen - expected),
        len(expected - chosen),
        int(any(item in expected for item in ranked)),
    )


def _metrics_from_counts(counts: np.ndarray) -> dict[str, float]:
    positive, tp, fp, fn, hits = (float(value) for value in counts.sum(axis=0))
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "hit_at_10": hits / positive if positive else 0.0,
    }


def paired_bootstrap(
    base_block: dict[str, Any], current_block: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    """Paired case bootstrap of CURRENT - BASE over the cases both runs scored."""
    left = {row["case_id"]: row for row in base_block.get("cases", [])}
    right = {row["case_id"]: row for row in current_block.get("cases", [])}
    if left.keys() != right.keys():
        raise NotComparableError("paired bootstrap needs the same cases in both runs")
    order = sorted(left)
    a = np.asarray([_case_counts(left[case_id]) for case_id in order], dtype=float)
    b = np.asarray([_case_counts(right[case_id]) for case_id in order], dtype=float)
    point_a, point_b = _metrics_from_counts(a), _metrics_from_counts(b)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point_a}
    for _ in range(samples):
        index = rng.integers(0, len(order), size=len(order))
        sample_a, sample_b = _metrics_from_counts(a[index]), _metrics_from_counts(b[index])
        for name in draws:
            draws[name].append(sample_b[name] - sample_a[name])
    out: dict[str, Any] = {"unit": "case", "cases": len(order), "samples": samples, "seed": seed}
    for name, values in draws.items():
        array = np.asarray(values)
        out[name] = {
            "base": point_a[name],
            "current": point_b[name],
            "delta": point_b[name] - point_a[name],
            "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
            "share_not_above_zero": float(np.mean(array <= 0.0)),
        }
    # the counts above must reproduce the artifact's own pooled estimates
    out["matches_artifact"] = all(
        abs(
            point[name]
            - float((block.get("pooled") or {}).get("metrics", {}).get(name, point[name]))
        )
        < 1e-12
        for point, block in ((point_a, base_block), (point_b, current_block))
        for name in ("micro_precision", "micro_recall", "micro_f1", "hit_at_10")
    )
    return out


def _dataset_deltas(base_block: dict[str, Any], current_block: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = ("micro_precision", "micro_recall", "micro_f1", "hit_at_10")
    for dataset in sorted(
        set(base_block.get("datasets") or {}) & set(current_block.get("datasets") or {})
    ):
        left = base_block["datasets"][dataset].get("metrics") or {}
        right = current_block["datasets"][dataset].get("metrics") or {}
        out[dataset] = {
            name: {
                "base": left[name],
                "current": right[name],
                "delta": float(right[name]) - float(left[name]),
            }
            for name in names
            if name in left and name in right
        }
    return out


def _unsupported_rethreshold(block: dict[str, Any]) -> str | None:
    inputs = block.get("fingerprint_inputs") or {}
    settings = inputs.get("settings") or {}
    if inputs.get("reranker") is not None:
        return "a reranker was active (its promote/demote step is not stored)"
    if str(settings.get("hierarchy_parent_mode", "off")) != "off":
        return "hierarchy-parent mode was on (its consistency pass is not stored)"
    refused = block.get("acceptance_model_disabled")
    if refused:
        # those cases ran the rule-based gate; a threshold says nothing about them
        reasons = "; ".join((refused.get("reasons") or {}).keys())
        return f"the acceptance model was refused in {refused.get('cases')} cases ({reasons})"
    rows = [row for case in block.get("cases", []) for row in case.get("candidates") or []]
    if not any(row.get("acceptance_probability") is not None for row in rows):
        return "no stored acceptance probabilities (no model, or an artifact from older code)"
    if any(row.get("cue_supported") is None for row in rows):
        return "candidate rows lack the runtime guards (benign/sparse/cue_supported)"
    return None


def _emulated_state(row: dict[str, Any], threshold: float) -> str:
    """The state ``_assign_states_for_group`` gives one candidate at ``threshold``."""
    stored = str(row.get("state"))
    if row.get("hierarchy_parent") or str(row.get("state_reason") or "").startswith(
        "OpenRouter verifier"
    ):
        return stored
    probability = row.get("acceptance_probability")
    if probability is None:
        return stored
    model_accepted = (
        float(probability) >= float(threshold) and not row.get("benign") and not row.get("sparse")
    )
    return ACCEPTED if model_accepted or row.get("cue_supported") else REVIEW


def _has_retrieval(row: dict[str, Any]) -> bool:
    return any(
        row.get(key) is not None
        for key in ("dense_score", "dense_rank", "dense_field", "lexical_rank", "lexical_score")
    )


def emulate_case(case: dict[str, Any], threshold: float) -> tuple[list[str], list[str]]:
    """``(accepted ids, state per candidate row)`` at ``threshold``; the accepted set follows
    ``_materialize_candidates``: a parent of a visible child that has no hierarchy, dense or
    lexical evidence of its own is not materialized."""
    rows = case.get("candidates") or []
    states = [_emulated_state(row, threshold) for row in rows]
    child_parents = {
        row.get("parent_technique_id")
        for row, state in zip(rows, states, strict=True)
        if state in VISIBLE_STATES and row.get("parent_technique_id")
    }
    accepted = []
    for row, state in zip(rows, states, strict=True):
        if state != ACCEPTED:
            continue
        if (
            row["technique_id"] in child_parents
            and not row.get("hierarchy_parent")
            and not _has_retrieval(row)
        ):
            continue
        accepted.append(str(row["technique_id"]))
    return sorted(set(accepted)), states


def _run_threshold(block: dict[str, Any]) -> float | None:
    model = (block.get("fingerprint_inputs") or {}).get("acceptance_model") or {}
    value = model.get("threshold")
    return float(value) if value is not None else None


def self_check(block: dict[str, Any]) -> dict[str, Any]:
    """Re-threshold at the run's own threshold: must reproduce every stored accepted set."""
    threshold = _run_threshold(block)
    if threshold is None:
        return {"threshold": None, "mismatches": None, "examples": []}
    mismatched = []
    for case in block.get("cases", []):
        accepted, _ = emulate_case(case, threshold)
        if _upper(accepted) != _upper(case.get("accepted_technique_ids")):
            mismatched.append(case["case_id"])
    return {"threshold": threshold, "mismatches": len(mismatched), "examples": mismatched[:10]}


def pr_curve(block: dict[str, Any], thresholds: Sequence[float] = DEFAULT_CURVE) -> dict[str, Any]:
    unsupported = _unsupported_rethreshold(block)
    if unsupported:
        return {"available": False, "reason": unsupported}
    own = _run_threshold(block)
    grid = sorted({*(float(t) for t in thresholds), *((own,) if own is not None else ())})
    cases = block.get("cases", [])
    points: list[dict[str, Any]] = []
    accepted_by_threshold: dict[float, set[tuple[str, str]]] = {}
    for threshold in grid:
        counts = []
        pairs: set[tuple[str, str]] = set()
        for case in cases:
            accepted, _ = emulate_case(case, threshold)
            counts.append(_case_counts(case, accepted))
            pairs.update((case["case_id"], technique_id.upper()) for technique_id in accepted)
        accepted_by_threshold[threshold] = pairs
        metrics = _metrics_from_counts(np.asarray(counts, dtype=float))
        points.append(
            {
                "threshold": threshold,
                "precision": metrics["micro_precision"],
                "recall": metrics["micro_recall"],
                "f1": metrics["micro_f1"],
                "accepted": len(pairs),
                "run_threshold": own is not None and threshold == own,
            }
        )
    # band precision: the precision of what lowering the threshold one grid step adds (over
    # labelled cases, as the micro precision; zero-label cases only count in "accepted")
    expected = {case["case_id"]: _upper(case.get("expected_technique_ids")) for case in cases}
    previous: set[tuple[str, str]] | None = None
    for index in range(len(grid) - 1, -1, -1):
        pairs = {pair for pair in accepted_by_threshold[grid[index]] if expected[pair[0]]}
        band = pairs - previous if previous is not None else pairs
        correct = sum(1 for case_id, technique_id in band if technique_id in expected[case_id])
        points[index]["band_accepted"] = len(band)
        points[index]["band_precision"] = correct / len(band) if band else None
        previous = pairs
    return {"available": True, "self_check": self_check(block), "points": points}


def rethreshold_artifact(
    artifact: dict[str, Any],
    threshold: float,
    *,
    kb: Any,
    modes: Sequence[str] | None = None,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
    source: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(new artifact, checks)``: accepted sets and accepted-set metrics recomputed at
    ``threshold``; ranking metrics (which need the full ranked list) are carried over. The
    intervals are redone with the run's own bootstrap samples and seed unless overridden."""
    from apt_classifier.evaluation import (
        TechniqueEvaluationCase,
        TechniquePrediction,
        bootstrap_technique_metrics,
        evaluate_technique_predictions,
        pipeline_fingerprint,
    )
    from scripts.benchmark_mapping import summarize

    seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else int((artifact.get("case_diagnostics") or {}).get("seed", 7))
    )
    if bootstrap_samples is None:
        bootstrap_samples = int(artifact.get("bootstrap_samples") or 300)
    output = copy.deepcopy(artifact)
    output.pop("artifact_hash", None)
    checks: dict[str, Any] = {}
    for mode in modes or sorted(artifact.get("results") or {}):
        block = output["results"][mode]
        unsupported = _unsupported_rethreshold(block)
        if unsupported:
            raise NotComparableError(f"{mode}: cannot re-threshold: {unsupported}")
        checks[mode] = self_check(block)
        cases: list[Any] = []
        predictions: list[Any] = []
        for case in block["cases"]:
            accepted, states = emulate_case(case, threshold)
            for row, state in zip(case.get("candidates") or [], states, strict=True):
                if state != row.get("state"):
                    row["state"] = state
                    row["state_reason"] = f"re-thresholded offline at {threshold}"
            expected = set(case.get("expected_technique_ids") or ())
            case["accepted_technique_ids"] = accepted
            case["expected_accepted"] = bool(expected & set(accepted))
            cases.append(
                TechniqueEvaluationCase(
                    case_id=case["case_id"],
                    document_id=case["case_id"],
                    text=str(case.get("text") or ""),
                    expected_technique_ids=tuple(case.get("expected_technique_ids") or ()),
                    source=f"rethreshold:{case.get('dataset')}",
                )
            )
            predictions.append(
                TechniquePrediction.from_ranked(
                    case["case_id"], case.get("ranked_technique_ids") or (), accepted=accepted
                )
            )
        datasets = {case["dataset"] for case in block["cases"]}
        by_case = {prediction.case_id: prediction for prediction in predictions}
        dataset_of = {case["case_id"]: case["dataset"] for case in block["cases"]}
        groups = [("pooled", cases)] + [
            (dataset, [case for case in cases if dataset_of[case.case_id] == dataset])
            for dataset in sorted(datasets)
        ]
        for label, subset in groups:
            target = block["pooled"] if label == "pooled" else block["datasets"][label]
            subset_predictions = [by_case[case.case_id] for case in subset]
            metrics = evaluate_technique_predictions(subset, subset_predictions, kb=kb)
            intervals = bootstrap_technique_metrics(
                subset, subset_predictions, kb=kb, samples=bootstrap_samples, seed=seed
            )
            for name in RANKING_METRICS:
                if name in (target.get("metrics") or {}):
                    metrics[name] = target["metrics"][name]
                if name in (target.get("bootstrap_confidence_intervals") or {}):
                    intervals[name] = target["bootstrap_confidence_intervals"][name]
            target["metrics"] = metrics
            target["bootstrap_confidence_intervals"] = intervals
            target["summary"] = summarize(metrics, intervals)
        inputs = block.setdefault("fingerprint_inputs", {})
        model = dict(inputs.get("acceptance_model") or {})
        original = model.get("threshold")
        model["threshold"] = float(threshold)
        inputs["acceptance_model"] = model
        block["pipeline_fingerprint"] = pipeline_fingerprint(inputs)
        block["rethresholded"] = {
            "threshold": float(threshold),
            "original_threshold": original,
            "self_check": checks[mode],
        }
    identity = output.get("acceptance_model")
    if isinstance(identity, dict):
        # the run-level identity (what run_ledger records): same weights file, but no longer
        # the threshold stored in it
        output["acceptance_model"] = {
            **identity,
            "threshold": float(threshold),
            "rethresholded_from": identity.get("threshold"),
        }
    output["rethresholded"] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_artifact": source,
        "source_artifact_hash": artifact.get("artifact_hash"),
        "threshold": float(threshold),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "carried_over_metrics": list(RANKING_METRICS),
    }
    return output, checks


def _load_kb(path: Path | None) -> Any:
    from apt_classifier.attack.kb import AttackKnowledgeBase

    if path is None:
        from apt_classifier.config import Settings

        path = Path(Settings().kb_path)
    if not Path(path).is_file():
        raise NotComparableError(f"knowledge base not found: {path} (pass --kb)")
    return AttackKnowledgeBase.load(str(path))


def cmd_mapping(args: argparse.Namespace) -> int:
    base = _load(args.base)
    current = _load(args.current) if args.current else None
    if args.rethreshold is not None:
        if current is not None:
            raise NotComparableError("--rethreshold reads one artifact")
        if args.output is None:
            raise NotComparableError("--rethreshold needs --output")
        output, checks = rethreshold_artifact(
            base,
            args.rethreshold,
            kb=_load_kb(args.kb),
            modes=args.modes,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            source=str(args.base),
        )
        broken = {mode: check for mode, check in checks.items() if check["mismatches"]}
        print(json.dumps({"self_check": checks}, indent=2, sort_keys=True))
        if broken and not args.allow_mismatch:
            print(
                "refusing to write: re-thresholding at the run's own threshold does not "
                f"reproduce its accepted sets ({broken}); pass --allow-mismatch to write anyway",
                file=sys.stderr,
            )
            return 2
        from apt_classifier.evaluation import write_artifact

        print(f"wrote {write_artifact(args.output, output)}")
        return 0
    modes = _modes(base, current, args.modes)
    report: dict[str, Any] = {
        "kind": "mapping",
        "base": str(args.base),
        "current": str(args.current) if args.current else None,
        "modes": modes,
    }
    status = 0
    if current is not None:
        if args.expect_identical:
            identity = compare_mapping_identity(base, current, modes=modes)
            report["identity"] = identity
            status = 0 if identity["identical"] else 1
        report["deltas"] = {}
        for mode in modes:
            entry: dict[str, Any] = {
                "datasets": _dataset_deltas(base["results"][mode], current["results"][mode])
            }
            if args.paired_bootstrap:
                if base.get("case_set_hash") != current.get("case_set_hash"):
                    raise NotComparableError("different case sets: no paired bootstrap")
                entry["paired_bootstrap"] = paired_bootstrap(
                    base["results"][mode],
                    current["results"][mode],
                    samples=args.paired_bootstrap,
                    seed=args.seed,
                )
            report["deltas"][mode] = entry
    if args.pr_curve:
        report["pr_curve"] = {
            label: {mode: pr_curve(artifact["results"][mode]) for mode in modes}
            for label, artifact in (("base", base), ("current", current))
            if artifact is not None
        }
    _emit(report, args.output)
    return status


# ----------------------------------------------------------------------------- attribution


def _keyed_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    seen: Counter[str] = Counter()
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        stem = f"{row.get('group_id')}|{row.get('report_id') or ''}"
        keyed[f"{stem}|{seen[stem]}"] = row
        seen[stem] += 1
    return keyed


SECTIONS = ("known", "unknown", "open_set")


def compare_attribution_identity(
    base: dict[str, Any], current: dict[str, Any], *, max_examples: int = 20
) -> dict[str, Any]:
    report: dict[str, Any] = {"sections": {}}
    identical = True
    for section in SECTIONS:
        left = _keyed_rows(base.get(section) or [])
        right = _keyed_rows(current.get(section) or [])
        differing: list[dict[str, Any]] = []
        fields_only: dict[str, set[str]] = {"base": set(), "current": set()}
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                differing.append(
                    {"row": key, "missing_in": "base" if key not in left else "current"}
                )
                continue
            keys, only_a, only_b = compare_records(left[key], right[key])
            fields_only["base"].update(only_a)
            fields_only["current"].update(only_b)
            if keys:
                differing.append({"row": key, "fields": keys[:12]})
        identical = identical and not differing
        report["sections"][section] = {
            "rows": len(left),
            "differing_rows": len(differing),
            "examples": differing[:max_examples],
            "fields_only_in_base": sorted(fields_only["base"]),
            "fields_only_in_current": sorted(fields_only["current"]),
        }
    keys, only_a, only_b = compare_records(base.get("metrics") or {}, current.get("metrics") or {})
    report["metric_differences"] = keys
    report["metrics_only_in_base"] = only_a
    report["metrics_only_in_current"] = only_b
    report["pipeline_fingerprint_equal"] = base.get("pipeline_fingerprint") == current.get(
        "pipeline_fingerprint"
    )
    report["identical"] = identical and not keys
    return report


def naming(artifact: dict[str, Any]) -> dict[str, Any]:
    """Who each run names. Known rows: correct when the named leader is the truth (rank 1);
    unknown-actor rows and open-set reports: any name is unsupported."""
    wrong: list[dict[str, Any]] = []
    named: dict[str, str] = {}
    counts = Counter()
    for section in SECTIONS:
        for key, row in _keyed_rows(artifact.get(section) or []).items():
            if row.get("stage") != NAMED_STAGE:
                continue
            named[f"{section}:{key}"] = str(row.get("leading"))
            counts[f"{section}_named"] += 1
            correct = section == "known" and row.get("rank") == 1
            counts[f"{section}_named_correct"] += int(correct)
            if not correct:
                wrong.append(
                    {
                        "section": section,
                        "group_id": row.get("group_id"),
                        "report_id": row.get("report_id"),
                        "leading": row.get("leading"),
                    }
                )
    return {"counts": dict(sorted(counts.items())), "wrong": wrong, "named": named}


def _report_precisions(rows: Sequence[dict[str, Any]]) -> list[float]:
    """Accepted precision per report with gold labels and at least one accepted technique."""
    values = []
    for row in rows:
        gold, accepted = (
            set(row.get("held_out_techniques") or ()),
            set(row.get("accepted_techniques") or ()),
        )
        if gold and accepted:
            values.append(len(gold & accepted) / len(accepted))
    return values


def technique_pr(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact.get("known") or []
    out = dict(technique_micro_metrics(rows))
    precisions = _report_precisions(rows)
    out["reports"] = len(rows)
    out["reports_without_accepted"] = sum(1 for row in rows if not row.get("accepted_techniques"))
    out["low_precision_share"] = (
        sum(1 for value in precisions if value < LOW_PRECISION) / len(precisions)
        if precisions
        else None
    )
    recalls = [row["technique_recall"] for row in rows if row.get("technique_recall") is not None]
    out["mean_technique_recall"] = float(np.mean(recalls)) if recalls else None
    return out


def combiner_shift(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered combiner refit trigger (plan section 7): (a) median accepted count per
    report shifts by more than 25%, (b) the share of reports with accepted precision below 0.30
    rises by more than 10 points, (c) top-1 falls (more than 0.02 on the folds, more than two
    reports on a report set) while the combiner-free order holds (``rrf_top1`` at or above)."""
    left, right = technique_pr(base), technique_pr(current)
    real_reports = "report_set" in (base.get("parameters") or {})
    reasons: list[str] = []
    median_a, median_b = (
        left.get("median_accepted_techniques"),
        right.get("median_accepted_techniques"),
    )
    shift = (median_b - median_a) / median_a if median_a else None
    if shift is not None and abs(shift) > 0.25:
        reasons.append(f"(a) median accepted count shifted {shift:+.1%}")
    low_a, low_b = left.get("low_precision_share"), right.get("low_precision_share")
    if low_a is not None and low_b is not None and low_b - low_a > 0.10:
        reasons.append(f"(b) low-precision report share rose {low_b - low_a:+.3f}")
    metrics_a, metrics_b = base.get("metrics") or {}, current.get("metrics") or {}
    rrf_holds = (
        metrics_a.get("rrf_top1") is not None
        and metrics_b.get("rrf_top1") is not None
        and float(metrics_b["rrf_top1"]) >= float(metrics_a["rrf_top1"])
    )
    if real_reports:
        drop = int(metrics_a.get("top1_count") or 0) - int(metrics_b.get("top1_count") or 0)
        top1_fell = drop > 2
    else:
        drop = float(metrics_a.get("top1") or 0.0) - float(metrics_b.get("top1") or 0.0)
        top1_fell = drop > 0.02
    if top1_fell and rrf_holds:
        reasons.append(f"(c) top-1 fell by {drop} while rrf_top1 held")
    return {
        "median_accepted": {"base": median_a, "current": median_b, "relative_shift": shift},
        "low_precision_share": {"base": low_a, "current": low_b},
        "top1_drop": drop,
        "rrf_top1_holds": rrf_holds,
        "triggered": bool(reasons),
        "reasons": reasons,
    }


def cmd_attribution(args: argparse.Namespace) -> int:
    base, current = _load(args.base), _load(args.current)
    report: dict[str, Any] = {
        "kind": "attribution",
        "base": str(args.base),
        "current": str(args.current),
    }
    status = 0
    keys = ("top1", "top5", "mrr", "named_correct", "named_risk", "false_named_attribution_rate")
    report["metrics"] = {
        key: {
            "base": (base.get("metrics") or {}).get(key),
            "current": (current.get("metrics") or {}).get(key),
        }
        for key in keys
    }
    if args.expect_identical:
        report["identity"] = compare_attribution_identity(base, current)
        status = 0 if report["identity"]["identical"] else 1
    if args.names:
        left, right = naming(base), naming(current)
        report["names"] = {
            "base": {"counts": left["counts"], "wrong": left["wrong"]},
            "current": {"counts": right["counts"], "wrong": right["wrong"]},
            "newly_named": sorted(right["named"].keys() - left["named"].keys()),
            "no_longer_named": sorted(left["named"].keys() - right["named"].keys()),
            "leader_changed": sorted(
                key
                for key in left["named"].keys() & right["named"].keys()
                if left["named"][key] != right["named"][key]
            ),
        }
    if args.technique_pr:
        left, right = technique_pr(base), technique_pr(current)
        report["technique_pr"] = {
            "base": left,
            "current": right,
            "delta": {
                key: (right[key] - left[key])
                if isinstance(left.get(key), int | float)
                and isinstance(right.get(key), int | float)
                else None
                for key in (
                    "technique_micro_precision",
                    "technique_micro_recall",
                    "median_accepted_techniques",
                    "mean_technique_recall",
                )
            },
        }
    if args.combiner_shift:
        report["combiner_shift"] = combiner_shift(base, current)
    _emit(report, args.output)
    return status


# ----------------------------------------------------------------------------- evidenceforge


def _red_herrings_accepted(cases: Sequence[dict[str, Any]]) -> int:
    return sum(
        1
        for case in cases
        if case.get("label_quality") == "red_herring" and case.get("accepted_technique_ids")
    )


def cmd_evidenceforge(args: argparse.Namespace) -> int:
    base, current = _load(args.base), _load(args.current)
    modes = sorted(set(base.get("results") or {}) & set(current.get("results") or {}))
    if not modes:
        raise NotComparableError("the artifacts share no EvidenceForge mode")
    report: dict[str, Any] = {
        "kind": "evidenceforge",
        "base": str(args.base),
        "current": str(args.current),
        "modes": {},
    }
    identical = True
    names = ("micro_precision", "micro_recall", "micro_f1", "hit_at_1", "hit_at_10")
    for mode in modes:
        left, right = base["results"][mode], current["results"][mode]
        left_cases = {case["case_id"]: case for case in left.get("cases", [])}
        right_cases = {case["case_id"]: case for case in right.get("cases", [])}
        differing = []
        input_changed = 0
        for case_id in sorted(left_cases.keys() | right_cases.keys()):
            a, b = left_cases.get(case_id), right_cases.get(case_id)
            if a is None or b is None:
                differing.append(
                    {"case_id": case_id, "missing_in": "base" if a is None else "current"}
                )
                continue
            input_changed += int(a.get("model_input_hash") != b.get("model_input_hash"))
            keys, _, _ = compare_records(a, b)
            if keys:
                differing.append({"case_id": case_id, "fields": keys})
        metric_keys, _, _ = compare_records(left.get("metrics") or {}, right.get("metrics") or {})
        mode_identical = not differing and not metric_keys
        identical = identical and mode_identical
        metrics_a, metrics_b = left.get("metrics") or {}, right.get("metrics") or {}
        report["modes"][mode] = {
            "identical": mode_identical,
            "cases": len(left_cases),
            "differing_cases": len(differing),
            "model_input_changed": input_changed,
            "examples": differing[:20],
            "metric_differences": metric_keys,
            "metrics": {
                name: {
                    "base": metrics_a.get(name),
                    "current": metrics_b.get(name),
                    "delta": float(metrics_b[name]) - float(metrics_a[name])
                    if name in metrics_a and name in metrics_b
                    else None,
                }
                for name in names
            },
            "red_herrings_accepted": {
                "base": _red_herrings_accepted(list(left_cases.values())),
                "current": _red_herrings_accepted(list(right_cases.values())),
            },
        }
    report["identical"] = identical
    _emit(report, args.output)
    return 1 if args.expect_identical and not identical else 0


# ----------------------------------------------------------------------------- summary


def summarize_ledger(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs = list(lines)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": [
            {
                key: run.get(key)
                for key in (
                    "run_id",
                    "stage",
                    "gate",
                    "verdict",
                    "recorded_at",
                    "git_commit",
                    "llm_cost_usd",
                )
            }
            | {
                "artifacts": [
                    {"path": item.get("path"), "artifact_hash": item.get("artifact_hash")}
                    for item in run.get("artifacts") or []
                ],
                "chroma_clean": (run.get("chroma") or {}).get("clean"),
            }
            for run in runs
        ],
        "verdicts": dict(sorted(Counter(str(run.get("verdict")) for run in runs).items())),
        "llm_cost_usd_total": round(sum(float(run.get("llm_cost_usd") or 0.0) for run in runs), 4),
        "git_commits": sorted(
            {str(run.get("git_commit")) for run in runs if run.get("git_commit")}
        ),
        "chroma_unclean_runs": [
            run.get("run_id") for run in runs if (run.get("chroma") or {}).get("clean") is False
        ],
    }


def cmd_summary(args: argparse.Namespace) -> int:
    lines = [
        json.loads(line)
        for line in args.ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _emit(summarize_ledger(lines), args.output)
    return 0


# ----------------------------------------------------------------------------- main


def _emit(report: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _write_json(output, report)
        print(f"wrote {output}", file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    mapping = commands.add_parser("mapping", help="benchmark_mapping.py artifacts")
    mapping.add_argument("base", type=Path)
    mapping.add_argument("current", type=Path, nargs="?")
    mapping.add_argument("--modes", nargs="+", help="result modes (default: all shared)")
    mapping.add_argument("--expect-identical", action="store_true")
    mapping.add_argument("--paired-bootstrap", type=int, default=0, metavar="N")
    mapping.add_argument("--seed", type=int, default=7)
    mapping.add_argument("--pr-curve", action="store_true")
    mapping.add_argument("--rethreshold", type=float, metavar="T")
    mapping.add_argument("--allow-mismatch", action="store_true")
    mapping.add_argument("--kb", type=Path, help="knowledge base for parent-aware metrics")
    mapping.add_argument("--bootstrap-samples", type=int, help="default: the artifact's (300)")
    mapping.add_argument("--bootstrap-seed", type=int, help="default: the artifact's seed")
    mapping.add_argument(
        "--output",
        type=Path,
        help="where to write the result: the comparison report, or with --rethreshold the "
        "re-thresholded artifact",
    )
    mapping.set_defaults(handler=cmd_mapping)

    attribution = commands.add_parser("attribution", help="attribution_benchmark.py artifacts")
    attribution.add_argument("base", type=Path)
    attribution.add_argument("current", type=Path)
    attribution.add_argument("--expect-identical", action="store_true")
    attribution.add_argument("--names", action="store_true")
    attribution.add_argument("--technique-pr", action="store_true")
    attribution.add_argument("--combiner-shift", action="store_true")
    attribution.add_argument("--output", type=Path)
    attribution.set_defaults(handler=cmd_attribution)

    evidenceforge = commands.add_parser("evidenceforge", help="evidenceforge_eval.py artifacts")
    evidenceforge.add_argument("base", type=Path)
    evidenceforge.add_argument("current", type=Path)
    evidenceforge.add_argument("--expect-identical", action="store_true")
    evidenceforge.add_argument("--output", type=Path)
    evidenceforge.set_defaults(handler=cmd_evidenceforge)

    summary = commands.add_parser("summary", help="compose the run ledger")
    summary.add_argument("--ledger", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    summary.set_defaults(handler=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except NotComparableError as error:
        print(f"not comparable: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
