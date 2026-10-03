"""Fit the calibrated confidence labels (display only) on the design folds; evaluate the rest.

Fits the three logistic regressions of ``apt_classifier/confidence_labels.py`` (P(leader right),
P(truth in the top two), P(truth in the top three); sklearn ``LogisticRegression(C=1.0)``,
lbfgs, raw features) on attribution-benchmark rows of the *design* seeds only (41 and 53;
removed actors count as wrong and their truth is never in the top three). Thresholds are chosen
on out-of-fold predictions from ``GroupKFold(5)`` by group id (both seeds hold out the same
groups) with one-sided 95% Clopper-Pearson bounds:

- strong: the smallest grid value at which the leader-error upper bound is <= 0.05;
- set: the smallest grid value at which, among the non-strong rows, the shown set misses the
  truth in <= 10% of rows with an upper bound <= 0.20.

Selection uses the probability rule alone; the insufficient-evidence -> open override is
definitional and applied at display time (and in every evaluation below). CTI-HAL and the
evaluation seeds are refused as design input. ``--evaluate`` artifacts are scored with the
fitted labels as report-only numbers; ``--evaluate-only`` scores them with the frozen artifact
and never refits (the pre-registered check on fresh seeds; ``--acceptance-bar`` adds A1-A3).
Offline, seconds, no OpenRouter and no Chroma.

    PYTHONPATH=. .venv/bin/python scripts/fit_confidence_labels.py \\
        --design data/audits/top3/design_seed41.json data/audits/top3/design_seed53.json \\
        --evaluate data/audits/attribution_heldout_current.json \\
            data/audits/attribution_cti_hal_current.json \\
            data/audits/attribution_cti_hal_as_reported.json

    PYTHONPATH=. .venv/bin/python scripts/fit_confidence_labels.py --evaluate-only \\
        --evaluate data/audits/top3/confirm_seed43.json data/audits/top3/confirm_seed47.json \\
        --acceptance-bar --report data/audits/confidence_labels_confirm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.combiner import AttributionCombiner  # noqa: E402
from apt_classifier.confidence_labels import (  # noqa: E402
    FEATURE_NAMES,
    FOLD_EVALUATION,
    INSUFFICIENT_STAGE,
    LOGIT_CLIP,
    MODELS,
    NAMED_STAGE,
    ConfidenceCalibrator,
    ConfidenceInputs,
    combiner_sha256,
    decide,
    features_from_row,
    row_shortlist,
)
from apt_classifier.config import Settings  # noqa: E402

FOLD_BENCHMARK = "attribution_text_heldout"
REPORT_BENCHMARK = "attribution_cti_hal_reports"
# seed 7 is the shipped benchmark's evaluation seed; 43 and 47 are the pre-registered fresh check
EVALUATION_SEEDS = frozenset({7, 43, 47})
DEFAULT_DESIGN = (
    REPO_ROOT / "data" / "audits" / "top3" / "design_seed41.json",
    REPO_ROOT / "data" / "audits" / "top3" / "design_seed53.json",
)
GRID = (0.5, 0.995, 0.005)
STRONG_ALPHA = 0.05
SET_MAX_MISS = 0.10
SET_BOUND = 0.20
CV_FOLDS = 5
ECE_BINS = 10
# pre-registered acceptance bar on the fresh seeds (spec section 8)
BAR_STRONG_POOLED = 0.05
BAR_STRONG_PER_SEED = 0.08
BAR_ECE = 0.06
BAR_SET_COVERAGE = 0.80


@dataclass(frozen=True)
class Row:
    """One benchmark row as the labels see it, with its truth."""

    inputs: ConfidenceInputs | None
    stage: str | None
    group_ids: tuple[str, ...]
    probabilities: dict[str, float]
    named_cluster: tuple[str, ...]
    truth: str | None  # None for a removed actor: never in the shortlist
    group: str  # the fold's group (the GroupKFold key)
    removed: bool
    leader_related_to_truth: bool

    @property
    def rank(self) -> int | None:
        if self.truth is None or self.truth not in self.group_ids:
            return None
        return self.group_ids.index(self.truth) + 1

    def hit(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def _display(path: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute)


def load_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") not in {FOLD_BENCHMARK, REPORT_BENCHMARK}:
        raise SystemExit(f"{path}: not an attribution benchmark artifact")
    payload["_path"] = path
    return payload


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload.get("parameters") or {}
    recorded = (payload.get("fingerprint_inputs") or {}).get("combiner_artifact")
    return {
        "source": _display(payload["_path"]),
        "artifact_hash": payload.get("artifact_hash"),
        "benchmark": payload.get("benchmark"),
        "seed": parameters.get("seed"),
        "actor_names": parameters.get("actor_names"),
        "source_fingerprint": (payload.get("fingerprint_inputs") or {}).get("source_fingerprint"),
        "combiner_artifact": recorded or "not recorded",
    }


def benchmark_rows(payload: dict[str, Any], *, removed: bool = True) -> list[Row]:
    """Known rows, plus the removed-actor rows (their leader is always wrong) when asked."""
    rows: list[Row] = []
    arms = ("known", "unknown") if removed else ("known",)
    for arm in arms:
        for item in payload.get(arm) or []:
            group_ids, probabilities = row_shortlist(item)
            rows.append(
                Row(
                    inputs=features_from_row(item),
                    stage=item.get("stage"),
                    group_ids=tuple(group_ids),
                    probabilities=probabilities,
                    named_cluster=tuple(item.get("named_cluster") or ()),
                    truth=str(item["group_id"]) if arm == "known" else None,
                    group=str(item["group_id"]),
                    removed=arm == "unknown",
                    leader_related_to_truth=bool(item.get("leader_related_to_truth")),
                )
            )
    return rows


# ----------------------------------------------------------------------------- statistics


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound of a binomial rate."""
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Two-sided (1 - alpha) interval of a binomial rate."""
    from scipy.stats import beta

    if n <= 0:
        return None
    low = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(low, 4), round(high, 4)]


def expected_calibration_error(
    p: Sequence[float], y: Sequence[bool], bins: int = ECE_BINS
) -> float | None:
    """Equal-width ECE (None without rows)."""
    probabilities = np.asarray(p, dtype=float)
    outcomes = np.asarray(y, dtype=float)
    if not len(probabilities):
        return None
    index = np.minimum((probabilities * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            share = mask.sum() / len(probabilities)
            total += share * abs(probabilities[mask].mean() - outcomes[mask].mean())
    return float(total)


def grid() -> list[float]:
    low, high, step = GRID
    count = int(round((high - low) / step)) + 1
    return [round(low + i * step, 3) for i in range(count)]


# ----------------------------------------------------------------------------- fitting


def design_matrix(rows: Sequence[Row], model: str, clip: float = LOGIT_CLIP) -> np.ndarray:
    return np.asarray([row.inputs.vector(model, clip=clip) for row in rows], dtype=float)  # type: ignore[union-attr]


def targets(rows: Sequence[Row], model: str) -> np.ndarray:
    k = MODELS.index(model) + 1
    return np.asarray([row.hit(k) for row in rows], dtype=int)


def fit_models(rows: Sequence[Row]) -> tuple[dict[str, float], dict[str, tuple[float, ...]]]:
    from sklearn.linear_model import LogisticRegression

    intercepts: dict[str, float] = {}
    weights: dict[str, tuple[float, ...]] = {}
    for model in MODELS:
        fitted = LogisticRegression(C=1.0).fit(design_matrix(rows, model), targets(rows, model))
        intercepts[model] = float(fitted.intercept_[0])
        weights[model] = tuple(float(w) for w in fitted.coef_[0])
    return intercepts, weights


def fold_calibrators(
    rows: Sequence[Row], template: ConfidenceCalibrator, folds: int = CV_FOLDS
) -> list[ConfidenceCalibrator]:
    """Per row, the calibrator fitted without that row's group (GroupKFold by group id)."""
    from sklearn.model_selection import GroupKFold

    assigned: list[ConfidenceCalibrator | None] = [None] * len(rows)
    groups = np.asarray([row.group for row in rows])
    for train, test in GroupKFold(n_splits=folds).split(np.zeros(len(rows)), groups=groups):
        intercepts, weights = fit_models([rows[i] for i in train])
        model = replace(template, intercepts=intercepts, weights=weights)
        for i in test:
            assigned[i] = model
    assert all(model is not None for model in assigned)
    return assigned  # type: ignore[return-value]


def select_thresholds(
    rows: Sequence[Row], predictions: Sequence[dict[str, float]], *, override: bool = False
) -> dict[str, Any]:
    """Smallest grid values meeting the strong and set rules (out-of-fold predictions).
    ``override`` applies the insufficient-evidence -> open rule first (a robustness check)."""
    eligible = [
        not override or (row.stage != INSUFFICIENT_STAGE and bool(row.group_ids)) for row in rows
    ]
    strong: dict[str, Any] | None = None
    for t in grid():
        chosen = [i for i, p in enumerate(predictions) if eligible[i] and p["leader"] >= t]
        errors = sum(1 for i in chosen if not rows[i].hit(1))
        bound = clopper_pearson_upper(errors, len(chosen), STRONG_ALPHA)
        if chosen and bound <= STRONG_ALPHA:
            strong = {"t": t, "n": len(chosen), "errors": errors, "upper": round(bound, 4)}
            break
    if strong is None:
        raise SystemExit("no strong threshold meets the bound on the design data")
    shown: dict[str, Any] | None = None
    for t in grid():
        n = misses = 0
        for i, p in enumerate(predictions):
            if not eligible[i] or p["leader"] >= strong["t"]:
                continue
            size = len(rows[i].group_ids)
            if size >= 2 and p["top2"] >= t:
                k = 2
            elif size >= 3 and p["top3"] >= t:
                k = 3
            else:
                continue
            n += 1
            misses += 0 if rows[i].hit(k) else 1
        bound = clopper_pearson_upper(misses, n, STRONG_ALPHA)
        if n and misses / n <= SET_MAX_MISS and bound <= SET_BOUND:
            shown = {"t": t, "n": n, "misses": misses, "upper": round(bound, 4)}
            break
    if shown is None:
        raise SystemExit("no set threshold meets the bound on the design data")
    return {"strong": strong, "set": shown}


# ----------------------------------------------------------------------------- evaluation


def evaluate(
    rows: Sequence[Row],
    calibrators: Sequence[ConfidenceCalibrator],
    *,
    description: str,
    meta: dict[str, Any] | None = None,
    variant: str = "known+removed",
) -> dict[str, Any]:
    """Label every row with the shared rule (``decide``) and count how each label held up."""
    usable = [(row, model) for row, model in zip(rows, calibrators, strict=True) if row.inputs]
    total = len(usable)
    counts = {
        "strong": {"n": 0, "right": 0},
        "set": {
            "n": 0,
            "right": 0,
            "by_size": {"2": {"n": 0, "right": 0}, "3": {"n": 0, "right": 0}},
        },
        "open": {"n": 0, "truth_in_top3": 0},
    }
    strong_errors = {"n": 0, "removed_actor": 0, "leader_related_to_truth": 0}
    named = {"n": 0, "strong": 0, "lower": 0}
    misses = {"n": 0, "labelled_open": 0}
    p_leader: list[float] = []
    p_top3: list[float] = []
    raw_leader: list[float] = []
    raw_top3: list[float] = []
    for row, model in usable:
        block = decide(
            model,
            row.inputs,
            stage=row.stage,
            group_ids=row.group_ids,
            probabilities=row.probabilities,
            named_cluster=row.named_cluster,
        )
        predicted = model.predict(row.inputs)  # type: ignore[arg-type]
        p_leader.append(predicted["leader"])
        p_top3.append(predicted["top3"])
        raw_leader.append(row.inputs.probabilities[0])  # type: ignore[union-attr]
        raw_top3.append(min(sum(row.inputs.probabilities), 1.0))  # type: ignore[union-attr]
        label = block["label"]
        right = row.truth is not None and row.truth in block["group_ids"]
        if label == "strong":
            counts["strong"]["n"] += 1
            counts["strong"]["right"] += int(right)
            if not right:
                strong_errors["n"] += 1
                strong_errors["removed_actor"] += int(row.removed)
                strong_errors["leader_related_to_truth"] += int(row.leader_related_to_truth)
        elif label == "set":
            size = str(block["set_size"])
            counts["set"]["n"] += 1
            counts["set"]["right"] += int(right)
            by_size = counts["set"]["by_size"].setdefault(size, {"n": 0, "right": 0})
            by_size["n"] += 1
            by_size["right"] += int(right)
        else:
            counts["open"]["n"] += 1
            counts["open"]["truth_in_top3"] += int(row.hit(3))
        if row.stage == NAMED_STAGE:
            named["n"] += 1
            named["strong" if label == "strong" else "lower"] += 1
        if not row.hit(3):
            misses["n"] += 1
            misses["labelled_open"] += int(label == "open")
    for key in ("strong", "set", "open"):
        counts[key]["share"] = round(counts[key]["n"] / total, 4) if total else None
    strong_n, strong_right = counts["strong"]["n"], counts["strong"]["right"]
    counts["strong"]["error_rate"] = (
        round((strong_n - strong_right) / strong_n, 4) if strong_n else None
    )
    counts["strong"]["error_upper_95"] = round(
        clopper_pearson_upper(strong_n - strong_right, strong_n), 4
    )
    set_n, set_right = counts["set"]["n"], counts["set"]["right"]
    counts["set"]["coverage"] = round(set_right / set_n, 4) if set_n else None
    counts["set"]["coverage_ci_95"] = clopper_pearson_interval(set_right, set_n)
    y_leader = [row.hit(1) for row, _ in usable]
    y_top3 = [row.hit(3) for row, _ in usable]
    return {
        "description": description,
        **(meta or {}),
        "variant": variant,
        "rows": total,
        "skipped_without_combiner": len(rows) - total,
        "labels": counts,
        "strong_errors": strong_errors,
        "named": named,
        "top3_misses": misses,
        "ece": {
            "bins": ECE_BINS,
            "leader": _rounded(expected_calibration_error(p_leader, y_leader)),
            "leader_raw": _rounded(expected_calibration_error(raw_leader, y_leader)),
            "top3": _rounded(expected_calibration_error(p_top3, y_top3)),
            "top3_raw": _rounded(expected_calibration_error(raw_top3, y_top3)),
        },
    }


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _evaluation_key(meta: dict[str, Any], taken: set[str]) -> str:
    if meta["benchmark"] == REPORT_BENCHMARK:
        key = f"cti_hal_{'as_reported' if meta['actor_names'] == 'as_reported' else 'anonymized'}"
    else:
        key = f"seed{meta['seed']}"
    candidate, suffix = key, 2
    while candidate in taken:
        candidate, suffix = f"{key}_{suffix}", suffix + 1
    return candidate


def _description(meta: dict[str, Any], variant: str) -> str:
    if meta["benchmark"] == REPORT_BENCHMARK:
        names = "as reported" if meta["actor_names"] == "as_reported" else "anonymized"
        return f"CTI-HAL real reports, actor names {names} (report only)"
    removed = "removed actors count as wrong" if variant == "known+removed" else "known only"
    return f"ATT&CK test folds, seed {meta['seed']} ({removed}; report only)"


def check_compatible(
    payload: dict[str, Any], attack_version: str, fingerprint: str, digest: str
) -> None:
    """Refuse rows from another ATT&CK version or (when recorded) another combiner artifact.
    Rows written before the benchmark recorded its combiner artifact pass on "active" alone."""
    path = payload["_path"]
    if str(payload.get("attack_version") or "") != attack_version:
        raise SystemExit(f"{path}: ATT&CK {payload.get('attack_version')} != {attack_version}")
    recorded = (payload.get("fingerprint_inputs") or {}).get("combiner_artifact")
    if isinstance(recorded, dict) and (
        recorded.get("sha256") != digest or recorded.get("pipeline_fingerprint") != fingerprint
    ):
        raise SystemExit(f"{path}: rows were scored by another combiner artifact")


def evaluate_sets(
    payloads: Sequence[dict[str, Any]], calibrator: ConfidenceCalibrator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for payload in payloads:
        check_compatible(
            payload,
            calibrator.attack_version,
            calibrator.combiner_fingerprint,
            calibrator.combiner_sha256,
        )
        meta = _meta(payload)
        key = _evaluation_key(meta, set(out))
        rows = benchmark_rows(payload, removed=True)
        out[key] = evaluate(
            rows,
            [calibrator] * len(rows),
            description=_description(meta, "known+removed"),
            meta=meta,
        )
        if meta["benchmark"] == FOLD_BENCHMARK and payload.get("unknown"):
            known = benchmark_rows(payload, removed=False)
            out[f"{key}_known_only"] = evaluate(
                known,
                [calibrator] * len(known),
                description=_description(meta, "known only"),
                meta=meta,
                variant="known only",
            )
    return out


def acceptance_bar(
    payloads: Sequence[dict[str, Any]], calibrator: ConfidenceCalibrator
) -> dict[str, Any]:
    """A1-A3 of the pre-registered bar, pooled over the ATT&CK fold sets given (known+removed).
    A4 (labels never change decisions, rankings or exports) is a test, not a measurement."""
    folds = [p for p in payloads if p.get("benchmark") == FOLD_BENCHMARK]
    if not folds:
        raise SystemExit("--acceptance-bar needs ATT&CK fold artifacts in --evaluate")
    per_seed = {}
    pooled_rows: list[Row] = []
    for payload in folds:
        rows = benchmark_rows(payload, removed=True)
        pooled_rows += rows
        result = evaluate(rows, [calibrator] * len(rows), description="", meta=_meta(payload))
        per_seed[str(result["seed"])] = result
    pooled = evaluate(pooled_rows, [calibrator] * len(pooled_rows), description="pooled fold sets")
    strong = pooled["labels"]["strong"]
    seed_rates = {
        seed: result["labels"]["strong"]["error_rate"] for seed, result in per_seed.items()
    }
    a1 = (
        strong["error_rate"] is not None
        and strong["error_rate"] <= BAR_STRONG_POOLED
        and all(rate is not None and rate <= BAR_STRONG_PER_SEED for rate in seed_rates.values())
    )
    a2 = pooled["ece"]["leader"] is not None and pooled["ece"]["leader"] <= BAR_ECE
    coverage = pooled["labels"]["set"]["coverage"]
    a3 = None if coverage is None else coverage >= BAR_SET_COVERAGE
    return {
        "seeds": sorted(per_seed),
        "rows": pooled["rows"],
        "A1_strong_error": {
            "pooled": strong["error_rate"],
            "pooled_upper_95": strong["error_upper_95"],
            "per_seed": seed_rates,
            "bar": {"pooled": BAR_STRONG_POOLED, "per_seed": BAR_STRONG_PER_SEED},
            "passed": a1,
        },
        "A2_ece_leader": {"pooled": pooled["ece"]["leader"], "bar": BAR_ECE, "passed": a2},
        "A3_set_coverage": {
            "pooled": coverage,
            "n": pooled["labels"]["set"]["n"],
            "ci_95": pooled["labels"]["set"]["coverage_ci_95"],
            "bar": BAR_SET_COVERAGE,
            "passed": a3,
            **({"note": "no 'one of these' rows"} if a3 is None else {}),
        },
        "A4_display_only": "tests/test_confidence_labels.py (decisions, rankings, exports)",
        "passed": bool(a1 and a2 and a3),
        "pooled": pooled,
        "per_seed": per_seed,
    }


# ----------------------------------------------------------------------------- main


def _combiner_pin(path: Path) -> tuple[str, str, str]:
    combiner = AttributionCombiner.load(path)
    if combiner is None:
        raise SystemExit(f"combiner artifact not found: {path}")
    return combiner.pipeline_fingerprint, combiner_sha256(path), combiner.attack_version


def _check_design(payloads: Sequence[dict[str, Any]], evaluate_hashes: set[str]) -> None:
    for payload in payloads:
        path = payload["_path"]
        seed = (payload.get("parameters") or {}).get("seed")
        if payload.get("benchmark") != FOLD_BENCHMARK:
            raise SystemExit(f"{path}: CTI-HAL reports are evaluation only, never design input")
        if seed in EVALUATION_SEEDS:
            raise SystemExit(f"{path}: seed {seed} is an evaluation seed, never design input")
        if payload.get("artifact_hash") in evaluate_hashes:
            raise SystemExit(f"{path}: also passed to --evaluate")
        rows = list(payload.get("known") or []) + list(payload.get("unknown") or [])
        inactive = [row for row in rows if features_from_row(row) is None]
        if inactive:
            raise SystemExit(f"{path}: {len(inactive)} rows without an active combiner")
    versions = {str(payload.get("attack_version") or "") for payload in payloads}
    if len(versions) != 1:
        raise SystemExit(f"design artifacts mix ATT&CK versions: {sorted(versions)}")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, nargs="+", default=list(DEFAULT_DESIGN))
    parser.add_argument("--evaluate", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="score --evaluate with the frozen --artifact; never refit or rewrite it",
    )
    parser.add_argument(
        "--acceptance-bar",
        action="store_true",
        help="A1-A3 of the pre-registered bar, pooled over the --evaluate fold sets",
    )
    # defaults follow the application settings, so the labels are pinned to the combiner the
    # app loads and written where it reads them
    settings = Settings()
    parser.add_argument(
        "--combiner",
        type=Path,
        default=Path(settings.attribution_combiner_path),
        help="the shipped combiner artifact the rows were scored with (pinned)",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(settings.confidence_labels_path),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    evaluated = [load_benchmark(path) for path in args.evaluate]
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    if args.evaluate_only:
        if not evaluated:
            parser.error("--evaluate-only needs --evaluate artifacts")
        calibrator = ConfidenceCalibrator.load(args.artifact)
        if calibrator is None:
            parser.error(f"artifact not found: {args.artifact}")
        design = {
            item.get("artifact_hash")
            for item in calibrator.provenance.get("design_artifacts", [])  # type: ignore[union-attr]
        }
        for payload in evaluated:
            if payload.get("artifact_hash") in design:
                raise SystemExit(f"{payload['_path']}: a design artifact, not an evaluation set")
        report = {
            "generated_at": generated_at,
            "mode": "evaluate_only (frozen artifact, no refit)",
            "artifact": _display(args.artifact),
            "artifact_sha256_24": hashlib.sha256(args.artifact.read_bytes()).hexdigest()[:24],
            "thresholds": calibrator.thresholds,
            "evaluation": evaluate_sets(evaluated, calibrator),  # type: ignore[arg-type]
        }
        if args.acceptance_bar:
            report["acceptance_bar"] = acceptance_bar(evaluated, calibrator)  # type: ignore[arg-type]
        target = args.report or REPO_ROOT / "data" / "audits" / "confidence_labels_frozen_eval.json"
        _write(target, report)
        _print_summary(report["evaluation"], report.get("acceptance_bar"))
        print(f"wrote {_display(target)}", file=sys.stderr)
        return 0

    import sklearn

    designs = [load_benchmark(path) for path in args.design]
    _check_design(designs, {str(p.get("artifact_hash")) for p in evaluated})
    fingerprint, digest, combiner_version = _combiner_pin(args.combiner)
    attack_version = str(designs[0].get("attack_version") or "")
    if combiner_version and combiner_version != attack_version:
        raise SystemExit(f"combiner ATT&CK {combiner_version} != design {attack_version}")
    for payload in designs:
        check_compatible(payload, attack_version, fingerprint, digest)
    rows = [row for payload in designs for row in benchmark_rows(payload, removed=True)]
    intercepts, weights = fit_models(rows)
    template = ConfidenceCalibrator(
        intercepts=intercepts,
        weights=weights,
        thresholds={"strong": 0.5, "set": 0.5},
        attack_version=attack_version,
        combiner_fingerprint=fingerprint,
        combiner_sha256=digest,
    )
    folds = fold_calibrators(rows, template)
    oof = [model.predict(row.inputs) for row, model in zip(rows, folds, strict=True)]  # type: ignore[arg-type]
    chosen = select_thresholds(rows, oof)
    thresholds = {"strong": chosen["strong"]["t"], "set": chosen["set"]["t"]}
    calibrator = replace(
        template,
        thresholds=thresholds,
        selection={
            "grid": list(GRID),
            "strong_alpha": STRONG_ALPHA,
            "set_max_miss": SET_MAX_MISS,
            "set_bound": SET_BOUND,
            "bound": "one-sided 95% Clopper-Pearson",
            "cv": f"GroupKFold({CV_FOLDS}) by group_id",
            "rule": "probability only; the insufficient-evidence -> open override applies at "
            "display and in every evaluation",
            "strong": chosen["strong"],
            "set": chosen["set"],
            "with_override": select_thresholds(rows, oof, override=True),
        },
        provenance={
            "fitted_at": generated_at,
            "variant": "known+removed",
            "rows": len(rows),
            "design_artifacts": [
                {
                    "path": _display(payload["_path"]),
                    "artifact_hash": payload.get("artifact_hash"),
                    "seed": (payload.get("parameters") or {}).get("seed"),
                    "source_fingerprint": (payload.get("fingerprint_inputs") or {}).get(
                        "source_fingerprint"
                    ),
                }
                for payload in designs
            ],
            "combiner_artifact": _display(args.combiner),
            "model": "LogisticRegression(C=1.0, lbfgs), raw features",
            "fit_script": "scripts/fit_confidence_labels.py",
            "sklearn": sklearn.__version__,
        },
    )
    folds = [replace(model, thresholds=thresholds) for model in folds]
    evaluation = {
        FOLD_EVALUATION: evaluate(
            rows,
            folds,
            description="ATT&CK test folds, seeds 41 and 53 (out-of-fold; removed actors count "
            "as wrong)",
            meta={"sources": [_display(p["_path"]) for p in designs]},
        ),
        **evaluate_sets(evaluated, calibrator),
    }
    calibrator = replace(calibrator, evaluation=evaluation)
    calibrator.save(args.artifact)
    report = {
        "generated_at": generated_at,
        "mode": "fit",
        "artifact": _display(args.artifact),
        "artifact_sha256_24": hashlib.sha256(args.artifact.read_bytes()).hexdigest()[:24],
        "feature_names": list(FEATURE_NAMES),
        "models": calibrator.to_dict()["models"],
        "thresholds": thresholds,
        "selection": calibrator.selection,
        "provenance": calibrator.provenance,
        "evaluation": evaluation,
    }
    target = args.report or REPO_ROOT / "data" / "audits" / "confidence_labels_eval.json"
    _write(target, report)
    for model in MODELS:
        coefficients = ", ".join(f"{w:.6f}" for w in weights[model])
        print(f"{model:6s} b0 {intercepts[model]:.6f}  w [{coefficients}]", file=sys.stderr)
    print(
        f"thresholds strong {thresholds['strong']} ({chosen['strong']}), "
        f"set {thresholds['set']} ({chosen['set']})",
        file=sys.stderr,
    )
    _print_summary(evaluation, None)
    print(f"wrote {_display(args.artifact)} and {_display(target)}", file=sys.stderr)
    return 0


def _print_summary(evaluation: dict[str, Any], bar: dict[str, Any] | None) -> None:
    for key, item in evaluation.items():
        labels = item["labels"]
        ece = item["ece"]
        print(
            f"{key:28s} n {item['rows']:4d}  strong {labels['strong']['n']}: "
            f"{labels['strong']['right']} right  set {labels['set']['n']}: "
            f"{labels['set']['right']}  open {labels['open']['n']}: "
            f"{labels['open']['truth_in_top3']} in top 3  ECE leader {ece['leader']:.3f} "
            f"({ece['leader_raw']:.3f}) top3 {ece['top3']:.3f} ({ece['top3_raw']:.3f})",
            file=sys.stderr,
        )
    if bar is not None:
        print(f"acceptance bar passed: {bar['passed']}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
