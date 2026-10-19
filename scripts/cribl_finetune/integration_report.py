"""The product-integration gate's report (prereg ``cribl_product_integration_2026-10-19``): the
mapping arms (off / proposer / head) on the public test splits and CTI-HAL with paired case
bootstraps, where the change comes from, the head-only candidates, the schema-4 model's weights,
and the no-regression checks (EvidenceForge clinic, public regression check, attribution).

  python scripts/cribl_finetune/integration_report.py [--dir data/audits/cribl_ft/integration]
  -> <dir>/integration_report.json and a table on stdout

Reads artifacts only (``scripts/benchmark_mapping.py``, ``scripts/evidenceforge_eval.py``,
``scripts/attribution_benchmark.py``); no model, no Chroma, no LLM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "data" / "audits" / "cribl_ft" / "integration"
HEAD_MODEL = REPO_ROOT / "data" / "models" / "acceptance" / "acceptance_model_technique_head.json"
ARMS = ("off", "proposer", "head", "multi")
SETS = ("public", "cti_hal")
BOOTSTRAP = 1000
SEED = 7
ACCEPTED_STATE = "auto_accepted"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def case_rows(artifact: dict[str, Any], mode: str = "hybrid") -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in artifact["results"][mode]["cases"]}


def counts(row: dict[str, Any]) -> tuple[int, int, int]:
    expected = set(row["expected_technique_ids"])
    accepted = set(row["accepted_technique_ids"])
    return len(expected & accepted), len(accepted - expected), len(expected - accepted)


def micro(triples: list[tuple[int, int, int]]) -> dict[str, float]:
    tp = sum(t[0] for t in triples)
    fp = sum(t[1] for t in triples)
    fn = sum(t[2] for t in triples)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1}


def paired(base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Point estimates of both arms and the paired case-bootstrap intervals of other - base."""
    ids = [case_id for case_id in base if case_id in other]
    a = [counts(base[case_id]) for case_id in ids]
    b = [counts(other[case_id]) for case_id in ids]
    point_a, point_b = micro(a), micro(b)
    rng = random.Random(SEED)
    deltas: dict[str, list[float]] = {name: [] for name in point_a}
    n = len(ids)
    for _ in range(BOOTSTRAP):
        sample = [rng.randrange(n) for _ in range(n)]
        ma = micro([a[i] for i in sample])
        mb = micro([b[i] for i in sample])
        for name in deltas:
            deltas[name].append(mb[name] - ma[name])
    out: dict[str, Any] = {"cases": n, "base": point_a, "arm": point_b, "delta": {}}
    for name, values in deltas.items():
        values.sort()
        out["delta"][name] = {
            "estimate": round(point_b[name] - point_a[name], 4),
            "ci95": [
                round(values[int(0.025 * (BOOTSTRAP - 1))], 4),
                round(values[int(0.975 * (BOOTSTRAP - 1))], 4),
            ],
        }
    return out


def breakdown(base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Where the arm changes things, by what the base arm did with the statement."""
    buckets: dict[str, dict[str, int]] = {}
    for case_id, row in base.items():
        if case_id not in other:
            continue
        expected = set(row["expected_technique_ids"])
        if not expected:
            continue
        accepted = set(row["accepted_technique_ids"])
        if not accepted:
            kind = "base_accepted_nothing"
        elif accepted & expected:
            kind = "base_accepted_a_right_one"
        else:
            kind = "base_accepted_only_wrong"
        after = set(other[case_id]["accepted_technique_ids"])
        bucket = buckets.setdefault(
            kind,
            {"statements": 0, "arm_right": 0, "arm_only_wrong": 0, "arm_nothing": 0},
        )
        bucket["statements"] += 1
        if after & expected:
            bucket["arm_right"] += 1
        elif after:
            bucket["arm_only_wrong"] += 1
        else:
            bucket["arm_nothing"] += 1
    return buckets


def head_only(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The candidates that exist only because the head proposed them (within the stored rows)."""
    appended = accepted = correct = accepted_correct = 0
    for row in rows.values():
        expected = set(row["expected_technique_ids"])
        for candidate in row.get("candidates", ()):
            if not candidate.get("th_only"):
                continue
            appended += 1
            right = candidate["technique_id"] in expected
            correct += int(right)
            if str(candidate.get("state", "")).lower().endswith(ACCEPTED_STATE):
                accepted += 1
                accepted_correct += int(right)
    return {
        "appended": appended,
        "correct": correct,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
    }


def multi_label(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recall on statements with two or more gold labels, and how often more than one technique
    is accepted (and how many of those extra acceptances are right)."""
    statements = labels = found = all_found = 0
    multi_accept = extra_right = extra_wrong = 0
    for row in rows.values():
        expected = set(row["expected_technique_ids"])
        accepted = set(row["accepted_technique_ids"])
        if len(accepted) > 1:
            multi_accept += 1
            right = len(accepted & expected)
            # the "extra" acceptances beyond one: right ones count as recovered answers
            extra_right += max(right - 1, 0)
            extra_wrong += len(accepted - expected) - (1 if right == 0 and accepted else 0)
        if len(expected) < 2:
            continue
        statements += 1
        labels += len(expected)
        found += len(expected & accepted)
        all_found += int(expected <= accepted)
    return {
        "multi_label_statements": statements,
        "labels": labels,
        "labels_found": found,
        "label_recall": round(found / labels, 4) if labels else None,
        "all_labels_found": all_found,
        "statements_with_more_than_one_accepted": multi_accept,
        "extra_acceptances_right": extra_right,
        "extra_acceptances_wrong": extra_wrong,
    }


def summary_block(artifact: dict[str, Any], mode: str = "hybrid") -> dict[str, Any]:
    run = artifact["results"][mode]
    pooled = run["pooled"]["summary"]
    out = {
        name: pooled[name]["estimate"]
        for name in ("micro_precision", "micro_recall", "micro_f1", "parent_aware_f1", "hit_at_1")
        if name in pooled
    }
    out["runtime_seconds"] = run.get("runtime_seconds")
    out["acceptance_model_active"] = run.get("acceptance_model_active")
    out["technique_head_errors"] = run.get("technique_head_errors")
    out["datasets"] = {
        name: {
            key: block["summary"][key]["estimate"]
            for key in ("micro_precision", "micro_recall", "micro_f1")
            if key in block["summary"]
        }
        for name, block in run.get("datasets", {}).items()
    }
    return out


def clinic_block(artifact: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode, block in (artifact.get("results") or {}).items():
        metrics = block.get("metrics") or {}
        out[mode] = {
            key: round(float(metrics[key]), 4)
            for key in ("micro_precision", "micro_recall", "micro_f1", "hit_at_1")
            if key in metrics
        }
    return out


def attribution_block(artifact: dict[str, Any]) -> dict[str, Any]:
    metrics = artifact.get("metrics") or {}
    return {
        key: metrics.get(key)
        for key in (
            "case_count",
            "deterministic_top1",
            "deterministic_top5",
            "mrr",
            "named_correct",
            "named_coverage",
            "named_risk",
            "false_named_attribution_rate",
            "mean_technique_recall",
        )
    }


def model_block(path: Path) -> dict[str, Any] | None:
    artifact = load(path)
    if artifact is None:
        return None
    weights = dict(zip(artifact["feature_names"], artifact["weights"], strict=True))
    metadata = artifact.get("metadata", {})
    return {
        "path": str(path),
        "feature_schema_version": artifact.get("feature_schema_version"),
        "threshold": artifact.get("threshold"),
        "dev_operating_point": metadata.get("dev_operating_point"),
        "technique_head_fingerprint": metadata.get("technique_head_fingerprint"),
        "head_weights": {k: round(v, 4) for k, v in weights.items() if k.startswith("th_")},
        "top_weights": {
            k: round(v, 4) for k, v in sorted(weights.items(), key=lambda item: -abs(item[1]))[:10]
        },
        "row_count": metadata.get("row_count"),
        "case_count": metadata.get("case_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model", type=Path, default=HEAD_MODEL)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "arms": {},
        "paired": {},
        "breakdown": {},
        "head_only": {},
        "multi_label": {},
    }
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ARMS:
        for name in SETS:
            artifact = load(args.dir / f"{arm}_{name}.json")
            if artifact is None:
                continue
            artifacts[(arm, name)] = artifact
            report["arms"][f"{arm}_{name}"] = summary_block(artifact)
            report["multi_label"][f"{arm}_{name}"] = multi_label(case_rows(artifact))
    for name in SETS:
        base = artifacts.get(("off", name))
        if base is None:
            continue
        base_rows = case_rows(base)
        for arm in ("proposer", "head", "multi"):
            other = artifacts.get((arm, name))
            if other is None:
                continue
            rows = case_rows(other)
            report["paired"][f"{arm}_vs_off_{name}"] = paired(base_rows, rows)
            report["breakdown"][f"{arm}_vs_off_{name}"] = breakdown(base_rows, rows)
            report["head_only"][f"{arm}_{name}"] = head_only(rows)
        # the multi-answer arm against the schema-4 head arm (its registered baseline)
        if ("head", name) in artifacts and ("multi", name) in artifacts:
            report["paired"][f"multi_vs_head_{name}"] = paired(
                case_rows(artifacts[("head", name)]), case_rows(artifacts[("multi", name)])
            )
    report["model"] = model_block(args.model)
    report["clinic"] = {
        arm: clinic_block(artifact)
        for arm in ("off", "head")
        if (artifact := load(args.dir / f"clinic_{arm}.json")) is not None
    }
    report["attribution"] = {
        f"{kind}_{arm}": attribution_block(artifact)
        for kind in ("cti_hal", "folds", "elastic")
        for arm in ("off", "head")
        if (artifact := load(args.dir / f"attribution_{kind}_{arm}.json")) is not None
    }
    regression = args.dir / "check_regression_head.log"
    report["check_regression"] = (
        regression.read_text(encoding="utf-8")[-2000:] if regression.is_file() else None
    )
    output = args.dir / "integration_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key, block in report["arms"].items():
        print(
            f"{key:18s} P {block.get('micro_precision')} R {block.get('micro_recall')} "
            f"F1 {block.get('micro_f1')} parent F1 {block.get('parent_aware_f1')} "
            f"hit@1 {block.get('hit_at_1')} ({block.get('runtime_seconds')}s)"
        )
    for key, block in report["paired"].items():
        print(
            f"{key}: "
            + ", ".join(f"{n} {v['estimate']:+.4f} {v['ci95']}" for n, v in block["delta"].items())
        )
    for key, block in report["head_only"].items():
        print(f"head-only {key}: {block}")
    for key, block in report["multi_label"].items():
        print(f"multi-label {key}: {block}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
