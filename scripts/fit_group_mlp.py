"""Fit the PyTorch group network (``apt_classifier/group_mlp.py``) and evaluate it.

    PYTHONPATH=. .venv/bin/python scripts/fit_group_mlp.py [--set key=value ...] [--save]

Training uses only the ATT&CK knowledge base. Evaluation sets (inputs: pre-extracted techniques and
named software):

- ``synthetic``: cases from the training sampler with another seed (in-distribution sanity check);
- ``campaigns`` (development, used for tuning): every ATT&CK campaign attributed to exactly one
  group with at least three techniques; only about 44% of their techniques are in the group's own
  technique list, so this measures transfer beyond the profile;
- ``cti_hal_gold`` (test): the 55 CTI-HAL reports with their human-labelled techniques;
- ``cti_hal_accepted`` (test): the same reports with the techniques and software the local pipeline
  accepted from the anonymized prose (``attribution_cti_hal_current.json``).

Campaign software comes from the raw STIX bundle. ``*_no_sw`` rows repeat a set with the software
block zeroed, so what software contributes is visible; ``cti_hal_gold`` has no software labels.

The group random forest (``PROFILE_FOREST_PATH``) is scored on the same vectors for comparison, and
for ``cti_hal_accepted`` the full pipeline's top-1 is read from the artifact. CTI-HAL files are
local evaluation data (no licence); the report stores metrics, not their text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.group_mlp import (  # noqa: E402
    DEFAULT_PARAMS,
    classification_metrics,
    fit_group_mlp,
    training_cases,
)
from apt_classifier.profile_forest import ProfileForest  # noqa: E402

AUDIT = REPO_ROOT / "data" / "audits" / "group_mlp_eval.json"
MODEL = REPO_ROOT / "data" / "models" / "mlp" / "group_mlp.npz"
CTI_HAL_REPORTS = REPO_ROOT / "data" / "benchmarks" / "cti_hal" / "reports.jsonl"
RAW_STIX = REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
CTI_HAL_RUN = REPO_ROOT / "data" / "audits" / "attribution_cti_hal_current.json"


def campaign_software(raw_path: Path) -> dict[str, set[str]]:
    """Campaign id -> software ids it used, from the raw ATT&CK STIX bundle (the knowledge base
    does not keep campaign software)."""
    if not raw_path.is_file():
        return {}
    objects = json.loads(raw_path.read_text())["objects"]
    external = {
        o["id"]: next(
            (
                r["external_id"]
                for r in o.get("external_references", [])
                if r.get("source_name") == "mitre-attack"
            ),
            None,
        )
        for o in objects
        if "id" in o
    }
    out: dict[str, set[str]] = {}
    for o in objects:
        if (
            o.get("type") == "relationship"
            and o.get("relationship_type") == "uses"
            and o["source_ref"].startswith("campaign--")
            and o["target_ref"].split("--")[0] in ("malware", "tool")
            and not o.get("revoked")
            and not o.get("x_mitre_deprecated")
        ):
            campaign, software = external.get(o["source_ref"]), external.get(o["target_ref"])
            if campaign and software:
                out.setdefault(campaign, set()).add(software)
    return out


def evaluation_sets(kb: Any, model: Any) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Named sets of (feature rows, true groups); ``*_no_sw`` hides the software block."""
    features = model.features
    known = set(model.group_ids)
    seed = int(model.metadata["params"]["seed"]) + 1000
    synthetic, synthetic_groups = training_cases(
        kb, features, {**model.metadata["params"], "seed": seed, "samples_per_group": 20}
    )
    sets: dict[str, tuple[np.ndarray, list[str]]] = {"synthetic": (synthetic, synthetic_groups)}
    index = set(features.space.techniques)
    used = campaign_software(RAW_STIX)
    rows, truth = [], []
    for campaign_id in sorted(kb.campaigns):
        campaign = kb.campaigns[campaign_id]
        techniques = [t for t in campaign.technique_ids if t in index]
        if len(campaign.group_ids) == 1 and campaign.group_ids[0] in known and len(techniques) >= 3:
            rows.append(features.vector(techniques, used.get(campaign_id, ())))
            truth.append(campaign.group_ids[0])
    sets["campaigns"] = (np.stack(rows), truth)
    if CTI_HAL_REPORTS.is_file():
        reports = [json.loads(line) for line in CTI_HAL_REPORTS.read_text().splitlines() if line]
        gold = [
            features.vector(t for s in r["statements"] for t in s["technique_ids"]) for r in reports
        ]
        sets["cti_hal_gold"] = (np.stack(gold), [r["group_id"] for r in reports])
    if CTI_HAL_RUN.is_file():
        run = json.loads(CTI_HAL_RUN.read_text())["known"]
        sets["cti_hal_accepted"] = (
            np.stack([features.vector(r["accepted_techniques"], r["software_ids"]) for r in run]),
            [r["group_id"] for r in run],
        )
    for name in ("campaigns", "cti_hal_accepted"):
        if name in sets:
            hidden = sets[name][0].copy()
            hidden[:, features.space.size :] = 0
            sets[f"{name}_no_sw"] = (hidden, sets[name][1])
    return sets


def ranked(probabilities: np.ndarray, group_ids: tuple[str, ...], k: int = 20) -> list[list[str]]:
    order = np.argsort(-probabilities, axis=1)[:, :k]
    return [[group_ids[i] for i in row] for row in order]


def _pipeline_ranking(row: dict[str, Any]) -> list[str]:
    """The pipeline's final ranking as far as the artifact records it: the true group at its
    ``rank`` and the recorded leader first when that is another group (placeholders elsewhere)."""
    rank = row.get("rank")
    out = [row.get("leading") or "?"] + [f"?{i}" for i in range(1, 20)]
    if rank and rank <= len(out):
        if rank == 1:
            out[0] = row["group_id"]
        else:
            out[rank - 1] = row["group_id"]
    return out


def _parse_sets(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in DEFAULT_PARAMS:
            raise SystemExit(f"unknown parameter {key!r}")
        out[key] = json.loads(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="auto (default), cuda or cpu")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--save", action="store_true", help=f"write the artifact ({MODEL})")
    parser.add_argument("--output", type=Path, default=MODEL)
    parser.add_argument("--report", type=Path, default=AUDIT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--prior-powers",
        default="0,0.5,1,1.5,2,3",
        help="extra prior strengths reported (no refit); the saved model keeps --set prior_power",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    params = _parse_sets(args.set)
    model = fit_group_mlp(kb, params=params, device=args.device, log=None if args.quiet else print)
    forest = None
    try:
        forest = ProfileForest.load(settings.profile_forest_path, device=args.device)
    except (ValueError, KeyError, OSError) as error:
        print(f"forest baseline unavailable: {error}")
    results: dict[str, Any] = {}
    for name, (vectors, truth) in evaluation_sets(kb, model).items():
        entry = {
            "mlp": classification_metrics(truth, ranked(model.predict(vectors), model.group_ids))
        }
        for power in [float(v) for v in args.prior_powers.split(",") if v]:
            entry[f"mlp@{power:g}"] = classification_metrics(
                truth, ranked(model.predict(vectors, prior_power=power), model.group_ids)
            )
        if forest is not None and forest.space.size == model.space.size:
            probabilities, _ = forest.predict(vectors[:, : forest.space.size])
            entry["forest"] = classification_metrics(truth, ranked(probabilities, forest.group_ids))
        results[name] = entry
    if CTI_HAL_RUN.is_file():
        run = json.loads(CTI_HAL_RUN.read_text())["known"]
        results["cti_hal_accepted"]["pipeline"] = classification_metrics(
            [r["group_id"] for r in run], [_pipeline_ranking(r) for r in run]
        )
    report = {
        "model": {
            "fingerprint": model.fingerprint,
            "device": model.device,
            "groups": len(model.group_ids),
            "features": model.features.size,
            "members": len(model.members),
            "metadata": model.metadata,
        },
        "results": {
            name: {
                who: {k: v for k, v in m.items() if k != "per_class"} for who, m in entry.items()
            }
            for name, entry in results.items()
        },
        "cti_hal_run": (
            {
                k: json.loads(CTI_HAL_RUN.read_text()).get(k)
                for k in ("generated_at", "artifact_hash")
            }
            if CTI_HAL_RUN.is_file()
            else None
        ),
        "per_class_cti_hal_gold": results.get("cti_hal_gold", {}).get("mlp", {}).get("per_class"),
        "note": "campaigns = development set (tuning); cti_hal_* = test; pipeline row = full local pipeline incl. software names",
    }
    if args.save:
        report["artifact"] = str(model.save(args.output))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"network on {model.device}; {model.metadata['seconds']}")
    print(f"{'set':18} {'model':9} {'cases':>5} {'top1':>6} {'top5':>6} {'mrr':>6} {'macroF1':>8}")
    for name, entry in report["results"].items():
        for who, m in entry.items():
            print(
                f"{name:18} {who:9} {m['cases']:>5} {m['top1']:>6.3f} {m['top5']:>6.3f} "
                f"{m['mrr']:>6.3f} {m['macro_f1']:>8.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
