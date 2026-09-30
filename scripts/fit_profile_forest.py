"""Fit the random-forest group and source-country models from the ATT&CK knowledge base.

    PYTHONPATH=. .venv/bin/python scripts/fit_profile_forest.py [--device auto|cuda|cpu]

Training runs on the GPU when one is available (``apt_classifier/torch_forest.py``) and uses
only the knowledge base: synthetic partial observations of each group's documented techniques,
motivations and target geographies (``apt_classifier/profile_forest.py``). It writes
``PROFILE_FOREST_PATH`` and ``data/audits/profile_forest_fit.json``. The report's accuracy is an
in-distribution sanity check on synthetic cases drawn with another seed; the real evaluation is
the attribution benchmark (``make benchmark-attribution`` and the CTI-HAL run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.profile_forest import (  # noqa: E402
    DEFAULT_PARAMS,
    fit_profile_forest,
    training_cases,
)

AUDIT = REPO_ROOT / "data" / "audits" / "profile_forest_fit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="auto (default), cuda or cpu")
    parser.add_argument("--output", type=Path, help="default: PROFILE_FOREST_PATH")
    parser.add_argument("--report", type=Path, default=AUDIT)
    parser.add_argument("--seed", type=int, default=int(DEFAULT_PARAMS["seed"]))
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    device = args.device or getattr(settings, "profile_forest_device", "auto")
    model = fit_profile_forest(kb, params={"seed": args.seed}, device=device)
    path = model.save(args.output or settings.profile_forest_path)
    # in-distribution check on synthetic cases from another seed
    check, groups, classes = training_cases(
        kb, model.space, {"seed": args.seed + 100, "samples_per_group": 20}
    )
    group_probs, country_probs = model.predict(check)
    group_top = [model.group_ids[i] for i in group_probs.argmax(1)]
    country_top = [model.country_classes[i] for i in country_probs.argmax(1)]
    report = {
        "artifact": str(path),
        "fingerprint": model.fingerprint,
        "device": model.device,
        "metadata": model.metadata,
        "synthetic_check": {
            "cases": int(check.shape[0]),
            "group_top1": round(
                float(np.mean([a == b for a, b in zip(group_top, groups, strict=True)])), 4
            ),
            "country_accuracy": round(
                float(np.mean([a == b for a, b in zip(country_top, classes, strict=True)])), 4
            ),
            "note": "in-distribution only; the attribution benchmarks are the evaluation",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    seconds = model.metadata["seconds"]
    print(
        f"profile forest -> {path} ({model.device}; {model.metadata['training_cases']} training cases; "
        f"group forest {model.group_forest.n_trees} trees {seconds['group_forest']}s, "
        f"country forest {model.country_forest.n_trees} trees {seconds['country_forest']}s)"
    )
    print(f"synthetic check: {report['synthetic_check']}")
    print(f"country classes: {model.metadata['country_groups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
