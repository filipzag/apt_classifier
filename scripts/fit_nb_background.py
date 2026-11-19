"""Build the Naive Bayes group-background reference (``apt_classifier/nb_background.py``).

Scores every group with the configured Naive Bayes member on the accepted techniques that
recorded attribution-benchmark runs of the design fold seeds (41 and 53) stored per case, and
writes each group's mean and spread of its case-level z-scores. A removed actor's fold scores
every group but that actor (the benchmark removes it from the knowledge base). Only ATT&CK fold
runs of the design seeds are accepted as reference; report sets, CTI-HAL, Elastic and the fresh
or evaluation seeds are refused.

``--evaluate`` scores the member before and after the correction on recorded runs of the
development report sets (vendor, campaigns, advisories; known cases only) as report-only numbers.
Offline, CPU only, about a minute; no Chroma, no head service, no OpenRouter.

    PYTHONPATH=. .venv/bin/python scripts/fit_nb_background.py \\
        --reference <design41 run>.json <design53 run>.json \\
        --evaluate <vendor run>.json <campaigns run>.json <advisories run>.json \\
        --output data/models/nb_background/nb_background_design.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.classifiers.base import ClassifierContext  # noqa: E402
from apt_classifier.classifiers.naive_bayes import BernoulliNaiveBayesClassifier  # noqa: E402
from apt_classifier.config import Settings  # noqa: E402
from apt_classifier.nb_background import (  # noqa: E402
    BackgroundNaiveBayesClassifier,
    fit_background,
)

FOLD_BENCHMARK = "attribution_text_heldout"
DESIGN_SEEDS = (41, 53)
DEVELOPMENT_REPORT_BENCHMARKS = (
    "attribution_vendor_reports",
    "attribution_campaigns_reports",
    "attribution_advisories_reports",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, allowed: tuple[str, ...], seeds: tuple[int, ...] | None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = data.get("benchmark")
    if benchmark not in allowed:
        raise SystemExit(
            f"{path}: benchmark {benchmark!r} is not allowed here ({', '.join(allowed)})"
        )
    seed = (data.get("parameters") or {}).get("seed")
    if seeds is not None and seed not in seeds:
        raise SystemExit(f"{path}: seed {seed} is not a design seed {seeds}")
    return data


def contexts(kb: Any, data: dict[str, Any], *, parts: tuple[str, ...]):
    """One classifier context per recorded case: its accepted techniques at their confidence and
    its negated techniques; a removed actor's case scores every group but that actor."""
    groups = set(kb.groups)
    for part in parts:
        for row in data.get(part) or []:
            observed = {t: float(c) for t, c in (row.get("accepted_confidences") or {}).items()}
            if not observed:
                continue
            eligible = groups - {row["group_id"]} if part == "unknown" else None
            yield (
                row,
                ClassifierContext(
                    kb=kb,
                    observed_techniques=observed,
                    technique_observations={},
                    negative_techniques=set(row.get("negative_techniques") or ()),
                    semantic_profile_scores={},
                    group_text_scores={},
                    software_ids=set(),
                    eligible_group_ids=eligible,
                ),
            )


def _truth_rank(result: Any, truth: str) -> int | None:
    """Strict rank: groups tied with the truth count against it."""
    scores = {entry.group_id: entry.score for entry in result.entries}
    if truth not in scores:
        return None
    return 1 + sum(1 for g, s in scores.items() if g != truth and s >= scores[truth])


def evaluate(kb: Any, member: Any, corrected: Any, paths: list[Path]) -> None:
    print("member alone on the development report sets (known cases, strict ranks):")
    rows = []
    for path in paths:
        data = _load(path, DEVELOPMENT_REPORT_BENCHMARKS, None)
        for row, context in contexts(kb, data, parts=("known",)):
            rows.append(
                (
                    data["benchmark"],
                    _truth_rank(member.classify(context), row["group_id"]),
                    _truth_rank(corrected.classify(context), row["group_id"]),
                )
            )
    for label, index in (("current", 1), ("corrected", 2)):
        ranks = [row[index] for row in rows]
        top1 = sum(rank == 1 for rank in ranks)
        top5 = sum(rank is not None and rank <= 5 for rank in ranks)
        mrr = statistics.fmean(1.0 / rank if rank else 0.0 for rank in ranks) if ranks else 0.0
        print(f"  {label:9s} top-1 {top1}/{len(ranks)}  top-5 {top5}  MRR {mrr:.3f}")
    gained = sum(1 for _, before, after in rows if after == 1 and before != 1)
    lost = sum(1 for _, before, after in rows if before == 1 and after != 1)
    print(f"  gained {gained}, lost {lost}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluate", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, help="default: NAIVE_BAYES_BACKGROUND_PATH")
    parser.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    member = BernoulliNaiveBayesClassifier(
        settings.nb_alpha, settings.nb_beta, settings.negative_evidence_weight
    )
    reference = []
    sources = []
    for path in args.reference:
        data = _load(path, (FOLD_BENCHMARK,), DESIGN_SEEDS)
        cases = list(contexts(kb, data, parts=("known", "unknown")))
        reference.extend(context for _, context in cases)
        sources.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "seed": data["parameters"]["seed"],
                "cases": len(cases),
                "attack_version": data.get("attack_version"),
            }
        )
    background = fit_background(kb, member, reference, settings, sources=sources)
    counts = [count for _, _, count in background.groups.values()]
    print(
        f"reference: {len(reference)} cases from {len(sources)} runs, {len(background.groups)} "
        f"groups (reference cases per group {min(counts)}-{max(counts)}), ATT&CK "
        f"{background.attack_version}, fingerprint {background.fingerprint}"
    )
    if args.evaluate:
        evaluate(kb, member, BackgroundNaiveBayesClassifier(member, background), args.evaluate)
    output = Path(args.output or settings.naive_bayes_background_path)
    if output.exists() and not args.force:
        print(f"not written: {output} exists (pass --force or a new --output)", file=sys.stderr)
        return 1
    background.save(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
