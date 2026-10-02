"""Offline replay of the software interview questions on attribution benchmark rows (docs 21.3.15 C).

The enumerated simulation (``scripts/attribution_benchmark.py --simulate-software-interview
enumerate``) re-runs the whole pipeline after every simulated answer, so a "found" can re-rank the
shortlist and change the gate window. This replay keeps each row's recorded gate window and order
fixed and replays only the naming gate's discrimination (``software_questions.replay_window`` over
the row's accepted techniques, behaviour matches, negatives and software), with the questions the
product would ask (the same pool and ranking functions), the "found" guard and the leader-only
"not found" rule. It is the C6 cross-check: its expected names must be within +-2 of the
pipeline's, per seed.

What it models beyond the separation bars: a "found" for the leader's malware adds the software
evidence family when the software classifier did not already rank the leader in its top ten, and
the maturity score moves by that family factor (the unresolved-activity check). The combiner's
re-ranking, the cluster step and any change of the window are not modelled (rows named as a
cluster count as named and are never asked).

Usage::

    PYTHONPATH=. .venv/bin/python scripts/software_interview_replay.py \\
        data/audits/fresh/software_seed43.json [--output replay.json]

The artifact's own settings overrides are applied (``Settings(**overrides)``), and removed-actor
rows are replayed on the knowledge base without the removed group, as the benchmark ran them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings
from apt_classifier.separation import (
    CandidateDiscrimination,
    DiscriminationResult,
    DiscriminationWeights,
)
from apt_classifier.software_questions import (
    leader_only_negative,
    rank_software_leads,
    replay_window,
    software_question_pool,
)
from scripts.attribution_benchmark import (
    SOFTWARE_INTERVIEW_FP,
    SOFTWARE_INTERVIEW_Q,
    summarize_software_interview,
)

NAMED = {"known_group_hypothesis", "known_cluster_hypothesis"}
SOFTWARE_CLASSIFIER = "software_tools"
SUPPORT_RANK_LIMIT = 10


def reduced_kb(kb: AttackKnowledgeBase, group_id: str) -> AttackKnowledgeBase:
    """The knowledge base without one group, as the benchmark's removed-actor arm builds it."""
    reduced = copy.deepcopy(kb)
    reduced.groups.pop(group_id, None)
    for campaign in reduced.campaigns.values():
        if group_id in getattr(campaign, "group_ids", []):
            campaign.group_ids = [gid for gid in campaign.group_ids if gid != group_id]
    for software in reduced.software.values():
        if group_id in getattr(software, "group_ids", []):
            software.group_ids = [gid for gid in software.group_ids if gid != group_id]
    reduced.recompute_statistics()
    return reduced


def row_context(row: dict[str, Any]) -> SimpleNamespace:
    """The evidence the gate scored, from the row's replay inputs (ids and scores only)."""
    return SimpleNamespace(
        observed_techniques=dict(row.get("accepted_confidences") or {}),
        negative_techniques=set(row.get("negative_techniques") or ()),
        parent_policy="ancestor_compatible",
        specific_behavior_matches=tuple(
            SimpleNamespace(
                group_id=group_id,
                observation_id=observation_id,
                technique_id=technique,
                score=score,
            )
            for group_id, observation_id, technique, score in row.get("window_behavior_matches")
            or ()
        ),
        software_ids=set(row.get("software_ids") or ()),
        decisive_software=dict(row.get("decisive_software") or {}),
        contrast_techniques=None,
    )


def row_discrimination(
    row: dict[str, Any], weights: DiscriminationWeights, floor: float
) -> DiscriminationResult:
    """The recorded gate window (order and channels) as a discrimination result."""
    scores = tuple(
        CandidateDiscrimination(
            group_id=str(item["group_id"]),
            rrf_score=float(item.get("rrf", 0.0) or 0.0),
            technique=float(item.get("technique", 0.0) or 0.0),
            behavior=float(item.get("behavior", 0.0) or 0.0),
            software=float(item.get("software", 0.0) or 0.0),
            negative=float(item.get("negative", 0.0) or 0.0),
            total=float(item.get("total", 0.0) or 0.0),
            soft=float(item.get("soft", 0.0) or 0.0),
        )
        for item in row.get("topk") or ()
    )
    leader = scores[0] if scores else None
    others = scores[1:]
    runner = max(others, key=lambda item: item.total) if others else None
    runner_total = runner.total if runner is not None else 0.0
    denominator = max(leader.total if leader else 0.0, runner_total, floor)
    separation = (
        (leader.total - runner_total) / denominator if leader is not None and denominator else 0.0
    )
    return DiscriminationResult(
        scores=scores,
        top_k=len(scores),
        leader_id=leader.group_id if leader is not None else None,
        runner_up_id=runner.group_id if runner is not None else None,
        separation=max(-1.0, min(1.0, separation)),
        floor=floor,
        weights=weights,
    )


class RowReplay:
    """The fixed-window gate and question selection for one benchmark row."""

    def __init__(self, kb: Any, row: dict[str, Any], settings: Settings, thresholds: dict) -> None:
        self.kb = kb
        self.row = row
        self.settings = settings
        self.thresholds = thresholds
        weights = DiscriminationWeights(
            technique=float(settings.attribution_separation_technique_weight),
            behavior=float(settings.attribution_separation_behavior_weight),
            software=float(settings.attribution_separation_software_weight),
            negative=float(settings.attribution_separation_negative_weight),
            soft=float(settings.attribution_separation_soft_weight),
        )
        self.recorded = row_discrimination(
            row, weights, float(settings.attribution_separation_floor)
        )
        self.context = row_context(row)
        self.document = set(self.context.software_ids)
        self.options = {
            "tool_weight_cap": float(settings.software_tool_weight_cap),
            "behavior_threshold": float(settings.behavior_similarity_threshold),
        }
        self.base = replay_window(kb, self.recorded, self.context, **self.options)
        self.leader = self.recorded.leader_id
        self.window = [item.group_id for item in self.recorded.scores]
        self.top3 = list(row.get("top3") or [])[:3]
        detail = (row.get("top3_detail") or [{}])[0]
        ranks = dict(detail.get("classifier_ranks") or {})
        rank = ranks.get(SOFTWARE_CLASSIFIER)
        self.software_supports_leader = rank is not None and int(rank) <= SUPPORT_RANK_LIMIT

    @property
    def matches_recording(self) -> bool:
        return abs(self.base.separation - float(self.row.get("separation") or 0.0)) <= 2e-3

    def _view(self, found: set[str]) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(self.context), "software_ids": self.document | found})

    def decide(self, found: set[str], not_found: set[str]) -> list[str]:
        """The named set the fixed-window gate gives with these answers ([] or [leader]). Rows
        named in the run keep their name (they are not asked while named)."""
        stage = self.row.get("stage")
        if stage in NAMED:
            return list(self.row.get("named_cluster") or []) or [str(self.row.get("leading"))]
        if (
            stage == "insufficient_evidence"
            or not self.leader
            or self.row.get("leader_guard_fired")
        ):
            return []
        if not found:
            return []  # without a found nothing the gate reads changes, and a not-found only lowers
        leader_software = set(self.kb.group_software.get(self.leader, ()) or ())
        if not found & leader_software:
            return []  # a found only enables a name of a group that documents it
        view = self._view(found)
        boost = float(self.settings.software_decisive_boost)
        replayed = replay_window(
            self.kb,
            self.recorded,
            view,
            software_ids=view.software_ids,
            decisive_boost=boost,
            **self.options,
        )
        # the families and maturity checks before the gate
        if stage == "unresolved_activity_cluster":
            factors = dict(self.row.get("maturity_factors") or {})
            minimum_families = int(self.thresholds.get("min_independent_evidence_families", 2))
            families = round(float(factors.get("independent_families", 0.0)) * minimum_families)
            if not self.software_supports_leader:
                families += 1  # the leader documents a found malware (checked above)
            factor = min(1.0, families / max(minimum_families, 1))
            maturity = float(self.row.get("maturity") or 0.0) + 0.25 * (
                factor - float(factors.get("independent_families", 0.0))
            )
            if families < minimum_families or maturity < float(
                self.thresholds.get("min_maturity_score", 0.5)
            ):
                return []
        mode = str(self.thresholds.get("gate_mode", "separation"))
        if mode == "maturity":
            return []  # a software answer cannot move the maturity gate in this replay
        effect = leader_only_negative(
            replayed, self.kb, not_found, float(self.settings.interview_software_negative_weight)
        )
        if effect.separation < float(self.thresholds.get("hypothesis_separation", 0.5)):
            return []
        if effect.total < float(self.thresholds.get("hypothesis_min_discriminative", 3.0)):
            return []
        if bool(self.settings.interview_software_found_requires_lead) and self.base.separation < 0:
            return []
        return [self.leader]

    def next_question(
        self, found: set[str], answered: set[str], asked: dict[str, int]
    ) -> tuple[str, str, int] | None:
        """(software id, target group, top-three place) of the next question, or None."""
        if sum(asked.values()) >= int(self.settings.interview_software_max_questions):
            return None
        pool = software_question_pool(
            self.kb,
            self.window,
            self.top3,
            self.document | found,
            answered,
            max_users=int(self.settings.interview_software_max_users),
        )
        view = self._view(found)
        if found:
            factor, triggers = _decisive(self.kb, view.software_ids, self.settings)
            view.decisive_software = {"factor": factor, "software_ids": triggers}
        leads = rank_software_leads(
            self.kb,
            self.recorded,
            view,
            pool,
            per_group=int(self.settings.interview_software_per_group),
            decisive_boost=float(self.settings.software_decisive_boost),
            negative_weight=float(self.settings.interview_software_negative_weight),
            asked=asked,
            **self.options,
        )
        return (leads[0].software_id, leads[0].group_id, leads[0].target_rank) if leads else None

    def enumerate(self, truth_software: set[str]) -> dict[str, Any]:
        baseline = self.decide(set(), set())
        nodes: list[dict[str, Any]] = []
        truth_asked: set[str] = set()

        def visit(
            path: tuple[str, ...], found: set[str], not_found: set[str], asked: dict[str, int]
        ) -> None:
            named = self.decide(found, not_found)
            selected = (
                None
                if named and not bool(self.settings.interview_software_when_named)
                else self.next_question(found, found | not_found, asked)
            )
            nodes.append({"path": "|".join(path), "named": named, "leaf": selected is None})
            if selected is None:
                return
            question, group_id, rank = selected
            if question in truth_software:
                truth_asked.add(question)
            counted = {**asked, group_id: asked.get(group_id, 0) + 1}
            for answer in ("F", "N"):
                visit(
                    (*path, f"{question}:{rank}:{answer}"),
                    found | ({question} if answer == "F" else set()),
                    not_found | ({question} if answer == "N" else set()),
                    counted,
                )

        visit((), set(), set(), {})
        return {"baseline_named": baseline, "truth_documents": sorted(truth_asked), "nodes": nodes}


def _decisive(kb: Any, software_ids: set[str], settings: Settings) -> tuple[float, list[str]]:
    from apt_classifier.classifiers.software import decisive_software_factor

    factor, triggers = decisive_software_factor(
        kb, set(software_ids), boost=float(settings.software_decisive_boost)
    )
    return factor, list(triggers)


def replay_artifact(path: Path, kb: AttackKnowledgeBase) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = dict((artifact.get("parameters") or {}).get("settings_overrides") or {})
    overrides.setdefault("interview_software_questions", True)
    settings = Settings(**overrides)
    thresholds = dict((artifact.get("fingerprint_inputs") or {}).get("thresholds") or {})
    reduced: dict[str, AttackKnowledgeBase] = {}
    rows: dict[str, list[dict[str, Any]]] = {"known": [], "unknown": []}
    checks = {"rows": 0, "skipped": 0, "separation_matches": 0}
    for arm in ("known", "unknown"):
        for row in artifact.get(arm) or []:
            if (
                not row.get("topk")
                or "accepted_confidences" not in row
                or "software_ids" not in row
            ):
                checks["skipped"] += 1
                continue
            group_id = str(row["group_id"])
            if arm == "unknown":
                base_kb = reduced.setdefault(group_id, reduced_kb(kb, group_id))
            else:
                base_kb = kb
            replay = RowReplay(base_kb, row, settings, thresholds)
            checks["rows"] += 1
            checks["separation_matches"] += replay.matches_recording
            truth = set(getattr(kb.groups.get(group_id), "software_ids", ()) or ())
            rows[arm].append({"group_id": group_id, "software_interview": replay.enumerate(truth)})
    summary = summarize_software_interview(
        rows["known"],
        rows["unknown"],
        cluster_naming=bool(getattr(settings, "attribution_cluster_naming", False)),
    )
    summary["criteria"].pop("C6", None)
    comparison = None
    pipeline = artifact.get("software_interview")
    if isinstance(pipeline, dict):
        differences = {
            f"q={q}": round(
                summary["expected"][f"q={q},fp=0.0"]["gained"]
                - pipeline["expected"][f"q={q},fp=0.0"]["gained"],
                4,
            )
            for q in SOFTWARE_INTERVIEW_Q
        }
        comparison = {
            "gained_replay_minus_pipeline": differences,
            "C6": all(abs(value) <= 2.0 for value in differences.values()),
        }
    return {
        "artifact": str(path),
        "seed": (artifact.get("parameters") or {}).get("seed"),
        "replay_checks": checks,
        "answer_model": {"q": list(SOFTWARE_INTERVIEW_Q), "fp": list(SOFTWARE_INTERVIEW_FP)},
        "replay": summary,
        "pipeline_comparison": comparison,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="write the replay (all artifacts) as JSON")
    args = parser.parse_args(argv)
    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    results = []
    for path in args.artifacts:
        result = replay_artifact(path, kb)
        results.append(result)
        checks = result["replay_checks"]
        print(
            f"{path}: seed {result['seed']}, {checks['rows']} rows replayed "
            f"({checks['separation_matches']} reproduce the recorded separation, "
            f"{checks['skipped']} without replay inputs)"
        )
        for key in ("q=0.1,fp=0.0", "q=0.3,fp=0.0", "q=1.0,fp=0.0", "q=0.3,fp=0.05"):
            item = result["replay"]["expected"][key]
            print(
                f"  {key:14s} gained {item['gained']:.2f} wrong added {item['wrong_added']:.2f} "
                f"removed named {item['removed_named']:.2f}"
            )
        if result["pipeline_comparison"] is not None:
            print(f"  versus the pipeline: {result['pipeline_comparison']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
