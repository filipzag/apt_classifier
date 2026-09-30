"""Tests for the held-out public benchmark loader, regression checker and collector.

No Chroma index or model is needed: these exercise case loading, leak filtering,
stratified sampling, artifact comparison and summary composition on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_mapping import (
    LEAKY_DATASETS,
    case_set_hash,
    dataset_of,
    load_cases,
)
from scripts.check_regression import compare
from scripts.collect_verification import (
    summarize_curated,
    summarize_evidenceforge,
    summarize_public,
)


def _write_split(root: Path, dataset: str, split: str, rows: list[dict]) -> None:
    path = root / f"{dataset}_{split}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(dataset: str, index: int, text: str, labels: list[str] | None = None) -> dict:
    labels = ["T1059.001"] if labels is None else labels
    return {
        "case_id": f"{dataset}-test-{index:06d}",
        "document_id": f"{dataset}-test-{index:06d}",
        "text": text,
        "expected_technique_ids": labels,
        "original_technique_ids": labels or ["T9999"],
        "source": f"tumeteor/mitre-ttp-mapping:{dataset}",
        "split": "test",
        "label_quality": "public_dataset",
    }


@pytest.fixture
def benchmark_root(tmp_path: Path) -> Path:
    rows = [_row("tram", i, f"tram sentence number {i} about powershell") for i in range(1, 21)]
    rows.append(_row("tram", 21, "The adversary dumped credentials from LSASS. (Citation: X)"))
    _write_split(tmp_path, "tram", "test", rows)
    expert = [_row("expert", i, f"expert paragraph {i}", ["T1105", "T1041"]) for i in range(1, 6)]
    expert.append(_row("expert", 6, "unscorable", []))  # original label could not be mapped
    _write_split(tmp_path, "expert", "test", expert)
    _write_split(tmp_path, "procedures", "test", [_row("procedures", 1, "leaky text")])
    return tmp_path


def test_dataset_is_taken_from_source_suffix():
    from apt_classifier.evaluation import TechniqueEvaluationCase

    case = TechniqueEvaluationCase.from_dict(_row("derived_procedures", 3, "x"))
    assert dataset_of(case) == "derived_procedures"


def test_procedures_dataset_is_refused_unless_allowed(benchmark_root: Path):
    assert "procedures" in LEAKY_DATASETS
    cases, diagnostics = load_cases(benchmark_root, datasets={"tram", "procedures"}, split="test")
    assert diagnostics["refused_leaky_datasets"] == ["procedures"]
    assert all(dataset_of(case) == "tram" for case in cases)
    cases, diagnostics = load_cases(
        benchmark_root, datasets={"procedures"}, split="test", allow_leaky=True
    )
    assert diagnostics["leaky_datasets_included"] == ["procedures"]
    assert len(cases) == 1


def test_indexed_text_is_dropped_and_counted(benchmark_root: Path):
    indexed = frozenset({"the adversary dumped credentials from lsass"})
    cases, diagnostics = load_cases(
        benchmark_root, datasets={"tram"}, split="test", indexed_texts=indexed
    )
    assert diagnostics["dropped_indexed_text"]["tram"] == 1
    assert len(cases) == 20
    assert not any("LSASS" in case.text for case in cases)


def test_unscorable_rows_are_dropped(benchmark_root: Path):
    cases, diagnostics = load_cases(benchmark_root, datasets={"expert"}, split="test")
    assert diagnostics["dropped_unscorable"]["expert"] == 1
    assert len(cases) == 5


def test_sampling_is_stratified_seeded_and_not_first_n(benchmark_root: Path):
    first, _ = load_cases(benchmark_root, datasets={"tram", "expert"}, split="test", sample=5)
    second, _ = load_cases(benchmark_root, datasets={"tram", "expert"}, split="test", sample=5)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    by_dataset: dict[str, list[str]] = {}
    for case in first:
        by_dataset.setdefault(dataset_of(case), []).append(case.case_id)
    assert len(by_dataset["tram"]) == 5
    assert len(by_dataset["expert"]) == 5
    assert by_dataset["tram"] != [f"tram-test-{i:06d}" for i in range(1, 6)]
    other, _ = load_cases(benchmark_root, datasets={"tram"}, split="test", sample=5, seed=11)
    assert [case.case_id for case in other] != by_dataset["tram"]


def test_case_set_hash_is_order_independent(benchmark_root: Path):
    cases, _ = load_cases(benchmark_root, datasets={"tram", "expert"}, split="test")
    assert case_set_hash(cases) == case_set_hash(list(reversed(cases)))


def _artifact(f1: float, hit10: float, *, case_hash: str = "abc") -> dict:
    metrics = {"micro_f1": f1, "hit_at_10": hit10, "micro_precision": 0.5, "case_count": 10}
    return {
        "case_set_hash": case_hash,
        "results": {
            "hybrid": {
                "pipeline_fingerprint": "fp",
                "pooled": {"metrics": metrics},
                "datasets": {"tram": {"metrics": metrics}},
            }
        },
    }


def test_compare_flags_regressions_beyond_tolerance():
    report = compare(_artifact(0.40, 0.80), _artifact(0.37, 0.80))
    assert report["same_case_set"] is True
    assert report["pooled"]["micro_f1"]["delta"] == pytest.approx(-0.03)
    assert any("micro_f1" in item for item in report["regressions"])
    report = compare(_artifact(0.40, 0.80), _artifact(0.39, 0.78))
    assert report["regressions"] == []
    report = compare(_artifact(0.40, 0.80), _artifact(0.41, 0.70, case_hash="other"))
    assert report["same_case_set"] is False
    assert any("hit_at_10" in item for item in report["regressions"])


def test_compare_requires_mode():
    with pytest.raises(KeyError):
        compare(_artifact(0.4, 0.8), _artifact(0.4, 0.8), mode="lexical")


def test_collector_summaries_read_only_from_artifacts():
    public = {
        "split": "test",
        "datasets": ["tram"],
        "case_count": 10,
        "case_set_hash": "abc",
        "results": {
            "hybrid": {
                "mode": "hybrid",
                "cues_enabled": True,
                "pipeline_fingerprint": "fp",
                "pooled": {
                    "metrics": {"case_count": 10, "micro_f1": 0.4, "hit_at_10": 0.8},
                    "bootstrap_confidence_intervals": {
                        "micro_f1": {
                            "estimate": 0.4,
                            "lower": 0.3,
                            "upper": 0.5,
                            "confidence": 0.95,
                        }
                    },
                },
                "datasets": {},
            }
        },
    }
    summary = summarize_public(public)
    assert summary["runs"]["hybrid"]["pooled"]["micro_f1"] == {"estimate": 0.4, "ci95": [0.3, 0.5]}
    curated = {
        "pipeline_fingerprint": "fp",
        "results": {
            "hybrid": {
                "mode": "hybrid",
                "positive_cases": 30,
                "hard_negative_cases": 6,
                "top1": 26,
                "top3": 29,
                "top10": 30,
                "abstain_count": 0,
                "accepted_set": {
                    "tp": 26,
                    "fp": 21,
                    "fn": 9,
                    "precision": 0.55,
                    "recall": 0.74,
                    "micro_f1": 0.63,
                    "accepted_fp_cases": 0,
                },
            }
        },
    }
    assert summarize_curated(curated)["runs"]["hybrid"]["top1"] == "26/30"
    assert summarize_curated(curated)["runs"]["hybrid"]["hard_negative_accepted_fp_cases"] == "0/6"
    evidenceforge = {
        "metadata": {
            "scenario_path": "data/evidenceforge/heldout-clinic/scenario.yaml",
            "generation_seed": 2024,
            "pipeline": {"pipeline": "local", "verify": False},
            "pipeline_fingerprint": "fp",
        },
        "results": {
            "narrative": {
                "metrics": {
                    "case_count": 21,
                    "micro_f1": 0.21,
                    "zero_label_false_positive_rate": 0.33,
                },
                "bootstrap_confidence_intervals": {},
                "coverage": {"coverage": 1.0, "unsupported_reasons": {}},
            }
        },
    }
    summary = summarize_evidenceforge(evidenceforge)
    assert summary["scenario"] == "heldout-clinic"
    assert summary["status"].startswith("held-out")
    assert summary["modes"]["narrative"]["zero_label_false_positive_rate"] == 0.33


def test_fold_store_hides_held_out_evidence():
    from types import SimpleNamespace

    from apt_classifier.vectorstore import VectorHit
    from scripts.attribution_benchmark import FoldStore

    held_text = "the adversary used wmic to run robocopy on hosts"

    class Store:
        def query_techniques(self, texts, *, n_results=10, fields=None):
            return [
                [
                    VectorHit(
                        "a",
                        "The adversary used WMIC to run robocopy on hosts.",
                        {"technique_id": "T1047"},
                        0.0,
                        1.0,
                    ),
                    VectorHit(
                        "b", "Adversaries may abuse WMI.", {"technique_id": "T1047"}, 0.2, 0.8
                    ),
                    VectorHit("c", "Other technique text.", {"technique_id": "T1059"}, 0.3, 0.7),
                ]
                for _ in texts
            ]

        def query_group_behaviors(self, texts, *, n_results=20):
            return [
                [
                    VectorHit("g1", "x", {"group_id": "G1", "technique_id": "T1047"}, 0.0, 1.0),
                    VectorHit("g1b", "y", {"group_id": "G1", "technique_id": "T1003"}, 0.2, 0.8),
                    VectorHit("g2", "z", {"group_id": "G2", "technique_id": "T1047"}, 0.3, 0.7),
                ]
                for _ in texts
            ]

        def query_group_profile(self, texts, *, document_type, n_results=20):
            return [
                VectorHit("p1", "p", {"group_id": "G1"}, 0.1, 0.9),
                VectorHit("p2", "p", {"group_id": "G2"}, 0.2, 0.8),
            ]

        extra_attribute = "delegated"

    proxy = FoldStore(
        Store(),
        group_id="G1",
        held_out_techniques=frozenset({"T1047"}),
        held_out_texts=frozenset({held_text}),
    )
    hits = proxy.query_techniques(["q"], n_results=2)[0]
    assert [hit.id for hit in hits] == ["b", "c"]
    behaviors = proxy.query_group_behaviors(["q"], n_results=5)[0]
    assert [hit.id for hit in behaviors] == ["g1b", "g2"]
    assert [hit.id for hit in proxy.query_group_profile(["q"], document_type="d")] == ["p1", "p2"]
    assert proxy.extra_attribute == "delegated"
    assert proxy.dropped == {"technique": 1, "behavior": 1, "profile": 0}
    hidden = FoldStore(
        Store(),
        group_id="G1",
        held_out_techniques=frozenset(),
        held_out_texts=frozenset(),
        hide_group_entirely=True,
    )
    assert [hit.id for hit in hidden.query_group_behaviors(["q"])[0]] == ["g2"]
    assert [hit.id for hit in hidden.query_group_profile(["q"], document_type="d")] == ["p2"]
    assert isinstance(SimpleNamespace(), SimpleNamespace)


def test_build_folds_anonymizes_actor_names_and_is_deterministic():
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
    from scripts.attribution_benchmark import build_folds

    kb = AttackKnowledgeBase(
        version="t",
        techniques={f"T{i}": Technique(id=f"T{i}", name=f"Tech {i}") for i in range(1, 8)},
        groups={
            "G1": Group(
                id="G1",
                name="Axiom",
                aliases=["Group 72"],
                technique_ids=[f"T{i}" for i in range(1, 7)],
                technique_behaviors={
                    f"T{i}": [f"Axiom used tool number {i} against victims while Group 72 watched."]
                    for i in range(1, 7)
                },
            ),
            "G2": Group(
                id="G2", name="Tiny", technique_ids=["T1"], technique_behaviors={"T1": ["x"]}
            ),
        },
    )
    folds = build_folds(
        kb, coverage=0.5, seed=1, min_behaviors=5, min_techniques=5, max_observations=12
    )
    assert [fold.group_id for fold in folds] == ["G1"]
    fold = folds[0]
    assert len(fold.observations) == 3
    for text in fold.observations:
        assert "axiom" not in text.lower() and "group 72" not in text.lower()
        assert "the adversary" in text.lower()
    again = build_folds(
        kb, coverage=0.5, seed=1, min_behaviors=5, min_techniques=5, max_observations=12
    )
    assert again[0].observations == fold.observations
    other = build_folds(
        kb, coverage=0.5, seed=2, min_behaviors=5, min_techniques=5, max_observations=12
    )
    assert (
        other[0].held_out_techniques != fold.held_out_techniques
        or other[0].observations != fold.observations
    )


def test_collector_summarizes_attribution_artifact():
    from scripts.collect_verification import compose, summarize_attribution

    payload = {
        "parameters": {"coverage": 0.4, "known_folds": 3, "unknown_folds": 1},
        "pipeline_fingerprint": "fp",
        "leak_control": {"dropped_hits": {"technique": 5, "behavior": 7, "profile": 0}},
        "metrics": {
            "known_count": 3,
            "unknown_count": 1,
            "top1": 0.667,
            "top1_ci95": [0.33, 1.0],
            "top5": 1.0,
            "mrr": 0.8,
            "median_rank": 1,
            "mean_technique_recall": 0.75,
            "unknown_recall": 1.0,
            "false_named_attribution_rate": 0.0,
            "named_coverage": 0.25,
            "named_risk": 0.0,
            "stage_distribution": {"known": {"known_group_shortlist": 3}, "unknown": {}},
        },
    }
    summary = summarize_attribution(payload)
    assert summary["top1"] == 0.667
    assert summary["leak_control"]["dropped_hits"]["behavior"] == 7
    report = compose(
        public=None,
        curated=None,
        evidenceforge={},
        public_path=Path("missing.json"),
        curated_path=Path("missing.json"),
        gates=None,
        attribution=payload,
        attribution_path=Path("attribution.json"),
    )
    assert report["attribution_text_heldout"]["unknown_recall"] == 1.0
    assert report["sources"]["public_heldout"]["status"] == "missing"


def test_public_benchmark_run_produces_per_case_rows_without_chroma():
    from types import SimpleNamespace

    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from apt_classifier.retrieve import LexicalTechniqueIndex
    from apt_classifier.vectorstore import VectorHit
    from scripts.benchmark_mapping import run

    kb = AttackKnowledgeBase(
        version="t",
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1003.001": Technique(
                id="T1003.001", name="LSASS Memory", is_subtechnique=True, parent_id="T1003"
            ),
        },
        groups={"G1": Group(id="G1", name="G", technique_ids=["T1059.001"])},
    )

    class Store:
        def query_techniques(self, texts, *, n_results, **kwargs):
            return [
                [
                    VectorHit(
                        "v",
                        "PowerShell commands",
                        {"technique_id": "T1059.001", "field": "description"},
                        0.1,
                        0.9,
                    )
                ]
                if "powershell" in text.lower()
                else []
                for text in texts
            ]

        def query_group_profile(self, texts, *, document_type, n_results):
            return []

        def query_group_behaviors(self, texts, *, n_results):
            return [[] for _ in texts]

    cases = [
        TechniqueEvaluationCase.from_dict(
            _row("tram", 1, "PowerShell ran an encoded command", ["T1059.001"])
        ),
        TechniqueEvaluationCase.from_dict(_row("tram", 2, "nothing here", ["T1003.001"])),
        TechniqueEvaluationCase.from_dict(
            _row("expert", 3, "PowerShell staged a loader", ["T1059.001", "T1105"])
        ),
    ]
    settings = SimpleNamespace(
        similarity_threshold=0.45,
        retrieval_top_k=30,
        technique_candidate_pool=12,
        technique_parent_policy="ancestor_compatible",
        technique_acceptance_threshold=0.75,
        technique_relative_margin=0.08,
        technique_mapping_preset="balanced",
        embedding_model="fake",
        embedding_model_revision="rev",
    )
    report = run(
        cases,
        mode="hybrid",
        settings=settings,
        kb=kb,
        store=Store(),
        lexical_index=LexicalTechniqueIndex.from_kb(kb),
        bootstrap_samples=20,
    )
    assert set(report["datasets"]) == {"tram", "expert"}
    assert report["pooled"]["summary"]["case_count"] == 3
    assert report["pooled"]["metrics"]["hit_at_1"] == pytest.approx(2 / 3)
    assert report["pooled"]["metrics"]["micro_recall"] == pytest.approx(2 / 4)
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows["tram-test-000001"]["expected_accepted"] is True
    assert rows["tram-test-000002"]["expected_in_pool"] is False
    first = rows["tram-test-000001"]["candidates"][0]
    assert first["technique_id"] == "T1059.001" and first["state"] == "auto_accepted"
    assert {"dense_score", "lexical_rank", "explicit_cue", "state_reason"} <= set(first)
    assert len(report["pipeline_fingerprint"]) == 64
    assert report["fingerprint_inputs"]["cues_enabled"] is True


def test_cti_hal_converter_builds_cases_and_reports(tmp_path: Path):
    from scripts.attribution_benchmark import load_report_folds
    from scripts.prepare_cti_hal import convert

    source = tmp_path / "cti_hal"
    (source / "data" / "fin6").mkdir(parents=True)
    (source / "data" / "unknowngroup").mkdir(parents=True)
    (source / "data" / "fin6" / "REPORT1.json").write_text(
        json.dumps(
            [
                {
                    "context": "Adfind was used to query Active Directory",
                    "technique": "T1018",
                    "metadata": {"sub_technique": None},
                },
                {
                    "context": "Adfind was used to query Active Directory",
                    "technique": "T1087",
                    "metadata": {"sub_technique": "T1087.002"},
                },
                {"context": "short", "technique": "T1105", "metadata": {}},
                {
                    "context": "data compressed with 7-Zip",
                    "technique": "T1560",
                    "metadata": {"sub_technique": "T1560.001"},
                },
                {"context": "unmapped label case", "technique": "T9999", "metadata": {}},
            ]
        ),
        encoding="utf-8",
    )
    (source / "data" / "unknowngroup" / "X.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "out"
    stats = convert(
        source, output, kb_techniques={"T1018", "T1087", "T1087.002", "T1105", "T1560", "T1560.001"}
    )
    assert stats["reports"] == 1 and stats["unknown_group"] == 1 and stats["unmapped_label"] == 1
    cases = [json.loads(line) for line in (output / "cti_hal_test.jsonl").read_text().splitlines()]
    assert len(cases) == 3  # two labels on one statement collapse into one multi-label case
    multi = next(case for case in cases if case["text"].startswith("Adfind"))
    assert multi["expected_technique_ids"] == ["T1018", "T1087.002"]
    assert multi["metadata"]["group_id"] == "G0037"
    folds = load_report_folds(output / "reports.jsonl", max_observations=30)
    assert len(folds) == 1 and folds[0].group_id == "G0037"
    assert folds[0].report_id == "cti_hal-fin6-REPORT1"
    assert len(folds[0].observations) == 2  # the one-word statement is dropped
    assert folds[0].held_out_texts == frozenset()


def test_attribution_summary_keeps_multiple_reports_per_group_distinct():
    from scripts.attribution_benchmark import summarize

    def row(group, rank, stage, report):
        return {
            "group_id": group,
            "report_id": report,
            "rank": rank,
            "top3": [group] if rank == 1 else ["G0999", group],
            "stage": stage,
            "technique_recall": 0.5,
            "observations": 5,
        }

    known = [
        row("G0037", 1, "known_group_hypothesis", "r1"),
        row("G0037", 2, "known_group_hypothesis", "r2"),
        row("G0037", 1, "known_group_shortlist", "r3"),
        row("G0046", 2, "known_group_hypothesis", "r4"),
    ]
    metrics = summarize(known, [])
    assert metrics["known_count"] == 4
    assert metrics["top1"] == pytest.approx(0.5)
    # three named hypotheses, one correct -> named risk 2/3, coverage 3/4
    assert metrics["named_coverage"] == pytest.approx(0.75)
    assert metrics["named_risk"] == pytest.approx(2 / 3)


def test_attribution_pipeline_overrides_keep_local_offline_and_let_set_win():
    from scripts.attribution_benchmark import pipeline_overrides

    local = pipeline_overrides("local", [])
    # the engine applies the mapping preset last, so the verifier is switched off via preset
    assert local["technique_mapping_preset"] == "local_only"
    assert local["cyberre_weight"] == 0.0
    # the LLM zero-shot vote left the ensemble: no pipeline sets its weight any more
    assert not any("zero_shot" in key for key in local)
    production = pipeline_overrides("production", [])
    assert "technique_verifier_mode" not in production
    assert production["cyberre_weight"] == 0.0
    forced = pipeline_overrides("production", ["cyberre_weight=0.2"])
    assert forced["cyberre_weight"] == 0.2
    with pytest.raises(ValueError):
        pipeline_overrides("production", ["nonsense"])
    with pytest.raises(ValueError):
        pipeline_overrides("staging", [])


def test_attribution_summary_reports_deterministic_ranking_and_llm_cost():
    from scripts.attribution_benchmark import summarize

    def row(group, rank, det_rank, cost, errors=0):
        return {
            "group_id": group,
            "report_id": f"r-{group}-{rank}",
            "rank": rank,
            "deterministic_rank": det_rank,
            "top3": [group] if rank == 1 else ["G0999", group],
            "stage": "known_group_shortlist",
            "technique_recall": 0.5,
            "observations": 5,
            "llm": {
                "verifier_calls": 5,
                "verifier_errors": errors,
                "verifier_cost_usd": cost,
                "cost_usd": cost,
            },
        }

    known = [row("G0037", 1, 2, 0.01), row("G0046", 2, 1, 0.02, errors=1)]
    metrics = summarize(known, [])
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["deterministic_top1"] == pytest.approx(0.5)
    assert metrics["deterministic_mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["llm"] == {
        "verifier_calls": 10,
        "verifier_errors": 1,
        "cost_usd": pytest.approx(0.03),
    }
    # rows without the production fields keep the old shape
    for item in known:
        item.pop("deterministic_rank")
        item.pop("llm")
    plain = summarize(known, [])
    assert "deterministic_top1" not in plain and "llm" not in plain


def test_attribution_production_detects_total_llm_failure():
    from scripts.attribution_benchmark import _all_llm_calls_failed

    failed = {"verifier_calls": 27, "verifier_errors": 27}
    partial = {"verifier_calls": 27, "verifier_errors": 3}
    none = {"verifier_calls": 0, "verifier_errors": 0}
    assert _all_llm_calls_failed(failed)
    assert not _all_llm_calls_failed(partial)
    assert not _all_llm_calls_failed(none)  # local pipeline: nothing was attempted


def test_attribution_sample_folds_spreads_over_groups():
    from scripts.attribution_benchmark import Fold, sample_folds

    def fold(group, report):
        return Fold(
            group_id=group,
            held_out_techniques=frozenset(),
            observations=("x",),
            held_out_texts=frozenset(),
            report_id=report,
        )

    folds = [fold("G0008", f"c{i}") for i in range(10)] + [fold("G0037", f"f{i}") for i in range(3)]
    chosen = sample_folds(folds, 6, seed=7)
    assert len(chosen) == 6
    groups = [f.group_id for f in chosen]
    assert groups.count("G0037") == 3 and groups.count("G0008") == 3
    assert sample_folds(folds, 6, seed=7) == chosen  # seeded
    assert sample_folds(folds, 50, seed=7) == folds


def test_collector_summarizes_production_attribution_artifact():
    from scripts.collect_verification import summarize_attribution

    payload = {
        "parameters": {"pipeline": "production", "sample": 10, "truncated_at": None},
        "pipeline_fingerprint": "fp",
        "metrics": {
            "known_count": 10,
            "top1": 0.5,
            "deterministic_top1": 0.5,
            "deterministic_top5": 0.8,
            "deterministic_mrr": 0.666,
            "llm": {
                "verifier_calls": 165,
                "verifier_errors": 0,
                "zero_shot_applied": 10,
                "cost_usd": 3.37,
            },
        },
    }
    summary = summarize_attribution(payload)
    assert summary["parameters"]["pipeline"] == "production"
    assert summary["llm"]["verifier_errors"] == 0
    assert summary["deterministic_mrr"] == 0.666


def test_attribution_summary_reports_separation_auc_and_named_correct():
    from scripts.attribution_benchmark import summarize

    def row(group, rank, sep, stage="known_group_shortlist"):
        return {
            "group_id": group,
            "report_id": f"r-{group}-{rank}-{sep}",
            "rank": rank,
            "top3": [group] if rank == 1 else ["G0999", group],
            "stage": stage,
            "technique_recall": 0.5,
            "observations": 5,
            "margin": 0.2 if rank == 1 else 0.05,
            "separation": sep,
        }

    known = [
        row("G0037", 1, 0.9, "known_group_hypothesis"),
        row("G0037", 1, 0.6),
        row("G0046", 2, 0.1),
        row("G0008", 3, 0.2),
    ]
    metrics = summarize(known, [])
    assert metrics["named_correct"] == 1
    assert metrics["separation_auc"] == pytest.approx(1.0)
    assert metrics["margin_auc"] == pytest.approx(1.0)
    for item in known:
        item.pop("separation")
    assert "separation_auc" not in summarize(known, [])


def test_gate_sweep_selects_a_clean_rule_on_synthetic_artifacts(tmp_path, capsys):
    from scripts.attribution_gate_sweep import main, recombine

    def row(group, rank, leader_channels, runner_channels, margin=0.2):
        topk = [
            {"group_id": "L", "rrf": 0.05, "total": 0.0, **leader_channels},
            {"group_id": "R", "rrf": 0.04, "total": 0.0, **runner_channels},
        ]
        return {
            "group_id": group,
            "rank": rank,
            "stage": "known_group_shortlist",
            "margin": margin,
            "maturity": 0.7,
            "separation": recombine(
                {"topk": topk}, w_t=1.0, w_b=2.0, w_s=1.5, w_n=0.3, floor=1.0, k=5
            )[0],
            "leader_specificity": 0.5,
            "topk": topk,
        }

    strong = {"technique": 2.0, "behavior": 0.5, "software": 1.0, "negative": 0.0}
    weak = {"technique": 0.1, "behavior": 0.0, "software": 0.0, "negative": 0.0}
    artifact = {
        "known": [
            row("A", 1, strong, weak),
            row("B", 1, strong, weak),
            row("C", 2, weak, strong),  # wrong leader with little exclusive evidence
        ],
        "unknown": [row("Z", None, weak, weak, margin=0.0)],
    }
    path = tmp_path / "attr.json"
    path.write_text(json.dumps(artifact))
    assert main([str(path), "--top", "3", "--weights-grid", "1"]) == 0
    out = capsys.readouterr().out
    assert "AUC separation" in out and "1.0" in out
    assert "clean rules" in out
    # the best clean rule names both correct leaders and none of the wrong one
    assert "attr=2(2)" in out
