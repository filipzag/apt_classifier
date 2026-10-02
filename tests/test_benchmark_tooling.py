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


def test_fold_anonymization_replaces_sentence_end_names_and_still_hides_the_index_form():
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
    from apt_classifier.vectorstore import normalize_procedure_text
    from scripts.attribution_benchmark import build_folds

    sentences = {
        "T1": "The loader was attributed to Axiom.",
        "T2": "Operators dropped Axiom.dll before Axiom's beacon ran.",
        "T3": "Collection was done by the Group 72.",
        "T4": "Axiom staged archives on the file server.",
        "T5": "Axiom deleted logs with a custom wiper.",
    }
    kb = AttackKnowledgeBase(
        version="t",
        techniques={tid: Technique(id=tid, name=tid) for tid in sentences},
        groups={
            "G1": Group(
                id="G1",
                name="Axiom",
                aliases=["Group 72"],
                technique_ids=list(sentences),
                technique_behaviors={tid: [text] for tid, text in sentences.items()},
            )
        },
    )
    fold = build_folds(
        kb, coverage=1.0, seed=1, min_behaviors=5, min_techniques=5, max_observations=12
    )[0]
    texts = set(fold.observations)
    assert "The loader was attributed to the adversary." in texts
    assert "Collection was done by the adversary." in texts
    # a dot followed by a word character protects file names; the possessive is replaced
    assert "Operators dropped Axiom.dll before the adversary's beacon ran." in texts
    # the technique index keeps the index pattern's form ("... attributed to Axiom."): both forms
    # are hidden, so the held-out sentence cannot come back through the index
    assert normalize_procedure_text("The loader was attributed to Axiom.") in fold.held_out_texts
    assert normalize_procedure_text("The loader was attributed to the adversary.") in (
        fold.held_out_texts
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


def test_report_folds_anonymize_actor_names_and_count_lost_software(tmp_path: Path):
    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software
    from scripts.attribution_benchmark import anonymize_report_folds, load_report_folds, main

    # "Carbanak" is a group (G0008) and a malware (S0030), as in ATT&CK
    kb = AttackKnowledgeBase(
        version="t",
        groups={
            "G0008": Group(id="G0008", name="Carbanak", aliases=["Anunak"]),
            "G0037": Group(id="G0037", name="FIN6", aliases=["ITG08"]),
            "G0046": Group(id="G0046", name="FIN7"),
        },
        software={
            "S0030": Software(
                id="S0030", name="Carbanak", software_type="malware", aliases=["Anunak"]
            ),
            "S0002": Software(id="S0002", name="Mimikatz", software_type="tool"),
        },
    )

    def statement(text: str) -> dict:
        return {"text": text, "technique_ids": ["T1059"]}

    reports = tmp_path / "reports.jsonl"
    rows = [
        {
            "report_id": "fin6-r1",
            "group_id": "G0037",
            "statements": [
                statement("FIN6 used Mimikatz to dump credentials"),
                statement("ITG08's operators staged the archives on a share"),
                statement("The FIN7 group reused the same loader"),
                statement("data  compressed with 7-Zip"),
            ],
        },
        {
            "report_id": "fin7-r1",
            "group_id": "G0046",
            "statements": [
                statement("the loader launched the CARBANAK DLL from the registry"),
                statement("scheduled tasks ran the loader nightly"),
            ],
        },
    ]
    reports.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    folds = load_report_folds(reports, max_observations=30)

    anonymized, records, totals = anonymize_report_folds(folds, kb)

    fin6, fin7 = anonymized
    assert fin6.observations == (
        "the adversary used Mimikatz to dump credentials",
        "the adversary's operators staged the archives on a share",
        "the adversary group reused the same loader",
        "data  compressed with 7-Zip",  # no name: left byte-identical
    )
    assert fin7.observations == (
        "the loader launched the adversary DLL from the registry",
        "scheduled tasks ran the loader nightly",
    )
    assert (fin6.group_id, fin6.report_id) == ("G0037", "fin6-r1")
    assert fin6.held_out_techniques == folds[0].held_out_techniques
    assert folds[0].observations[0].startswith("FIN6")  # the input folds are not modified
    assert records[0] == {
        "statements_changed": 3,
        "replacements": 3,
        "names": {"fin6": 1, "fin7": 1, "itg08": 1},
        "own_group_named": True,
        "software_matches_removed": 0,  # Mimikatz is not a group name and stays
        "software_ids_removed": [],
    }
    # the FIN7 report only names Carbanak, and that mention is the malware
    assert records[1]["own_group_named"] is False
    assert records[1]["software_matches_removed"] == 1
    assert records[1]["software_ids_removed"] == ["S0030"]
    assert totals["reports_changed"] == 2 and totals["reports_naming_own_group"] == 1
    assert (totals["statements"], totals["statements_changed"], totals["replacements"]) == (6, 4, 4)
    assert totals["software_matches_removed"] == 1
    assert totals["software_ids_removed"] == {"S0030": 1}
    with pytest.raises(SystemExit):  # the ATT&CK folds are always anonymized
        main(["--anonymize-actors"])


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


def test_attribution_default_output_matches_the_makefile_and_never_collides():
    import shlex

    from scripts.attribution_benchmark import REPO_ROOT, default_output
    from scripts.report_anonymizer import LEVELS

    modes = [
        (pipeline, real_reports, anonymize)
        for pipeline in ("local", "production")
        for real_reports, anonymize in ((False, False), (True, True), (True, False))
    ]
    paths = [default_output(p, real_reports=r, anonymize_actors=a) for p, r, a in modes]
    # the Elastic set and the other anonymization levels add their own files
    paths += [
        default_output(p, real_reports=True, anonymize_actors=False, report_set=s, anonymization=lv)
        for p in ("local", "production")
        for s, levels in (("cti_hal", ("strict", "tradecraft")), ("elastic", LEVELS + (None,)))
        for lv in levels
    ]
    assert len(set(paths)) == len(paths)  # no mode overwrites another mode's artifact
    assert (
        default_output(
            "local",
            real_reports=True,
            anonymize_actors=False,
            report_set="elastic",
            anonymization="strict",
        ).name
        == "attribution_elastic_current.json"
    )  # strict is the Elastic headline
    assert (
        default_output(
            "local",
            real_reports=True,
            anonymize_actors=True,
            report_set="cti_hal",
            anonymization="names",
        ).name
        == "attribution_cti_hal_current.json"
    )  # names stays the CTI-HAL headline
    assert default_output("local", real_reports=True, anonymize_actors=False).name == (
        "attribution_cti_hal_as_reported.json"
    )
    # every Makefile call writes where the same flags write without --output
    calls = 0
    for line in (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if "scripts/attribution_benchmark.py" not in line:
            continue
        words = shlex.split(line)
        flags = words[words.index("scripts/attribution_benchmark.py") + 1 :]
        expected = default_output(
            flags[flags.index("--pipeline") + 1] if "--pipeline" in flags else "local",
            real_reports="--cti-hal" in flags,
            anonymize_actors="--anonymize-actors" in flags,
            report_set=flags[flags.index("--report-set") + 1]
            if "--report-set" in flags
            else "cti_hal",
            # exact token: "--anonymize" must not match "--anonymize-actors"
            anonymization=flags[flags.index("--anonymize") + 1] if "--anonymize" in flags else None,
        )
        written = (
            REPO_ROOT / flags[flags.index("--output") + 1] if "--output" in flags else expected
        )
        assert written == expected, line
        calls += 1
    # benchmark-attribution, benchmark-cti-hal (2), benchmark-elastic (4), three production targets
    assert calls >= 10


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


def test_collector_keeps_as_reported_cti_hal_beside_the_anonymized_headline():
    from scripts.collect_verification import compose, cti_hal_headline_warning

    def artifact(top1: float, actor_names: str | None = None) -> dict:
        parameters: dict = {"known_folds": 55}
        if actor_names:
            parameters["actor_names"] = actor_names
        return {"parameters": parameters, "metrics": {"known_count": 55, "top1": top1}}

    def report(headline: dict) -> dict:
        return compose(
            public=None,
            curated=None,
            evidenceforge={},
            public_path=Path("missing.json"),
            curated_path=Path("missing.json"),
            gates=None,
            cti_hal_attribution=headline,
            cti_hal_attribution_path=Path("attribution_cti_hal_current.json"),
            cti_hal_attribution_as_reported=artifact(0.545, "as_reported"),
            cti_hal_attribution_as_reported_path=Path("attribution_cti_hal_as_reported.json"),
        )

    current = report(artifact(0.436, "anonymized"))
    assert current["cti_hal_attribution"]["top1"] == 0.436
    assert current["cti_hal_attribution"]["parameters"]["actor_names"] == "anonymized"
    assert current["cti_hal_attribution_as_reported"]["top1"] == 0.545
    assert current["sources"]["cti_hal_attribution_as_reported"]["path"].endswith(
        "as_reported.json"
    )
    assert "warnings" not in current
    # artifacts from before 2026-09-30 have no actor_names: they read the names as reported
    stale = report(artifact(0.545))
    assert stale["warnings"] == [cti_hal_headline_warning(artifact(0.545))]
    assert cti_hal_headline_warning(artifact(0.545, "as_reported")) is not None
    assert cti_hal_headline_warning(None) is None


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


def test_unknown_arm_refits_only_the_voting_forest(monkeypatch):
    from types import SimpleNamespace

    from scripts import attribution_benchmark as bench

    calls = {}

    def fake_fit(kb, *, params, device, reuse_group_forest=None):
        calls.update(kb=kb, params=params, device=device, reuse=reuse_group_forest)
        return "refit"

    monkeypatch.setattr(bench, "fit_profile_forest", fake_fit)
    model = SimpleNamespace(metadata={"params": {"seed": 7}}, device="cpu")
    engine = SimpleNamespace(
        profile_forest=model,
        default_weights=lambda: {"profile_forest": 0.0, "source_country": 1.0},
    )
    assert bench._forest_votes(engine)
    assert bench._refit_forest_without_actor(engine, "reduced") == "refit"
    # the silent group forest is reused, the voting country forest is refitted
    assert calls["reuse"] is model and calls["params"] == {"seed": 7} and calls["kb"] == "reduced"
    engine.default_weights = lambda: {"profile_forest": 0.5, "source_country": 0.0}
    bench._refit_forest_without_actor(engine, "reduced")
    assert calls["reuse"] is None
    engine.default_weights = lambda: {"profile_forest": 0.0, "source_country": 0.0}
    assert not bench._forest_votes(engine)
    assert not bench._forest_votes(
        SimpleNamespace(profile_forest=None, default_weights=lambda: {"source_country": 1.0})
    )


def test_attribution_summary_reports_top_one_given_top_three_and_the_guard():
    from scripts.attribution_benchmark import summarize

    def row(group, rank, guard=False):
        return {
            "group_id": group,
            "report_id": f"r-{group}-{rank}",
            "rank": rank,
            "top3": [group] if rank == 1 else ["G0999", group],
            "stage": "known_group_shortlist",
            "technique_recall": 0.5,
            "observations": 5,
            "leader_guard_fired": guard,
        }

    known = [row("G1", 1), row("G2", 2), row("G3", 1, guard=True), row("G4", 3), row("G5", 7)]
    known[4]["top3_related_to_truth"] = True  # a documented-overlap group made the top three
    for item in known[:4]:
        item["top3_related_to_truth"] = False
    metrics = summarize(known, [])
    assert metrics["top3"] == pytest.approx(4 / 5)
    assert metrics["cluster_top3"] == pytest.approx(5 / 5)
    assert metrics["top1_given_top3"] == pytest.approx(2 / 4)
    assert metrics["pairwise_12"] == pytest.approx(2 / 3)
    assert metrics["leader_guard_fired"] == 1
    # rows recorded before the instrumentation keep the old shape
    for item in known:
        item.pop("leader_guard_fired")
    assert "leader_guard_fired" not in summarize(known, [])


def test_attribution_benchmark_lists_rows_scored_without_an_enabled_combiner():
    from scripts.attribution_benchmark import _inactive_combiner_rows

    rows = [
        {"group_id": "G1", "combiner_status": "active"},
        {"group_id": "G2", "report_id": "r2", "combiner_status": "incompatible artifact: x"},
        {"group_id": "G3", "combiner_status": None},  # no fusion, no status
    ]
    assert _inactive_combiner_rows(rows) == ["G2:r2"]


# ----------------------------------------------------------------------------- instrumentation


def test_output_guard_refuses_shared_paths_and_overwrites(tmp_path: Path):
    from scripts.run_tooling import guard_output_paths

    repo = tmp_path / "repo"
    shared = tmp_path / "shared_models"
    (repo / "data").mkdir(parents=True)
    shared.mkdir()
    # a worktree's data/models is a symlink to the main checkout's shared models
    (repo / "data" / "models").symlink_to(shared, target_is_directory=True)
    inside = repo / "data" / "audits" / "model.json"
    through_link = repo / "data" / "models" / "model.json"
    assert guard_output_paths([inside, None], repo_root=repo) == []
    problems = guard_output_paths([through_link], repo_root=repo)
    assert len(problems) == 1 and "--allow-shared-output" in problems[0]
    assert guard_output_paths([through_link], repo_root=repo, allow_shared=True) == []
    inside.parent.mkdir(parents=True)
    inside.write_text("{}", encoding="utf-8")
    problems = guard_output_paths([inside], repo_root=repo)
    assert len(problems) == 1 and "--overwrite" in problems[0]
    assert guard_output_paths([inside], repo_root=repo, overwrite=True) == []


@pytest.mark.parametrize(
    "module, extra, suffix",
    [
        ("scripts.fit_acceptance_model", [], ".json"),
        ("scripts.build_exemplar_index", [], ".npz"),
        (
            "scripts.fit_attribution_combiner",
            ["--lists-cache", "{tmp}/lists.jsonl.gz", "--report", "{tmp}/cv.json"],
            ".json",
        ),
    ],
)
def test_fit_scripts_refuse_unsafe_outputs_before_any_work(module, extra, suffix, tmp_path, capsys):
    import importlib

    main = importlib.import_module(module).main
    existing = tmp_path / f"existing{suffix}"
    existing.write_text("{}", encoding="utf-8")
    flags = [item.format(tmp=tmp_path) for item in extra]
    # tmp_path lies outside the repository: refused without --allow-shared-output
    with pytest.raises(SystemExit) as outside:
        main(["--output", str(tmp_path / f"new{suffix}"), *flags])
    assert outside.value.code == 2
    assert "--allow-shared-output" in capsys.readouterr().err
    # allowed outside, but an existing file is never replaced without --overwrite
    with pytest.raises(SystemExit) as overwrite:
        main(["--output", str(existing), "--allow-shared-output", *flags])
    assert overwrite.value.code == 2
    assert "--overwrite" in capsys.readouterr().err


def test_exemplar_build_guards_the_file_numpy_writes(tmp_path, capsys):
    """Regression: numpy appends .npz to an --output without it, so the guard must check that
    file, not the path as typed (which never exists and let the library be replaced)."""
    import numpy as np

    from apt_classifier.exemplars import ExemplarIndex, npz_path
    from scripts.build_exemplar_index import main

    assert npz_path(tmp_path / "lib") == tmp_path / "lib.npz"
    assert npz_path(tmp_path / "lib.v2") == tmp_path / "lib.v2.npz"
    assert npz_path(tmp_path / "lib.npz") == tmp_path / "lib.npz"
    shipped = tmp_path / "exemplar_index.npz"
    shipped.write_bytes(b"the shipped library")
    with pytest.raises(SystemExit) as refused:
        main(["--output", str(tmp_path / "exemplar_index"), "--allow-shared-output"])
    assert refused.value.code == 2 and "--overwrite" in capsys.readouterr().err
    assert shipped.read_bytes() == b"the shipped library"
    # save writes, and reports, the same file the guard checked
    index = ExemplarIndex(
        embeddings=np.eye(2, dtype=np.float32),
        labels=(("T1059",), ("T1105",)),
        texts=("a", "b"),
        sources=("tram", "tram"),
        metadata={"kind": "technique_exemplar_index", "schema_version": 1},
    )
    written = index.save(tmp_path / "other")
    assert written == tmp_path / "other.npz" and written.is_file()
    assert not (tmp_path / "other").exists()


def test_makefile_fit_targets_pass_overwrite():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    for script in (
        "fit_acceptance_model.py",
        "build_exemplar_index.py",
        "fit_attribution_combiner.py",
    ):
        lines = [line for line in makefile.splitlines() if f"scripts/{script}" in line]
        assert lines and all("--overwrite" in line for line in lines), script
        # in a worktree the default outputs resolve outside the repository and stay refused
        assert all("--allow-shared-output" not in line for line in lines), script


def _fake_mapping_pipeline():
    from types import SimpleNamespace

    from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
    from apt_classifier.retrieve import LexicalTechniqueIndex
    from apt_classifier.vectorstore import VectorHit

    kb = AttackKnowledgeBase(
        version="t",
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1105": Technique(id="T1105", name="Ingress Tool Transfer"),
        },
    )

    class Store:
        def query_techniques(self, texts, *, n_results, **kwargs):
            out = []
            for text in texts:
                lower, hits = text.lower(), []
                if "powershell" in lower:
                    hits.append(
                        VectorHit(
                            "a",
                            "PowerShell",
                            {"technique_id": "T1059.001", "field": "description"},
                            0.1,
                            0.9,
                        )
                    )
                if "download" in lower:
                    hits.append(
                        VectorHit(
                            "b",
                            "Transfer",
                            {"technique_id": "T1105", "field": "description"},
                            0.3,
                            0.7,
                        )
                    )
                out.append(hits)
            return out

        def query_group_profile(self, texts, *, document_type, n_results):
            return []

        def query_group_behaviors(self, texts, *, n_results):
            return [[] for _ in texts]

    class Model:
        """Stand-in acceptance model: p is the dense score (0 without a dense hit)."""

        metadata: dict = {}

        def __init__(self, threshold: float):
            self.threshold = threshold

        def probability(self, features):
            return 0.05 + 0.9 * float(features.get("dense_score", 0.0))

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
    return kb, Store(), LexicalTechniqueIndex.from_kb(kb), Model, settings


def test_candidate_rows_record_probability_exemplar_features_and_guards():
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from scripts.benchmark_mapping import run

    kb, store, lexical, Model, settings = _fake_mapping_pipeline()
    cases = [
        TechniqueEvaluationCase.from_dict(
            _row("tram", 1, "PowerShell ran an encoded command and a download", ["T1059.001"])
        ),
        # benign context: similarity alone may not accept
        TechniqueEvaluationCase.from_dict(
            _row("tram", 2, "Routine maintenance used PowerShell scripts to download", ["T1105"])
        ),
        # fewer than three distinct words
        TechniqueEvaluationCase.from_dict(_row("tram", 3, "powershell download", ["T1105"])),
    ]
    block = run(
        cases,
        mode="hybrid",
        settings=settings,
        kb=kb,
        store=store,
        lexical_index=lexical,
        acceptance_model=Model(0.5),
        bootstrap_samples=10,
    )
    rows = {row["case_id"]: row for row in block["cases"]}
    first = {item["technique_id"]: item for item in rows["tram-test-000001"]["candidates"]}
    powershell = first["T1059.001"]
    assert powershell["acceptance_probability"] == pytest.approx(0.05 + 0.9 * 0.9)
    assert powershell["cue_supported"] is True  # explicit cue + dense 0.9 >= 0.45
    assert powershell["benign"] is False and powershell["sparse"] is False
    assert powershell["parent_technique_id"] == "T1059"
    # no exemplar index: the features are absent, not zero votes
    assert powershell["ex_vote"] is None and powershell["exemplar_only"] is False
    assert {
        "ex_family",
        "ex_maxsim_rel",
        "ex_rank_inv",
        "ex_count",
        "ex_top_sim",
        "ex_vote_x_top",
    } <= set(powershell)
    # p clears the threshold, but a benign or sparse context forbids similarity-only acceptance
    benign = {item["technique_id"]: item for item in rows["tram-test-000002"]["candidates"]}
    assert benign["T1059.001"]["benign"] is True and benign["T1059.001"]["sparse"] is False
    assert benign["T1059.001"]["acceptance_probability"] > 0.5
    assert benign["T1059.001"]["state"] == "needs_review"
    sparse = {item["technique_id"]: item for item in rows["tram-test-000003"]["candidates"]}
    assert sparse["T1059.001"]["sparse"] is True and sparse["T1059.001"]["state"] == "needs_review"


def test_benchmark_run_records_a_model_refused_inside_retrieval():
    """Regression: a --modes value outside a stored contract runs the rule-based gate; the run
    block must say so (not active, the reason, no model threshold in its fingerprint) instead of
    claiming the calibrated model ran. A contract-less (legacy) model still runs in every mode,
    as it did before contracts."""
    from apt_classifier.acceptance import legacy_contract, retrieval_contract
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from apt_classifier.retrieve import retrieval_kwargs
    from scripts.benchmark_mapping import run

    kb, store, lexical, Model, settings = _fake_mapping_pipeline()
    cases = [
        TechniqueEvaluationCase.from_dict(
            _row("tram", 1, "PowerShell ran an encoded command and a download", ["T1059.001"])
        ),
        TechniqueEvaluationCase.from_dict(
            _row("tram", 2, "The operator ran a powershell download cradle", ["T1105"])
        ),
    ]
    stored = Model(0.5)
    stored.contract = retrieval_contract(
        retrieval_kwargs(settings, exemplar_index=None, lexical_index=lexical), None
    )
    legacy = Model(0.5)
    legacy.contract = legacy_contract({})

    def block(mode, model):
        return run(
            cases,
            mode=mode,
            settings=settings,
            kb=kb,
            store=store,
            lexical_index=lexical,
            acceptance_model=model,
            bootstrap_samples=5,
        )

    def probabilities(result):
        return [
            row["acceptance_probability"] for case in result["cases"] for row in case["candidates"]
        ]

    hybrid = block("hybrid", stored)
    assert hybrid["acceptance_model_active"] is True
    assert "acceptance_model_disabled" not in hybrid
    assert hybrid["fingerprint_inputs"]["acceptance_model"]["threshold"] == 0.5
    assert all(value is not None for value in probabilities(hybrid))
    for mode in ("lexical", "legacy_dense"):
        refused = block(mode, stored)
        assert refused["acceptance_model_active"] is False, mode
        assert refused["acceptance_model_disabled"]["cases"] == len(cases)
        (reason,) = refused["acceptance_model_disabled"]["reasons"]
        assert f"retrieval_mode '{mode}' (fitted with 'hybrid')" in reason
        assert all(value is None for value in probabilities(refused))
        # the fingerprint is the rule-based pipeline's, not one with the model's threshold
        assert refused["fingerprint_inputs"]["acceptance_model"] is None
        assert refused["pipeline_fingerprint"] == block(mode, None)["pipeline_fingerprint"]
        # the legacy model is not held to the retrieval mode and scores every case
        unconstrained = block(mode, legacy)
        assert unconstrained["acceptance_model_active"] is True, mode
        assert "acceptance_model_disabled" not in unconstrained
        assert any(value is not None for value in probabilities(unconstrained))


def test_exemplar_columns_carry_the_eight_model_features():
    from types import SimpleNamespace

    from apt_classifier.exemplars import FEATURE_NAMES
    from scripts.benchmark_mapping import _exemplar_columns

    features = {name: 0.25 for name in FEATURE_NAMES} | {"exemplar_only": 1.0}
    evidence = SimpleNamespace(metadata={"features": features, "exemplar_only": True})
    columns = _exemplar_columns(evidence)
    assert set(columns) == set(FEATURE_NAMES) and len(columns) == 8
    assert columns["ex_vote"] == 0.25 and columns["exemplar_only"] is True


def test_attribution_fingerprint_covers_the_mapping_modules():
    from scripts.attribution_benchmark import FINGERPRINTED_SOURCES

    names = {path.name for path in FINGERPRINTED_SOURCES}
    assert {"exemplars.py", "acceptance.py", "vectorstore.py", "retrieve.py"} <= names
    assert all(path.is_file() for path in FINGERPRINTED_SOURCES)


def test_attribution_summary_adds_technique_micro_metrics():
    from scripts.attribution_benchmark import summarize

    def row(group, rank, accepted, gold):
        return {
            "group_id": group,
            "report_id": f"r-{group}",
            "rank": rank,
            "top3": [group],
            "stage": "known_group_shortlist",
            "technique_recall": len(set(accepted) & set(gold)) / len(gold) if gold else None,
            "accepted_techniques": accepted,
            "held_out_techniques": gold,
        }

    known = [
        row("G1", 1, ["T1", "T2", "T3"], ["T1", "T2"]),
        row("G2", 2, ["T4"], ["T5", "T6"]),
        row("G3", 1, ["T7", "T8"], []),  # no gold labels: only the median counts it
    ]
    metrics = summarize(known, [])
    assert metrics["technique_micro_precision"] == pytest.approx(2 / 4)
    assert metrics["technique_micro_recall"] == pytest.approx(2 / 4)
    assert metrics["technique_micro_counts"] == {
        "true_positive": 2,
        "accepted": 4,
        "gold": 4,
        "reports": 2,
    }
    assert metrics["median_accepted_techniques"] == 2.0


def test_evidenceforge_metadata_records_run_identity(tmp_path: Path):
    from types import SimpleNamespace

    from scripts.evidenceforge_eval import collect_run_metadata

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "GROUND_TRUTH.json").write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        embedding_model="embed",
        embedding_model_revision="rev",
        technique_verifier_model="verify-model",
        reranker_model="reranker",
        reranker_model_revision="reranker-rev",
        technique_retrieval_mode="hybrid",
        technique_candidate_pool=12,
        retrieval_top_k=30,
        similarity_threshold=0.45,
        technique_acceptance_threshold=0.75,
        technique_relative_margin=0.08,
        attack_version="latest",
        acceptance_model_path=tmp_path / "acceptance.json",
        acceptance_model_fallback_path=tmp_path / "fallback.json",
        technique_exemplar_index_path=tmp_path / "exemplars.npz",
    )
    (tmp_path / "acceptance.json").write_text("{}", encoding="utf-8")

    class Index:
        fingerprint = "fp123"

        def __len__(self):
            return 3

    def engine(threshold):
        return SimpleNamespace(
            acceptance_model=SimpleNamespace(threshold=threshold, exemplar_fingerprint="fp123"),
            acceptance_model_status="active",
            exemplar_index=Index(),
            exemplar_status="active",
        )

    def metadata(**extra):
        return collect_run_metadata(
            bundle_dir=tmp_path,
            scenario_path=scenario,
            settings=settings,
            kb=SimpleNamespace(version="15.1"),
            preset="balanced",
            pipeline="local",
            **extra,
        )

    plain = metadata()
    assert plain["generated_at"] and "git_commit" in plain and plain["acceptance_model"] is None
    first, second = metadata(engine=engine(0.21)), metadata(engine=engine(0.30))
    assert first["acceptance_model"]["threshold"] == 0.21
    assert first["acceptance_model"]["sha256"]  # the artifact actually loaded is hashed
    assert first["exemplars"]["fingerprint"] == "fp123" and first["exemplars"]["count"] == 3
    # a different acceptance artifact never shares a pipeline fingerprint
    assert (
        len(
            {
                plain["pipeline_fingerprint"],
                first["pipeline_fingerprint"],
                second["pipeline_fingerprint"],
            }
        )
        == 3
    )


class _ReadOnlyCollection:
    def __init__(self, name, *, count, metadata, configuration=None, ids=None):
        self.name, self.id = name, f"id-{name}"
        self._count, self.metadata = count, metadata
        self.configuration_json = configuration or {"hnsw": {"space": "cosine", "ef_search": 100}}
        self._ids = ids or [f"{name}-{index}" for index in range(count)]
        self.documents = {item: f"text of {item}" for item in self._ids}
        self.vectors = {item: [float(len(item)), 0.25, -1.0] for item in self._ids}
        self.paged_embeddings = False

    def count(self):
        return self._count

    def get(self, *, include, limit=None, offset=None, ids=None):
        assert set(include) <= {"documents", "metadatas", "embeddings"}
        selected = list(ids) if ids is not None else self._ids[offset : offset + limit]
        # vectors only for an explicit sample of ids, never paged through the whole collection
        self.paged_embeddings |= "embeddings" in include and ids is None
        payload = {"ids": selected}
        if "documents" in include:
            payload["documents"] = [self.documents[item] for item in selected]
        if "metadatas" in include:
            payload["metadatas"] = [{"field": "name"} for _ in selected]
        if "embeddings" in include:
            payload["embeddings"] = [self.vectors[item] for item in selected]
        return payload

    def __getattr__(self, name):
        if name in {"add", "upsert", "update", "delete", "modify"}:
            raise AssertionError(f"snapshot called the write method {name}")
        raise AttributeError(name)


class _ReadOnlyClient:
    def __init__(self, collections):
        self.collections = collections

    def list_collections(self):
        return list(self.collections)

    def get_version(self):
        return "1.0.0"

    def __getattr__(self, name):
        if name in {"create_collection", "get_or_create_collection", "delete_collection", "reset"}:
            raise AssertionError(f"snapshot called the write method {name}")
        raise AttributeError(name)


def test_chroma_snapshot_reads_only_and_diff_flags_changes(tmp_path: Path, capsys):
    from scripts.chroma_snapshot import diff_snapshots, main, take_snapshot

    techniques = _ReadOnlyCollection("attack_techniques", count=3, metadata={"schema_version": 4})
    groups = _ReadOnlyCollection("attack_groups", count=2, metadata={"kind": "group_profiles"})
    before = take_snapshot(_ReadOnlyClient([techniques, groups]), hash_ids=True)
    assert before["collections"]["attack_techniques"]["count"] == 3
    assert before["collections"]["attack_techniques"]["ids_sha256"]
    assert before["server_version"] == "1.0.0"

    exemplars = _ReadOnlyCollection("technique_exemplars-0f8d-s1-m32-efc400", count=1, metadata={})
    after = take_snapshot(_ReadOnlyClient([techniques, groups, exemplars]), hash_ids=True)
    assert diff_snapshots(before, after)["clean"]  # only an allowed new collection

    stray = _ReadOnlyCollection("attack_techniques_copy", count=1, metadata={})
    unexpected = diff_snapshots(before, take_snapshot(_ReadOnlyClient([techniques, groups, stray])))
    assert not unexpected["clean"] and unexpected["unexpected_added"] == ["attack_techniques_copy"]

    changed = _ReadOnlyCollection("attack_techniques", count=4, metadata={"schema_version": 5})
    report = diff_snapshots(before, take_snapshot(_ReadOnlyClient([changed, groups])))
    assert not report["clean"] and set(report["changed"]["attack_techniques"]) == {
        "count",
        "metadata",
    }
    assert diff_snapshots(before, take_snapshot(_ReadOnlyClient([techniques])))["removed"] == [
        "attack_groups"
    ]

    path = tmp_path / "before.json"
    assert (
        main(
            ["--output", str(path), "--hash-ids", "--no-hash-content"],
            client=_ReadOnlyClient([techniques, groups]),
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(["--diff", str(path), "--hash-ids"], client=_ReadOnlyClient([techniques, groups])) == 0
    )
    assert main(["--diff", str(path)], client=_ReadOnlyClient([changed, groups])) == 1
    assert not techniques.paged_embeddings and not groups.paged_embeddings


def _snapshot_main(args, client):
    from scripts.chroma_snapshot import main

    return main([str(item) for item in args], client=client)


def test_chroma_snapshot_never_writes_over_the_baseline_it_compares(tmp_path: Path, capsys):
    """Regression: --output naming the --diff file (a copy-pasted "before" command) replaced the
    baseline before it was read, and the diff reported clean."""
    from tests.chroma_fakes import FakeClient

    client = FakeClient()
    techniques = client.seed("attack_techniques", metadata={"schema_version": 4})
    techniques.add(ids=["T1:name:0000"], embeddings=[[0.1, 0.2, 0.3]], documents=["a"])
    client.calls.clear()  # the test's own setup write
    before = tmp_path / "R13_before.json"
    assert _snapshot_main(["--output", before], client) == 0
    baseline = before.read_text(encoding="utf-8")
    techniques.add(ids=["T1:name:0001"], embeddings=[[0.3, 0.2, 0.1]], documents=["b"])
    client.seed("rogue", metadata={})
    client.calls.clear()  # what a peer session did meanwhile
    capsys.readouterr()
    for args in (
        ["--diff", before, "--output", before],
        ["--diff", before, "--output", before, "--overwrite"],
        ["--diff", before, "--diff-output", before],
        ["--output", before],  # re-running the "before" command
    ):
        with pytest.raises(SystemExit) as refused:
            _snapshot_main(args, client)
        assert refused.value.code == 2, args
        assert before.read_text(encoding="utf-8") == baseline, args
    assert "already exists" in capsys.readouterr().err
    after = tmp_path / "R13_after.json"
    assert _snapshot_main(["--diff", before, "--output", after], client) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["unexpected_added"] == ["rogue"] and "attack_techniques" in report["changed"]
    assert _snapshot_main(["--output", after, "--overwrite"], client) == 0
    assert not client.writes()


def test_chroma_snapshot_sees_a_same_shape_reingest(tmp_path: Path, capsys):
    """Regression: a compatible make ingest upserts in place (same collection id, record ids,
    count and metadata); only the content hashes see new text or re-embedded vectors."""
    import numpy as np

    from scripts.chroma_snapshot import diff_snapshots, take_snapshot
    from tests.chroma_fakes import FakeClient

    client = FakeClient()
    techniques = client.seed("attack_techniques", metadata={"schema_version": 4})
    ids = [f"T{index}:procedure:0000" for index in range(40)]
    vectors = np.random.default_rng(3).normal(size=(40, 8)).astype(np.float32)
    techniques.add(
        ids=ids,
        embeddings=vectors,
        documents=[f"sentence {index}" for index in range(40)],
        metadatas=[{"field": "procedure"} for _ in ids],
    )
    client.calls.clear()  # the test's own setup write
    before = take_snapshot(client, hash_content=True, embedding_sample=16)
    shallow = take_snapshot(client, hash_ids=True)
    entry = before["collections"]["attack_techniques"]
    assert entry["records_read"] == 40 and entry["embeddings_sample_size"] == 16
    assert diff_snapshots(before, take_snapshot(client, hash_content=True, embedding_sample=16))[
        "clean"
    ]
    # the same text re-embedded elsewhere: tiny float differences on every vector
    for index, item in enumerate(ids):
        techniques.tamper(item, embedding=vectors[index] + np.float32(1e-6))
    reembedded = diff_snapshots(
        before, take_snapshot(client, hash_content=True, embedding_sample=16)
    )
    assert not reembedded["clean"]
    assert set(reembedded["changed"]["attack_techniques"]) == {"embeddings_sample_sha256"}
    # a regenerated sentence with the same per-technique count
    techniques.tamper(ids[7], document="a different synthetic sentence")
    rewritten = diff_snapshots(
        before, take_snapshot(client, hash_content=True, embedding_sample=16)
    )
    assert "records_sha256" in rewritten["changed"]["attack_techniques"]
    # an ids-only snapshot misses both (the defect)
    assert diff_snapshots(shallow, take_snapshot(client, hash_ids=True))["clean"]
    # a diff hashes at least what its baseline hashed, even when told not to
    path = tmp_path / "before.json"
    path.write_text(json.dumps(before), encoding="utf-8")
    capsys.readouterr()
    assert _snapshot_main(["--diff", path, "--no-hash-content"], client) == 1
    # and an offline AFTER file lacking a hash BEFORE recorded is not clean
    lacking = tmp_path / "after.json"
    lacking.write_text(json.dumps(take_snapshot(client)), encoding="utf-8")
    diff_path = tmp_path / "diff.json"
    args = ["--diff", path, "--after", lacking, "--diff-output", diff_path]
    assert _snapshot_main(args, client) == 1
    report = json.loads(diff_path.read_text(encoding="utf-8"))
    assert report["not_compared"]["attack_techniques"] == [
        "ids_sha256",
        "records_sha256",
        "embeddings_sample_sha256",
    ]
    capsys.readouterr()
    assert not client.writes()


def test_run_ledger_appends_one_redacted_line_per_run(tmp_path: Path, capsys):
    from scripts.chroma_snapshot import take_snapshot
    from scripts.run_ledger import main

    artifact = tmp_path / "run.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_hash": "h1",
                "generated_at": "2026-10-02T00:00:00+00:00",
                "parameters": {"pipeline": "local"},
                "pipeline_fingerprint": "pf",
                "fingerprint_inputs": {"source_fingerprint": "sf", "combiner": "active"},
                "acceptance_model": {
                    "path": "data/models/acceptance/acceptance_model.json",
                    "threshold": 0.21,
                },
                "metrics": {"llm": {"verifier_calls": 0, "verifier_errors": 0, "cost_usd": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    collection = _ReadOnlyCollection("attack_techniques", count=1, metadata={})
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(take_snapshot(_ReadOnlyClient([collection]))), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    environ = {
        "TECHNIQUE_EXEMPLAR_BACKEND": "npz",
        "OPENROUTER_API_KEY": "sk-secret",
        "HOME": "/home/someone",
    }
    for run_id in ("R01", "R02"):
        args = ["append", "--ledger", str(ledger), "--run-id", run_id, "--artifact", str(artifact)]
        args += ["--snapshot-before", str(snapshot), "--snapshot-after", str(snapshot)]
        args += ["--command", "scripts/benchmark_mapping.py --output x", "--verdict", "recorded"]
        assert main(args, environ=environ) == 0
    capsys.readouterr()
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [line["run_id"] for line in lines] == ["R01", "R02"]
    entry = lines[0]
    assert entry["env"] == {"OPENROUTER_API_KEY": "<set>", "TECHNIQUE_EXEMPLAR_BACKEND": "npz"}
    assert "sk-secret" not in ledger.read_text(encoding="utf-8")
    described = entry["artifacts"][0]
    assert described["artifact_hash"] == "h1" and described["pipeline_fingerprints"] == {
        "artifact": "pf"
    }
    assert (
        described["source_fingerprints"] == {"artifact": "sf"} and described["combiner"] == "active"
    )
    assert entry["llm_cost_usd"] == 0.0 and entry["chroma"]["clean"] is True
    assert entry["argv"][0] == "scripts/benchmark_mapping.py"


def test_fingerprints_cover_the_exemplar_store():
    from scripts.attribution_benchmark import FINGERPRINTED_SOURCES as ATTRIBUTION
    from scripts.benchmark_mapping import FINGERPRINTED_SOURCES as MAPPING

    for sources in (ATTRIBUTION, MAPPING):
        assert "exemplar_store.py" in {path.name for path in sources}
        assert all(path.is_file() for path in sources)


def test_exemplar_backend_flags_and_requirement():
    import argparse

    from apt_classifier import exemplar_store
    from scripts.run_tooling import (
        add_exemplar_backend_arguments,
        exemplar_backend_problem,
        exemplar_identity,
        exemplar_setting_overrides,
        exemplar_signature,
    )

    parser = argparse.ArgumentParser()
    add_exemplar_backend_arguments(parser)
    assert exemplar_setting_overrides(parser.parse_args([])) == {}
    args = parser.parse_args(
        [
            "--exemplar-backend",
            "chroma",
            "--exemplar-search",
            "hnsw",
            "--require-exemplar-backend",
            "chroma",
        ]
    )
    assert exemplar_setting_overrides(args) == {
        "technique_exemplar_backend": "chroma",
        "technique_exemplar_search": "hnsw",
    }

    class Chroma:
        backend = "chroma"
        fingerprint = "fp"

        def __len__(self):
            return 2

        def signature(self):
            return {"backend": "chroma", "search": "hnsw", "fingerprint": "fp"}

    class Npz(Chroma):
        backend = "npz"

        def signature(self):
            return {"backend": "npz", "search": "exact", "fingerprint": "fp"}

    exemplar_store.reset_query_totals()
    assert exemplar_backend_problem(Npz(), None) is None  # not required
    assert exemplar_backend_problem(Npz(), "npz") is None
    assert "loaded from npz" in exemplar_backend_problem(Npz(), "chroma")
    assert "no exemplar library" in exemplar_backend_problem(None, "chroma")
    assert exemplar_backend_problem(Chroma(), "chroma") is None
    exemplar_store._add_totals({"queries": 3, "search_fallbacks": 3})  # noqa: SLF001
    assert "fell back" in exemplar_backend_problem(Chroma(), "chroma")
    identity = exemplar_identity(argparse.Namespace(), Chroma(), "active")
    assert identity["backend"] == "chroma" and identity["signature"]["search"] == "hnsw"
    assert identity["query_totals"]["search_fallbacks"] == 3
    exemplar_store.reset_query_totals()
    assert (
        exemplar_signature(object()) is None
        and exemplar_identity(None, None, "off")["backend"] is None
    )


# ----- fit / serve alignment (retrieval_kwargs, runtime guards, threshold selection) -----


class _StopRetrieval(Exception):
    pass


def _capture_retrieve(monkeypatch, module) -> dict:
    captured: dict = {}

    def recorder(ledger, store, kb, **kwargs):
        captured.update(kwargs)
        raise _StopRetrieval

    monkeypatch.setattr(module, "retrieve_techniques", recorder)
    return captured


CALL_SITE_KEYS = {"lexical_index", "acceptance_model", "reranker", "finalize", "cues_enabled"}


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"technique_exemplar_k": 128, "technique_dense_relative_margin": 0.2},
        {"technique_include_synthetic": True, "technique_segment_acceptance": True},
    ],
)
def test_engine_benchmark_and_fit_retrieve_with_the_same_arguments(monkeypatch, changes):
    """The serving call, the public benchmark and the acceptance fit cannot drift: they pass
    retrieve_techniques the same settings-driven arguments (the fit used to add the synthetic
    sentences and leave out segment acceptance)."""
    import apt_classifier.session as session_module
    import scripts.benchmark_mapping as benchmark_module
    import scripts.fit_acceptance_model as fit_module
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from apt_classifier.retrieve import LexicalTechniqueIndex, lexical_fields
    from tests.test_session import FakeLLM, FakeStore, _kb, _ledger, _settings

    engine = session_module.AttributionEngine(_settings(**changes), _kb(), FakeStore(), FakeLLM())
    settings = engine.settings
    served = _capture_retrieve(monkeypatch, session_module)
    with pytest.raises(_StopRetrieval):
        engine.map_techniques(_ledger("PowerShell executed an encoded command."), mode="local")
    case = TechniqueEvaluationCase.from_dict(_row("tram", 1, "PowerShell ran a script"))
    lexical = LexicalTechniqueIndex.from_kb(engine.kb, fields=lexical_fields(settings))
    benchmarked = _capture_retrieve(monkeypatch, benchmark_module)
    with pytest.raises(_StopRetrieval):
        benchmark_module.run(
            [case],
            mode=settings.technique_retrieval_mode,
            settings=settings,
            kb=engine.kb,
            store=FakeStore(),
            lexical_index=lexical,
            exemplar_index=engine.exemplar_index,
        )
    fitted = _capture_retrieve(monkeypatch, fit_module)
    with pytest.raises(_StopRetrieval):
        fit_module.extract_rows(
            [case],
            settings=settings,
            kb=engine.kb,
            store=FakeStore(),
            lexical_index=lexical,
            exemplar_index=engine.exemplar_index,
        )
    shared = {key: value for key, value in served.items() if key not in CALL_SITE_KEYS}
    assert {k: v for k, v in benchmarked.items() if k not in CALL_SITE_KEYS} == shared
    assert {k: v for k, v in fitted.items() if k not in CALL_SITE_KEYS} == shared
    assert fitted["acceptance_model"] is None
    assert (
        served["lexical_index"].fields
        == benchmarked["lexical_index"].fields
        == fitted["lexical_index"].fields
    )
    assert shared["include_synthetic"] is bool(changes.get("technique_include_synthetic", False))


def test_fit_rows_carry_the_runtime_guards_and_dump_without_pickles(tmp_path):
    import numpy as np

    from apt_classifier.acceptance import FEATURE_NAMES, choose_threshold_runtime
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from scripts.fit_acceptance_model import dump_rows, extract_rows

    kb, store, lexical, _, settings = _fake_mapping_pipeline()
    cases = [
        TechniqueEvaluationCase.from_dict(
            _row("tram", 1, "PowerShell ran an encoded command and a download", ["T1059.001"])
        ),
        TechniqueEvaluationCase.from_dict(
            _row("tram", 2, "Routine maintenance used PowerShell scripts to download", ["T1105"])
        ),
        TechniqueEvaluationCase.from_dict(_row("tram", 3, "powershell download", ["T1105"])),
    ]
    fitted = extract_rows(cases, settings=settings, kb=kb, store=store, lexical_index=lexical)
    count = len(fitted.rows)
    assert count == len(fitted.labels) == len(fitted.case_ids) == len(fitted.flags)
    assert len(fitted.technique_ids) == count and fitted.expected_counts == {
        case.case_id: 1 for case in cases
    }
    flags = {
        (case_id, technique_id): row
        for case_id, technique_id, row in zip(
            fitted.case_ids, fitted.technique_ids, fitted.flags, strict=True
        )
    }
    first = flags[("tram-test-000001", "T1059.001")]
    assert set(first) == {
        "benign",
        "sparse",
        "hierarchy_parent",
        "cue_supported",
        "parent_suppressed",
    }
    assert first["cue_supported"] and not first["benign"]
    assert flags[("tram-test-000002", "T1059.001")]["benign"]
    assert flags[("tram-test-000003", "T1059.001")]["sparse"]
    # a lexical hit of its own keeps the parent of a visible child in the accepted set
    assert not flags[("tram-test-000001", "T1059")]["parent_suppressed"]
    blocked, forced = fitted.masks()
    for row, never, always in zip(fitted.flags, blocked, forced, strict=True):
        assert never == any(
            row[key] for key in ("benign", "sparse", "hierarchy_parent", "parent_suppressed")
        )
        assert always == (row["cue_supported"] and not row["hierarchy_parent"])
    choice = choose_threshold_runtime(
        [0.5] * count,
        fitted.labels,
        fitted.case_ids,
        fitted.expected_counts,
        blocked=blocked,
        forced=forced,
        technique_ids=fitted.technique_ids,
    )
    assert choice["forced_rows"] == sum(forced)
    path = dump_rows(
        tmp_path / "rows",  # no suffix: written exactly where asked
        fitted,
        datasets={case.case_id: "tram" for case in cases},
        in_sample=[0.5] * count,
        out_of_fold=None,
        folds=None,
        groups=None,
        summary={"threshold": 0.5},
    )
    assert path == tmp_path / "rows" and path.is_file()
    with np.load(path, allow_pickle=False) as payload:
        assert payload["features"].shape == (count, len(FEATURE_NAMES))
        assert list(payload["feature_names"]) == list(FEATURE_NAMES)
        assert payload["blocked"].tolist() == blocked
        assert payload["parent_suppressed"].shape == (count,)
        assert np.isnan(payload["p_out_of_fold"]).all() and (payload["folds"] == -1).all()
        assert json.loads(str(payload["summary"])) == {"threshold": 0.5}


def test_fit_script_threshold_flags_and_dump_guard(monkeypatch, tmp_path, capsys):
    import scripts.fit_acceptance_model as fit_module

    def no_work(*args, **kwargs):
        raise AssertionError("the guard must refuse before any work")

    monkeypatch.setattr(fit_module, "Settings", no_work)
    inside = fit_module.REPO_ROOT / "data" / "audits" / "never_written_by_tests.json"
    assert not inside.exists()
    with pytest.raises(SystemExit) as refused:
        fit_module.main(["--output", str(inside), "--dump-rows", str(tmp_path / "rows.npz")])
    assert refused.value.code == 2
    assert "rows.npz" in capsys.readouterr().err and not inside.exists()
    with pytest.raises(SystemExit):
        fit_module.main(["--threshold-selection", "nested"])
    assert "invalid choice" in capsys.readouterr().err


def test_threshold_selection_out_of_fold_with_the_runtime_rule():
    from apt_classifier.acceptance import AcceptanceModel, choose_threshold
    from apt_classifier.evaluation import TechniqueEvaluationCase
    from scripts.fit_acceptance_model import extract_rows, select_threshold

    kb, store, lexical, _, settings = _fake_mapping_pipeline()
    texts = [
        ("PowerShell ran an encoded command", ["T1059.001"]),
        ("the operator used powershell to download a tool", ["T1105"]),
        ("a download of the second stage followed", ["T1105"]),
        ("powershell scripts collected host details", ["T1059.001"]),
        ("they used powershell and a download cradle", ["T1059.001", "T1105"]),
        ("files were staged after a download from the server", ["T1105"]),
    ]
    cases = [
        TechniqueEvaluationCase.from_dict(
            _row("tram" if index % 2 else "expert", index, text, labels)
        )
        for index, (text, labels) in enumerate(texts * 2)
    ]

    class Embedder:
        def embed(self, values):
            # one direction per case, so no two cases are near-copies
            return [
                [1.0 if i == j else 0.0 for j in range(len(values))] for i in range(len(values))
            ]

    fitted = extract_rows(cases, settings=settings, kb=kb, store=store, lexical_index=lexical)
    model = AcceptanceModel.fit(fitted.rows, fitted.labels, attack_version="t")
    probabilities = [model.probability(row) for row in fitted.rows]
    common = {
        "cases": cases,
        "datasets": {case.case_id: case.source.rsplit(":", 1)[-1] for case in cases},
        "embedder": Embedder(),
        "attack_version": "t",
        "l2": 1.0,
        "min_precision": 0.5,
        "cv_folds": 3,
    }
    in_sample = select_threshold(fitted, probabilities, method="in_sample", **common)
    assert in_sample.operating_point == choose_threshold(
        probabilities, fitted.labels, fitted.case_ids, fitted.expected_counts, min_precision=0.5
    )
    assert in_sample.out_of_fold is None and in_sample.selection["runtime_rule_exact"]
    oof = select_threshold(fitted, probabilities, method="oof_runtime", **common)
    assert oof.out_of_fold.shape == (len(fitted.rows),)
    assert set(oof.folds) == {case.case_id for case in cases} and set(oof.folds.values()) == {
        0,
        1,
        2,
    }
    assert oof.operating_point["rule"] == "runtime"
    assert oof.selection["out_of_fold_runtime"] == oof.operating_point
    assert oof.selection["cv"]["groups"] == len(cases)
    assert oof.selection["in_sample_gap"] == pytest.approx(
        abs(oof.operating_point["threshold"] - oof.selection["in_sample_runtime"]["threshold"])
    )
    with pytest.raises(ValueError):
        select_threshold(fitted, probabilities, method="nested", **common)
