import json
from types import SimpleNamespace

import pytest

from apt_classifier.evaluation import (
    AttackIdMap,
    AttributionEvaluationCase,
    AttributionPrediction,
    TechniqueEvaluationCase,
    TechniquePrediction,
    bootstrap_interval,
    duplicate_text_groups,
    evaluate_attribution_predictions,
    evaluate_technique_predictions,
    grouped_split,
    load_jsonl_cases,
    pipeline_fingerprint,
    time_ordered_split,
)


def test_technique_case_parses_serialized_labels():
    case = TechniqueEvaluationCase.from_dict(
        {"id": "one", "document_id": "doc", "text1": "PowerShell ran.", "labels": "['T1059.001']"}
    )
    assert case.expected_technique_ids == ("T1059.001",)


def test_attack_id_map_follows_revoked_by_relationship():
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--old",
                "revoked": True,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T1000"}],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--new",
                "external_references": [{"source_name": "mitre-attack", "external_id": "T2000"}],
            },
            {
                "type": "relationship",
                "relationship_type": "revoked-by",
                "source_ref": "attack-pattern--old",
                "target_ref": "attack-pattern--new",
            },
        ]
    }
    mapping = AttackIdMap.from_bundle(bundle)
    remap = mapping.remap(("T1000", "T9999"), {"T2000"})
    assert remap.mapped == ("T2000",)
    assert remap.replacements == {"T1000": "T2000"}
    assert remap.unscorable == ("T9999",)


def test_grouped_split_keeps_documents_together():
    cases = [
        TechniqueEvaluationCase(f"case-{index}", f"doc-{index // 2}", str(index), ("T1",))
        for index in range(20)
    ]
    splits = grouped_split(cases)
    document_splits = {
        case.document_id: split for split, values in splits.items() for case in values
    }
    assert len(document_splits) == 10
    assert all(
        document_splits[case.document_id] == split
        for split, values in splits.items()
        for case in values
    )


def test_time_split_orders_whole_documents():
    cases = [
        AttributionEvaluationCase("early", "early-doc", "G1", published_at="2020-01-01"),
        AttributionEvaluationCase("middle", "middle-doc", "G1", published_at="2021-01-01"),
        AttributionEvaluationCase("late", "late-doc", None, published_at="2022-01-01"),
    ]
    splits = time_ordered_split(cases, train_fraction=0.34, dev_fraction=0.33)
    assert splits["train"][0].case_id == "early"
    assert splits["dev"][0].case_id == "middle"
    assert splits["test"][0].case_id == "late"


def test_duplicate_detection_normalizes_text():
    cases = [
        TechniqueEvaluationCase("one", "doc-one", "PowerShell -ENC", ("T1",)),
        TechniqueEvaluationCase("two", "doc-two", "powershell enc", ("T1",)),
    ]
    groups = duplicate_text_groups(cases)
    assert tuple(groups.values()) == (("one", "two"),)


def test_technique_metrics_include_negative_and_hierarchy_scores():
    cases = [
        TechniqueEvaluationCase("positive", "doc-one", "x", ("T1.001",)),
        TechniqueEvaluationCase("negative", "doc-two", "y", ()),
    ]
    predictions = [
        TechniquePrediction.from_ranked("positive", ("T1", "T2"), accepted=("T1",)),
        TechniquePrediction.from_ranked("negative", (), accepted=()),
    ]
    kb = SimpleNamespace(
        techniques={
            "T1": SimpleNamespace(parent_id=None),
            "T1.001": SimpleNamespace(parent_id="T1"),
        }
    )
    metrics = evaluate_technique_predictions(cases, predictions, kb=kb)
    assert metrics["micro_f1"] == 0.0
    assert metrics["parent_aware_f1"] == 1.0
    assert metrics["zero_label_false_positive_rate"] == 0.0
    assert metrics["abstention_recall"] == 1.0


def test_attribution_metrics_measure_unknown_rejection():
    cases = [
        AttributionEvaluationCase("known", "doc-one", "G1"),
        AttributionEvaluationCase("unknown", "doc-two", None),
    ]
    predictions = [
        AttributionPrediction("known", ("G1", "G2"), True),
        AttributionPrediction("unknown", ("G2",), False),
    ]
    metrics = evaluate_attribution_predictions(cases, predictions)
    assert metrics["top1"] == 1.0
    assert metrics["unknown_recall"] == 1.0
    assert metrics["false_named_attribution_rate"] == 0.0
    assert metrics["named_risk"] == 0.0


def test_bootstrap_and_fingerprint_are_deterministic():
    first = bootstrap_interval(
        [0.0, 1.0, 1.0], lambda values: sum(values) / len(values), samples=50
    )
    second = bootstrap_interval(
        [0.0, 1.0, 1.0], lambda values: sum(values) / len(values), samples=50
    )
    assert first == second
    assert pipeline_fingerprint({"b": 2, "a": 1}) == pipeline_fingerprint({"a": 1, "b": 2})


def test_load_jsonl_rejects_non_object(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_jsonl_cases(path, TechniqueEvaluationCase)
