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


HIERARCHY_KB = SimpleNamespace(
    techniques={
        "T1566": SimpleNamespace(parent_id=None),
        "T1566.001": SimpleNamespace(parent_id="T1566"),
        "T1566.002": SimpleNamespace(parent_id="T1566"),
        "T1059": SimpleNamespace(parent_id=None),
        "T1059.001": SimpleNamespace(parent_id="T1059"),
        "T1105": SimpleNamespace(parent_id=None),
    }
)


def test_parent_aware_metrics_use_a_maximum_matching():
    """Worked example: a first-fit pairing could spend T1566.001 on the parent T1566 and leave
    T1566.002 (which can only pair with T1566) unmatched; the maximum matching pairs both. In
    one process the old pairing failed this only under some hash seeds, so the regression guard
    for it is the subprocess test below, which runs the same case under several seeds."""
    cases = [TechniqueEvaluationCase("c", "doc", "x", ("T1566", "T1566.001"))]
    predictions = [TechniquePrediction.from_ranked("c", ("T1566.001", "T1566.002"), accepted=None)]
    metrics = evaluate_technique_predictions(cases, predictions, kb=HIERARCHY_KB)
    assert metrics["micro_precision"] == 0.5
    assert metrics["parent_aware_precision"] == 1.0
    assert metrics["parent_aware_recall"] == 1.0


def test_hierarchy_matching_is_maximum_on_every_small_case():
    """Brute force over every subset pair of a small hierarchy: the matching size equals the
    best one-to-one pairing found by trying every permutation."""
    from itertools import combinations, permutations

    from apt_classifier.evaluation import _hierarchy_compatible, _hierarchy_matching

    ids = sorted(HIERARCHY_KB.techniques)
    subsets = [set(combo) for size in range(4) for combo in combinations(ids, size)]

    def brute_force(accepted: set[str], expected: set[str]) -> int:
        left, right = sorted(accepted), sorted(expected)
        if len(left) > len(right):
            left, right = right, left
            swapped = True
        else:
            swapped = False
        best = 0
        for chosen in permutations(right, len(left)):
            pairs = zip(left, chosen, strict=True)
            best = max(
                best,
                sum(
                    _hierarchy_compatible(b, a, HIERARCHY_KB)
                    if swapped
                    else _hierarchy_compatible(a, b, HIERARCHY_KB)
                    for a, b in pairs
                ),
            )
        return best

    for accepted in subsets:
        for expected in subsets:
            assert _hierarchy_matching(accepted, expected, HIERARCHY_KB) == brute_force(
                accepted, expected
            ), (accepted, expected)


def test_parent_aware_metrics_do_not_depend_on_the_hash_seed():
    """Set iteration order of strings follows PYTHONHASHSEED; the metric must not."""
    import os
    import subprocess
    import sys

    script = (
        "from types import SimpleNamespace as N\n"
        "from apt_classifier.evaluation import TechniqueEvaluationCase as C, "
        "TechniquePrediction as P, evaluate_technique_predictions as e\n"
        "kb = N(techniques={'T1566': N(parent_id=None), 'T1566.001': N(parent_id='T1566'),"
        " 'T1566.002': N(parent_id='T1566')})\n"
        "m = e([C('c', 'd', 'x', ('T1566', 'T1566.001'))],"
        " [P.from_ranked('c', ('T1566.001', 'T1566.002'))], kb=kb)\n"
        "print(m['parent_aware_f1'])\n"
    )
    values = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        values.add(completed.stdout.strip())
    assert values == {"1.0"}


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
