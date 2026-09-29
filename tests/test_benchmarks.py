"""Tests for benchmark script case definitions and importability.

These tests do not require a live Chroma index. They verify that the
benchmark case definitions are well-formed and the scripts can be imported
without errors.
"""

from scripts.curated_retrieval_benchmark import CASES as CURATED_CASES
from scripts.ensemble_benchmark import CLASSIFIERS as ENSEMBLE_CLASSIFIERS
from scripts.otrf_benchmark import CASES as OTRF_CASES


def test_curated_cases_have_valid_structure():
    assert len(CURATED_CASES) >= 30
    for case in CURATED_CASES:
        assert case.label
        assert case.text
        assert isinstance(case.expected_technique_ids, tuple)
        assert case.category in {
            "positive",
            "multi_label",
            "hard_negative",
            "hierarchy_parent",
            "hierarchy_subtechnique",
        }


def test_curated_cases_include_hard_negatives():
    hard_negatives = [c for c in CURATED_CASES if c.category == "hard_negative"]
    assert len(hard_negatives) >= 5
    for case in hard_negatives:
        assert case.expected_technique_ids == ()


def test_curated_cases_include_multi_label():
    multi_label = [c for c in CURATED_CASES if c.category == "multi_label"]
    assert len(multi_label) >= 4
    for case in multi_label:
        assert len(case.expected_technique_ids) >= 2


def test_curated_cases_include_hierarchy():
    parents = [c for c in CURATED_CASES if c.category == "hierarchy_parent"]
    subtechniques = [c for c in CURATED_CASES if c.category == "hierarchy_subtechnique"]
    assert len(parents) >= 4
    assert len(subtechniques) >= 2


def test_otrf_cases_have_valid_structure():
    assert len(OTRF_CASES) >= 8
    for case in OTRF_CASES:
        assert case.label
        assert case.text
        assert case.expected_technique_id


def test_ensemble_classifiers_are_instantiated():
    assert len(ENSEMBLE_CLASSIFIERS) >= 7
    for classifier in ENSEMBLE_CLASSIFIERS:
        assert hasattr(classifier, "name")
        assert hasattr(classifier, "classify")
