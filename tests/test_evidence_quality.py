"""Tests for evidence quality, corroboration, and documented specificity."""

from types import SimpleNamespace

from apt_classifier.evidence.quality import (
    EvidenceQuality,
    InformationCredibility,
    SourceReliability,
    build_corroboration_summary,
    compute_software_specificity,
    compute_technique_specificity,
    default_credibility,
    default_reliability,
)


def _group(gid, technique_ids=None, software_ids=None):
    return SimpleNamespace(
        id=gid,
        technique_ids=technique_ids or [],
        software_ids=software_ids or [],
    )


def _kb(groups=None):
    return SimpleNamespace(groups=groups or {})


def test_default_reliability_edr_is_a():
    assert default_reliability("edr") is SourceReliability.A


def test_default_reliability_unknown_is_unrated():
    assert default_reliability("unknown") is SourceReliability.UNRATED


def test_default_credibility_edr_is_one():
    assert default_credibility("edr") is InformationCredibility.ONE


def test_evidence_quality_is_rated():
    q = EvidenceQuality(
        reliability=SourceReliability.A,
        credibility=InformationCredibility.ONE,
    )
    assert q.is_rated
    assert q.numeric_reliability == 6
    assert q.numeric_credibility == 6


def test_evidence_quality_unrated_is_not_rated():
    q = EvidenceQuality()
    assert not q.is_rated
    assert q.numeric_reliability == 0
    assert q.numeric_credibility == 0


def test_corroboration_independent_sources():
    quality1 = EvidenceQuality(
        reliability=SourceReliability.A,
        credibility=InformationCredibility.ONE,
        independence_key="report-a",
    )
    quality2 = EvidenceQuality(
        reliability=SourceReliability.B,
        credibility=InformationCredibility.TWO,
        independence_key="report-b",
    )
    summary = build_corroboration_summary(
        "T1059.001",
        [("prov1", quality1), ("prov2", quality2)],
    )
    assert summary.independent_source_count == 2
    assert summary.is_corroborated
    assert summary.effective_quality > 0.0


def test_corroboration_duplicate_independence_key_not_independent():
    quality1 = EvidenceQuality(
        reliability=SourceReliability.A,
        credibility=InformationCredibility.ONE,
        independence_key="report-a",
    )
    quality2 = EvidenceQuality(
        reliability=SourceReliability.B,
        credibility=InformationCredibility.TWO,
        independence_key="report-a",  # Same key — syndicated
    )
    summary = build_corroboration_summary(
        "T1059.001",
        [("prov1", quality1), ("prov2", quality2)],
    )
    assert summary.independent_source_count == 1
    assert not summary.is_corroborated
    assert "Single independent source" in summary.corroboration_note


def test_corroboration_no_independence_keys():
    quality1 = EvidenceQuality(
        reliability=SourceReliability.A, credibility=InformationCredibility.ONE
    )
    quality2 = EvidenceQuality(
        reliability=SourceReliability.B, credibility=InformationCredibility.TWO
    )
    summary = build_corroboration_summary(
        "T1059.001",
        [("prov1", quality1), ("prov2", quality2)],
    )
    assert summary.independent_source_count == 0
    assert not summary.is_corroborated
    assert "uncorroborated" in summary.corroboration_note


def test_technique_specificity_rare_technique():
    kb = _kb(
        {
            "G1": _group("G1", technique_ids=["T1059.001"]),
            "G2": _group("G2", technique_ids=["T1003"]),
            "G3": _group("G3", technique_ids=["T1003"]),
        }
    )
    score = compute_technique_specificity("T1059.001", kb)
    assert score.signal_type == "technique"
    assert score.group_count == 1
    assert score.total_groups == 3
    assert abs(score.specificity - (1 - 1 / 3)) < 1e-9
    assert score.is_group_specific
    assert "rare" in score.note.lower() or "incomplete" in score.note.lower()


def test_technique_specificity_common_technique():
    kb = _kb(
        {
            "G1": _group("G1", technique_ids=["T1059.001", "T1003"]),
            "G2": _group("G2", technique_ids=["T1059.001", "T1003"]),
            "G3": _group("G3", technique_ids=["T1059.001"]),
        }
    )
    score = compute_technique_specificity("T1059.001", kb)
    assert score.group_count == 3
    assert score.total_groups == 3
    assert abs(score.specificity - 0.0) < 1e-9
    assert not score.is_group_specific
    assert "widely shared" in score.note.lower()


def test_technique_specificity_no_groups():
    kb = _kb({})
    score = compute_technique_specificity("T1059.001", kb)
    assert score.total_groups == 0
    assert "undefined" in score.note.lower()


def test_software_specificity():
    kb = _kb(
        {
            "G1": _group("G1", software_ids=["S0001"]),
            "G2": _group("G2", software_ids=["S0002"]),
            "G3": _group("G3", software_ids=["S0002"]),
        }
    )
    score = compute_software_specificity("S0001", kb)
    assert score.signal_type == "software"
    assert score.group_count == 1
    assert score.total_groups == 3
    assert score.is_group_specific


def test_corroboration_effective_quality_weights_by_best_per_key():
    quality1 = EvidenceQuality(
        reliability=SourceReliability.A,
        credibility=InformationCredibility.ONE,
        independence_key="report-a",
    )
    quality2 = EvidenceQuality(
        reliability=SourceReliability.C,
        credibility=InformationCredibility.THREE,
        independence_key="report-a",  # Same key, lower quality
    )
    quality3 = EvidenceQuality(
        reliability=SourceReliability.B,
        credibility=InformationCredibility.TWO,
        independence_key="report-b",
    )
    summary = build_corroboration_summary(
        "T1059.001",
        [("prov1", quality1), ("prov2", quality2), ("prov3", quality3)],
    )
    assert summary.independent_source_count == 2
    assert summary.is_corroborated
    # The best quality for report-a is quality1 (A1=12), report-b is quality3 (B2=10)
    # effective_quality = ((12/12) + (10/12)) / 2
    expected = ((12 / 12) + (10 / 12)) / 2
    assert abs(summary.effective_quality - expected) < 1e-9
