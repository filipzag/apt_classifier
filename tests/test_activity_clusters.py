"""Tests for opt-in local longitudinal activity clustering."""

from types import SimpleNamespace

from apt_classifier.activity_clusters import (
    ActivityClusterStore,
    CaseSignature,
    build_case_signature,
    compute_similarity,
)


def _state(techniques=None, software=None, observations=None):
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            observed_techniques=techniques or {},
        ),
        context=SimpleNamespace(
            software_ids=software or set(),
        ),
        ledger=SimpleNamespace(
            observations=observations or [],
        ),
    )


def _observation(obs_id, metadata=None):
    return SimpleNamespace(
        id=obs_id,
        text="activity",
        metadata=metadata or {},
    )


def test_build_case_signature_excludes_raw_logs():
    obs = [
        _observation(
            "o1",
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "ioc": "evil.com",
                "victim": "Target Inc",
                "sector": "finance",
                "raw_log": "SECRET_KEY=abc123\nmalicious raw log line",
            },
        ),
    ]
    state = _state(
        techniques={"T1059.001": 0.9, "T1003.001": 0.8},
        software={"S0001"},
        observations=obs,
    )
    sig = build_case_signature("case-1", state)
    assert sig.case_id == "case-1"
    assert sig.techniques == ("T1003.001", "T1059.001")
    assert sig.software == ("S0001",)
    assert "evil.com" in sig.infrastructure
    assert "Target Inc" in sig.victimology
    assert "finance" in sig.victimology
    assert sig.temporal_features["first_seen"] == "2024-01-01T10:00:00Z"
    # Raw logs must not be in the signature
    assert "raw_log" not in str(sig.to_dict())
    assert "SECRET_KEY" not in str(sig.to_dict())


def test_case_signature_fingerprint_is_stable():
    state = _state(techniques={"T1": 0.9}, software={"S1"})
    sig1 = build_case_signature("case-1", state)
    sig2 = build_case_signature("case-1", state)
    # Fingerprints should match for same data (ignoring created_at)
    sig1_fixed = CaseSignature(
        case_id=sig1.case_id,
        techniques=sig1.techniques,
        software=sig1.software,
        infrastructure=sig1.infrastructure,
        victimology=sig1.victimology,
        temporal_features=sig1.temporal_features,
        evidence_quality_summary=sig1.evidence_quality_summary,
    )
    sig2_fixed = CaseSignature(
        case_id=sig2.case_id,
        techniques=sig2.techniques,
        software=sig2.software,
        infrastructure=sig2.infrastructure,
        victimology=sig2.victimology,
        temporal_features=sig2.temporal_features,
        evidence_quality_summary=sig2.evidence_quality_summary,
    )
    assert sig1_fixed.fingerprint() == sig2_fixed.fingerprint()


def test_compute_similarity_identical_signatures():
    sig_a = CaseSignature(
        case_id="a",
        techniques=("T1", "T2"),
        software=("S1",),
        infrastructure=("evil.com",),
        victimology=("Target",),
        temporal_features={"first_seen": "2024-01-01", "last_seen": "2024-01-02"},
        evidence_quality_summary={},
    )
    sig_b = CaseSignature(
        case_id="b",
        techniques=("T1", "T2"),
        software=("S1",),
        infrastructure=("evil.com",),
        victimology=("Target",),
        temporal_features={"first_seen": "2024-01-01", "last_seen": "2024-01-02"},
        evidence_quality_summary={},
    )
    suggestion = compute_similarity(sig_a, sig_b)
    assert suggestion.overall_similarity > 0.9
    assert suggestion.technique_similarity == 1.0
    assert suggestion.software_similarity == 1.0


def test_compute_similarity_no_overlap():
    sig_a = CaseSignature(
        case_id="a",
        techniques=("T1",),
        software=("S1",),
        infrastructure=(),
        victimology=(),
        temporal_features={},
        evidence_quality_summary={},
    )
    sig_b = CaseSignature(
        case_id="b",
        techniques=("T2",),
        software=("S2",),
        infrastructure=(),
        victimology=(),
        temporal_features={},
        evidence_quality_summary={},
    )
    suggestion = compute_similarity(sig_a, sig_b)
    assert suggestion.overall_similarity == 0.0
    assert suggestion.technique_similarity == 0.0


def test_activity_store_store_and_retrieve(tmp_path):
    store = ActivityClusterStore(tmp_path / "clusters.db")
    sig = CaseSignature(
        case_id="case-1",
        techniques=("T1", "T2"),
        software=("S1",),
        infrastructure=(),
        victimology=(),
        temporal_features={},
        evidence_quality_summary={},
    )
    store.store_signature(sig)
    retrieved = store.get_signature("case-1")
    assert retrieved is not None
    assert retrieved.case_id == "case-1"
    assert retrieved.techniques == ("T1", "T2")


def test_activity_store_find_related(tmp_path):
    store = ActivityClusterStore(tmp_path / "clusters.db")
    sig_a = CaseSignature(
        case_id="case-a",
        techniques=("T1", "T2"),
        software=("S1",),
        infrastructure=("evil.com",),
        victimology=("Target",),
        temporal_features={"first_seen": "2024-01-01", "last_seen": "2024-01-02"},
        evidence_quality_summary={},
    )
    sig_b = CaseSignature(
        case_id="case-b",
        techniques=("T1", "T3"),
        software=("S1",),
        infrastructure=("evil.com",),
        victimology=("Target",),
        temporal_features={"first_seen": "2024-01-03", "last_seen": "2024-01-04"},
        evidence_quality_summary={},
    )
    store.store_signature(sig_a)
    store.store_signature(sig_b)
    # Find related for a new case similar to sig_a
    sig_new = CaseSignature(
        case_id="case-new",
        techniques=("T1", "T2"),
        software=("S1",),
        infrastructure=("evil.com",),
        victimology=("Target",),
        temporal_features={"first_seen": "2024-01-01", "last_seen": "2024-01-02"},
        evidence_quality_summary={},
    )
    related = store.find_related(sig_new, min_similarity=0.3)
    assert len(related) >= 1
    # case-a should be more similar than case-b
    assert related[0].suggested_case_id == "case-a"
    assert related[0].overall_similarity > related[1].overall_similarity


def test_activity_store_add_and_get_links(tmp_path):
    store = ActivityClusterStore(tmp_path / "clusters.db")
    store.add_link("case-a", "case-b", "related", "Same TTPs")
    links = store.get_links("case-a")
    assert len(links) == 1
    assert links[0]["case_id_a"] == "case-a"
    assert links[0]["case_id_b"] == "case-b"
    assert links[0]["link_type"] == "related"


def test_activity_store_delete_signature(tmp_path):
    store = ActivityClusterStore(tmp_path / "clusters.db")
    sig = CaseSignature(
        case_id="case-1",
        techniques=("T1",),
        software=(),
        infrastructure=(),
        victimology=(),
        temporal_features={},
        evidence_quality_summary={},
    )
    store.store_signature(sig)
    store.add_link("case-1", "case-2", "related")
    store.delete_signature("case-1")
    assert store.get_signature("case-1") is None
    assert store.get_links("case-1") == []


def test_activity_store_export_all(tmp_path):
    store = ActivityClusterStore(tmp_path / "clusters.db")
    sig = CaseSignature(
        case_id="case-1",
        techniques=("T1",),
        software=("S1",),
        infrastructure=(),
        victimology=(),
        temporal_features={},
        evidence_quality_summary={},
    )
    store.store_signature(sig)
    store.add_link("case-1", "case-2", "related")
    exported = store.export_all()
    assert len(exported["signatures"]) == 1
    assert exported["signatures"][0]["case_id"] == "case-1"
    assert len(exported["links"]) == 1
