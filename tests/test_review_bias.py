from apt_classifier.review.bias import measure_fame_bias
from apt_classifier.review.blinding import blind_payload, create_blinding_map
from apt_classifier.review.context import ReviewPayload


def test_fame_bias_reports_promotion_and_correlation():
    report = measure_fame_bias(
        {"G1": 3, "G2": 2, "G3": 1},
        {"G1": 1, "G2": 2, "G3": 3},
        {"G1": 10.0, "G2": 5.0, "G3": 1.0},
    )
    assert report.promotion_when_named["G1"] == 2
    assert report.promotion_when_named["G3"] == -2
    assert report.prominence_correlation > 0.9


def test_blinding_removes_identity_strings(synthetic_kb):
    payload = ReviewPayload(
        evidence={"observations": [{"text": "Alpha used RedKit during Operation One"}]},
        candidates=(
            {
                "candidate_id": "G1",
                "profile": {"name": "Alpha", "aliases": ["A-Team"]},
                "known_techniques": ["T1", "T2"],
                "known_software": ["S1"],
                "campaigns": ["C1"],
                "prominence": 10,
            },
        ),
        classifier_results=(
            {"classifier": "test", "applicable": True, "entries": [{"group_id": "G1"}]},
        ),
        fusion={"entries": [{"group_id": "G1", "rank": 1}]},
        reference={"techniques": {"T1": {"name": "One"}}},
    )
    mapping = create_blinding_map(["G1"])
    blinded = blind_payload(payload, synthetic_kb, mapping).to_dict()
    text = str(blinded)
    for forbidden in ("Alpha", "A-Team", "RedKit", "Operation One", "G1", "S1", "C1"):
        assert forbidden.lower() not in text.lower()
    assert "Candidate A" in text
    assert "T1" in text
