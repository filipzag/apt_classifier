"""Tests for open-set attribution stages and calibrated rejection."""

from types import SimpleNamespace

from apt_classifier.attribution import (
    AttributionStage,
    MaturityThresholds,
    assess_attribution,
)
from apt_classifier.fusion import FusionEntry, FusionResult


def _fusion_entry(group_id, score, rank=1, rank_std=2.0, classifier_ranks=None):
    return FusionEntry(
        group_id=group_id,
        score=score,
        relative_score=score,
        rank=rank,
        classifier_ranks=classifier_ranks or {"nb": 1, "jaccard": 2},
        rank_min=1,
        rank_max=3,
        rank_std=rank_std,
        contested=False,
    )


def _fusion(entries):
    return FusionResult(
        entries=tuple(entries),
        active_classifiers=("nb", "jaccard"),
        correlations={},
        k=60,
        effective_weights={},
    )


def _context(techniques=None, narrative="word " * 50):
    return SimpleNamespace(
        observed_techniques=techniques or {},
        narrative=narrative,
    )


def _kb(groups=None, techniques=None, group_techniques=None):
    return SimpleNamespace(
        version="19.2",
        groups=groups or {},
        techniques=techniques or {},
        group_techniques=group_techniques or {},
    )


def test_insufficient_evidence_when_no_techniques():
    kb = _kb()
    fusion = _fusion([_fusion_entry("G1", 80.0)])
    context = _context(techniques={})
    decision = assess_attribution(kb, fusion, context)
    assert decision.stage is AttributionStage.INSUFFICIENT_EVIDENCE
    assert decision.cluster_id is None
    assert "Insufficient" in decision.rationale


def test_insufficient_evidence_when_no_fusion_entries():
    kb = _kb()
    fusion = _fusion([])
    context = _context(techniques={"T1": 0.9})
    decision = assess_attribution(kb, fusion, context)
    assert decision.stage is AttributionStage.INSUFFICIENT_EVIDENCE


def test_unresolved_activity_cluster_when_maturity_low():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    # Low maturity: only 1 family, high rank_std
    fusion = _fusion(
        [
            _fusion_entry("G1", 50.0, rank_std=8.0, classifier_ranks={"nb": 1}),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision = assess_attribution(kb, fusion, context)
    assert decision.stage is AttributionStage.UNRESOLVED_ACTIVITY_CLUSTER
    assert decision.cluster_id is not None
    assert decision.cluster_id.startswith("CL-UNK-")
    assert decision.leading_group_id == "G1"


def test_known_group_shortlist_when_maturity_moderate():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    # Moderate maturity but below hypothesis threshold
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.9,
        hypothesis_margin=0.5,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 70.0, rank_std=2.0, classifier_ranks={"nb": 1, "jaccard": 2}),
            _fusion_entry(
                "G2", 60.0, rank=2, rank_std=2.0, classifier_ranks={"nb": 2, "jaccard": 1}
            ),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision = assess_attribution(kb, fusion, context, thresholds=thresholds)
    assert decision.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert decision.leading_group_id == "G1"


def test_known_group_hypothesis_when_maturity_high():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 90.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
            _fusion_entry(
                "G2", 50.0, rank=2, rank_std=2.0, classifier_ranks={"nb": 2, "jaccard": 2}
            ),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision = assess_attribution(kb, fusion, context, thresholds=thresholds)
    assert decision.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    assert decision.leading_group_id == "G1"
    assert decision.leading_group_name == "Alpha"
    assert "hypothesis" in decision.rationale.lower()
    assert not decision.is_calibrated


def test_no_numeric_probability_without_calibration_artifact():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 90.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision = assess_attribution(kb, fusion, context, thresholds=thresholds)
    assert decision.calibrated_probability is None
    assert "qualitative" in decision.calibration_note.lower()


def test_calibration_artifact_validated():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 90.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    # Valid artifact
    artifact = {
        "attack_version": "19.2",
        "pipeline_fingerprint": "abc123",
        "calibration_hash": "def456",
        "sample_count": 100,
        "calibration_curve": {"0.5": 0.6, "0.9": 0.85},
    }
    decision = assess_attribution(
        kb, fusion, context, thresholds=thresholds, calibration_artifact=artifact
    )
    assert decision.is_calibrated
    assert decision.calibrated_probability is not None


def test_calibration_artifact_version_mismatch_rejected():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 90.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    artifact = {
        "attack_version": "15.1",  # Mismatch
        "pipeline_fingerprint": "abc123",
        "calibration_hash": "def456",
        "sample_count": 100,
    }
    decision = assess_attribution(
        kb, fusion, context, thresholds=thresholds, calibration_artifact=artifact
    )
    assert not decision.is_calibrated
    assert "mismatch" in decision.calibration_note.lower()


def test_calibration_artifact_insufficient_samples_rejected():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 90.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
        ]
    )
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    artifact = {
        "attack_version": "19.2",
        "pipeline_fingerprint": "abc123",
        "calibration_hash": "def456",
        "sample_count": 10,  # Too few
    }
    decision = assess_attribution(
        kb, fusion, context, thresholds=thresholds, calibration_artifact=artifact
    )
    assert not decision.is_calibrated
    assert "insufficient" in decision.calibration_note.lower()


def test_cluster_id_is_stable():
    kb = _kb()
    fusion = _fusion([_fusion_entry("G1", 50.0, rank_std=8.0, classifier_ranks={"nb": 1})])
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision1 = assess_attribution(kb, fusion, context)
    decision2 = assess_attribution(kb, fusion, context)
    assert decision1.cluster_id == decision2.cluster_id


def test_decision_to_dict_serializable():
    kb = _kb(
        groups={"G1": SimpleNamespace(id="G1", name="Alpha")},
        group_techniques={"G1": ["T1", "T2"]},
    )
    fusion = _fusion([_fusion_entry("G1", 50.0, rank_std=8.0, classifier_ranks={"nb": 1})])
    context = _context(techniques={"T1": 0.9, "T2": 0.8})
    decision = assess_attribution(kb, fusion, context)
    d = decision.to_dict()
    assert "stage" in d
    assert "maturity_score" in d
    assert "maturity_factors" in d
    assert isinstance(d["stage"], str)


def test_decision_carries_documented_overlap_of_the_leader():
    from types import SimpleNamespace

    from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software
    from apt_classifier.attribution import assess_attribution
    from apt_classifier.fusion import FusionEntry, FusionResult

    kb = AttackKnowledgeBase(
        version="t",
        groups={
            "G0008": Group(
                id="G0008",
                name="Carbanak",
                description="shares tooling with FIN7",
                software_ids=["S1"],
                technique_ids=["T1", "T2"],
            ),
            "G0046": Group(
                id="G0046", name="FIN7", description="", software_ids=["S1"], technique_ids=["T1"]
            ),
        },
        software={"S1": Software(id="S1", name="Carbanak malware")},
    )
    kb.recompute_statistics()
    entries = (
        FusionEntry(
            "G0008", 1.0, 100.0, 1, {"idf_tfidf_cosine": 1, "weighted_jaccard": 1}, 1, 1, 0.0, False
        ),
        FusionEntry(
            "G0046", 0.5, 50.0, 2, {"idf_tfidf_cosine": 2, "weighted_jaccard": 2}, 2, 2, 0.0, False
        ),
    )
    fusion = FusionResult(entries, ("idf_tfidf_cosine", "weighted_jaccard"), {}, 20)
    context = SimpleNamespace(observed_techniques={"T1": 0.9, "T2": 0.8}, narrative="a " * 40)
    decision = assess_attribution(kb, fusion, context)
    assert "G0046" in decision.related_leading_groups
    assert decision.to_dict()["related_leading_groups"]["G0046"]


def _separation_kb():
    return _kb(
        groups={
            "G1": SimpleNamespace(id="G1", name="Alpha"),
            "G2": SimpleNamespace(id="G2", name="Beta"),
        },
        techniques={
            "T1": SimpleNamespace(parent_id=None),
            "T2": SimpleNamespace(parent_id=None),
            "T3": SimpleNamespace(parent_id=None),
        },
        group_techniques={"G1": ["T1", "T2", "T3"], "G2": ["T2", "T3"]},
    )


def _two_candidates(leader_score=70.0, runner_score=60.0):
    return _fusion(
        [
            _fusion_entry(
                "G1", leader_score, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}
            ),
            _fusion_entry(
                "G2", runner_score, rank=2, rank_std=1.0, classifier_ranks={"nb": 2, "jaccard": 2}
            ),
        ]
    )


def test_separation_gate_names_high_separation_and_blocks_shared_evidence():
    kb = _separation_kb()
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.99,  # maturity alone would never name
        hypothesis_margin=0.05,
        gate_mode="separation",
        hypothesis_separation=0.5,
        hypothesis_min_discriminative=0.2,
    )
    # T1 is exclusive to Alpha and documented for one of two groups (specificity 0.5)
    exclusive = assess_attribution(
        kb, _two_candidates(), _context(techniques={"T1": 1.0, "T2": 1.0}), thresholds=thresholds
    )
    assert exclusive.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    assert exclusive.gate_mode == "separation"
    assert exclusive.separation >= 0.5
    assert exclusive.runner_up_group_id == "G2"
    assert "separation" in exclusive.rationale
    assert exclusive.to_dict()["separation_factors"]["leader"]["technique"] > 0
    # only shared evidence: separation 0 -> shortlist, even though margin passes
    shared = assess_attribution(
        kb, _two_candidates(), _context(techniques={"T2": 1.0, "T3": 1.0}), thresholds=thresholds
    )
    assert shared.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert shared.separation == 0.0
    assert "separation" in shared.rationale and "discriminative evidence" in shared.rationale
    # in separation mode the RRF margin is informational: a near-tie RRF pair with exclusive
    # evidence is still named
    tie = assess_attribution(
        kb,
        _two_candidates(70.0, 69.0),
        _context(techniques={"T1": 1.0, "T2": 1.0}),
        thresholds=thresholds,
    )
    assert tie.margin < 0.05 and tie.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def test_maturity_gate_mode_ignores_separation_and_both_requires_both():
    kb = _separation_kb()
    base = dict(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.3,
        hypothesis_margin=0.01,
    )
    context = _context(techniques={"T2": 1.0, "T3": 1.0})  # shared evidence only
    maturity_only = assess_attribution(
        kb, _two_candidates(90.0, 50.0), context, thresholds=MaturityThresholds(**base)
    )
    assert maturity_only.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    assert maturity_only.gate_mode == "maturity" and maturity_only.separation == 0.0
    both = assess_attribution(
        kb,
        _two_candidates(90.0, 50.0),
        context,
        thresholds=MaturityThresholds(**base, gate_mode="both", hypothesis_separation=0.5),
    )
    assert both.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "separation 0.00 < 0.50" in both.rationale


def test_separation_gate_never_names_a_group_promoted_above_the_scored_leader():
    from apt_classifier.separation import discriminate

    kb = _separation_kb()
    thresholds = MaturityThresholds(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.99,
        hypothesis_margin=0.0,
        gate_mode="separation",
        hypothesis_separation=0.5,
        hypothesis_min_discriminative=0.2,
    )
    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    fusion = _two_candidates()
    scored = discriminate(kb, fusion, context)  # leader G1, exclusive T1 -> clears the bar
    assert scored.leader_id == "G1" and scored.separation >= 0.5
    # a re-ranking that put G2 first while the separation still describes G1
    swapped = FusionResult(
        entries=tuple(
            FusionEntry(
                group_id=entry.group_id,
                score=entry.score,
                relative_score=entry.relative_score,
                rank=2 if entry.group_id == "G1" else 1,
                classifier_ranks=entry.classifier_ranks,
                rank_min=entry.rank_min,
                rank_max=entry.rank_max,
                rank_std=entry.rank_std,
                contested=entry.contested,
            )
            for entry in sorted(fusion.entries, key=lambda e: e.group_id != "G2")
        ),
        active_classifiers=fusion.active_classifiers,
        correlations=fusion.correlations,
        k=fusion.k,
        effective_weights=fusion.effective_weights,
    )
    decision = assess_attribution(
        kb, swapped, context, thresholds=thresholds, discrimination=scored
    )
    assert decision.leading_group_id == "G2"
    assert decision.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "re-ranking placed G2 above G1" in decision.rationale
    # the consistent case still names
    consistent = assess_attribution(
        kb, fusion, context, thresholds=thresholds, discrimination=scored
    )
    assert consistent.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def _cluster_thresholds(**overrides):
    base = dict(
        min_techniques=2,
        min_independent_evidence_families=2,
        min_maturity_score=0.3,
        hypothesis_maturity_score=0.99,
        hypothesis_margin=0.0,
        gate_mode="separation",
        hypothesis_separation=0.5,
        hypothesis_min_discriminative=0.2,
        cluster_naming=True,
        cluster_min_probability=0.8,
        cluster_min_discriminative=0.2,
    )
    base.update(overrides)
    return MaturityThresholds(**base)


def _candidate(**overrides):
    base = {
        "members": ["G1", "G2"],
        "reason": "ATT&CK: Alpha may be linked to Beta",
        "link": "same_activity",
        "nested": False,
        "containment": 0.5,
        "pair_separation": 0.1,
        "separation": 0.7,
        "discriminative": 1.0,
        "runner_up_id": None,
        "probability": 0.9,
    }
    base.update(overrides)
    return base


def test_documented_cluster_is_named_when_the_pair_clears_its_bars():
    kb = _separation_kb()
    shared = _context(techniques={"T2": 1.0, "T3": 1.0})  # nothing separates the two groups
    combiner = {"leader_probability": 0.5, "cluster": _candidate()}
    off = assess_attribution(
        kb,
        _two_candidates(),
        shared,
        thresholds=_cluster_thresholds(cluster_naming=False),
        combiner=combiner,
    )
    assert off.stage is AttributionStage.KNOWN_GROUP_SHORTLIST and off.named_cluster == ()
    on = assess_attribution(
        kb, _two_candidates(), shared, thresholds=_cluster_thresholds(), combiner=combiner
    )
    assert on.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    assert on.named_cluster == ("G1", "G2") and on.leading_group_id == "G1"
    assert on.cluster_probability == 0.9 and "Documented-overlap cluster" in on.rationale
    assert on.to_dict()["named_cluster"] == ["G1", "G2"]
    # each bar on its own keeps the shortlist
    for bad in (
        {"probability": 0.79},
        {"probability": None},
        {"separation": 0.49},
        {"discriminative": 0.1},
        {"members": ["G2", "G1"]},  # the candidate must belong to the leader
    ):
        held = assess_attribution(
            kb,
            _two_candidates(),
            shared,
            thresholds=_cluster_thresholds(),
            combiner={"leader_probability": 0.5, "cluster": _candidate(**bad)},
        )
        assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST, bad
    # nested scope ignores a non-nesting pair; the maturity gate never names clusters
    for thresholds in (
        _cluster_thresholds(cluster_scope="nested"),
        _cluster_thresholds(gate_mode="maturity"),
    ):
        decision = assess_attribution(
            kb, _two_candidates(), shared, thresholds=thresholds, combiner=combiner
        )
        assert decision.stage is not AttributionStage.KNOWN_CLUSTER_HYPOTHESIS


def test_documented_scope_never_names_a_pair_that_only_shares_rare_malware():
    kb = _separation_kb()
    shared = _context(techniques={"T2": 1.0, "T3": 1.0})
    malware_only = _candidate(
        reason="shared rarely used software KillDisk (S0607)", link="shared_software"
    )
    combiner = {"leader_probability": 0.5, "cluster": malware_only}
    default = assess_attribution(
        kb, _two_candidates(), shared, thresholds=_cluster_thresholds(), combiner=combiner
    )
    assert default.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert default.named_cluster == ()
    # the pre-registered full scope names it; nesting profiles qualify under the default scope
    full = assess_attribution(
        kb,
        _two_candidates(),
        shared,
        thresholds=_cluster_thresholds(cluster_scope="full"),
        combiner=combiner,
    )
    assert full.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    nested = assess_attribution(
        kb,
        _two_candidates(),
        shared,
        thresholds=_cluster_thresholds(),
        combiner={
            "leader_probability": 0.5,
            "cluster": _candidate(link="nesting", nested=True, containment=1.0),
        },
    )
    assert nested.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    # a description that calls the pair distinct (APT30 / Naikon nest fully) never names it,
    # and neither does a mere mention
    for candidate in (
        _candidate(link="description", nested=True, containment=1.0),
        _candidate(link="description"),
    ):
        distinct = assess_attribution(
            kb,
            _two_candidates(),
            shared,
            thresholds=_cluster_thresholds(),
            combiner={"leader_probability": 0.5, "cluster": candidate},
        )
        assert distinct.stage is not AttributionStage.KNOWN_CLUSTER_HYPOTHESIS, candidate
    # a candidate recorded before the link field existed counts only when it nests
    legacy = {key: value for key, value in _candidate().items() if key != "link"}
    held = assess_attribution(
        kb,
        _two_candidates(),
        shared,
        thresholds=_cluster_thresholds(),
        combiner={"leader_probability": 0.5, "cluster": legacy},
    )
    assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST


def test_no_cluster_when_the_leader_consistency_guard_fires():
    from apt_classifier.separation import discriminate

    kb = _separation_kb()
    context = _context(techniques={"T2": 1.0, "T3": 1.0})
    swapped = _fusion(
        [
            _fusion_entry("G2", 70.0, rank_std=1.0, classifier_ranks={"nb": 1, "jaccard": 1}),
            _fusion_entry(
                "G1", 60.0, rank=2, rank_std=1.0, classifier_ranks={"nb": 2, "jaccard": 2}
            ),
        ]
    )
    scored_for_g1 = discriminate(kb, _two_candidates(), context)
    decision = assess_attribution(
        kb,
        swapped,
        context,
        thresholds=_cluster_thresholds(),
        discrimination=scored_for_g1,
        combiner={"cluster": _candidate(members=["G2", "G1"])},
    )
    assert decision.stage is AttributionStage.KNOWN_GROUP_SHORTLIST


def test_a_single_name_with_a_nested_twin_becomes_the_cluster_or_is_held_back():
    kb = _separation_kb()
    exclusive = _context(techniques={"T1": 1.0, "T2": 1.0})  # Alpha clears alone on T1
    thresholds = _cluster_thresholds()
    plain = assess_attribution(kb, _two_candidates(), exclusive, thresholds=thresholds)
    assert plain.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    nested = _candidate(nested=True, containment=1.0)
    upgraded = assess_attribution(
        kb, _two_candidates(), exclusive, thresholds=thresholds, combiner={"cluster": nested}
    )
    assert upgraded.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    failing = _candidate(nested=True, containment=1.0, probability=0.5)
    held = assess_attribution(
        kb, _two_candidates(), exclusive, thresholds=thresholds, combiner={"cluster": failing}
    )
    assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "documented nested twin" in held.rationale
    # a non-nesting related group does not touch a single name
    single = assess_attribution(
        kb, _two_candidates(), exclusive, thresholds=thresholds, combiner={"cluster": _candidate()}
    )
    assert single.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


# ----- software interview answers in the gate (software_questions.py, docs 21.3.15 C) -----


def _software_gate_kb(group_count=4, seed=0):
    """Groups G1..Gn, each with malware only it documents (S1xx) and pairs sharing some (S2xx)."""
    import random

    rng = random.Random(seed)
    ids = [f"G{index}" for index in range(1, group_count + 1)]
    group_software = {gid: {f"S1{index:02d}"} for index, gid in enumerate(ids, 1)}
    for number in range(3):
        for gid in rng.sample(ids, 2):
            group_software[gid].add(f"S2{number:02d}")
    software_groups: dict[str, set[str]] = {}
    for gid, items in group_software.items():
        for item in items:
            software_groups.setdefault(item, set()).add(gid)
    kb = _kb(
        groups={gid: SimpleNamespace(id=gid, name=f"Group {gid}") for gid in ids},
        techniques={"T1": SimpleNamespace(parent_id=None), "T2": SimpleNamespace(parent_id=None)},
        group_techniques={gid: ["T1", "T2"] for gid in ids},
    )
    kb.group_software = group_software
    kb.software_groups = software_groups
    kb.software = {
        item: SimpleNamespace(id=item, name=item, software_type="malware")
        for item in software_groups
    }
    return kb, ids


def _random_gate(rng, ids):
    from apt_classifier.separation import CandidateDiscrimination, DiscriminationResult

    window = rng.sample(ids, rng.randint(2, len(ids)))
    totals = [rng.uniform(0.0, 7.0) for _ in window]
    scores = tuple(
        CandidateDiscrimination(gid, 0.1, total, 0.0, 0.0, 0.0, total)
        for gid, total in zip(window, totals, strict=True)
    )
    runner = max(scores[1:], key=lambda item: item.total)
    denominator = max(scores[0].total, runner.total, 1.0)
    discrimination = DiscriminationResult(
        scores=scores,
        top_k=len(scores),
        leader_id=window[0],
        runner_up_id=runner.group_id,
        separation=max(-1.0, min(1.0, (scores[0].total - runner.total) / denominator)),
        floor=1.0,
    )
    fusion = _fusion(
        [
            _fusion_entry(gid, 90.0 - 5 * rank, rank=rank + 1, rank_std=1.0)
            for rank, gid in enumerate(window)
        ]
    )
    cluster = None
    if rng.random() < 0.6:
        cluster = _candidate(
            members=[window[0], window[1]],
            nested=rng.random() < 0.3,
            separation=rng.uniform(0.3, 1.0),
            discriminative=rng.uniform(2.0, 9.0),
            probability=rng.uniform(0.7, 1.0),
        )
    return fusion, discrimination, cluster


def _named(decision):
    if decision.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS:
        return set(decision.named_cluster)
    if decision.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS:
        return {decision.leading_group_id}
    return set()


def test_not_found_answers_never_add_or_change_a_name_property():
    """Random gate windows, random cluster candidates and random "not found" sets: the
    separation and the leader's mass only go down, failed conditions only accumulate, a
    shortlist never becomes a name, and a name is either kept or dropped, never changed."""
    import random
    from collections import Counter

    from apt_classifier.attribution import _hypothesis_gate_failures
    from apt_classifier.software_questions import leader_only_negative

    def kinds(items):
        return {item.split(" ")[0] for item in items}

    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    demotions: Counter = Counter()
    named_bases = 0
    for seed in range(12):
        rng = random.Random(seed)
        kb, ids = _software_gate_kb(group_count=rng.randint(3, 6), seed=seed)
        all_software = sorted(kb.software_groups)
        for _ in range(60):
            fusion, discrimination, cluster = _random_gate(rng, ids)
            thresholds = _cluster_thresholds(
                cluster_naming=rng.random() < 0.7,
                hypothesis_min_discriminative=3.0,
                cluster_min_discriminative=3.5,
            )
            combiner = {"leader_probability": 0.5, "cluster": cluster}
            weight = rng.choice([0.3, 1.0, 3.0])
            base = assess_attribution(
                kb,
                fusion,
                context,
                thresholds=thresholds,
                discrimination=discrimination,
                combiner=combiner,
            )
            chosen = set(rng.sample(all_software, rng.randint(1, 4)))
            for extra in (set(), set(rng.sample(all_software, 2))):
                answers = chosen | extra
                after = assess_attribution(
                    kb,
                    fusion,
                    context,
                    thresholds=thresholds,
                    discrimination=discrimination,
                    combiner=combiner,
                    negative_software=answers,
                    negative_software_weight=weight,
                )
                effect = leader_only_negative(discrimination, kb, answers, weight)
                assert effect.total <= discrimination.leader_total
                assert effect.separation <= discrimination.separation + 1e-12
                assert after.separation <= base.separation + 1e-9
                assert _named(after) in (_named(base), set()), (base.rationale, after.rationale)
                if base.stage is AttributionStage.KNOWN_GROUP_SHORTLIST:
                    assert after.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
                # every naming-bar condition that failed before still fails after
                penalized = SimpleNamespace(
                    separation=effect.separation,
                    leader_total=effect.total,
                    leader_specificity=discrimination.leader_specificity,
                )
                assert kinds(_hypothesis_gate_failures(thresholds, 1.0, 1.0, discrimination)) <= (
                    kinds(_hypothesis_gate_failures(thresholds, 1.0, 1.0, penalized))
                )
                if after.stage is not base.stage:
                    assert after.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
                    assert "not-found interview answers" in after.rationale
                    demotions[base.stage] += 1
                named_bases += bool(_named(base))
                # more "not found" answers never do less
                more = assess_attribution(
                    kb,
                    fusion,
                    context,
                    thresholds=thresholds,
                    discrimination=discrimination,
                    combiner=combiner,
                    negative_software=answers | set(all_software[:3]),
                    negative_software_weight=weight,
                )
                assert _named(more) in (_named(after), set())
                assert more.separation <= after.separation + 1e-9
    # the property was exercised on both kinds of names
    assert named_bases > 100
    assert demotions[AttributionStage.KNOWN_GROUP_HYPOTHESIS] > 5
    assert demotions[AttributionStage.KNOWN_CLUSTER_HYPOTHESIS] > 5


def _gate_with_software(leader_total=4.0, runner_total=0.5):
    from apt_classifier.separation import CandidateDiscrimination, DiscriminationResult

    kb, _ = _software_gate_kb(group_count=3, seed=1)
    scores = (
        CandidateDiscrimination("G1", 0.1, leader_total, 0.0, 0.0, 0.0, leader_total),
        CandidateDiscrimination("G2", 0.1, runner_total, 0.0, 0.0, 0.0, runner_total),
        CandidateDiscrimination("G3", 0.1, 0.2, 0.0, 0.0, 0.0, 0.2),
    )
    discrimination = DiscriminationResult(
        scores=scores,
        top_k=3,
        leader_id="G1",
        runner_up_id="G2",
        separation=(leader_total - runner_total) / max(leader_total, runner_total, 1.0),
        floor=1.0,
    )
    fusion = _fusion(
        [
            _fusion_entry("G1", 80.0, rank_std=1.0),
            _fusion_entry("G2", 60.0, rank=2, rank_std=1.0),
            _fusion_entry("G3", 50.0, rank=3, rank_std=1.0),
        ]
    )
    return kb, fusion, discrimination


def test_not_found_for_the_leaders_malware_moves_the_name_to_the_shortlist():
    kb, fusion, discrimination = _gate_with_software(leader_total=3.2)
    thresholds = _cluster_thresholds(cluster_naming=False, hypothesis_min_discriminative=3.0)
    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    named = assess_attribution(
        kb, fusion, context, thresholds=thresholds, discrimination=discrimination
    )
    assert named.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    # a rival's malware: nothing changes
    rival = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        negative_software={"S102"},
    )
    assert rival.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    assert rival.separation == named.separation
    assert rival.separation_factors["negative_software"]["applied"] == []
    own = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        negative_software={"S101"},
    )
    assert own.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    factors = own.separation_factors["negative_software"]
    assert factors["applied"] == ["S101"] and factors["penalty"] == 0.3
    assert factors["separation_before"] == round(discrimination.separation, 4)
    assert "discriminative evidence 2.90 < 3.00 after the not-found" in own.rationale


def test_found_guard_blocks_a_name_that_rests_on_interview_answers_alone():
    kb, fusion, discrimination = _gate_with_software()
    thresholds = _cluster_thresholds(cluster_naming=False)
    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    guard = {"found": ["S101"], "separation_without": -0.2, "named_without": False}
    blocked = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        software_guard={**guard, "requires_lead": True},
    )
    assert blocked.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "would rest on interview answers alone" in blocked.rationale
    # with the guard off, or a lead of its own without the found software, the name stands
    for kept in (
        {**guard, "requires_lead": False},
        {**guard, "requires_lead": True, "separation_without": 0.1},
    ):
        decision = assess_attribution(
            kb,
            fusion,
            context,
            thresholds=thresholds,
            discrimination=discrimination,
            software_guard=kept,
        )
        assert decision.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def test_a_found_only_enables_a_name_of_a_group_that_documents_it():
    kb, fusion, discrimination = _gate_with_software()
    thresholds = _cluster_thresholds(cluster_naming=False)
    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    base = {"requires_lead": True, "separation_without": 0.3}
    # found for G2's malware, G1 named: refused while the case would be unnamed without it
    other = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        software_guard={**base, "found": ["S102"], "named_without": False},
    )
    assert other.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "documents none of the found malware" in other.rationale
    own = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        software_guard={**base, "found": ["S102", "S101"], "named_without": False},
    )
    assert own.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    # named without the interview software anyway: the found did not enable the name
    anyway = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        software_guard={**base, "found": ["S102"], "named_without": True},
    )
    assert anyway.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS


def test_software_answers_and_documented_clusters_compose_in_a_fixed_order():
    kb, fusion, discrimination = _gate_with_software(leader_total=2.0, runner_total=1.6)
    context = _context(techniques={"T1": 1.0, "T2": 1.0})
    thresholds = _cluster_thresholds(
        hypothesis_min_discriminative=1.0, cluster_min_discriminative=3.5
    )
    cluster = _candidate(members=["G1", "G2"], separation=0.9, discriminative=3.7)
    combiner = {"leader_probability": 0.5, "cluster": cluster}
    named = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        combiner=combiner,
    )
    assert named.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    # a not found for the twin's own malware lowers the merged pair: shortlist, never a single
    for answers in ({"S102"}, {"S101"}):
        demoted = assess_attribution(
            kb,
            fusion,
            context,
            thresholds=thresholds,
            discrimination=discrimination,
            combiner=combiner,
            negative_software=answers,
        )
        assert demoted.stage is AttributionStage.KNOWN_GROUP_SHORTLIST, answers
        assert "the cluster's discriminative evidence" in demoted.rationale
    # malware no member documents: the cluster stands
    kept = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        combiner=combiner,
        negative_software={"S103"},
    )
    assert kept.stage is AttributionStage.KNOWN_CLUSTER_HYPOTHESIS
    # a single name demoted by a not found is never re-sent to the cluster step
    single_kb, single_fusion, single = _gate_with_software(leader_total=3.2, runner_total=0.5)
    exclusive = assess_attribution(
        single_kb,
        single_fusion,
        context,
        thresholds=_cluster_thresholds(hypothesis_min_discriminative=3.0),
        discrimination=single,
        combiner={"cluster": _candidate(members=["G1", "G2"])},
    )
    assert exclusive.stage is AttributionStage.KNOWN_GROUP_HYPOTHESIS
    held = assess_attribution(
        single_kb,
        single_fusion,
        context,
        thresholds=_cluster_thresholds(hypothesis_min_discriminative=3.0),
        discrimination=single,
        combiner={"cluster": _candidate(members=["G1", "G2"])},
        negative_software={"S101"},
    )
    assert held.stage is AttributionStage.KNOWN_GROUP_SHORTLIST and held.named_cluster == ()
    # the found guard covers clusters: without the found software the pair must still lead
    guard = {"found": ["S101"], "requires_lead": True, "named_without": False}
    for value, stage in (
        (-0.1, AttributionStage.KNOWN_GROUP_SHORTLIST),
        (0.2, AttributionStage.KNOWN_CLUSTER_HYPOTHESIS),
    ):
        decision = assess_attribution(
            kb,
            fusion,
            context,
            thresholds=thresholds,
            discrimination=discrimination,
            combiner=combiner,
            software_guard={
                **guard,
                "separation_without": 0.0,
                "cluster_separation_without": value,
            },
        )
        assert decision.stage is stage, value
    # a found that only a non-member documents never names the cluster
    decision = assess_attribution(
        kb,
        fusion,
        context,
        thresholds=thresholds,
        discrimination=discrimination,
        combiner=combiner,
        software_guard={
            **guard,
            "found": ["S103"],
            "separation_without": 0.0,
            "cluster_separation_without": 0.5,
        },
    )
    assert decision.stage is AttributionStage.KNOWN_GROUP_SHORTLIST
    assert "documents none of the found malware" in decision.rationale
