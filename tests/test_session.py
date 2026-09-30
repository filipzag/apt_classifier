"""End-to-end tests for ``AttributionEngine`` orchestration.

These build a real engine over a fake vector store and a fake LLM client and exercise
``analyze`` -> ``rescore`` eligibility, an LLM-free ensemble,
threshold wiring, and deterministic stage decisions.
"""

from __future__ import annotations

import pytest

import apt_classifier.session as session_module
from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.attribution import (
    _evidence_family_count,
    _generate_cluster_id,
)
from apt_classifier.config import Settings
from apt_classifier.cyberre import CYBERRE_CLASSIFIER_NAME, CyberRERankedTechnique, CyberREResult
from apt_classifier.evidence.models import (
    EvidenceLedger,
    EvidenceStatus,
    Observation,
    Provenance,
    QuestionAnswer,
)
from apt_classifier.fusion import FusionEntry, FusionResult
from apt_classifier.llm import LLMResponse, LLMUsage
from apt_classifier.session import _CLASSIFIER_FAMILIES, AttributionEngine
from apt_classifier.vectorstore import VectorHit


class FakeStore:
    """Dense hits keyed on observation text; no group-profile or behavior vectors."""

    def query_techniques(self, texts, *, n_results, **kwargs):
        hits = []
        for text in texts:
            lowered = text.lower()
            if "powershell" in lowered:
                hits.append(
                    [
                        VectorHit(
                            "T1059.001:description:0000",
                            "Adversaries may abuse PowerShell commands and scripts.",
                            {
                                "technique_id": "T1059.001",
                                "name": "PowerShell",
                                "parent_id": "T1059",
                                "field": "description",
                            },
                            0.15,
                            0.85,
                        )
                    ]
                )
            elif "network share" in lowered:
                hits.append(
                    [
                        VectorHit(
                            "T1021.002:description:0000",
                            "Adversaries may use SMB to interact with remote admin shares.",
                            {
                                "technique_id": "T1021.002",
                                "name": "SMB/Windows Admin Shares",
                                "parent_id": "T1021",
                                "field": "description",
                            },
                            0.12,
                            0.88,
                        ),
                        VectorHit(
                            "T1003.001:description:0001",
                            "Credential material may be copied alongside file transfers.",
                            {
                                "technique_id": "T1003.001",
                                "name": "LSASS Memory",
                                "parent_id": "T1003",
                                "field": "description",
                            },
                            0.22,
                            0.78,
                        ),
                    ]
                )
            elif "lsass" in lowered:
                hits.append(
                    [
                        VectorHit(
                            "T1003.001:description:0000",
                            "Adversaries may access credential material stored in LSASS memory.",
                            {
                                "technique_id": "T1003.001",
                                "name": "LSASS Memory",
                                "parent_id": "T1003",
                                "field": "description",
                            },
                            0.15,
                            0.85,
                        )
                    ]
                )
            else:
                hits.append([])
        return hits

    def query_group_profile(self, texts, *, document_type, n_results):
        return []

    def query_group_behaviors(self, texts, *, n_results):
        return [[] for _ in texts]


class FakeLLM:
    def complete(self, messages, **kwargs):  # pragma: no cover - never reached in these tests
        raise AssertionError("LLM must not be called")

    def close(self) -> None:
        return None


def _kb() -> AttackKnowledgeBase:
    kb = AttackKnowledgeBase(
        version="test",
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1003": Technique(id="T1003", name="OS Credential Dumping"),
            "T1003.001": Technique(
                id="T1003.001", name="LSASS Memory", is_subtechnique=True, parent_id="T1003"
            ),
            "T1021.002": Technique(
                id="T1021.002",
                name="SMB/Windows Admin Shares",
                is_subtechnique=True,
                parent_id="T1021",
            ),
        },
        groups={
            "G1": Group(id="G1", name="Alpha", technique_ids=["T1059.001"]),
            "G2": Group(id="G2", name="Bravo", technique_ids=["T1003.001", "T1021.002"]),
            "G3": Group(id="G3", name="Charlie", technique_ids=["T1059.001", "T1003.001"]),
        },
    )
    kb.recompute_statistics()
    return kb


def _settings(**overrides) -> Settings:
    base = {
        "technique_mapping_preset": "local_only",
        "cyberre_weight": 0.0,
        "reranker_enabled": False,
        "profile_forest_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def _ledger(*texts: str) -> EvidenceLedger:
    return EvidenceLedger(
        observations=[
            Observation(text=text, provenance=Provenance(source="free_text")) for text in texts
        ]
    )


def _groups(fusion: FusionResult) -> list[str]:
    return [entry.group_id for entry in fusion.entries]


def test_rescore_after_positive_answer_can_add_previously_ineligible_group():
    engine = AttributionEngine(_settings(), _kb(), FakeStore(), FakeLLM())
    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    assert "T1059.001" in state.context.observed_techniques
    # Only groups with technical overlap are eligible; Bravo has none yet.
    assert set(_groups(state.fusion)) == {"G1", "G3"}

    state.ledger.add_question_answer(
        QuestionAnswer(
            question="Was LSASS memory accessed?",
            technique_id="T1003.001",
            status=EvidenceStatus.POSITIVE,
        )
    )
    engine.rescore(state)
    assert "T1003.001" in state.context.observed_techniques
    # Bravo documents T1003.001, so it must become eligible on the second pass.
    assert "G2" in _groups(state.fusion)
    assert state.context.eligible_group_ids == {"G1", "G2", "G3"}


def test_restart_interview_does_not_keep_a_shrunken_candidate_pool():
    engine = AttributionEngine(_settings(), _kb(), FakeStore(), FakeLLM())
    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    state.ledger.add_question_answer(
        QuestionAnswer(question="q", technique_id="T1003.001", status=EvidenceStatus.POSITIVE)
    )
    engine.rescore(state)
    assert "G2" in _groups(state.fusion)
    engine.restart_interview(state)
    assert set(_groups(state.fusion)) == {"G1", "G3"}
    state.ledger.add_question_answer(
        QuestionAnswer(question="q", technique_id="T1003.001", status=EvidenceStatus.POSITIVE)
    )
    engine.rescore(state)
    assert "G2" in _groups(state.fusion)


def test_the_ensemble_makes_no_llm_call_even_with_a_client(monkeypatch):
    """The LLM zero-shot vote left the ensemble (2026-09-30): with a client configured, analysis
    and rescoring make no model call (FakeLLM raises on any), no LLM opinion family or weight
    remains, and the reviewer receives the fusion the stage decision reads."""
    engine = AttributionEngine(_settings(), _kb(), FakeStore(), FakeLLM())
    assert "llm_zero_shot" not in engine.default_weights()
    assert "llm_opinion" not in engine.family_weights()
    assert not hasattr(engine, "run_zero_shot")
    captured: dict[str, FusionResult] = {}

    def fake_review(client, kb, ledger, classifier_results, fusion, **kwargs):
        captured["fusion"] = fusion
        captured["classifiers"] = tuple(result.classifier for result in classifier_results)
        return "review"

    monkeypatch.setattr(session_module, "run_review_pipeline", fake_review)

    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    assert not hasattr(state, "zero_shot")
    assert state.fusion is state.deterministic_fusion
    assert state.attribution_decision is not None
    assert not any("zero_shot" in name for name in state.fusion.active_classifiers)

    engine.run_review(state)
    assert captured["fusion"] is state.deterministic_fusion

    engine.rescore(state)
    assert state.fusion is state.deterministic_fusion


def _rrf_rel_combiner(engine: AttributionEngine, weight: float):
    """A hand-made combiner for ``engine``'s pipeline whose only feature is the fused score
    (``rrf_rel``): a positive weight follows RRF, a negative one prefers its last candidate."""
    import numpy as np

    from apt_classifier.combiner import FEATURE_NAMES, NULL_FEATURE_NAMES, AttributionCombiner
    from apt_classifier.combiner import pipeline_fingerprint as fingerprint_of

    weights = np.zeros(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("rrf_rel")] = weight
    return AttributionCombiner(
        weights=weights,
        means=np.zeros(len(FEATURE_NAMES)),
        scales=np.ones(len(FEATURE_NAMES)),
        attack_version="test",
        pipeline_fingerprint=fingerprint_of(
            rrf_k=int(engine.settings.rrf_k),
            weights=engine.default_weights(),
            family_weights=engine.family_weights(),
        ),
        null_weights=np.zeros(len(NULL_FEATURE_NAMES)),
    )


@pytest.mark.parametrize("with_combiner", [False, True], ids=["rrf", "combiner"])
def test_cyberre_vote_reaches_the_analyst_ranking_but_not_the_reviewer_or_the_stage(
    monkeypatch, tmp_path, with_combiner
):
    """After an analyst runs CyberRE its vote re-orders the analyst-facing ``fusion`` (here it
    promotes Bravo), while the reviewer and the stage decision read the deterministic fusion:
    exactly the pre-scan fusion, with that fusion's discrimination and combiner."""
    overrides = {"cyberre_weight": 0.35, "attribution_combiner_enabled": with_combiner}
    if with_combiner:
        plain = AttributionEngine(
            _settings(attribution_combiner_enabled=False), _kb(), FakeStore(), FakeLLM()
        )
        model = _rrf_rel_combiner(plain, 8.0)
        overrides["attribution_combiner_path"] = model.save(tmp_path / "combiner.json")
    engine = AttributionEngine(_settings(**overrides), _kb(), FakeStore(), FakeLLM())
    assert (engine.combiner is not None) is with_combiner

    def fake_scan(client, kb, observation, **kwargs):
        # CyberRE "infers" SMB admin shares, which only Bravo documents
        ranking = (CyberRERankedTechnique("T1021.002", "SMB/Windows Admin Shares", 1, "fake"),)
        return CyberREResult(
            observation.id, observation.text, (), ranking, "fake", 0, LLMUsage(0, 0, 0.0)
        )

    stage_inputs: list[dict] = []
    real_assess = session_module.assess_attribution

    def spy_assess(kb, fusion, context, **kwargs):
        stage_inputs.append({"fusion": fusion, **kwargs})
        return real_assess(kb, fusion, context, **kwargs)

    reviewed: dict = {}

    def fake_review(client, kb, ledger, classifier_results, fusion, **kwargs):
        reviewed["fusion"] = fusion
        reviewed["classifiers"] = {result.classifier for result in classifier_results}
        return "review"

    monkeypatch.setattr(session_module, "run_cyberre_scan", fake_scan)
    monkeypatch.setattr(session_module, "assess_attribution", spy_assess)
    monkeypatch.setattr(session_module, "run_review_pipeline", fake_review)

    state = engine.analyze(
        _ledger("PowerShell executed an encoded command.", "LSASS memory was read.")
    )
    assert state.fusion is state.deterministic_fusion
    before = state.deterministic_fusion
    decision = state.attribution_decision
    discrimination = state.context.discrimination
    combiner = state.context.combiner
    assert _groups(before)[0] == "G3"
    assert (combiner is not None) is with_combiner

    engine.run_cyberre(state, state.ledger.observations[0].id)
    assert CYBERRE_CLASSIFIER_NAME in state.fusion.active_classifiers
    assert _groups(state.fusion)[0] == "G2"  # the vote promoted Bravo for the analyst
    deterministic = state.deterministic_fusion
    assert CYBERRE_CLASSIFIER_NAME not in deterministic.active_classifiers
    assert not any(CYBERRE_CLASSIFIER_NAME in e.classifier_ranks for e in deterministic.entries)
    assert deterministic == before
    # the stage read that fusion, with its own discrimination and combiner, and did not move
    assert stage_inputs[-1]["fusion"] is deterministic
    assert stage_inputs[-1]["discrimination"] == discrimination
    assert stage_inputs[-1]["combiner"] == combiner
    assert state.attribution_decision == decision
    assert state.attribution_decision.leading_group_id == "G3"

    engine.run_review(state)
    assert reviewed["fusion"] is deterministic
    assert CYBERRE_CLASSIFIER_NAME not in reviewed["classifiers"]


def test_attribution_thresholds_come_from_settings():
    engine = AttributionEngine(
        _settings(attribution_min_techniques=5, attribution_hypothesis_margin=0.42),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    assert engine.attribution_thresholds.min_techniques == 5
    assert engine.attribution_thresholds.hypothesis_margin == 0.42
    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    assert state.attribution_decision is not None
    assert state.attribution_decision.stage.value == "insufficient_evidence"
    assert "minimum 5" in state.attribution_decision.rationale


def test_default_weights_follow_settings():
    settings = _settings(software_weight=0.33, motivation_weight=0.44)
    engine = AttributionEngine(settings, _kb(), FakeStore(), FakeLLM())
    weights = engine.default_weights()
    assert weights["software_tools"] == 0.33
    assert weights["motivation_alignment"] == 0.44


def _entry(group_id: str, ranks: dict[str, int]) -> FusionEntry:
    return FusionEntry(
        group_id=group_id,
        score=1.0,
        relative_score=100.0,
        rank=1,
        classifier_ranks=ranks,
        rank_min=min(ranks.values()),
        rank_max=max(ranks.values()),
        rank_std=0.0,
        contested=False,
    )


def test_family_count_only_counts_supporting_classifiers():
    fusion = FusionResult(
        entries=(_entry("G1", {"bernoulli_naive_bayes": 50, "specific_behavior": 1}),),
        active_classifiers=("bernoulli_naive_bayes", "specific_behavior"),
        correlations={},
        k=60,
    )
    assert _evidence_family_count(fusion, _CLASSIFIER_FAMILIES) == 1
    fusion = FusionResult(
        entries=(_entry("G1", {"bernoulli_naive_bayes": 2, "specific_behavior": 1}),),
        active_classifiers=("bernoulli_naive_bayes", "specific_behavior"),
        correlations={},
        k=60,
    )
    assert _evidence_family_count(fusion, _CLASSIFIER_FAMILIES) == 2


class _Context:
    def __init__(self, observed):
        self.observed_techniques = observed


def test_cluster_id_depends_on_evidence_not_only_leader():
    fusion = FusionResult(
        entries=(_entry("G1", {"idf_tfidf_cosine": 1}), _entry("G2", {"idf_tfidf_cosine": 2})),
        active_classifiers=("idf_tfidf_cosine",),
        correlations={},
        k=60,
    )
    first = _generate_cluster_id(fusion, _Context({"T1059.001": 0.9}))
    same = _generate_cluster_id(fusion, _Context({"T1059.001": 0.9}))
    other = _generate_cluster_id(fusion, _Context({"T1003.001": 0.9}))
    assert first == same
    assert first != other
    assert first.startswith("CL-UNK-")


class VerifierLLM:
    """Fake OpenRouter client that records verifier payloads and supports one candidate."""

    def __init__(self, supported_id: str, span: str) -> None:
        self.supported_id = supported_id
        self.span = span
        self.payloads: list[str] = []

    def complete(self, messages, **kwargs):
        self.payloads.append(messages[-1]["content"])
        parsed = {
            "decisions": [
                {
                    "technique_id": self.supported_id,
                    "verdict": "supported",
                    "support_span": self.span,
                    "reason": "matches",
                }
            ]
        }
        return LLMResponse("{}", parsed, LLMUsage(10, 5, 0.0), kwargs.get("model", ""), {})

    def close(self) -> None:
        return None


def test_ambiguous_verifier_scope_skips_locally_accepted_primary_and_honours_verdicts():
    text = "Files were copied to the network share of the file server."
    llm = VerifierLLM("T1003.001", "copied to the network share")
    engine = AttributionEngine(
        _settings(technique_mapping_preset="balanced"), _kb(), FakeStore(), llm
    )
    local, _ = engine.map_techniques(_ledger(text), mode="local")
    states = {item.technique_id: item.state.value for item in local.candidates}
    assert states["T1021.002"] == "auto_accepted"
    assert states["T1003.001"] == "needs_review"

    retrieval, verifications = engine.map_techniques(_ledger(text), mode="production")
    assert len(llm.payloads) == 1
    payload = llm.payloads[0]
    assert '"technique_id": "T1003.001"' in payload
    assert '"technique_id": "T1021.002"' not in payload
    states = {item.technique_id: item.state.value for item in retrieval.candidates}
    assert states["T1021.002"] == "auto_accepted"
    assert states["T1003.001"] == "auto_accepted"
    assert set(retrieval.observed_techniques) == {"T1021.002", "T1003.001"}
    assert retrieval.diagnostics["verifier_mode"] == "ambiguous"
    assert list(verifications)


def test_engine_loads_acceptance_model_fail_closed(tmp_path):
    from apt_classifier.acceptance import AcceptanceModel, load_acceptance_model

    disabled = AttributionEngine(
        _settings(acceptance_model_enabled=False), _kb(), FakeStore(), FakeLLM()
    )
    assert disabled.acceptance_model is None
    assert disabled.acceptance_model_status == "disabled"
    missing = AttributionEngine(
        _settings(acceptance_model_path=tmp_path / "nope.json"), _kb(), FakeStore(), FakeLLM()
    )
    assert missing.acceptance_model is None
    assert "not found" in missing.acceptance_model_status
    rows = [dict.fromkeys(["dense_score"], 0.9), dict.fromkeys(["dense_score"], 0.2)]
    model = AcceptanceModel.fit(rows * 5, [1, 0] * 5, attack_version="other-version")
    path = model.save(tmp_path / "model.json")
    settings = _settings(acceptance_model_path=path)
    stale = AttributionEngine(settings, _kb(), FakeStore(), FakeLLM())
    assert stale.acceptance_model is None
    assert "incompatible" in stale.acceptance_model_status
    model.attack_version = "test"
    model.save(path)
    live = AttributionEngine(settings, _kb(), FakeStore(), FakeLLM())
    assert live.acceptance_model is not None
    assert load_acceptance_model(settings, _kb())[1] == "active"
    state = live.analyze(_ledger("PowerShell executed an encoded command."))
    assert state.retrieval.diagnostics["acceptance_model_active"] is True


def test_separation_settings_reach_thresholds_and_context():
    engine = AttributionEngine(
        _settings(
            attribution_gate_mode="separation",
            attribution_hypothesis_separation=0.7,
            attribution_hypothesis_min_discriminative=2.0,
            attribution_separation_behavior_weight=3.0,
        ),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    assert engine.attribution_thresholds.gate_mode == "separation"
    assert engine.attribution_thresholds.hypothesis_separation == 0.7
    assert engine.attribution_thresholds.hypothesis_min_discriminative == 2.0
    assert engine.discrimination_weights.behavior == 3.0
    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    assert state.context.discrimination is not None
    assert state.attribution_decision.gate_mode == "separation"


def test_rerank_blend_zero_is_the_exact_rrf_fusion_and_blend_keeps_top_k(monkeypatch):
    plain = AttributionEngine(_settings(), _kb(), FakeStore(), FakeLLM())
    blended = AttributionEngine(
        _settings(attribution_rerank_blend=1.0), _kb(), FakeStore(), FakeLLM()
    )
    ledger_text = "PowerShell executed an encoded command."
    a = plain.analyze(_ledger(ledger_text)).deterministic_fusion
    b = blended.analyze(_ledger(ledger_text)).deterministic_fusion
    top_k = int(getattr(blended.settings, "attribution_separation_top_k", 5))
    assert {e.group_id for e in a.entries[:top_k]} == {e.group_id for e in b.entries[:top_k]}
    assert [e.group_id for e in a.entries[top_k:]] == [e.group_id for e in b.entries[top_k:]]
    assert b.entries[0].relative_score == pytest.approx(100.0)


def test_rerank_top_k_contrasts_a_longer_shortlist_but_the_gate_keeps_its_own(monkeypatch):
    """ATTRIBUTION_RERANK_TOP_K re-ranks over a longer shortlist than the naming gate scores:
    the recorded discrimination (gate) still has ATTRIBUTION_SEPARATION_TOP_K candidates, the
    re-ranked top-K set is the RRF top-K set, and 0 means the same shortlist as the gate."""
    text = "PowerShell executed an encoded command."
    plain = AttributionEngine(
        _settings(attribution_rerank_blend=0.0), _kb(), FakeStore(), FakeLLM()
    )
    same = AttributionEngine(
        _settings(attribution_rerank_blend=1.0, attribution_rerank_top_k=0),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    longer = AttributionEngine(
        _settings(attribution_rerank_blend=1.0, attribution_rerank_top_k=4),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    rrf = plain.analyze(_ledger(text))
    a = same.analyze(_ledger(text))
    b = longer.analyze(_ledger(text))
    gate_k = int(getattr(longer.settings, "attribution_separation_top_k", 5))
    assert a.context.discrimination.top_k == min(gate_k, len(rrf.deterministic_fusion.entries))
    assert b.context.discrimination.top_k == min(gate_k, len(rrf.deterministic_fusion.entries))
    order = [e.group_id for e in rrf.deterministic_fusion.entries]
    assert {e.group_id for e in b.deterministic_fusion.entries[:4]} == set(order[:4])
    assert [e.group_id for e in b.deterministic_fusion.entries[4:]] == order[4:]
    assert b.deterministic_fusion.entries[0].relative_score == pytest.approx(100.0)


def test_prior_verifications_are_reused_without_calling_the_verifier():
    """Verdicts saved with a case are applied on re-analysis (in production and local mode) and
    the covered candidates are never sent to the verifier again."""
    from apt_classifier.session import collect_verifications, verifications_from_records

    text = "Files were copied to the network share of the file server."
    first = VerifierLLM("T1003.001", "copied to the network share")
    engine = AttributionEngine(
        _settings(technique_mapping_preset="balanced"), _kb(), FakeStore(), first
    )
    retrieval, _ = engine.map_techniques(_ledger(text), mode="production")
    assert len(first.payloads) == 1
    records = collect_verifications(retrieval)
    assert [(r["technique_id"], r["verdict"]) for r in records] == [("T1003.001", "supported")]
    assert records[0]["observation_id"] == retrieval.candidates[0].observation_id
    prior = verifications_from_records(records)
    assert set(prior) == {(records[0]["observation_id"], "T1003.001")}

    # a second engine whose verifier would contradict the saved verdict is never asked
    second = VerifierLLM("T1021.002", "copied")
    replay = AttributionEngine(
        _settings(technique_mapping_preset="balanced"), _kb(), FakeStore(), second
    )
    again, _ = replay.map_techniques(_ledger(text), mode="production", prior_verifications=prior)
    assert second.payloads == []
    states = {item.technique_id: item.state.value for item in again.candidates}
    assert states["T1003.001"] == "auto_accepted"
    assert again.diagnostics["verifier_reused"] == 1
    assert set(again.observed_techniques) == {"T1021.002", "T1003.001"}
    # local mode (no client) applies the saved verdict too
    local = AttributionEngine(_settings(), _kb(), FakeStore(), None)  # type: ignore[arg-type]
    offline, _ = local.map_techniques(_ledger(text), mode="local", prior_verifications=prior)
    assert {i.technique_id: i.state.value for i in offline.candidates}[
        "T1003.001"
    ] == "auto_accepted"
    # records that are not final verdicts are ignored
    assert verifications_from_records([{**records[0], "verdict": "error"}, {"bad": 1}]) == {}
    assert collect_verifications(offline)  # the applied verdict is collected again


def test_combiner_orders_the_shortlist_and_the_gate_scores_the_new_leader(monkeypatch, tmp_path):
    """With a combiner loaded the shortlist follows its probabilities, the discrimination is
    recomputed on that order (leader-consistency guard never fires spuriously), the stage
    carries the calibrated leader probability, and disabling it restores the RRF path."""
    from apt_classifier.combiner import AttributionCombiner

    text = "PowerShell executed an encoded command."
    plain = AttributionEngine(
        _settings(attribution_combiner_enabled=False), _kb(), FakeStore(), FakeLLM()
    )
    baseline = plain.analyze(_ledger(text))
    assert plain.combiner is None and plain.combiner_status == "disabled"
    assert baseline.context.combiner is None
    # a hand-made combiner that prefers the *last* RRF candidate (negative weight on rrf_rel)
    model = _rrf_rel_combiner(plain, -8.0)
    path = model.save(tmp_path / "combiner.json")
    engine = AttributionEngine(
        _settings(
            attribution_combiner_enabled=True,
            attribution_combiner_path=path,
            attribution_combiner_blend=1.0,
        ),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    assert engine.combiner is not None and engine.combiner_status == "active"
    state = engine.analyze(_ledger(text))
    info = state.context.combiner
    assert info is not None and info["status"] == "active"
    rrf_order = [g for g, _ in sorted(info["rrf_ranks"].items(), key=lambda item: item[1])]
    combined = [e.group_id for e in state.deterministic_fusion.entries[: len(rrf_order)]]
    assert set(combined) == set(rrf_order)
    if len(rrf_order) > 1:
        assert combined[0] == rrf_order[-1]  # the model's preference wins over RRF
    assert abs(sum(info["probabilities"].values()) + (info["null_probability"] or 0.0) - 1.0) < 0.01
    assert state.context.discrimination.leader_id == state.deterministic_fusion.entries[0].group_id
    decision = state.attribution_decision
    assert decision.is_calibrated and decision.calibrated_probability == info["leader_probability"]
    # a stale fingerprint disables the combiner fail-closed
    stale = AttributionCombiner.from_dict({**model.to_dict(), "pipeline_fingerprint": "old"})
    stale_path = stale.save(tmp_path / "stale.json")
    disabled = AttributionEngine(
        _settings(attribution_combiner_enabled=True, attribution_combiner_path=stale_path),
        _kb(),
        FakeStore(),
        FakeLLM(),
    )
    assert disabled.combiner is None and "fingerprint" in disabled.combiner_status


def test_profile_forest_members_are_wired_and_swappable():
    engine = AttributionEngine(_settings(), _kb(), FakeStore(), FakeLLM())
    weights = engine.default_weights()
    # both members are wired; the votes stay at 0 by default (docs 21.3.11)
    assert weights["profile_forest"] == 0.0 and weights["source_country"] == 0.0
    assert {"learned_profile", "origin"} <= set(engine.family_weights())
    members = [c for c in engine.classifiers if c.name in {"profile_forest", "source_country"}]
    assert len(members) == 2 and all(c.model is None for c in members)
    assert engine.profile_forest_status == "disabled"
    sentinel = object()
    engine.set_profile_forest(sentinel)
    assert all(c.model is sentinel for c in members)
    assert engine.profile_forest_status.startswith("active")
    # without a model both members abstain and the analysis is unchanged
    engine.set_profile_forest(None)
    state = engine.analyze(_ledger("PowerShell executed an encoded command."))
    by_name = {r.classifier: r for r in state.classifier_results}
    assert not by_name["profile_forest"].applicable and not by_name["source_country"].applicable
