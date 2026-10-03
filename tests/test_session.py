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
    # both members are wired; the source-country vote is on by default, the group forest silent
    assert weights["profile_forest"] == 0.0 and weights["source_country"] == 1.0
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


def test_group_mlp_at_weight_zero_leaves_the_pipeline_unchanged(tmp_path):
    """Off (the default) the network is not loaded and adds no weight key, so the fusion and
    the combiner fingerprint are exactly the pre-network ones; with a weight it votes in the
    learned_profile family and the fingerprint changes (the combiner must be refitted)."""
    pytest.importorskip("torch")
    from apt_classifier.group_mlp import GROUP_MLP_NAME, fit_group_mlp

    kb = _kb()
    model = fit_group_mlp(
        kb,
        params={"samples_per_group": 20, "hidden": [16], "epochs": 2, "size_min": 1, "size_max": 2},
        device="cpu",
    )
    path = model.save(tmp_path / "mlp.npz")
    off = AttributionEngine(_settings(group_mlp_path=path), kb, FakeStore(), FakeLLM())
    plain = AttributionEngine(
        _settings(group_mlp_enabled=False, group_mlp_path=path), kb, FakeStore(), FakeLLM()
    )
    assert off.group_mlp is None and off.group_mlp_status == "disabled (weight 0)"
    assert GROUP_MLP_NAME not in off.default_weights()
    assert off.combiner_fingerprint == plain.combiner_fingerprint
    assert all(c.name != GROUP_MLP_NAME for c in off.classifiers)

    on = AttributionEngine(
        _settings(group_mlp_path=path, group_mlp_weight=1.0), kb, FakeStore(), FakeLLM()
    )
    assert on.group_mlp is not None and on.group_mlp_status.startswith("active")
    assert on.default_weights()[GROUP_MLP_NAME] == 1.0
    assert _CLASSIFIER_FAMILIES[GROUP_MLP_NAME] == "learned_profile"
    assert on.combiner_fingerprint != plain.combiner_fingerprint
    state = on.analyze(_ledger("Mimikatz dumped LSASS credentials"))
    assert any(r.classifier == GROUP_MLP_NAME for r in state.classifier_results)
    on.set_group_mlp(None)
    assert on.group_mlp_status == "disabled"

    missing = AttributionEngine(
        _settings(group_mlp_path=tmp_path / "none.npz", group_mlp_weight=1.0),
        kb,
        FakeStore(),
        FakeLLM(),
    )
    assert missing.group_mlp is None and "not found" in missing.group_mlp_status


def test_auto_exemplar_backend_without_a_store_client_uses_the_npz_offline(tmp_path):
    """A store without a Chroma client (every test fake) never reaches the network: ``auto``
    falls back to the npz library with a visible reason, strict ``chroma`` loads none."""
    from apt_classifier.exemplars import ExemplarIndex

    class Embedder:
        def embed(self, texts, batch_size=None):
            return [[1.0, float("powershell" in text.lower()), 0.5] for text in texts]

    settings = Settings()
    index = ExemplarIndex.build(
        [
            ("powershell ran a script", ["T1059.001"], "tram"),
            ("lsass was read", ["T1003.001"], "tram"),
        ],
        Embedder(),
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_model_revision,
        attack_version="test",
    )
    path = index.save(tmp_path / "exemplars.npz")
    store = FakeStore()
    assert not hasattr(store, "client")
    engine = AttributionEngine(
        _settings(technique_exemplar_backend="auto", technique_exemplar_index_path=path),
        _kb(),
        store,
        FakeLLM(),
    )
    assert engine.exemplar_index is not None and engine.exemplar_index.backend == "npz"
    assert engine.exemplar_index.fingerprint == index.fingerprint
    assert engine.exemplar_status.startswith("active (npz fallback: no Chroma client")
    strict = AttributionEngine(
        _settings(technique_exemplar_backend="chroma", technique_exemplar_index_path=path),
        _kb(),
        store,
        FakeLLM(),
    )
    assert strict.exemplar_index is None
    assert strict.exemplar_status.startswith("chroma exemplar store unavailable")
    # the analysis still runs (the acceptance model falls back without exemplars)
    state = strict.analyze(_ledger("PowerShell executed an encoded command."))
    assert "T1059.001" in state.context.observed_techniques


# ----- software interview answers (software_questions.py, docs 21.3.15 C) -----


def _software_engine(tmp_path=None, *, combiner: bool = False, **overrides):
    from tests.test_software_questions import software_engine, software_kb, software_settings

    if not combiner:
        return software_engine(**overrides)
    plain = software_engine(**overrides)
    path = _rrf_rel_combiner(plain, 8.0).save(tmp_path / "combiner.json")
    engine = AttributionEngine(
        software_settings(
            attribution_combiner_enabled=True, attribution_combiner_path=path, **overrides
        ),
        software_kb(),
        FakeStore(),
        None,
    )
    assert engine.combiner is not None
    return engine


def _software_answer(software_id, status, question="Did you find it?"):
    return QuestionAnswer(
        question=question, kind="software", software_id=software_id, status=status
    )


def _snapshot(state):
    discrimination = state.context.discrimination
    return (
        [(e.group_id, e.score, e.rank) for e in state.fusion.entries],
        dict((state.context.combiner or {}).get("probabilities") or {}),
        discrimination.to_dict() if discrimination is not None else None,
        set(state.context.negative_techniques),
        set(state.context.software_ids),
    )


@pytest.mark.parametrize("with_combiner", [False, True], ids=["rrf", "combiner"])
def test_a_not_found_answer_leaves_scoring_untouched(tmp_path, with_combiner):
    """A "not found" reaches only the naming gate: fusion, combiner probabilities, the gate's
    discrimination and the technique negatives are exactly as before."""
    from tests.test_software_questions import LSASS, POWERSHELL

    engine = _software_engine(tmp_path, combiner=with_combiner)
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    before = _snapshot(state)
    state.ledger.add_question_answer(_software_answer("S0103", EvidenceStatus.NEGATIVE))
    engine.rescore(state)
    assert _snapshot(state) == before
    assert state.context.negative_software_ids == {"S0103"}
    # "not checked" records nothing at all
    state.ledger.add_question_answer(_software_answer("S0101", EvidenceStatus.UNKNOWN))
    engine.rescore(state)
    assert _snapshot(state) == before and state.context.negative_software_ids == {"S0103"}


def test_a_found_answer_counts_like_software_named_in_the_text_and_restart_clears_it():
    from tests.test_software_questions import LSASS, POWERSHELL

    engine = _software_engine()
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    assert state.context.document_software_ids == set() and state.context.software_ids == set()
    question = engine.next_question(state)
    assert question is not None and question.kind == "software"
    engine.answer_question(state, "found")
    answer = state.ledger.question_answers[-1]
    assert answer.kind == "software" and answer.software_id == question.software_id
    assert answer.technique_id is None
    assert answer.provenance.source == "analyst_interview"
    assert state.context.software_ids == {question.software_id}
    assert state.context.interview_found_software == {question.software_id}
    assert state.context.decisive_software["software_ids"] == [question.software_id]
    # the next software question never repeats it
    follow_up = engine.next_question(state)
    assert follow_up is None or follow_up.software_id != question.software_id
    state.current_question = follow_up
    if follow_up is not None and follow_up.kind == "software":
        engine.answer_question(state, "not found")
        assert state.context.negative_software_ids == {follow_up.software_id}
    engine.restart_interview(state)
    assert state.context.software_ids == set() and state.context.negative_software_ids == set()
    assert state.context.interview_found_software == set()
    # a software question rejects "not visible"
    question = engine.next_question(state)
    with pytest.raises(ValueError, match="found, not found or not checked"):
        engine.answer_question(state, EvidenceStatus.NOT_VISIBLE)
    assert question is not None and not state.ledger.question_answers


def test_software_questions_follow_the_settings_and_stop_when_named():
    from tests.test_software_questions import LSASS, POWERSHELL

    off = _software_engine(interview_software_questions=False)
    state = off.analyze(_ledger(POWERSHELL, LSASS))
    assert off.software_question(state) is None
    question = off.next_question(state)
    assert question is not None and question.kind == "technique"
    engine = _software_engine(interview_software_max_questions=1)
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    engine.next_question(state)
    engine.answer_question(state, "not checked")
    assert engine.software_question(state) is None  # cap reached
    # a case already named (the text names Charlie's own malware) is not asked
    named = _software_engine()
    state = named.analyze(_ledger(f"{POWERSHELL} The Greenfin implant was loaded.", LSASS))
    assert state.attribution_decision.stage.value == "known_group_hypothesis"
    assert named.software_question(state) is None
    asking = _software_engine(interview_software_when_named=True)
    state = asking.analyze(_ledger(f"{POWERSHELL} The Greenfin implant was loaded.", LSASS))
    assert asking.software_question(state) is not None


def test_a_saved_case_with_software_answers_reloads_to_the_identical_decision():
    from apt_classifier.evidence.models import EvidenceLedger
    from tests.test_software_questions import LSASS, POWERSHELL

    engine = _software_engine()
    text = f"{POWERSHELL} The Greenfin implant was loaded."
    state = engine.analyze(_ledger(text, LSASS))
    assert state.attribution_decision.stage.value == "known_group_hypothesis"
    # "not found" for Charlie's two other malware entries blocks the name ...
    for software_id in ("S0104", "S0108"):
        state.ledger.add_question_answer(
            _software_answer(software_id, EvidenceStatus.NEGATIVE, f"Did you find {software_id}?")
        )
    engine.rescore(state)
    blocked = state.attribution_decision
    assert blocked.stage.value == "known_group_shortlist"
    assert blocked.separation_factors["negative_software"]["applied"] == ["S0104", "S0108"]
    # ... and stays blocked when the case is reopened from disk
    reopened = engine.analyze(EvidenceLedger.from_dict(state.ledger.to_dict()))
    assert reopened.attribution_decision.to_dict() == blocked.to_dict()
    assert reopened.context.negative_software_ids == {"S0104", "S0108"}
    # a "found" survives the reload too, guard included
    other = engine.analyze(_ledger(POWERSHELL, LSASS))
    engine.next_question(other)
    engine.answer_question(other, "found")
    reloaded = engine.analyze(EvidenceLedger.from_dict(other.ledger.to_dict()))
    assert reloaded.attribution_decision.to_dict() == other.attribution_decision.to_dict()
    assert reloaded.context.software_ids == other.context.software_ids
    assert reloaded.context.interview_software_guard == other.context.interview_software_guard


def test_a_found_for_one_group_never_names_another():
    """Alpha's malware found while Charlie leads: Alpha takes the lead but its separation without
    the answer is negative, so nothing is named; Charlie is never named on Alpha's malware."""
    from tests.test_software_questions import LSASS, POWERSHELL

    engine = _software_engine()
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    for software_id in ("S0101", "S0102"):
        trial = engine.analyze(_ledger(POWERSHELL, LSASS))
        trial.ledger.add_question_answer(_software_answer(software_id, EvidenceStatus.POSITIVE))
        engine.rescore(trial)
        decision = trial.attribution_decision
        owner = engine.kb.software[software_id].group_ids[0]
        if decision.stage.value == "known_group_hypothesis":
            assert decision.leading_group_id == owner
        guard = trial.context.interview_software_guard
        assert guard["found"] == [software_id] and guard["document_software_ids"] == []
    assert state.attribution_decision.stage.value != "known_group_hypothesis"


def test_old_ledgers_load_and_question_ids_do_not_change():
    import hashlib

    from apt_classifier.evidence.models import EvidenceLedger

    legacy = {
        "observations": [],
        "question_answers": [
            {"question": "Was LSASS read?", "technique_id": "T1003.001", "status": "negative"}
        ],
    }
    ledger = EvidenceLedger.from_dict(legacy)
    answer = ledger.question_answers[0]
    material = "\x1f".join(("t1003.001", "was lsass read?"))
    expected = f"question-{hashlib.sha256(material.encode()).hexdigest()[:16]}"
    assert answer.question_id == expected and answer.kind == "technique"
    assert answer.software_id is None
    software = _software_answer("s0101", EvidenceStatus.POSITIVE, question="Did you find X?")
    assert software.software_id == "S0101" and software.question_id.startswith("question-")
    assert software.question_id != QuestionAnswer(question="Did you find X?").question_id
    roundtrip = EvidenceLedger.from_dict({**legacy, "question_answers": [software.to_dict()]})
    assert roundtrip.question_answers[0] == software
