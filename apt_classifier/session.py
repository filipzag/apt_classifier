from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import ValidationError

from apt_classifier.acceptance import load_acceptance_model
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attribution import (
    AttributionDecision,
    MaturityThresholds,
    assess_attribution,
)
from apt_classifier.classifiers.base import ClassifierContext, ClassifierResult
from apt_classifier.classifiers.naive_bayes import posterior
from apt_classifier.classifiers.registry import default_classifiers, run_classifiers
from apt_classifier.classifiers.software import decisive_software_factor
from apt_classifier.combiner import (
    apply_combiner_order,
    combiner_inputs,
    load_attribution_combiner,
    pipeline_fingerprint,
)
from apt_classifier.config import Settings
from apt_classifier.cyberre import (
    CYBERRE_CLASSIFIER_NAME,
    CyberREResult,
    TechniqueCandidateVerificationResult,
    run_cyberre_scan,
    verify_technique_candidates,
)
from apt_classifier.evidence.models import (
    EvidenceLedger,
    EvidenceStatus,
    Observation,
    QuestionAnswer,
    TechniqueVerification,
    TechniqueVerificationVerdict,
)
from apt_classifier.exemplars import exemplar_kwargs, load_exemplar_index
from apt_classifier.fusion import FusionResult, reciprocal_rank_fusion
from apt_classifier.llm import OpenRouterClient
from apt_classifier.origin import OriginAndThreatAssessment, assess_origin_and_threat_type
from apt_classifier.presets import apply_preset
from apt_classifier.questions import (
    INTERVIEW_REVISION,
    QuestionCandidate,
    SelectedQuestion,
    interview_stop_reason,
    select_question_locally,
    select_question_with_llm,
    shortlist_questions,
)
from apt_classifier.reranker import LocalReranker, RerankerConfig
from apt_classifier.retrieve import (
    LEXICAL_DEFAULT_FIELDS,
    LexicalTechniqueIndex,
    RetrievalResult,
    apply_candidate_decisions,
    build_classifier_context,
    finalize_retrieval_candidates,
    refresh_context_answers,
    retrieve_techniques,
)
from apt_classifier.review.pipeline import ReviewOutcome, run_review_pipeline
from apt_classifier.separation import (
    DiscriminationWeights,
    apply_discriminative_rerank,
    discriminate,
)
from apt_classifier.threat_intel import (
    ThreatIntelMatch,
    ThreatIntelSnapshot,
    match_threat_intel_to_case,
)
from apt_classifier.vectorstore import ChromaVectorStore

_TECHNICAL_EVIDENCE_CLASSIFIERS = frozenset(
    {
        "idf_tfidf_cosine",
        "weighted_jaccard",
        "specific_behavior",
        "campaign_profile",
        "software_tools",
    }
)
_CLASSIFIER_FAMILIES = {
    "bernoulli_naive_bayes": "technique_overlap",
    "idf_tfidf_cosine": "technique_overlap",
    "weighted_jaccard": "technique_overlap",
    "semantic_centroid": "technique_overlap",
    "specific_behavior": "actor_behavior",
    "software_tools": "software",
    "campaign_profile": "campaign",
    "group_description": "context",
    "strategic_context": "context",
    "motivation_alignment": "motivation",
    CYBERRE_CLASSIFIER_NAME: "cyberre_llm",
}


@dataclass
class AnalysisState:
    ledger: EvidenceLedger
    retrieval: RetrievalResult
    context: ClassifierContext
    classifier_results: tuple[ClassifierResult, ...]
    fusion: FusionResult
    bayes_posterior: dict[str, float]
    origin_assessment: OriginAndThreatAssessment
    weights: dict[str, float] = field(default_factory=dict)
    # Fusion the isolated reviewer and the attribution stage read. It equals ``fusion`` since the
    # LLM zero-shot vote left the ensemble (2026-09-30); kept so callers and saved states are unchanged.
    deterministic_fusion: FusionResult | None = None
    current_question: SelectedQuestion | None = None
    interview_complete: bool = False
    interview_stop_reason: str | None = None
    interview_revision: int = INTERVIEW_REVISION
    threat_intel_snapshot: ThreatIntelSnapshot | None = None
    threat_intel_matches: tuple[ThreatIntelMatch, ...] = ()
    cyberre_results: dict[str, CyberREResult] = field(default_factory=dict)
    technique_verifications: dict[str, TechniqueCandidateVerificationResult] = field(
        default_factory=dict
    )
    review: ReviewOutcome | None = None
    telemetry_profile: Any = None
    attribution_decision: AttributionDecision | None = None


_FINAL_VERDICTS = frozenset(
    {
        TechniqueVerificationVerdict.SUPPORTED,
        TechniqueVerificationVerdict.UNSUPPORTED,
        TechniqueVerificationVerdict.UNCERTAIN,
    }
)


def collect_verifications(retrieval: Any) -> list[dict[str, Any]]:
    """Verifier verdicts worth keeping with a case: one record per (observation, technique)
    that received a final verdict. ``not_run`` and ``error`` are not verdicts and are left out
    so the next analysis retries them."""
    records: list[dict[str, Any]] = []
    for candidate in getattr(retrieval, "candidates", ()) or ():
        verification = getattr(candidate, "verification", None)
        if verification is None or verification.verdict not in _FINAL_VERDICTS:
            continue
        records.append(
            {
                "observation_id": candidate.observation_id,
                "technique_id": candidate.technique_id,
                **verification.to_dict(),
            }
        )
    return records


def verifications_from_records(
    records: Any,
) -> dict[tuple[str, str], TechniqueVerification]:
    """Inverse of :func:`collect_verifications`; malformed or non-final records are skipped."""
    prior: dict[tuple[str, str], TechniqueVerification] = {}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        payload = {k: v for k, v in record.items() if k not in {"observation_id", "technique_id"}}
        try:
            verification = TechniqueVerification.model_validate(payload)
        except ValidationError:
            continue
        if verification.verdict not in _FINAL_VERDICTS:
            continue
        key = (str(record.get("observation_id", "")), str(record.get("technique_id", "")).upper())
        if all(key):
            prior[key] = verification
    return prior


def _apply_prior_verifications(
    retrieval: Any, prior: dict[tuple[str, str], TechniqueVerification]
) -> tuple[Any, int]:
    """Attach stored verdicts to the matching candidates (observation ids are content hashes,
    so a verdict follows its observation across re-analyses)."""
    updated = []
    reused = 0
    for candidate in retrieval.candidates:
        verification = prior.get((candidate.observation_id, candidate.technique_id))
        if verification is None:
            updated.append(candidate)
            continue
        updated.append(candidate.model_copy(update={"verification": verification}))
        reused += 1
    diagnostics = dict(retrieval.diagnostics or {})
    diagnostics["verifier_reused"] = reused
    return replace(retrieval, candidates=tuple(updated), diagnostics=diagnostics), reused


class AttributionEngine:
    def __init__(
        self,
        settings: Settings,
        knowledge_base: AttackKnowledgeBase,
        vector_store: ChromaVectorStore,
        llm: OpenRouterClient,
    ) -> None:
        self.settings = apply_preset(settings)
        self.kb = knowledge_base
        self.store = vector_store
        self.llm = llm
        include_synthetic = bool(getattr(self.settings, "technique_include_synthetic", False))
        self.lexical_index = LexicalTechniqueIndex.from_kb(
            knowledge_base,
            fields=tuple(
                field
                for field in LEXICAL_DEFAULT_FIELDS
                if include_synthetic or field != "synthetic"
            ),
        )
        self.reranker = LocalReranker(
            RerankerConfig(
                enabled=bool(getattr(self.settings, "reranker_enabled", False)),
                model_name=str(getattr(self.settings, "reranker_model", "")),
                model_revision=getattr(self.settings, "reranker_model_revision", None),
                max_candidates=int(getattr(self.settings, "reranker_max_candidates", 20)),
                attack_version=str(getattr(knowledge_base, "version", "")) or None,
                expected_base_model=getattr(self.settings, "reranker_expected_base_model", None),
                expected_base_model_revision=getattr(
                    self.settings, "reranker_expected_base_model_revision", None
                ),
                expected_artifact_fingerprint=getattr(
                    self.settings, "reranker_expected_artifact_fingerprint", None
                ),
            )
        )
        self.classifiers = default_classifiers(
            self.settings.nb_alpha,
            self.settings.nb_beta,
            self.settings.negative_evidence_weight,
            software_tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
            idf_length_norm=float(getattr(self.settings, "idf_overlap_length_norm", 1.0)),
            software_idf=bool(getattr(self.settings, "software_idf_enabled", True)),
        )
        self.telemetry_profile: Any = None
        self.exemplar_index, self.exemplar_status = load_exemplar_index(self.settings)
        self.acceptance_model, self.acceptance_model_status = load_acceptance_model(
            self.settings, knowledge_base, self.exemplar_index
        )
        self.combiner_fingerprint = pipeline_fingerprint(
            rrf_k=int(self.settings.rrf_k),
            weights=self.default_weights(),
            family_weights=self.family_weights(),
        )
        self.combiner, self.combiner_status = load_attribution_combiner(
            self.settings, knowledge_base, fingerprint=self.combiner_fingerprint
        )
        self.attribution_thresholds = MaturityThresholds(
            min_techniques=int(getattr(self.settings, "attribution_min_techniques", 2)),
            min_independent_evidence_families=int(
                getattr(self.settings, "attribution_min_independent_families", 2)
            ),
            min_maturity_score=float(getattr(self.settings, "attribution_min_maturity", 0.5)),
            hypothesis_maturity_score=float(
                getattr(self.settings, "attribution_hypothesis_maturity", 0.7)
            ),
            hypothesis_margin=float(getattr(self.settings, "attribution_hypothesis_margin", 0.10)),
            gate_mode=str(getattr(self.settings, "attribution_gate_mode", "separation")),
            hypothesis_separation=float(
                getattr(self.settings, "attribution_hypothesis_separation", 0.5)
            ),
            hypothesis_min_discriminative=float(
                getattr(self.settings, "attribution_hypothesis_min_discriminative", 3.0)
            ),
            hypothesis_specificity=float(
                getattr(self.settings, "attribution_hypothesis_specificity", 0.0)
            ),
        )

    def map_techniques(
        self,
        ledger: EvidenceLedger,
        *,
        verify: bool | None = None,
        mode: str = "production",
        prior_verifications: dict[tuple[str, str], TechniqueVerification] | None = None,
    ) -> tuple[RetrievalResult, dict[str, TechniqueCandidateVerificationResult]]:
        """Map a ledger through one shared local or production pipeline.

        ``local`` is deterministic and never calls OpenRouter. ``production``
        uses the configured verifier unless the caller explicitly supplies
        ``verify``.  Both modes collect identical retrieval evidence and pass
        through the same final acceptance gate. ``prior_verifications`` (verdicts saved with
        the case, keyed by observation and technique id) are applied in both modes before any
        verifier call, and candidates that carry one are never sent to the verifier again, so
        reopening a case is deterministic and free.
        """
        if mode not in {"local", "production"}:
            raise ValueError("mapping mode must be 'local' or 'production'")
        if verify is None:
            # No client (local mode, benchmarks) means no verification attempts at all rather
            # than one failing call per observation.
            verify = (
                mode == "production"
                and self.llm is not None
                and str(getattr(self.settings, "technique_verifier_mode", "ambiguous")) != "off"
            )
        if mode == "local" and verify:
            raise ValueError("local mapping mode cannot enable OpenRouter verification")
        retrieval = retrieve_techniques(
            ledger,
            self.store,
            self.kb,
            threshold=self.settings.similarity_threshold,
            top_k=self.settings.retrieval_top_k,
            behavior_threshold=self.settings.behavior_similarity_threshold,
            behavior_top_k=self.settings.behavior_retrieval_top_k,
            retrieval_mode=getattr(self.settings, "technique_retrieval_mode", "hybrid"),
            candidate_pool=getattr(self.settings, "technique_candidate_pool", 12),
            include_detection=getattr(self.settings, "technique_include_detection", False),
            parent_policy=getattr(self.settings, "technique_parent_policy", "ancestor_compatible"),
            lexical_index=self.lexical_index,
            acceptance_threshold=getattr(self.settings, "technique_acceptance_threshold", 0.75),
            relative_margin=getattr(self.settings, "technique_relative_margin", 0.08),
            reranker=self.reranker if self.reranker.available else None,
            finalize=False,
            acceptance_model=self.acceptance_model,
            rank_by_acceptance=bool(getattr(self.settings, "technique_rank_by_acceptance", False)),
            corroboration_accept_observations=int(
                getattr(self.settings, "technique_corroboration_accept_observations", 0)
            ),
            corroboration_accept_min_probability=float(
                getattr(self.settings, "technique_corroboration_accept_min_probability", 0.2)
            ),
            rrf_k=int(getattr(self.settings, "technique_rrf_k", 60)),
            corroboration_bonus=float(getattr(self.settings, "technique_corroboration_bonus", 0.0)),
            segment_acceptance=bool(getattr(self.settings, "technique_segment_acceptance", False)),
            hierarchy_parent_mode=str(
                getattr(self.settings, "technique_hierarchy_parent_mode", "off")
            ),
            include_synthetic=bool(getattr(self.settings, "technique_include_synthetic", False)),
            **exemplar_kwargs(self.settings, self.exemplar_index),
        )
        verifications: dict[str, TechniqueCandidateVerificationResult] = {}
        # Local gating runs first so that the verifier's ``ambiguous`` scope means what it
        # says: only candidates still NEEDS_REVIEW after deterministic acceptance are sent.
        # Verdicts are then folded in by a second, idempotent finalization pass.
        retrieval = self._finalize(retrieval, ledger)
        reused: set[tuple[str, str]] = set()
        if prior_verifications:
            retrieval, count = _apply_prior_verifications(retrieval, prior_verifications)
            if count:
                reused = {
                    (c.observation_id, c.technique_id)
                    for c in retrieval.candidates
                    if (c.observation_id, c.technique_id) in prior_verifications
                }
                retrieval = self._finalize(retrieval, ledger)
        if verify:
            retrieval, verifications = self._verify_retrieval_candidates(
                ledger, retrieval, exclude=reused
            )
            retrieval = self._finalize(retrieval, ledger)
        retrieval = apply_candidate_decisions(retrieval, ledger, self.kb)
        return retrieval, verifications

    def _finalize(self, retrieval: RetrievalResult, ledger: EvidenceLedger) -> RetrievalResult:
        return finalize_retrieval_candidates(
            retrieval,
            ledger,
            self.kb,
            retrieval_mode=getattr(self.settings, "technique_retrieval_mode", "hybrid"),
            threshold=self.settings.similarity_threshold,
            acceptance_threshold=getattr(self.settings, "technique_acceptance_threshold", 0.75),
            relative_margin=getattr(self.settings, "technique_relative_margin", 0.08),
            reranker=self.reranker if self.reranker.available else None,
            acceptance_model=self.acceptance_model,
            rank_by_acceptance=bool(getattr(self.settings, "technique_rank_by_acceptance", False)),
            corroboration_accept_observations=int(
                getattr(self.settings, "technique_corroboration_accept_observations", 0)
            ),
            corroboration_accept_min_probability=float(
                getattr(self.settings, "technique_corroboration_accept_min_probability", 0.2)
            ),
            corroboration_bonus=float(getattr(self.settings, "technique_corroboration_bonus", 0.0)),
            segment_acceptance=bool(getattr(self.settings, "technique_segment_acceptance", False)),
            hierarchy_parent_mode=str(
                getattr(self.settings, "technique_hierarchy_parent_mode", "off")
            ),
        )

    def analyze(
        self,
        ledger: EvidenceLedger,
        *,
        weights: dict[str, float] | None = None,
        prior_verifications: dict[tuple[str, str], TechniqueVerification] | None = None,
    ) -> AnalysisState:
        retrieval, technique_verifications = self.map_techniques(
            ledger, prior_verifications=prior_verifications
        )
        context = build_classifier_context(
            self.kb,
            ledger,
            retrieval,
            telemetry_profile=self.telemetry_profile,
            soft_min_probability=float(
                getattr(self.settings, "attribution_soft_evidence_min_probability", 0.0)
            ),
            soft_weight=float(getattr(self.settings, "attribution_soft_evidence_weight", 1.0)),
            soft_max_groups=int(getattr(self.settings, "attribution_soft_evidence_max_groups", 0)),
            contrast_min_probability=float(
                getattr(self.settings, "attribution_contrast_min_probability", 0.0)
            ),
            contrast_weight=float(getattr(self.settings, "attribution_contrast_weight", 1.0)),
        )
        merged_weights = self.default_weights()
        merged_weights.update(weights or {})
        results, fusion, probabilities = self._score(context, merged_weights)
        state = AnalysisState(
            ledger=ledger,
            retrieval=retrieval,
            context=context,
            classifier_results=results,
            fusion=fusion,
            bayes_posterior=probabilities,
            origin_assessment=assess_origin_and_threat_type(self.kb, fusion),
            weights=merged_weights,
            technique_verifications=technique_verifications,
            telemetry_profile=self.telemetry_profile,
            deterministic_fusion=fusion,
        )
        state.attribution_decision = self._assess_stage(state)
        return state

    def _assess_stage(self, state: AnalysisState) -> AttributionDecision:
        """Stage decision from the deterministic fusion only (LLM votes never raise it)."""
        return assess_attribution(
            self.kb,
            state.deterministic_fusion or state.fusion,
            state.context,
            thresholds=self.attribution_thresholds,
            classifier_families=_CLASSIFIER_FAMILIES,
            discrimination=getattr(state.context, "discrimination", None),
            combiner=getattr(state.context, "combiner", None),
        )

    def set_telemetry_profile(self, profile: Any) -> None:
        """Set the per-case telemetry profile used for negative-answer validation.

        Negative interview answers are only discounted against a profile when
        ``TELEMETRY_VALIDATION_ENABLED`` is true; with the flag off every negative answer
        counts, exactly as when no profile has been supplied. The Streamlit UI does not
        build profiles yet, so this is an API/evaluator entry point.
        """
        if not bool(getattr(self.settings, "telemetry_validation_enabled", True)):
            self.telemetry_profile = None
            return
        self.telemetry_profile = profile

    def _verify_retrieval_candidates(
        self,
        ledger: EvidenceLedger,
        retrieval: RetrievalResult,
        *,
        exclude: set[tuple[str, str]] | None = None,
    ) -> tuple[RetrievalResult, dict[str, TechniqueCandidateVerificationResult]]:
        scope = str(getattr(self.settings, "technique_verifier_mode", "ambiguous"))
        if scope == "off" or not retrieval.candidates:
            return retrieval, {}
        observations = {observation.id: observation for observation in ledger.observations}
        by_observation: dict[str, list[Any]] = {}
        for candidate in retrieval.candidates:
            if exclude and (candidate.observation_id, candidate.technique_id) in exclude:
                continue  # verdict already on file
            by_observation.setdefault(candidate.observation_id, []).append(candidate)
        replacements: dict[tuple[str, str], Any] = {}
        results: dict[str, TechniqueCandidateVerificationResult] = {}
        for observation_id, candidates in by_observation.items():
            observation = observations.get(observation_id)
            if observation is None:
                continue
            result = verify_technique_candidates(
                self.llm,
                self.kb,
                observation,
                candidates,
                model=getattr(
                    self.settings,
                    "technique_verifier_model",
                    self.settings.extract_model,
                ),
                effort=getattr(self.settings, "technique_verifier_effort", "low"),
                scope=scope,
            )
            results[observation_id] = result
            for candidate in result.candidates:
                replacements[(candidate.observation_id, candidate.technique_id)] = candidate
        updated = tuple(
            replacements.get((candidate.observation_id, candidate.technique_id), candidate)
            for candidate in retrieval.candidates
        )
        diagnostics = dict(retrieval.diagnostics or {})
        diagnostics.update(
            {
                "verifier_reused": int(diagnostics.get("verifier_reused", 0) or 0),
                "verifier_mode": scope,
                # observations that produced a model request (an observation whose remaining
                # candidates need no verification costs nothing and is not a call)
                "verifier_calls": sum(1 for result in results.values() if result.prompt),
                "verifier_errors": sum(result.error is not None for result in results.values()),
                "verifier_cost_usd": sum(result.usage.cost_usd for result in results.values()),
            }
        )
        return replace(retrieval, candidates=updated, diagnostics=diagnostics), results

    def rescore(self, state: AnalysisState) -> AnalysisState:
        refresh_context_answers(
            state.context,
            state.ledger,
            kb=self.kb,
            telemetry_profile=getattr(state, "telemetry_profile", None),
        )
        merged_weights = self.default_weights()
        merged_weights.update(state.weights)
        results, fusion, probabilities = self._score(state.context, merged_weights)
        state.classifier_results = results
        state.fusion = fusion
        state.deterministic_fusion = fusion
        state.bayes_posterior = probabilities
        state.origin_assessment = assess_origin_and_threat_type(self.kb, fusion)
        if state.threat_intel_snapshot is not None:
            state.threat_intel_matches = match_threat_intel_to_case(
                self.kb, state, state.threat_intel_snapshot
            )
        state.current_question = None
        state.interview_stop_reason = None
        state.review = None
        state.attribution_decision = self._assess_stage(state)
        return state

    def attach_threat_intel(
        self, state: AnalysisState, snapshot: ThreatIntelSnapshot
    ) -> tuple[ThreatIntelMatch, ...]:
        state.threat_intel_snapshot = snapshot
        state.threat_intel_matches = match_threat_intel_to_case(self.kb, state, snapshot)
        return state.threat_intel_matches

    def import_observations(
        self, state: AnalysisState, observations: list[Observation] | tuple[Observation, ...]
    ) -> AnalysisState:
        state.ledger.extend_observations(observations)
        updated = self.analyze(
            state.ledger,
            weights=state.weights,
            prior_verifications=verifications_from_records(
                collect_verifications(getattr(state, "retrieval", None))
            ),
        )
        if state.threat_intel_snapshot is not None:
            self.attach_threat_intel(updated, state.threat_intel_snapshot)
        return updated

    def ensure_interview_current(self, state: AnalysisState) -> bool:
        if getattr(state, "interview_revision", 0) == INTERVIEW_REVISION:
            return False
        self.restart_interview(state)
        return True

    def restart_interview(self, state: AnalysisState) -> AnalysisState:
        state.ledger.question_answers.clear()
        state.context.observed_techniques = dict(state.retrieval.observed_techniques)
        state.context.technique_observations = dict(state.retrieval.technique_observations)
        state.context.negative_techniques = set()
        state.current_question = None
        state.interview_complete = False
        state.interview_stop_reason = None
        state.interview_revision = INTERVIEW_REVISION
        return self.rescore(state)

    def interview_hypothesis_weights(self, state: AnalysisState) -> dict[str, float]:
        entries = tuple(getattr(getattr(state, "fusion", None), "entries", ()))[:10]
        weights = {
            entry.group_id: (max(float(entry.relative_score), 0.0) / 100.0) ** 2
            for entry in entries
            if float(entry.relative_score) > 0
        }
        total = sum(weights.values())
        if total > 0:
            return {group_id: value / total for group_id, value in weights.items()}
        return dict(state.bayes_posterior)

    def question_shortlist(self, state: AnalysisState) -> list[QuestionCandidate]:
        answered = {
            answer.technique_id for answer in state.ledger.question_answers if answer.technique_id
        }
        answered.update(state.context.observed_techniques)
        return shortlist_questions(
            self.kb,
            self.interview_hypothesis_weights(state),
            answered,
            alpha=self.settings.nb_alpha,
            beta=self.settings.nb_beta,
            top_n=10,
        )

    def next_question(self, state: AnalysisState) -> SelectedQuestion | None:
        if state.interview_complete:
            return None
        candidates = self.question_shortlist(state)
        stop_reason = interview_stop_reason(
            self.interview_hypothesis_weights(state),
            candidates,
            len(state.ledger.question_answers),
            max_questions=self.settings.max_questions,
        )
        if stop_reason is not None:
            state.current_question = None
            state.interview_complete = True
            state.interview_stop_reason = stop_reason
            return None
        state.current_question = (
            select_question_locally(candidates)
            if self.llm is None
            else select_question_with_llm(
                self.llm,
                candidates,
                state.context.narrative,
                model=self.settings.extract_model,
                effort=self.settings.question_effort,
            )
        )
        state.interview_complete = state.current_question is None
        state.interview_stop_reason = "selection_failed" if state.current_question is None else None
        return state.current_question

    def answer_question(
        self,
        state: AnalysisState,
        status: EvidenceStatus | str,
        answer_text: str | None = None,
    ) -> AnalysisState:
        question = state.current_question
        if question is None:
            raise ValueError("No current question to answer")
        state.ledger.add_question_answer(
            QuestionAnswer(
                question=question.question,
                technique_id=question.technique_id,
                status=status,
                answer_text=answer_text,
                metadata={
                    "information_gain": question.information_gain,
                    "probability_yes": question.probability_yes,
                    "profile_support": question.profile_support,
                    "supporting_group_count": question.supporting_group_count,
                    "candidate_group_count": question.candidate_group_count,
                    "investigation_priority": question.investigation_priority,
                    "hypothesis": question.hypothesis,
                    "investigation_steps": list(question.investigation_steps),
                },
            )
        )
        return self.rescore(state)

    def run_cyberre(
        self,
        state: AnalysisState,
        observation_id: str,
        *,
        use_prefilter: bool | None = None,
    ) -> CyberREResult:
        observation = next(
            (
                item
                for item in state.ledger.observations
                if item.id == observation_id
                and item.status is EvidenceStatus.POSITIVE
                and item.kind not in {"ioc", "software"}
            ),
            None,
        )
        if observation is None:
            raise ValueError("CyberRE-LLM requires a positive action observation")
        prefilter_store = self._cyberre_prefilter_store(use_prefilter)
        result = run_cyberre_scan(
            self.llm,
            self.kb,
            observation,
            model=self.settings.cyberre_model,
            effort=self.settings.cyberre_effort,
            max_ranked_techniques=self.settings.cyberre_max_ranked_techniques,
            prefilter_store=prefilter_store,
            prefilter_top_k=getattr(self.settings, "cyberre_prefilter_top_k", 50),
        )
        state.cyberre_results[observation_id] = result
        self._aggregate_cyberre_results(state)
        return self.rescore(state).cyberre_results[observation_id]

    def run_cyberre_batch(
        self,
        state: AnalysisState,
        *,
        use_prefilter: bool | None = None,
        observation_ids: list[str] | None = None,
    ) -> dict[str, CyberREResult]:
        """Run CyberRE-LLM across multiple positive action observations.

        When ``observation_ids`` is omitted, every positive action observation
        that has not already been scanned is processed. Already-scanned
        observations are skipped unless explicitly listed.
        """
        candidates = [
            observation
            for observation in state.ledger.observations
            if observation.status is EvidenceStatus.POSITIVE
            and observation.kind not in {"ioc", "software"}
        ]
        if observation_ids is not None:
            wanted = set(observation_ids)
            candidates = [item for item in candidates if item.id in wanted]
        else:
            candidates = [item for item in candidates if item.id not in state.cyberre_results]
        if not candidates:
            return dict(state.cyberre_results)
        prefilter_store = self._cyberre_prefilter_store(use_prefilter)
        prefilter_top_k = getattr(self.settings, "cyberre_prefilter_top_k", 50)
        for observation in candidates:
            result = run_cyberre_scan(
                self.llm,
                self.kb,
                observation,
                model=self.settings.cyberre_model,
                effort=self.settings.cyberre_effort,
                max_ranked_techniques=self.settings.cyberre_max_ranked_techniques,
                prefilter_store=prefilter_store,
                prefilter_top_k=prefilter_top_k,
            )
            state.cyberre_results[observation.id] = result
        self._aggregate_cyberre_results(state)
        self.rescore(state)
        return dict(state.cyberre_results)

    def _cyberre_prefilter_store(self, use_prefilter: bool | None) -> Any:
        if use_prefilter is None:
            use_prefilter = bool(getattr(self.settings, "cyberre_prefilter_enabled", True))
        return self.store if use_prefilter else None

    def _aggregate_cyberre_results(self, state: AnalysisState) -> None:
        scores: dict[str, float] = {}
        observation_ids: dict[str, set[str]] = {}
        for scan in state.cyberre_results.values():
            for technique_id, score in scan.technique_scores().items():
                scores[technique_id] = max(scores.get(technique_id, 0.0), score)
                observation_ids.setdefault(technique_id, set()).add(scan.observation_id)
        state.context.cyberre_techniques = scores
        state.context.cyberre_technique_observations = {
            technique_id: tuple(sorted(ids)) for technique_id, ids in observation_ids.items()
        }

    def run_review(self, state: AnalysisState, top_n: int = 10) -> ReviewOutcome:
        review_results = tuple(
            result
            for result in state.classifier_results
            if result.classifier != CYBERRE_CLASSIFIER_NAME
        )
        # The reviewer must not see an LLM vote: pass the deterministic fusion, not a ranking
        # that includes CyberRE-LLM, alongside the LLM-free classifier results.
        state.review = run_review_pipeline(
            self.llm,
            self.kb,
            state.ledger,
            review_results,
            state.deterministic_fusion or state.fusion,
            model=self.settings.review_model,
            effort=self.settings.review_effort,
            top_n=top_n,
            audit_dir=self.settings.audit_dir,
        )
        return state.review

    def _fuse(
        self,
        results: tuple[ClassifierResult, ...],
        weights: dict[str, float],
        eligible_groups: set[str] | None,
        context: ClassifierContext,
    ) -> FusionResult:
        """The single fusion path: every ranking the engine produces goes through it.

        Applies the decisive-evidence weights, family-aware RRF, then the candidate-relative
        discrimination (stored on the context for the stage decision and the UI) and, when
        ``ATTRIBUTION_RERANK_BLEND`` > 0, the discriminative re-ordering of the top-K.
        """
        fused_weights, fused_family_weights = self._decisive_evidence_weights(context, weights)
        fusion = reciprocal_rank_fusion(
            results,
            fused_weights,
            self.settings.rrf_k,
            eligible_group_ids=eligible_groups,
            classifier_families=_CLASSIFIER_FAMILIES,
            family_weights=fused_family_weights,
            tie_breaker=getattr(self.kb, "prominence", None),
        )
        discrimination = discriminate(
            self.kb,
            fusion,
            context,
            self.discrimination_weights,
            top_k=int(getattr(self.settings, "attribution_separation_top_k", 5)),
            floor=float(getattr(self.settings, "attribution_separation_floor", 1.0)),
            tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
            behavior_threshold=float(self.settings.behavior_similarity_threshold),
            soft_techniques=getattr(context, "contrast_techniques", None) or None,
        )
        context.discrimination = discrimination
        if self.combiner is not None and fusion.entries:
            # Learned ordering of the shortlist: features from the same inputs RRF used, then
            # the discrimination is recomputed on the new order so the gate scores the leader
            # it will see (leader-consistency guard). The discriminative blend is skipped: its
            # channels are combiner features already.
            top_n = int(getattr(self.settings, "attribution_combiner_top_n", 20))
            basis = discriminate(
                self.kb,
                fusion,
                context,
                self.discrimination_weights,
                top_k=top_n,
                floor=float(getattr(self.settings, "attribution_separation_floor", 1.0)),
                tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
                behavior_threshold=float(self.settings.behavior_similarity_threshold),
                soft_techniques=getattr(context, "contrast_techniques", None) or None,
            )
            inputs = combiner_inputs(
                results,
                fusion,
                basis,
                context,
                self.kb,
                top_n=top_n,
                weights=fused_weights,
                tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
            )
            probabilities, null_probability = self.combiner.probabilities(inputs)
            rrf_ranks = {entry.group_id: int(entry.rank) for entry in fusion.entries[:top_n]}
            fusion = apply_combiner_order(
                fusion,
                probabilities,
                blend=float(getattr(self.settings, "attribution_combiner_blend", 1.0)),
                top_n=top_n,
            )
            leader = fusion.entries[0].group_id if fusion.entries else None
            context.combiner = {
                "status": self.combiner_status,
                "top_n": top_n,
                "probabilities": {gid: round(p, 4) for gid, p in probabilities.items()},
                "leader_probability": (
                    round(probabilities.get(leader, 0.0), 4) if leader is not None else None
                ),
                "null_probability": (
                    round(null_probability, 4) if null_probability is not None else None
                ),
                "rrf_ranks": rrf_ranks,
            }
            # The gate compares the new leader against the union of both shortlists (the
            # RRF top-K and the combiner top-K): a competitor the re-ordering pushed down must
            # still count as shared evidence, or a promoted leader could look falsely exclusive.
            gate_k = int(getattr(self.settings, "attribution_separation_top_k", 5))
            rrf_top = [gid for gid, rank in sorted(rrf_ranks.items(), key=lambda i: i[1])][:gate_k]
            combined_top = [entry.group_id for entry in fusion.entries[:gate_k]]
            gate_ids = []
            for gid in combined_top + rrf_top:
                if gid not in gate_ids:
                    gate_ids.append(gid)
            by_id = {entry.group_id: entry for entry in fusion.entries}
            gate_fusion = replace(
                fusion, entries=tuple(by_id[gid] for gid in gate_ids if gid in by_id)
            )
            context.discrimination = discriminate(
                self.kb,
                gate_fusion,
                context,
                self.discrimination_weights,
                top_k=len(gate_ids),
                floor=float(getattr(self.settings, "attribution_separation_floor", 1.0)),
                tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
                behavior_threshold=float(self.settings.behavior_similarity_threshold),
                soft_techniques=getattr(context, "contrast_techniques", None) or None,
            )
            return fusion
        context.combiner = None
        blend = float(getattr(self.settings, "attribution_rerank_blend", 0.5))
        if blend > 0:
            # The re-ranking may contrast a longer shortlist than the naming gate scores
            # (ATTRIBUTION_RERANK_TOP_K); 0 means "same shortlist as the gate".
            rerank_k = int(getattr(self.settings, "attribution_rerank_top_k", 0) or 0)
            rerank_basis = discrimination
            if rerank_k > 0 and rerank_k != discrimination.top_k:
                rerank_basis = discriminate(
                    self.kb,
                    fusion,
                    context,
                    self.discrimination_weights,
                    top_k=rerank_k,
                    floor=float(getattr(self.settings, "attribution_separation_floor", 1.0)),
                    tool_weight_cap=float(getattr(self.settings, "software_tool_weight_cap", 1.5)),
                    behavior_threshold=float(self.settings.behavior_similarity_threshold),
                    soft_techniques=getattr(context, "contrast_techniques", None) or None,
                )
            fusion = apply_discriminative_rerank(
                fusion,
                rerank_basis,
                blend,
                float(getattr(self.settings, "attribution_separation_floor", 1.0)),
            )
        return fusion

    @property
    def discrimination_weights(self) -> DiscriminationWeights:
        return DiscriminationWeights(
            technique=float(getattr(self.settings, "attribution_separation_technique_weight", 1.0)),
            behavior=float(getattr(self.settings, "attribution_separation_behavior_weight", 2.0)),
            software=float(getattr(self.settings, "attribution_separation_software_weight", 1.0)),
            negative=float(getattr(self.settings, "attribution_separation_negative_weight", 0.3)),
            soft=float(getattr(self.settings, "attribution_separation_soft_weight", 1.0)),
        )

    def _decisive_evidence_weights(
        self, context: ClassifierContext, weights: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Scale the software vote when the case names a bespoke implant (see
        ``decisive_software_factor``); the factor is recorded on the context for display."""
        family_weights = self.family_weights()
        factor, triggers = decisive_software_factor(
            self.kb,
            set(getattr(context, "software_ids", ()) or ()),
            boost=float(getattr(self.settings, "software_decisive_boost", 0.0)),
        )
        context.decisive_software = {"factor": factor, "software_ids": list(triggers)}
        if factor <= 1.0:
            return dict(weights), family_weights
        boosted = dict(weights)
        boosted["software_tools"] = float(boosted.get("software_tools", 1.0)) * factor
        family_weights["software"] = float(family_weights.get("software", 1.0)) * factor
        return boosted, family_weights

    def family_weights(self) -> dict[str, float]:
        """Per-family RRF budgets from settings (single definition for every fusion call)."""
        return {
            "technique_overlap": self.settings.technique_family_weight,
            "context": self.settings.context_family_weight,
            "motivation": self.settings.motivation_family_weight,
            "actor_behavior": getattr(self.settings, "actor_behavior_family_weight", 1.50),
            "software": getattr(self.settings, "software_family_weight", 1.20),
            "campaign": getattr(self.settings, "campaign_family_weight", 1.00),
            "cyberre_llm": getattr(self.settings, "cyberre_family_weight", 0.35),
        }

    def default_weights(self) -> dict[str, float]:
        return {
            "campaign_profile": self.settings.campaign_weight,
            "group_description": self.settings.group_description_weight,
            "semantic_centroid": self.settings.semantic_centroid_weight,
            "specific_behavior": self.settings.specific_behavior_weight,
            "strategic_context": self.settings.strategic_context_weight,
            "motivation_alignment": self.settings.motivation_weight,
            CYBERRE_CLASSIFIER_NAME: self.settings.cyberre_weight,
            "software_tools": self.settings.software_weight,
            "bernoulli_naive_bayes": getattr(self.settings, "naive_bayes_weight", 1.0),
            "idf_tfidf_cosine": getattr(self.settings, "idf_overlap_weight", 1.0),
            "weighted_jaccard": getattr(self.settings, "jaccard_weight", 1.0),
        }

    def _score(
        self, context: ClassifierContext, weights: dict[str, float]
    ) -> tuple[tuple[ClassifierResult, ...], FusionResult, dict[str, float]]:
        # Eligibility is derived from this pass's technical-evidence results and applied
        # to fusion only. It must not constrain the classifiers themselves: a context is
        # reused across rescoring passes, and intersecting with the previous eligible set
        # would let the candidate pool only ever shrink.
        context.eligible_group_ids = None
        results = run_classifiers(context, self.classifiers)
        eligible_groups: set[str] = set()
        reasons: dict[str, tuple[str, ...]] = {}
        for result in results:
            if not result.applicable or result.classifier not in _TECHNICAL_EVIDENCE_CLASSIFIERS:
                continue
            for entry in result.entries:
                eligible_groups.add(entry.group_id)
                existing = reasons.setdefault(entry.group_id, ())
                if entry.classifier not in existing:
                    reasons[entry.group_id] = (*existing, entry.classifier)
        context.eligible_group_ids = eligible_groups
        context.group_eligibility_reasons = reasons
        fusion = self._fuse(results, weights, eligible_groups, context)
        bayes = next(
            (result for result in results if result.classifier == "bernoulli_naive_bayes"),
            None,
        )
        probabilities = (
            posterior(bayes, self.settings.posterior_temperature) if bayes is not None else {}
        )
        return results, fusion, probabilities
