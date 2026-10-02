"""Headless case service: the engine, the case store and every analyst operation.

The HTTP API (and any other front end) drives the analysis through this class so behaviour
cannot drift between interfaces: one place builds the evidence ledger (LLM extraction in
production mode, sentence split in local mode), runs ``AttributionEngine.analyze``, keeps what
is still valid across re-analyses (live-intel matches, CyberRE scans), applies technique
decisions and interview answers, and persists the ledger after every change. State lives in
memory per case id and is rebuilt from the saved ledger on demand.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.canvas import build_analyst_canvas, build_candidate_dossier
from apt_classifier.cases import CaseRecord, CaseStore
from apt_classifier.confidence_labels import confidence_label, load_confidence_labels
from apt_classifier.config import Settings, get_settings
from apt_classifier.diamond import build_diamond_model
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evidence.extract import extract_evidence
from apt_classifier.evidence.local_extract import local_observations
from apt_classifier.evidence.models import (
    EvidenceLedger,
    EvidenceStatus,
    StrategicContext,
    TechniqueReviewAction,
    TechniqueReviewDecision,
)
from apt_classifier.llm import OpenRouterClient
from apt_classifier.platforms import ledger_platforms, normalize_platforms, set_ledger_platforms
from apt_classifier.questions import parse_interview_answer
from apt_classifier.report import export_document
from apt_classifier.separation import discriminate, distinguishing_techniques
from apt_classifier.session import (
    AnalysisState,
    AttributionEngine,
    collect_verifications,
    verifications_from_records,
)
from apt_classifier.threat_intel import PublicThreatIntelClient, ThreatIntelSnapshot
from apt_classifier.top3_view import build_top3_view, top3_summary
from apt_classifier.vectorstore import ChromaVectorStore

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "case_observation.txt"
INTEL_CACHE_SECONDS = 1800.0


class UploadedBlob:
    """Minimal file-like object the structured-evidence parser understands."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buffer = io.BytesIO(data)

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buffer.seek(offset, whence)


@dataclass
class CaseMeta:
    case_id: str
    name: str
    created_at: str | None = None
    updated_at: str | None = None
    ingestion: dict[str, Any] = field(default_factory=dict)
    owner: dict[str, Any] | None = None  # {"id", "name", "email"} of the creating user

    @property
    def owner_id(self) -> str | None:
        return str(self.owner.get("id")) if self.owner and self.owner.get("id") else None


@dataclass
class IntakePayload:
    name: str = ""
    narrative: str = ""
    phases: dict[str, str] = field(default_factory=dict)
    alerts: str = ""
    iocs: str = ""
    files: list[UploadedBlob] = field(default_factory=list)
    target_geographies: str = ""
    target_industries: str = ""
    motivations: str = ""
    objectives: str = ""
    suspected_origins: str = ""
    # Target OS: every operating system the activity touched (Windows, Linux, macOS, ESXi); empty =
    # unknown. Techniques documented only for other operating systems are not mapped.
    target_platforms: str = ""

    def free_text(self) -> str:
        parts = [self.narrative.strip()] if self.narrative.strip() else []
        for phase, value in self.phases.items():
            if value and value.strip():
                parts.append(f"{phase}:\n{value.strip()}")
        return "\n\n".join(parts)

    def context(self) -> StrategicContext:
        return StrategicContext(
            target_geographies=self.target_geographies,
            target_industries=self.target_industries,
            motivations=self.motivations,
            objectives=self.objectives,
            suspected_origins=self.suspected_origins,
        )

    def has_input(self) -> bool:
        return bool(self.free_text() or self.alerts.strip() or self.iocs.strip() or self.files)


class CaseNotFound(KeyError):
    pass


class ServiceUnavailable(RuntimeError):
    """An operation needs OpenRouter (or another external service) that is not configured."""


def llm_available(settings: Settings) -> bool:
    try:
        settings.require_api_key()
    except RuntimeError:
        return False
    return True


class CaseService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine: AttributionEngine | None = None,
        cases_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        if engine is None:
            kb = AttackKnowledgeBase.load(self.settings.kb_path)
            embedder = SentenceTransformerEmbedder.from_settings(self.settings)
            store = ChromaVectorStore(self.settings, embedder)
            client = OpenRouterClient(self.settings) if llm_available(self.settings) else None
            engine = AttributionEngine(self.settings, kb, store, client)  # type: ignore[arg-type]
        self.engine = engine
        # Calibrated confidence labels for the Top-three view (display only): pinned to the
        # combiner the engine loaded, so they read exactly the probabilities it records.
        self.confidence_labels, self.confidence_labels_status = load_confidence_labels(
            self.settings,
            getattr(engine, "kb", None),
            getattr(engine, "combiner", None),
            getattr(getattr(engine, "settings", None), "attribution_combiner_path", None),
        )
        self.store = CaseStore(cases_dir or (self.settings.data_dir / "cases"))
        self._states: dict[str, AnalysisState] = {}
        self._meta: dict[str, CaseMeta] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()
        # The engine shares one embedder and one vector-store client; analyses are serialised.
        self._engine_lock = threading.RLock()
        self._intel_cache: dict[tuple[int, bool], tuple[float, ThreatIntelSnapshot]] = {}
        self._elastic_batches: dict[str, Any] = {}

    # ----- system -----
    @property
    def local_mode(self) -> bool:
        return getattr(self.engine, "llm", None) is None

    def system_info(self) -> dict[str, Any]:
        settings = self.engine.settings
        usage = getattr(getattr(self.engine, "llm", None), "total_usage", None)
        index: dict[str, Any]
        try:
            counts = self.engine.store.health()["counts"]
            index = {
                "ok": True,
                "technique_sentences": counts.technique_sentences,
                "group_documents": counts.group_documents,
                "actor_behavior_documents": counts.actor_behavior_documents,
            }
        except Exception as exc:  # noqa: BLE001 - health is informational
            index = {"ok": False, "error": str(exc)[:200]}
        status = getattr(self.engine, "acceptance_model_status", None)
        forest = getattr(self.engine, "profile_forest", None)
        # the device can change after loading (CPU fallback when the GPU fills up)
        forest_status = (
            f"active ({forest.device})"
            if forest is not None and hasattr(forest, "device")
            else str(getattr(self.engine, "profile_forest_status", "unavailable"))
        )
        return {
            "mode": "local" if self.local_mode else "production",
            "attack_version": str(getattr(self.engine.kb, "version", "") or ""),
            "groups": len(getattr(self.engine.kb, "groups", {}) or {}),
            "techniques": len(getattr(self.engine.kb, "techniques", {}) or {}),
            "embedding_model": settings.embedding_model,
            "extract_model": settings.extract_model,
            "review_model": settings.review_model,
            "preset": str(getattr(settings, "technique_mapping_preset", "balanced")),
            "gate_mode": str(getattr(settings, "attribution_gate_mode", "separation")),
            "acceptance_model": status
            if isinstance(status, str | dict | type(None))
            else str(status),
            "combiner": str(getattr(self.engine, "combiner_status", "unavailable")),
            "confidence_labels": str(self.confidence_labels_status),
            "exemplars": str(getattr(self.engine, "exemplar_status", "unavailable")),
            "profile_forest": forest_status,
            "group_mlp": str(getattr(self.engine, "group_mlp_status", "unavailable")),
            "index": index,
            "llm_spend_usd": round(float(getattr(usage, "cost_usd", 0.0) or 0.0), 4),
            "cases": len(self.store.list()),
            "elastic_mcp": bool(getattr(settings, "elastic_mcp_url", None)),
            "classifier_weights": self.engine.default_weights(),
        }

    def sample_text(self) -> str:
        return SAMPLE_PATH.read_text(encoding="utf-8") if SAMPLE_PATH.is_file() else ""

    def search_techniques(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        needle = query.strip().casefold()
        techniques = getattr(self.engine.kb, "techniques", {}) or {}
        rows = []
        for technique_id, technique in techniques.items():
            name = str(getattr(technique, "name", "") or "")
            if not needle or needle in technique_id.casefold() or needle in name.casefold():
                rows.append({"id": technique_id, "name": name})
        rows.sort(key=lambda row: (not row["id"].casefold().startswith(needle), row["id"]))
        return rows[: max(1, limit)]

    # ----- helpers -----
    def _lock(self, case_id: str) -> threading.RLock:
        with self._global_lock:
            return self._locks.setdefault(case_id, threading.RLock())

    def _build_ledger(self, payload: IntakePayload, diagnostics: dict[str, Any]) -> EvidenceLedger:
        context = payload.context()
        files = list(payload.files) or None
        if self.local_mode:
            ledger = extract_evidence(
                None,
                iocs=payload.iocs or None,
                structured_files=files,
                structured_diagnostics=diagnostics,
                kb=self.engine.kb,
                strategic_context=context,
            )
            ledger.observations.extend(local_observations(payload.free_text()))
            ledger.observations.extend(local_observations(payload.alerts, source="alerts"))
            diagnostics["normalization"] = "local_sentence_split"
            return ledger
        return extract_evidence(
            self.engine.llm,
            free_text=payload.free_text() or None,
            alerts=payload.alerts or None,
            structured_files=files,
            structured_diagnostics=diagnostics,
            iocs=payload.iocs or None,
            kb=self.engine.kb,
            source_names={"alerts": "analyst_input"},
            strategic_context=context,
        )

    def _reanalyze(
        self,
        ledger: EvidenceLedger,
        weights: dict[str, float] | None,
        previous: AnalysisState | None,
        *,
        prior_verifications: dict[tuple[str, str], Any] | None = None,
    ) -> AnalysisState:
        if prior_verifications is None and previous is not None:
            prior_verifications = verifications_from_records(
                collect_verifications(previous.retrieval)
            )
        with self._engine_lock:
            state = self.engine.analyze(
                ledger,
                weights=weights or None,
                prior_verifications=prior_verifications or None,
            )
            if previous is not None:
                if previous.threat_intel_snapshot is not None:
                    self.engine.attach_threat_intel(state, previous.threat_intel_snapshot)
                live = {observation.id for observation in ledger.observations}
                kept = {
                    oid: result
                    for oid, result in (previous.cyberre_results or {}).items()
                    if oid in live
                }
                if kept:
                    state.cyberre_results = kept
                    self.engine._aggregate_cyberre_results(state)  # noqa: SLF001
                    state = self.engine.rescore(state)
        return state

    @staticmethod
    def summary_for(state: AnalysisState) -> dict[str, Any]:
        decision = getattr(state, "attribution_decision", None)
        leader = state.fusion.entries[0].group_id if state.fusion.entries else None
        return {
            "observations": len(state.ledger.observations),
            "stage": decision.stage.value if decision else None,
            "leading_group": decision.leading_group_id if decision else leader,
            "named_cluster": list(getattr(decision, "named_cluster", ()) or ()) if decision else [],
            "accepted_techniques": len(getattr(state.retrieval, "observed_techniques", {}) or {}),
            "answers": len(state.ledger.question_answers),
        }

    def _save(self, case_id: str) -> None:
        state = self._states[case_id]
        meta = self._meta[case_id]
        self.store.save(
            case_id=case_id,
            name=meta.name,
            ledger=state.ledger,
            weights=state.weights,
            summary=self.summary_for(state),
            verifications=collect_verifications(state.retrieval),
            owner=meta.owner,
        )
        record = next((r for r in self.store.list() if r.case_id == case_id), None)
        if record is not None:
            meta.created_at, meta.updated_at = record.created_at, record.updated_at

    # ----- cases -----
    def list_cases(self) -> list[CaseRecord]:
        return self.store.list()

    def create_case(self, payload: IntakePayload, *, owner: dict[str, Any] | None = None) -> str:
        if not payload.has_input():
            raise ValueError("add at least one observation, alert, indicator or evidence file")
        # validated before extraction so a typo never costs an LLM call
        platforms = normalize_platforms(payload.target_platforms, strict=True)
        diagnostics: dict[str, Any] = {}
        with self._engine_lock:
            ledger = self._build_ledger(payload, diagnostics)
        if not ledger.observations:
            raise ValueError(
                "nothing usable was extracted: add concrete behaviour (commands, files, registry "
                "keys, network activity) or a log file"
            )
        case_id = uuid4().hex[:12]
        ledger.case_id = case_id
        set_ledger_platforms(ledger, platforms)
        state = self._reanalyze(ledger, None, None)
        name = payload.name.strip()[:120] or f"case-{case_id[:8]}"
        with self._lock(case_id):
            self._states[case_id] = state
            self._meta[case_id] = CaseMeta(
                case_id=case_id, name=name, ingestion=diagnostics, owner=dict(owner or {}) or None
            )
            self._save(case_id)
        return case_id

    def get(self, case_id: str) -> tuple[AnalysisState, CaseMeta]:
        with self._lock(case_id):
            if case_id not in self._states:
                try:
                    payload = self.store.load(case_id)
                except (FileNotFoundError, ValueError) as exc:
                    raise CaseNotFound(case_id) from exc
                saved_records = list(payload.get("verifications") or [])
                state = self._reanalyze(
                    payload["ledger"],
                    payload.get("weights") or None,
                    None,
                    prior_verifications=verifications_from_records(saved_records),
                )
                self._states[case_id] = state
                self._meta[case_id] = CaseMeta(
                    case_id=case_id,
                    name=payload["name"],
                    created_at=payload.get("created_at"),
                    updated_at=payload.get("updated_at"),
                    owner=payload.get("owner"),
                )
                # A case saved before verdict persistence (or with new candidates) just paid
                # for verifier calls: keep those verdicts so the next reopen is free.
                fresh = collect_verifications(state.retrieval)
                new_verdicts = {(r["observation_id"], r["technique_id"]) for r in fresh} - {
                    (r.get("observation_id"), r.get("technique_id")) for r in saved_records
                }
                if new_verdicts:
                    self._save(case_id)
            return self._states[case_id], self._meta[case_id]

    def rename(self, case_id: str, name: str) -> CaseMeta:
        _, meta = self.get(case_id)
        with self._lock(case_id):
            meta.name = name.strip()[:120] or meta.name
            self._save(case_id)
        return meta

    def delete(self, case_id: str) -> bool:
        with self._lock(case_id):
            self._states.pop(case_id, None)
            self._meta.pop(case_id, None)
            return self.store.delete(case_id)

    def rescore(self, case_id: str, weights: dict[str, float]) -> AnalysisState:
        state, _ = self.get(case_id)
        defaults = self.engine.default_weights()
        cleaned = {
            name: max(0.0, float(value))
            for name, value in weights.items()
            if name in defaults and value is not None
        }
        with self._lock(case_id):
            updated = self._reanalyze(state.ledger, cleaned, state)
            self._states[case_id] = updated
            self._save(case_id)
        return updated

    # ----- target operating systems -----
    def set_target_platforms(self, case_id: str, platforms: Any) -> AnalysisState:
        """Change the case's Target OS and map the observations again.

        Techniques documented only for other operating systems leave the candidate pools and
        compatible ones take their place. Saved verifier verdicts are reused, so in production
        mode only candidates that are new to the case are sent to the verifier.
        """
        values = normalize_platforms(platforms, strict=True)
        state, _ = self.get(case_id)
        if values == ledger_platforms(state.ledger):
            return state
        with self._lock(case_id):
            set_ledger_platforms(state.ledger, values)
            updated = self._reanalyze(state.ledger, state.weights, state)
            self._states[case_id] = updated
            self._save(case_id)
        return updated

    # ----- technique review -----
    def apply_decisions(self, case_id: str, decisions: list[dict[str, Any]]) -> AnalysisState:
        state, _ = self.get(case_id)
        techniques = getattr(self.engine.kb, "techniques", {})
        parsed: list[TechniqueReviewDecision] = []
        for item in decisions:
            action = str(item.get("action", "")).lower()
            if action not in {"confirm", "reject", "replace"}:
                continue
            replacement = str(item.get("replacement") or "").strip().upper() or None
            if action == "replace" and replacement not in techniques:
                raise ValueError(f"{replacement or '(empty)'} is not an ATT&CK technique id")
            parsed.append(
                TechniqueReviewDecision(
                    observation_id=str(item["observation_id"]),
                    technique_id=str(item["technique_id"]),
                    action=TechniqueReviewAction(action),
                    replacement_technique_id=replacement if action == "replace" else None,
                    rationale=(str(item.get("rationale") or "").strip() or None),
                )
            )
        if not parsed:
            return state
        with self._lock(case_id):
            for decision in parsed:
                state.ledger.record_technique_review(decision)
            updated = self._reanalyze(state.ledger, state.weights, state)
            self._states[case_id] = updated
            self._save(case_id)
        return updated

    # ----- interview -----
    def interview(self, case_id: str) -> dict[str, Any]:
        state, _ = self.get(case_id)
        with self._lock(case_id), self._engine_lock:
            self.engine.ensure_interview_current(state)
            selection_error: str | None = None
            if (
                state.current_question is None
                and not state.interview_complete
                and state.fusion.entries
            ):
                try:
                    self.engine.next_question(state)
                except Exception as exc:  # noqa: BLE001 - reported, not fatal
                    selection_error = f"{type(exc).__name__}: {exc}"[:300]
            question = state.current_question
            leads = self.engine.question_shortlist(state) if state.fusion.entries else []
            weights = self.engine.interview_hypothesis_weights(state)
            return {
                "complete": bool(state.interview_complete),
                "stop_reason": state.interview_stop_reason,
                "selection_error": selection_error,
                "answered": [
                    {
                        "question": a.question,
                        "technique_id": a.technique_id,
                        "status": a.status.value,
                        "text": a.answer_text,
                    }
                    for a in state.ledger.question_answers
                ],
                "max_questions": int(self.engine.settings.max_questions),
                "question": (
                    {
                        "technique_id": question.technique_id,
                        "question": question.question,
                        "hypothesis": question.hypothesis,
                        "investigation_steps": list(question.investigation_steps),
                        "profile_support": question.profile_support,
                        "supporting_group_count": question.supporting_group_count,
                        "candidate_group_count": question.candidate_group_count,
                        "probability_yes": question.probability_yes,
                        "information_gain": question.information_gain,
                        "investigation_priority": question.investigation_priority,
                    }
                    if question is not None
                    else None
                ),
                "leads": [
                    {
                        "technique_id": lead.technique_id,
                        "technique_name": lead.technique_name,
                        "priority": lead.investigation_priority,
                        "profile_support": lead.profile_support,
                        "supporting_group_count": lead.supporting_group_count,
                        "candidate_group_count": lead.candidate_group_count,
                        "first_step": lead.investigation_steps[0]
                        if lead.investigation_steps
                        else "",
                    }
                    for lead in leads[:8]
                ],
                "weights": [
                    {"group_id": gid, "weight": weights[gid]}
                    for gid in sorted(weights, key=lambda g: -weights[g])
                ],
            }

    def answer(self, case_id: str, status: str, text: str | None = None) -> AnalysisState:
        state, _ = self.get(case_id)
        parsed = parse_interview_answer(status)
        if parsed is None:
            try:
                parsed = EvidenceStatus(status)
            except ValueError:
                raise ValueError("answer must be yes, no, unknown or not visible") from None
        with self._lock(case_id), self._engine_lock:
            if state.current_question is None:
                raise ValueError("there is no open question to answer")
            self.engine.answer_question(state, parsed, (text or "").strip() or None)
            try:
                self.engine.next_question(state)
            finally:
                self._save(case_id)
        return state

    def restart_interview(self, case_id: str) -> AnalysisState:
        state, _ = self.get(case_id)
        with self._lock(case_id), self._engine_lock:
            self.engine.restart_interview(state)
            self._save(case_id)
        return state

    # ----- LLM steps -----
    def _require_llm(self, what: str) -> None:
        if self.local_mode:
            raise ServiceUnavailable(
                f"{what} needs an OpenRouter key; the service runs in local mode"
            )

    def run_review(self, case_id: str) -> AnalysisState:
        self._require_llm("the independent AI review")
        state, _ = self.get(case_id)
        with self._lock(case_id), self._engine_lock:
            self.engine.run_review(state)
        return state

    def run_cyberre(
        self, case_id: str, observation_id: str | None, *, use_prefilter: bool | None = None
    ) -> AnalysisState:
        self._require_llm("CyberRE-LLM")
        state, _ = self.get(case_id)
        with self._lock(case_id), self._engine_lock:
            if observation_id:
                self.engine.run_cyberre(state, observation_id, use_prefilter=use_prefilter)
            else:
                self.engine.run_cyberre_batch(state, use_prefilter=use_prefilter)
            self._save(case_id)
        return state

    # ----- live threat intelligence -----
    def fetch_intel(
        self, *, days: int | None = None, include_claims: bool | None = None, refresh: bool = False
    ) -> ThreatIntelSnapshot:
        settings = self.engine.settings
        key = (
            int(days or settings.threat_intel_days),
            bool(settings.include_ransomware_claims if include_claims is None else include_claims),
        )
        cached = self._intel_cache.get(key)
        now = time.monotonic()
        if cached is not None and not refresh and now - cached[0] < INTEL_CACHE_SECONDS:
            return cached[1]
        client = PublicThreatIntelClient(timeout=float(settings.threat_intel_timeout))
        try:
            snapshot = client.fetch(
                days=key[0],
                include_ransomware_claims=key[1],
                max_items_per_source=int(settings.threat_intel_max_items_per_source),
            )
        finally:
            client.close()
        self._intel_cache[key] = (now, snapshot)
        return snapshot

    def attach_intel(self, case_id: str, snapshot: ThreatIntelSnapshot) -> AnalysisState:
        state, _ = self.get(case_id)
        with self._lock(case_id), self._engine_lock:
            self.engine.attach_threat_intel(state, snapshot)
        return state

    # ----- Elasticsearch (MCP) evidence gathering -----
    def elastic_search(self, case_id: str, lookback_hours: int | None = None) -> Any:
        from apt_classifier.elastic_mcp import ElasticMCPClient, elastic_mcp_configured

        if not elastic_mcp_configured(self.engine.settings):
            raise ServiceUnavailable("set ELASTIC_MCP_URL and ELASTIC_MCP_API_KEY to enable it")
        state, _ = self.get(case_id)
        question = state.current_question
        if question is None:
            raise ValueError("the interview has no open question to search for")
        hours = int(lookback_hours or self.engine.settings.elastic_mcp_lookback_hours)
        batch = ElasticMCPClient.from_settings(self.engine.settings).gather(
            question, lookback_hours=hours
        )
        self._elastic_batches[f"{case_id}:{question.technique_id}"] = batch
        return batch

    def elastic_import(self, case_id: str, candidate_ids: list[str]) -> AnalysisState:
        state, _ = self.get(case_id)
        question = state.current_question
        key = f"{case_id}:{question.technique_id}" if question else None
        batch = self._elastic_batches.get(key) if key else None
        if batch is None:
            raise ValueError("search first; no staged Elasticsearch results for this lead")
        by_id = {candidate.id: candidate for candidate in batch.candidates}
        chosen = [by_id[cid].to_observation() for cid in candidate_ids if cid in by_id]
        if not chosen:
            raise ValueError("no staged candidate matches the selected ids")
        with self._lock(case_id), self._engine_lock:
            updated = self.engine.import_observations(state, chosen)
            live = {o.id for o in updated.ledger.observations}
            kept = {
                oid: result for oid, result in (state.cyberre_results or {}).items() if oid in live
            }
            if kept:
                updated.cyberre_results = kept
                self.engine._aggregate_cyberre_results(updated)  # noqa: SLF001
                updated = self.engine.rescore(updated)
            self._states[case_id] = updated
            self._elastic_batches.pop(key, None)
            self._save(case_id)
        return updated

    # ----- distinguishing the shortlist -----
    def distinguish(
        self, case_id: str, *, top_k: int | None = None, limit: int = 12
    ) -> dict[str, Any]:
        """What separates the top-K, and which pending candidates or interview leads would
        change the separation the most (counterfactual replays of the discrimination).

        By default K is the naming gate's own window (the combiner's top five plus the RRF
        top five, 5-8 groups), replayed in the recorded order, so the separation shown equals
        the decision's."""
        state, _ = self.get(case_id)
        kb = self.engine.kb
        settings = self.engine.settings
        fusion = state.deterministic_fusion or state.fusion
        context = state.context
        recorded = getattr(context, "discrimination", None)
        recorded_order = [item.group_id for item in recorded.scores] if recorded is not None else []
        gate_k = len(recorded_order)
        k = max(2, min(int(top_k if top_k is not None else (gate_k or 10)), 15))
        weights = self.engine.discrimination_weights
        options = {
            "top_k": k,
            "floor": float(getattr(settings, "attribution_separation_floor", 1.0)),
            "tool_weight_cap": float(getattr(settings, "software_tool_weight_cap", 1.5)),
            "behavior_threshold": float(settings.behavior_similarity_threshold),
        }
        soft = dict(getattr(context, "contrast_techniques", {}) or {})
        # The stage decision scored the gate window in its recorded order (leader first; with the
        # combiner the window adds RRF top-five groups the combiner ranked lower). Whenever K
        # covers that window, replay it first, then the next groups in fusion order, so the
        # separation shown here equals the decision's at K = window size.
        by_id = fusion.by_group()
        if recorded_order and k >= gate_k and all(gid in by_id for gid in recorded_order):
            window = set(recorded_order)
            ordered = [by_id[gid] for gid in recorded_order] + [
                entry for entry in fusion.entries if entry.group_id not in window
            ]
        else:
            ordered = list(fusion.entries)
        basis = replace(fusion, entries=tuple(ordered))
        names = {
            gid: (g.name if g else gid)
            for gid, g in ((e.group_id, kb.groups.get(e.group_id)) for e in basis.entries[:k])
        }

        def replay(
            extra_observed: dict[str, float] | None = None,
            extra_negative: set[str] | None = None,
        ):
            observed = dict(context.observed_techniques)
            observed.update(extra_observed or {})
            negative = set(context.negative_techniques) | set(extra_negative or ())
            view = SimpleNamespace(
                observed_techniques=observed,
                negative_techniques=negative,
                parent_policy=context.parent_policy,
                specific_behavior_matches=getattr(context, "specific_behavior_matches", ()),
                software_ids=context.software_ids,
                decisive_software=getattr(context, "decisive_software", {}),
            )
            soft_view = {
                tid: v for tid, v in soft.items() if tid not in observed and tid not in negative
            }
            return discriminate(
                kb, basis, view, weights, soft_techniques=soft_view or None, **options
            )

        base = replay()
        rows = distinguishing_techniques(kb, basis, context, top_k=k, soft_techniques=soft or None)
        techniques = kb.techniques
        technique_name = lambda tid: getattr(techniques.get(tid), "name", "") or ""  # noqa: E731

        def delta(result) -> float:
            return round(result.separation - base.separation, 4)

        def strongest(result) -> str | None:
            return (
                max(result.scores, key=lambda c: (c.total, -c.rrf_score)).group_id
                if result.scores
                else None
            )

        # pending candidates, one row per technique (best probability), ranked by |delta|
        texts = {o.id: o.text for o in state.ledger.observations}
        best: dict[str, Any] = {}
        for candidate in getattr(state.retrieval, "candidates", ()) or ():
            if candidate.state.value != "needs_review":
                continue
            tid = candidate.technique_id
            if tid in context.observed_techniques or tid in context.negative_techniques:
                continue
            if candidate.evidence_for("hierarchy_parent") is not None:
                continue
            evidence = candidate.evidence_for("acceptance_model")
            probability = (
                float(evidence.score)
                if evidence is not None and evidence.score is not None
                else 0.0
            )
            if tid not in best or probability > best[tid][0]:
                best[tid] = (probability, candidate)
        pending = []
        for tid, (probability, candidate) in best.items():
            result = replay({tid: 1.0})
            row_match = next((r for r in rows if r.technique_id == tid), None)
            if row_match is None:
                # below the contrast floor: probe the technique's place in the shortlist
                probe = distinguishing_techniques(
                    kb, basis, context, top_k=k, soft_techniques={tid: 1.0}
                )
                row_match = next((r for r in probe if r.technique_id == tid), None)
            if row_match is None or not row_match.documented_by:
                continue
            documented_by = list(row_match.documented_by)
            exclusivity, specificity = row_match.exclusivity, row_match.specificity
            pending.append(
                {
                    "technique_id": tid,
                    "technique_name": candidate.technique_name or technique_name(tid),
                    "probability": round(probability, 3),
                    "observation_id": candidate.observation_id,
                    "observation": texts.get(candidate.observation_id, candidate.observation_id)[
                        :200
                    ],
                    "documented_by": [
                        {"group_id": gid, "name": names.get(gid, gid)} for gid in documented_by
                    ],
                    "exclusivity": round(exclusivity, 3),
                    "specificity": round(specificity, 3),
                    "delta_separation": delta(result),
                    "new_leader_id": strongest(result),
                    "new_leader_name": names.get(strongest(result) or "", strongest(result)),
                    "keeps_leader": strongest(result) in (None, base.leader_id),
                }
            )
        # expected change: the size of the move weighted by how likely the mapping is
        pending.sort(
            key=lambda item: (
                -abs(item["delta_separation"]) * max(item["probability"], 0.01),
                -item["probability"],
            )
        )

        # interview leads: what a yes or a no would do to the separation
        leads = []
        with self._lock(case_id), self._engine_lock:
            shortlist_leads = self.engine.question_shortlist(state) if fusion.entries else []
        for lead in shortlist_leads[:10]:
            yes = replay({lead.technique_id: 1.0})
            no = replay(None, {lead.technique_id})
            leads.append(
                {
                    "technique_id": lead.technique_id,
                    "technique_name": lead.technique_name,
                    "priority": round(float(lead.investigation_priority), 3),
                    "profile_support": round(float(lead.profile_support), 3),
                    "supporting_group_count": lead.supporting_group_count,
                    "candidate_group_count": lead.candidate_group_count,
                    "delta_yes": delta(yes),
                    "delta_no": delta(no),
                    "leader_after_yes": names.get(strongest(yes) or "", strongest(yes)),
                    "first_step": lead.investigation_steps[0] if lead.investigation_steps else "",
                }
            )
        leads.sort(key=lambda item: -max(abs(item["delta_yes"]), abs(item["delta_no"])))

        by_group = base.by_group()
        return {
            "top_k": k,
            "gate_top_k": gate_k,
            "separation": round(base.separation, 4),
            "leader_total": round(base.leader_total, 4),
            "leader": {
                "group_id": base.leader_id,
                "name": names.get(base.leader_id or "", base.leader_id),
            },
            "runner_up": {
                "group_id": base.runner_up_id,
                "name": names.get(base.runner_up_id or "", base.runner_up_id),
            },
            "runner_up_related": bool(base.runner_up_related),
            "thresholds": {
                "separation": float(getattr(settings, "attribution_hypothesis_separation", 0.5)),
                "min_discriminative": float(
                    getattr(settings, "attribution_hypothesis_min_discriminative", 3.0)
                ),
                "gate_mode": str(getattr(settings, "attribution_gate_mode", "separation")),
            },
            "contrast_enabled": bool(soft),
            "shortlist": [
                {
                    "group_id": e.group_id,
                    "name": names.get(e.group_id, e.group_id),
                    "rank": int(e.rank),
                    "relative_score": round(float(e.relative_score), 1),
                    **(
                        {
                            key: round(float(getattr(by_group[e.group_id], key)), 4)
                            for key in (
                                "technique",
                                "behavior",
                                "software",
                                "soft",
                                "negative",
                                "total",
                            )
                        }
                        if e.group_id in by_group
                        else {}
                    ),
                }
                for e in basis.entries[:k]
            ],
            "techniques": [
                {**row.to_dict(), "technique_name": technique_name(row.technique_id)}
                for row in rows
            ],
            "pending": pending[: max(1, limit)],
            "leads": leads[:8],
        }

    def top3_view(self, state: Any) -> dict[str, Any]:
        """Head-to-head of the decision's first three candidates (display only, docs 10.10)."""
        settings = self.engine.settings
        thresholds = self.engine.attribution_thresholds
        view = build_top3_view(
            self.engine.kb,
            state,
            floor=float(getattr(settings, "attribution_separation_floor", 1.0)),
            separation_bar=float(getattr(thresholds, "hypothesis_separation", 0.5)),
            min_discriminative=float(getattr(thresholds, "hypothesis_min_discriminative", 3.0)),
            behavior_threshold=float(settings.behavior_similarity_threshold),
        )
        view["confidence"] = self.confidence(state)
        return view

    def confidence(self, state: Any) -> dict[str, Any] | None:
        """Calibrated confidence label (display only, never a decision); None when off."""
        try:
            return confidence_label(self.confidence_labels, state, kb=self.engine.kb)
        except Exception:  # noqa: BLE001 - a secondary chip must not break the view
            return None

    def top3(self, case_id: str) -> dict[str, Any]:
        state, _ = self.get(case_id)
        return self.top3_view(state)

    def top3_summary(self, state: Any) -> dict[str, Any] | None:
        """State and one-line summary for the case payload; never fails the payload."""
        try:
            return top3_summary(self.top3_view(state))
        except Exception:  # noqa: BLE001 - an explanatory badge must not break a case view
            return None

    # ----- views and exports -----
    def dossier(self, case_id: str, group_id: str) -> dict[str, Any]:
        state, _ = self.get(case_id)
        if group_id not in getattr(self.engine.kb, "groups", {}):
            raise ValueError(f"{group_id} is not an ATT&CK group id")
        return build_candidate_dossier(self.engine.kb, state, group_id)

    def canvas(self, case_id: str, *, top_groups: int = 8, max_evidence: int = 12) -> Any:
        state, _ = self.get(case_id)
        return build_analyst_canvas(
            self.engine.kb,
            state,
            top_groups=max(1, min(int(top_groups), 15)),
            max_evidence=max(1, min(int(max_evidence), 40)),
        )

    def diamond(
        self,
        case_id: str,
        *,
        top_groups: int = 8,
        max_capabilities: int = 15,
        max_infrastructure: int = 15,
    ) -> Any:
        state, _ = self.get(case_id)
        return build_diamond_model(
            self.engine.kb,
            state,
            top_groups=max(1, min(int(top_groups), 15)),
            max_capabilities=max(1, min(int(max_capabilities), 40)),
            max_infrastructure=max(1, min(int(max_infrastructure), 40)),
        )

    def export(self, case_id: str, kind: str) -> tuple[str, str, str]:
        state, meta = self.get(case_id)
        return export_document(self.engine.kb, state, kind, case_name=meta.name)
