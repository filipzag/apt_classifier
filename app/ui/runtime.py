"""Per-session engine, analysis state and case persistence for the UI.

Heavy resources (knowledge base, embedder, vector store) are cached once per process.
The `AttributionEngine` itself is cheap to build and holds the *session's* settings, so it
lives in `st.session_state`: two analysts changing sliders no longer overwrite each other's
settings on a shared engine, and the LLM client (with its running cost) is carried over
when the engine is rebuilt after a settings change.
"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

import streamlit as st

from app.ui.cases import CaseStore
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.config import Settings, get_settings
from apt_classifier.embeddings import SentenceTransformerEmbedder
from apt_classifier.evidence.models import EvidenceLedger
from apt_classifier.llm import OpenRouterClient
from apt_classifier.session import AnalysisState, AttributionEngine
from apt_classifier.vectorstore import ChromaVectorStore

STATE_KEY = "analysis_state"
CASE_NAME_KEY = "case_name"
STALE_KEY = "results_stale"


@st.cache_resource(show_spinner="Loading ATT&CK-BERT and connecting to Chroma…")
def load_resources(kb_path: str, kb_mtime: int) -> tuple[AttackKnowledgeBase, ChromaVectorStore]:
    del kb_mtime  # part of the cache key only
    settings = get_settings()
    kb = AttackKnowledgeBase.load(kb_path)
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    return kb, ChromaVectorStore(settings, embedder)


def llm_available(settings: Settings) -> bool:
    try:
        settings.require_api_key()
    except RuntimeError:
        return False
    return True


def local_only(settings: Settings) -> bool:
    """True when no OpenRouter key is configured: the UI runs without any LLM step."""
    return not llm_available(settings)


def _settings_key(settings: Settings) -> str:
    return hashlib.sha256(settings.model_dump_json().encode("utf-8")).hexdigest()


def get_engine(settings: Settings) -> AttributionEngine:
    """The session's engine, rebuilt only when its settings change."""
    key = _settings_key(settings)
    cached = st.session_state.get("_engine")
    if cached is not None and cached[0] == key:
        return cached[1]
    kb, store = load_resources(str(settings.kb_path), settings.kb_path.stat().st_mtime_ns)
    previous_client = cached[1].llm if cached is not None else None
    client = previous_client
    if client is None and llm_available(settings):
        client = OpenRouterClient(settings)
    engine = AttributionEngine(settings, kb, store, client)  # type: ignore[arg-type]
    st.session_state["_engine"] = (key, engine)
    return engine


def current_state() -> AnalysisState | None:
    return st.session_state.get(STATE_KEY)


def set_state(state: AnalysisState | None, *, stale: bool = False) -> None:
    st.session_state[STATE_KEY] = state
    st.session_state[STALE_KEY] = stale


def mark_stale(value: bool = True) -> None:
    if current_state() is not None:
        st.session_state[STALE_KEY] = value


def case_name() -> str:
    name = str(st.session_state.get(CASE_NAME_KEY) or "").strip()
    if name:
        return name
    state = current_state()
    return f"case-{state.ledger.case_id[:8]}" if state is not None and state.ledger.case_id else ""


def new_case_id() -> str:
    return uuid4().hex[:12]


def case_store(settings: Settings) -> CaseStore:
    return CaseStore(settings.data_dir / "cases")


def summary_for(state: AnalysisState) -> dict[str, Any]:
    decision = getattr(state, "attribution_decision", None)
    return {
        "observations": len(state.ledger.observations),
        "stage": decision.stage.value if decision else None,
        "leading_group": decision.leading_group_id if decision else None,
        "accepted_techniques": len(getattr(state.retrieval, "observed_techniques", {}) or {}),
    }


def autosave(settings: Settings, state: AnalysisState) -> None:
    """Persist the ledger after every change that alters it; never raises into the page."""
    if not state.ledger.case_id:
        state.ledger.case_id = new_case_id()
    try:
        case_store(settings).save(
            case_id=state.ledger.case_id,
            name=case_name() or f"case-{state.ledger.case_id[:8]}",
            ledger=state.ledger,
            weights=state.weights,
            summary=summary_for(state),
        )
    except Exception as exc:  # noqa: BLE001 - saving must never break the analysis
        st.toast(f"Case was not saved to disk: {exc}", icon="⚠️")


def reanalyze(
    engine: AttributionEngine,
    ledger: EvidenceLedger,
    weights: dict[str, float],
    *,
    previous: AnalysisState | None,
) -> AnalysisState:
    """Full analysis of ``ledger`` that keeps what is still valid from ``previous``.

    Live threat-intel matches are re-attached, CyberRE scans for observations that still
    exist are kept and re-fused, the interview cursor is rebuilt by the engine. The
    independent AI review is dropped: it was written for the old ranking.
    """
    state = engine.analyze(ledger, weights=weights)
    if previous is not None:
        if previous.threat_intel_snapshot is not None:
            engine.attach_threat_intel(state, previous.threat_intel_snapshot)
        live_ids = {observation.id for observation in ledger.observations}
        kept = {
            observation_id: result
            for observation_id, result in (previous.cyberre_results or {}).items()
            if observation_id in live_ids
        }
        if kept:
            state.cyberre_results = kept
            engine._aggregate_cyberre_results(state)  # noqa: SLF001 - engine-internal fold-in
            state = engine.rescore(state)
    return state


def clear_case() -> None:
    for key in (STATE_KEY, CASE_NAME_KEY, STALE_KEY, "structured_ingestion_diagnostics"):
        st.session_state.pop(key, None)
    for key in [
        k
        for k in st.session_state
        if str(k).startswith(
            ("action_", "replace_", "rationale_", "elastic-", "cyberre-", "interview-", "phase_")
        )
    ]:
        st.session_state.pop(key, None)
