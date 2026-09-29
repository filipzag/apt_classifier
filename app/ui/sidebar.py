"""Sidebar: case manager, mode banner, preset and advanced settings."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app.ui import runtime
from app.ui.guard import run_step
from apt_classifier.config import Settings
from apt_classifier.presets import PRESETS, apply_preset, preset_choices
from apt_classifier.session import AttributionEngine, verifications_from_records

_HEALTH_TTL_SECONDS = 60.0


def _store_health(engine: AttributionEngine) -> dict[str, Any] | None:
    """Chroma health, refreshed at most once a minute per session (it was every rerun)."""
    cached = st.session_state.get("_store_health")
    now = time.monotonic()
    if cached is not None and now - cached[0] < _HEALTH_TTL_SECONDS:
        return cached[1]
    try:
        health = engine.store.health()
        counts = health["counts"]
        value: dict[str, Any] | None = {
            "technique_sentences": counts.technique_sentences,
            "group_documents": counts.group_documents,
            "actor_behavior_documents": counts.actor_behavior_documents,
        }
    except Exception as exc:  # noqa: BLE001
        value = {"error": str(exc)[:200]}
    st.session_state["_store_health"] = (now, value)
    return value


def _render_case_manager(
    settings: Settings, engine: AttributionEngine, weights: dict[str, float]
) -> None:
    store = runtime.case_store(settings)
    state = runtime.current_state()
    st.sidebar.markdown("#### Case")
    current_case_id = state.ledger.case_id if state is not None else "none"
    name = st.sidebar.text_input(
        "Case name",
        value=runtime.case_name(),
        key=f"case_name_input_{current_case_id}",
        placeholder="e.g. INC-2041 finance workstation",
        disabled=state is None,
        help="Saved with the evidence ledger under data/cases.",
    )
    if state is not None and name.strip() and name.strip() != runtime.case_name():
        st.session_state[runtime.CASE_NAME_KEY] = name.strip()
        runtime.autosave(settings, state)
    records = store.list()
    columns = st.sidebar.columns(2)
    if columns[0].button("New case", use_container_width=True, help="Start from empty evidence."):
        runtime.clear_case()
        st.rerun()
    if columns[1].button(
        "Save now",
        use_container_width=True,
        disabled=state is None,
        help="Cases are saved automatically after every analysis; this forces a save.",
    ):
        runtime.autosave(settings, state)
        st.sidebar.success("Saved.")
    if records:
        options = {record.case_id: record for record in records}
        current_id = state.ledger.case_id if state is not None else None
        chosen = st.sidebar.selectbox(
            "Open a saved case",
            options=list(options),
            format_func=lambda case_id: options[case_id].label,
            index=list(options).index(current_id) if current_id in options else 0,
            key="open_case_select",
        )
        open_col, delete_col = st.sidebar.columns(2)
        if open_col.button("Open", use_container_width=True, disabled=chosen == current_id):
            result = run_step("Opening the case", store.load, chosen)
            if result.ok:
                payload = result.value
                with st.spinner("Re-running the analysis on the saved evidence…"):
                    loaded = run_step(
                        "Analysing the saved evidence",
                        runtime.reanalyze,
                        engine,
                        payload["ledger"],
                        payload.get("weights") or weights,
                        previous=None,
                        prior_verifications=verifications_from_records(
                            payload.get("verifications")
                        ),
                    )
                if loaded.ok:
                    st.session_state[runtime.CASE_NAME_KEY] = payload["name"]
                    runtime.set_state(loaded.value)
                    st.rerun()
        with delete_col.popover("Delete", use_container_width=True):
            st.caption(f"Delete **{options[chosen].name}** from disk? This cannot be undone.")
            if st.button("Delete this case", type="primary", key="confirm_delete_case"):
                store.delete(chosen)
                if chosen == current_id:
                    runtime.clear_case()
                st.rerun()
    else:
        st.sidebar.caption("No saved cases yet. Cases are saved automatically after analysis.")


def render_sidebar(
    settings: Settings, engine: AttributionEngine
) -> tuple[Settings, dict[str, float], bool]:
    """Returns (settings with sidebar overrides, classifier weights, rescore requested)."""
    st.sidebar.markdown("### APT Attribution Assistant")
    if runtime.local_only(settings):
        st.sidebar.warning(
            "**Local mode.** No OpenRouter key is configured: free text is split into sentences "
            "locally, and the verifier, LLM vote, CyberRE and AI review are unavailable. "
            "Retrieval, gating and ranking are unaffected.",
            icon="🔌",
        )
    _render_case_manager(settings, engine, engine.default_weights())
    st.sidebar.divider()
    st.sidebar.markdown("#### Analysis settings")
    preset_keys = list(preset_choices().keys())
    current_preset = getattr(settings, "technique_mapping_preset", "balanced")
    preset_choice = st.sidebar.selectbox(
        "Mapping preset",
        options=preset_keys,
        format_func=lambda key: PRESETS[key].label,
        index=preset_keys.index(current_preset) if current_preset in preset_keys else 0,
        help="Balanced is the default. Conservative accepts less and sends more to the verifier. "
        "Exploratory widens the candidate list. Local only never calls OpenRouter.",
    )
    st.sidebar.caption(PRESETS[preset_choice].description)
    settings = apply_preset(
        settings.model_copy(update={"technique_mapping_preset": preset_choice}), preset_choice
    )
    with st.sidebar.expander("Advanced: retrieval", expanded=False):
        similarity = st.slider(
            "Technique similarity floor", 0.0, 1.0, settings.similarity_threshold, 0.01
        )
        top_k = st.slider("Vector matches per observation", 1, 50, settings.retrieval_top_k)
        behavior_similarity = st.slider(
            "Actor-behavior similarity floor",
            0.0,
            1.0,
            settings.behavior_similarity_threshold,
            0.01,
        )
        behavior_top_k = st.slider(
            "Actor-behavior matches per observation", 1, 50, settings.behavior_retrieval_top_k
        )
        max_questions = st.slider("Maximum interview questions", 1, 20, settings.max_questions)
    with st.sidebar.expander("Advanced: ensemble weights", expanded=False):
        st.caption("Family budgets cap how much correlated classifiers can contribute together.")
        technique_family_weight = st.slider(
            "Technique-overlap family", 0.0, 3.0, settings.technique_family_weight, 0.05
        )
        context_family_weight = st.slider(
            "Victimology / context family", 0.0, 3.0, settings.context_family_weight, 0.05
        )
        motivation_family_weight = st.slider(
            "Motivation family", 0.0, 3.0, settings.motivation_family_weight, 0.05
        )
        llm_zero_shot_family_weight = st.slider(
            "LLM opinion family", 0.0, 2.0, settings.llm_zero_shot_family_weight, 0.05
        )
        cyberre_family_weight = st.slider(
            "CyberRE-LLM family", 0.0, 2.0, getattr(settings, "cyberre_family_weight", 0.35), 0.05
        )
        st.caption("Individual classifiers")
        default_weights = engine.default_weights()
        weights = {
            classifier.name: st.slider(
                classifier.name.replace("_", " "),
                0.0,
                3.0,
                default_weights.get(classifier.name, 1.0),
                0.05,
            )
            for classifier in engine.classifiers
        }
        weights["llm_zero_shot"] = st.slider(
            "LLM zero-shot vote",
            0.0,
            2.0,
            0.0 if runtime.local_only(settings) else default_weights.get("llm_zero_shot", 0.5),
            0.05,
            help="Bounded LLM opinion folded into the ranking after deterministic fusion. 0 disables it.",
        )
    updated = settings.model_copy(
        update={
            "similarity_threshold": similarity,
            "retrieval_top_k": top_k,
            "behavior_similarity_threshold": behavior_similarity,
            "behavior_retrieval_top_k": behavior_top_k,
            "technique_family_weight": technique_family_weight,
            "context_family_weight": context_family_weight,
            "motivation_family_weight": motivation_family_weight,
            "llm_zero_shot_family_weight": llm_zero_shot_family_weight,
            "cyberre_family_weight": cyberre_family_weight,
            "max_questions": max_questions,
            "llm_zero_shot_weight": weights["llm_zero_shot"],
        }
    )
    state = runtime.current_state()
    changed = state is not None and (
        runtime._settings_key(updated) != runtime._settings_key(engine.settings)
        or {k: round(v, 4) for k, v in weights.items()}
        != {k: round(v, 4) for k, v in state.weights.items() if k in weights}
    )
    if changed:
        runtime.mark_stale(True)
    rescore = st.sidebar.button(
        "Re-score with these settings",
        type="primary",
        use_container_width=True,
        disabled=state is None or not changed,
        help="Re-runs retrieval and ranking on the current evidence with the settings above.",
    )
    with st.sidebar.expander("System", expanded=False):
        health = _store_health(engine)
        if health is None or "error" in (health or {}):
            st.error(f"Chroma unavailable: {(health or {}).get('error', 'unknown')}")
        else:
            st.caption(
                f"Chroma online · {health['technique_sentences']:,} technique sentences · "
                f"{health['group_documents']:,} group vectors · "
                f"{health['actor_behavior_documents']:,} actor behaviours"
            )
            if health["actor_behavior_documents"] == 0:
                st.warning("Actor-behaviour index missing. Run `make ingest`.")
        st.caption(f"ATT&CK {engine.kb.version} · embedder {settings.embedding_model}")
        st.caption(f"Extractor {settings.extract_model} · reviewer {settings.review_model}")
        st.caption(f"Acceptance model: {engine.acceptance_model_status}")
        usage = getattr(engine.llm, "total_usage", None)
        if usage is not None:
            st.metric("OpenRouter spend this session", f"${usage.cost_usd:.4f}")
            st.caption(f"{usage.prompt_tokens:,} input + {usage.completion_tokens:,} output tokens")
    return updated, weights, rescore
