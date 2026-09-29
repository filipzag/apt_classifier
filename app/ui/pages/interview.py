"""Adaptive interview: predicted unobserved behaviours as collection leads, one question at a time."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import goto, group_name, horizontal_bars, status_strip
from app.ui.guard import run_step
from apt_classifier.config import Settings
from apt_classifier.elastic_mcp import ElasticMCPClient, elastic_mcp_configured
from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.questions import parse_interview_answer
from apt_classifier.session import AnalysisState, AttributionEngine

STOP_MESSAGES = {
    "question_limit": "The configured question limit was reached. Restart the interview or continue to the assessment.",
    "no_candidates": "No unanswered ATT&CK behaviour is supported by the current candidates. Add more concrete evidence in Intake.",
    "low_priority": "The remaining predicted behaviours are below the minimum investigation priority. Continue to the assessment.",
    "concentrated_low_value": "The candidate weights are concentrated and the remaining behaviours would not discriminate. Continue to the assessment.",
    "selection_failed": "The question selector did not return a usable lead. Restart the interview to retry.",
}


def _answer_label(status: EvidenceStatus) -> str:
    return {
        EvidenceStatus.POSITIVE: "Yes — observed",
        EvidenceStatus.NEGATIVE: "No — not observed",
        EvidenceStatus.UNKNOWN: "Unknown",
        EvidenceStatus.NOT_VISIBLE: "Not visible in available telemetry",
    }[status]


def _elastic_batch_key(state: AnalysisState, question: Any) -> str:
    return f"{state.ledger.case_id}:{question.technique_id}"


def _render_elastic_gathering(
    engine: AttributionEngine, settings: Settings, state: AnalysisState, question: Any
) -> None:
    configured = elastic_mcp_configured(engine.settings)
    with st.expander("Gather evidence from Elasticsearch (MCP)", expanded=False):
        st.caption(
            "Read-only search through Elastic Agent Builder. Results are staged for your review "
            "and do not affect the ranking until you import them."
        )
        if not configured:
            st.info(
                "Set ELASTIC_MCP_URL and ELASTIC_MCP_API_KEY in .env and restart to enable this."
            )
            return
        lookback = st.number_input(
            "Search lookback (hours)",
            1,
            24 * 90,
            engine.settings.elastic_mcp_lookback_hours,
            1,
            key=f"elastic-lookback-{state.ledger.case_id}",
        )
        batches = st.session_state.setdefault("elastic_mcp_batches", {})
        batch_key = _elastic_batch_key(state, question)
        if st.button("Search this lead", type="primary", key=f"elastic-search-{batch_key}"):
            with st.spinner("Searching Elasticsearch through MCP…"):
                result = run_step(
                    "Elasticsearch search",
                    lambda: ElasticMCPClient.from_settings(engine.settings).gather(
                        question, lookback_hours=int(lookback)
                    ),
                )
            if result.ok:
                batches[batch_key] = result.value
        batch = batches.get(batch_key)
        if batch is None:
            return
        if batch.esql_query:
            with st.expander("Executed query (auditable)", expanded=False):
                st.code(batch.esql_query, language="sql")
        if not batch.ok:
            st.error(f"Search failed: {batch.error}")
            return
        if not batch.candidates:
            st.info(
                "No event candidates were returned. Absence is not negative evidence unless the telemetry was complete."
            )
            return
        by_id = {candidate.id: candidate for candidate in batch.candidates}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Timestamp": c.timestamp or "",
                        "Index": c.index or "",
                        "Event ID": c.event_id or "",
                        "Excerpt": c.text[:400],
                    }
                    for c in batch.candidates
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        selected = st.multiselect(
            "Verified events to import as observations",
            options=list(by_id),
            format_func=lambda cid: by_id[cid].text[:160],
            key=f"elastic-select-{batch_key}",
        )
        columns = st.columns(2)
        if columns[0].button(
            "Import selected and re-rank",
            type="primary",
            disabled=not selected,
            key=f"elastic-import-{batch_key}",
        ):
            observations = [by_id[item].to_observation() for item in selected]
            result = run_step(
                "Importing observations", engine.import_observations, state, observations
            )
            if result.ok:
                batches.pop(batch_key, None)
                runtime.set_state(result.value)
                runtime.autosave(settings, result.value)
                st.rerun()
        if columns[1].button("Discard results", key=f"elastic-discard-{batch_key}"):
            batches.pop(batch_key, None)
            st.rerun()


def _render_weights(engine: AttributionEngine, state: AnalysisState) -> None:
    weights = engine.interview_hypothesis_weights(state)
    if not weights:
        return
    with st.expander("Which candidates drive the questions", expanded=False):
        st.caption("Normalised squared relative scores of the top candidates. Not probabilities.")
        ordered = sorted(weights, key=lambda g: -weights[g])
        horizontal_bars(
            [group_name(engine, g) for g in ordered],
            [weights[g] for g in ordered],
            value_title="hypothesis weight",
        )


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## 3 · Interview")
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    if not state.fusion.entries:
        st.info(
            "No candidate group is ranked yet, so there is nothing to ask about. Add evidence in Intake."
        )
        return
    if engine.ensure_interview_current(state):
        st.info("The saved interview used an older question model and was reset.")
    answered = len(state.ledger.question_answers)
    top = st.columns([3, 1])
    top[0].progress(
        min(1.0, answered / max(engine.settings.max_questions, 1)),
        text=f"{answered} of {engine.settings.max_questions} questions answered",
    )
    if top[1].button(
        "Restart interview",
        use_container_width=True,
        disabled=not state.ledger.question_answers
        and state.current_question is None
        and not state.interview_complete,
    ):
        engine.restart_interview(state)
        runtime.autosave(settings, state)
        st.rerun()
    st.caption(
        "Each question predicts a behaviour the shortlisted groups would show but that has not "
        "been observed. Answering narrows the shortlist. Predictions are collection leads, not "
        "findings."
    )
    leads = engine.question_shortlist(state)
    if leads:
        with st.expander("All predicted leads", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Priority": round(lead.investigation_priority, 3),
                            "Technique": f"{lead.technique_name} ({lead.technique_id})",
                            "Profiles supporting": f"{lead.supporting_group_count}/{lead.candidate_group_count}",
                            "First collection step": lead.investigation_steps[0]
                            if lead.investigation_steps
                            else "",
                        }
                        for lead in leads[:8]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
    for answer in state.ledger.question_answers:
        with st.chat_message("assistant"):
            st.markdown(answer.question)
            st.caption(answer.technique_id or "Follow-up evidence")
        with st.chat_message("user"):
            st.markdown(f"**{_answer_label(answer.status)}**")
            if answer.answer_text:
                st.caption(answer.answer_text)
    if state.current_question is None and not state.interview_complete:
        with st.spinner("Selecting the next discriminating question…"):
            result = run_step("Selecting the next question", engine.next_question, state)
        if not result.ok:
            if st.button("Retry", type="primary"):
                st.rerun()
            return
    if state.interview_complete:
        with st.chat_message("assistant"):
            st.markdown(
                STOP_MESSAGES.get(state.interview_stop_reason or "", "The interview stopped.")
            )
        _render_weights(engine, state)
        if st.button("Continue to assessment →", type="primary"):
            goto("Assessment")
        return
    question = state.current_question
    if question is None:
        st.error("The interview could not produce a question. Restart it to retry.")
        return
    with st.chat_message("assistant"):
        st.markdown(f"**{question.question}**")
        st.caption(question.hypothesis)
        with st.expander("What to inspect before answering", expanded=False):
            st.caption(
                "Event IDs are collection pivots, not proof: correlate host, user, time, process "
                "tree and command line."
            )
            for step in question.investigation_steps:
                st.markdown(f"- {step}")
            st.caption(
                f"Profile support {question.profile_support:.0%} "
                f"({question.supporting_group_count}/{question.candidate_group_count}) · "
                f"information gain {question.information_gain:.3f} bits · technique {question.technique_id}"
            )
    _render_elastic_gathering(engine, settings, state, question)
    options = (
        ("Yes", EvidenceStatus.POSITIVE),
        ("No", EvidenceStatus.NEGATIVE),
        ("Unknown", EvidenceStatus.UNKNOWN),
        ("Not visible", EvidenceStatus.NOT_VISIBLE),
    )
    selected: EvidenceStatus | None = None
    columns = st.columns(4)
    for column, (label, status) in zip(columns, options, strict=True):
        if column.button(
            label,
            use_container_width=True,
            key=f"interview-{answered}-{question.technique_id}-{status.value}",
        ):
            selected = status
    typed = st.chat_input("Or type: yes / no / unknown / not visible, with a note")
    if typed:
        selected = parse_interview_answer(typed)
        if selected is None:
            st.error("Reply with yes, no, unknown, or not visible.")
            return
    if selected is not None:
        with st.spinner("Recording the answer and re-ranking…"):
            recorded = run_step(
                "Recording the answer", engine.answer_question, state, selected, typed or None
            )
            if recorded.ok:
                run_step("Selecting the next question", engine.next_question, state)
        runtime.autosave(settings, state)
        st.rerun()
    _render_weights(engine, state)
