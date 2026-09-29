"""Technique review: one interactive table of every candidate mapping with a decision column."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import candidate_counts, candidate_state_label, goto, status_strip
from app.ui.guard import run_step
from apt_classifier.config import Settings
from apt_classifier.evidence.models import TechniqueReviewAction, TechniqueReviewDecision
from apt_classifier.session import AttributionEngine

DECISIONS = ("keep", "confirm", "reject", "replace")


def _evidence_summary(candidate: Any) -> str:
    parts: list[str] = []
    for item in candidate.evidence:
        if item.source == "acceptance_model" and item.score is not None:
            parts.append(f"p={item.score:.2f}")
        elif item.source == "dense" and item.score is not None:
            parts.append(f"dense {item.score:.2f} ({item.field or '?'})")
        elif item.source == "lexical" and item.rank is not None:
            parts.append(f"lexical #{item.rank}")
        elif item.source in {"explicit_cue", "structured_signal"}:
            parts.append("cue")
    return " · ".join(parts)


def _frame(state: Any, observations: dict[str, Any], order: dict[str, int]) -> pd.DataFrame:
    rows = []
    for candidate in state.retrieval.candidates:
        observation = observations.get(candidate.observation_id)
        text = observation.text if observation is not None else candidate.observation_id
        verification = candidate.verification
        rows.append(
            {
                "key": f"{candidate.observation_id}|{candidate.technique_id}",
                "#": order.get(candidate.observation_id, 0) + 1,
                "Observation": text,
                "Technique": candidate.technique_id,
                "Name": candidate.technique_name,
                "Status": candidate_state_label(candidate.state.value),
                "state_value": candidate.state.value,
                "Rank": candidate.fused_rank,
                "Evidence": _evidence_summary(candidate),
                "Why": candidate.state_reason or "",
                "Verifier": (
                    f"{verification.verdict.value}: {verification.reason or ''}".strip(": ")
                    if verification is not None and verification.verdict.value != "not_run"
                    else ""
                ),
                "Decision": "keep",
                "Replacement": "",
                "Rationale": "",
            }
        )
    return pd.DataFrame(rows)


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## 2 · Technique review")
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    candidates = getattr(state.retrieval, "candidates", ())
    if not candidates:
        st.info("No technique candidates were retrieved for this case.")
        return
    counts = candidate_counts(state)
    st.caption(
        "Every observation was matched against ATT&CK. **Accepted** mappings vote in the group "
        "ranking; **needs review** candidates are shown for you to confirm or reject; **rejected** "
        "ones were ruled out by the verifier or by you. Decisions are saved with the case and the "
        "ranking is recomputed."
    )
    observations = {obs.id: obs for obs in state.ledger.observations}
    order = {obs.id: index for index, obs in enumerate(state.ledger.observations)}
    filters = st.columns([2, 2, 3])
    show_states = filters[0].multiselect(
        "Show",
        options=["auto_accepted", "needs_review", "auto_rejected"],
        default=["auto_accepted", "needs_review"],
        format_func=candidate_state_label,
    )
    top_only = filters[1].toggle(
        "Top 3 per observation only",
        value=counts["needs_review"] > 30,
        help="Hide low-ranked candidates to keep the table short.",
    )
    search = filters[2].text_input("Search observation or technique", placeholder="lsass, T1059…")
    frame = _frame(state, observations, order)
    view = frame[frame["state_value"].isin(show_states)]
    if top_only:
        view = view[view["Rank"] <= 3]
    if search.strip():
        needle = search.strip().casefold()
        view = view[
            view["Observation"].str.casefold().str.contains(needle, regex=False)
            | view["Technique"].str.casefold().str.contains(needle, regex=False)
            | view["Name"].str.casefold().str.contains(needle, regex=False)
        ]
    view = view.sort_values(["#", "Rank"]).reset_index(drop=True)
    st.caption(
        f"{len(view)} of {len(frame)} candidates shown · {counts['auto_accepted']} accepted · "
        f"{counts['needs_review']} need review · {counts['auto_rejected']} rejected"
    )
    edited = st.data_editor(
        view.drop(columns=["state_value"]),
        key=f"technique_review_editor_{state.ledger.case_id}_{len(state.ledger.technique_reviews)}",
        use_container_width=True,
        hide_index=True,
        disabled=[
            "key",
            "#",
            "Observation",
            "Technique",
            "Name",
            "Status",
            "Rank",
            "Evidence",
            "Why",
            "Verifier",
        ],
        column_order=[
            "#",
            "Observation",
            "Technique",
            "Name",
            "Status",
            "Decision",
            "Replacement",
            "Rationale",
            "Evidence",
            "Why",
            "Verifier",
        ],
        column_config={
            "#": st.column_config.NumberColumn("Obs", width="small"),
            "Observation": st.column_config.TextColumn(width="large"),
            "Technique": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="small"),
            "Decision": st.column_config.SelectboxColumn(
                options=list(DECISIONS),
                required=True,
                width="small",
                help="confirm = trust this mapping; reject = drop it; replace = map to another technique",
            ),
            "Replacement": st.column_config.TextColumn(
                width="small", help="ATT&CK id (T####.###) used when Decision is replace"
            ),
            "Rationale": st.column_config.TextColumn(width="medium"),
            "Evidence": st.column_config.TextColumn(width="medium"),
            "Why": st.column_config.TextColumn("Gate reason", width="medium"),
        },
        height=min(640, 60 + 36 * max(len(view), 1)),
    )
    decisions: list[TechniqueReviewDecision] = []
    problems: list[str] = []
    for _, row in edited.iterrows():
        decision = str(row.get("Decision") or "keep")
        if decision == "keep":
            continue
        observation_id, technique_id = str(row["key"]).split("|", 1)
        rationale = str(row.get("Rationale") or "").strip() or None
        if decision == "confirm":
            decisions.append(
                TechniqueReviewDecision(
                    observation_id=observation_id,
                    technique_id=technique_id,
                    action=TechniqueReviewAction.CONFIRM,
                    rationale=rationale,
                )
            )
        elif decision == "reject":
            decisions.append(
                TechniqueReviewDecision(
                    observation_id=observation_id,
                    technique_id=technique_id,
                    action=TechniqueReviewAction.REJECT,
                    rationale=rationale,
                )
            )
        elif decision == "replace":
            replacement = str(row.get("Replacement") or "").strip().upper()
            if replacement not in engine.kb.techniques:
                problems.append(
                    f"{technique_id} on observation {row['#']}: replacement {replacement or '(empty)'} "
                    "is not an ATT&CK technique id"
                )
                continue
            decisions.append(
                TechniqueReviewDecision(
                    observation_id=observation_id,
                    technique_id=technique_id,
                    action=TechniqueReviewAction.REPLACE,
                    replacement_technique_id=replacement,
                    rationale=rationale,
                )
            )
    for problem in problems:
        st.error(problem)
    actions = st.columns([2, 2, 3])
    if actions[0].button(
        f"Apply {len(decisions)} decision(s) and re-rank",
        type="primary",
        use_container_width=True,
        disabled=not decisions or bool(problems),
    ):
        for decision in decisions:
            state.ledger.record_technique_review(decision)
        with st.spinner("Recomputing technique mappings and the group ranking…"):
            result = run_step(
                "Re-ranking", runtime.reanalyze, engine, state.ledger, state.weights, previous=state
            )
        if result.ok:
            runtime.set_state(result.value)
            runtime.autosave(settings, result.value)
            st.toast(f"Applied {len(decisions)} decision(s).", icon="✅")
            st.rerun()
    if actions[1].button("Continue to interview →", use_container_width=True):
        goto("Interview")
    applied = state.ledger.technique_reviews
    if applied:
        with st.expander(
            f"{len(applied)} decision(s) already applied to this case", expanded=False
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Observation": (
                                observations[d.observation_id].text[:100]
                                if d.observation_id in observations
                                else d.observation_id
                            ),
                            "Technique": d.technique_id,
                            "Decision": d.action.value,
                            "Replacement": d.replacement_technique_id or "",
                            "Rationale": d.rationale or "",
                        }
                        for d in applied
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
