"""Assessment: the stage decision, the ranking, the evidence behind it, review and export."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import (
    STAGE_INFO,
    assessment_markdown,
    benchmark_note,
    classifier_frame,
    correlation_frame,
    fusion_frame,
    goto,
    group_name,
    horizontal_bars,
    stage_badge,
    stage_glossary,
    status_strip,
)
from app.ui.guard import run_step
from apt_classifier.attack_flow import build_attack_flow
from apt_classifier.canvas import build_analyst_canvas, build_candidate_dossier
from apt_classifier.config import Settings
from apt_classifier.navigator import export_layers
from apt_classifier.overlap_cluster import recorded_cluster_scope, split_related
from apt_classifier.session import AnalysisState, AttributionEngine


def _overlap_advice(stage: str) -> str:
    """What a documented same-activity link means for this decision."""
    if stage == "known_group_hypothesis":
        return "Here the evidence separates the leader from them, so the leader is named alone."
    if stage == "known_cluster_hypothesis":
        return "The named cluster above reports them together."
    return "When the evidence cannot separate them, report the cluster rather than one name."


def _render_stage(engine: AttributionEngine, state: AnalysisState) -> None:
    decision = getattr(state, "attribution_decision", None)
    if decision is None:
        st.info("No attribution decision is available for this case.")
        return
    info = STAGE_INFO.get(decision.stage.value, {})
    named = decision.stage.value in {"known_group_hypothesis", "known_cluster_hypothesis"}
    with st.container(border=True):
        head = st.columns([2.2, 1, 1, 1])
        head[0].markdown(f"### {stage_badge(decision.stage.value)}")
        head[0].caption(info.get("summary", ""))
        thresholds = engine.attribution_thresholds
        head[1].metric(
            "Maturity",
            f"{decision.maturity_score:.2f}",
            help="0-1 score from evidence volume, independent families, classifier agreement, "
            "margin and technique specificity.",
        )
        head[1].caption(f"naming bar {thresholds.hypothesis_maturity_score:.2f}")
        head[2].metric(
            "Margin",
            f"{decision.margin:.2f}",
            help="Gap between the first and second candidate's relative index (0-1).",
        )
        head[2].caption(f"naming bar {thresholds.hypothesis_margin:.2f}")
        separation = float(getattr(decision, "separation", 0.0) or 0.0)
        head[3].metric(
            "Separation",
            f"{separation:.2f}",
            help="Evidence the leader has that its best-supported alternative cannot explain, "
            "as a share of the leader's discriminative evidence (-1 to 1). Shared evidence "
            "counts for neither side.",
        )
        gate_mode = str(getattr(thresholds, "gate_mode", "maturity"))
        if gate_mode != "maturity":
            head[3].caption(f"naming bar {thresholds.hypothesis_separation:.2f}")
        else:
            head[3].caption("informational")
        members = tuple(getattr(decision, "named_cluster", ()) or ())
        if decision.stage.value == "known_cluster_hypothesis" and members:
            from apt_classifier.overlap_cluster import cluster_label

            st.markdown(
                f"**Named documented-overlap cluster:** {cluster_label(engine.kb, members)} · "
                "report the cluster, not either member"
            )
        elif decision.leading_group_name:
            role = "Named hypothesis" if named else "Closest documented group (not an attribution)"
            st.markdown(
                f"**{role}:** {decision.leading_group_name} ({decision.leading_group_id}) · "
                f"relative index {decision.leading_score:.1f}"
            )
        runner = getattr(decision, "runner_up_group_id", None)
        if runner:
            st.caption(
                f"Best-supported alternative: {group_name(engine, runner)}"
                + (" (documented-related)" if getattr(decision, "runner_up_related", False) else "")
            )
        related = getattr(decision, "related_leading_groups", {}) or {}
        same, other = split_related(
            engine.kb,
            decision.leading_group_id or "",
            related,
            scope=recorded_cluster_scope(
                state, str(getattr(engine.attribution_thresholds, "cluster_scope", "documented"))
            ),
            named_cluster=tuple(getattr(decision, "named_cluster", ()) or ()),
        )
        if other:
            st.info(
                "**ATT&CK-linked groups.** ATT&CK links the leader to "
                + "; ".join(f"{group_name(engine, gid)} ({reason})" for gid, reason, _ in other)
                + ", but does not document them as the same actor. Keep them apart: look for "
                "evidence only one of them documents.",
                icon="🔗",
            )
        if same:
            st.warning(
                "**Documented overlap.** ATT&CK documents the leader and "
                + "; ".join(f"{group_name(engine, gid)} ({reason})" for gid, reason in same)
                + " as the same or overlapping activity. "
                + _overlap_advice(decision.stage.value),
                icon="🔗",
            )
        decisive = getattr(state.context, "decisive_software", {}) or {}
        if float(decisive.get("factor", 1.0) or 1.0) > 1.0:
            names = ", ".join(
                f"{engine.kb.software[sid].name} ({sid})" if sid in engine.kb.software else sid
                for sid in decisive.get("software_ids", [])
            )
            st.info(
                f"**Decisive tooling.** The evidence names {names}, malware that ATT&CK documents "
                f"for very few groups, so the software vote was weighted ×{decisive['factor']:.1f}.",
                icon="🧬",
            )
        if not named:
            st.info(
                f"**No named group meets the evidence bar.** {info.get('advice', '')} "
                "The ranking below is a shortlist of the closest documented groups.",
                icon="ℹ️",
            )
        st.caption(decision.rationale)
        note = benchmark_note(str(engine.settings.audit_dir))
        if note:
            st.caption(note)
        if decision.maturity_factors:
            names = sorted(decision.maturity_factors)
            horizontal_bars(
                [name.replace("_", " ") for name in names],
                [decision.maturity_factors[name] for name in names],
                value_title="factor score (0-1)",
                height=170,
            )
        if decision.cluster_id:
            st.caption(
                f"Activity cluster id (stable hash of the evidence): `{decision.cluster_id}`"
            )
    stage_glossary()


def _render_ranking(engine: AttributionEngine, state: AnalysisState) -> str | None:
    st.markdown("### Candidate groups")
    st.caption(
        "Family-aware reciprocal-rank fusion over the documented ATT&CK overlap. The relative "
        "index compares candidates within this case only; it is not a probability. Select a row "
        "to open the candidate's dossier."
    )
    if not state.fusion.entries:
        st.error("No candidate has documented technical support. Add evidence in Intake.")
        return None
    entries = state.fusion.entries[:15]
    horizontal_bars(
        [f"{e.rank}. {group_name(engine, e.group_id)}" for e in entries],
        [e.relative_score for e in entries],
        value_title="relative ensemble index (within this case)",
    )
    frame = fusion_frame(engine, state, limit=20)
    event = st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"ranking_table_{state.ledger.case_id}",
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        return state.fusion.entries[int(rows[0])].group_id
    return state.fusion.entries[0].group_id


def _render_dossier(engine: AttributionEngine, state: AnalysisState, group_id: str) -> None:
    dossier = build_candidate_dossier(engine.kb, state, group_id)
    with st.container(border=True):
        st.markdown(f"#### {group_name(engine, group_id)}")
        top = st.columns(3)
        top[0].metric("Rank", dossier["rank"])
        top[1].metric("Relative index", f"{dossier['relative_score']:.1f}")
        top[2].metric("Classifiers disagree", "yes" if dossier["contested"] else "no")
        tabs = st.tabs(["Why it ranks here", "Documented profile", "Gaps"])
        with tabs[0]:
            if dossier["matched_techniques"]:
                st.markdown("**Observed techniques this group documents**")
                st.dataframe(
                    pd.DataFrame(dossier["matched_techniques"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if dossier["specific_behaviors"]:
                st.markdown("**Actor-specific behaviour matches**")
                st.dataframe(
                    pd.DataFrame(dossier["specific_behaviors"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if dossier["tools"]:
                st.markdown(
                    "**Named tools:** " + ", ".join(item["name"] for item in dossier["tools"])
                )
            st.markdown("**Classifier ranks**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Classifier": n, "Rank": r}
                        for n, r in sorted(dossier["classifier_ranks"].items())
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with tabs[1]:
            st.write(dossier["description"] or "No ATT&CK group description available.")
            for item in dossier["country_evidence"]:
                st.caption(f"{item['country']}: {item['excerpt']}")
        with tabs[2]:
            if dossier["unmatched_techniques"]:
                st.markdown("**Observed but not documented for this group**")
                st.dataframe(
                    pd.DataFrame(dossier["unmatched_techniques"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if dossier["contradicting_techniques"]:
                st.markdown(
                    "**Contradicted by interview answers:** "
                    + ", ".join(dossier["contradicting_techniques"])
                )
            if not dossier["unmatched_techniques"] and not dossier["contradicting_techniques"]:
                st.caption("No gaps recorded.")


def _render_evidence_detail(engine: AttributionEngine, state: AnalysisState) -> None:
    with st.expander("Evidence detail: techniques, behaviours, context, origin", expanded=False):
        detected = pd.DataFrame(
            [
                {
                    "Technique": f"{m.technique_name} ({m.technique_id})",
                    "Confidence": round(m.score, 3),
                    "Observations": len(m.observation_ids),
                    "Matched ATT&CK text": m.matched_text,
                }
                for m in state.retrieval.matches
            ]
        )
        st.markdown("**Accepted ATT&CK techniques**")
        st.dataframe(detected, use_container_width=True, hide_index=True)
        if state.retrieval.specific_behavior_matches:
            observations = {o.id: o.text for o in state.ledger.observations}
            st.markdown("**Actor-specific behaviour matches**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Candidate": group_name(engine, m.group_id),
                            "Technique": m.technique_id,
                            "Similarity": round(m.score, 3),
                            "Observed": observations.get(m.observation_id, m.observation_id),
                            "Documented": m.matched_text,
                        }
                        for m in sorted(
                            state.retrieval.specific_behavior_matches,
                            key=lambda i: (-i.score, i.group_id),
                        )[:40]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        strategic = next(
            (
                r
                for r in state.classifier_results
                if r.classifier == "strategic_context" and r.applicable
            ),
            None,
        )
        if strategic is not None:
            st.markdown("**Strategic-context matches**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Rank": e.rank,
                            "Candidate": group_name(engine, e.group_id),
                            "Coverage": round(e.score, 3),
                            "Factors": ", ".join(e.signal_ids),
                        }
                        for e in strategic.entries[:20]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        origin = state.origin_assessment.origin
        threat_type = state.origin_assessment.threat_type
        st.markdown(
            "**Origin and intrusion type** (aggregated from ATT&CK reporting on the ranked groups; never a geolocation or proof of sponsorship)"
        )
        columns = st.columns(3)
        columns[0].metric(
            "Most supported origin",
            origin.top_country or "Unresolved",
            origin.confidence,
            delta_color="off",
        )
        columns[1].metric(
            "Intrusion type", threat_type.label, threat_type.confidence, delta_color="off"
        )
        columns[2].metric("Origin-bearing coverage", f"{origin.candidate_coverage:.0f}%")
        if origin.hypotheses:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Country": h.country,
                            "Relative support": round(h.relative_support, 1),
                            "Supporting candidates": ", ".join(
                                group_name(engine, g) for g in h.supporting_group_ids
                            ),
                        }
                        for h in origin.hypotheses
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_diagnostics(engine: AttributionEngine, state: AnalysisState) -> None:
    with st.expander("Classifier diagnostics", expanded=False):
        st.markdown("**Per-classifier ranking matrix**")
        st.dataframe(
            classifier_frame(engine, state.classifier_results),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Rank correlation between classifiers (Kendall τ)**")
        st.dataframe(
            correlation_frame(state.classifier_results, state.fusion.correlations).round(2),
            use_container_width=True,
        )
        st.markdown("**Effective weights after family normalisation**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Classifier": n, "Effective weight": round(w, 4)}
                    for n, w in state.fusion.effective_weights.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        diagnostics = getattr(state.retrieval, "diagnostics", {}) or {}
        if diagnostics:
            st.markdown("**Retrieval diagnostics**")
            st.json(diagnostics, expanded=False)
        reasons = getattr(state.context, "group_eligibility_reasons", {}) or {}
        if reasons:
            st.markdown("**Why each candidate is eligible**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Group": group_name(engine, g), "Eligible via": ", ".join(r)}
                        for g, r in sorted(reasons.items())
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_review(engine: AttributionEngine, settings: Settings, state: AnalysisState) -> None:
    st.markdown("### Independent AI review")
    st.caption(
        "Four blinded and unblinded reviews plus a devil's advocate, reconciled into an advisory "
        "opinion. It never changes the deterministic ranking. It costs several model calls."
    )
    if runtime.local_only(settings):
        st.info("Unavailable in local mode (no OpenRouter key and no local LLM).")
    else:
        columns = st.columns(2)
        if columns[0].button("Run independent review", type="primary", use_container_width=True):
            with st.status(
                "Running blinded, unblinded, devil's-advocate and reconciliation reviews…"
            ) as status:
                result = run_step("Independent review", engine.run_review, state)
                status.update(state="complete" if result.ok else "error")
            if result.ok:
                runtime.set_state(state)
                st.rerun()
    if state.review is None:
        return
    final = state.review.final.parsed
    with st.container(border=True):
        st.markdown(f"#### {final.get('headline', 'Independent AI review')}")
        st.caption(
            f"Review confidence: **{str(final.get('confidence_label', 'low')).upper()}** · advisory only"
        )
        st.write(final.get("assessment", ""))
        advisory = final.get("advisory_ranking", [])
        if advisory:
            baseline = {e.group_id: e.rank for e in state.fusion.entries}
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Advisory rank": item.get("rank"),
                            "Candidate": group_name(engine, str(item.get("candidate_id", ""))),
                            "Deterministic rank": baseline.get(str(item.get("candidate_id", ""))),
                            "Justification": item.get("justification"),
                        }
                        for item in advisory
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        bias = state.review.fame_bias
        with st.expander("Fame-bias check and devil's advocate", expanded=False):
            columns = st.columns(2)
            columns[0].metric("Blinded vs named Kendall τ", f"{bias.kendall_tau:.3f}")
            columns[1].metric(
                "Prominence/promotion correlation", f"{bias.prominence_correlation:.3f}"
            )
            st.write(state.review.advocate.parsed.get("counter_case", ""))
            for item in state.review.advocate.parsed.get("missing_evidence", []) or []:
                st.markdown(f"- Missing: {item}")
        if final.get("limitations"):
            with st.expander("Limitations and disconfirmation tests", expanded=False):
                for item in final.get("limitations", []):
                    st.markdown(f"- {item}")
                for item in final.get("disconfirmation_tests", []):
                    st.markdown(f"- Test: {item}")
        if state.review.audit_path:
            st.caption(f"Audit written to `{state.review.audit_path}`")


def _export_payload(
    engine: AttributionEngine, state: AnalysisState
) -> tuple[str, dict[str, Any], Any]:
    decision = getattr(state, "attribution_decision", None)
    navigator_layers = export_layers(
        engine.kb,
        observed_techniques=dict(getattr(state.retrieval, "observed_techniques", {})),
        negative_techniques=getattr(state.context, "negative_techniques", set()),
        candidate_technique_ids=[m.technique_id for m in getattr(state.retrieval, "matches", ())],
        profile=getattr(state, "telemetry_profile", None),
    )
    attack_flow = build_attack_flow(engine.kb, state, case_id=state.ledger.case_id)
    payload = {
        "case_id": state.ledger.case_id,
        "case_name": runtime.case_name(),
        "ledger": state.ledger.to_dict(),
        "fusion": state.fusion.to_dict(),
        "attribution_decision": decision.to_dict() if decision is not None else None,
        "origin_and_threat_type": state.origin_assessment.to_dict(),
        "cyberre_results": {
            k: v.to_dict() for k, v in getattr(state, "cyberre_results", {}).items()
        },
        "analyst_canvas": build_analyst_canvas(
            engine.kb, state, top_groups=10, max_evidence=20
        ).to_dict(),
        "navigator_layers": navigator_layers,
        "attack_flow": attack_flow.to_dict(),
        "live_threat_intel": {
            "snapshot": state.threat_intel_snapshot.to_dict()
            if state.threat_intel_snapshot is not None
            else None,
            "matches": [m.to_dict() for m in state.threat_intel_matches],
            "voting": False,
        },
        "review": state.review.to_dict() if state.review is not None else None,
    }
    return json.dumps(payload, indent=2, default=str), navigator_layers, attack_flow


def _render_export(engine: AttributionEngine, state: AnalysisState) -> None:
    st.markdown("### Export")
    st.caption("Available at any point; the report states which steps were run.")
    built = run_step("Preparing exports", _export_payload, engine, state)
    if not built.ok:
        return
    export_json, navigator_layers, attack_flow = built.value
    stem = f"apt-{runtime.case_name() or state.ledger.case_id}".replace(" ", "-")
    columns = st.columns(4)
    columns[0].download_button(
        "Markdown report",
        assessment_markdown(engine, state),
        file_name=f"{stem}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    columns[1].download_button(
        "JSON audit",
        export_json,
        file_name=f"{stem}.json",
        mime="application/json",
        use_container_width=True,
    )
    columns[2].download_button(
        "Navigator layers",
        json.dumps(navigator_layers, indent=2, default=str),
        file_name=f"{stem}-navigator.json",
        mime="application/json",
        use_container_width=True,
    )
    columns[3].download_button(
        "Attack Flow (STIX)",
        json.dumps(attack_flow.to_stix(), indent=2, default=str),
        file_name=f"{stem}-attack-flow.json",
        mime="application/json",
        use_container_width=True,
    )


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## 4 · Assessment")
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    _render_stage(engine, state)
    selected = _render_ranking(engine, state)
    if selected is not None:
        _render_dossier(engine, state, selected)
    _render_evidence_detail(engine, state)
    _render_diagnostics(engine, state)
    st.divider()
    _render_review(engine, settings, state)
    st.divider()
    _render_export(engine, state)
    if (
        not state.ledger.question_answers
        and not state.interview_complete
        and st.button("Sharpen the ranking with the interview →")
    ):
        goto("Interview")
