"""Shared frames, badges, the status strip and "what next" guidance."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from apt_classifier.classifiers.base import ClassifierResult
from apt_classifier.session import AnalysisState, AttributionEngine

# Stage vocabulary shown to analysts. Icons carry the meaning together with the label so the
# state is never expressed by colour alone.
STAGE_INFO: dict[str, dict[str, str]] = {
    "insufficient_evidence": {
        "label": "Insufficient evidence",
        "icon": "⬜",
        "summary": "Too little technique evidence to rank groups meaningfully.",
        "advice": "Add concrete behaviour (commands, persistence, lateral movement) or logs.",
    },
    "unresolved_activity_cluster": {
        "label": "Unresolved activity cluster",
        "icon": "🔶",
        "summary": "Techniques were mapped but no documented group stands out.",
        "advice": "Treat this as an unattributed cluster; the shortlist is a starting point only.",
    },
    "known_group_shortlist": {
        "label": "Known-group shortlist",
        "icon": "🔷",
        "summary": "Several documented groups overlap the evidence; none clears the naming bar.",
        "advice": "Answer interview questions or add discriminating evidence to separate them.",
    },
    "known_group_hypothesis": {
        "label": "Named group hypothesis",
        "icon": "🟢",
        "summary": "One documented group clears the maturity and margin bar.",
        "advice": "Check the documented-overlap note and run the independent review before reporting.",
    },
}

CANDIDATE_STATE_INFO: dict[str, tuple[str, str]] = {
    "auto_accepted": ("✅", "Accepted"),
    "needs_review": ("❔", "Needs review"),
    "auto_rejected": ("⛔", "Rejected"),
}


def group_name(engine: AttributionEngine, group_id: str) -> str:
    group = engine.kb.groups.get(group_id)
    return f"{group.name} ({group_id})" if group else group_id


def stage_badge(stage_value: str | None) -> str:
    info = STAGE_INFO.get(stage_value or "", None)
    if info is None:
        return "⬜ No analysis yet"
    return f"{info['icon']} {info['label']}"


def candidate_state_label(state_value: str) -> str:
    icon, label = CANDIDATE_STATE_INFO.get(state_value, ("•", state_value))
    return f"{icon} {label}"


@lru_cache(maxsize=1)
def benchmark_note(audits_dir: str) -> str | None:
    """A one-line, artifact-backed statement about the naming bar (or None if unavailable)."""
    path = Path(audits_dir) / "verification_results_current.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    parts: list[str] = []
    folds = payload.get("attribution_text_heldout") or {}
    real = payload.get("cti_hal_attribution") or {}
    if folds.get("named_risk") is not None:
        parts.append(
            f"ATT&CK folds: named risk {float(folds['named_risk']):.2f} at "
            f"{float(folds.get('named_coverage') or 0):.0%} coverage"
        )
    if real.get("named_risk") is not None:
        parts.append(
            f"real reports: named risk {float(real['named_risk']):.2f} at "
            f"{float(real.get('named_coverage') or 0):.0%} coverage"
        )
    if not parts:
        return None
    return "Measured on the attribution benchmarks (" + "; ".join(parts) + ")."


def fusion_frame(engine: AttributionEngine, state: AnalysisState, limit: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": entry.rank,
                "Candidate": group_name(engine, entry.group_id),
                "Relative ensemble index": round(entry.relative_score, 2),
                "Best classifier rank": entry.rank_min,
                "Worst classifier rank": entry.rank_max,
                "Rank σ": round(entry.rank_std, 2),
                "Voting classifiers": len(entry.classifier_ranks),
                "Contested": entry.contested,
            }
            for entry in state.fusion.entries[:limit]
        ]
    )


def classifier_frame(
    engine: AttributionEngine, results: tuple[ClassifierResult, ...]
) -> pd.DataFrame:
    candidate_ids = {
        entry.group_id for result in results if result.applicable for entry in result.entries[:15]
    }
    rows = []
    for group_id in candidate_ids:
        row: dict[str, Any] = {"Candidate": group_name(engine, group_id)}
        for result in results:
            item = result.by_group().get(group_id)
            row[result.classifier] = item.rank if item else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    sort_columns = [result.classifier for result in results if result.applicable]
    if frame.empty or not sort_columns:
        return frame
    return frame.sort_values(by=sort_columns, na_position="last")


def correlation_frame(
    results: tuple[ClassifierResult, ...], correlations: dict[str, float]
) -> pd.DataFrame:
    names = [result.classifier for result in results if result.applicable]
    return pd.DataFrame(
        [[correlations.get(f"{left}|{right}", 0.0) for right in names] for left in names],
        index=names,
        columns=names,
    )


def evidence_table(state: AnalysisState) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Source": observation.provenance.source,
                "Kind": observation.kind,
                "Observation": observation.text,
                "Occurrences": observation.occurrence_count,
                "Lines": ", ".join(map(str, observation.line_numbers)),
            }
            for observation in state.ledger.observations
        ]
    )


def candidate_counts(state: AnalysisState | None) -> dict[str, int]:
    counts = {"auto_accepted": 0, "needs_review": 0, "auto_rejected": 0}
    if state is None:
        return counts
    for candidate in getattr(state.retrieval, "candidates", ()):
        counts[candidate.state.value] = counts.get(candidate.state.value, 0) + 1
    return counts


def assessment_markdown(engine: AttributionEngine, state: AnalysisState) -> str:
    """Analyst report. Works with or without the AI review; the deterministic result leads."""
    decision = getattr(state, "attribution_decision", None)
    lines = ["# APT attribution assessment", ""]
    if decision is not None:
        info = STAGE_INFO.get(decision.stage.value, {})
        lines += [
            f"**Attribution stage:** {info.get('label', decision.stage.value)}",
            f"**Maturity:** {decision.maturity_score:.2f} · **Margin:** {decision.margin:.2f} · "
            f"**Separation:** {float(getattr(decision, 'separation', 0.0) or 0.0):.2f}",
        ]
        runner = getattr(decision, "runner_up_group_id", None)
        if runner:
            lines.append(
                f"**Best-supported alternative:** {group_name(engine, runner)}"
                + (" (documented-related)" if getattr(decision, "runner_up_related", False) else "")
            )
        if decision.leading_group_name:
            role = (
                "Leading hypothesis"
                if decision.stage.value == "known_group_hypothesis"
                else "Closest documented group (not an attribution)"
            )
            lines.append(f"**{role}:** {decision.leading_group_name} ({decision.leading_group_id})")
        related = getattr(decision, "related_leading_groups", {}) or {}
        if related:
            lines.append(
                "**Documented overlap:** "
                + "; ".join(
                    f"{group_name(engine, gid)}: {reason}" for gid, reason in related.items()
                )
            )
        decisive = getattr(state.context, "decisive_software", {}) or {}
        if float(decisive.get("factor", 1.0) or 1.0) > 1.0:
            lines.append(
                "**Decisive tooling:** named malware documented for very few groups ("
                + ", ".join(decisive.get("software_ids", []))
                + f"); software vote weighted ×{float(decisive['factor']):.1f}."
            )
        lines += ["", decision.rationale, ""]
    lines += [
        "## Deterministic ranking (authoritative)",
        "Family-aware reciprocal-rank fusion over the documented ATT&CK overlap. The relative "
        "index compares candidates within this case; it is not a probability.",
    ]
    for entry in state.fusion.entries[:15]:
        lines.append(
            f"{entry.rank}. **{group_name(engine, entry.group_id)}** — relative index "
            f"{entry.relative_score:.1f}{' (contested)' if entry.contested else ''}"
        )
    accepted = getattr(state.retrieval, "observed_techniques", {}) or {}
    if accepted:
        lines += ["", "## Accepted ATT&CK techniques"]
        names = {m.technique_id: m.technique_name for m in getattr(state.retrieval, "matches", ())}
        lines += [f"- {tid} {names.get(tid, '')}".rstrip() for tid in sorted(accepted)]
    origin = state.origin_assessment.origin
    threat_type = state.origin_assessment.threat_type
    lines += [
        "",
        "## Origin and intrusion-type hypotheses (from ATT&CK reporting on the ranked groups)",
        f"- Most supported origin: {origin.top_country or 'Unresolved'} ({origin.confidence})",
        f"- Intrusion type: {threat_type.label} ({threat_type.confidence})",
    ]
    if state.review is not None:
        final = state.review.final.parsed
        lines += [
            "",
            f"## Independent AI review: {final.get('headline', '')}".rstrip(": "),
            f"Confidence: {final.get('confidence_label', 'low')}",
            "",
            str(final.get("assessment", "")),
        ]
        advisory = final.get("advisory_ranking", [])
        if advisory:
            lines += ["", "Advisory ordering (not applied to the deterministic ranking):"]
            lines += [
                f"{item.get('rank')}. {group_name(engine, str(item.get('candidate_id', '')))} — "
                f"{item.get('justification', '')}"
                for item in advisory
            ]
        if final.get("limitations"):
            lines += ["", "Limitations:"] + [f"- {item}" for item in final["limitations"]]
        if final.get("disconfirmation_tests"):
            lines += ["", "Disconfirmation tests:"] + [
                f"- {item}" for item in final["disconfirmation_tests"]
            ]
    if state.threat_intel_snapshot is not None and state.threat_intel_matches:
        items = state.threat_intel_snapshot.item_by_id()
        lines += ["", "## Non-voting live threat-intelligence matches"]
        lines += [
            f"- [{items[m.item_id].source}] {items[m.item_id].title} ({m.verification}): "
            f"{'; '.join(m.reasons)}"
            for m in state.threat_intel_matches[:10]
            if m.item_id in items
        ]
    lines += [
        "",
        "## Caveats",
        "- Ranking measures overlap with documented ATT&CK tradecraft; shared tooling, reporting "
        "gaps and false flags can invalidate attribution.",
        "- Interview leads are predictions, not findings.",
    ]
    return "\n".join(lines)


def next_step(state: AnalysisState | None) -> tuple[str, str]:
    """(message, page key) telling the analyst what to do next."""
    if state is None:
        return "Add evidence and run the analysis.", "Intake"
    counts = candidate_counts(state)
    if not state.fusion.entries:
        return (
            "No candidate group has documented overlap yet. Add more concrete evidence.",
            "Intake",
        )
    reviewed = len(getattr(state.ledger, "technique_reviews", []) or [])
    if counts["needs_review"] and not reviewed:
        return (
            f"{counts['needs_review']} technique candidates need a decision. Confirm or reject "
            "them so only trusted mappings vote.",
            "Technique Review",
        )
    if not state.ledger.question_answers and not state.interview_complete:
        return "Answer the interview leads to separate the shortlisted groups.", "Interview"
    return (
        "Read the assessment, then export the report or run the independent review.",
        "Assessment",
    )


def horizontal_bars(
    labels: list[str], values: list[float], *, value_title: str, height: int | None = None
) -> None:
    """Static horizontal bar chart in rank order (no zoom or pan on scroll, no legend)."""
    import altair as alt

    frame = pd.DataFrame({"label": labels, "value": values})
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=value_title),
            y=alt.Y("label:N", sort=None, title=None),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=height or max(120, 24 * len(labels) + 30))
    )
    st.altair_chart(chart, use_container_width=True)


def goto(page_key: str) -> None:
    page = (st.session_state.get("_navigation_pages") or {}).get(page_key)
    if page is not None:
        st.switch_page(page)


SHORT_STAGE = {
    "insufficient_evidence": "Insufficient",
    "unresolved_activity_cluster": "Cluster",
    "known_group_shortlist": "Shortlist",
    "known_group_hypothesis": "Named",
}


def status_strip(engine: AttributionEngine, state: AnalysisState | None, case_name: str) -> None:
    """Compact case header shown on every page (five short values so it survives narrow screens)."""
    counts = candidate_counts(state)
    decision = getattr(state, "attribution_decision", None) if state is not None else None
    stage_value = decision.stage.value if decision else None
    icon = STAGE_INFO.get(stage_value or "", {}).get("icon", "⬜")
    columns = st.columns([1.8, 1, 1.4, 1, 1.6])
    columns[0].metric("Case", (case_name[:22] if state is not None else "—") or "—")
    columns[1].metric("Observations", len(state.ledger.observations) if state else 0)
    columns[2].metric(
        "Techniques",
        f"{counts['auto_accepted']} ✓ · {counts['needs_review']} ?",
        help="accepted · needing review",
    )
    columns[3].metric("Candidates", len(state.fusion.entries) if state else 0)
    columns[4].metric(
        "Stage",
        f"{icon} {SHORT_STAGE.get(stage_value or '', 'None')}",
        help=STAGE_INFO.get(stage_value or "", {}).get("summary", "No analysis yet"),
    )
    message, page_key = next_step(state)
    hint, button = st.columns([5, 1.2])
    hint.caption(f"**Next:** {message}")
    if button.button("Go", key=f"next-step-{page_key}", use_container_width=True):
        goto(page_key)
    if st.session_state.get("results_stale"):
        st.warning(
            "Settings or weights changed since this result was computed. Use **Re-score** in the "
            "sidebar to apply them.",
            icon="⚠️",
        )


def stage_glossary() -> None:
    with st.expander("How to read the attribution stage", expanded=False):
        for info in STAGE_INFO.values():
            st.markdown(f"{info['icon']} **{info['label']}** — {info['summary']} {info['advice']}")
        st.markdown(
            "**Maturity** combines technique count, independent evidence families, technique "
            "specificity and corroboration into one 0-1 score. **Margin** is the gap between the "
            "first and second candidate. A group is *named* only when both clear the configured "
            "bar; the bar was set so that no wrong name was issued on the benchmarks. "
            "**Documented overlap** means ATT&CK itself cross-references the two groups or "
            "records tooling shared by very few groups, so this evidence usually cannot tell "
            "them apart."
        )
