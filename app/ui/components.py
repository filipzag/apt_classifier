"""Shared frames, badges, the status strip and "what next" guidance."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from apt_classifier import report
from apt_classifier.classifiers.base import ClassifierResult
from apt_classifier.session import AnalysisState, AttributionEngine

# Stage vocabulary shown to analysts. Icons carry the meaning together with the label so the
# state is never expressed by colour alone.
STAGE_INFO = report.STAGE_INFO

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
    """Analyst report (shared builder in ``apt_classifier.report``)."""
    from app.ui.runtime import case_name

    return report.assessment_markdown(engine.kb, state, case_name=case_name())


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


SHORT_STAGE = report.SHORT_STAGE


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
            "**Documented overlap** means ATT&CK documents the two groups as the same or "
            "overlapping activity (a reviewed table of its own description text) or their "
            "profiles nest, so this evidence usually cannot tell them apart. Groups ATT&CK only "
            "links through rarely shared malware, or names to call them distinct, are different "
            "actors."
        )
