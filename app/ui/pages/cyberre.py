"""CyberRE-LLM: analyst-triggered LLM technique filter and ranker per observation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import status_strip
from app.ui.guard import run_step
from apt_classifier.config import Settings
from apt_classifier.evidence.models import EvidenceStatus
from apt_classifier.session import AttributionEngine


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## CyberRE-LLM")
    st.caption(
        "An independent LLM pass that filters ATT&CK techniques against one observation and ranks "
        "the relevant ones. Its result is a bounded, non-authoritative vote in the ranking."
    )
    if state is None:
        st.info("Run an analysis on the Intake page first.")
        return
    if runtime.local_only(settings):
        st.info("Unavailable in local mode (no OpenRouter key).")
        return
    observations = [
        o
        for o in state.ledger.observations
        if o.status is EvidenceStatus.POSITIVE and o.kind not in {"ioc", "software"}
    ]
    if not observations:
        st.info("No positive action observations are available.")
        return
    by_id = {o.id: o for o in observations}
    prefilter_top_k = int(getattr(engine.settings, "cyberre_prefilter_top_k", 50))
    use_prefilter = st.checkbox(
        "Use the ATTACK-BERT pre-filter (recommended)",
        value=bool(getattr(engine.settings, "cyberre_prefilter_enabled", True)),
        help=f"Send only the top {prefilter_top_k} vector matches per observation instead of the whole knowledge base.",
        key=f"cyberre-prefilter-{state.ledger.case_id}",
    )
    calls = prefilter_top_k if use_prefilter else len(engine.kb.techniques)
    results = getattr(state, "cyberre_results", {})
    pending = [o for o in observations if o.id not in results]
    st.caption(
        f"About {calls} filter calls plus one ranking call per observation · "
        f"{len(observations) - len(pending)} of {len(observations)} observations scanned."
    )
    columns = st.columns([2, 1, 1])
    selected_id = columns[0].selectbox(
        "Observation",
        list(by_id),
        format_func=lambda oid: by_id[oid].text[:160],
        key=f"cyberre-observation-{state.ledger.case_id}",
    )
    if columns[1].button(
        "Scan this observation",
        type="primary",
        use_container_width=True,
        key=f"cyberre-run-{state.ledger.case_id}-{selected_id}",
    ):
        with st.status(f"Filtering {calls} techniques and ranking…") as status:
            result = run_step(
                "CyberRE scan", engine.run_cyberre, state, selected_id, use_prefilter=use_prefilter
            )
            status.update(state="complete" if result.ok else "error")
        if result.ok:
            runtime.autosave(settings, state)
            st.rerun()
    if columns[2].button(
        f"Scan all remaining ({len(pending)})",
        use_container_width=True,
        disabled=not pending,
        key=f"cyberre-batch-{state.ledger.case_id}",
    ):
        with st.status(
            f"Scanning {len(pending)} observations, about {len(pending) * calls} calls…"
        ) as status:
            result = run_step(
                "CyberRE batch scan", engine.run_cyberre_batch, state, use_prefilter=use_prefilter
            )
            status.update(state="complete" if result.ok else "error")
        if result.ok:
            runtime.autosave(settings, state)
            st.rerun()
    if not results:
        return
    inspected_id = st.selectbox(
        "Inspect a scanned observation",
        list(results),
        format_func=lambda oid: results[oid].observation_text[:160],
        key=f"cyberre-inspect-{state.ledger.case_id}",
    )
    result = results.get(inspected_id)
    if result is None:
        return
    metrics = st.columns(4)
    metrics[0].metric("Filter calls", result.filter_calls)
    metrics[1].metric("Kept by filter", len(result.filtered))
    metrics[2].metric("Ranked", len(result.ranking))
    metrics[3].metric("Vote", "Active" if result.applicable else "Abstained")
    st.caption(
        f"Model {result.model} · {result.usage.prompt_tokens + result.usage.completion_tokens:,} tokens · ${result.usage.cost_usd:.4f}"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Rank": i.rank,
                "Technique": f"{i.technique_name} ({i.technique_id})",
                "Reason": i.reason,
            }
            for i in result.ranking
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Filter decisions", expanded=False):
        st.dataframe(
            pd.DataFrame(
                {"Technique": f"{i.technique_name} ({i.technique_id})", "Reason": i.reason}
                for i in result.filtered
            ),
            use_container_width=True,
            hide_index=True,
        )
