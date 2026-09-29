"""Live threat intelligence: non-voting enrichment from public feeds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import status_strip
from app.ui.guard import run_step
from apt_classifier.config import Settings
from apt_classifier.session import AttributionEngine
from apt_classifier.threat_intel import PublicThreatIntelClient, ThreatIntelSnapshot


@st.cache_data(ttl=1800, show_spinner=False)
def load_public_threat_intel(
    days: int, include_ransomware_claims: bool, timeout: float, max_items_per_source: int
) -> ThreatIntelSnapshot:
    client = PublicThreatIntelClient(timeout=timeout)
    try:
        return client.fetch(
            days=days,
            include_ransomware_claims=include_ransomware_claims,
            max_items_per_source=max_items_per_source,
        )
    finally:
        client.close()


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## Live intelligence")
    st.caption(
        "Public feeds (CISA KEV and advisories, ransomware claim sites). Matches enrich the case "
        "but never vote, and are withheld from the independent reviewer."
    )
    controls = st.columns([1, 1.2, 0.8])
    days = controls[0].slider(
        "Lookback days", 1, 90, engine.settings.threat_intel_days, key="threat_intel_days"
    )
    include_claims = controls[1].toggle(
        "Include ransomware victim claims",
        value=engine.settings.include_ransomware_claims,
        help="Copied from extortion sites; not confirmed breaches.",
    )
    if controls[2].button("Refresh feeds", use_container_width=True):
        load_public_threat_intel.clear()
    with st.spinner("Fetching public feeds…"):
        fetched = run_step(
            "Fetching feeds",
            load_public_threat_intel,
            days,
            include_claims,
            engine.settings.threat_intel_timeout,
            engine.settings.threat_intel_max_items_per_source,
        )
    if not fetched.ok:
        return
    snapshot: ThreatIntelSnapshot = fetched.value
    st.session_state["latest_threat_intel_snapshot"] = snapshot
    matches = engine.attach_threat_intel(state, snapshot) if state is not None else ()
    authoritative = sum(item.verification == "authoritative" for item in snapshot.items)
    claims = sum(item.verification == "unverified_extortion_claim" for item in snapshot.items)
    metrics = st.columns(5)
    metrics[0].metric("Records", len(snapshot.items))
    metrics[1].metric("Authoritative", authoritative)
    metrics[2].metric("Unverified claims", claims)
    metrics[3].metric("Matches to this case", len(matches))
    metrics[4].metric("Fetched", snapshot.fetched_at[:16].replace("T", " ") + " UTC")
    offline = [source for source in snapshot.sources if not source.ok]
    if offline:
        st.warning(
            "Unavailable sources: "
            + ", ".join(f"{s.source} ({s.error or 'error'})" for s in offline)
        )
    if matches:
        st.markdown("### Matches to this case")
        st.caption(
            "Explicit overlap only: actor alias, CVE, named tool, product, industry or geography."
        )
        items = snapshot.item_by_id()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Score": m.score,
                        "Published": items[m.item_id].published_at[:10],
                        "Source": items[m.item_id].source,
                        "Title": items[m.item_id].title,
                        "Reasons": "; ".join(m.reasons),
                        "Verification": m.verification.replace("_", " ").title(),
                        "URL": items[m.item_id].url,
                    }
                    for m in matches
                    if m.item_id in items
                ]
            ),
            use_container_width=True,
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Source URL")},
        )
    elif state is not None:
        st.info("No explicit overlap between this case and the fetched records.")
    kinds = sorted({item.kind for item in snapshot.items})
    sources = sorted({item.source for item in snapshot.items})
    filters = st.columns(3)
    selected_kinds = filters[0].multiselect(
        "Record types", kinds, default=kinds, format_func=lambda v: v.replace("_", " ").title()
    )
    selected_sources = filters[1].multiselect("Sources", sources, default=sources)
    search = (
        filters[2]
        .text_input("Search", placeholder="Actor, CVE, product, victim, sector…")
        .casefold()
        .strip()
    )
    match_by_item = {m.item_id: m for m in matches}
    filtered = [
        item
        for item in snapshot.items
        if item.kind in selected_kinds
        and item.source in selected_sources
        and (
            not search
            or search
            in " ".join(
                (
                    item.title,
                    item.summary,
                    *item.actors,
                    *item.cves,
                    *item.products,
                    *item.industries,
                )
            ).casefold()
        )
    ]
    st.markdown("### Feed records")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Published": item.published_at[:10],
                    "Source": item.source,
                    "Type": item.kind.replace("_", " ").title(),
                    "Title": item.title,
                    "Actors": ", ".join(item.actors),
                    "CVEs": ", ".join(item.cves),
                    "Verification": item.verification.replace("_", " ").title(),
                    "Case match": round(match_by_item[item.id].score, 2)
                    if item.id in match_by_item
                    else None,
                    "URL": item.url,
                }
                for item in filtered
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={"URL": st.column_config.LinkColumn("Source URL")},
    )
