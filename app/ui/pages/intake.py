"""Intake: describe or upload the evidence and run the analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui import runtime
from app.ui.components import evidence_table, goto, status_strip
from app.ui.guard import run_step
from app.ui.local_extract import local_observations
from apt_classifier.config import Settings
from apt_classifier.evidence.extract import extract_evidence
from apt_classifier.evidence.models import EvidenceLedger, StrategicContext
from apt_classifier.evidence.standardise import standardise_observations
from apt_classifier.llm import prose_normalisation
from apt_classifier.session import AttributionEngine

SAMPLE_PATH = Path(__file__).resolve().parents[3] / "data" / "samples" / "case_observation.txt"
PHASES = (
    "Initial access",
    "Execution",
    "Persistence",
    "Privilege escalation",
    "Credential access",
    "Lateral movement",
    "Command and control",
    "Exfiltration / impact",
)


def _phase_key(phase: str) -> str:
    return "phase_" + phase.lower().replace(" ", "_").replace("/", "_")


def _collect_free_text() -> str:
    parts: list[str] = []
    narrative = str(st.session_state.get("intake_narrative") or "").strip()
    if narrative:
        parts.append(narrative)
    for phase in PHASES:
        value = str(st.session_state.get(_phase_key(phase)) or "").strip()
        if value:
            parts.append(f"{phase}:\n{value}")
    return "\n\n".join(parts)


def _build_ledger(
    engine: AttributionEngine,
    settings: Settings,
    *,
    free_text: str,
    alerts: str,
    iocs: str,
    files: list[Any] | None,
    context: StrategicContext,
    diagnostics: dict[str, Any],
) -> EvidenceLedger:
    prose_mode, prose_model = prose_normalisation(
        None if runtime.local_only(settings) else engine.llm, settings
    )
    if prose_mode != "extract":
        # No OpenRouter extraction: structured files and IOCs are parsed locally by the
        # extractor; prose is split into sentences here so nothing is silently dropped, and a
        # local model then rewrites each sentence 1:1 (docs 16.4a).
        ledger = extract_evidence(
            None,
            iocs=iocs or None,
            structured_files=files or None,
            structured_diagnostics=diagnostics,
            kb=engine.kb,
            strategic_context=context,
        )
        prose = local_observations(free_text)
        if prose_mode == "sentences":
            prose = standardise_observations(
                prose, engine.llm, model=str(prose_model), diagnostics=diagnostics
            )
        ledger.observations.extend(prose)
        ledger.observations.extend(local_observations(alerts, source="alerts"))
        diagnostics["normalization"] = (
            "local_llm_sentences" if prose_mode == "sentences" else "local_sentence_split"
        )
        return ledger
    return extract_evidence(
        engine.llm,
        free_text=free_text or None,
        alerts=alerts or None,
        structured_files=files or None,
        structured_diagnostics=diagnostics,
        iocs=iocs or None,
        kb=engine.kb,
        source_names={"alerts": "analyst_input"},
        strategic_context=context,
    )


def render(engine: AttributionEngine, settings: Settings, weights: dict[str, float]) -> None:
    state = runtime.current_state()
    status_strip(engine, state, runtime.case_name())
    st.markdown("## 1 · Evidence intake")
    st.caption(
        "Describe what was observed, paste alerts and indicators, or upload logs. The evidence is "
        "normalised into atomic observations, mapped to ATT&CK techniques and ranked against "
        "documented groups. Nothing here is sent anywhere except the configured model provider."
    )
    header = st.columns([1.3, 1.1, 3.6])
    if header[0].button("Load sample case", help="Fills the narrative with a worked example."):
        st.session_state["intake_narrative"] = SAMPLE_PATH.read_text(encoding="utf-8")
        st.rerun()
    if header[1].button("Clear inputs", help="Empties every intake field."):
        for key in ["intake_narrative", "intake_alerts", "intake_iocs", *map(_phase_key, PHASES)]:
            st.session_state.pop(key, None)
        st.rerun()

    st.text_area(
        "What happened? (incident narrative, report excerpts, analyst notes)",
        key="intake_narrative",
        height=220,
        placeholder="The actor gained access through a phishing email carrying a macro-enabled "
        "document, executed a PowerShell downloader, created a scheduled task for persistence, "
        "dumped LSASS with a renamed procdump, and moved laterally over SMB using stolen "
        "credentials…",
    )
    with st.expander("Add observations by kill-chain phase (optional)", expanded=False):
        columns = st.columns(2)
        for index, phase in enumerate(PHASES):
            columns[index % 2].text_area(
                phase, key=_phase_key(phase), height=90, placeholder=f"Observed {phase.lower()}…"
            )
    with st.expander("Alerts, indicators and log files (optional)", expanded=False):
        st.text_area(
            "EDR / SIEM alerts",
            key="intake_alerts",
            height=120,
            placeholder="Alert titles, descriptions or exported alert JSON…",
        )
        st.text_area(
            "Indicators and named tooling",
            key="intake_iocs",
            height=90,
            placeholder="Domains, IPs, hashes, Mimikatz, Cobalt Strike…",
        )
        files = st.file_uploader(
            "Structured evidence files",
            type=["json", "jsonl", "ndjson", "xml", "txt"],
            accept_multiple_files=True,
            help="Windows/Sysmon JSON or JSONL, Elastic exports, Windows Event XML, eCAR JSONL. "
            "The format is detected from the content. Files above 16 MB are truncated.",
        )
    with st.expander("Strategic targeting context (optional)", expanded=False):
        st.caption(
            "Only facts supported by incident or intelligence reporting. Context re-ranks "
            "technically supported candidates; it cannot introduce a group with no TTP or tool link."
        )
        columns = st.columns(2)
        geographies = columns[0].text_input("Victim country / region", key="ctx_geo")
        industries = columns[1].text_input("Targeted industry / sector", key="ctx_industry")
        motivations = columns[0].text_input("Suspected motivation", key="ctx_motivation")
        objectives = columns[1].text_input("Likely end objective", key="ctx_objective")
        origins = st.text_input(
            "Suspected sponsor / origin",
            key="ctx_origin",
            help="Enter only when supported by independent evidence, never by this tool's output.",
        )
    context = StrategicContext(
        target_geographies=geographies,
        target_industries=industries,
        motivations=motivations,
        objectives=objectives,
        suspected_origins=origins,
    )

    free_text = _collect_free_text()
    alerts = str(st.session_state.get("intake_alerts") or "").strip()
    iocs = str(st.session_state.get("intake_iocs") or "").strip()
    has_input = any((free_text, alerts, iocs, files))
    st.divider()
    action = st.columns([2, 3])
    if state is not None:
        action[1].caption(
            "Analysing replaces the current case's evidence. Use **New case** in the sidebar to "
            "keep this one and start another."
        )
    if action[0].button(
        "Analyse evidence", type="primary", use_container_width=True, disabled=not has_input
    ):
        diagnostics: dict[str, Any] = {}
        with st.status("Analysing…", expanded=True) as status:
            st.write("Normalising evidence into observations")
            built = run_step(
                "Evidence normalisation",
                _build_ledger,
                engine,
                settings,
                free_text=free_text,
                alerts=alerts,
                iocs=iocs,
                files=list(files or []),
                context=context,
                diagnostics=diagnostics,
            )
            if not built.ok:
                status.update(label="Analysis stopped", state="error")
                return
            ledger: EvidenceLedger = built.value
            if not ledger.observations:
                status.update(label="No observations found", state="error")
                st.warning(
                    "Nothing usable was extracted. Add concrete behaviour (commands, files, "
                    "registry keys, network activity) or a log file."
                )
                return
            st.write(
                f"{len(ledger.observations)} observations · mapping ATT&CK techniques and ranking groups"
            )
            ledger.case_id = runtime.new_case_id()
            analysed = run_step(
                "Technique mapping and ranking",
                runtime.reanalyze,
                engine,
                ledger,
                weights,
                previous=None,
            )
            if not analysed.ok:
                status.update(label="Analysis stopped", state="error")
                return
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["structured_ingestion_diagnostics"] = diagnostics
        if not st.session_state.get(runtime.CASE_NAME_KEY):
            st.session_state[runtime.CASE_NAME_KEY] = f"case-{ledger.case_id[:8]}"
        runtime.set_state(analysed.value)
        runtime.autosave(settings, analysed.value)
        goto("Technique Review")
        st.rerun()

    if state is None:
        if not has_input:
            st.info("Add evidence above, or load the sample case, then press **Analyse evidence**.")
        return
    st.markdown("### Current case evidence")
    ingestion = st.session_state.get("structured_ingestion_diagnostics", {})
    if ingestion:
        st.caption(
            f"Structured ingestion: {ingestion.get('parsed', 0)} parsed · "
            f"{ingestion.get('skipped', 0)} skipped · {ingestion.get('source_count', 0)} sources"
            + (" · local sentence split (no LLM)" if ingestion.get("normalization") else "")
        )
    if not state.fusion.entries:
        technique_count = len(state.retrieval.observed_techniques)
        st.warning(
            "No ATT&CK technique cleared the evidence bar, so no group can be ranked. Add more "
            "specific behaviour or logs."
            if technique_count == 0
            else f"{technique_count} technique(s) were mapped but no documented group overlaps them. "
            "Add named tooling or more distinctive behaviour."
        )
    st.dataframe(evidence_table(state), use_container_width=True, hide_index=True)
    if not state.ledger.strategic_context.is_empty:
        with st.expander("Strategic context in force"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Factor": name.replace("_", " ").title(), "Values": ", ".join(values)}
                        for name, values in state.ledger.strategic_context.field_values().items()
                        if values
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
