"""Vestige — Streamlit entry point (legacy UI).

Bootstraps settings, the knowledge base and the per-session engine, then hands off to the
page modules under ``app/ui/pages``. Workflow order: Intake → Technique Review → Interview →
Assessment; Canvas, Diamond Model, CyberRE-LLM and Live Intel are analysis views.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui import runtime  # noqa: E402
from app.ui.guard import run_step  # noqa: E402
from app.ui.pages import (  # noqa: E402
    assessment,
    canvas,
    cyberre,
    diamond,
    intake,
    intel,
    interview,
    technique_review,
)
from app.ui.sidebar import render_sidebar  # noqa: E402
from apt_classifier.config import get_settings  # noqa: E402
from apt_classifier.ingest import ingest  # noqa: E402

st.set_page_config(page_title="Vestige", layout="wide", page_icon="🛡️")

settings = get_settings()
settings.ensure_directories()

st.title("Vestige")
st.caption(
    "Ranks hypotheses by overlap with documented ATT&CK tradecraft and says when the evidence "
    "does not support a name. Shared tooling, reporting gaps and false flags can invalidate "
    "attribution; treat every output as a lead."
)

if not settings.kb_path.exists():
    st.info("The local ATT&CK knowledge base has not been ingested yet.")
    if st.button("Download ATT&CK and build the index", type="primary"):
        with st.status(
            "Downloading ATT&CK, loading ATTACK-BERT and embedding the index…"
        ) as status:
            result = run_step("Ingestion", ingest, settings, reset=True)
            status.update(state="complete" if result.ok else "error")
        if result.ok:
            st.success(
                f"Ingested {len(result.value.knowledge_base.techniques)} techniques and "
                f"{len(result.value.knowledge_base.groups)} groups."
            )
            st.cache_resource.clear()
            st.rerun()
    st.stop()

base_settings = st.session_state.get("_effective_settings", settings)
engine_result = run_step("Starting the analysis engine", runtime.get_engine, base_settings)
if not engine_result.ok:
    st.stop()
engine = engine_result.value

effective_settings, classifier_weights, rescore_requested = render_sidebar(settings, engine)
if effective_settings.model_dump_json() != base_settings.model_dump_json():
    st.session_state["_effective_settings"] = effective_settings
if rescore_requested:
    state = runtime.current_state()
    if state is not None:
        engine = runtime.get_engine(effective_settings)
        with st.spinner("Re-scoring with the new settings…"):
            result = run_step(
                "Re-scoring",
                runtime.reanalyze,
                engine,
                state.ledger,
                classifier_weights,
                previous=state,
            )
        if result.ok:
            runtime.set_state(result.value)
            runtime.autosave(settings, result.value)
            st.rerun()

pages = {
    "Intake": st.Page(
        lambda: intake.render(engine, effective_settings, classifier_weights),
        title="1 · Intake",
        url_path="intake",
        default=True,
    ),
    "Technique Review": st.Page(
        lambda: technique_review.render(engine, effective_settings, classifier_weights),
        title="2 · Technique Review",
        url_path="technique-review",
    ),
    "Interview": st.Page(
        lambda: interview.render(engine, effective_settings, classifier_weights),
        title="3 · Interview",
        url_path="interview",
    ),
    "Assessment": st.Page(
        lambda: assessment.render(engine, effective_settings, classifier_weights),
        title="4 · Assessment",
        url_path="assessment",
    ),
    "Canvas": st.Page(
        lambda: canvas.render(engine, effective_settings, classifier_weights),
        title="Canvas",
        url_path="canvas",
    ),
    "Diamond Model": st.Page(
        lambda: diamond.render(engine, effective_settings, classifier_weights),
        title="Diamond Model",
        url_path="diamond-model",
    ),
    "CyberRE-LLM": st.Page(
        lambda: cyberre.render(engine, effective_settings, classifier_weights),
        title="CyberRE-LLM",
        url_path="cyberre",
    ),
    "Live Intel": st.Page(
        lambda: intel.render(engine, effective_settings, classifier_weights),
        title="Live Intel",
        url_path="live-intel",
    ),
}
st.session_state["_navigation_pages"] = pages
navigation = st.navigation(
    {
        "Workflow": [
            pages["Intake"],
            pages["Technique Review"],
            pages["Interview"],
            pages["Assessment"],
        ],
        "Analysis": [pages["Canvas"], pages["Diamond Model"], pages["CyberRE-LLM"]],
        "Intelligence": [pages["Live Intel"]],
    },
    position="top",
)
navigation.run()
