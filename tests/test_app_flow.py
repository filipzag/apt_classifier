"""End-to-end UI flow through the real local pipeline (no LLM).

Runs only when the ATT&CK knowledge base exists locally (the Chroma index must be reachable
too). It drives the app in local mode (no OpenRouter key): the Intake page analyses the
sample case and saves it, and every workflow page renders the analysed case without an
exception.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from apt_classifier.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb" / "attack_kb.json"
SAMPLE = ROOT / "data" / "samples" / "case_observation.txt"
PAGES = ("Technique Review", "Interview", "Assessment", "Canvas", "Diamond Model", "CyberRE-LLM")

pytestmark = pytest.mark.skipif(
    not KB.exists() or os.environ.get("APT_SKIP_APP_FLOW") == "1",
    reason="requires the ingested knowledge base and Chroma index",
)


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [str(item.value) for item in app.exception]
    assert not app.error, [str(item.value) for item in app.error]


@pytest.fixture
def local_mode(monkeypatch, tmp_path):
    # an empty env var overrides the key in .env, which is what makes the app run local-only
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("LOCAL_LLM_ENABLED", "false")
    get_settings.cache_clear()
    from app.ui import runtime

    monkeypatch.setattr(
        runtime, "case_store", lambda settings: runtime.CaseStore(tmp_path / "cases")
    )
    yield tmp_path / "cases"
    get_settings.cache_clear()


def test_intake_analyses_sample_in_local_mode_and_saves_the_case(local_mode):
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    app.run()
    _assert_clean(app)
    assert app.title[0].value == "Vestige"
    assert any("Local mode" in item.value for item in app.sidebar.warning)
    narrative = next(area for area in app.text_area if area.key == "intake_narrative")
    narrative.set_value(SAMPLE.read_text(encoding="utf-8"))
    app.run()
    analyse = next(button for button in app.button if "Analyse evidence" in button.label)
    analyse.click()
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    state = app.session_state["analysis_state"]
    assert state.ledger.observations
    assert state.retrieval.candidates
    assert list(local_mode.glob("*.json")), "the analysed case should be saved automatically"


def _page_script(page_key: str) -> None:
    from pathlib import Path

    import streamlit as st

    import apt_classifier
    from app.ui import runtime
    from app.ui.local_extract import local_observations
    from app.ui.pages import assessment, canvas, cyberre, diamond, interview, technique_review
    from apt_classifier.config import get_settings
    from apt_classifier.evidence.models import EvidenceLedger

    renderers = {
        "Technique Review": technique_review.render,
        "Interview": interview.render,
        "Assessment": assessment.render,
        "Canvas": canvas.render,
        "Diamond Model": diamond.render,
        "CyberRE-LLM": cyberre.render,
    }
    settings = get_settings()
    engine = runtime.get_engine(settings)
    if runtime.current_state() is None:
        root = Path(apt_classifier.__file__).resolve().parents[1]
        sample = (root / "data" / "samples" / "case_observation.txt").read_text(encoding="utf-8")
        ledger = EvidenceLedger(observations=local_observations(sample))
        ledger.case_id = "test-case"
        st.session_state[runtime.CASE_NAME_KEY] = "flow test"
        runtime.set_state(
            runtime.reanalyze(engine, ledger, engine.default_weights(), previous=None)
        )
    st.session_state["_navigation_pages"] = {}
    renderers[page_key](engine, settings, engine.default_weights())


@pytest.mark.parametrize("page_key", PAGES)
def test_every_page_renders_the_analysed_case(local_mode, page_key):
    app = AppTest.from_function(_page_script, kwargs={"page_key": page_key}, default_timeout=300)
    app.run()
    _assert_clean(app)
    assert app.session_state["analysis_state"].fusion.entries
    if page_key == "Assessment":
        # exports are rendered without the AI review (the section header precedes the buttons)
        assert any("Export" in item.value for item in app.markdown)
        assert app.session_state["analysis_state"].review is None
