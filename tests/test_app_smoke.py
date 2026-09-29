from pathlib import Path

from streamlit.testing.v1 import AppTest

from apt_classifier.config import get_settings


def test_app_starts_and_requests_initial_ingestion(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    app = AppTest.from_file(str(Path("app/streamlit_app.py"))).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "APT Attribution Assistant"
    assert any("not been ingested" in item.value for item in app.info)
    get_settings.cache_clear()
