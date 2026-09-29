"""Unit tests for the UI support modules (no Streamlit server needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ui.cases import CaseStore, safe_case_id
from app.ui.components import STAGE_INFO, candidate_state_label, next_step, stage_badge
from app.ui.guard import describe_error
from app.ui.local_extract import local_observations
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance


def test_case_store_roundtrips_ledger_and_lists_newest_first(tmp_path):
    store = CaseStore(tmp_path / "cases")
    ledger = EvidenceLedger(
        observations=[
            Observation(
                text="The actor dumped LSASS memory.", provenance=Provenance(source="free_text")
            )
        ]
    )
    ledger.case_id = "abc123"
    path = store.save(
        case_id="abc123",
        name="INC-1",
        ledger=ledger,
        weights={"naive_bayes": 1.0},
        summary={"observations": 1, "stage": "known_group_shortlist", "leading_group": "G0016"},
    )
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["kind"] == "apt_case" and payload["name"] == "INC-1"
    second = EvidenceLedger(
        observations=[
            Observation(text="Scheduled task created.", provenance=Provenance(source="free_text"))
        ]
    )
    second.case_id = "def456"
    store.save(case_id="def456", name="INC-2", ledger=second, summary={"observations": 1})
    records = store.list()
    assert [record.case_id for record in records][0] == "def456"
    assert records[1].stage == "known_group_shortlist"
    loaded = store.load("abc123")
    assert loaded["name"] == "INC-1"
    assert loaded["ledger"].observations[0].text == "The actor dumped LSASS memory."
    assert loaded["ledger"].case_id == "abc123"
    assert loaded["weights"] == {"naive_bayes": 1.0}
    assert store.delete("abc123") and not store.delete("abc123")
    assert [record.case_id for record in store.list()] == ["def456"]


def test_case_store_rejects_unsafe_ids(tmp_path):
    store = CaseStore(tmp_path)
    with pytest.raises(ValueError):
        safe_case_id("../etc/passwd")
    with pytest.raises(ValueError):
        store.load("../x")
    (tmp_path / "junk.json").write_text("{not json")
    (tmp_path / "other.json").write_text(json.dumps({"kind": "something_else"}))
    assert store.list() == []


def test_local_observations_split_sentences_with_provenance():
    text = "First line. The actor ran mimikatz and dumped credentials.\n\nok\nA scheduled task persisted the loader."
    observations = local_observations(text)
    texts = [o.text for o in observations]
    assert "The actor ran mimikatz and dumped credentials." in texts
    assert "A scheduled task persisted the loader." in texts
    assert all(len(t.split()) >= 3 for t in texts)  # "ok" and "First line." dropped
    first = next(o for o in observations if o.text.startswith("The actor"))
    assert first.provenance.source == "free_text"
    assert first.provenance.line_numbers == [1]
    assert first.provenance.metadata["normalization"] == "local_sentence_split"
    assert local_observations("") == []
    # duplicates collapse
    assert len(local_observations("Dumped LSASS memory now.\nDumped LSASS memory now.")) == 1


def test_describe_error_maps_operational_failures():
    assert "credit" in describe_error(RuntimeError("Client error '402 Payment Required'"))
    assert "API key" in describe_error(RuntimeError("401 Unauthorized"))
    assert "rate limit" in describe_error(RuntimeError("HTTP 429 rate limit"))
    assert "timed out" in describe_error(TimeoutError("read timed out"))
    assert "Chroma" in describe_error(ConnectionError("connection refused"))
    assert describe_error(ValueError("weird")).startswith("The step failed")


def test_stage_and_candidate_labels_are_textual_not_colour_only():
    for stage, info in STAGE_INFO.items():
        assert info["label"] in stage_badge(stage)
    assert "No analysis" in stage_badge(None)
    assert "Accepted" in candidate_state_label("auto_accepted")
    assert "Rejected" in candidate_state_label("auto_rejected")


def _fake_state(*, entries, candidates, reviews=(), answers=(), complete=False):
    ledger = SimpleNamespace(
        observations=[1, 2], technique_reviews=list(reviews), question_answers=list(answers)
    )
    retrieval = SimpleNamespace(
        candidates=[SimpleNamespace(state=SimpleNamespace(value=v)) for v in candidates]
    )
    fusion = SimpleNamespace(entries=list(entries))
    return SimpleNamespace(
        ledger=ledger, retrieval=retrieval, fusion=fusion, interview_complete=complete
    )


def test_next_step_guides_through_the_workflow():
    assert next_step(None)[1] == "Intake"
    assert next_step(_fake_state(entries=[], candidates=["needs_review"]))[1] == "Intake"
    message, page = next_step(
        _fake_state(entries=[1], candidates=["needs_review", "auto_accepted"])
    )
    assert page == "Technique Review" and "1 technique" in message
    assert next_step(_fake_state(entries=[1], candidates=["auto_accepted"]))[1] == "Interview"
    assert (
        next_step(_fake_state(entries=[1], candidates=["auto_accepted"], answers=[1]))[1]
        == "Assessment"
    )
    assert (
        next_step(
            _fake_state(entries=[1], candidates=["needs_review"], reviews=[1], complete=True)
        )[1]
        == "Assessment"
    )
