from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apt_classifier.evaluation import TechniqueEvaluationCase
from scripts.evidenceforge_eval import (
    ScenarioIntent,
    align_event_to_intent,
    build_narrative_case_set,
    build_narrative_text,
    build_source_case_set,
    collect_run_metadata,
    reject_conflicting_labels,
    run_cases,
)


def _intent() -> ScenarioIntent:
    return ScenarioIntent(
        record_id="evt-1#0",
        storyline_id="evt-1",
        event_type="process",
        technique_id="T1033",
        observable_attributes={"command_line": "id", "process_name": "/usr/bin/id"},
        section="storyline",
    )


def _event() -> dict:
    return {
        "record_id": "evt-1#0",
        "intent_id": "intent-1",
        "storyline_id": "evt-1",
        "kind": "process",
        "time": "2024-01-01T00:00:00Z",
        "system": "host-1",
        "activity": "LABEL-BEARING attacker prose",
        "description": "System Owner/User Discovery",
        "attributes": {
            "command_line": "id",
            "process_name": "/usr/bin/id",
            "technique": "T1033",
            "description": "label prose",
        },
    }


def test_narrative_excludes_all_label_bearing_prose() -> None:
    text = build_narrative_text(_event())

    assert "command_line: id" in text
    assert "process_name: /usr/bin/id" in text
    assert "LABEL-BEARING" not in text
    assert "System Owner/User Discovery" not in text
    assert "T1033" not in text
    assert "label prose" not in text


def test_exact_alignment_rejects_attribute_and_type_mismatches() -> None:
    event = _event()
    intent, reason = align_event_to_intent(event, [_intent()])
    assert intent == _intent()
    assert reason is None

    changed = {**event, "attributes": {"command_line": "whoami"}}
    assert align_event_to_intent(changed, [_intent()])[1] == "observable_attribute_mismatch"
    changed = {**event, "kind": "connection"}
    assert align_event_to_intent(changed, [_intent()])[1] == "event_type_mismatch"


def test_narrative_and_native_sources_are_separate_and_exact() -> None:
    event = _event()
    narrative = build_narrative_case_set([event], [_intent()])
    matching = {
        "timestamp_ms": 1704067200000,
        "hostname": "host-1",
        "object": "PROCESS",
        "action": "CREATE",
        "properties": {"command_line": "id", "image_path": "/usr/bin/id"},
    }
    nearby_but_wrong = {
        **matching,
        "properties": {"command_line": "uname -a", "image_path": "/usr/bin/uname"},
    }
    ecar = build_source_case_set(
        [event], [_intent()], [matching, nearby_but_wrong], source="ecar", window_seconds=5
    )
    windows = build_source_case_set([event], [_intent()], [], source="windows")

    assert narrative.cases[0].source == "evidenceforge_narrative"
    assert ecar.cases[0].source == "evidenceforge_ecar"
    assert ecar.cases[0].metadata["selected_record_count"] == 1
    assert not windows.cases
    assert windows.unsupported[0]["reason"] == "no_exact_observable_record_match"


def test_identical_input_with_conflicting_labels_is_rejected() -> None:
    cases = [
        TechniqueEvaluationCase("one", "doc", "Command: id", ("T1033",)),
        TechniqueEvaluationCase("two", "doc", "command id", ("T1057",)),
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        reject_conflicting_labels(cases)


def test_output_metadata_records_reproducibility_fields(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("storyline: []\n", encoding="utf-8")
    (tmp_path / "GROUND_TRUTH.json").write_text("{}", encoding="utf-8")
    (tmp_path / "OBSERVATION_MANIFEST.json").write_text("{}", encoding="utf-8")
    (tmp_path / "GENERATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "generation_seed": 19,
                "evidenceforge_version": "2.0.1",
                "evidenceforge_commit": "abc123",
                "compiled_sha256": "bundle-hash",
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        embedding_model="embed",
        embedding_model_revision="rev",
        technique_verifier_model="verify-model",
        reranker_model="reranker",
        reranker_model_revision="reranker-rev",
        technique_retrieval_mode="hybrid",
        technique_candidate_pool=12,
        retrieval_top_k=30,
        similarity_threshold=0.45,
        technique_acceptance_threshold=0.75,
        technique_relative_margin=0.08,
        attack_version="latest",
    )
    metadata = collect_run_metadata(
        bundle_dir=tmp_path,
        scenario_path=scenario,
        settings=settings,
        kb=SimpleNamespace(version="15.1"),
        preset="balanced",
        pipeline="local",
    )

    assert metadata["generation_seed"] == 19
    assert metadata["evidenceforge_version"] == "2.0.1"
    assert metadata["evidenceforge_commit"] == "abc123"
    assert metadata["scenario_hash"]
    assert metadata["ground_truth_hash"]
    assert metadata["bundle_manifest_hashes"]["GENERATION_MANIFEST.json"]
    assert metadata["attack_version"] == "15.1"
    assert metadata["models"]["embedding_revision"] == "rev"
    assert metadata["pipeline"]["preset"] == "balanced"
    assert metadata["pipeline_fingerprint"]


def test_local_run_uses_shared_mapper_with_verification_disabled() -> None:
    calls = []

    class FakeEngine:
        def map_techniques(self, ledger, *, verify, mode):
            calls.append((ledger.observations[0].text, verify, mode))
            candidate = SimpleNamespace(technique_id="T1033", fused_rank=1)
            retrieval = SimpleNamespace(candidates=(candidate,), observed_techniques={"T1033": 1})
            return retrieval, {}

    case = TechniqueEvaluationCase("case", "doc", "command_line: id", ("T1033",))
    predictions = run_cases([case], engine=FakeEngine(), verify=False)

    assert calls == [("command_line: id", False, "local")]
    assert predictions[0].accepted_technique_ids == ("T1033",)
