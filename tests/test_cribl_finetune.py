"""The dependency-free parts of the cribl-decision experiment (scripts/cribl_finetune)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from scripts.cribl_finetune import common
from scripts.cribl_finetune.export_cases import (
    build_catalog,
    case_row,
    drop_near_duplicates,
    known_candidates,
)


def _options(count: int) -> list[dict[str, str]]:
    candidates = [
        common.technique_option(f"T10{index:02d}", f"Name {index}", f"Text {index}.")
        for index in range(count)
    ]
    return common.c1_options(candidates)


def test_cut_description_keeps_short_text_and_cuts_at_sentence_end():
    assert common.cut_description("Short  text.") == "Short text."
    text = "First sentence here. " + "x" * 30 + ". " + "Word " * 120
    cut = common.cut_description(text, limit=60)
    assert cut == "First sentence here. " + "x" * 30 + "."


def test_cut_description_skips_abbreviations_and_lowercase_continuations():
    text = "Use tools, e.g. curl and wget to fetch. " + "More words follow here " * 30
    assert common.cut_description(text, limit=50) == "Use tools, e.g. curl and wget to fetch."
    no_end = "word " * 200
    cut = common.cut_description(no_end, limit=23)
    assert cut.endswith("…") and not cut[:-1].endswith(" ") and len(cut) <= 24


def test_combine_rotations_geometric_mean_with_floor():
    single = common.combine_rotations([{"a": 0.7, "b": 0.3}], ["a", "b"])
    assert single == {"a": 0.7, "b": 0.3}
    combined = common.combine_rotations([{"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}], ["a", "b"])
    a = math.sqrt(1.0 * 0.5)
    b = math.sqrt(0.005 * 0.5)
    assert combined["a"] == pytest.approx(a / (a + b))
    assert math.fsum(combined.values()) == pytest.approx(1.0)


def test_rotation_keeps_none_last_and_counts():
    options = _options(10)
    assert len(options) == 11 and options[-1]["id"] == common.NONE_OF_THESE
    rotated = common.rotate(options, 1)
    assert rotated[-1]["id"] == common.NONE_OF_THESE
    assert [item["id"] for item in rotated[:-1]] == [item["id"] for item in options[1:-1]] + [
        options[0]["id"]
    ]
    assert common.rotation_count(options) == 3
    assert common.rotation_count(_options(1)) == 1


def test_c1_options_cap_at_ten_and_label_name_id():
    options = _options(12)
    assert len(options) == 11
    assert options[0]["label"] == "Name 0 (T1000)" and options[0]["what"] == "Text 0."


@pytest.mark.parametrize("fmt", common.FORMATS)
def test_render_prompt_letters_and_decision_slot(fmt):
    options = _options(3)
    prompt = common.render_prompt(
        common.c1_state("Ran whoami."), common.C1_INSTRUCTIONS, options, fmt
    )
    assert prompt.endswith(common.ANSWER_CUE)
    assert "Name 2 (T1002)" in prompt and common.NONE_CRITERION in prompt
    marker = '"D"' if fmt == "json" else "D. "
    assert marker in prompt and ('"E"' if fmt == "json" else "E. ") not in prompt
    assert common.letter_tokens(4) == [" A", " B", " C", " D"]


def test_render_prompt_refuses_too_many_options():
    with pytest.raises(ValueError):
        common.render_prompt("s", "q", [{"label": "x", "what": "y"}] * 27)


def test_target_distribution_gold_or_none():
    options = _options(3)
    assert common.target_distribution(options, ["T1001"]) == [0.0, 1.0, 0.0, 0.0]
    assert common.target_distribution(options, ["T1000", "T1002"]) == [0.5, 0.0, 0.5, 0.0]
    assert common.target_distribution(options, ["T9999"]) == [0.0, 0.0, 0.0, 1.0]


def test_hits_exact_and_parent_aware():
    assert common.hits("T1059.001", ["T1059.001"]) == (True, True)
    assert common.hits("T1059", ["T1059.001"]) == (False, True)
    assert common.hits("T1059.003", ["T1059.001"]) == (False, True)
    assert common.hits("T1082", ["T1059"]) == (False, False)
    assert common.hits(None, ["T1059"]) == (False, False)


def test_pick_from_none_and_ties():
    assert common.pick_from({"a": 0.2, common.NONE_OF_THESE: 0.8}, ["a", common.NONE_OF_THESE]) == (
        None,
        0.8,
    )
    assert common.pick_from({"b": 0.5, "a": 0.5}, ["a", "b"]) == ("a", 0.5)


def test_forced_pick_never_abstains_and_renormalises():
    distribution = {"a": 0.1, "b": 0.3, common.NONE_OF_THESE: 0.6}
    label, probability = common.forced_pick(distribution, ["a", "b"])
    assert label == "b" and probability == pytest.approx(0.75)
    assert common.forced_pick({"b": 0.2, "a": 0.2, common.NONE_OF_THESE: 0.6}, ["a", "b"])[0] == "a"
    assert common.forced_pick({common.NONE_OF_THESE: 1.0}, []) == (None, None)
    row = common.forced_row({"case_id": "1", "distribution": distribution}, ["a", "b"])
    assert (row["pick"], row["p_none"]) == ("b", 0.6)
    assert common.forced_row({"case_id": "1", "status": "error"}, [])["pick"] is None


def test_score_block_counts_misses_and_ece():
    rows = [
        {"gold_ids": ["T1"], "pick": "T1", "p_pick": 0.9},
        {"gold_ids": ["T1"], "pick": "T2", "p_pick": 0.6},
        {"gold_ids": ["T1"], "pick": None, "p_pick": None},
    ]
    block = common.score_block(rows)
    assert block["exact"] == pytest.approx(1 / 3, abs=1e-6)
    assert block["no_pick"] == 1
    assert block["ece"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)


def test_paired_delta_and_bootstrap():
    base = [{"case_id": str(i), "dataset": "d", "gold_ids": ["T1"], "pick": "T2"} for i in range(4)]
    arm = [{**row, "pick": "T1" if i < 2 else "T2"} for i, row in enumerate(base)]
    delta = common.paired_delta(base, arm, samples=50)
    assert delta["pooled"]["delta"]["exact"] == 0.5
    low, high = delta["pooled"]["ci95"]["exact"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert common.bootstrap_mean_differences({"x": []}) is None


def test_reference_rows_use_status():
    cases = [
        {
            "case_id": "1",
            "dataset": "d",
            "gold_ids": ["T1"],
            "pipeline_rank1": "T1",
            "jev_c1": {"status": "no_pick", "pick": None, "p_final": None},
        },
        {
            "case_id": "2",
            "dataset": "d",
            "gold_ids": ["T1"],
            "pipeline_rank1": "T2",
            "jev_c1": None,
        },
    ]
    assert [row["pick"] for row in common.reference_rows(cases, "pipeline")] == ["T1", "T2"]
    jev = common.reference_rows(cases, "jev_c1")
    assert len(jev) == 1 and jev[0]["pick"] is None
    cases[0]["candidates"] = [{"id": "T2"}, {"id": "T1"}]
    cases[0]["jev_c1"]["distribution"] = {"T1": 0.2, "T2": 0.2, common.NONE_OF_THESE: 0.6}
    forced = common.reference_rows(cases, "jev_c1_forced")
    assert len(forced) == 1 and forced[0]["pick"] == "T2"  # a Jev no-pick is re-decided
    cases[0]["jev_c1"]["status"] = "error"
    assert common.reference_rows(cases, "jev_c1_forced")[0]["pick"] is None


def test_rescore_auroc_counts_ties_half():
    from scripts.cribl_finetune.rescore import auroc

    assert auroc([(0.9, True), (0.1, False)]) == 1.0
    assert auroc([(0.5, True), (0.5, False)]) == 0.5
    assert auroc([(0.2, True), (0.8, False), (0.5, True)]) == 0.0
    assert auroc([(0.2, True)]) is None


def _technique(technique_id, tactics=(), parent=None):
    return SimpleNamespace(
        id=technique_id,
        name=f"Name {technique_id} (Citation: X)",
        description="Does things. (Citation: Y)",
        tactics=tactics,
        is_subtechnique=parent is not None,
        parent_id=parent,
    )


def test_build_catalog_follows_judge_rules():
    bundle = {
        "objects": [
            {"type": "x-mitre-tactic", "x_mitre_shortname": "discovery"},
            {"type": "x-mitre-tactic", "x_mitre_shortname": "gone", "revoked": True},
            {
                "type": "attack-pattern",
                "revoked": True,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T1003"}],
            },
        ]
    }
    kb = SimpleNamespace(
        techniques={
            "T1001": _technique("T1001", ("discovery",)),
            "T1001.001": _technique("T1001.001", parent="T1001"),
            "T1002": _technique("T1002", ("gone",)),
            "T1003": _technique("T1003", ("discovery",)),
        }
    )
    catalog = build_catalog(bundle, kb)
    assert list(catalog) == ["T1001", "T1001.001"]
    assert catalog["T1001"]["description"] == "Does things."
    assert known_candidates(catalog, ["T1002", "T1001.001", "T1001", "T1001"]) == [
        "T1001.001",
        "T1001",
    ]
    row = case_row(
        {
            "case_id": "c",
            "dataset": "tram",
            "text": "t",
            "expected_technique_ids": ["T1001"],
            "ranked_technique_ids": ["T1002", "T1001"],
        },
        catalog,
        "test",
        ["T1001"],
    )
    assert row["gold_in_candidates"] and row["pipeline_rank1"] == "T1002"
    assert row["candidates"][0]["label"] == "Name T1001 (T1001)"


def test_drop_near_duplicates():
    rows = [
        {"text": "The actor ran whoami on the host."},
        {"text": "Totally different words."},
        {"text": "Totally different words!"},
    ]
    kept, counts = drop_near_duplicates(rows, ["the actor ran WHOAMI on the host"])
    assert [row["text"] for row in kept] == ["Totally different words."]
    assert counts == {"held_out_duplicates": 1, "train_repeats": 1}
