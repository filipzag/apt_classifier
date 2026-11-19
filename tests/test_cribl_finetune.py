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


def test_family_target_spreads_over_a_gold_parents_family():
    options = common.c1_options(
        [
            common.technique_option(item, item, "x")
            for item in ("T1566.001", "T1566", "T1059.001", "T1027")
        ]
    )
    assert common.family_target_distribution(options, ["T1566"]) == [0.5, 0.5, 0.0, 0.0, 0.0]
    assert common.family_target_distribution(options, ["T1059.001"]) == [0, 0, 1.0, 0, 0]
    assert common.family_target_distribution(options, ["T1059"]) == [0, 0, 1.0, 0, 0]
    assert common.family_target_distribution(options, ["T1105"]) == [0, 0, 0, 0, 1.0]


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


def test_annoctr_sentences_group_technique_links():
    from scripts.cribl_finetune.prepare_annoctr import (
        clean_sentence,
        record_sentence,
        sentence_cases,
        technique_id,
    )

    assert clean_sentence("top vector for [ransomware](https://x.com/a_(b)) and  more") == (
        "top vector for ransomware and more"
    )
    assert clean_sentence('follow-on [malware attacks](/us/malware "Malware").![](data:x)') == (
        "follow-on malware attacks."
    )
    # AnnoCTR's context_* fields are stripped; _context_* keep the spaces around the mention
    glued = {"context_left": "then installs", "mention": "ransomware", "context_right": "now."}
    assert record_sentence(glued) == "then installsransomwarenow."
    spaced = {**glued, "_context_left": "then installs ", "_context_right": " now."}
    assert record_sentence(spaced) == "then installs ransomware now."
    assert technique_id("https://attack.mitre.org/techniques/T1566/001") == "T1566.001"
    assert technique_id("https://attack.mitre.org/software/S0332") is None
    record = {"document": "d", "_context_left": "They used ", "_context_right": " to log in."}
    records = [
        {
            **record,
            "mention": "[phishing](u)",
            "entity_type": "TECHNIQUE",
            "entity_class": "CE",
            "label_link": "https://attack.mitre.org/techniques/T1566",
        },
        {
            **record,
            "mention": "[phishing](u)",
            "entity_type": "TECHNIQUE",
            "entity_class": "CI",
            "label_link": "https://attack.mitre.org/techniques/T1078",
        },
        {
            **record,
            "mention": "Emotet",
            "entity_type": "MALWARE",
            "entity_class": "CE",
            "label_link": "https://attack.mitre.org/software/S0367",
        },
    ]
    (case,) = sentence_cases(records, "train")
    assert case["text"] == "They used phishing to log in."
    assert case["original_technique_ids"] == ["T1566", "T1078"]
    assert case["metadata"]["mention_classes"] == {"CE": 1, "CI": 1}


def test_family_question_has_no_options_and_targets_gold_families():
    prompt = common.render_family_prompt(common.c1_state("They  used PowerShell."))
    assert prompt.endswith(f"<think>\n\n</think>\n\n{common.ANSWER_CUE}")
    assert '"sentence": "They used PowerShell."' in prompt
    assert "Options:" not in prompt
    families = ["T1059", "T1566", "T1486"]
    assert common.family_targets(["T1566.001", "T1059.001", "T1566"], families) == [1, 0]
    assert common.family_targets(["T9999"], families) == []


def test_multi_family_adds_confident_families_and_scores_micro_f1():
    from scripts.cribl_finetune.multi_family import TOP1_ONLY, families_named, micro_prf

    row = {
        "gold_ids": ["T1059.001", "T1105"],
        "top": [["T1059", 0.5], ["T1105", 0.3], ["T1027", 0.1], ["T1566", 0.05]],
    }
    assert families_named(row, TOP1_ONLY) == ["T1059"]
    assert families_named(row, 0.25) == ["T1059", "T1105"]
    assert families_named(row, 0.05) == ["T1059", "T1105", "T1027"]  # at most three
    assert micro_prf([row], TOP1_ONLY)["f1"] == pytest.approx(2 / 3)
    assert micro_prf([row], 0.25)["f1"] == 1.0


def test_rival_margin_is_zero_past_the_margin_and_pushes_gold_up():
    torch = pytest.importorskip("torch")
    from scripts.cribl_finetune.losses import rival_margin

    logits = torch.tensor([3.0, 0.5, 0.0, -1.0], requires_grad=True)
    log_p = torch.log_softmax(logits, dim=0)
    assert float(rival_margin(log_p, [0], 1.0)) == 0.0  # gold beats the rival by 2.5 nats
    close = torch.tensor([1.0, 0.8, 0.0, -1.0], requires_grad=True)
    term = rival_margin(torch.log_softmax(close, dim=0), [0], 1.0)
    assert float(term) == pytest.approx(0.8)
    term.backward()
    assert close.grad[0] < 0 < close.grad[1]  # gradient descent raises gold, lowers the rival
    two_gold = torch.log_softmax(torch.tensor([1.0, 1.0, 2.0]), dim=0)
    # the gold families' mass counts together: gap = log(2e) - 2 = ln 2 - 1, term = 2 - ln 2
    assert float(rival_margin(two_gold, [0, 1], 1.0)) == pytest.approx(2 - math.log(2), abs=1e-5)


def test_technique_head_scores_within_family_and_picks_among_top_families():
    torch = pytest.importorskip("torch")
    from scripts.cribl_finetune.hierarchy import (
        joint_log_probs,
        pick,
        technique_index,
        technique_targets,
        within_family_log_probs,
    )

    texts = {
        "T1059": [{"id": "T1059"}, {"id": "T1059.001"}, {"id": "T1059.003"}],
        "T1105": [{"id": "T1105"}],
    }
    techniques, group = technique_index(["T1059", "T1105"], texts)
    assert techniques == ["T1059", "T1059.001", "T1059.003", "T1105"]
    assert group == [0, 0, 0, 1]
    assert technique_targets(["T1059.001", "T9999", "T1059.001"], techniques) == [1]
    group = torch.tensor(group)
    logits = torch.tensor([[0.0, 2.0, 1.0, 5.0]], requires_grad=True)
    within = within_family_log_probs(logits, group, 2)
    expected = torch.log_softmax(torch.tensor([0.0, 2.0, 1.0]), dim=0)
    assert torch.allclose(within[0, :3], expected)
    assert float(within[0, 3]) == pytest.approx(0.0)  # a family without sub-techniques: P = 1
    (-within[0, 1]).backward()
    assert logits.grad[0, 1] < 0 < logits.grad[0, 2]  # descent raises the gold sibling only
    assert float(logits.grad[0, 3]) == 0.0
    family_log_p = torch.log(torch.tensor([[0.55, 0.45]]))
    joint = joint_log_probs(family_log_p, within.detach(), group)
    assert float(joint.exp().sum()) == pytest.approx(1.0)
    # P(T1105) = 0.45 beats P(T1059.001) = 0.55 * 0.665: the second family's lone technique wins
    assert pick(joint[0], family_log_p[0], group, None) == 3
    assert pick(joint[0], family_log_p[0], group, 1) == 1  # limited to the top family


def test_advisory_sentences_keep_tagged_behaviour_and_strip_tags():
    from scripts.cribl_finetune.prepare_advisories import (
        listing_links,
        names_present,
        resolve_ids,
        tagged_sentences,
        technique_name_only,
    )

    text = (
        "TLP:CLEAR\n\nBianLian group actors use PowerShell [T1059.001] and Windows Command\n"
        "Shell [T1059.003] to disable antivirus [TA0005]. They read the registry.\n"
        "Page 3 of 20 | Product ID: AA24-001A\n\n"
        "End Update The actors used Cobalt Strike [S0154] for command and control (T1071.001).\n\n"
        "Indicator Removal: File Deletion [T1070.004]\n"
    )
    found = tagged_sentences(text)
    assert found[0] == (
        "BianLian group actors use PowerShell and Windows Command Shell to disable antivirus.",
        ["T1059.001", "T1059.003"],
    )
    assert found[1] == (
        "The actors used Cobalt Strike for command and control.",
        ["T1071.001"],
    )
    assert technique_name_only(found[2][0], {"indicator removal file deletion"})
    assert len(found) == 3  # the untagged sentence is not a row
    assert resolve_ids(["T1064", "T1059.001", "T9999"], {"T1064": "T1059"}, {"T1059"}) == [
        "T1059",
        "T1059.001",
    ]
    assert names_present("Conti actors and TrickBot", {"G0102": ["TrickBot", "Wizard Spider"]}) == [
        "G0102: TrickBot"
    ]
    assert names_present("NotTrickBot-ish", {"G0102": ["TrickBot"]}) == []
    html = '<a href="/CSA/2024/241120.pdf">x</a><a href="/CSA/2024/241120.pdf">y</a>'
    assert listing_links(html + '<a href="/CSA/2023/230101.pdf">', 2024) == ["2024/241120"]


def test_cleaning_rules_drop_broken_rows_and_keep_short_sentences():
    from scripts.cribl_finetune.clean_cases import broken_reason

    assert broken_reason("has utilized  during and.") == "fragment"
    assert broken_reason("*Figure 5: Encrypted files*") == "figure"
    assert broken_reason("| DarkSide | August 2020 | Under 5 | Exploit | Yes |") == "table_row"
    assert broken_reason("T1020 Automated exfiltration") == "technique_id"
    hashes = " ".join(["a" * 64, "b" * 64, "c" * 64])
    assert broken_reason(f"Filename {hashes}") == "ioc_list"
    for text in (
        "uses DNS for C2.",
        "T9000 can take screenshots of the desktop.",
        "Figure 4 shows a PDF that pretends to be a Request for Quotation for an expansion plan.",
    ):
        assert broken_reason(text) is None, text


def test_atomic_rows_are_name_and_description_without_ids_or_placeholders():
    from scripts.cribl_finetune.prepare_atomic import atomic_tests, atomic_text

    test = {
        "name": "Dump LSASS via comsvcs",
        "auto_generated_guid": "g1",
        "description": "Uses `rundll32` to dump #{process} for T1003.001 testing.",
    }
    index = {
        "credential-access": {"T1003.001": {"atomic_tests": [test]}},
        "defense-evasion": {"T1003.001": {"atomic_tests": [test]}},
    }
    assert [(t, g) for t, g, _ in atomic_tests(index)] == [("T1003.001", "g1")]
    assert atomic_text(test) == "Dump LSASS via comsvcs. Uses rundll32 to dump for testing."


def test_cti_rationale_paragraphs_join_gold_rows_and_keep_first_order():
    from scripts.cribl_finetune.prepare_cti_rationale import paragraphs, statement_texts

    gold = [
        {
            "report_id": 7,
            "chunk_id": "P1",
            "vendor": "v",
            "chunk_text": "A  b\nc.",
            "technique_id": "T1059.001",
        },
        {
            "report_id": 7,
            "chunk_id": "P1",
            "vendor": "v",
            "chunk_text": "A  b\nc.",
            "technique_id": "T1105",
        },
        {
            "report_id": 7,
            "chunk_id": "P1",
            "vendor": "v",
            "chunk_text": "A  b\nc.",
            "technique_id": "T1059.001",
        },
        {
            "report_id": 7,
            "chunk_id": "P2",
            "vendor": "v",
            "chunk_text": "D.",
            "technique_id": "T1486",
        },
    ]
    found = paragraphs(gold)
    assert [(p["report_id"], p["chunk_id"], p["text"], p["ids"]) for p in found] == [
        ("7", "P1", "A b c.", ["T1059.001", "T1105"]),
        ("7", "P2", "D.", ["T1486"]),
    ]
    rows = [{"statements": [{"text": "x"}, {"text": ""}, {"section": "s"}]}]
    assert statement_texts(rows) == ["x"]


def test_yes_no_loss_scores_each_family_on_its_own():
    torch = pytest.importorskip("torch")
    from scripts.cribl_finetune.losses import yes_no

    logits = torch.tensor([2.0, 1.0, -3.0, -4.0], requires_grad=True)
    loss = yes_no(logits, [0, 1], 0.1)
    softplus = torch.nn.functional.softplus
    expected = softplus(torch.tensor(-2.0)) + softplus(torch.tensor(-1.0))
    expected += 0.1 * (softplus(torch.tensor(-3.0)) + softplus(torch.tensor(-4.0)))
    assert float(loss) == pytest.approx(float(expected), abs=1e-6)
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] < 0  # both labelled families pushed up
    assert logits.grad[2] > 0 and logits.grad[3] > 0  # the others pushed down, weakly
    assert float(yes_no(torch.tensor([2.0, 5.0]), [0], 0.0)) == pytest.approx(
        float(softplus(torch.tensor(-2.0))), abs=1e-6
    )  # weight 0: an unlabelled family costs nothing however high it scores


def test_multi_answer_matches_the_budget_and_names_each_family_best_technique():
    from scripts.cribl_finetune.multi_answer import answers, budget_threshold, hits, multi_block

    def row(case, top, gold, techniques=()):
        return {
            "case_id": case,
            "status": "ok",
            "top": top,
            "technique_top": list(techniques),
            "gold_ids": gold,
            "gold_families": sorted({item.split(".")[0] for item in gold}),
        }

    rows = [
        row(
            "a",
            [["T1566", 0.6], ["T1204", 0.3], ["T1059", 0.05]],
            ["T1204.002"],
            [["T1566.001", 0.5], ["T1204.002", 0.25], ["T1204", 0.05], ["T1059.001", 0.04]],
        ),
        row("b", [["T1105", 0.9], ["T1059", 0.06], ["T1027", 0.02]], ["T1105"], [["T1105", 0.9]]),
        row("c", [["T1027", 0.5], ["T1140", 0.4], ["T1001", 0.08]], ["T1140", "T1027"]),
        {"case_id": "d", "status": "error", "gold_ids": ["T1003"], "gold_families": ["T1003"]},
    ]
    assert budget_threshold(rows, 1.0) == math.inf
    cut = budget_threshold(rows, 1.5)  # two extra answers over four sentences: 0.4 and 0.3
    assert cut == 0.3
    assert answers(rows[0], cut) == [("T1566", "T1566.001"), ("T1204", "T1204.002")]
    assert answers(rows[3], cut) == []  # an error row answers nothing (a miss)
    assert hits(rows[0], cut) == (True, True) and hits(rows[0], math.inf) == (False, False)
    block = multi_block(rows, cut)
    assert block["answers_per_sentence"] == pytest.approx(5 / 4)
    assert block["family_hit"] == pytest.approx(3 / 4)
    assert block["several_labels_all_found"] == 1.0  # c: both labelled families answered
    assert (block["extras"], block["extras_matching_a_label"]) == (2, 2)


def test_multi_answer_recall_is_the_share_of_labels_found():
    from scripts.cribl_finetune.multi_answer import recall

    row = {
        "case_id": "p",
        "status": "ok",
        "top": [["T1566", 0.9], ["T1204", 0.6], ["T1059", 0.2]],
        "technique_top": [["T1566.001", 0.8], ["T1204.002", 0.5], ["T1059.001", 0.1]],
        "gold_ids": ["T1566.001", "T1204.002", "T1105"],
        "gold_families": ["T1105", "T1204", "T1566"],
    }
    assert recall(row, 0.5) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert recall(row, math.inf) == (pytest.approx(1 / 3), pytest.approx(1 / 3))


def test_parent_relabel_keeps_order_and_drops_repeats():
    from scripts.cribl_finetune.relabel_parents import parent_labels, relabel

    assert parent_labels(["T1059.001", "T1105", "T1059.003", "T1059"]) == ["T1059", "T1105"]
    rows, counts = relabel(
        [
            {"case_id": "a", "gold_ids": ["T1059.001", "T1059.003"], "text": "x"},
            {"case_id": "b", "gold_ids": ["T1105"], "text": "y"},
        ]
    )
    assert [row["gold_ids"] for row in rows] == [["T1059"], ["T1105"]]
    assert rows[0]["text"] == "x" and rows[0]["case_id"] == "a"
    assert counts == {"cases": 2, "labels_before": 3, "labels_after": 2, "cases_changed": 1}


def test_parent_rescore_forgives_granularity_but_not_the_family():
    from scripts.cribl_finetune.parent_rescore import block, compare, measures

    def row(case_id, pick, technique, gold, families=None):
        out = {
            "case_id": case_id,
            "dataset": "d",
            "status": "ok",
            "gold_ids": gold,
            "gold_families": families or sorted({g.split(".")[0] for g in gold}),
            "pick": pick,
            "top": [[pick, 0.8]],
        }
        if technique is not None:
            out["technique_pick"] = technique
        return out

    sub_for_parent = row("a", "T1059", "T1059.001", ["T1059"])  # granularity: forgiven
    parent_for_sub = row("b", "T1059", "T1059", ["T1059.003"])  # granularity: forgiven
    sibling = row("c", "T1059", "T1059.001", ["T1059.003"])  # same family, wrong sibling
    wrong = row("d", "T1105", "T1105", ["T1059.001"])
    assert measures(sub_for_parent) == {"family_top1": 1.0, "parent_level": 1.0, "exact": 0.0}
    assert measures(parent_for_sub)["parent_level"] == 1.0
    assert measures(sibling) == {"family_top1": 1.0, "parent_level": 1.0, "exact": 0.0}
    assert measures(wrong) == {"family_top1": 0.0, "parent_level": 0.0, "exact": 0.0}
    family_only = row("e", "T1059", None, ["T1059.001"])  # no technique head: the family answer
    assert measures(family_only) == {"family_top1": 1.0, "parent_level": 1.0, "exact": None}
    error = {"case_id": "f", "dataset": "d", "status": "error", "gold_ids": ["T1003"]}
    assert measures(error) == {"family_top1": 0.0, "parent_level": 0.0, "exact": 0.0}
    scored = block([sub_for_parent, parent_for_sub, sibling, wrong])
    assert (scored["parent_level"], scored["exact"]) == (0.75, 0.0)
    assert block([family_only])["exact"] is None
    delta = compare([wrong], [row("d", "T1059", "T1059.001", ["T1059.001"])])
    assert delta["delta"] == {"family_top1": 1.0, "parent_level": 1.0, "exact": 1.0}


def test_family_prompt_templates_share_the_body():
    from scripts.cribl_finetune.common import ANSWER_CUE, c1_state, render_family_prompt

    state = c1_state("The actor ran PowerShell.")
    qwen, gemma = render_family_prompt(state), render_family_prompt(state, "gemma")
    assert qwen == render_family_prompt(state, "qwen")
    assert qwen.startswith("<|im_start|>user\n") and qwen.endswith(f"</think>\n\n{ANSWER_CUE}")
    assert gemma.startswith("<bos><start_of_turn>user\n")
    assert gemma.endswith(f"<end_of_turn>\n<start_of_turn>model\n{ANSWER_CUE}")
    body = qwen.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert body in gemma and "The actor ran PowerShell." in body
    llama = render_family_prompt(state, "llama3")
    assert llama.startswith("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n")
    assert llama.endswith(f"<|start_header_id|>assistant<|end_header_id|>\n\n{ANSWER_CUE}")
    assert body in llama
    with pytest.raises(ValueError):
        render_family_prompt(state, "llama")


def test_lora_shape_and_base_registry():
    pytest.importorskip("peft")
    from scripts.cribl_finetune.decision_model import BASES, DEFAULT_BASE, base_spec, lora_shape

    assert lora_shape() == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none"}
    assert lora_shape(64)["lora_alpha"] == 128 and lora_shape(8, 8)["lora_alpha"] == 8
    with pytest.raises(ValueError):
        lora_shape(0)
    assert base_spec(DEFAULT_BASE).prompt_template == "qwen"
    assert base_spec("gemma-4-e4b").prompt_template == "gemma"
    assert base_spec("gemma-4-e4b").loader == "image_text_to_text"
    assert base_spec("gemma-4-e4b").cpu_modules == ("model.language_model.embed_tokens_per_layer",)
    assert {spec.repo for spec in BASES.values()} == {
        "Qwen/Qwen3.5-4B-Base",
        "google/gemma-4-E4B",
        "fdtn-ai/Foundation-Sec-8B",
    }
    assert base_spec("foundation-sec-8b").loader == "causal_lm"
    assert base_spec("foundation-sec-8b").prompt_template == "llama3"
    with pytest.raises(ValueError):
        base_spec("llama")


def test_group_prompt_and_state():
    from scripts.cribl_finetune.common import (
        ANSWER_CUE,
        GROUP_INSTRUCTIONS,
        group_state,
        render_family_prompt,
    )

    state = group_state(
        "T1059.001 Command and Scripting Interpreter: PowerShell; T1105 Ingress Tool Transfer"
    )
    assert set(state) == {"techniques", "source"}
    prompt = render_family_prompt(state, "qwen", "group")
    assert GROUP_INSTRUCTIONS in prompt and prompt.endswith(ANSWER_CUE)
    assert "Reply with the group." in prompt and "T1105 Ingress Tool Transfer" in prompt
    with pytest.raises(ValueError):
        render_family_prompt(state, "qwen", "malware")


def test_group_cases_draw_and_text():
    from scripts.cribl_finetune.export_group_cases import (
        case_text,
        draw_cases,
        extracted_per_report,
    )

    profiles = {
        "G0001": ["T1059", "T1059.001", "T1105", "T1566", "T1566.001", "T1027"],
        "G0002": ["T1003", "T1003.001", "T1021", "T1021.002", "T1105", "T1047", "T1053.005"],
    }
    parents = {
        "T1059.001": "T1059",
        "T1566.001": "T1566",
        "T1003.001": "T1003",
        "T1021.002": "T1021",
        "T1053.005": "T1053",
    }
    draws = draw_cases(
        profiles, parents, {"samples_per_group": 50, "size_min": 3, "size_max": 5, "seed": 1}
    )
    assert len(draws) == 100 and {g for g, _ in draws} == set(profiles)
    for group_id, techniques in draws:
        assert len(techniques) == len(set(techniques))
        own = set(profiles[group_id]) | {parents[t] for t in profiles[group_id] if t in parents}
        true = [t for t in techniques if t in own]
        assert 1 <= len(true) <= 5  # 3-5 drawn, a parent swap can merge two
        assert len(techniques) - len(true) <= 8  # precision >= 0.4 on 5 true techniques
    labels = {
        "T1105": "T1105 Ingress Tool Transfer",
        "T1059.001": "T1059.001 Command and Scripting Interpreter: PowerShell",
    }
    assert case_text(["T1105", "T1059.001", "T9999"], labels) == (
        "T1105 Ingress Tool Transfer; T1059.001 Command and Scripting Interpreter: PowerShell; T9999"
    )
    reports = [
        {
            "report_id": "r1",
            "statements": [{"text": "A  ran  PowerShell."}, {"text": "B dumped LSASS."}],
        },
        {"report_id": "r2", "statements": [{"text": "C sent phishing mail."}]},
    ]
    cases = [
        {"case_id": "a", "text": "A ran PowerShell."},
        {"case_id": "b", "text": "B dumped LSASS."},
        {"case_id": "c", "text": "C sent phishing mail."},
    ]
    predictions = [
        {"case_id": "a", "status": "ok", "technique_pick": "T1059.001"},
        {"case_id": "b", "status": "ok", "technique_pick": "T1003.001"},
        {"case_id": "c", "status": "error"},
    ]
    assert extracted_per_report(reports, cases, predictions) == {"r1": ["T1003.001", "T1059.001"]}


def test_group_baseline_rows_rank_by_score():
    from scripts.cribl_finetune.group_baseline import jaccard_scores, ranked_row

    row = {"case_id": "x", "dataset": "group_vendor", "gold_ids": ["G0002"]}
    out = ranked_row(row, {"G0001": 0.2, "G0002": 0.5, "G0003": 0.3})
    assert out["pick"] == "G0002" and out["gold_rank"] == 1 and out["top"][0] == ["G0002", 0.5]
    profiles = {"G0001": frozenset({"T1059", "T1105"}), "G0002": frozenset({"T1003", "T1003.001"})}
    scores = jaccard_scores(["T1003.001"], profiles, {"T1003.001": "T1003"})
    assert scores["G0002"] == 1.0 and scores["G0001"] == 0.0


def test_group_mlp_noise_measure_and_draw_params():
    from scripts.cribl_finetune.refit_group_mlp import (
        draw_params,
        false_positive_kind,
        measure_noise,
        report_sets,
    )

    cases = [
        {"case_id": "cti_rationale-r1-P1", "gold_ids": ["T1059.001", "T1105"]},
        {"case_id": "cti_rationale-r1-P2", "gold_ids": ["T1566"]},
        {"case_id": "cti_rationale-r2-P1", "gold_ids": ["T1003"]},
    ]
    predictions = [
        {"case_id": "cti_rationale-r1-P1", "status": "ok", "technique_pick": "T1059"},
        {"case_id": "cti_rationale-r1-P2", "status": "ok", "technique_pick": "T1566.001"},
        {"case_id": "cti_rationale-r2-P1", "status": "error"},
    ]
    sets = report_sets(cases, predictions)
    assert sets["r1"] == ({"T1059.001", "T1105", "T1566"}, {"T1059", "T1566.001"})
    assert sets["r2"] == ({"T1003"}, set())
    gold = {"T1059.001", "T1105", "T1566"}
    assert false_positive_kind("T1059", gold) == "granularity"
    assert false_positive_kind("T1566.001", gold) == "granularity"
    assert false_positive_kind("T1059.003", gold) == "sibling"
    assert false_positive_kind("T1003", gold) == "other"
    noise = measure_noise(sets)
    assert noise["reports"] == 1 and noise["precision_exact"]["median"] == 0.0
    assert noise["precision_parent"]["median"] == 1.0
    params = draw_params(
        {
            "extracted_size": {"q10": 8, "q90": 30},
            "precision_parent": {"median": 0.5, "q10": 0.3, "q90": 0.8},
        }
    )
    assert params == {"size_min": 4, "size_max": 15, "precision_min": 0.3, "precision_max": 0.8}


def test_actor_prompt_and_report_aggregation():
    from scripts.cribl_finetune.common import ACTOR_INSTRUCTIONS, c1_state, render_family_prompt
    from scripts.cribl_finetune.export_actor_cases import dev_subsample, training_rows
    from scripts.cribl_finetune.group_text_eval import aggregate_reports

    prompt = render_family_prompt(c1_state("The adversary used PowerShell."), "qwen", "actor")
    assert ACTOR_INSTRUCTIONS in prompt and "Reply with the group." in prompt
    records = [
        {
            "text": "the adversary used PowerShell",
            "groups": ["G0001"],
            "source": "group",
            "owner": "G0001",
        },
        {
            "text": "shared tool text",
            "groups": ["G0001", "G0002", "G0003", "G0004"],
            "source": "software",
            "owner": "S1",
        },
        {"text": "unlisted group text", "groups": ["G0009"], "source": "group", "owner": "G0009"},
        {"text": "two groups", "groups": ["G0001", "G0002"], "source": "campaign", "owner": "C1"},
    ]
    rows = training_rows(records, {"G0001", "G0002", "G0003"})
    assert [row["gold_ids"] for row in rows] == [["G0001"], ["G0001", "G0002"]]
    statements = [
        {"case_id": f"v-r{r}-{i}", "report_id": f"r{r}", "gold_ids": ["G0001"]}
        for r in (1, 2)
        for i in range(4)
    ]
    dev = dev_subsample(statements, 2, 1)
    assert len(dev) == 4 and {row["report_id"] for row in dev} == {"r1", "r2"}
    cases = [
        {"case_id": "a", "report_id": "r1", "gold_ids": ["G0002"]},
        {"case_id": "b", "report_id": "r1", "gold_ids": ["G0002"]},
        {"case_id": "c", "report_id": "r2", "gold_ids": ["G0001"]},
    ]
    predictions = [
        {"case_id": "a", "status": "ok", "top": [["G0001", 0.6], ["G0002", 0.4]]},
        {"case_id": "b", "status": "ok", "top": [["G0002", 0.9]]},
        {"case_id": "c", "status": "error"},
    ]
    out = aggregate_reports(cases, predictions, ["G0001", "G0002", "G0003"])
    assert len(out) == 1 and out[0]["case_id"] == "r1"
    assert (
        out[0]["pick"] == "G0002"
        and out[0]["top"][0] == ["G0002", 0.65]
        and out[0]["gold_rank"] == 1
    )


def test_external_bench_scoring_rules():
    from scripts.cribl_finetune.external_bench import (
        ate_scores,
        ate_sets,
        bullets,
        connected,
        taa_score,
    )

    assert bullets("- Targeted X - Used hardcoded names - C2 via SharePoint") == [
        "Used hardcoded names",
        "C2 via SharePoint",
    ]  # a two-word bullet is dropped
    cases = [
        {"case_id": "a-1", "report_id": "a", "gold_ids": ["T1059", "T1105"]},
        {"case_id": "a-2", "report_id": "a", "gold_ids": ["T1059", "T1105"]},
        {"case_id": "b-1", "report_id": "b", "gold_ids": ["T1003"]},
    ]
    predictions = [
        {
            "case_id": "a-1",
            "status": "ok",
            "top": [["T1059", 0.9]],
            "technique_top": [["T1059.001", 0.8]],
        },
        {
            "case_id": "a-2",
            "status": "ok",
            "top": [["T1566", 0.7]],
            "technique_top": [["T1566.001", 0.6]],
        },
        {"case_id": "b-1", "status": "error"},
    ]
    import math

    sets = ate_sets(cases, predictions, math.inf)
    assert sets == {"a": ({"T1059", "T1105"}, {"T1059", "T1566"}), "b": ({"T1003"}, set())}
    scored = ate_scores(sets)
    assert scored["micro_f1"] == 0.4 and scored["mean_f1"] == 0.25  # a: P 0.5 R 0.5; b: 0
    aliases = {"apt41": {"winnti group"}, "winnti group": {"apt41"}}
    related = {"apt41": {"apt17"}, "apt17": {"apt41"}}
    assert connected("APT41", "Winnti Group", aliases)
    assert not connected("APT41", "APT17", aliases) and connected(
        "APT41", "APT17", aliases, related
    )
    assert taa_score("APT41", "Winnti Group", aliases, related)["combined"] == 1.0
    assert taa_score("APT41", "APT17", aliases, related)["combined"] == 0.5
    assert taa_score("APT41", "Lazarus", aliases, related)["combined"] == 0.0
