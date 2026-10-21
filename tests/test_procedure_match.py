"""The technique-conditioned procedure member (apt_classifier/procedure_match.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.procedure_match import (
    PROCEDURE_MATCH_NAME,
    ProcedureMatchClassifier,
    ProcedureMatchModel,
    load_procedure_match_for,
    procedure_match_bound_fingerprint,
    procedure_match_params,
    procedure_match_votes,
)
from apt_classifier.text_group import TextGroupModel, build_corpus, prominence_log_prior


def _kb() -> AttackKnowledgeBase:
    techniques = {
        "T1566": Technique(id="T1566", name="Phishing"),
        "T1566.001": Technique(
            id="T1566.001", name="Spearphishing Attachment", is_subtechnique=True, parent_id="T1566"
        ),
        "T1566.002": Technique(
            id="T1566.002", name="Spearphishing Link", is_subtechnique=True, parent_id="T1566"
        ),
        "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
        "T1105": Technique(id="T1105", name="Ingress Tool Transfer"),
    }
    groups = {
        "G1": Group(
            id="G1",
            name="Alpha",
            description="Alpha is an espionage group.",
            technique_ids=["T1566.001", "T1059"],
            software_ids=["S1"],
            technique_behaviors={
                "T1566.001": ["Alpha sent spearphishing emails with macro documents attached."],
                "T1059": ["Alpha ran a batch script to collect host details."],
            },
        ),
        "G2": Group(
            id="G2",
            name="Bravo",
            description="Bravo is a financially motivated group.",
            technique_ids=["T1566.002", "T1059"],
            software_ids=["S2"],
            technique_behaviors={
                "T1566.002": ["Bravo sent spearphishing emails with links to a fake login page."],
                "T1059": ["Bravo used a batch script to collect host details."],
            },
        ),
        "G3": Group(
            id="G3",
            name="Charlie",
            description="Charlie is a destructive group.",
            technique_ids=["T1105"],
            software_ids=["S2"],
            technique_behaviors={"T1105": ["Charlie downloaded a wiper from its server."]},
        ),
    }
    software = {
        "S1": Software(
            id="S1",
            name="AlphaRat",
            software_type="malware",
            technique_ids=["T1105", "T1059"],
            group_ids=["G1"],
        ),
        "S2": Software(
            id="S2",
            name="SharedTool",
            software_type="tool",
            technique_ids=["T1105"],
            group_ids=["G2", "G3"],
        ),
    }
    kb = AttackKnowledgeBase(
        techniques=techniques, groups=groups, software=software, version="20.0"
    )
    return kb


def _records_with_software(kb: AttackKnowledgeBase) -> list[dict]:
    """The text member's corpus plus two software procedure records (the raw bundle is not
    available in a test, so they are appended by hand with the same fields)."""
    records = build_corpus(kb, None)
    records.append(
        {
            "text": "AlphaRat downloads additional payloads from its command server over https.",
            "source": "software",
            "owner": "S1",
            "groups": ["G1"],
            "technique_id": "T1105",
        }
    )
    records.append(
        {
            "text": "SharedTool downloads files from a remote server to the victim host.",
            "source": "software",
            "owner": "S2",
            "groups": ["G2", "G3"],
            "technique_id": "T1105",
        }
    )
    return records


def _base(kb: AttackKnowledgeBase) -> TextGroupModel:
    group_ids = tuple(sorted(kb.groups))
    return TextGroupModel(
        _records_with_software(kb),
        group_ids,
        prominence_log_prior(kb, group_ids),
        params={"min_df": 1},
        metadata={"attack_version": "20.0"},
    )


def test_votes_are_conditioned_on_the_accepted_technique():
    kb = _kb()
    model = ProcedureMatchModel(_base(kb), kb, {"family": "exact", "link_weight": 0.0})
    statement = ["the operators sent spearphishing emails with macro documents attached"]
    votes, techniques, details = model.votes({"T1566.001": statement})
    g = {gid: i for i, gid in enumerate(model.group_ids)}
    assert techniques == ["T1566.001"]
    # only Alpha documents this behaviour under the attachment sub-technique
    assert votes[0, g["G1"]] > 0 and votes[0, g["G2"]] == 0 and votes[0, g["G3"]] == 0
    assert details["G1"][0] == "T1566.001" and "group:" in details["G1"][2]
    # the same sentence filed under the link sub-technique reaches nobody: Bravo's link
    # sentence shares the words but Alpha's attachment sentence is not under T1566.002
    votes_link, _, _ = model.votes({"T1566.002": statement})
    assert votes_link[0, g["G1"]] == 0
    # the parent rule lets a parent-level acceptance read both sub-techniques, never siblings
    parent = ProcedureMatchModel(_base(kb), kb, {"family": "parent", "link_weight": 0.0})
    assert parent.family("T1566") == ("T1566", "T1566.001", "T1566.002")
    assert parent.family("T1566.001") == ("T1566.001", "T1566")
    votes_parent, _, _ = parent.votes({"T1566": statement})
    assert votes_parent[0, g["G1"]] > votes_parent[0, g["G2"]] > 0


def test_software_route_links_a_tool_sentence_to_its_users_with_rarity_and_the_seen_bonus():
    kb = _kb()
    params = {"family": "exact", "link_weight": 0.0, "seen_bonus": 1.0, "min_similarity": 0.0}
    model = ProcedureMatchModel(_base(kb), kb, params)
    g = {gid: i for i, gid in enumerate(model.group_ids)}
    statement = ["the malware downloads additional payloads from its command server over https"]
    votes, _, details = model.votes({"T1105": statement})
    # AlphaRat's sentence votes for Alpha; the shared tool's sentence for Bravo and Charlie
    assert votes[0, g["G1"]] > 0 and votes[0, g["G2"]] > 0 and votes[0, g["G3"]] > 0
    assert "S1" in details["G1"][2]
    # rarity: the single-user malware outweighs the two-user tool (same cosine order)
    assert model.rarity("S1") > model.rarity("S2") > 0
    # naming the software in the report doubles its route
    seen, _, seen_details = model.votes({"T1105": statement}, {"S1"})
    assert seen[0, g["G1"]] == pytest.approx(2.0 * votes[0, g["G1"]])
    assert "(named)" in seen_details["G1"][2]
    assert seen[0, g["G2"]] == votes[0, g["G2"]]
    # the route can be switched off: the two groups reached only through software go silent
    # (Charlie keeps the weak vote of its own sentence)
    off = ProcedureMatchModel(_base(kb), kb, {**params, "software_route": False})
    off_votes = off.votes({"T1105": statement})[0]
    assert off_votes[0, g["G1"]] == 0 and off_votes[0, g["G2"]] == 0


def test_link_route_votes_without_text_for_named_software_documented_for_the_technique():
    kb = _kb()
    model = ProcedureMatchModel(
        _base(kb), kb, {"family": "exact", "software_route": False, "link_weight": 1.0}
    )
    g = {gid: i for i, gid in enumerate(model.group_ids)}
    # AlphaRat is documented for T1105 and used by Alpha: the link alone votes
    votes, _, details = model.votes({"T1105": ["nothing in common with the corpus"]}, {"S1"})
    assert votes[0, g["G1"]] == pytest.approx(model.rarity("S1"))
    assert "documented for T1105" in details["G1"][2]
    # not documented for the accepted technique: no vote
    assert model.votes({"T1566.001": ["x y z"]}, {"S1"})[0].sum() == 0


def test_scores_aggregate_over_the_accepted_techniques_and_softmax_over_reached_groups():
    kb = _kb()
    model = ProcedureMatchModel(_base(kb), kb, {"family": "exact", "aggregate": "mean"})
    case = {
        "T1566.001": ["the operators sent spearphishing emails with macro documents attached"],
        "T1059": ["a batch script collected host details"],
    }
    scores, details = model.scores(case)
    assert max(scores, key=scores.get) == "G1" and sum(scores.values()) == pytest.approx(1.0)
    assert "G3" not in scores  # no technique reached Charlie
    assert model.scores({"T1105": ["the weather was mild"]}) == ({}, {})
    assert model.scores({}) == ({}, {})
    with pytest.raises(ValueError):
        ProcedureMatchModel(_base(kb), kb, {"aggregate": "max"})
    with pytest.raises(ValueError):
        ProcedureMatchModel(_base(kb), kb, {"family": "siblings"})


def test_without_hides_the_fold_and_the_removed_actor():
    kb = _kb()
    model = ProcedureMatchModel(_base(kb), kb, {"family": "exact", "link_weight": 0.0})
    statement = {"T1566.001": ["the operators sent spearphishing emails with macro documents"]}
    assert "G1" in model.scores(statement)[0]
    hidden = model.without(
        kb,
        group_id="G1",
        techniques=["T1566.001"],
        texts=["Alpha sent spearphishing emails with macro documents attached."],
    )
    assert "G1" not in hidden.scores(statement)[0]
    assert hidden.fingerprint != model.fingerprint
    reduced = _kb()
    del reduced.groups["G1"]
    gone = model.without(reduced, group_id="G1", remove_group=True)
    assert "G1" not in gone.group_ids
    # the removed actor leaves its software's user list, so the software route cannot name it
    assert "G1" not in gone.scores({"T1105": ["downloads additional payloads over https"]})[0]
    assert model.without(kb, group_id="G1", techniques=["T1566.001"]) is model.without(
        kb, group_id="G1", techniques=["T1566.001"]
    )


def test_classifier_reads_the_technique_statements_and_respects_eligibility():
    kb = _kb()
    model = ProcedureMatchModel(_base(kb), kb, {"family": "exact"})
    context = SimpleNamespace(
        kb=kb,
        technique_statements={
            "T1566.001": ("the operators sent spearphishing emails with macro documents",)
        },
        software_ids=set(),
        eligible_group_ids={"G1", "G3"},
        observed_techniques={},
    )
    result = ProcedureMatchClassifier(model).classify(context)
    assert result.applicable and result.classifier == PROCEDURE_MATCH_NAME
    assert [e.group_id for e in result.entries] == ["G1"]
    assert any("T1566.001" in s for s in result.by_group()["G1"].signal_ids)
    context.technique_statements = {}
    assert not ProcedureMatchClassifier(model).classify(context).applicable
    assert not ProcedureMatchClassifier(None).classify(context).applicable


def test_votes_only_with_a_weight_and_the_loader_fails_closed(tmp_path):
    kb = _kb()
    assert not procedure_match_votes(
        SimpleNamespace(procedure_match_enabled=True, procedure_match_weight=0)
    )
    settings = SimpleNamespace(
        procedure_match_enabled=True,
        procedure_match_weight=1.0,
        procedure_match_min_similarity=0.2,
        procedure_match_family="exact",
        procedure_match_software_route=False,
        procedure_match_seen_bonus=0.5,
        procedure_match_link_weight=0.0,
        procedure_match_aggregate="mean",
        procedure_match_prior=0.0,
        text_group_path=tmp_path / "missing.json.gz",
    )
    assert procedure_match_params(settings)["family"] == "exact"
    model, status = load_procedure_match_for(settings, kb)
    assert model is None and "not found" in status
    base = _base(kb)
    model, status = load_procedure_match_for(settings, kb, base)
    assert model is not None and status == "active" and model.params["min_similarity"] == 0.2
    settings.procedure_match_aggregate = "median"
    assert load_procedure_match_for(settings, kb, base)[0] is None
    assert procedure_match_bound_fingerprint("abc", model) != procedure_match_bound_fingerprint(
        "abd", model
    )


def test_member_never_counts_as_an_independent_family_and_has_combiner_columns():
    from apt_classifier.attribution import NOT_INDEPENDENT_EVIDENCE
    from apt_classifier.combiner import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, OPTIONAL_CLASSIFIERS

    assert PROCEDURE_MATCH_NAME in NOT_INDEPENDENT_EVIDENCE
    assert PROCEDURE_MATCH_NAME in OPTIONAL_CLASSIFIERS and FEATURE_SCHEMA_VERSION >= 3
    assert "procedure_match_score_rel" in FEATURE_NAMES
    assert "procedure_match_inv_rank_x_log_accepted" in FEATURE_NAMES
