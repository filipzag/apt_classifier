"""The behaviour-text member: statement-level matching against the text member's corpus."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.behavior_text import (
    BEHAVIOR_TEXT_NAME,
    BehaviorTextClassifier,
    BehaviorTextModel,
    behavior_text_bound_fingerprint,
    behavior_text_params,
    behavior_text_votes,
    load_behavior_text_for,
)
from apt_classifier.text_group import TextGroupModel, build_corpus, prominence_log_prior


def _kb() -> AttackKnowledgeBase:
    techniques = {f"T{i}": Technique(id=f"T{i}", name=f"Technique {i}") for i in range(1, 6)}
    groups = {
        "G1": Group(
            id="G1",
            name="Alpha",
            description="Alpha is an espionage group targeting ministries.",
            technique_ids=["T1", "T2"],
            technique_behaviors={
                "T1": ["Alpha has used spearphishing attachments with macro documents."],
                "T2": ["Alpha staged archives on a compromised sharepoint server."],
            },
        ),
        "G2": Group(
            id="G2",
            name="Bravo",
            description="Bravo is a financially motivated group targeting point of sale systems.",
            technique_ids=["T3", "T4"],
            technique_behaviors={
                "T3": ["Bravo has scraped payment card data from point of sale memory."],
                "T4": ["Bravo used a custom backdoor over raw sockets for command and control."],
            },
        ),
        "G3": Group(
            id="G3",
            name="Charlie",
            description="Charlie is a destructive group attacking power grid operators.",
            technique_ids=["T5"],
            technique_behaviors={
                "T5": ["Charlie deployed wiper malware against power grid workstations."]
            },
        ),
    }
    return AttackKnowledgeBase(techniques=techniques, groups=groups, version="19.2")


def _base(kb: AttackKnowledgeBase, **params) -> TextGroupModel:
    records = build_corpus(kb, None)
    group_ids = tuple(sorted(kb.groups))
    return TextGroupModel(
        records,
        group_ids,
        prominence_log_prior(kb, group_ids),
        params={"min_df": 1, **params},
        metadata={"attack_version": "19.2"},
    )


def test_votes_follow_the_statements_and_the_mean_rewards_consistent_groups():
    kb = _kb()
    model = BehaviorTextModel(_base(kb), {"top_k": 5})
    statements = [
        "the operators sent spearphishing attachments carrying macro documents",
        "stolen archives were staged on a sharepoint server before exfiltration",
        "a custom backdoor spoke over raw sockets for command and control",
    ]
    votes = model.votes(statements)
    assert votes.shape == (3, 3)
    g = {gid: i for i, gid in enumerate(model.group_ids)}
    # the first two statements reach Alpha best, the third Bravo; a stray shared word (Charlie's
    # "operators") only gives a weak vote
    assert votes[0, g["G1"]] > votes[0, g["G3"]] > 0 and votes[1, g["G1"]] > 0
    assert votes[2, g["G2"]] > 0 and votes[2, g["G2"]] == votes[2].max()
    scores = model.scores(statements)
    assert max(scores, key=scores.get) == "G1" and scores["G1"] > scores["G2"] > scores["G3"]
    assert sum(scores.values()) == pytest.approx(1.0)
    # nothing in common with the corpus: no score at all
    assert model.scores(["the weather was mild and the market closed early"]) == {}
    assert model.scores([]) == {}
    # top5 aggregation and a prior are accepted; an unknown aggregate is not
    assert BehaviorTextModel(_base(kb), {"aggregate": "top5", "prior": 0.5}).scores(statements)
    with pytest.raises(ValueError):
        BehaviorTextModel(_base(kb), {"aggregate": "max"})


def test_without_hides_the_fold_like_the_text_member():
    kb = _kb()
    model = BehaviorTextModel(_base(kb), {"top_k": 5})
    hidden = model.without(
        kb,
        group_id="G1",
        techniques=["T1"],
        texts=["Alpha has used spearphishing attachments with macro documents."],
    )
    statement = ["the operators sent spearphishing attachments carrying macro documents"]
    assert "G1" in model.scores(statement)
    # the held-out procedure is gone, so Alpha is no longer reached by that statement
    assert "G1" not in hidden.scores(statement)
    assert hidden.fingerprint != model.fingerprint
    # the removed-actor arm drops the group entirely
    reduced = _kb()
    del reduced.groups["G1"]
    gone = model.without(reduced, group_id="G1", remove_group=True)
    assert "G1" not in gone.group_ids
    assert model.without(kb, group_id="G1", techniques=["T1"]) is model.without(
        kb, group_id="G1", techniques=["T1"]
    )  # cached


def test_classifier_reads_the_statements_and_respects_eligibility():
    kb = _kb()
    model = BehaviorTextModel(_base(kb), {"top_k": 5})
    context = SimpleNamespace(
        kb=kb,
        statements=("the operators sent spearphishing attachments carrying macro documents",),
        narrative="",
        eligible_group_ids={"G1", "G3"},
        observed_techniques={},
    )
    result = BehaviorTextClassifier(model).classify(context)
    assert result.applicable and result.classifier == BEHAVIOR_TEXT_NAME
    # Bravo is not eligible and never appears; Alpha ranks first
    assert [e.group_id for e in result.entries][0] == "G1"
    assert "G2" not in {e.group_id for e in result.entries}
    # without statements the narrative is split into sentences; without a model it abstains
    context.statements = ()
    context.narrative = "The operators sent spearphishing attachments. Archives went to sharepoint."
    assert BehaviorTextClassifier(model).classify(context).applicable
    assert not BehaviorTextClassifier(None).classify(context).applicable


def test_votes_only_with_a_weight_and_the_loader_fails_closed(tmp_path):
    kb = _kb()
    assert not behavior_text_votes(
        SimpleNamespace(behavior_text_enabled=True, behavior_text_weight=0)
    )
    assert behavior_text_votes(
        SimpleNamespace(behavior_text_enabled=True, behavior_text_weight=1.0)
    )
    settings = SimpleNamespace(
        behavior_text_enabled=True,
        behavior_text_weight=1.0,
        behavior_text_top_k=10,
        behavior_text_aggregate="top5",
        behavior_text_prior=0.25,
        text_group_path=tmp_path / "missing.json.gz",
    )
    assert behavior_text_params(settings) == {
        "top_k": 10,
        "min_similarity": 0.0,
        "aggregate": "top5",
        "prior": 0.25,
    }
    model, status = load_behavior_text_for(settings, kb)
    assert model is None and "not found" in status
    base = _base(kb)
    model, status = load_behavior_text_for(settings, kb, base)
    assert model is not None and status == "active" and model.params["top_k"] == 10
    settings.behavior_text_aggregate = "median"
    assert load_behavior_text_for(settings, kb, base)[0] is None
    off, status = load_behavior_text_for(SimpleNamespace(behavior_text_weight=0), kb, base)
    assert off is None and status.startswith("disabled")
    assert behavior_text_bound_fingerprint("abc", model) != behavior_text_bound_fingerprint(
        "abd", model
    )


def test_member_never_counts_as_an_independent_family():
    from apt_classifier.attribution import NOT_INDEPENDENT_EVIDENCE

    assert BEHAVIOR_TEXT_NAME in NOT_INDEPENDENT_EVIDENCE


def test_scores_are_size_blind_for_an_equal_match():
    """A group with many sentences gets no head start: votes are per statement, best match."""
    kb = _kb()
    big = kb.groups["G2"]
    big.technique_behaviors["T3"] = big.technique_behaviors["T3"] * 20
    model = BehaviorTextModel(_base(kb), {"top_k": 50})
    votes = model.votes(["payment card data was scraped from point of sale memory"])
    g = {gid: i for i, gid in enumerate(model.group_ids)}
    assert votes[0, g["G2"]] <= 1.0 + 1e-6
    # twenty identical sentences move the best vote by at most a few percent (the IDF over
    # sentences shifts); they never multiply it
    once = BehaviorTextModel(_base(_kb()), {"top_k": 50}).votes(
        ["payment card data was scraped from point of sale memory"]
    )
    assert once[0, g["G2"]] > 0
    assert abs(votes[0, g["G2"]] - once[0, g["G2"]]) <= 0.03 * once[0, g["G2"]]
