"""Software interview questions (apt_classifier/software_questions.py, docs 21.3.15 C)."""

from __future__ import annotations

import re

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique
from apt_classifier.config import Settings
from apt_classifier.separation import CandidateDiscrimination, DiscriminationResult
from apt_classifier.session import AttributionEngine
from apt_classifier.software_questions import (
    SOFTWARE_HYPOTHESIS,
    group_name_pattern,
    leader_only_negative,
    penalized_separation,
    rank_software_leads,
    replay_window,
    software_question_pool,
    software_selected_question,
)
from tests.test_session import FakeStore, _ledger

POWERSHELL = "PowerShell executed an encoded command."
LSASS = "Credentials were dumped from LSASS memory."


def software_kb() -> AttackKnowledgeBase:
    """Three groups; each owns malware no other group uses. Charlie leads on PowerShell + LSASS
    (it documents both), Alpha and Bravo document one each."""
    kb = AttackKnowledgeBase(
        version="test",
        techniques={
            "T1059": Technique(id="T1059", name="Command and Scripting Interpreter"),
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", is_subtechnique=True, parent_id="T1059"
            ),
            "T1003": Technique(id="T1003", name="OS Credential Dumping"),
            "T1003.001": Technique(
                id="T1003.001", name="LSASS Memory", is_subtechnique=True, parent_id="T1003"
            ),
            "T1021.002": Technique(
                id="T1021.002",
                name="SMB/Windows Admin Shares",
                is_subtechnique=True,
                parent_id="T1021",
            ),
        },
        groups={
            "G1": Group(
                id="G1",
                name="Alpha",
                aliases=["Alpha", "Redteam Panda"],
                technique_ids=["T1059.001"],
                software_ids=["S0101", "S0105"],
            ),
            "G2": Group(
                id="G2",
                name="Bravo",
                technique_ids=["T1003.001", "T1021.002"],
                software_ids=["S0102", "S0106", "S0107"],
            ),
            "G3": Group(
                id="G3",
                name="Charlie",
                technique_ids=["T1059.001", "T1003.001"],
                software_ids=["S0103", "S0104", "S0106", "S0108"],
            ),
        },
        software={
            # an alias that is a group's alias must never be shown
            "S0101": Software(
                id="S0101",
                name="Redfang",
                software_type="malware",
                aliases=["Redfang", "Redteam Panda", "RF-Loader"],
                platforms=["Windows"],
                group_ids=["G1"],
                technique_ids=["T1059.001"],
            ),
            "S0102": Software(
                id="S0102",
                name="Bluefin",
                software_type="malware",
                group_ids=["G2"],
                technique_ids=["T1003.001"],
            ),
            "S0103": Software(
                id="S0103",
                name="Greenfin",
                software_type="malware",
                group_ids=["G3"],
                technique_ids=["T1059.001", "T1003.001"],
            ),
            "S0104": Software(
                id="S0104", name="Goldfin", software_type="malware", group_ids=["G3"]
            ),
            # malware named like a group: asking about it would name the group
            "S0105": Software(id="S0105", name="Alpha", software_type="malware", group_ids=["G1"]),
            # two users: decisive, but not exclusive to one candidate
            "S0106": Software(
                id="S0106", name="Twinfin", software_type="malware", group_ids=["G2", "G3"]
            ),
            # a tool is never asked about
            "S0107": Software(id="S0107", name="Wrench", software_type="tool", group_ids=["G2"]),
            "S0108": Software(
                id="S0108", name="Silverfin", software_type="malware", group_ids=["G3"]
            ),
        },
    )
    kb.recompute_statistics()
    return kb


def software_settings(**overrides) -> Settings:
    base = {
        "technique_mapping_preset": "local_only",
        "cyberre_weight": 0.0,
        "reranker_enabled": False,
        "profile_forest_enabled": False,
        "group_mlp_enabled": False,
        "attribution_combiner_enabled": False,
        "attribution_min_maturity": 0.0,
        "interview_software_questions": True,
    }
    base.update(overrides)
    return Settings(**base)


def software_engine(**overrides) -> AttributionEngine:
    return AttributionEngine(software_settings(**overrides), software_kb(), FakeStore(), None)


def test_pool_asks_only_about_malware_one_top_three_candidate_alone_documents():
    kb = software_kb()
    window = ["G3", "G1", "G2"]
    pool = software_question_pool(kb, window, window, set(), set())
    assert [(item.software_id, item.group_id, item.target_rank) for item in pool] == [
        ("S0101", "G1", 2),
        ("S0102", "G2", 3),
        ("S0103", "G3", 1),
        ("S0104", "G3", 1),
        ("S0108", "G3", 1),
    ]  # no tool (S0107), no two-user malware (S0106), no malware named like a group (S0105)
    # named in the case or already asked: skipped
    pool = software_question_pool(kb, window, window, {"S0103"}, {"S0101", "S0108"})
    assert [item.software_id for item in pool] == ["S0102", "S0104"]
    # the users must be inside the gate window, and the candidate in the decision's top three
    narrow = software_question_pool(kb, ["G3", "G1"], ["G3", "G1"], set(), set())
    assert narrow and "S0102" not in {item.software_id for item in narrow}
    assert {item.group_id for item in software_question_pool(kb, window, ["G3"], (), ())} == {"G3"}
    # the looser variant: two users, both in the window, one of them in the top three
    loose = software_question_pool(kb, window, ["G3", "G1"], set(), set(), max_users=2)
    assert ("S0106", "G3") in {(item.software_id, item.group_id) for item in loose}
    assert "S0106" not in {
        item.software_id for item in software_question_pool(kb, window, window, (), (), 2)
    }  # both users in the top three: not exclusive to one candidate


def test_wording_never_names_a_group_or_alias():
    kb = software_kb()
    pattern = group_name_pattern(kb)
    assert pattern is not None and pattern.search("the Redteam Panda loader")
    engine = AttributionEngine(software_settings(), kb, FakeStore(), None)
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    leads = engine.software_leads(state)
    assert leads
    for item in software_question_pool(kb, ["G1", "G2", "G3"], ["G1", "G2", "G3"], (), ()):
        lead = next((lead for lead in leads if lead.software_id == item.software_id), None)
        if lead is None:
            continue
        question = software_selected_question(kb, lead, platforms=("Windows",), window_size=3)
        text = " ".join((question.question, question.hypothesis, *question.investigation_steps))
        assert not pattern.search(text), text
        assert question.question == (
            f"Did you find {kb.software[lead.software_id].name} ({lead.software_id}) or its "
            "known artefacts?"
        )
        assert question.hypothesis == SOFTWARE_HYPOTHESIS
        assert question.kind == "software" and question.technique_id == ""
        assert question.asked_at and question.target_rank == lead.target_rank
    redfang = software_selected_question(
        kb, next(lead for lead in leads if lead.software_id == "S0101")
    )
    steps = " ".join(redfang.investigation_steps)
    assert "RF-Loader" in steps and "Redteam Panda" not in steps
    assert "PowerShell (T1059.001)" in steps and "Windows" in steps


def test_leads_are_ranked_by_the_separation_change_then_techniques_then_id():
    engine = software_engine()
    state = engine.analyze(_ledger(POWERSHELL, LSASS))
    context = state.context
    discrimination = context.discrimination
    assert [item.group_id for item in discrimination.scores] == ["G3", "G1", "G2"]
    # the replay of the recorded window is the gate's own discrimination
    replayed = replay_window(engine.kb, discrimination, context)
    assert replayed.separation == discrimination.separation
    assert [item.total for item in replayed.scores] == [
        item.total for item in discrimination.scores
    ]
    pool = software_question_pool(engine.kb, ["G3", "G1", "G2"], ["G3", "G1", "G2"], (), ())
    leads = rank_software_leads(
        engine.kb, discrimination, context, pool, per_group=3, decisive_boost=2.0
    )
    keys = [(-abs(lead.delta_found), -lead.technique_count, lead.software_id) for lead in leads]
    assert keys == sorted(keys)
    # G1 and G2 tie on |delta| and technique count: the id breaks the tie
    assert [lead.software_id for lead in leads[:2]] == ["S0101", "S0102"]
    # a found for the leader's malware raises the separation, for a rival's lowers it
    by_id = {lead.software_id: lead for lead in leads}
    assert by_id["S0103"].delta_found > 0 > by_id["S0101"].delta_found
    # a not found only ever lowers, and only for the leader's own malware
    assert by_id["S0103"].delta_not_found < 0
    assert by_id["S0101"].delta_not_found == by_id["S0102"].delta_not_found == 0.0
    # one item per top-three candidate by default
    capped = rank_software_leads(engine.kb, discrimination, context, pool, decisive_boost=2.0)
    assert sorted(lead.group_id for lead in capped) == ["G1", "G2", "G3"]
    assert [lead.to_dict() for lead in engine.software_leads(state)] == [
        lead.to_dict() for lead in capped
    ]


def _result(totals: list[tuple[str, float]], floor: float = 1.0) -> DiscriminationResult:
    scores = tuple(
        CandidateDiscrimination(gid, 0.1, total, 0.0, 0.0, 0.0, total) for gid, total in totals
    )
    leader, others = scores[0], scores[1:]
    runner = max(others, key=lambda item: item.total) if others else None
    runner_total = runner.total if runner is not None else 0.0
    denominator = max(leader.total, runner_total, floor)
    return DiscriminationResult(
        scores=scores,
        top_k=len(scores),
        leader_id=leader.group_id,
        runner_up_id=runner.group_id if runner is not None else None,
        separation=(leader.total - runner_total) / denominator,
        floor=floor,
    )


def test_not_found_lowers_only_the_leader_and_only_for_its_own_malware():
    kb = software_kb()
    result = _result([("G3", 3.4), ("G1", 0.2), ("G2", 0.1)])
    rival = leader_only_negative(result, kb, {"S0101"}, 0.3)
    assert rival.penalty == 0.0 and rival.applied == ()
    assert rival.separation == result.separation and rival.total == 3.4
    own = leader_only_negative(result, kb, {"S0103", "S0104", "S0101"}, 0.3)
    assert own.applied == ("S0103", "S0104") and own.penalty == pytest.approx(0.6)
    assert own.total == pytest.approx(2.8) and own.separation < result.separation
    assert own.separation == pytest.approx((2.8 - 0.2) / 2.8)
    # shared with a window member: less exclusive, a smaller penalty
    shared = leader_only_negative(result, kb, {"S0106"}, 0.3)
    assert 0 < shared.penalty < 0.3


@pytest.mark.parametrize("seed", range(5))
def test_penalized_separation_never_increases(seed):
    import random

    rng = random.Random(seed)
    for _ in range(200):
        total, runner = rng.uniform(0, 6), rng.uniform(0, 6)
        floor = rng.choice([0.5, 1.0, 2.0])
        base = penalized_separation(total, runner, 0.0, floor)
        penalty = rng.uniform(0, 4)
        reduced, separation = penalized_separation(total, runner, penalty, floor)
        assert reduced <= total and separation <= base[1] + 1e-12
        more = penalized_separation(total, runner, penalty + rng.uniform(0, 2), floor)
        assert more[1] <= separation + 1e-12


def test_distinguish_lists_the_same_software_rows_the_interview_asks_from(tmp_path):
    from apt_classifier.service import CaseService, IntakePayload

    engine = software_engine()
    service = CaseService(engine.settings, engine=engine, cases_dir=tmp_path)
    case_id = service.create_case(IntakePayload(narrative=f"{POWERSHELL} {LSASS}"))
    state, _ = service.get(case_id)
    expected = [lead.to_dict() for lead in engine.software_leads(state)]
    rows = service.distinguish(case_id)["software"]
    assert expected and [{k: v for k, v in row.items() if k != "group_name"} for row in rows] == (
        expected
    )
    assert {row["group_name"] for row in rows} <= {"Alpha", "Bravo", "Charlie"}
    interview = service.interview(case_id)
    question = interview["question"]
    assert question["kind"] == "software" and question["software_id"] == rows[0]["software_id"]
    assert re.fullmatch(
        r"Did you find .+ \(S\d{4}\) or its known artefacts\?", question["question"]
    )


def test_the_per_group_cap_holds_over_the_whole_interview():
    """One question per top-three candidate by default: once each candidate was asked about,
    no further software question comes, although Charlie still has unasked malware (with two
    per candidate Charlie gets a second one; Alpha and Bravo have nothing else to ask about)."""
    for per_group, expected in ((1, 3), (2, 4)):
        engine = software_engine(
            interview_software_max_questions=6, interview_software_per_group=per_group
        )
        state = engine.analyze(_ledger(POWERSHELL, LSASS))
        targets = []
        while (question := engine.software_question(state)) is not None:
            targets.append(question.target_group_id)
            state.current_question = question
            engine.answer_question(state, "not checked")
        assert len(targets) == expected, (per_group, targets)
        assert max(targets.count(gid) for gid in set(targets)) == per_group
        assert [answer.metadata["target_group_id"] for answer in state.ledger.question_answers] == (
            targets
        )
