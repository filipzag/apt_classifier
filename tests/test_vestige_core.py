"""The lean core: exact incremental TF-IDF, the date and hiding rules, the naming rule, scoring."""

from __future__ import annotations

import numpy as np
import pytest

from apt_classifier.text_group import TextGroupModel
from vestige_core.decide import Proposal, decide
from vestige_core.evaluate import paired, score, smallest_margin
from vestige_core.rank import Ranking, TextRanker

ACTORS = ("G1", "G2", "G3")
ROWS = [
    ("p1", "G1", "the adversary used powershell to download a loader from a staging server"),
    ("p1", "G1", "the loader injected shellcode into a legitimate process for persistence"),
    ("p2", "G1", "powershell scripts downloaded the loader and created a scheduled task"),
    ("p3", "G2", "the ransomware encrypted file shares after deleting volume shadow copies"),
    ("p3", "G2", "operators exfiltrated data with rclone before encrypting file shares"),
    ("p4", "G2", "rclone uploaded stolen data and the ransomware deleted shadow copies"),
    ("p5", "G3", "wiper malware destroyed industrial control systems at energy firms"),
    ("p6", "G3", "the wiper overwrote the master boot record of energy company servers"),
]
POSTS = {
    "p1": {"source": "a", "available_from": "2024-01-01"},
    "p2": {"source": "b", "available_from": "2024-02-01"},
    "p3": {"source": "a", "available_from": "2024-01-15"},
    "p4": {"source": "c", "available_from": "2024-03-01"},
    "p5": {"source": "b", "available_from": "2024-01-20"},
    "p6": {"source": "c", "available_from": "2024-04-01"},
}
PRIOR = np.asarray([0.3, -0.1, -0.2])
QUERY = "powershell downloaded a loader that injected shellcode and deleted shadow copies"


def reference(readable: set[str]) -> dict[str, float]:
    """Store text's way: a fresh TextGroupModel slice over the readable rows."""
    full = TextGroupModel(
        [{"text": t, "source": "store", "owner": k, "groups": [g]} for k, g, t in ROWS],
        ACTORS,
        PRIOR,
    )
    from scipy import sparse

    keep = [i for i, (k, _, _) in enumerate(ROWS) if k in readable]
    counts = full._counts[keep].tocsc()
    df = np.diff(counts.indptr)
    counts = (counts @ sparse.diags((df >= 2).astype(np.float32))).tocsr()
    counts.eliminate_zeros()
    model = TextGroupModel(
        [full.records[i] for i in keep],
        ACTORS,
        PRIOR,
        params=full.params,
        _vectorizer=full._vectorizer,
        _counts=counts,
    )
    return model.scores(QUERY)


@pytest.mark.parametrize(
    ("as_of", "hidden"),
    [(None, set()), ("2024-02-15", set()), ("2024-12-31", {"p2"}), ("2024-03-01", {"p1"})],
)
def test_incremental_equals_slice(as_of, hidden):
    ranker = TextRanker(ROWS, POSTS, ACTORS, PRIOR)
    got = dict(ranker.rank(QUERY, ranker.weights(as_of, hidden)).order)
    readable = {
        k
        for k, p in POSTS.items()
        if (as_of is None or p["available_from"] < as_of) and k not in hidden
    }
    want = reference(readable)
    assert set(got) == set(want)
    for actor, value in want.items():
        assert got[actor] == pytest.approx(value, abs=1e-12)


def test_date_rule_is_strictly_before():
    ranker = TextRanker(ROWS, POSTS, ACTORS)
    w = ranker.weights("2024-01-15", ())
    readable = {ranker.unit_posts[i] for i in np.flatnonzero(w)}
    assert readable == {"p1"}  # p3 is available on the day itself, so it is not read


def test_hidden_and_actor_filters():
    ranker = TextRanker(ROWS, POSTS, ACTORS)
    w = ranker.weights(None, {"p1", "p2"}, allowed_actors={"G1", "G2"})
    readable = {ranker.unit_posts[i] for i in np.flatnonzero(w)}
    assert readable == {"p3", "p4"}
    assert ranker.rank(QUERY, np.zeros(len(w), np.float32)).first is None


def test_unknown_actor_rows_are_dropped_but_shape_the_vocabulary():
    rows = [*ROWS, ("p9", "misp:Other", "powershell loader shellcode staging server")]
    ranker = TextRanker(rows, POSTS, ACTORS)
    assert "p9" not in ranker.unit_posts
    extended = TextRanker(
        rows, {**POSTS, "p9": {"available_from": "2024-01-02"}}, (*ACTORS, "misp:Other")
    )
    assert "misp:Other" in dict(extended.rank(QUERY, extended.weights(None)).order)


def test_ranking_margin_and_rank():
    r = Ranking((("G1", 0.6), ("G2", 0.3), ("G3", 0.1)))
    assert r.first == "G1" and r.margin == pytest.approx(0.3) and r.rank_of("G3") == 3
    assert Ranking(()).margin is None


@pytest.mark.parametrize(
    ("first", "lead", "proposal", "expected"),
    [
        ("G1", 0.5, None, "G1"),
        ("G1", 0.1, None, None),
        ("G1", None, None, None),
        (None, 0.9, None, None),
        ("G1", 0.5, Proposal("G1", "name"), "G1"),
        ("G1", 0.5, Proposal("G2", "name"), None),
        ("G1", 0.5, Proposal("G1", "open"), None),
        ("G1", 0.5, Proposal(None, "name"), None),
    ],
)
def test_decide_truth_table(first, lead, proposal, expected):
    assert decide(first, lead, 0.2, proposal) == expected


def _row(rid, truth, first, margin):
    return {"report_id": rid, "group_id": truth, "accept": [], "first": first, "margin": margin}


def test_score_and_margin_choice():
    rows = [
        _row("a", "G1", "G1", 0.9),
        _row("b", "G1", "G2", 0.2),
        _row("c", None, "G3", 0.5),
        _row("d", "G2", "G2", 0.6),
    ]
    out = score(rows, 0.55)
    assert out["core"] == {
        "first": 2,
        "named": 2,
        "named_wrong": 0,
        "open_named": 0,
        "wrong_total": 0,
    }
    assert smallest_margin(rows, 0.0) == pytest.approx(0.6)
    assert smallest_margin(rows, 0.25) == pytest.approx(0.5)


def test_paired_counts():
    rows = list(range(10))
    out = paired(rows, lambda r: r < 6, lambda r: r < 3)
    assert out["gained"] == 3 and out["lost"] == 0 and out["sign_p"] == pytest.approx(0.25)
