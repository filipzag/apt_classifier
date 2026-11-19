"""Store text as a ranking-only lead (prereg store_text_lead_rule_2026-11-10): store text alone on
a dated store, the rule, the inputs it refuses, the paired scoring, the registered reading and the
threshold's pool."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apt_classifier.knowledge_store.members import StoreState
from apt_classifier.knowledge_store.sources import PUBLISHERS
from scripts import store_text_lead as lead
from tests.test_knowledge_store_members import BRAVO, _kb, _members, _payload

T = lead.THRESHOLD
SHORTLIST = "known_group_shortlist"
NAMED = "known_group_hypothesis"
CLUSTER = "known_cluster_hypothesis"
THIN = "insufficient_evidence"
SCRIPT_SHA = lead.file_sha256(Path(lead.__file__))


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _fold(report_id, statements, *, as_of="2025-03-01", keys=(), group="G2", accept=(), raw=None):
    """A report as the benchmark loads it: ``raw`` are its statements as filed (what the store's
    copy rule reads), ``statements`` what is left of them at the reading."""
    return SimpleNamespace(
        report_id=report_id,
        as_of=as_of,
        store_keys=tuple(keys),
        store_statements=tuple(raw or statements),
        observations=tuple(statements),
        group_id=group,
        accept=frozenset(accept),
    )


def _case(report_id, group, leading, *, stage=SHORTLIST, topk=None, accept=(), cluster=()):
    order = list(topk) if topk is not None else [leading]
    truth = {group, *accept} - {None}
    rank = next((i for i, g in enumerate(order, 1) if g in truth), None)
    return {
        "report_id": report_id,
        "group_id": group,
        "accept": list(accept),
        "leading": leading,
        "stage": stage,
        "named_cluster": list(cluster),
        "topk": [{"group_id": g} for g in order],
        "lenient_rank": rank,
        "observations": 3,
    }


def _entry(first, margin, hidden_posts=0):
    return {
        "first": first,
        "margin": margin,
        "scored_groups": 0 if first is None else (1 if margin is None else 2),
        "top": [],
        "hidden_posts": hidden_posts,
    }


def _row(report_id, group, whole, hidden=(None, None), *, reading="behavior", **extra):
    row = {
        "report_id": report_id,
        "reading": reading,
        "set": "known" if group else "open_set",
        "group_id": group,
        "accept": [],
        "published": "2025-06-01",
        "as_of": "2025-06-01",
        "pin": "before_2025",
        "publisher": "talos",
        "statements": 3,
        "whole_store": _entry(*whole, hidden_posts=1),
        "publisher_hidden": _entry(*hidden, hidden_posts=4),
    }
    row.update(extra)
    return row


def _product(known, open_set=(), *, reading="behavior"):
    inputs = {
        "report_inputs": {"/a/reports.jsonl": "aa", "/a/open_set.jsonl": "bb"},
        "report_rules": "rr",
        "weights": {"text_group": 1.0, "behavior_text": 1.0},
        "combiner": "active",
        "acceptance_model": lead.REGISTERED_ACCEPTANCE,
    }
    if reading == "behavior":
        inputs["mask_store_names"] = 5
    return {
        "pipeline_fingerprint": "pf",
        "fingerprint_inputs": inputs,
        "parameters": {
            "anonymization": reading,
            "pipeline": "local",
            "settings_overrides": dict(lead.LOCAL_OVERRIDES),
            "truncated_at": None,
            "sample": None,
            "limit": None,
            "report_observations": None,
            "report_sample_seed": None,
        },
        "attack_version": "20.0",
        # the benchmark's record of its masking: the same totals, counted in another order
        "leak_control": {
            "actor_name_replacement": {"reports": 2, "names": {"x": 1}, "lost": ["a", "b"]},
            "dropped_hits": 0,
        },
        "known": list(known),
        "open_set": list(open_set),
        "unknown": [],
    }


def _store(rows):
    totals = {"known": {"reports": 2, "names": {"x": 1}, "lost": ["b", "a"]}, "open_set": None}
    return {
        "kind": lead.KIND,
        "schema": lead.SCHEMA,
        "script_sha256": SCRIPT_SHA,
        "inputs": {"/b/reports.jsonl": "aa", "/b/open_set.jsonl": "bb"},
        "attack_version": "20.0",
        "store": dict(lead.REGISTERED_STORE),
        "masking_names": 5,
        "masking": {"behavior": json.loads(json.dumps(totals)), "strict": totals},
        "report_rules": "rr",
        "readings": ["behavior", "strict"],
        "arms": list(lead.ARMS),
        "rows": list(rows),
    }


def _frozen(**files):
    """The decisive set's registration, as far as the script reads it."""
    hashes = {"reports.jsonl": "aa", "open_set.jsonl": "bb", "candidates.json": "cc", **files}
    return {"id": lead.REGISTERED_SET, "frozen": {"files_sha256": hashes}}


def _pairs(gains=0, losses=0, same=0, *, reading="behavior", hidden=(None, None)):
    """A report set where the whole-store rule gains and loses the given numbers of reports."""
    cases, rows = [], []
    for kind, count in (("gain", gains), ("loss", losses), ("same", same)):
        for number in range(count):
            report_id = f"{kind}-{number}"
            leading = "G9" if kind == "gain" else "G1"
            first = {"gain": "G1", "loss": "G9", "same": "G1"}[kind]
            cases.append(_case(report_id, "G1", leading, topk=[leading, "G5"]))
            rows.append(_row(report_id, "G1", (first, T * 2), hidden, reading=reading))
    return cases, rows


# --- store text alone ---------------------------------------------------------------------------


def test_ranking_breaks_ties_by_group_id_and_one_group_has_no_margin():
    order = lead.ranked({"G3": 0.2, "G1": 0.2, "G2": 0.6})
    assert order == [("G2", 0.6), ("G1", 0.2), ("G3", 0.2)]
    assert lead.margin_of(order) == pytest.approx(0.4)
    assert lead.margin_of(lead.ranked({"G1": 1.0})) is None
    assert lead.margin_of([]) is None
    # the first minus the second, not the last
    three = {"G1": 0.5, "G2": 0.3, "G3": 0.2}
    assert lead.margin_of(lead.ranked(three)) == pytest.approx(0.2)
    assert lead.arm_entry(three, 0)["margin"] == pytest.approx(0.2)
    entry = lead.arm_entry({"G1": 0.5, "G2": 0.5}, 3)
    assert entry["first"] == "G1" and entry["margin"] == 0.0 and entry["hidden_posts"] == 3
    assert lead.arm_entry({}, 0) == {
        "first": None,
        "margin": None,
        "scored_groups": 0,
        "top": [],
        "hidden_posts": 0,
    }


def test_the_report_files_give_each_report_its_source(tmp_path):
    reports = _write(
        tmp_path / "reports.jsonl",
        [{"report_id": "r1", "source": "securelist", "published": "2025-03-01T10:00:00"}],
    )
    open_set = _write(tmp_path / "open_set.jsonl", [{"report_id": "o1", "source": "talos"}])
    assert lead.read_sources(reports, open_set, None) == {
        "r1": {"source": "securelist", "published": "2025-03-01"},
        "o1": {"source": "talos", "published": ""},
    }
    with pytest.raises(ValueError, match="listed twice"):
        lead.read_sources(reports, reports)
    with pytest.raises(ValueError, match="without report_id"):
        lead.read_sources(_write(tmp_path / "bad.jsonl", [{"source": "talos"}]))


def test_store_text_alone_reads_the_dated_store_with_and_without_the_own_publisher(tmp_path):
    members = _members(tmp_path)
    state = StoreState({None: members})
    wiper = (
        "A wiper overwrote the master boot record through group policy.",
        "The loader decrypted its payload in memory.",
    )
    masked = ("A wiper overwrote the master boot record.", "The loader decrypted its payload.")
    # as filed the report has a third statement, which the reading drops
    filed = (*wiper, "The report attributes the intrusion to a known cluster.")
    items = [
        ("known", "behavior", _fold("r1", masked, raw=filed)),
        ("known", "strict", _fold("r1", wiper)),
        ("known", "strict", _fold("r2", wiper, as_of="2024-07-15", keys=("b/1",))),
        ("open_set", "strict", _fold("r3", wiper, as_of="", group=None)),
    ]
    sources = {
        "r1": {"source": "securelist", "published": "2025-03-01"},
        "r2": {"source": "", "published": "2024-07-15"},
        "r3": {"source": "talos", "published": ""},
    }
    seen = []
    rows = lead.store_text_rows(
        items,
        sources,
        state,
        _kb(),
        no_store="0001-01-01",
        publishers=PUBLISHERS,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert [(r["report_id"], r["reading"], r["set"]) for r in rows] == [
        ("r1", "behavior", "known"),
        ("r1", "strict", "known"),
        ("r2", "strict", "known"),
        ("r3", "strict", "open_set"),
    ]
    assert seen == [(1, 3), (2, 3), (3, 3)]
    first = rows[1]
    # the whole store holds the later Bravo post on the wiper; with its publisher hidden
    # (securelist and kaspersky_ics are one) Alpha's loader sentence matches best
    assert first["publisher"] == "kaspersky" and first["statements"] == 2
    assert first["as_of"] == "2025-03-01" and first["published"] == "2025-03-01"
    assert first["whole_store"]["first"] == "G2" and first["whole_store"]["margin"] > 0
    assert first["whole_store"]["scored_groups"] == 2 and first["whole_store"]["hidden_posts"] == 0
    assert first["whole_store"]["top"][0][0] == "G2" and len(first["whole_store"]["top"]) == 2
    assert first["publisher_hidden"]["first"] == "G1" and first["publisher_hidden"]["margin"] > 0
    assert first["publisher_hidden"]["hidden_posts"] == 4  # b/2, l/2, l/3, l/4
    # each reading is scored on its own statements: masked, they share words with Alpha alone
    # once Bravo's later post is hidden, and one scored group has no margin
    assert rows[0]["statements"] == 2 and rows[0]["whole_store"]["first"] == "G2"
    assert rows[0]["whole_store"]["margin"] < first["whole_store"]["margin"]
    assert rows[0]["publisher_hidden"]["scored_groups"] == 1
    assert rows[0]["publisher_hidden"]["margin"] is None
    # no source in the report file: the publisher of the report's own store post; the post and
    # its copy are hidden either way and eset's other post is dated after the report
    own = rows[2]
    assert own["publisher"] == "eset"
    assert own["whole_store"]["hidden_posts"] == own["publisher_hidden"]["hidden_posts"] == 2
    assert own["whole_store"]["first"] == "G1"  # Bravo's only earlier post is its own
    # an undated report reads nothing
    undated = rows[3]
    assert undated["as_of"] == "0001-01-01" and undated["group_id"] is None
    assert undated["whole_store"] == undated["publisher_hidden"] == lead.arm_entry({}, 0)
    assert state.case.as_of is None and not state.case.hidden  # left clean

    # a report whose text the store holds under another key: the copy rule hides those posts
    # (three shared statements), so store text cannot lead with the report's own copy
    copied = [line.replace("Bravo", "Sandstorm") for line in BRAVO]
    copy = lead.store_text_rows(
        [("known", "strict", _fold("r4", copied, as_of="2024-07-15"))],
        {"r4": {"source": "talos", "published": "2024-07-15"}},
        state,
        _kb(),
        no_store="0001-01-01",
        publishers=PUBLISHERS,
    )
    assert copy[0]["whole_store"]["hidden_posts"] == 2  # b/1 and its copy
    assert copy[0]["whole_store"]["first"] != "G2"

    with pytest.raises(ValueError, match="no source"):
        lead.store_text_rows(
            [("known", "strict", _fold("r9", wiper))],
            {},
            state,
            _kb(),
            no_store="0001-01-01",
            publishers=PUBLISHERS,
        )
    with pytest.raises(ValueError, match="twice at one reading"):
        lead.store_text_rows(
            [("known", "strict", _fold("r1", wiper)), ("known", "strict", _fold("r1", wiper))],
            sources,
            state,
            _kb(),
            no_store="0001-01-01",
            publishers=PUBLISHERS,
        )


def test_a_dated_report_reads_the_pinned_copy_of_the_store(tmp_path):
    # the copy for reports before 2026-03 labels the later wiper post with another group
    pinned = _payload()
    for row in pinned["text"]:
        if row["key"] == "b/2":
            row["group"] = "G3"
    state = StoreState(
        {
            None: _members(tmp_path),
            "before_2025": _members(tmp_path, pinned, pin="before_2025"),
        }
    )
    wiper = ("A wiper overwrote the master boot record through group policy.",)
    rows = lead.store_text_rows(
        [
            ("known", "strict", _fold("dated", wiper)),
            ("known", "strict", _fold("undated", wiper, as_of="")),
        ],
        {"dated": {"source": "talos"}, "undated": {"source": "talos"}},
        state,
        _kb(),
        no_store="0001-01-01",
        publishers=PUBLISHERS,
    )
    assert [row["pin"] for row in rows] == ["before_2025", "before_2025"]
    assert rows[0]["whole_store"]["first"] == "G3"  # the whole store would say G2
    assert rows[1]["whole_store"]["first"] is None  # no date: nothing is read


def test_reports_are_loaded_and_masked_as_the_benchmark_does_it():
    calls = []

    class Bench:
        @staticmethod
        def load_report_folds(path, *, max_observations, anonymizer, sample_seed, kb):
            calls.append(("load", path.name, max_observations, anonymizer, sample_seed, kb))
            if path.name == "open.jsonl":
                return [SimpleNamespace(group_id=None, name="o1")]
            folds = [SimpleNamespace(group_id="G1", name="k1")]
            if path.name == "mixed.jsonl":
                folds.append(SimpleNamespace(group_id=None, name="k-open"))
            return folds

        @staticmethod
        def anonymize_report_folds(folds, kb, *, level, extra_names):
            calls.append(("mask", [fold.name for fold in folds], level, extra_names, kb))
            return list(folds), [], {"reports": len(folds), "level": level}

    class Dropper:
        def __init__(self, kb, level):
            self.built_with = (kb, level)

    known, open_folds, totals = lead.load_reading(
        Bench, Dropper, "kb", Path("mixed.jsonl"), Path("open.jsonl"), "behavior", ["x"]
    )
    assert [fold.name for fold in known] == ["k1"]
    assert [fold.name for fold in open_folds] == ["k-open", "o1"]
    loads = [call for call in calls if call[0] == "load"]
    # every statement of every report (no cap, no sample), attribution prose dropped at load
    assert [(c[1], c[2], c[4], c[5]) for c in loads] == [
        ("mixed.jsonl", None, None, "kb"),
        ("open.jsonl", None, None, "kb"),
    ]
    assert loads[0][3] is loads[1][3] and loads[0][3].built_with == ("kb", "behavior")
    # the store's names are masked at the behaviour reading, for known and open-set reports
    assert [call for call in calls if call[0] == "mask"] == [
        ("mask", ["k1"], "behavior", ("x",), "kb"),
        ("mask", ["k-open", "o1"], "behavior", ("x",), "kb"),
    ]
    assert totals == {
        "known": {"reports": 1, "level": "behavior"},
        "open_set": {"reports": 2, "level": "behavior"},
    }
    calls.clear()
    known, open_folds, totals = lead.load_reading(
        Bench, Dropper, "kb", Path("known.jsonl"), None, "strict", ["x"]
    )
    # ... and at no other reading; without open-set reports there is nothing to mask for them
    assert [call[1:4] for call in calls if call[0] == "mask"] == [(["k1"], "strict", ())]
    assert open_folds == [] and totals["open_set"] is None


# --- the rule -----------------------------------------------------------------------------------


def test_store_text_leads_only_an_unnamed_report_at_or_above_the_threshold():
    case = _case("r", "G1", "G9")
    assert lead.decide(case, _entry("G1", T), T) == {"named": False, "leads": True, "leader": "G1"}
    assert lead.decide(case, _entry("G1", T - 1e-9), T)["leader"] == "G9"
    assert lead.decide(case, _entry("G1", None), T)["leads"] is False  # one scored group
    assert lead.decide(case, _entry(None, None), T)["leader"] == "G9"  # nothing readable
    assert lead.decide(case, _entry("G1", 0.0), T)["leads"] is False  # a tie of the first two
    for stage in (NAMED, CLUSTER):
        named = _case("r", "G1", "G9", stage=stage)
        assert lead.decide(named, _entry("G1", 0.9), T) == {
            "named": True,
            "leads": False,
            "leader": "G9",
        }
    unresolved = _case("r", "G1", "G9", stage="unresolved_activity_cluster")
    assert lead.decide(unresolved, _entry("G1", 0.9), T)["leads"] is True


def test_a_lead_moves_to_the_front_of_the_first_five():
    case = _case("r", "G1", "G2", topk=["G2", "G3", "G4", "G5", "G1", "G6"])
    assert lead.top_five(case) == ["G2", "G3", "G4", "G5", "G1"]
    assert lead.top_five(case, "G7") == ["G7", "G2", "G3", "G4", "G5"]  # pushes the fifth out
    assert lead.top_five(case, "G4") == ["G4", "G2", "G3", "G5", "G1"]
    assert lead.truth_of(_case("r", "G1", "G2", accept=["G3"])) == {"G1", "G3"}
    assert lead.truth_of(_case("r", None, "G2")) == set()


def test_scoring_pairs_the_product_with_the_rule_and_leaves_the_names_alone():
    known = [
        _case("gain", "G1", "G9", topk=["G9", "G5"]),
        _case("loss", "G1", "G1", topk=["G1", "G5"]),
        _case("weak", "G1", "G9", topk=["G9", "G1"]),  # below the threshold: no lead
        _case("named", "G2", "G2", stage=NAMED, topk=["G2", "G3"]),
        _case("wrong-name", "G2", "G3", stage=NAMED, topk=["G3", "G2"]),
        _case("cluster", "G4", "G3", stage=CLUSTER, topk=["G3", "G4"], cluster=["G3", "G4"]),
        _case("accepted", "G2", "G9", topk=["G9"], accept=["G7"]),
        _case("push-out", "G3", "G9", topk=["G9", "G5", "G6", "G7", "G3"]),
    ]
    rows = [
        _row("gain", "G1", ("G1", T), ("G1", T)),
        _row("loss", "G1", ("G8", 0.5), ("G1", 0.5)),
        _row("weak", "G1", ("G1", T / 2), ("G1", T / 2)),
        _row("named", "G2", ("G1", 0.9), ("G1", 0.9)),
        _row("wrong-name", "G2", ("G2", 0.9), ("G2", 0.9)),
        _row("cluster", "G4", ("G1", 0.9), ("G1", 0.9)),
        _row("accepted", "G2", ("G7", 0.2), (None, None), accept=["G7"]),
        _row("push-out", "G3", ("G8", 0.3), (None, None)),
        _row("open-lead", None, ("G1", 0.2), (None, None)),
        _row("open-named", None, ("G1", 0.9), ("G1", 0.9)),
        _row("gain", "G1", ("G1", 0.0), reading="strict"),  # another reading: not read
    ]
    open_set = [_case("open-lead", None, "G5"), _case("open-named", None, "G5", stage=NAMED)]
    known[6]["accept"] = ["G7"]
    result = lead.score(_product(known, open_set), _store(rows))
    assert result["reading"] == "behavior" and result["registered_threshold"] is True
    assert result["frozen_set_checked"] is False
    assert (result["known"], result["groups"], result["open_set"]) == (8, 4, 2)
    assert result["product_first_by_rank"] == 2  # loss and named
    assert result["names"] == {
        "known": {"named": 3, "right": 2, "wrong": 1},  # the cluster holds the truth
        "open_set": {"named": 1},
    }
    assert result["known_without_other_publisher_posts"] == 0
    whole = result["arms"]["whole_store"]
    assert (whole["known"], whole["not_named"]) == (8, 5)
    assert (whole["first_before"], whole["first_after"]) == (2, 3)
    assert whole["gained"] == ["gain", "accepted"] and whole["lost"] == ["loss"]
    assert whole["difference"] == pytest.approx(0.125)
    assert (whole["leads"], whole["leads_right"], whole["leads_wrong"]) == (4, 2, 2)
    assert whole["leader_changes"] == 4
    # first five: gain and accepted enter, the truth of the loss stays, the lead on push-out
    # drops its truth from them
    assert (whole["top5_before"], whole["top5_after"]) == (6, 7)
    assert whole["mcnemar_p"] == 1.0 and whole["sign_test_one_sided_p"] == 0.5
    assert whole["open_set"] == {"reports": 2, "not_named": 1, "leads": 1, "leader_changes": 1}
    by_id = {case["report_id"]: case for case in whole["cases"]}
    assert by_id["wrong-name"]["leader"] == "G3" and by_id["wrong-name"]["leads"] is False
    assert by_id["weak"]["leader"] == "G9" and by_id["gain"]["leader"] == "G1"
    hidden = result["arms"]["publisher_hidden"]
    assert (hidden["first_before"], hidden["first_after"]) == (2, 3)
    assert hidden["gained"] == ["gain"] and hidden["lost"] == []
    assert hidden["leader_changes"] == 1 and hidden["leads"] == 2  # the lead on loss agrees
    assert hidden["open_set"]["leads"] == 0
    # per group: G1 1/3 -> 1/3 (gain and loss), G2 1/3 -> 2/3, G3 0, G4 0
    assert whole["macro_first_before"] == pytest.approx(1 / 6, abs=1e-4)
    assert whole["macro_first_after"] == pytest.approx(0.25, abs=1e-4)
    assert result["known_without_product_leader"] == 0


def test_a_report_without_a_product_leader_reads_the_first_of_its_ranking():
    # with too little evidence the product reports no leader, yet ranks the groups
    known = [
        _case("kept", "G1", None, stage=THIN, topk=["G1", "G5"]),
        _case("same", "G1", None, stage=THIN, topk=["G1", "G5"]),
        _case("lost", "G1", None, stage=THIN, topk=["G1", "G5"]),
        _case("gain", "G1", None, stage=THIN, topk=["G5", "G1"]),
        _case("empty", "G1", None, stage=THIN, topk=[]),
    ]
    rows = [
        _row("kept", "G1", (None, None)),
        _row("same", "G1", ("G1", 0.5)),  # a right lead where the ranking was right: no gain
        _row("lost", "G1", ("G9", 0.5)),  # a wrong lead where the ranking was right: a loss
        _row("gain", "G1", ("G1", 0.5)),
        _row("empty", "G1", ("G1", 0.5)),
    ]
    assert lead.product_leader(known[0]) == "G1" and lead.product_leader(known[4]) is None
    assert lead.product_leader(_case("r", "G1", "G9", topk=["G5", "G9"])) == "G9"
    result = lead.score(_product(known), _store(rows))
    whole = result["arms"]["whole_store"]
    assert (whole["first_before"], whole["first_after"]) == (3, 4)
    assert whole["gained"] == ["gain", "empty"] and whole["lost"] == ["lost"]
    assert result["product_first_by_rank"] == whole["first_before"]
    assert result["known_without_product_leader"] == 5
    assert whole["leader_changes"] == 3  # the right lead on "same" changes nothing
    by_id = {case["report_id"]: case for case in whole["cases"]}
    assert by_id["kept"]["product_leader"] == "G1" and by_id["kept"]["leader"] == "G1"
    assert by_id["empty"]["product_leader"] is None
    assert (whole["top5_before"], whole["top5_after"]) == (4, 5)


def test_an_accepted_alternative_counts_for_the_product_too():
    row = _row("alt", "G2", ("G2", 0.5), accept=["G7"])
    case = _case("alt", "G2", "G7", topk=["G7", "G2"], accept=["G7"])
    whole = lead.score(_product([case]), _store([row]))["arms"]["whole_store"]
    assert (whole["first_before"], whole["first_after"]) == (1, 1)
    assert whole["gained"] == [] and whole["lost"] == [] and whole["leads"] == 1
    named = _case("alt", "G2", "G7", stage=NAMED, topk=["G7", "G2"], accept=["G7"])
    names = lead.score(_product([named]), _store([row]))["names"]
    assert names["known"] == {"named": 1, "right": 1, "wrong": 0}


# --- refused inputs -----------------------------------------------------------------------------


def _good():
    cases, rows = _pairs(gains=1, same=1)
    return _product(cases), _store(rows)


def test_a_matching_pair_of_files_scores():
    product, store = _good()
    assert lead.check_inputs(product, store, threshold=T) == "behavior"
    assert lead.score(product, store)["arms"]["whole_store"]["gained"] == ["gain-0"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, s: s.update(kind="other"), "not a store-text file"),
        (lambda p, s: p["parameters"].update(anonymization="tradecraft"), "reading"),
        (lambda p, s: p["parameters"].update(truncated_at=30), "every report in full"),
        (lambda p, s: p["parameters"].update(limit=5), "every report in full"),
        (lambda p, s: p["parameters"].update(report_observations=40), "every report in full"),
        (lambda p, s: p["fingerprint_inputs"]["weights"].update(store_text=1.0), "store members"),
        (
            lambda p, s: p["fingerprint_inputs"]["weights"].update(store_software=0.5),
            "store members",
        ),
        (lambda p, s: p["fingerprint_inputs"].update(combiner="inactive"), "combiner"),
        (lambda p, s: s.update(arms=["whole_store"]), "both arms"),
        (lambda p, s: s["rows"][0].pop("publisher_hidden"), "no publisher_hidden entry"),
        (lambda p, s: s.update(report_rules="other"), "different rules"),
        (lambda p, s: p["fingerprint_inputs"].pop("report_rules"), "different rules"),
        (
            lambda p, s: s["masking"]["behavior"]["known"].update(reports=3),
            "masked the known reports differently",
        ),
        (
            lambda p, s: p["leak_control"].update(open_set_actor_name_replacement={"reports": 1}),
            "masked the open-set reports differently",
        ),
        (lambda p, s: p.update(leak_control="none"), "masked the known reports differently"),
        (lambda p, s: p["fingerprint_inputs"].update(mask_store_names=4), "store names masked"),
        (lambda p, s: p["fingerprint_inputs"].pop("mask_store_names"), "store names masked"),
        (lambda p, s: s.update(attack_version="19.2"), "ATT&CK"),
        (lambda p, s: s["inputs"].update({"/b/reports.jsonl": "zz"}), "different report files"),
        (lambda p, s: p["fingerprint_inputs"].update(report_inputs={}), "different report files"),
        (lambda p, s: p.update(unknown=[{"report_id": "x"}]), "fold cases"),
        (lambda p, s: s["rows"].pop(), "same reports"),
        (lambda p, s: p["known"].pop(), "same reports"),
        (lambda p, s: s["rows"].append(dict(s["rows"][0])), "twice in the store-text file"),
        (lambda p, s: p["known"].append(dict(p["known"][0])), "listed twice"),
        (lambda p, s: s["rows"][0].update(group_id="G7"), "disagree on its group"),
        (lambda p, s: s["rows"][0].update(accept=["G7"]), "accepted groups"),
        (lambda p, s: s["rows"][0].update(statements=2), "statements in the product"),
    ],
)
def test_inputs_that_do_not_belong_together_are_refused(change, message):
    product, store = _good()
    change(product, store)
    with pytest.raises(ValueError, match=message):
        lead.score(product, store)


def test_the_strict_reading_masks_no_store_names():
    cases, rows = _pairs(gains=1, reading="strict")
    assert lead.score(_product(cases, reading="strict"), _store(rows))["reading"] == "strict"
    product = _product(cases, reading="strict")
    product["fingerprint_inputs"]["mask_store_names"] = 5
    with pytest.raises(ValueError, match="outside the behaviour reading"):
        lead.score(product, _store(rows))


def test_a_store_text_file_without_masking_records_scores_in_development_only():
    # files written before the masking was recorded (the first fresh2 run)
    product, store = _good()
    del store["masking"], store["report_rules"]
    result = lead.score(product, store)
    assert result["inputs"]["masking_compared"] is False
    assert lead.score(*_good())["inputs"]["masking_compared"] is True
    assert lead.canonical({"b": [2, 1], "a": {"c": ["y", "x"]}, "d": [{"k": 1}, {"k": 0}]}) == {
        "b": [1, 2],
        "a": {"c": ["x", "y"]},
        "d": [{"k": 1}, {"k": 0}],  # a list of records keeps its order
    }


def test_the_frozen_set_needs_its_hashes_the_registered_threshold_and_registration():
    frozen = _frozen()
    product, store = _good()
    assert lead.score(product, store, frozen=frozen)["frozen_set_checked"] is True
    with pytest.raises(ValueError, match="registered threshold"):
        lead.score(product, store, threshold=0.02, frozen=frozen)
    assert lead.score(product, store, threshold=0.02)["registered_threshold"] is False
    with pytest.raises(ValueError, match="not the frozen set"):
        lead.score(product, store, frozen=_frozen(**{"open_set.jsonl": "zz"}))
    one_file = {"id": lead.REGISTERED_SET, "frozen": {"files_sha256": {"reports.jsonl": "aa"}}}
    with pytest.raises(ValueError, match="no frozen hashes"):
        lead.score(product, store, frozen=one_file)
    with pytest.raises(ValueError, match="not the registration"):
        lead.score(product, store, frozen={**frozen, "id": "a_later_set"})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, s: p["parameters"].update(pipeline="production"), "local pipeline"),
        (
            lambda p, s: p["parameters"]["settings_overrides"].update(text_group_weight=0.0),
            "product's defaults",
        ),
        (lambda p, s: p["fingerprint_inputs"].update(acceptance_model="active"), "technique head"),
        (lambda p, s: s.update(script_sha256="older"), "another version of this script"),
        (lambda p, s: s["store"].update(before_2025="rebuilt"), "another before_2025 store file"),
        (lambda p, s: s["store"].pop("whole"), "another whole store file"),
        (lambda p, s: s["rows"][0].update(pin="before_2026_03"), "before_2025 copy"),
        (lambda p, s: s.pop("masking"), "does not record how"),
        (lambda p, s: s.pop("report_rules"), "does not record how"),
    ],
)
def test_the_registered_run_refuses_what_the_registration_does_not_name(change, message):
    product, store = _good()
    change(product, store)
    lead.score(product, store)  # a development scoring takes it
    with pytest.raises(ValueError, match=message):
        lead.score(product, store, frozen=_frozen())


def test_the_registrations_hold_what_the_script_reads():
    docs = Path(lead.__file__).resolve().parent.parent / "docs" / "preregistration"
    frozen = json.loads((docs / f"{lead.REGISTERED_SET}.json").read_text(encoding="utf-8"))
    files = frozen["frozen"]["files_sha256"]
    assert frozen["id"] == lead.REGISTERED_SET
    assert len({files["reports.jsonl"], files["open_set.jsonl"]}) == 2
    rule = json.loads((docs / "store_text_lead_rule_2026-11-10.json").read_text(encoding="utf-8"))
    assert rule["threshold"]["value"] == lead.THRESHOLD
    assert set(lead.REGISTERED_STORE.values()) <= set(
        rule["store_text_alone"]["store_files"].values()
    )
    # the registered script: an edit before the decisive run needs an amendment of the hash
    assert rule["code"]["script_sha256"] == SCRIPT_SHA


# --- statistics and the registered reading ------------------------------------------------------


def test_the_exact_tests_and_the_interval():
    assert lead.sign_test(5, 0) == 0.0312 and lead.sign_test(4, 0) == 0.0625
    assert lead.sign_test(7, 1) == 0.0352 and lead.sign_test(6, 1) == 0.0625
    assert lead.sign_test(9, 2) == 0.0327 and lead.sign_test(0, 0) == 1.0
    assert lead.sign_test(1, 4) > 0.9
    assert lead.mcnemar(0, 0) == 1.0 and lead.mcnemar(5, 0) == 0.0625
    assert lead.mcnemar(9, 1) == 0.0215 and lead.mcnemar(1, 9) == 0.0215
    after, before = [1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1]
    interval = lead.bootstrap_diff(after, before)
    assert interval == lead.bootstrap_diff(after, before)  # fixed seed
    assert interval[0] <= 1 / 6 <= interval[1] and -1 <= interval[0] < interval[1] <= 1
    assert lead.bootstrap_diff([1, 1], [1, 1]) == [0.0, 0.0]
    assert lead.bootstrap_diff([], []) is None
    assert lead.macro(["a", "a", "b"], [1, 0, 1]) == 0.75 and lead.macro([], []) is None
    # the registered level, on both sides of it
    for gained, lost, passes in (
        (5, 0, True),
        (4, 0, False),
        (7, 1, True),
        (6, 1, False),
        (10, 3, True),  # p 0.0461
        (8, 2, False),  # p 0.0547
        (3, 3, False),
        (0, 0, False),
    ):
        assert lead.beyond_chance(gained, lost) is passes


def _scores(behavior, strict, *, hidden_behavior=(0, 0), hidden_strict=(0, 0)):
    """Score files with the given (gains, losses) per reading and arm."""

    def one(reading, whole, hidden):
        cases, rows = [], []
        kinds = (("gain", max(whole[0], hidden[0])), ("loss", max(whole[1], hidden[1]) + 1))
        for quota, (kind, count) in enumerate(kinds):
            for number in range(count):
                report_id = f"{kind}-{number}"
                # the product is wrong where a lead can gain and right where it can lose
                cases.append(_case(report_id, "G1", "G9" if kind == "gain" else "G1"))
                leads = ("G1" if kind == "gain" else "G9", T * 2)
                rows.append(
                    _row(
                        report_id,
                        "G1",
                        leads if number < whole[quota] else (None, None),
                        leads if number < hidden[quota] else (None, None),
                        reading=reading,
                    )
                )
        return lead.score(_product(cases, reading=reading), _store(rows))

    return one("behavior", behavior, hidden_behavior), one("strict", strict, hidden_strict)


def test_the_registered_reading_needs_gains_beyond_chance_and_no_losing_arm():
    result = lead.verdict(*_scores((5, 0), (2, 1), hidden_behavior=(1, 1)))
    assert result["supported"] is True and all(result["checks"].values())
    assert result["primary"]["gained"] == 5 and result["primary"]["sign_test_one_sided_p"] == 0.0312
    assert result["frozen_set_checked"] is False
    # the file says what it was made from
    assert result["threshold"] == T and result["script_sha256"] == SCRIPT_SHA
    assert result["inputs"]["store"] == lead.REGISTERED_STORE
    assert lead.verdict(*_scores((10, 3), (0, 0)))["supported"] is True
    assert lead.verdict(*_scores((8, 2), (0, 0)))["supported"] is False
    # four gains alone are within chance
    few = lead.verdict(*_scores((4, 0), (2, 0)))
    assert few["supported"] is False
    assert few["checks"]["behaviour, whole store: gains beyond chance"] is False
    assert lead.verdict(*_scores((7, 1), (0, 0)))["supported"] is True
    assert lead.verdict(*_scores((6, 1), (0, 0)))["supported"] is False
    # an arm that loses more than it gains
    for changes, failed in (
        ({"hidden_behavior": (0, 1)}, "behaviour, own publisher hidden"),
        ({"hidden_strict": (1, 2)}, "strict, own publisher hidden"),
    ):
        losing = lead.verdict(*_scores((6, 0), (2, 2), **changes))
        assert losing["supported"] is False
        assert losing["checks"][f"{failed}: gained at least as many as lost"] is False
    assert lead.verdict(*_scores((6, 0), (1, 2)))["supported"] is False


def test_the_registered_reading_refuses_other_files():
    behavior, strict = _scores((5, 0), (1, 0))
    with pytest.raises(ValueError, match="strict reading"):
        lead.verdict(behavior, behavior)
    with pytest.raises(ValueError, match="behavior reading"):
        lead.verdict(strict, strict)
    cases, rows = _pairs(gains=1, reading="strict")
    with pytest.raises(ValueError, match="registered threshold"):
        lead.verdict(
            behavior, lead.score(_product(cases, reading="strict"), _store(rows), threshold=0.5)
        )
    store = _store(rows)
    store["inputs"] = {"/c/reports.jsonl": "cc"}
    product = _product(cases, reading="strict")
    product["fingerprint_inputs"]["report_inputs"] = {"/c/reports.jsonl": "cc"}
    with pytest.raises(ValueError, match="same report set"):
        lead.verdict(behavior, lead.score(product, store))
    # one reading scored on a rebuilt store, or by another version of the script
    rebuilt = json.loads(json.dumps(strict))
    rebuilt["inputs"]["store"]["before_2025"] = "rebuilt"
    with pytest.raises(ValueError, match="differ in what they read"):
        lead.verdict(behavior, rebuilt)
    with pytest.raises(ValueError, match="different versions"):
        lead.verdict(behavior, dict(strict, script_sha256="older"))
    # the threshold itself is read, not the stored flag
    with pytest.raises(ValueError, match="registered threshold"):
        lead.verdict(behavior, dict(strict, threshold=0.02))
    # score files of the registered run must come from this version of the script
    checked = [dict(item, frozen_set_checked=True) for item in (behavior, strict)]
    assert lead.verdict(*checked)["frozen_set_checked"] is True
    with pytest.raises(ValueError, match="another version of this script"):
        lead.verdict(*(dict(item, script_sha256="older") for item in checked))


# --- the threshold ------------------------------------------------------------------------------


def test_the_threshold_is_the_smallest_margin_of_the_top_share():
    assert lead.quantile_threshold([0.5, 0.1, 0.3, 0.2], 0.5) == 0.3
    assert lead.quantile_threshold([0.5, 0.1, 0.3, 0.2], 0.25) == 0.5
    assert lead.quantile_threshold([0.5, 0.1, 0.3, 0.2], 1.0) == 0.1
    assert lead.quantile_threshold([0.4], 0.01) == 0.4
    # of an odd number the middle value belongs to the top half
    assert lead.quantile_threshold([0.5, 0.4, 0.3, 0.2, 0.1], 0.5) == 0.3
    assert lead.quantile_threshold([7, 6, 5, 4, 3, 2, 1], 0.5) == 4
    assert lead.quantile_threshold(range(214, 0, -1), 0.5) == 108  # the 107th of 214
    with pytest.raises(ValueError):
        lead.quantile_threshold([], 0.5)


def test_the_threshold_pool_is_the_earlier_posts_with_a_margin():
    rows = [
        _row("a", "G1", ("G1", 0.4), published="2023-05-01"),
        _row("b", None, ("G1", 0.2), published="2023-12-31"),  # the open set counts
        _row("c", "G1", ("G1", None), published="2022-01-01"),  # one scored group
        _row("d", "G1", (None, None), published="2021-01-01"),  # nothing readable
        _row("e", "G1", ("G1", 0.9), published="2024-01-01"),  # from the cut on
        _row("f", "G1", ("G1", 0.8), published=""),  # undated
        _row("g", "G1", ("G1", 0.1), ("G1", 0.7), published="2020-01-01"),
        _row("a", "G1", ("G1", 0.6), published="2023-05-01", reading="strict"),
    ]
    store = _store(rows)
    assert sorted(lead.threshold_pool(store)) == [0.1, 0.2, 0.4]
    assert lead.threshold_pool(store, reading="strict") == [0.6]
    derived = lead.derive_threshold(store)
    assert derived["registered"] == T and derived["share"] == 0.5
    assert derived["readings"]["behavior"] == {"posts": 3, "threshold": 0.2, "rounded": 0.2}
    assert derived["readings"]["strict"]["posts"] == 1
    with pytest.raises(ValueError, match="not a store-text file"):
        lead.derive_threshold({"kind": "other"})


# --- command line -------------------------------------------------------------------------------


def test_the_command_line_scores_files_and_refuses_with_exit_code_2(tmp_path, capsys):
    paths = {}
    for reading in lead.READINGS:
        cases, rows = _pairs(gains=5, same=2, reading=reading)
        product = tmp_path / f"product_{reading}.json"
        product.write_text(json.dumps(_product(cases, reading=reading)), encoding="utf-8")
        store = tmp_path / f"store_{reading}.json"
        store.write_text(json.dumps(_store(rows)), encoding="utf-8")
        paths[reading] = tmp_path / "out" / f"score_{reading}.json"
        assert lead.main(["score", str(product), str(store), "--output", str(paths[reading])]) == 0
    written = json.loads(paths["behavior"].read_text(encoding="utf-8"))
    assert written["kind"] == lead.SCORE_KIND and written["arms"]["whole_store"]["first_after"] == 7
    assert written["script_sha256"] == lead.file_sha256(Path(lead.__file__))
    assert "first 2 -> 7 of 7" in capsys.readouterr().out
    assert lead.main(["verdict", str(paths["behavior"]), str(paths["strict"])]) == 0
    out = capsys.readouterr().out
    assert "the rule is supported (not the frozen decisive set)" in out
    assert lead.main(["threshold", str(tmp_path / "store_behavior.json")]) == 0
    assert f"registered: {T}" in capsys.readouterr().out
    # a store-text file of other reports
    other = _store(_pairs(gains=5, same=2)[1])
    other["inputs"] = {"/c/reports.jsonl": "cc"}
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps(other), encoding="utf-8")
    assert lead.main(["score", str(tmp_path / "product_behavior.json"), str(bad)]) == 2
    assert "refused: the product run and the store-text file read different" in (
        capsys.readouterr().err
    )
    # a file that is not there is refused, not a traceback
    assert lead.main(["score", str(tmp_path / "product_behavior.json"), str(tmp_path / "x")]) == 2
    assert "refused:" in capsys.readouterr().err


def test_the_command_line_applies_the_registered_checks(tmp_path, capsys):
    cases, rows = _pairs(gains=5, same=2)
    product = tmp_path / "product.json"
    product.write_text(json.dumps(_product(cases)), encoding="utf-8")
    store = tmp_path / "store.json"
    store.write_text(json.dumps(_store(rows)), encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(_frozen()), encoding="utf-8")
    out = tmp_path / "score.json"
    command = ["score", str(product), str(store), "--frozen", str(frozen)]
    assert lead.main([*command, "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["frozen_set_checked"] is True
    capsys.readouterr()
    assert lead.main([*command, "--threshold", "0.02"]) == 2
    assert "refused: the registered threshold" in capsys.readouterr().err
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_frozen(**{"reports.jsonl": "zz"})), encoding="utf-8")
    assert lead.main(["score", str(product), str(store), "--frozen", str(other)]) == 2
    assert "refused: the report files are not the frozen set" in capsys.readouterr().err
