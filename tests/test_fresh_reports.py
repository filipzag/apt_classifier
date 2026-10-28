"""Fresh-report head-to-head set (``prepare_fresh_reports.py``) and its scoring
(``fresh_h2h_score.py``): actor names, the registered label rule, masking, dates and the
registered combination. Synthetic knowledge base, MISP list and artifacts; nothing is downloaded."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from scripts import fresh_h2h_score as h2h
from scripts import prepare_fresh_reports as fresh


def _kb():
    def group(name, aliases=()):
        return SimpleNamespace(name=name, aliases=tuple(aliases))

    return SimpleNamespace(
        groups={
            "G0032": group("Lazarus Group", ["Lazarus Group", "Diamond Sleet"]),
            "G0007": group("APT28", ["APT28", "Fancy Bear", "Sednit"]),
            "G0129": group("Mustang Panda", ["Mustang Panda"]),
            "G1014": group("LuminousMoth", ["LuminousMoth"]),
            "G1045": group("Salt Typhoon", ["Salt Typhoon", "GhostEmperor"]),
        },
        software={"S0001": SimpleNamespace(name="Sednit", aliases=("Sofacy",))},
    )


def _misp():
    return {
        "values": [
            {
                "uuid": "u-apt28",
                "value": "APT28",
                "meta": {
                    "synonyms": ["Fighting Ursa", "Sofacy"],
                    "refs": ["https://attack.mitre.org/groups/G0007/"],
                },
            },
            {
                "uuid": "u-mustang",
                "value": "MUSTANG PANDA",
                "meta": {"synonyms": ["Stately Taurus", "LuminousMoth"]},
            },
            {"uuid": "u-ghost", "value": "GhostEmperor", "meta": {}},
            {
                "uuid": "u-estries",
                "value": "Earth Estries",
                "related": [{"dest-uuid": "u-ghost", "type": "similar"}],
            },
            {"uuid": "u-void", "value": "Void Blizzard", "meta": {"synonyms": ["Laundry Bear"]}},
        ]
    }


def _dictionary():
    dictionary = fresh.actor_dictionary(_kb(), _misp())
    return dictionary, fresh.name_pattern(list(dictionary["surface"]))


def test_actor_dictionary_maps_vendor_names_and_keeps_attack_names():
    surface = _dictionary()[0]["surface"]
    assert surface["Lazarus"] == "G0032"  # short form of "Lazarus Group"
    assert surface["Fighting Ursa"] == "G0007"  # MISP synonym, ATT&CK reference
    assert surface["Stately Taurus"] == "G0129"  # MISP actor mapped by its own name
    assert surface["LuminousMoth"] == "G1014"  # an ATT&CK name keeps its own group
    assert "Sednit" not in surface and "Sofacy" not in surface  # also software names
    assert surface["Earth Estries"] == "related:G1045"  # MISP links it to an ATT&CK group
    assert surface["Laundry Bear"] == "misp:Void Blizzard"


def _statements(lead, rest, n=20):
    return [lead, *([rest] * (n - 1))]


def test_label_known_open_and_excluded():
    dictionary, pattern = _dictionary()
    known = fresh.label_report(
        "Fighting Ursa targets ministries",
        _statements("Fancy Bear used a new loader.", "APT28 then moved laterally."),
        dictionary,
        pattern,
    )
    assert known["label"] == "known" and known["group_id"] == "G0007"
    open_ = fresh.label_report(
        "Storm-2999 hits telecoms",
        _statements("Storm-2999 used web shells.", "The actor Storm-2999 persisted."),
        dictionary,
        pattern,
    )
    assert open_["label"] == "open" and open_["actor"] == "code:Storm-2999"
    related = fresh.label_report(
        "Earth Estries returns",
        _statements("Earth Estries used a backdoor.", "Earth Estries exfiltrated data."),
        dictionary,
        pattern,
    )
    assert related == {**related, "label": "excluded", "reason": "leader_related_to_attack_group"}
    mixed = fresh.label_report(
        "Two actors",
        _statements("APT28 and Lazarus both appear.", "APT28 and Lazarus again."),
        dictionary,
        pattern,
    )
    assert mixed["label"] == "excluded" and mixed["reason"] == "no_dominant_actor"
    late = fresh.label_report(
        "A campaign",
        [*["Generic statement about phishing."] * 12, *["APT28 did it."] * 8],
        dictionary,
        pattern,
    )
    assert late["label"] == "excluded"  # named only after the lead statements
    short = fresh.label_report("APT28", ["APT28 did it."] * 5, dictionary, pattern)
    assert short["reason"] == "too_short"


def test_mask_removes_actor_names_and_cluster_codes_but_not_software():
    _, pattern = _dictionary()
    text = "Fighting Ursa (UNC1234) dropped Sednit; Lazarus and Storm-0001 too."
    assert fresh.mask_text(text, pattern) == (
        "the adversary (the adversary) dropped Sednit; the adversary and the adversary too."
    )


def test_dates_from_urls_and_pages():
    assert fresh.url_date(
        "https://www.microsoft.com/en-us/security/blog/2026/07/14/x/", "ymd"
    ) == date(2026, 7, 14)
    assert fresh.url_date("https://www.trendmicro.com/en_us/research/26/h/x.html", "trend") == (
        date(2026, 8, 1)
    )
    assert fresh.url_date(
        "https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-194a", "cisa"
    ) == date(2026, 7, 13)
    html = '<meta property="article:published_time" content="2026-09-02T10:00:00+00:00">'
    assert fresh.page_date(html) == date(2026, 9, 2)
    assert fresh.page_date('{"datePublished": "2026-08-30"}') == date(2026, 8, 30)
    assert fresh.normalise_url("https://www.Example.com/a/b/?x=1#y") == "example.com/a/b"


def _product(report_id, truth, leading, top5, stage="unresolved_activity_cluster", arm="known"):
    return {
        "report_id": report_id,
        "group_id": truth,
        "leading": leading,
        "top3": top5[:3],
        "topk": [{"group_id": g} for g in top5],
        "stage": stage,
        "named_cluster": [],
        "accept": [],
    }


def test_combination_and_scores():
    product = {
        "known": [
            _product("r1", "G1", "G1", ["G1", "G2"], stage="known_group_hypothesis"),
            _product("r2", "G2", "G3", ["G3", "G2", "G4"]),
            _product("r3", "G5", "G6", ["G6", "G7"]),
        ],
        "open_set": [_product("o1", None, "G8", ["G8"], stage="known_group_hypothesis")],
    }
    llm = {
        "rows": [
            {
                "report_id": "r1",
                "arm": "known",
                "group_id": "G1",
                "candidates": ["G1"],
                "decision": "name",
            },
            {
                "report_id": "r2",
                "arm": "known",
                "group_id": "G2",
                "candidates": ["G2", "G3"],
                "decision": "name",
            },
            {
                "report_id": "r3",
                "arm": "known",
                "group_id": "G5",
                "candidates": ["G5"],
                "decision": "name",
            },
            {
                "report_id": "o1",
                "arm": "open_set",
                "group_id": None,
                "candidates": ["G9"],
                "decision": "not_in_attack",
            },
        ]
    }
    result = h2h.score(product, llm)
    arms = result["arms"]
    assert arms["product"]["first_right"] == 1 and arms["product"]["names_open_set"] == 1
    assert arms["llm"]["first_right"] == 3 and arms["llm"]["names_right"] == 3
    # r2: the LLM's G2 is in the product's top five -> first choice G2; r3: G5 is not -> G6
    assert arms["combination"]["first_right"] == 2
    # a name only where both agree and the LLM names (r1)
    assert arms["combination"]["names_known"] == 1 and arms["combination"]["names_right"] == 1
    assert arms["combination"]["names_open_set"] == 0
    assert result["pairs"]["llm_vs_product"]["mcnemar_first"]["only_first"] == 2
