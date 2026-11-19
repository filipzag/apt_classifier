"""Fresh-report head-to-head set (``prepare_fresh_reports.py``) and its scoring
(``fresh_h2h_score.py``): actor names, the registered label rule, masking, dates and the
registered combination. Synthetic knowledge base, MISP list and artifacts; nothing is downloaded."""

from __future__ import annotations

import json
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
            "G0080": group("Cobalt Group", ["Cobalt Group", "Cobalt"]),
            "G0006": group("APT1", ["APT1", "Comment Group"]),
            "G0047": group("Gamaredon Group", ["Gamaredon Group"]),
        },
        software={
            "S0001": SimpleNamespace(name="Sednit", aliases=("Sofacy",)),
            "S0154": SimpleNamespace(name="Cobalt Strike", aliases=()),
        },
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


def _dictionary(tmp_path=None, monkeypatch=None):
    if monkeypatch is not None:
        words = tmp_path / "words"
        words.write_text("comment\nLazarus\nbeijing\n", encoding="utf-8")
        monkeypatch.setattr(fresh, "WORDS", words)
    dictionary = fresh.actor_dictionary(_kb(), _misp())
    return dictionary, fresh.name_pattern(list(dictionary["surface"]))


def test_actor_dictionary_maps_vendor_names_and_keeps_attack_names(tmp_path, monkeypatch):
    surface = _dictionary(tmp_path, monkeypatch)[0]["surface"]
    assert surface["Lazarus"] == "G0032"  # short form; a proper noun, not an ordinary word
    assert surface["Gamaredon"] == "G0047"
    assert "Comment" not in surface  # short form that is an ordinary word
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


def test_mask_removes_actor_names_and_cluster_codes_but_not_software(tmp_path, monkeypatch):
    dictionary, pattern = _dictionary(tmp_path, monkeypatch)
    software = dictionary["software_pattern"]
    text = "Fighting Ursa (UNC1234) dropped Sednit and Cobalt Strike; Lazarus and Storm-0001 too."
    assert fresh.mask_text(text, pattern, software) == (
        "the adversary (the adversary) dropped Sednit and Cobalt Strike; the adversary and "
        "the adversary too."
    )
    # a name joined to a lower-case word by a hyphen is masked; a code-like suffix is not a name
    assert fresh.mask_text(
        "Laundry Bear-linked hosts; a handoff-to-Gamaredon; Lazarus-2", pattern, software
    ) == ("the adversary-linked hosts; a handoff-to-the adversary; Lazarus-2")
    # "Cobalt" inside "Cobalt Strike" is the software, not the Cobalt Group
    counts = fresh.actor_mentions(["Cobalt Strike beacons", "Cobalt used it"], dictionary, pattern)
    assert counts["G0080"] == 1


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


def test_page_dates_prefer_the_research_date_and_reject_batch_stamps():
    html = (
        '<meta name="published-date" content="2020-07-28">'
        '<script>{"datePublished": "2026-06-29"}</script>'
    )
    assert fresh.page_date(html) == date(2020, 7, 28)
    visible = "<html><nav>Menu</nav><span>20 Oct 2017</span> ... related: 22 Oct 2026</html>"
    assert fresh.page_date(visible, "symantec") == date(2017, 10, 20)
    pages = {f"rapid7-{i}": {"date": date(2026, 6, 16)} for i in range(5)}
    pages["rapid7-x"] = {"date": date(2026, 5, 6)}
    batches = fresh.batch_dates(pages, {key: "rapid7" for key in pages})
    assert batches == {"rapid7": {date(2026, 6, 16)}}
    sitemap = {"source": "rapid7", "url_date": None, "method": "sitemap", "listed": "2026-06-16"}
    assert fresh.publication_date(sitemap, date(2026, 6, 16), batches) is None
    assert fresh.publication_date(sitemap, date(2026, 5, 6), batches) == date(2026, 5, 6)
    feed = {**sitemap, "method": "feed", "listed": "2026-07-23"}
    assert fresh.publication_date(feed, date(2026, 6, 16), batches) == date(2026, 7, 23)
    assert fresh.LIVING_TITLE_RE.search("Threat Brief: Credential Attacks (Updated August 18)")
    assert not fresh.LIVING_TITLE_RE.search("UAT-7810 continues building ORB networks")


def _answer(report_id, truth, names, probability, decision="name", arm="known"):
    raw = [{"group_id": "G9999", "group_name": name, "probability": probability} for name in names]
    return {
        "report_id": report_id,
        "arm": arm,
        "group_id": truth,
        "candidates": [],
        "raw_candidates": raw,
        "decision": decision,
    }


def test_name_matching_confidence_cutoff_and_switch():
    names = h2h.group_names(_kb().groups)
    assert names["apt28"] == "G0007" and names["fancybear"] == "G0007"
    product = {
        "known": [
            _product("r1", "G0007", "G0047", ["G0047", "G0007"]),
            _product("r2", "G0032", "G0032", ["G0032"], stage="known_group_hypothesis"),
            _product("r3", "G0129", "G0129", ["G0129"]),
            _product("r4", "G0047", "G0047", ["G0047"]),
        ],
        "open_set": [_product("o1", None, "G0006", ["G0006"])],
    }
    llm = {
        "rows": [
            # the right name with an invented id: matched by its name
            _answer("r1", "G0007", ["APT 28"], 0.99),
            _answer("r2", "G0032", ["Lazarus Group"], 0.98),
            # a name ATT&CK does not list is no ATT&CK name
            _answer("r3", "G0129", ["Void Blizzard", "Mustang Panda"], 0.99),
            _answer("r4", "G0047", ["APT1"], 0.3, decision="open"),
            _answer("o1", None, ["APT1"], 0.95, arm="open_set"),
        ]
    }
    result = h2h.score(product, llm, names=names, min_probability=0.99, switch_probability=0.5)
    arms = result["arms"]
    assert arms["llm"]["names_known"] == 2 and arms["llm"]["names_right"] == 2
    assert arms["llm"]["names_outside_attack_known"] == 1 and arms["llm"]["names_open_set"] == 1
    # only r1 is named at 0.99; r3's 0.99 is a name outside ATT&CK
    assert arms["llm_confident"]["names_given"] == 1 and arms["llm_confident"]["names_right"] == 1
    # r2: the product's leader agrees with the model's name below the cut-off
    assert arms["agreed_or_confident"]["names_right"] == 2
    # first choice: the model where it states >= 0.5 (r1, r2, r3 -> Mustang Panda), else the
    # product (r4)
    assert arms["switch"]["first_right"] == 4 and arms["llm"]["first_right"] == 3
    assert arms["switch"]["names_open_set"] == 0
    assert h2h.exact_interval(7, 10) == [0.3475, 0.9333]
    assert h2h.exact_interval(17, 17)[0] == 0.8049 and h2h.exact_interval(0, 0) is None


def test_second_set_leaves_out_every_url_the_first_listing_considered(tmp_path):
    listed = tmp_path / "candidates.json"
    listed.write_text(
        json.dumps({"candidates": [{"url": "https://www.Example.com/a/b/?x=1"}]}), encoding="utf-8"
    )
    other = tmp_path / "reports.jsonl"
    other.write_text(json.dumps({"url": "https://example.com/c"}) + "\n", encoding="utf-8")
    bundle = {"objects": [{"external_references": [{"url": "https://example.com/d"}]}]}
    urls = fresh.excluded_url_set(bundle, [other], [listed])
    assert urls == {"example.com/a/b", "example.com/c", "example.com/d"}


def test_decisive_set_from_the_store_keeps_its_dates_and_leaves_out_what_leaks(
    tmp_path, monkeypatch
):
    from apt_classifier.knowledge_store.fetch import FetchResult
    from apt_classifier.knowledge_store.store import KnowledgeStore

    def html(title, published, sentences):
        body = "".join(f"<p>{sentence}</p>" for sentence in sentences)
        return (
            f'<html><head><title>{title}</title><meta property="article:published_time" '
            f'content="{published}T10:00:00+00:00"></head><body><article>{body}</article>'
            "</body></html>"
        )

    report = [
        "APT28 deployed a new backdoor against foreign ministries.",
        *(f"APT28 used implant number {n} to collect documents." for n in range(3)),
        *(f"The operators staged archive number {n} on a remote server." for n in range(3, 18)),
    ]
    filler = [f"The intruders renamed tool number {n} before running it." for n in range(8)]
    posts = {
        "v.test/report": ("APT28 targets ministries", "2025-03-10", report, None),
        "v.test/early": ("Early", "2024-12-01", filler, None),
        "v.test/copy": ("Copy", "2025-02-01", [*filler, "One more line is here."], None),
        "v.test/shown": ("Shown", "2025-04-01", report[1:], None),
        "v.test/edited": ("Edited", "2025-05-01", report[2:], "2026-06-01"),
        "v.test/cited": ("Cited", "2025-06-01", report[3:], None),
        "v.test/late": ("Late", "2026-04-01", report[4:], None),
    }
    store = KnowledgeStore(tmp_path / "store")
    store.add_posts(
        {"key": key, "source": "v", "url": f"https://{key}", "lastmod": lastmod}
        for key, (_title, _day, _lines, lastmod) in posts.items()
    )
    for key, (title, day, lines, _lastmod) in posts.items():
        page = html(title, day, lines).encode()
        store.record_fetch(
            key,
            "v",
            FetchResult("fetched", http_status=200, final_url=f"https://{key}", content=page),
            at="2026-11-03T10:00:00+00:00",
        )
    store.parse()
    store.finalise(attack_urls=["https://v.test/cited"])
    store.close()

    window = (date(2025, 1, 1), date(2026, 2, 28))
    listed = fresh.store_candidates(
        tmp_path / "store", set(), window=window, shown={"v.test/shown"}
    )
    assert listed["per_source"]["v"] == {
        "candidate": 1,
        "copy_of_earlier_post": 1,
        "shown_before_registration": 1,
        "available_after_window": 1,
        "cited_or_in_another_set": 1,
    }
    (candidate,) = listed["candidates"]
    assert candidate["store_key"] == "v.test/report" and candidate["raw"].endswith(".html.gz")
    assert (candidate["published"], candidate["available_from"]) == ("2025-03-10", "2025-03-10")

    class _Anonymizer:  # the strict anonymiser reads a real knowledge base
        def __init__(self, kb, level):
            pass

        def heading_is_attribution(self, heading):
            return False

    words = tmp_path / "words"
    words.write_text("comment\n", encoding="utf-8")
    monkeypatch.setattr(fresh, "WORDS", words)
    monkeypatch.setattr(fresh, "ReportAnonymizer", _Anonymizer)
    output = tmp_path / "decisive"
    kb = _kb()
    kb.version = "test"
    manifest = fresh.build(listed, output, kb=kb, misp=_misp(), window=window)
    assert (manifest["known_reports"], manifest["open_set_reports"]) == (1, 0)
    row = json.loads((output / "reports.jsonl").read_text(encoding="utf-8"))
    assert (row["group_id"], row["published"], row["available_from"]) == (
        "G0007",
        "2025-03-10",
        "2025-03-10",
    )
    assert row["store_key"] == "v.test/report"
    assert row["title"] == "the adversary targets ministries"
    assert not any("APT28" in statement["text"] for statement in row["statements"])
