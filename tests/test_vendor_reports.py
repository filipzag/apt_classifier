"""Vendor-report development set (``prepare_vendor_reports.py``): the fixed selection rule, page
reading, site chrome, the mention check and the fallback order. Synthetic pages and knowledge
base only; nothing is downloaded."""

from __future__ import annotations

import re
from types import SimpleNamespace

from scripts import prepare_vendor_reports as vendor

_ANONYMIZER = SimpleNamespace(
    heading_is_attribution=lambda heading: "attribution" in heading.lower()
)


def _group(name, aliases=(), techniques=50):
    return SimpleNamespace(
        name=name, aliases=tuple(aliases), technique_ids=tuple(f"T{i}" for i in range(techniques))
    )


def _kb():
    return SimpleNamespace(
        version="19.2",
        groups={
            "G0094": _group("Kimsuky", ["Springtail"], 130),
            "G0007": _group("APT28", ["Fancy Bear"], 93),
            "G0016": _group("APT29", [], 80),  # CTI-HAL: never selected
            "G0200": _group("Small One", [], 12),
            "G0201": _group("Tiny", [], 3),  # below the technique bar
        },
    )


def _set(group_id, refs):
    return {
        "type": "intrusion-set",
        "id": f"intrusion-set--{group_id}",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": group_id},
            *refs,
        ],
    }


def _ref(name, url, year, month="May", day=1):
    return {"source_name": name, "url": url, "description": f"V. ({year}, {month} {day}). T."}


def test_select_follows_the_fixed_rule():
    shared = _ref("Shared", "https://unit42.paloaltonetworks.com/shared/", 2025)
    bundle = {
        "objects": [
            _set(
                "G0094",
                [
                    _ref("Old", "https://www.security.com/threat-intelligence/old", 2020),
                    _ref("New", "https://www.security.com/threat-intelligence/new", 2024),
                    _ref("Pdf", "https://www.security.com/threat-intelligence/x.pdf", 2026),
                    _ref("News", "https://arstechnica.com/story", 2026),
                    shared,
                ],
            ),
            _set("G0007", [_ref("A", "https://www.volexity.com/blog/a", 2024), shared]),
            _set("G0016", [_ref("C", "https://www.volexity.com/blog/c", 2024)]),
            _set("G0200", [_ref("S", "https://blog.talosintelligence.com/s", 2021)]),
            _set("G0201", [_ref("T", "https://blog.talosintelligence.com/t", 2021)]),
        ]
    }
    labels = vendor.select(bundle, _kb())
    assert [entry["group_id"] for entry in labels["reports"]] == ["G0094", "G0007", "G0200"]
    kimsuky = labels["reports"][0]
    # newest first; PDFs, news and a page ATT&CK cites for another group too are left out
    assert [c["source_name"] for c in kimsuky["candidates"]] == ["New", "Old"]
    assert kimsuky["candidates"][0]["date"] == "2024-05-01"
    assert kimsuky["stratum"] == "large" and labels["reports"][2]["stratum"] == "small"
    assert labels["eligible_groups"] == 3


def _page(paragraphs, *, extra=""):
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    return (
        "<html><head><title>Report</title><script>var x = 1;</script></head><body>"
        "<nav><ul><li>Home page of the vendor site</li></ul></nav>"
        f"<article><h1>Report</h1>{body}"
        "<p><a href='/a'>A long list of links to</a> <a href='/b'>other posts here</a></p>"
        "<p>Figure 3 shows the infection chain in detail.</p>"
        "<pre>powershell -enc AAAA BBBB CCCC</pre>"
        "<h2>Indicators of Compromise</h2><p>The hashes listed below are all SHA256 values.</p>"
        "<h2>Related posts</h2><p>Another story about some other threat group.</p>"
        f"{extra}</article><footer><p>Copyright notice for the whole site here.</p></footer>"
        "</body></html>"
    )


def test_parse_page_reads_the_article_prose_only():
    page = vendor.parse_page(
        _page(["Springtail sent spearphishing emails to the targets.", "It used a backdoor."])
    )
    texts = [line["text"] for line in page["lines"]]
    assert texts == ["Springtail sent spearphishing emails to the targets.", "It used a backdoor."]
    assert page["dropped"]["link_lines"] == 1 and page["dropped"]["figure_lines"] == 1
    assert page["dropped"]["section:Indicators of Compromise"] == 1
    assert page["dropped"]["section:Related posts"] == 1
    assert page["lines"][0]["path"] == ("Report",)


def test_site_chrome_is_dropped_and_archives_count_as_their_origin():
    assert vendor.site_of("https://web.archive.org/web/2023/https://research.nccgroup.com/x") == (
        "nccgroup.com"
    )
    assert vendor.site_of("https://us-cert.cisa.gov/ncas/alerts/x") == "cisa.gov"
    banner = "We use cookies to give you the best experience on this site."
    first = {**vendor.parse_page(_page(["Springtail used a new backdoor.", banner])), "site": "s"}
    second = {**vendor.parse_page(_page(["Another actor used a loader.", banner])), "site": "s"}
    chrome = vendor.repeated_lines({("G1", 0): first, ("G2", 0): second})
    assert chrome == {"s": frozenset({banner})}
    statements = vendor.statements_of(first, _ANONYMIZER, chrome["s"])
    assert [statement["text"] for statement in statements] == ["Springtail used a new backdoor."]


def _sentences(name, count, other=None, other_count=0):
    lines = [
        f"In step {index} {name} ran command number {index} on the host." for index in range(count)
    ]
    lines += [f"Analysts also saw {other} activity number {index}." for index in range(other_count)]
    return lines


def test_mention_check_needs_the_most_mentions_and_enough_sentences():
    patterns = vendor.group_name_patterns(_kb())
    assert isinstance(patterns["G0094"], re.Pattern)

    def statements(lines):
        return [{"text": line} for line in lines]

    passed, mentions = vendor.mention_check(
        statements(_sentences("Springtail", 16)), "G0094", patterns
    )
    assert passed and mentions == {"G0094": 16}
    # a tie with another group fails
    tie = _sentences("Kimsuky", 8) + _sentences("Fancy Bear", 8)
    assert not vendor.mention_check(statements(tie), "G0094", patterns)[0]
    # too few sentences fails; aliases are case-sensitive proper nouns
    assert not vendor.mention_check(statements(_sentences("Kimsuky", 14)), "G0094", patterns)[0]
    assert not vendor.mention_check(statements(_sentences("kimsuky", 16)), "G0094", patterns)[0]


def test_choose_skips_pages_read_as_not_a_report(tmp_path):
    patterns = vendor.group_name_patterns(_kb())
    good = {**vendor.parse_page(_page(_sentences("Kimsuky", 16))), "site": "a"}
    pages = {("G0094", 0): good, ("G0094", 1): {**good, "site": "b"}}
    entry = {"group_id": "G0094", "candidates": [{"url": "u0"}, {"url": "u1"}]}
    choice, checks = vendor.choose(entry, pages, {}, anonymizer=_ANONYMIZER, patterns=patterns)
    assert choice["index"] == 0 and checks[0]["passed"]
    entry["not_a_report"] = [0]
    choice, checks = vendor.choose(entry, pages, {}, anonymizer=_ANONYMIZER, patterns=patterns)
    assert choice["index"] == 1 and checks[0] == {"index": 0, "not_a_report": True}
    entry["not_a_report"] = [0, 1]
    choice, _ = vendor.choose(entry, pages, {}, anonymizer=_ANONYMIZER, patterns=patterns)
    assert choice is None


def test_transient_fetch_failures_are_retried():
    assert vendor._transient({"status": "http_429"})
    assert vendor._transient({"status": "http_503"})
    assert vendor._transient({"status": "fetch_error", "error": "ReadTimeout: timed out"})
    assert not vendor._transient({"status": "http_404"})
    assert not vendor._transient({"status": "fetched"})
