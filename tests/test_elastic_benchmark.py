"""Elastic Security Labs real-report benchmark: anonymizer levels, feed preparation, loading.

No network, Chroma or model: a synthetic knowledge base, an in-test RSS feed and a tiny STIX
bundle with one revoked technique.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software, Technique


@pytest.fixture
def kb() -> AttackKnowledgeBase:
    groups = {
        "G0032": Group(id="G0032", name="Lazarus Group", aliases=["HIDDEN COBRA"]),
        "G0082": Group(id="G0082", name="APT38", aliases=["Bluenoroff"]),
        "G0050": Group(id="G0050", name="APT32", aliases=["OceanLotus", "SeaLotus"]),
        "G0065": Group(id="G0065", name="Leviathan", aliases=["APT40"]),
        "G1006": Group(id="G1006", name="Earth Lusca", aliases=["CHROMIUM"]),
        "G1040": Group(id="G1040", name="Play"),
        "G0047": Group(id="G0047", name="Gamaredon Group", aliases=["Primitive Bear"]),
    }
    software = {
        "S0625": Software(id="S0625", name="Cuba", software_type="malware"),
        "S0158": Software(id="S0158", name="PHOREAL", software_type="malware", group_ids=["G0050"]),
        "S0029": Software(id="S0029", name="PsExec", software_type="tool"),
        "S0147": Software(
            id="S0147", name="Pteranodon", software_type="malware", group_ids=["G0047"]
        ),
    }
    techniques = {
        technique_id: Technique(id=technique_id, name=technique_id)
        for technique_id in ("T1059", "T1059.001", "T1547", "T1547.001", "T1566")
    }
    groups["G0050"].software_ids = ["S0158"]
    groups["G0047"].software_ids = ["S0147"]
    return AttackKnowledgeBase(version="t", groups=groups, software=software, techniques=techniques)


def _folds(texts: list[str], group_id: str | None = "G0050"):
    from scripts.attribution_benchmark import Fold

    return [
        Fold(
            group_id=group_id,
            held_out_techniques=frozenset(),
            observations=tuple(texts),
            held_out_texts=frozenset(),
            report_id="r1",
        )
    ]


def test_strict_masks_origin_compounds_places_clusters_and_object_ids(kb):
    from scripts.attribution_benchmark import anonymize_report_folds

    texts = [
        "The DPRK-aligned actor used a Vietnam–based server and an Iran-Linked proxy.",
        "DPRK's operators and DPRK’s tools targeted Ukraninan officials.",
        "The Democratic People's Republic of Korea runs Group G0050 tools such as S0158.",
        "REF 8747 and REF7001-related infrastructure; TA0001 stays; TA428 tooling.",
        "A Russian speaker targeted a Malaysian-focused government in Hanoi.",
        "The coverage map focuses on the Iraqi-Kurdistan region.",
        "Lazarus used state-sponsored nation-state tooling against Southeast Asian banks.",
        "OceanLotus staged files in Southeast Asia.",
    ]
    (fold,), records, totals = anonymize_report_folds(_folds(texts), kb, level="strict")
    assert fold.observations == (
        "The actor used a server and a proxy.",
        "an undisclosed country's operators and an undisclosed country’s tools targeted officials.",
        "The undisclosed country runs Group tools such as.",
        "the adversary and the adversary-related infrastructure; stays; the adversary tooling.",
        "A speaker targeted an undisclosed-focused government in an undisclosed city.",
        "The coverage map focuses on the undisclosed region.",
        "the adversary used tooling against banks.",
        "the adversary staged files in an undisclosed region.",
    )
    strict = records[0]["strict"]
    assert strict["replacements"]["origin_compounds"] == 3
    assert strict["replacements"]["cluster_names"] == 3
    assert strict["replacements"]["attack_ids"] == 3
    assert strict["replacements"]["short_forms"] == 1
    assert strict["replacements"]["sponsor_terms"] == 2
    assert strict["software_lost_beyond_names"] == []
    assert strict["residuals"] == []
    assert totals["strict"]["level"] == "strict"
    # the names pass is recorded exactly as at the names level
    assert set(records[0]) == {
        "statements_changed",
        "replacements",
        "names",
        "own_group_named",
        "software_matches_removed",
        "software_ids_removed",
        "strict",
    }
    assert records[0]["names"] == {"oceanlotus": 1}


def test_strict_keeps_codenames_software_words_and_locale_codes(kb):
    from scripts.attribution_benchmark import anonymize_report_folds

    texts = [
        "CUBA ransomware and the KREMLIN toolkit; Cuba ransomware encrypts files.",
        "The Chromium extension API was abused while CHROMIUM was active.",
        "Users play a video set to en-US with US-ASCII encoding.",
    ]
    (fold,), records, _ = anonymize_report_folds(_folds(texts), kb, level="strict")
    assert fold.observations == (
        "CUBA ransomware and the KREMLIN toolkit; Cuba ransomware encrypts files.",
        "The Chromium extension API was abused while the adversary was active.",
        "Users play a video set to en-US with US-ASCII encoding.",
    )
    strict = records[0]["strict"]
    assert strict["codenames_kept"] == {"CUBA": 1, "KREMLIN": 1}
    assert strict["protected_by_software"] == {"Cuba": 1}
    # at the names level the ordinary word is replaced, as it always was
    (names_fold,), names_records, _ = anonymize_report_folds(_folds(texts[1:2]), kb)
    assert names_fold.observations == (
        "the adversary extension API was abused while the adversary was active.",
    )
    assert "strict" not in names_records[0]


def test_tradecraft_masks_malware_names_but_keeps_tools(kb):
    from scripts.attribution_benchmark import anonymize_report_folds

    texts = ["the PHOREAL backdoor ran PsExec on a Chinese-speaking operator's host."]
    (fold,), records, totals = anonymize_report_folds(_folds(texts), kb, level="tradecraft")
    assert fold.observations == ("the malware backdoor ran PsExec on an operator's host.",)
    assert records[0]["strict"]["malware_masked"] == {"S0158": 1}
    assert records[0]["strict"]["tools_kept"] == {"S0029": 1}
    assert records[0]["strict"]["software_lost_beyond_names"] == ["S0158"]
    assert totals["strict"]["malware_masked"] == {"S0158": 1}


def test_attribution_flags_need_an_actor_and_a_cue_and_spread_to_the_paragraph(kb):
    from scripts.report_anonymizer import ReportAnonymizer

    anonymizer = ReportAnonymizer(kb, "strict")
    texts = [
        "Throughout our analysis, we observed infrastructure known to belong to APT32.",
        "APT32 is a well-known group associated with several campaigns.",
        "A loader documented by a vendor has also been tied to the same domains.",
        "The loader wrote a scheduled task.",  # same paragraph, no cue: dropped with it
        "This named pipe doesn't have any security attributes.",
        "The DPRK-aligned actor used a VPS.",  # an origin compound is masked, not a cue
        "We assess with moderate confidence that the macro ran twice.",  # no actor reference
        "Victims were in banks.",
    ]
    flags = anonymizer.attribution_flags(
        texts,
        paragraphs=[1, 1, 1, 1, 2, 3, 4, 5],
        section_flags=[False] * 7 + [True],
    )
    assert flags == [
        "sentence",
        "sentence",
        "paragraph",
        "paragraph",
        None,
        None,
        None,
        "section",
    ]
    assert anonymizer.heading_is_attribution("Possible APT40/Leviathan connection")
    assert anonymizer.heading_is_attribution("DPRK attribution: why this links to the campaign")
    assert not anonymizer.heading_is_attribution("Campaign analysis")
    assert not anonymizer.heading_is_attribution("Execution flow")


def test_residual_audit_lists_names_the_patterns_cannot_touch(kb):
    from scripts.report_anonymizer import ReportAnonymizer

    anonymizer = ReportAnonymizer(kb, "strict")
    found = anonymizer.residuals(["see the manage-apt32 write-up", "the adversary used PsExec"])
    assert [item["term"] for item in found] == ["APT32"]
    assert anonymizer.residuals(["play the Chromium build", "a Cuba sample"]) == []


def _feed(items: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<item><title>{slug}</title><link>https://www.elastic.co/security-labs/threat-command/"
        f"{slug}</link><pubDate>Thu, 02 Mar 2023 00:00:00 GMT</pubDate>"
        f"<content:encoded><![CDATA[{html}]]></content:encoded></item>"
        for slug, html in items
    )
    return (
        '<?xml version="1.0"?><rss xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        f"<channel>{body}</channel></rss>"
    )


LABELLED_POST = """
<h2>Preamble</h2>
<p>The OceanLotus operators delivered a <code>tscon32.dll</code> loader through a lure.</p>
<p>| Path | Hash |
| --- | --- |
| C:\\Temp\\a.dll | abc |</p>
<pre><code>rule PHOREAL { strings: $a = "x" condition: $a }</code></pre>
<p>Figure 1 - The loader chain observed in the intrusion</p>
<p>![Execution chain](https://example.org/a.png)</p>
<h2>Execution</h2>
<h3>The undetected version of PHOREAL</h3>
<p>The PHOREAL backdoor created a service and contacted its C2 over ICMP.</p>
</p>loose = decode(b"x")<p>
<h2>Campaign associations</h2>
<p>The tooling overlaps with earlier public reporting by several vendors.</p>
<h2>Possible APT32 connection</h2>
<p>The strings match an earlier advisory sample closely.</p>
<h2>Observed adversary tactics and techniques</h2>
<h4>Techniques</h4>
<ul><li><a href="https://attack.mitre.org/techniques/T1060/">Registry Run Keys</a></li>
<li><a href="https://attack.mitre.org/techniques/T1059/001/">PowerShell</a></li></ul>
<h2>References</h2>
<p>The full advisory describes the operators in detail.</p>
"""
OPEN_POST = "<h2>Overview</h2><p>The REF7707 operators moved laterally with PsExec over SMB.</p>"


def _bundle(path: Path) -> Path:
    def pattern(stix_id: str, external_id: str) -> dict:
        return {
            "type": "attack-pattern",
            "id": stix_id,
            "external_references": [{"source_name": "mitre-attack", "external_id": external_id}],
        }

    path.write_text(
        json.dumps(
            {
                "objects": [
                    pattern("attack-pattern--old", "T1060"),
                    pattern("attack-pattern--new", "T1547.001"),
                    {
                        "type": "relationship",
                        "id": "relationship--1",
                        "relationship_type": "revoked-by",
                        "source_ref": "attack-pattern--old",
                        "target_ref": "attack-pattern--new",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _labels(path: Path, **overrides) -> Path:
    labels = {
        "reports": [
            {
                "slug": "phoreal-post",
                "group_id": "G0050",
                "accept": [],
                "attribution_sections": ["Campaign associations"],
                "activity_year": 2022,
            }
        ],
        "open_set": [
            {
                "slug": "ref7707-post",
                "cluster": "REF7707",
                "kind": "apt_like",
                "attribution_sections": [],
            }
        ],
    }
    labels.update(overrides)
    path.write_text(json.dumps(labels), encoding="utf-8")
    return path


def test_prepare_reads_prose_like_the_product_and_keeps_labels_separate(kb, tmp_path: Path):
    from scripts.prepare_elastic_reports import prepare

    feed = tmp_path / "feed.xml"
    feed.write_text(
        _feed(
            [("phoreal-post", LABELLED_POST), ("ref7707-post", OPEN_POST), ("other", "<p>x</p>")]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "elastic"
    manifest = prepare(
        feed,
        _labels(tmp_path / "labels.json"),
        output,
        kb=kb,
        raw_bundle=_bundle(tmp_path / "b.json"),
    )
    (row,) = [json.loads(line) for line in (output / "reports.jsonl").read_text().splitlines()]
    texts = [statement["text"] for statement in row["statements"]]
    assert texts == [
        "The OceanLotus operators delivered a tscon32.dll loader through a lure.",
        "The PHOREAL backdoor created a service and contacted its C2 over ICMP.",
        "The tooling overlaps with earlier public reporting by several vendors.",
        "The strings match an earlier advisory sample closely.",
    ]
    sections = {statement["text"][:12]: statement for statement in row["statements"]}
    assert sections["The PHOREAL "]["section"] == "Execution > The undetected version of PHOREAL"
    assert sections["The tooling "]["section_attribution"] is True  # curated heading
    assert sections["The strings "]["section_attribution"] is True  # heading rule
    assert sections["The OceanLot"]["section_attribution"] is False
    # revoked T1060 resolves to T1547.001; links in dropped sections still label the report
    assert row["technique_ids"] == ["T1059.001", "T1547.001"]
    assert row["group_id"] == "G0050" and row["report_id"] == "elastic-phoreal-post"
    (open_row,) = [
        json.loads(line) for line in (output / "reports_open_set.jsonl").read_text().splitlines()
    ]
    assert open_row["group_id"] is None and open_row["kind"] == "apt_like"
    stats = manifest["per_report"]["elastic-phoreal-post"]
    assert stats["dropped_markup"] == {
        "figure_lines": 1,
        "image_lines": 1,
        "loose_lines": 1,
        "pre_blocks": 1,
        "table_lines": 3,
    }
    assert set(stats["dropped_sections"]) == {
        "Observed adversary tactics and techniques",
        "References",
    }
    assert stats["truth_exclusive_software"] == ["S0158"]
    assert stats["leak_audit"]["statements_equal_to_attack_procedures"] == 0
    assert [drop["reason"] for drop in stats["strict_attribution_drops"]] == ["section", "section"]
    assert manifest["reports"] == 1 and manifest["open_set"] == 1


def test_prepare_refuses_unknown_slugs_groups_and_headings(kb, tmp_path: Path):
    from scripts.prepare_elastic_reports import prepare

    feed = tmp_path / "feed.xml"
    feed.write_text(_feed([("phoreal-post", LABELLED_POST), ("ref7707-post", OPEN_POST)]))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="not in the feed"):
        prepare(feed, _labels(tmp_path / "a.json", open_set=[{"slug": "nope"}]), output, kb=kb)
    bad_group = [{"slug": "phoreal-post", "group_id": "G9999", "attribution_sections": []}]
    with pytest.raises(ValueError, match="not in the KB"):
        prepare(feed, _labels(tmp_path / "b.json", reports=bad_group), output, kb=kb)
    bad_heading = [{"slug": "phoreal-post", "group_id": "G0050", "attribution_sections": ["Nope"]}]
    with pytest.raises(ValueError, match="not found"):
        prepare(feed, _labels(tmp_path / "c.json", reports=bad_heading), output, kb=kb)


def test_load_report_folds_open_set_accept_strict_drops_and_sampling(kb, tmp_path: Path):
    from scripts.attribution_benchmark import load_report_folds
    from scripts.report_anonymizer import ReportAnonymizer

    def statement(text: str, paragraph: int, section: bool = False) -> dict:
        return {
            "text": text,
            "technique_ids": [],
            "paragraph": paragraph,
            "section_attribution": section,
        }

    rows = [
        {
            "report_id": "elastic-a",
            "group_id": "G0032",
            "accept": ["G0082"],
            "technique_ids": ["T1059"],
            "statements": [
                statement("We assess with high confidence that this is Lazarus Group activity.", 1),
                statement("The loader wrote a launch agent for persistence.", 2),
                statement("The tooling overlaps with earlier reporting.", 3, section=True),
                statement("The implant beaconed every sixty seconds.", 4),
                statement("The implant collected browser data.", 5),
            ],
        },
        {
            "report_id": "elastic-b",
            "group_id": None,
            "kind": "commodity",
            "statements": [statement("The miner disabled the endpoint agent first.", 1)],
        },
    ]
    path = tmp_path / "reports.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    plain = load_report_folds(path, max_observations=None)
    assert [fold.group_id for fold in plain] == ["G0032", None]
    assert plain[0].accept == frozenset({"G0082"}) and plain[1].kind == "commodity"
    assert plain[0].held_out_techniques == frozenset({"T1059"})
    assert len(plain[0].observations) == 5 and plain[0].attribution_dropped == ()
    strict = load_report_folds(path, max_observations=2, anonymizer=ReportAnonymizer(kb))
    assert strict[0].observations == (
        "The loader wrote a launch agent for persistence.",
        "The implant beaconed every sixty seconds.",
    )
    assert dict(strict[0].attribution_dropped) == {"section": 1, "sentence": 1}
    sampled = load_report_folds(path, max_observations=2, sample_seed=3)
    again = load_report_folds(path, max_observations=2, sample_seed=3)
    assert sampled[0].observations == again[0].observations  # seeded, in report order
    order = [plain[0].observations.index(text) for text in sampled[0].observations]
    assert order == sorted(order)


def test_summary_adds_wilson_lenient_and_group_macro_and_open_set_metrics():
    from scripts.attribution_benchmark import open_set_metrics, summarize

    named = "known_group_hypothesis"

    def row(group, rank, stage, report, lenient=None, accept=()):
        return {
            "group_id": group,
            "report_id": report,
            "rank": rank,
            "lenient_rank": lenient if lenient is not None else rank,
            "accept": list(accept),
            "top3": [group] if rank == 1 else ["G0082", group],
            "stage": stage,
            "technique_recall": None if report == "r3" else 0.5,
            "observations": 5,
        }

    known = [
        row("G0032", 2, named, "r1", lenient=1, accept=["G0082"]),
        row("G0050", 1, named, "r2"),
        row("G0050", 3, "known_group_shortlist", "r3"),
    ]
    metrics = summarize(known, [])
    assert metrics["top1_count"] == 1
    low, high = metrics["top1_wilson95"]
    assert 0.0 < low < 1 / 3 < high < 1.0
    assert metrics["lenient_top1"] == pytest.approx(2 / 3)
    assert metrics["lenient_named_correct"] == 2
    assert metrics["group_macro_top1"] == pytest.approx(0.25)  # G0032 0/1, G0050 1/2
    assert metrics["mean_technique_recall"] == pytest.approx(0.5)
    assert metrics["technique_recall_reports"] == 2
    open_rows = [
        {"stage": named, "leading": "G0032", "top3": ["G0032"], "kind": "apt_like"},
        {
            "stage": "known_group_shortlist",
            "leading": "G0050",
            "top3": ["G0050"],
            "kind": "commodity",
        },
    ]
    result = open_set_metrics(open_rows, known)
    assert result["named"] == 1 and result["named_groups"] == {"G0032": 1}
    assert result["by_kind"] == {
        "apt_like": {"reports": 1, "named": 1},
        "commodity": {"reports": 1, "named": 0},
    }
    # one wrong strict name on the labelled set plus one open-set name over three names
    assert result["combined_named_risk"] == pytest.approx(2 / 3)


def test_collector_keeps_elastic_slots_and_warns_when_the_headline_is_not_strict(tmp_path: Path):
    from scripts.collect_verification import compose

    def artifact(level: str | None) -> dict:
        return {
            "parameters": {"report_set": "elastic", "anonymization": level},
            "metrics": {"top1": 0.5, "top1_wilson95": [0.2, 0.8], "open_set": {"named": 0}},
        }

    common = {
        "public": None,
        "curated": None,
        "evidenceforge": {},
        "public_path": tmp_path / "p.json",
        "curated_path": tmp_path / "c.json",
        "gates": None,
    }
    report = compose(
        **common,
        elastic_attribution=artifact("strict"),
        elastic_attribution_path=tmp_path / "e.json",
        elastic_attribution_as_reported=artifact(None),
        elastic_attribution_as_reported_path=tmp_path / "a.json",
    )
    assert report["elastic_attribution"]["top1_wilson95"] == [0.2, 0.8]
    assert report["elastic_attribution"]["open_set"] == {"named": 0}
    assert report["elastic_attribution_as_reported"]["parameters"]["anonymization"] is None
    assert "warnings" not in report
    report = compose(
        **common,
        elastic_attribution=artifact("names"),
        elastic_attribution_path=tmp_path / "e.json",
    )
    assert any("elastic_attribution" in warning for warning in report["warnings"])
