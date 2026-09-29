from datetime import UTC, datetime
from types import SimpleNamespace

import httpx

from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Software
from apt_classifier.evidence.models import EvidenceLedger, Observation, Provenance, StrategicContext
from apt_classifier.threat_intel import (
    CISA_ADVISORIES_URL,
    CISA_KEV_URL,
    RANSOMWARE_LIVE_URL,
    PublicThreatIntelClient,
    ThreatIntelSnapshot,
    match_threat_intel_to_case,
    parse_cisa_advisories,
    parse_cisa_kev,
    parse_ransomware_claims,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_public_feed_parsers_normalize_dates_provenance_and_verification():
    kev = parse_cisa_kev(
        {
            "vulnerabilities": [
                {
                    "cveID": "CVE-2026-1234",
                    "vendorProject": "Contoso",
                    "product": "Gateway",
                    "vulnerabilityName": "Contoso Gateway Remote Code Execution",
                    "dateAdded": "2026-08-30",
                    "shortDescription": "Actively exploited remote code execution.",
                    "requiredAction": "Apply the update.",
                    "knownRansomwareCampaignUse": "Known",
                },
                {
                    "cveID": "CVE-2020-0001",
                    "vendorProject": "OldCo",
                    "product": "Old Product",
                    "vulnerabilityName": "Old issue",
                    "dateAdded": "2020-01-01",
                },
            ]
        },
        now=NOW,
        days=30,
    )
    rss = parse_cisa_advisories(
        """<?xml version="1.0"?><rss><channel><item>
        <title>State actors exploit Contoso gateways</title>
        <link>https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-001a</link>
        <description><![CDATA[Campaign activity involving CVE-2026-1234.]]></description>
        <pubDate>Mon, 31 Aug 2026 12:00:00 +0000</pubDate>
        </item></channel></rss>""",
        now=NOW,
        days=30,
    )
    claims = parse_ransomware_claims(
        [
            {
                "victim": "Example Manufacturer",
                "group": "ExampleLocker",
                "activity": "Manufacturing",
                "country": "US",
                "domain": "example.invalid",
                "description": "The operator claims data theft.",
                "discovered": "2026-08-31T10:00:00+00:00",
                "url": "https://www.ransomware.live/id/example",
            }
        ],
        now=NOW,
        days=30,
    )

    assert [item.cves for item in kev] == [("CVE-2026-1234",)]
    assert kev[0].verification == "authoritative"
    assert "known-ransomware-use" in kev[0].tags
    assert rss[0].kind == "campaign_advisory"
    assert rss[0].cves == ("CVE-2026-1234",)
    assert claims[0].kind == "claimed_breach"
    assert claims[0].verification == "unverified_extortion_claim"
    assert claims[0].actors == ("ExampleLocker",)
    assert claims[0].industries == ("Manufacturing",)


def test_live_intel_matching_requires_explicit_case_overlap():
    kb = AttackKnowledgeBase(
        groups={
            "G1": Group(id="G1", name="Alpha Group", aliases=["Alpha"]),
            "G2": Group(id="G2", name="Unrelated Group"),
        },
        software={
            "S1": Software(id="S1", name="RedKit", aliases=["Red Kit"]),
        },
    )
    observation = Observation(
        text="RedKit exploited CVE-2026-1234 on the gateway.",
        provenance=Provenance(source="alert"),
    )
    state = SimpleNamespace(
        ledger=EvidenceLedger(
            observations=[observation],
            strategic_context=StrategicContext(
                target_geographies=["US"], target_industries=["Manufacturing"]
            ),
        ),
        context=SimpleNamespace(software_ids={"S1"}),
        fusion=SimpleNamespace(
            entries=(
                SimpleNamespace(group_id="G1", rank=1),
                SimpleNamespace(group_id="G2", rank=2),
            )
        ),
    )
    matching = parse_cisa_advisories(
        """<rss><channel><item><title>Alpha campaign uses RedKit</title>
        <link>https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-002a</link>
        <description>CVE-2026-1234 exploitation affecting manufacturers.</description>
        <pubDate>Mon, 31 Aug 2026 12:00:00 +0000</pubDate></item></channel></rss>""",
        now=NOW,
        days=30,
    )[0]
    unrelated = parse_cisa_advisories(
        """<rss><channel><item><title>Unrelated product update</title>
        <link>https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-003a</link>
        <description>No shared case signals.</description>
        <pubDate>Mon, 31 Aug 2026 12:00:00 +0000</pubDate></item></channel></rss>""",
        now=NOW,
        days=30,
    )[0]
    snapshot = ThreatIntelSnapshot(
        fetched_at=NOW.isoformat(),
        items=(matching, unrelated),
        sources=(),
    )

    matches = match_threat_intel_to_case(kb, state, snapshot)

    assert len(matches) == 1
    assert matches[0].item_id == matching.id
    assert matches[0].group_ids == ("G1",)
    assert matches[0].observation_ids == (observation.id,)
    assert any("CVE overlap" in reason for reason in matches[0].reasons)
    assert any("Ranked actor reference" in reason for reason in matches[0].reasons)
    assert any("Named tool" in reason for reason in matches[0].reasons)


def test_ransomware_claim_context_match_remains_non_voting_and_labeled():
    kb = AttackKnowledgeBase(groups={"G1": Group(id="G1", name="Alpha")})
    state = SimpleNamespace(
        ledger=EvidenceLedger(
            strategic_context=StrategicContext(
                target_geographies=["US"], target_industries=["Manufacturing"]
            )
        ),
        context=SimpleNamespace(software_ids=set()),
        fusion=SimpleNamespace(entries=(SimpleNamespace(group_id="G1", rank=1),)),
    )
    claim = parse_ransomware_claims(
        [
            {
                "victim": "Claimed Victim",
                "group": "UnknownLocker",
                "activity": "Manufacturing",
                "country": "US",
                "discovered": "2026-08-31T10:00:00+00:00",
                "url": "https://www.ransomware.live/id/claim",
            }
        ],
        now=NOW,
        days=30,
    )[0]
    snapshot = ThreatIntelSnapshot(NOW.isoformat(), (claim,), ())

    matches = match_threat_intel_to_case(kb, state, snapshot)

    assert matches[0].verification == "unverified_extortion_claim"
    assert matches[0].group_ids == ()
    assert matches[0].score < 2.0


def test_client_keeps_healthy_sources_when_one_feed_fails():
    def handler(request):
        if str(request.url) == CISA_KEV_URL:
            return httpx.Response(
                200,
                json={
                    "vulnerabilities": [
                        {
                            "cveID": "CVE-2026-1234",
                            "vulnerabilityName": "Recent issue",
                            "dateAdded": "2026-08-31",
                        }
                    ]
                },
            )
        if str(request.url) == CISA_ADVISORIES_URL:
            return httpx.Response(503, text="unavailable")
        if str(request.url) == RANSOMWARE_LIVE_URL:
            return httpx.Response(
                200,
                json=[
                    {
                        "victim": "Claimed Victim",
                        "group": "ExampleLocker",
                        "discovered": "2026-08-31T10:00:00+00:00",
                    }
                ],
            )
        raise AssertionError(request.url)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PublicThreatIntelClient(client=http_client)
    snapshot = client.fetch(now=NOW, days=30, include_ransomware_claims=True)

    assert {item.source for item in snapshot.items} == {"CISA KEV", "ransomware.live"}
    status = {source.source: source for source in snapshot.sources}
    assert status["CISA KEV"].ok
    assert not status["CISA Alerts & Advisories"].ok
    assert status["ransomware.live"].ok
