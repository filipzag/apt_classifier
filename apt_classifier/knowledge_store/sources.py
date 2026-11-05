"""The report sources the knowledge store reads (docs/KNOWLEDGE_STORE_DESIGN.md).

Every source was approved by the product owner for the fresh report sets and can be listed with a
plain request; a new source needs the owner's approval, and analysts cannot add URLs (the Live
Intel rule, docs 24.1). The fresh-set builder (``scripts/prepare_fresh_reports.py``) lists the same
sources, so the test sets and the store date their posts the same way.
"""

from __future__ import annotations

from typing import Any

# Each source: how its posts are listed. ``sitemaps``: sitemap URLs (an index is followed into
# children matching ``child``); ``feeds``: RSS/Atom URLs, ``{page}`` paged until older than the
# listing start; ``post``: a post URL pattern; ``dated``: how a URL encodes its date, if it does.
SOURCES: dict[str, dict[str, Any]] = {
    "unit42": {
        "sitemaps": ["https://unit42.paloaltonetworks.com/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://unit42\.paloaltonetworks\.com/[a-z0-9-]+/$",
    },
    "microsoft": {
        "feeds": ["https://www.microsoft.com/en-us/security/blog/feed/?paged={page}"],
        "post": r"^https://www\.microsoft\.com/en-us/security/blog/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/$",
        "dated": "ymd",
    },
    "eset": {
        "feeds": ["https://www.welivesecurity.com/en/rss/feed/"],
        "post": r"^https://www\.welivesecurity\.com/en/[a-z0-9-]+/[a-z0-9-]+/$",
    },
    "securelist": {
        "sitemaps": ["https://securelist.com/sitemap.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://securelist\.com/[a-z0-9-]+/\d+/$",
    },
    "checkpoint": {
        "sitemaps": ["https://research.checkpoint.com/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://research\.checkpoint\.com/20\d{2}/[a-z0-9-]+/$",
    },
    "sentinelone": {
        "sitemaps": ["https://www.sentinelone.com/sitemap-labs.xml"],
        "post": r"^https://www\.sentinelone\.com/labs/[a-z0-9-]+/$",
    },
    "talos": {
        "sitemaps": ["https://blog.talosintelligence.com/sitemap-posts.xml"],
        "post": r"^https://blog\.talosintelligence\.com/[a-z0-9-]+/$",
    },
    "zscaler": {
        "sitemaps": ["https://www.zscaler.com/sitemap.xml"],
        "post": r"^https://www\.zscaler\.com/blogs/security-research/[a-z0-9-]+$",
    },
    "proofpoint": {
        "sitemaps": [f"https://www.proofpoint.com/us/sitemap.xml?page={n}" for n in range(1, 6)],
        "post": r"^https://www\.proofpoint\.com/us/blog/threat-insight/[a-z0-9-]+$",
    },
    "google": {
        "sitemaps": ["https://cloud.google.com/transform/sitemapsummary/cloudblog"],
        "child": r"/cloudblog/en/2026-(?:0[2-9]|1[0-2])-",
        "feeds": ["https://feeds.feedburner.com/threatintelligence/pvexyqv7v0v"],
        "post": r"^https://cloud\.google\.com/blog/topics/threat-intelligence/[a-z0-9-]+/?$",
    },
    "trendmicro": {
        "sitemaps": ["https://www.trendmicro.com/en_us/sitemap.xml"],
        "post": r"^https://www\.trendmicro\.com/en_us/research/\d{2}/[a-l]/[a-z0-9-]+\.html$",
        "dated": "trend",
    },
    "symantec": {
        "sitemaps": ["https://www.security.com/sitemap.xml"],
        "post": r"^https://www\.security\.com/threat-intelligence/[a-z0-9-]+$",
    },
    "recordedfuture": {
        "sitemaps": ["https://www.recordedfuture.com/sitemap-main.xml"],
        "feeds": ["https://www.recordedfuture.com/feed"],
        "post": r"^https://www\.recordedfuture\.com/research/[a-z0-9-]+$",
    },
    "volexity": {
        "sitemaps": ["https://www.volexity.com/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://www\.volexity\.com/blog/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/$",
        "dated": "ymd",
    },
    # added 2026-10-29 (amendment): the approved sources that can be listed with dates
    "asec": {
        "feeds": ["https://asec.ahnlab.com/en/feed/?paged={page}"],
        "post": r"^https://asec\.ahnlab\.com/en/\d+/?$",
    },
    "seqrite": {
        "feeds": ["https://www.seqrite.com/blog/feed/?paged={page}"],
        "post": r"^https://www\.seqrite\.com/blog/[a-z0-9-]+/?$",
    },
    "bitdefender": {
        "feeds": ["https://www.bitdefender.com/blog/api/rss/labs/"],
        "post": r"^https://www\.bitdefender\.com/en-us/blog/labs/[a-z0-9-]+/?$",
    },
    "silentpush": {
        "sitemaps": ["https://www.silentpush.com/sitemap_index.xml"],
        "child": r"post|blog",
        "post": r"^https://www\.silentpush\.com/blog/[a-z0-9-]+/?$",
    },
    "fortinet": {
        "sitemaps": ["https://www.fortinet.com/sitemap-index.xml"],
        "child": r"blog",
        "post": r"^https://www\.fortinet\.com/blog/threat-research/[a-z0-9-]+$",
    },
    "arcticwolf": {
        "sitemaps": ["https://arcticwolf.com/sitemap_index.xml"],
        "child": r"post|blog|resource",
        "post": r"^https://arcticwolf\.com/resources/blog/[a-z0-9-]+/$",
    },
    "rapid7": {
        "sitemaps": ["https://www.rapid7.com/sitemap.xml"],
        "child": r"blog",
        "post": r"^https://www\.rapid7\.com/blog/post/[a-z0-9-]+/?$",
    },
    "hunt": {
        "sitemaps": ["https://hunt.io/sitemap.xml"],
        "post": r"^https://hunt\.io/blog/[a-z0-9-]+/?$",
    },
    "cisa": {
        "sitemaps": ["https://www.cisa.gov/default/sitemap.xml"],
        "post": r"^https://www\.cisa\.gov/news-events/cybersecurity-advisories/aa\d{2}-\d{3}[a-z]$",
        "dated": "cisa",
    },
    # added 2026-11-02 for the second set (prereg llm_confidence_cutoff_2026-11-02): the
    # approved sources whose posts can be listed with a plain request
    "groupib": {
        "sitemaps": ["https://www.group-ib.com/sitemap.xml"],
        "child": r"blog|post",
        # translated copies ("...-es/") are left out: the duplicate check compares statements
        "post": r"^https://www\.group-ib\.com/blog/(?![a-z0-9-]+-(?:es|pt|br|fr|de|ar|tr|id)/$)"
        r"[a-z0-9-]+/$",
    },
    "strikeready": {
        "sitemaps": ["https://www.strikeready.com/sitemap.xml"],
        "post": r"^https://www\.strikeready\.com/blog/[^/?#]+$",
    },
    "kaspersky_ics": {
        "sitemaps": ["https://ics-cert.kaspersky.com/sitemap.xml"],
        "child": r"publication",
        "post": r"^https://ics-cert\.kaspersky\.com/publications/(?:reports|blog)/\d{4}/\d{2}/\d{2}/"
        r"[a-z0-9-]+/$",
        "dated": "ymd",
    },
    "jpcert": {
        "sitemaps": ["https://blogs.jpcert.or.jp/sitemap.xml"],
        "post": r"^https://blogs\.jpcert\.or\.jp/en/\d{4}/\d{2}/[A-Za-z0-9_.-]+\.html$",
    },
    "gendigital": {
        "sitemaps": ["https://www.gendigital.com/blogentries/sitemap.xml"],
        "post": r"^https://www\.gendigital\.com/blog/insights/research/[a-z0-9-]+$",
    },
    "resecurity": {
        "sitemaps": ["https://www.resecurity.com/sitemap.xml"],
        "post": r"^https://www\.resecurity\.com/blog/article/[a-z0-9-]+$",
    },
    "intrinsec": {
        "sitemaps": ["https://www.intrinsec.com/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://www\.intrinsec\.com/[a-z0-9_-]+/$",
    },
    "infoblox": {
        "sitemaps": ["https://www.infoblox.com/blog/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://www\.infoblox\.com/blog/threat-intelligence/(?:[a-z0-9-]+/)?[a-z0-9-]+/$",
    },
    "threatbook": {
        "sitemaps": ["https://threatbook.io/sitemap.xml"],
        "post": r"^https://threatbook\.io/blog/[a-z0-9-]+$",
    },
    "sekoia": {
        # the feed reaches back to June 2026; the blog has no sitemap
        "feeds": ["https://blog.sekoia.io/feed/?paged={page}"],
        "post": r"^https://www\.sekoia\.com/blog/[a-z0-9-]+/?$",
    },
}

# The store reads further back than the fresh sets did (from 2022): Google's monthly child
# sitemaps for every year from 2022 (the fresh builder lists 2026 only).
STORE_OVERRIDES: dict[str, dict[str, Any]] = {
    "google": {"child": r"/cloudblog/en/20(?:2[2-9])-"},
}

# The store names itself (the fresh-set builder sent a browser string); robots.txt rules for this
# token, or for "*", apply.
ROBOTS_AGENT = "VestigeKnowledgeStore"
USER_AGENT = (
    f"{ROBOTS_AGENT}/1.0 (local threat-report reader; one request per second per site; "
    "honours robots.txt)"
)
# sources run by one publisher count once when facts are corroborated
PUBLISHERS = {"securelist": "kaspersky", "kaspersky_ics": "kaspersky"}
# sites kept out while their benchmark sets are in use (Elastic Security Labs: docs 21.3.4c)
EXCLUDED_SITES = ("elastic.co",)


def store_sources(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """The sources the store lists, with the store's overrides applied."""
    return {
        name: {**config, **STORE_OVERRIDES.get(name, {})}
        for name, config in SOURCES.items()
        if not names or name in names
    }
