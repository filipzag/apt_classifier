"""Fresh real reports: vendor research posts published after the frontier model's training cutoff.

A head-to-head test set (prereg ``fresh_reports_head_to_head_2026-10-28.json``): the product and a
frontier LLM read the same anonymised statements of reports no model and no ATT&CK release has
seen. Window: posts published from ``WINDOW_START`` (after Claude Opus 5.5's June 2026 cutoff)
to ``WINDOW_END``; the URLs MITRE ATT&CK cites are left out, so ATT&CK's own texts and profiles
do not contain the report either.

Steps, each re-runnable:

``--list``      candidate post URLs per source (sitemaps with lastmod, feeds, dated URLs), listed
                leniently from ``LIST_FROM``; URLs ATT&CK cites and URLs of the other report sets
                are dropped -> ``candidates.json``
``--aliases``   the MISP galaxy threat-actor list (vendor names for actors) -> ``misp_threat_actor.json``
``--download``  each candidate page, one request per second -> ``raw/`` and ``download_log.json``
(build)         parse every page (``prepare_vendor_reports.parse_page``), date it from the page,
                keep the window, label it by the registered rule, mask every actor name, cap
                each group, drop duplicates -> ``reports.jsonl`` (ATT&CK groups),
                ``open_set.jsonl`` (actors ATT&CK does not document), ``MANIFEST.json``

The labelling rule (``label_report``) reads the post as reported, before any masking: actor names
are ATT&CK group names and aliases plus MISP threat-actor names (mapped to an ATT&CK group by an
ATT&CK reference or an exact shared name; ambiguous names are left out; names that are also ATT&CK
software names never count, so a malware write-up is not a group report), counted
case-sensitively as whole words over the title and the statements.

* known: the leading ATT&CK group has at least ``MIN_MENTIONS`` mentions, at least
  ``DOMINANCE`` times the next ATT&CK group's, more than any actor ATT&CK does not document,
  and is named in the title or the first ``LEAD_STATEMENTS`` statements;
* open set: an actor ATT&CK does not document (a MISP actor without an ATT&CK group, or a vendor
  cluster code such as UNC1234 or Storm-1234 that is no ATT&CK alias) leads the same way and no
  ATT&CK group is named more than ``OPEN_MAX_ATTACK_MENTIONS`` time;
* anything else (several actors, overlaps, no actor) is left out and logged.

Every actor name the rule knows and every vendor cluster code is then replaced by "the
adversary" in the statements (ATT&CK software names are kept, as in the product's anonymiser),
so both systems read the same masked text; the benchmark's own anonymiser runs on top. At most
``GROUP_CAP`` reports per group (newest first) and one copy of a cross-posted report are kept.

Outputs under ``data/benchmarks/fresh/`` (gitignored; the text belongs to each vendor, CISA
advisories are US government works; local evaluation only, do not redistribute).
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from scripts.prepare_vendor_reports import (  # noqa: E402
    USER_AGENT,
    _transient,
    fetch,
    parse_page,
    repeated_lines,
    statements_of,
)
from scripts.report_anonymizer import ReportAnonymizer  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "data" / "benchmarks" / "fresh"
LICENCE = (
    "(c) each vendor (CISA advisories: US government works); local evaluation only, "
    "do not redistribute"
)
MISP_URL = "https://raw.githubusercontent.com/MISP/misp-galaxy/main/clusters/threat-actor.json"
WINDOW_START = date(2026, 7, 1)
WINDOW_END = date(2026, 10, 27)
LIST_FROM = date(2026, 6, 20)
MIN_STATEMENTS = 15
MIN_MENTIONS = 3
DOMINANCE = 2.0
LEAD_STATEMENTS = 10
OPEN_MAX_ATTACK_MENTIONS = 1
GROUP_CAP = 8
DUPLICATE_SHARE = 0.5
MASK = "the adversary"
# vendor cluster codes (an actor ATT&CK does not document unless the code is an ATT&CK alias)
CLUSTER_CODE_RE = re.compile(
    r"(?<![\w-])(?:UNC\d{3,5}|Storm-\d{4}|DEV-\d{4}|TA\d{3,4}|TAG-\d{2,4}|UAC-\d{4}|UAT-\d{4}"
    r"|CL-(?:STA|CRI|UNK)-\d{4}|TGR-(?:STA|CRI|UNK)-\d{4}|Earth [A-Z][a-z]+|Water [A-Z][a-z]+"
    r"|Void [A-Z][a-z]+|SHADOW-[A-Z]+-\d+|REF\d{4}|STAC\d{3,5}|TAC\d{3,5})(?![\w-])"
)
_TREND_MONTH = {letter: index for index, letter in enumerate("abcdefghijkl", 1)}
# MISP relations that tie an unmapped actor to an ATT&CK group's activity
RELATION_TYPES = frozenset(
    {"similar", "overlaps", "linked-to", "part-of", "related-to", "associated-with"}
)

# Each source: how its posts are listed. ``sitemaps``: sitemap URLs (an index is followed into
# children matching ``child``); ``feeds``: RSS/Atom URLs, ``{page}`` paged until older than
# LIST_FROM; ``post``: a post URL pattern; ``dated``: how a URL encodes its date, if it does.
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
        "feeds": ["https://feeds.feedburner.com/threatintelligence/pvexyqv7v0v"],
        "post": r"^https://cloud\.google\.com/blog/topics/threat-intelligence/[a-z0-9-]+",
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
        "feeds": ["https://www.recordedfuture.com/feed"],
        "post": r"^https://www\.recordedfuture\.com/research/[a-z0-9-]+$",
    },
    "volexity": {
        "sitemaps": ["https://www.volexity.com/sitemap_index.xml"],
        "child": r"/post-sitemap\d*\.xml$",
        "post": r"^https://www\.volexity\.com/blog/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/$",
        "dated": "ymd",
    },
    "cisa": {
        "sitemaps": ["https://www.cisa.gov/default/sitemap.xml"],
        "post": r"^https://www\.cisa\.gov/news-events/cybersecurity-advisories/aa\d{2}-\d{3}[a-z]$",
        "dated": "cisa",
    },
}


# --------------------------------------------------------------------------- listing


def _get(url: str, *, timeout: float = 40.0) -> str | None:
    import httpx

    try:
        response = httpx.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True
        )
    except httpx.HTTPError:
        return None
    return response.text if response.status_code == 200 else None


def normalise_url(url: str) -> str:
    """Scheme-less, lower-case host, no query or fragment, no trailing slash (for matching)."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    return f"{host}{parsed.path}".rstrip("/").lower()


def url_date(url: str, kind: str | None) -> date | None:
    """The publication date a URL encodes (``ymd``: /YYYY/MM/DD/, ``trend``: /research/YY/m/,
    first of the month, ``cisa``: aaYY-DDD day of the year)."""
    try:
        if kind == "ymd":
            match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
            return date(int(match[1]), int(match[2]), int(match[3])) if match else None
        if kind == "trend":
            match = re.search(r"/research/(\d{2})/([a-l])/", url)
            return date(2000 + int(match[1]), _TREND_MONTH[match[2]], 1) if match else None
        if kind == "cisa":
            match = re.search(r"/aa(\d{2})-(\d{3})[a-z]$", url)
            if match:
                return date.fromordinal(
                    date(2000 + int(match[1]), 1, 1).toordinal() + int(match[2]) - 1
                )
    except ValueError:
        return None
    return None


def _iso_date(text: str) -> date | None:
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", text or "")
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def sitemap_entries(url: str, child: str | None, *, seen: set[str] | None = None):
    """(loc, lastmod) of every URL in a sitemap, following an index into matching children."""
    seen = seen if seen is not None else set()
    if url in seen:
        return
    seen.add(url)
    text = _get(url, timeout=90.0)
    if text is None:
        print(f"  sitemap unavailable: {url}", flush=True)
        return
    blocks = re.findall(r"<(sitemap|url)>(.*?)</\1>", text, flags=re.S)
    for kind, block in blocks:
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block, flags=re.S)
        if not loc:
            continue
        lastmod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block, flags=re.S)
        location = loc.group(1).replace("&amp;", "&")
        if kind == "sitemap":
            if child is None or re.search(child, location):
                modified = _iso_date(lastmod.group(1)) if lastmod else None
                if modified is None or modified >= LIST_FROM:
                    yield from sitemap_entries(location, child, seen=seen)
            continue
        yield location, (_iso_date(lastmod.group(1)) if lastmod else None)


def feed_entries(url: str):
    """(link, published) of every item, paging ``{page}`` until older than LIST_FROM."""
    pages = range(1, 40) if "{page}" in url else range(1, 2)
    seen: set[str] = set()
    for page in pages:
        text = _get(url.format(page=page))
        if text is None:
            return
        items = re.findall(r"<(item|entry)\b.*?</\1>", text, flags=re.S)
        blocks = re.findall(r"<(?:item|entry)\b(.*?)</(?:item|entry)>", text, flags=re.S)
        if not items:
            return
        oldest = None
        fresh = 0
        for block in blocks:
            link = re.search(r"<link>\s*(.*?)\s*</link>", block, flags=re.S) or re.search(
                r"<link[^>]*href=\"([^\"]+)\"", block
            )
            if not link:
                continue
            location = link.group(1).replace("&amp;", "&").strip()
            published = None
            stamp = re.search(r"<pubDate>(.*?)</pubDate>", block, flags=re.S)
            if stamp:
                try:
                    published = email.utils.parsedate_to_datetime(stamp.group(1).strip()).date()
                except (TypeError, ValueError):
                    published = None
            else:
                stamp = re.search(r"<(?:published|updated)>(.*?)</", block, flags=re.S)
                published = _iso_date(stamp.group(1)) if stamp else None
            if location in seen:
                continue
            seen.add(location)
            fresh += 1
            if published is not None:
                oldest = published if oldest is None else min(oldest, published)
            yield location, published
        if fresh == 0 or (oldest is not None and oldest < LIST_FROM):
            return


def list_candidates(excluded_urls: set[str]) -> dict[str, Any]:
    """Every source's post URLs listed (leniently) from LIST_FROM; the window is applied to the
    date read from each page at build time."""
    candidates: dict[str, dict[str, Any]] = {}
    per_source: dict[str, Counter[str]] = {}
    for source, config in SOURCES.items():
        counts: Counter[str] = Counter()
        entries = []
        for sitemap in config.get("sitemaps", ()):
            entries.extend(
                (loc, modified, "sitemap")
                for loc, modified in sitemap_entries(sitemap, config.get("child"))
            )
        for feed in config.get("feeds", ()):
            entries.extend((link, published, "feed") for link, published in feed_entries(feed))
        for location, listed, method in entries:
            location = location.split("#")[0]
            if not re.search(config["post"], location):
                counts["not_a_post_url"] += 1
                continue
            encoded = url_date(location, config.get("dated"))
            when = encoded or listed
            if when is not None and when < LIST_FROM:
                counts["listed_before_window"] += 1
                continue
            key = normalise_url(location)
            if key in excluded_urls:
                counts["cited_or_in_another_set"] += 1
                continue
            if key in candidates:
                counts["listed_twice"] += 1
                continue
            counts["candidate"] += 1
            candidates[key] = {
                "source": source,
                "url": location,
                "listed": listed.isoformat() if listed else None,
                "url_date": encoded.isoformat() if encoded else None,
                "method": method,
            }
        per_source[source] = counts
        print(f"{source:15s} {dict(counts)}", flush=True)
    rows = sorted(candidates.values(), key=lambda row: (row["source"], row["url"]))
    for index, row in enumerate(rows):
        row["id"] = f"{row['source']}-{hashlib.sha1(row['url'].encode()).hexdigest()[:12]}"
        row["index"] = index
    return {
        "listed_at": datetime.now(UTC).isoformat(),
        "list_from": LIST_FROM.isoformat(),
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "per_source": {source: dict(counts) for source, counts in per_source.items()},
        "candidates": rows,
    }


def excluded_url_set(bundle: dict[str, Any], other_sets: list[Path]) -> set[str]:
    """URLs ATT&CK cites anywhere, plus the URLs of the other report sets."""
    urls = {
        normalise_url(reference["url"])
        for item in bundle.get("objects", [])
        for reference in item.get("external_references", []) or []
        if reference.get("url")
    }
    for path in other_sets:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                url = json.loads(line).get("url")
                if url:
                    urls.add(normalise_url(url))
    return urls


# --------------------------------------------------------------------------- page dates


def page_date(html: str) -> date | None:
    """The publication date a page states (Open Graph / article meta, JSON-LD, ``<time>``)."""
    patterns = (
        r"<meta[^>]+(?:property|name)=\"(?:article:published_time|og:published_time"
        r"|datePublished|publish-date|publication_date|pubdate|date|dc\.date)\"[^>]+"
        r"content=\"([^\"]+)\"",
        r"<meta[^>]+content=\"([^\"]+)\"[^>]+(?:property|name)=\"(?:article:published_time"
        r"|og:published_time|datePublished|publish-date|publication_date|pubdate)\"",
        r"\"datePublished\"\s*:\s*\"([^\"]+)\"",
        r"<time[^>]+datetime=\"([^\"]+)\"",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.I):
            found = _iso_date(match.group(1))
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------- actor names


def _attack_id(item: dict[str, Any]) -> str | None:
    return next(
        (
            reference["external_id"]
            for reference in item.get("external_references", []) or []
            if reference.get("source_name") == "mitre-attack"
        ),
        None,
    )


def software_names(kb: AttackKnowledgeBase) -> set[str]:
    names: set[str] = set()
    for software in kb.software.values():
        names.update(name for name in (software.name, *software.aliases) if name)
    return {name.lower() for name in names}


def _name_forms(name: str) -> list[str]:
    """A group name and its short form ("Lazarus Group" -> "Lazarus", as vendors write it)."""
    forms = [name]
    if name.endswith(" Group") and len(name) - len(" Group") > 3:
        forms.append(name[: -len(" Group")])
    return forms


def _misp_names(cluster: dict[str, Any]) -> list[str]:
    names = [cluster.get("value") or "", *((cluster.get("meta") or {}).get("synonyms") or [])]
    return [name.strip() for name in names if isinstance(name, str) and name.strip()]


def actor_dictionary(kb: AttackKnowledgeBase, misp: dict[str, Any] | None) -> dict[str, Any]:
    """Surface names -> actor.

    ATT&CK group names and aliases (and the short form of a "... Group" name) name their group
    (``G####``) and always win a name. A MISP threat actor maps to an ATT&CK group by its ATT&CK
    reference, else by its own name, else by its synonyms when they point at one group; a MISP
    actor whose names point at several groups is left out; an unmapped one is ``misp:<name>``,
    or ``related:<G####>`` when MISP links it (similar, overlaps, ...) to a mapped actor. A name
    two actors share, a name that is also an ATT&CK software name and a name of three characters
    or fewer are left out."""
    software = software_names(kb)
    attack_names: dict[str, set[str]] = {}
    for group_id, group in kb.groups.items():
        for name in (group.name, *group.aliases):
            for form in _name_forms(name) if name else ():
                attack_names.setdefault(form.lower(), set()).add(group_id)
    surface: dict[str, str] = {}
    origin: dict[str, str] = {}
    clashes: set[str] = set()
    for group_id, group in kb.groups.items():
        for name in (group.name, *group.aliases):
            for form in _name_forms(name) if name else ():
                if len(form) <= 3 or form.lower() in software:
                    continue
                if len(attack_names[form.lower()]) > 1:
                    clashes.add(form.lower())
                    continue
                surface[form] = group_id
                origin[form] = "attack"
    clusters = [cluster for cluster in (misp or {}).get("values", []) if cluster.get("uuid")]
    mapping: dict[str, str | None] = {}
    stats: Counter[str] = Counter()
    for cluster in clusters:
        names = _misp_names(cluster)
        refs = " ".join((cluster.get("meta") or {}).get("refs") or [])
        by_ref = set(re.findall(r"attack\.mitre\.org/groups/(G\d{4})", refs)) & set(kb.groups)
        own = attack_names.get((cluster.get("value") or "").strip().lower(), set())
        by_name = {
            group_id
            for name in names
            if len(attack_names.get(name.lower(), ())) == 1
            for group_id in attack_names[name.lower()]
        }
        if len(by_ref) == 1:
            actor: str | None = next(iter(by_ref))
        elif len(own) == 1:
            actor = next(iter(own))
        elif len(by_name) == 1:
            actor = next(iter(by_name))
        elif by_ref or by_name:
            actor = None
        else:
            actor = f"misp:{names[0]}" if names else None
        stats[
            "ambiguous"
            if actor is None
            else ("unmapped" if actor.startswith("misp:") else "mapped")
        ] += 1
        mapping[cluster["uuid"]] = actor
    links: dict[str, set[str]] = {}
    for cluster in clusters:
        for relation in cluster.get("related") or ():
            if relation.get("type") in RELATION_TYPES and relation.get("dest-uuid") in mapping:
                links.setdefault(cluster["uuid"], set()).add(relation["dest-uuid"])
                links.setdefault(relation["dest-uuid"], set()).add(cluster["uuid"])
    for uuid, actor in list(mapping.items()):
        if actor is None or not actor.startswith("misp:"):
            continue
        linked = sorted(
            {
                other
                for neighbour in links.get(uuid, ())
                if (other := mapping.get(neighbour)) and is_attack(other)
            }
        )
        if linked:
            mapping[uuid] = "related:" + ",".join(linked)
            stats["related"] += 1
    misp_surface: dict[str, set[str]] = {}
    for cluster in clusters:
        actor = mapping.get(cluster["uuid"])
        if actor is None:
            continue
        for name in _misp_names(cluster):
            if len(name) <= 3 or name.lower() in software or name.lower() in attack_names:
                continue
            misp_surface.setdefault(name, set()).add(actor)
    for name, actors in misp_surface.items():
        if len(actors) > 1:
            clashes.add(name.lower())
            continue
        surface[name] = next(iter(actors))
        origin[name] = "misp"
    return {
        "surface": surface,
        "origin": origin,
        "stats": {
            "names": len(surface),
            "misp_mapped_clusters": stats["mapped"],
            "misp_unmapped_clusters": stats["unmapped"] - stats["related"],
            "misp_related_clusters": stats["related"],
            "misp_ambiguous_clusters": stats["ambiguous"],
            "clashing_names": len(clashes),
        },
    }


def name_pattern(names: list[str]) -> re.Pattern[str] | None:
    names = sorted(set(names), key=lambda name: (-len(name), name))
    if not names:
        return None
    return re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(name) for name in names) + r")(?![\w-])"
    )


def actor_mentions(
    texts: list[str], dictionary: dict[str, Any], pattern: re.Pattern[str] | None
) -> Counter[str]:
    """Mentions per actor: dictionary names (case-sensitive, whole words) and cluster codes that
    are no dictionary name (``code:<code>``)."""
    surface = dictionary["surface"]
    counts: Counter[str] = Counter()
    for text in texts:
        spans: list[tuple[int, int]] = []
        if pattern is not None:
            for match in pattern.finditer(text):
                counts[surface[match.group(0)]] += 1
                spans.append(match.span())
        for match in CLUSTER_CODE_RE.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            counts[f"code:{match.group(0)}"] += 1
    return counts


def is_attack(actor: str) -> bool:
    return bool(re.fullmatch(r"G\d{4}", actor))


def label_report(
    title: str,
    statements: list[str],
    dictionary: dict[str, Any],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    """The registered label: ``known`` (an ATT&CK group), ``open`` (an actor ATT&CK does not
    document) or ``excluded``, with the reason and the counts."""
    counts = actor_mentions([title, *statements], dictionary, pattern)
    lead = set(actor_mentions([title, *statements[:LEAD_STATEMENTS]], dictionary, pattern))
    attack = sorted(
        ((count, actor) for actor, count in counts.items() if is_attack(actor)), reverse=True
    )
    other = sorted(
        ((count, actor) for actor, count in counts.items() if not is_attack(actor)), reverse=True
    )
    top = dict(sorted(counts.items(), key=lambda item: -item[1])[:6])
    if len(statements) < MIN_STATEMENTS:
        return {"label": "excluded", "reason": "too_short", "mentions": top}
    if attack:
        count, actor = attack[0]
        second = attack[1][0] if len(attack) > 1 else 0
        best_other = other[0][0] if other else 0
        if (
            count >= MIN_MENTIONS
            and count >= DOMINANCE * second
            and count > best_other
            and actor in lead
        ):
            return {"label": "known", "group_id": actor, "mentions": top}
    if other:
        count, actor = other[0]
        second = other[1][0] if len(other) > 1 else 0
        attack_max = attack[0][0] if attack else 0
        if (
            count >= MIN_MENTIONS
            and count >= DOMINANCE * second
            and attack_max <= OPEN_MAX_ATTACK_MENTIONS
            and actor in lead
        ):
            if actor.startswith("related:"):
                return {
                    "label": "excluded",
                    "reason": "leader_related_to_attack_group",
                    "mentions": top,
                }
            return {"label": "open", "actor": actor, "mentions": top}
    if not counts:
        return {"label": "excluded", "reason": "no_actor", "mentions": top}
    return {"label": "excluded", "reason": "no_dominant_actor", "mentions": top}


def mask_text(text: str, pattern: re.Pattern[str] | None) -> str:
    """Every dictionary actor name and every vendor cluster code -> "the adversary"."""
    if pattern is not None:
        text = pattern.sub(MASK, text)
    return CLUSTER_CODE_RE.sub(MASK, text)


# --------------------------------------------------------------------------- download / build


def _raw_path(output: Path, candidate: dict[str, Any]) -> Path:
    return output / "raw" / f"{candidate['id']}.html"


def download(candidates: dict[str, Any], output: Path, *, delay: float = 1.0) -> dict[str, Any]:
    log_path = output / "download_log.json"
    log: dict[str, list[dict[str, Any]]] = (
        json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {}
    )
    done = 0
    for candidate in candidates["candidates"]:
        path = _raw_path(output, candidate)
        if path.is_file():
            continue
        attempts = log.get(candidate["id"], [])
        if attempts and (len(attempts) >= 2 or not _transient(attempts[-1])):
            continue
        if attempts and attempts[-1].get("status") == "http_429":
            time.sleep(10)
        result = fetch(candidate["url"], path, timeout=90 if attempts else 45)
        log.setdefault(candidate["id"], []).append({"url": candidate["url"], **result})
        done += 1
        print(f"{candidate['id']} {result['status']}", flush=True)
        if done % 25 == 0:
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
        time.sleep(delay)
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def build(
    candidates: dict[str, Any],
    output: Path,
    *,
    kb: AttackKnowledgeBase,
    misp: dict[str, Any] | None,
) -> dict[str, Any]:
    anonymizer = ReportAnonymizer(kb, "strict")
    dictionary = actor_dictionary(kb, misp)
    pattern = name_pattern(list(dictionary["surface"]))
    pages: dict[str, dict[str, Any]] = {}
    for candidate in candidates["candidates"]:
        path = _raw_path(output, candidate)
        if path.is_file():
            html = path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_page(html)
            pages[candidate["id"]] = {
                **parsed,
                "site": candidate["source"],
                "path": path,
                "date": page_date(html),
            }
    chrome = repeated_lines(pages)
    excluded: Counter[str] = Counter()
    decisions: dict[str, Any] = {}
    kept: list[dict[str, Any]] = []
    for candidate in candidates["candidates"]:
        page = pages.get(candidate["id"])
        if page is None:
            excluded["not_downloaded"] += 1
            decisions[candidate["id"]] = {"label": "excluded", "reason": "not_downloaded"}
            continue
        published = (
            page["date"]
            or (date.fromisoformat(candidate["url_date"]) if candidate["url_date"] else None)
            or (date.fromisoformat(candidate["listed"]) if candidate["listed"] else None)
        )
        if published is None or not WINDOW_START <= published <= WINDOW_END:
            reason = "no_date" if published is None else "outside_window"
            excluded[reason] += 1
            decisions[candidate["id"]] = {"label": "excluded", "reason": reason}
            continue
        statements = statements_of(page, anonymizer, chrome.get(page["site"], frozenset()))
        texts = [statement["text"] for statement in statements]
        decision = label_report(page["title"], texts, dictionary, pattern)
        decision["published"] = published.isoformat()
        decisions[candidate["id"]] = decision
        if decision["label"] == "excluded":
            excluded[decision["reason"]] += 1
            continue
        kept.append(
            {
                "candidate": candidate,
                "page": page,
                "published": published,
                "statements": statements,
                "decision": decision,
            }
        )
    # one copy of a cross-posted report: the earliest published
    kept.sort(key=lambda item: (item["published"], item["candidate"]["id"]))
    unique: list[dict[str, Any]] = []
    for item in kept:
        texts = {statement["text"] for statement in item["statements"]}
        twin = next(
            (
                other
                for other in unique
                if _overlap(texts, {s["text"] for s in other["statements"]}) >= DUPLICATE_SHARE
            ),
            None,
        )
        if twin is not None:
            excluded["duplicate"] += 1
            decisions[item["candidate"]["id"]]["duplicate_of"] = twin["candidate"]["id"]
            continue
        unique.append(item)
    # at most GROUP_CAP reports per actor, newest first
    by_actor: dict[str, list[dict[str, Any]]] = {}
    for item in unique:
        actor = item["decision"].get("group_id") or item["decision"].get("actor")
        by_actor.setdefault(actor, []).append(item)
    chosen: list[dict[str, Any]] = []
    for _actor, items in by_actor.items():
        items.sort(key=lambda item: (item["published"], item["candidate"]["id"]), reverse=True)
        chosen.extend(items[:GROUP_CAP])
        for item in items[GROUP_CAP:]:
            excluded["over_group_cap"] += 1
            decisions[item["candidate"]["id"]]["over_cap"] = True
    chosen.sort(key=lambda item: (item["published"], item["candidate"]["id"]))
    known_rows, open_rows = [], []
    for item in chosen:
        candidate, page, decision = item["candidate"], item["page"], item["decision"]
        statements = [
            {**statement, "text": mask_text(statement["text"], pattern)}
            for statement in item["statements"]
        ]
        row = {
            "report_id": f"fresh-{candidate['id']}",
            "group_id": decision.get("group_id"),
            "accept": [],
            "kind": "fresh",
            "source": candidate["source"],
            "title": mask_text(page["title"], pattern),
            "url": candidate["url"],
            "published": item["published"].isoformat(),
            "statements": statements,
        }
        if decision["label"] == "known":
            known_rows.append(row)
        else:
            open_rows.append({**row, "actor": decision["actor"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("reports.jsonl", known_rows), ("open_set.jsonl", open_rows)):
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output / "labels.json").write_text(
        json.dumps(decisions, indent=1, sort_keys=True), encoding="utf-8"
    )
    groups = Counter(row["group_id"] for row in known_rows)
    manifest = {
        "source": "vendor research posts and CISA advisories published "
        f"{WINDOW_START}..{WINDOW_END}, not cited by ATT&CK (scripts/prepare_fresh_reports.py)",
        "licence": LICENCE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "attack_version": kb.version,
        "rule": {
            "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
            "min_statements": MIN_STATEMENTS,
            "min_mentions": MIN_MENTIONS,
            "dominance": DOMINANCE,
            "lead_statements": LEAD_STATEMENTS,
            "open_max_attack_mentions": OPEN_MAX_ATTACK_MENTIONS,
            "group_cap": GROUP_CAP,
            "duplicate_share": DUPLICATE_SHARE,
        },
        "dictionary": dictionary["stats"],
        "candidates": len(candidates["candidates"]),
        "downloaded": len(pages),
        "known_reports": len(known_rows),
        "known_groups": len(groups),
        "open_set_reports": len(open_rows),
        "per_source": dict(Counter(row["source"] for row in known_rows + open_rows)),
        "per_group": dict(groups.most_common()),
        "excluded": dict(excluded),
        "page_sha256": {
            item["candidate"]["id"]: hashlib.sha256(item["page"]["path"].read_bytes()).hexdigest()
            for item in chosen
        },
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="list candidate post URLs")
    parser.add_argument("--aliases", action="store_true", help="download the MISP actor list")
    parser.add_argument("--download", action="store_true", help="download candidate pages")
    parser.add_argument("--kb", type=Path, default=REPO_ROOT / "data" / "kb" / "attack_kb.json")
    parser.add_argument(
        "--bundle", type=Path, default=REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
    )
    parser.add_argument(
        "--other-sets",
        nargs="*",
        type=Path,
        default=[
            REPO_ROOT / "data" / "benchmarks" / name
            for name in (
                "vendor/reports.jsonl",
                "advisories/reports.jsonl",
                "campaigns/reports.jsonl",
                "elastic/reports.jsonl",
                "elastic/open_set.jsonl",
                "cti_hal/reports.jsonl",
                "cti_bench/product_taa_reports.jsonl",
                "athenabench/product_taa_reports.jsonl",
            )
        ],
    )
    args = parser.parse_args(argv)
    kb = AttackKnowledgeBase.load(args.kb)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.json"
    misp_path = output / "misp_threat_actor.json"
    if args.list:
        listed = list_candidates(excluded_url_set(bundle, args.other_sets))
        candidates_path.write_text(json.dumps(listed, indent=1), encoding="utf-8")
        print(f"{len(listed['candidates'])} candidates -> {candidates_path}")
        return 0
    if args.aliases:
        text = _get(MISP_URL, timeout=120.0)
        if text is None:
            print("MISP galaxy list unavailable", file=sys.stderr)
            return 1
        misp_path.write_text(text, encoding="utf-8")
        print(f"{len(json.loads(text).get('values', []))} MISP actors -> {misp_path}")
        return 0
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if args.download:
        download(candidates, output)
        return 0
    misp = json.loads(misp_path.read_text(encoding="utf-8")) if misp_path.is_file() else None
    manifest = build(candidates, output, kb=kb, misp=misp)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "candidates",
                    "downloaded",
                    "known_reports",
                    "known_groups",
                    "open_set_reports",
                    "per_source",
                    "excluded",
                    "dictionary",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
