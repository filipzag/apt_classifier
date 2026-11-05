"""Listing and dating posts: sitemaps, feeds, URL dates and the dates pages state.

Shared by the knowledge store and the fresh-set builder (``scripts/prepare_fresh_reports.py``,
which re-exports these names), so a test report and the store's posts are dated by one rule.
Fetching is passed in (``get``): the builder makes plain requests, the store goes through its
polite fetcher (robots.txt, one request per second per site).
"""

from __future__ import annotations

import email.utils
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any
from urllib.parse import urlparse

Getter = Callable[..., "str | None"]

_TREND_MONTH = {letter: index for index, letter in enumerate("abcdefghijkl", 1)}
_MONTHS_SHORT = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
# living documents ("Threat Brief: ... (Updated August 18)") carry content from after their date
LIVING_TITLE_RE = re.compile(r"\(Updated\b|\bUpdated (?:on )?[A-Z][a-z]+ \d", re.IGNORECASE)
# a page date shared by this many pages of one source is a re-dating stamp, not a publication date
BATCH_PAGES = 5


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


def sitemap_entries(
    url: str,
    child: str | None,
    *,
    since: date,
    get: Getter,
    seen: set[str] | None = None,
    max_sitemaps: int | None = None,
) -> Iterator[tuple[str, date | None]]:
    """(loc, lastmod) of every URL in a sitemap, following an index into matching children
    (children last modified before ``since`` are skipped; at most ``max_sitemaps`` files)."""
    seen = seen if seen is not None else set()
    if url in seen or (max_sitemaps is not None and len(seen) >= max_sitemaps):
        return
    seen.add(url)
    text = get(url, timeout=90.0)
    if text is None:
        print(f"  sitemap unavailable: {url}", flush=True)
        return
    blocks = re.findall(r"<(sitemap|url)>(.*?)</\1>", text, flags=re.S)
    for kind, block in blocks:
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block, flags=re.S)
        if not loc:
            continue
        lastmod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block, flags=re.S)
        location = re.sub(r"^<!\[CDATA\[(.*)\]\]>$", r"\1", loc.group(1).strip())
        location = location.replace("&amp;", "&")
        if kind == "sitemap":
            if child is None or re.search(child, location):
                modified = _iso_date(lastmod.group(1)) if lastmod else None
                if modified is None or modified >= since:
                    yield from sitemap_entries(
                        location, child, since=since, get=get, seen=seen, max_sitemaps=max_sitemaps
                    )
            continue
        yield location, (_iso_date(lastmod.group(1)) if lastmod else None)


def feed_entries(
    url: str, *, since: date, get: Getter, max_pages: int = 39
) -> Iterator[tuple[str, date | None]]:
    """(link, published) of every item, paging ``{page}`` until older than ``since``."""
    pages = range(1, max_pages + 1) if "{page}" in url else range(1, 2)
    seen: set[str] = set()
    for page in pages:
        text = get(url.format(page=page))
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
        if fresh == 0 or (oldest is not None and oldest < since):
            return


# --------------------------------------------------------------------------- page dates


# meta names that state a publication date, most specific first (Recorded Future's
# "published-date" is the research date; its JSON-LD datePublished is a re-dating stamp)
_DATE_META = (
    "published-date",
    "article:published_time",
    "og:published_time",
    "publish-date",
    "publication_date",
    "pubdate",
    "datePublished",
    "date",
    "dc.date",
)
_MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_VISIBLE_DATE_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_RE}),?\s+(20\d\d)\b|\b({_MONTH_RE})\s+(\d{{1,2}}),?\s+(20\d\d)\b"
)
# sources whose pages state the post date only as visible text, first on the page
VISIBLE_DATE_SOURCES = frozenset({"symantec", "rapid7"})


def _visible_date(html: str) -> date | None:
    match = _VISIBLE_DATE_RE.search(html)
    if not match:
        return None
    day, month, year = (
        (match[1], match[2], match[3]) if match[1] else (match[5], match[4], match[6])
    )
    try:
        return date(int(year), _MONTHS_SHORT[month[:3].lower()], int(day))
    except (KeyError, ValueError):
        return None


def page_date(html: str, source: str | None = None) -> date | None:
    """The publication date a page states: meta names in ``_DATE_META`` order, then JSON-LD
    ``datePublished`` and ``<time>``; for ``VISIBLE_DATE_SOURCES`` the first date on the page."""
    if source in VISIBLE_DATE_SOURCES:
        return _visible_date(html)
    for name in _DATE_META:
        escaped = re.escape(name)
        for pattern in (
            rf"<meta[^>]+(?:property|name|itemprop)=\"{escaped}\"[^>]+content=\"([^\"]+)\"",
            rf"<meta[^>]+content=\"([^\"]+)\"[^>]+(?:property|name|itemprop)=\"{escaped}\"",
        ):
            for match in re.finditer(pattern, html, flags=re.I):
                found = _iso_date(match.group(1))
                if found is not None:
                    return found
    for pattern in (r"\"datePublished\"\s*:\s*\"([^\"]+)\"", r"<time[^>]+datetime=\"([^\"]+)\""):
        for match in re.finditer(pattern, html, flags=re.I):
            found = _iso_date(match.group(1))
            if found is not None:
                return found
    return None


def batch_dates(
    pages: dict[str, dict[str, Any]], sources: dict[str, str], *, batch_pages: int = BATCH_PAGES
) -> dict[str, set[date]]:
    """Per source, the page dates shared by ``batch_pages`` or more pages: a site migration or
    re-dating stamps one date on many old pages (Rapid7, Recorded Future), so such a date is no
    publication date."""
    counts: dict[str, Counter[date]] = {}
    for page_id, page in pages.items():
        if page.get("date") is not None:
            counts.setdefault(sources[page_id], Counter())[page["date"]] += 1
    return {
        source: {day for day, count in counted.items() if count >= batch_pages}
        for source, counted in counts.items()
    }


def publication_date(
    candidate: dict[str, Any], stated: date | None, batches: dict[str, set[date]]
) -> date | None:
    """The page's stated date unless it is a batch date, else the URL's date, else a feed's
    publication date; a sitemap's lastmod is a modification date and never counts."""
    if stated is not None and stated not in batches.get(candidate["source"], set()):
        return stated
    if candidate.get("url_date"):
        return date.fromisoformat(candidate["url_date"])
    if candidate.get("method") == "feed" and candidate.get("listed"):
        return date.fromisoformat(candidate["listed"])
    return None


# --------------------------------------------------------------------------- the store's listing


def _list_source(
    source: str,
    config: Mapping[str, Any],
    *,
    since: date,
    get: Getter,
    max_feed_pages: int,
    robots_sitemaps: Callable[[str], list[str]] | None,
    max_sitemaps: int,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    entries: list[tuple[str, date | None, str]] = []
    sitemaps = list(config.get("sitemaps", ()))
    feeds = list(config.get("feeds", ()))
    # a source listed by one unpaged feed only reaches its last few posts: its robots.txt may
    # name sitemaps that reach further back (capped, and still filtered by the post pattern)
    if not sitemaps and robots_sitemaps is not None and not any("{page}" in f for f in feeds):
        host = urlparse(feeds[0]).netloc if feeds else ""
        sitemaps = robots_sitemaps(host) if host else []
        counts["robots_sitemaps"] = len(sitemaps)
    seen: set[str] = set()
    for sitemap in sitemaps:
        entries.extend(
            (loc, modified, "sitemap")
            for loc, modified in sitemap_entries(
                sitemap,
                config.get("child"),
                since=since,
                get=get,
                seen=seen,
                max_sitemaps=max_sitemaps,
            )
        )
    for feed in feeds:
        entries.extend(
            (link, published, "feed")
            for link, published in feed_entries(
                feed, since=since, get=get, max_pages=max_feed_pages
            )
        )
    rows: dict[str, dict[str, Any]] = {}
    for location, listed, method in entries:
        location = location.split("#")[0]
        if not re.search(config["post"], location):
            counts["not_a_post_url"] += 1
            continue
        encoded = url_date(location, config.get("dated"))
        # a lastmod before ``since`` means the post is older still; a later lastmod says nothing
        when = encoded or listed
        if when is not None and when < since:
            counts["before_since"] += 1
            continue
        key = normalise_url(location)
        row = rows.get(key)
        if row is None:
            row = rows[key] = {
                "key": key,
                "source": source,
                "url": location,
                "url_date": encoded.isoformat() if encoded else None,
                "feed_date": None,
                "lastmod": None,
            }
            counts["post"] += 1
        else:
            counts["listed_twice"] += 1
        if listed is not None:
            row["feed_date" if method == "feed" else "lastmod"] = listed.isoformat()
    return rows, counts


def list_posts(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    since: date,
    get: Getter,
    max_feed_pages: int = 400,
    robots_sitemaps: Callable[[str], list[str]] | None = None,
    max_sitemaps: int = 200,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Every post URL of every source listed from ``since`` (sources in parallel; the fetcher
    paces each site), with its URL date, feed date and sitemap lastmod when listed."""
    rows: dict[str, dict[str, Any]] = {}
    per_source: dict[str, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            source: pool.submit(
                _list_source,
                source,
                config,
                since=since,
                get=get,
                max_feed_pages=max_feed_pages,
                robots_sitemaps=robots_sitemaps,
                max_sitemaps=max_sitemaps,
            )
            for source, config in sources.items()
        }
        for source, future in futures.items():
            listed, counts = future.result()
            for key, row in listed.items():
                if key in rows:  # the same post under two sources: the first keeps it
                    counts["listed_by_another_source"] += 1
                    continue
                rows[key] = row
            per_source[source] = dict(counts)
            print(f"{source:15s} {dict(counts)}", flush=True)
    return list(rows.values()), per_source
