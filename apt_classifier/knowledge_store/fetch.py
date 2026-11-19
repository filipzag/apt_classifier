"""Polite fetching for the knowledge store: robots.txt, one request per second per site, no workarounds.

The rules (docs/KNOWLEDGE_STORE_DESIGN.md, "Where the reports come from"):

- plain GET requests that name the reader (``sources.USER_AGENT``), conditional on refresh
  (``If-None-Match`` / ``If-Modified-Since``);
- robots.txt is honoured for the reader's token or ``*``, including a crawl delay longer than the
  one-second floor. As RFC 9309 says, a missing robots.txt (a 4xx) allows everything and an
  unreachable one (5xx, network error) disallows the site for the run; a robots.txt the site
  refuses to the reader (401/403) is read as "keep out", stricter than the RFC;
- a refusal (401, 403, 451), a rate limit (429) or ``HOLDING_PAGES_TO_STOP`` bot-protection
  holding pages in a row ("202 Accepted", Check Point's answer on 2026-11-04) stops that site for
  the rest of the run. It is recorded and never retried with other headers or another client.

Requests to one site are serialised and spaced; different sites run in parallel threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from apt_classifier.knowledge_store.sources import ROBOTS_AGENT, USER_AGENT

MIN_INTERVAL = 1.0
REFUSED = frozenset({401, 403, 451})
GONE = frozenset({404, 410})
# "202 Accepted" holding pages in a row that stop a site like a refusal (one can be a slow page)
HOLDING_PAGES_TO_STOP = 5


@dataclass
class FetchResult:
    """One request's outcome. ``status``: fetched, not_modified, robots_disallowed, site_stopped,
    refused, rate_limited, gone, transient, not_html or error."""

    status: str
    http_status: int | None = None
    final_url: str | None = None
    content: bytes | None = None
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None
    detail: str | None = None

    @property
    def text(self) -> str | None:
        return self.content.decode("utf-8", errors="replace") if self.content is not None else None


class PoliteFetcher:
    """The store's only way onto the network (thread-safe; one lock and one clock per site)."""

    def __init__(
        self,
        *,
        user_agent: str = USER_AGENT,
        robots_agent: str = ROBOTS_AGENT,
        min_interval: float = MIN_INTERVAL,
        timeout: float = 45.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        host_intervals: dict[str, float] | None = None,
    ) -> None:
        self.robots_agent = robots_agent
        self.min_interval = float(min_interval)
        # longer spacing for particular sites (those that rate-limited an earlier run)
        self.host_intervals = dict(host_intervals or {})
        self._clock = clock
        self._sleep = sleep
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Language": "en"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self.robots_status: dict[str, str] = {}
        self.stopped: dict[str, str] = {}
        self._holding: dict[str, int] = {}
        self.requests: dict[str, int] = {}
        self._next: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _lock(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def crawl_delay(self, host: str) -> float:
        parser = self._robots.get(host)
        delay = parser.crawl_delay(self.robots_agent) if parser is not None else None
        return max(self.min_interval, float(delay or 0.0), self.host_intervals.get(host, 0.0))

    def _request(
        self, host: str, url: str, headers: dict[str, str], timeout: float | None = None
    ) -> httpx.Response | httpx.HTTPError:
        """One request, spaced from the site's previous one (the caller holds the site's lock)."""
        wait = self._next.get(host, 0.0) - self._clock()
        if wait > 0:
            self._sleep(wait)
        self.requests[host] = self.requests.get(host, 0) + 1
        try:
            if timeout is None:
                return self._client.get(url, headers=headers)
            return self._client.get(url, headers=headers, timeout=timeout)
        except httpx.HTTPError as error:
            return error
        finally:
            self._next[host] = self._clock() + self.crawl_delay(host)

    def _robots_for(self, scheme: str, host: str) -> RobotFileParser | None:
        """The site's robots.txt rules, read once per run (the caller holds the site's lock);
        ``None`` disallows the whole site."""
        if host in self._robots:
            return self._robots[host]
        response = self._request(host, f"{scheme}://{host}/robots.txt", {"Accept": "text/plain"})
        parser: RobotFileParser | None = RobotFileParser()
        if isinstance(response, httpx.HTTPError) or response.status_code >= 500:
            parser, status = None, "unreachable"
        elif response.status_code in (401, 403):
            parser, status = None, f"http_{response.status_code}"
        elif response.status_code >= 400:
            parser.parse([])
            parser.allow_all = True
            status = "missing"
        else:
            parser.parse(response.text.splitlines())
            status = "ok"
        self._robots[host] = parser
        self.robots_status[host] = status
        return parser

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        with self._lock(host):
            parser = self._robots_for(parsed.scheme or "https", host)
        return parser is not None and parser.can_fetch(self.robots_agent, url)

    def robots_sitemaps(self, host: str) -> list[str]:
        """The sitemaps a site's robots.txt names (none when robots.txt disallows the site)."""
        with self._lock(host):
            parser = self._robots_for("https", host)
        return list(parser.site_maps() or []) if parser is not None else []

    def get(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        html_only: bool = True,
        timeout: float | None = None,
    ) -> FetchResult:
        host = urlparse(url).netloc.lower()
        if host in self.stopped:
            return FetchResult("site_stopped", detail=self.stopped[host])
        if not self.allowed(url):
            return FetchResult("robots_disallowed", detail=self.robots_status.get(host))
        headers = {"Accept": "text/html,application/xhtml+xml" if html_only else "*/*"}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        with self._lock(host):
            if host in self.stopped:
                return FetchResult("site_stopped", detail=self.stopped[host])
            response = self._request(host, url, headers, timeout)
            if isinstance(response, httpx.HTTPError):
                transient = isinstance(response, httpx.TimeoutException | httpx.NetworkError)
                return FetchResult(
                    "transient" if transient else "error",
                    detail=f"{type(response).__name__}: {response}"[:200],
                )
            code = response.status_code
            if code in REFUSED or code == 429:
                self.stopped[host] = f"http_{code}"
            elif code == 202:
                self._holding[host] = self._holding.get(host, 0) + 1
                if self._holding[host] >= HOLDING_PAGES_TO_STOP:
                    self.stopped[host] = "http_202"
            else:
                self._holding.pop(host, None)
        result = FetchResult(
            "error",
            http_status=code,
            final_url=str(response.url),
            content_type=response.headers.get("content-type", "")[:80],
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
        )
        if code == 304:
            result.status = "not_modified"
        elif code == 200:
            if html_only and "html" not in result.content_type.lower():
                result.status = "not_html"
            else:
                result.status = "fetched"
                result.content = response.content
        elif code in REFUSED:
            result.status = "refused"
        elif code == 429:
            result.status = "rate_limited"
        elif code in GONE:
            result.status = "gone"
        elif code >= 500:
            result.status = "transient"
        return result

    def get_text(self, url: str, *, timeout: float | None = None) -> str | None:
        """A listing's text (sitemap, feed) or ``None``; the same rules as any other request."""
        result = self.get(url, html_only=False, timeout=timeout)
        return result.text if result.status == "fetched" else None
