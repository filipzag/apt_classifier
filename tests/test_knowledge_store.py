"""Knowledge store, step 1: listing, polite fetching, storage, dates and the dated view."""

from __future__ import annotations

import json
from datetime import date

import httpx

from apt_classifier.knowledge_store.fetch import FetchResult, PoliteFetcher
from apt_classifier.knowledge_store.listing import list_posts
from apt_classifier.knowledge_store.store import KnowledgeStore
from scripts import knowledge_store as cli


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def _fetcher(routes: dict, log: list[str], clock: _Clock | None = None, **kwargs) -> PoliteFetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        log.append(url)
        route = routes.get(url)
        if route is None:
            return httpx.Response(404)
        if callable(route):
            return route(request)
        status, body = route
        kind = "text/plain" if url.endswith("robots.txt") else "text/html; charset=utf-8"
        return httpx.Response(status, content=body.encode(), headers={"content-type": kind})

    clock = clock or _Clock()
    return PoliteFetcher(
        transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep, **kwargs
    )


def _html(title: str, published: str, sentences: list[str]) -> str:
    body = "".join(f"<p>{sentence}</p>" for sentence in sentences)
    return (
        f'<html><head><title>{title}</title><meta property="article:published_time" '
        f'content="{published}T10:00:00+00:00"></head><body><article>{body}</article></body></html>'
    )


def test_listing_follows_sitemap_index_and_stops_feeds_at_since():
    pages = {
        "https://a.test/index.xml": (
            "<sitemapindex><sitemap><loc>https://a.test/post-sitemap1.xml</loc>"
            "<lastmod>2021-06-01</lastmod></sitemap><sitemap><loc>https://a.test/post-sitemap2.xml"
            "</loc><lastmod>2025-01-01</lastmod></sitemap></sitemapindex>"
        ),
        "https://a.test/post-sitemap2.xml": (
            "<urlset><url><loc>https://a.test/new-post/</loc><lastmod>2025-01-01</lastmod></url>"
            "<url><loc>https://a.test/old-post/</loc><lastmod>2021-12-01</lastmod></url>"
            "<url><loc>https://a.test/about/x</loc></url></urlset>"
        ),
        "https://b.test/feed/?paged=1": (
            "<rss><item><link>https://b.test/p/1</link><pubDate>Mon, 06 Jan 2025 10:00:00 GMT"
            "</pubDate></item></rss>"
        ),
        "https://b.test/feed/?paged=2": (
            "<rss><item><link>https://b.test/p/2</link><pubDate>Fri, 01 Jan 2021 10:00:00 GMT"
            "</pubDate></item></rss>"
        ),
    }
    requested: list[str] = []

    def get(url, timeout=None):
        requested.append(url)
        return pages.get(url)

    sources = {
        "a": {
            "sitemaps": ["https://a.test/index.xml"],
            "child": r"post-sitemap",
            "post": r"^https://a\.test/[a-z-]+/$",
        },
        "b": {"feeds": ["https://b.test/feed/?paged={page}"], "post": r"^https://b\.test/p/\d+$"},
    }
    rows, counts = list_posts(sources, since=date(2022, 1, 1), get=get, workers=1)
    by_key = {row["key"]: row for row in rows}
    assert set(by_key) == {"a.test/new-post", "b.test/p/1"}
    assert by_key["a.test/new-post"]["lastmod"] == "2025-01-01"
    assert by_key["b.test/p/1"]["feed_date"] == "2025-01-06"
    assert counts["a"]["before_since"] == 1 and counts["a"]["not_a_post_url"] == 1
    assert "https://a.test/post-sitemap1.xml" not in requested  # its lastmod is before since
    assert "https://b.test/feed/?paged=3" not in requested  # page 2 reached 2021


def test_fetcher_honours_robots_crawl_delay_and_disallow():
    log: list[str] = []
    clock = _Clock()
    routes = {
        "https://s.test/robots.txt": (200, "User-agent: *\nCrawl-delay: 2\nDisallow: /private/\n"),
        "https://s.test/a": (200, "<p>a</p>"),
        "https://s.test/b": (200, "<p>b</p>"),
    }
    with _fetcher(routes, log, clock) as fetcher:
        assert fetcher.get("https://s.test/private/x").status == "robots_disallowed"
        assert fetcher.get("https://s.test/a").status == "fetched"
        assert fetcher.get("https://s.test/b").status == "fetched"
        assert fetcher.crawl_delay("s.test") == 2.0
    assert "https://s.test/private/x" not in log
    # one second after robots.txt, then the site's two-second crawl delay
    assert clock.sleeps == [1.0, 2.0]
    assert fetcher.robots_status == {"s.test": "ok"}


def test_refusal_stops_the_site_and_nothing_is_retried_differently():
    log: list[str] = []
    routes = {"https://r.test/a": (403, "no"), "https://r.test/b": (200, "<p>b</p>")}
    with _fetcher(routes, log) as fetcher:
        first = fetcher.get("https://r.test/a")
        second = fetcher.get("https://r.test/b")
    assert first.status == "refused" and first.http_status == 403
    assert second.status == "site_stopped"
    # robots.txt (missing: allowed) and the one refused page, nothing more
    assert log == ["https://r.test/robots.txt", "https://r.test/a"]
    assert fetcher.robots_status["r.test"] == "missing"


def test_rate_limit_conditional_requests_and_unreachable_robots():
    log: list[str] = []

    def conditional(request: httpx.Request) -> httpx.Response:
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(
            200, content=b"<p>c</p>", headers={"etag": '"v1"', "content-type": "text/html"}
        )

    routes = {
        "https://c.test/robots.txt": (200, ""),
        "https://c.test/page": conditional,
        "https://l.test/robots.txt": (200, ""),
        "https://l.test/a": (429, "slow down"),
        "https://u.test/robots.txt": (503, "down"),
    }
    with _fetcher(routes, log) as fetcher:
        fresh = fetcher.get("https://c.test/page")
        again = fetcher.get("https://c.test/page", etag=fresh.etag)
        limited = fetcher.get("https://l.test/a")
        after = fetcher.get("https://l.test/b")
        unreachable = fetcher.get("https://u.test/a")
    assert (fresh.status, fresh.etag, again.status) == ("fetched", '"v1"', "not_modified")
    assert (limited.status, after.status) == ("rate_limited", "site_stopped")
    assert unreachable.status == "robots_disallowed"
    assert fetcher.robots_status["u.test"] == "unreachable"


def test_bot_protection_holding_pages_in_a_row_stop_the_site(tmp_path):
    from apt_classifier.knowledge_store.fetch import HOLDING_PAGES_TO_STOP

    routes = {"https://h.test/robots.txt": (200, ""), "https://s.test/robots.txt": (200, "")}
    routes.update({f"https://h.test/{n}": (202, "please wait") for n in range(10)})
    routes.update(
        {"https://s.test/0": (202, "please wait"), "https://s.test/1": (200, "<p>ok</p>")}
    )
    routes.update({f"https://s.test/{n}": (202, "please wait") for n in range(2, 6)})
    log: list[str] = []
    with _fetcher(routes, log) as fetcher:
        held = [fetcher.get(f"https://h.test/{n}") for n in range(HOLDING_PAGES_TO_STOP + 1)]
        slow = [fetcher.get(f"https://s.test/{n}") for n in range(6)]
    assert [r.status for r in held[:-1]] == ["error"] * HOLDING_PAGES_TO_STOP
    assert held[-1].status == "site_stopped" and fetcher.stopped["h.test"] == "http_202"
    assert [r.status for r in slow] == ["error", "fetched", "error", "error", "error", "error"]
    assert "s.test" not in fetcher.stopped  # a page in between resets the count
    assert sum(url.startswith("https://h.test/") for url in log) == HOLDING_PAGES_TO_STOP + 1
    store = KnowledgeStore(tmp_path / "store")
    store.record_site("h.test", robots="ok", crawl_delay=None, stopped="http_202")
    assert store.stopped_sites() == {"h.test"}  # left alone like a refusal, for 30 days


def _fetched(store: KnowledgeStore, key: str, source: str, html: str, at: str) -> None:
    store.record_fetch(
        key,
        source,
        FetchResult("fetched", http_status=200, final_url=f"https://{key}", content=html.encode()),
        at=at,
    )


def test_store_dates_duplicates_and_the_dated_view(tmp_path):
    store = KnowledgeStore(tmp_path / "store")
    sentences = [f"The adversary deployed implant number {n} on the server." for n in range(8)]
    rows = [
        {"key": "v.test/original", "source": "v", "url": "https://v.test/original"},
        {"key": "v.test/copy", "source": "v", "url": "https://v.test/copy"},
        {"key": "v.test/living", "source": "v", "url": "https://v.test/living"},
        {
            "key": "v.test/edited",
            "source": "v",
            "url": "https://v.test/edited",
            "lastmod": "2025-06-01",
        },
        {"key": "v.test/cited", "source": "v", "url": "https://v.test/cited"},
    ]
    assert store.add_posts(rows) == 5
    at = "2026-11-03T10:00:00+00:00"
    _fetched(store, "v.test/original", "v", _html("Original", "2025-01-10", sentences), at)
    _fetched(
        store,
        "v.test/copy",
        "v",
        _html("Copy", "2025-02-01", [*sentences, "Another line here."]),
        at,
    )
    living = _html(
        "Threat Brief (Updated March 3)", "2025-01-05", ["The actor used a new loader."] * 1
    )
    _fetched(store, "v.test/living", "v", living, at)
    _fetched(
        store, "v.test/edited", "v", _html("Edited", "2025-01-01", ["Edited post text here."]), at
    )
    _fetched(
        store, "v.test/cited", "v", _html("Cited", "2024-03-01", ["Cited post text here."]), at
    )
    month = "https://www.trendmicro.com/en_us/research/25/c/x.html"
    store.add_posts(
        [
            {
                "key": "trendmicro.com/x",
                "source": "trendmicro",
                "url": month,
                "url_date": "2025-03-01",
            }
        ]
    )
    undated = "<html><head><title>T</title></head><body><article><p>Some text is here.</p></article></body></html>"
    _fetched(store, "trendmicro.com/x", "trendmicro", undated, at)
    assert store.parse() == {"parsed": 6}
    store.finalise(attack_urls=["https://www.v.test/cited/"])
    page = {row["key"]: dict(row) for row in store.db.execute("SELECT * FROM pages")}
    assert page["v.test/copy"]["duplicate_of"] == "v.test/original"
    assert page["v.test/original"]["statements"] == 8
    assert (page["v.test/living"]["available_from"], page["v.test/living"]["date_basis"]) == (
        "2026-11-03",
        "living:fetched",
    )
    assert page["v.test/edited"]["available_from"] == "2025-06-01"
    assert page["v.test/cited"]["attack_cited"] == 1
    # a month-only URL date counts from the end of the month
    assert page["trendmicro.com/x"]["published"] == "2025-03-01"
    assert page["trendmicro.com/x"]["available_from"] == "2025-03-31"
    early = {page["key"] for page in store.view(date(2025, 2, 1)).pages()}
    assert early == {"v.test/original", "v.test/cited"}  # the copy is a duplicate, edited is later
    later = {page["key"] for page in store.view(date(2025, 7, 1)).pages()}
    assert later == {"v.test/original", "v.test/cited", "v.test/edited", "trendmicro.com/x"}
    assert store.view(date(2025, 1, 10)).count() == 1  # nothing from the view's own day
    status = store.status()
    assert status["per_source"]["v"]["by_year"] == {"2024": 1, "2025": 3}


def test_seed_imports_a_report_sets_downloads(tmp_path):
    report_set = tmp_path / "fresh"
    (report_set / "raw").mkdir(parents=True)
    candidates = [
        {
            "id": "x-1",
            "source": "x",
            "url": "https://x.test/a/",
            "listed": "2026-03-02",
            "method": "feed",
            "url_date": None,
        },
        {
            "id": "x-2",
            "source": "x",
            "url": "https://x.test/b/",
            "listed": None,
            "method": "sitemap",
            "url_date": None,
        },
    ]
    (report_set / "candidates.json").write_text(json.dumps({"candidates": candidates}))
    (report_set / "raw" / "x-1.html").write_text(_html("A", "2026-03-02", ["Some text is here."]))
    store = KnowledgeStore(tmp_path / "store")
    assert store.seed(report_set) == {"seeded": 1, "no_download": 1}
    assert store.seed(report_set) == {"already_stored": 1, "no_download": 1}
    post = dict(store.db.execute("SELECT * FROM posts WHERE key = 'x.test/a'").fetchone())
    assert (post["seeded_from"], post["feed_date"], post["status"]) == (
        "fresh",
        "2026-03-02",
        "fetched",
    )
    assert store.pending() == []


def test_fetch_run_records_refusals_and_leaves_the_site_alone(tmp_path):
    log: list[str] = []
    page = _html("Post", "2025-05-05", ["The adversary used a scheduled task for persistence."])
    routes = {
        "https://ok.test/robots.txt": (200, ""),
        "https://ok.test/a": (200, page),
        "https://ok.test/b": (200, page.replace("Post", "Other post")),
        "https://no.test/robots.txt": (200, ""),
        "https://no.test/c": (403, "refused"),
        "https://no.test/d": (200, page),
    }
    store = KnowledgeStore(tmp_path / "store")
    store.add_posts(
        {"key": f"{host}/{name}", "source": host.split(".")[0], "url": f"https://{host}/{name}"}
        for host, name in (("ok.test", "a"), ("ok.test", "b"), ("no.test", "c"), ("no.test", "d"))
    )
    with _fetcher(routes, log, min_interval=0.0) as fetcher:
        counts = cli.run_fetch(store, fetcher, sources=None, workers=2, max_attempts=3, limit=None)
    assert counts == {"fetched": 2, "refused": 1, "site_stopped": 1}
    assert "https://no.test/d" not in log
    assert store.stopped_sites() == {"no.test"}
    assert [row["key"] for row in store.pending()] == ["no.test/d"]
    assert store.pending(skip_hosts=store.stopped_sites()) == []
    cli.run_parse(store, attack_bundle=tmp_path / "missing.json")
    totals = store.status()["totals"]
    # both pages dated; one statement each is too short to call them copies
    assert (totals["dated"], totals["duplicates"]) == (2, 0)


def test_slower_spacing_for_a_site_that_rate_limited_before():
    clock = _Clock()
    routes = {"https://s.test/robots.txt": (200, ""), "https://s.test/a": (200, "<p>a</p>")}
    with _fetcher(routes, [], clock, host_intervals={"s.test": 10.0}) as fetcher:
        fetcher.get("https://s.test/a")
        fetcher.get("https://s.test/a")
    assert clock.sleeps == [10.0, 10.0]


def test_policy_leaves_out_imported_posts_the_rules_would_not_allow(tmp_path):
    store = KnowledgeStore(tmp_path / "store")
    page = _html("Post", "2026-04-01", ["The adversary used a loader called X."])
    store.add_posts(
        {"key": f"{host}/{name}", "source": host.split(".")[0], "url": f"https://{host}/{name}"}
        for host, name in (
            ("deny.test", "a"),
            ("refuse.test", "b"),
            ("fine.test", "c"),
            ("fine.test", "own"),
        )
    )
    for key in ("deny.test/a", "refuse.test/b", "fine.test/c"):
        _fetched(store, key, key.split(".")[0], page, "2026-05-01T00:00:00+00:00")
        store.db.execute("UPDATE posts SET seeded_from = 'fresh' WHERE key = ?", (key,))
    _fetched(store, "fine.test/own", "fine", page, "2026-05-01T00:00:00+00:00")
    store.record_site("refuse.test", robots="ok", crawl_delay=1.0, stopped="http_403")
    routes = {
        "https://deny.test/robots.txt": (200, "User-agent: *\nDisallow: /a\n"),
        "https://fine.test/robots.txt": (200, ""),
    }
    with _fetcher(routes, [], min_interval=0.0) as fetcher:
        counts = cli.run_policy(store, fetcher)
    assert counts == {"site_refuses_reader": 1, "robots_disallowed": 1}
    store.parse()
    store.finalise()
    assert {page["key"] for page in store.view().pages()} == {"fine.test/c", "fine.test/own"}


def test_a_later_run_keeps_an_earlier_stop(tmp_path):
    store = KnowledgeStore(tmp_path / "store")
    store.record_site("slow.test", robots="ok", crawl_delay=1.0, stopped="http_429")
    store.record_site("slow.test", robots="ok", crawl_delay=1.0, stopped=None)
    assert store.rate_limited_sites() == {"slow.test"}
    assert store.stopped_sites() == {"slow.test"}  # within the one-day wait
