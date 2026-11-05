"""The knowledge store's database: every listed post, its download, its parsed text and its dates.

SQLite (standard library) in ``<root>/store.sqlite`` and the downloaded pages, gzipped, in
``<root>/raw/<source>/``. This is step 1 of docs/KNOWLEDGE_STORE_DESIGN.md: posts and pages.
The facts read from them (names, uses links, procedure text) come with step 2. The text belongs to
each publisher (CISA advisories are US government works): the root is local and gitignored, and
nothing under it is ever committed or redistributed.

Dates. ``published`` is the fresh-set builder's rule (``listing.publication_date``): the date the
page states unless it is a batch stamp, else the URL's date, else a feed's date, never a sitemap
lastmod. ``available_from`` is the first day the text as stored could have been read, and it is
what every dated view uses:

- for a living document ("(Updated ...)" titles) it is the day it was fetched;
- for a post a sitemap says was modified more than ``UPDATE_GRACE_DAYS`` after publication it is
  that lastmod. A lastmod shared by ``BATCH_PAGES`` posts of a source is a re-dating stamp and
  does not count;
- a post dated only by a month in its URL (Trend Micro's ``/research/25/c/``) counts from the
  last day of that month;
- otherwise it is the publication date.

When in doubt the later date wins: a fact missing from an early view costs recall, while a fact
from the future would flatter every time-split test.

Duplicates (the same post at two addresses, syndicated copies) share at least ``DUPLICATE_SHARE``
of their statements, the fresh-set builder's rule. The copy available first is kept and the others
point to it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from apt_classifier.knowledge_store.check import MODEL, MODEL_CUTOFF, PROMPT_VERSION
from apt_classifier.knowledge_store.extract import INTRODUCED_SHARE
from apt_classifier.knowledge_store.fetch import FetchResult
from apt_classifier.knowledge_store.listing import (
    BATCH_PAGES,
    LIVING_TITLE_RE,
    batch_dates,
    normalise_url,
    page_date,
    publication_date,
)
from apt_classifier.knowledge_store.parse import parse_page, statement_texts
from apt_classifier.knowledge_store.sources import PUBLISHERS, SOURCES

SCHEMA_VERSION = 1
# bump when parse_page or the stored line format changes: every page is parsed again
PARSER_VERSION = 1
UPDATE_GRACE_DAYS = 30
# a line on this many pages of one source is site chrome (menus, cookie notices, sign-up prompts)
CHROME_MIN_PAGES = 5
DUPLICATE_SHARE = 0.5
DUPLICATE_MIN_STATEMENTS = 5
# a statement on more pages than this is boilerplate the chrome rule missed; it links nothing
DUPLICATE_MAX_POSTING = 20
# statuses worth another attempt on a later run (at most ``max_attempts`` in all)
RETRY_STATUSES = ("pending", "transient", "error", "rate_limited", "site_stopped")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS posts (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    url_date TEXT,
    feed_date TEXT,
    lastmod TEXT,
    first_listed TEXT NOT NULL,
    last_listed TEXT NOT NULL,
    excluded TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    http_status INTEGER,
    final_url TEXT,
    fetched_at TEXT,
    etag TEXT,
    last_modified TEXT,
    raw_path TEXT,
    raw_sha256 TEXT,
    raw_bytes INTEGER,
    seeded_from TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    key TEXT PRIMARY KEY REFERENCES posts(key),
    raw_sha256 TEXT NOT NULL,
    parser_version INTEGER NOT NULL,
    title TEXT,
    stated_date TEXT,
    lines BLOB NOT NULL,
    n_lines INTEGER NOT NULL,
    statements INTEGER,
    published TEXT,
    date_basis TEXT,
    available_from TEXT,
    living INTEGER NOT NULL DEFAULT 0,
    duplicate_of TEXT,
    attack_cited INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS fetch_log (
    key TEXT NOT NULL, at TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS sites (
    host TEXT PRIMARY KEY, robots TEXT, crawl_delay REAL, stopped TEXT, stopped_at TEXT,
    checked_at TEXT
);
CREATE TABLE IF NOT EXISTS readings (
    key TEXT PRIMARY KEY REFERENCES posts(key),
    version INTEGER NOT NULL,
    label TEXT NOT NULL,
    actor TEXT,
    reason TEXT,
    mentions TEXT
);
CREATE TABLE IF NOT EXISTS links (
    key TEXT NOT NULL REFERENCES posts(key),
    actor TEXT NOT NULL,
    software_id TEXT NOT NULL,
    software_name TEXT NOT NULL,
    software_kind TEXT NOT NULL,
    rule TEXT NOT NULL,
    hedge TEXT NOT NULL,
    statement INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    sentence TEXT,
    checked TEXT,
    checked_by TEXT
);
CREATE TABLE IF NOT EXISTS claims (
    key TEXT NOT NULL REFERENCES posts(key),
    actor_a TEXT NOT NULL,
    actor_b TEXT NOT NULL,
    relation TEXT NOT NULL,
    hedge TEXT NOT NULL,
    statement INTEGER NOT NULL,
    sentence TEXT
);
CREATE TABLE IF NOT EXISTS software (id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS name_posts (
    software_id TEXT NOT NULL, key TEXT NOT NULL, introduced INTEGER NOT NULL,
    PRIMARY KEY (software_id, key)
);
CREATE INDEX IF NOT EXISTS name_posts_key ON name_posts(key);
CREATE TABLE IF NOT EXISTS link_checks (
    key TEXT NOT NULL, rule TEXT NOT NULL, statement INTEGER NOT NULL, actor TEXT NOT NULL,
    software_id TEXT NOT NULL, model TEXT NOT NULL, prompt_version INTEGER NOT NULL,
    verdict TEXT NOT NULL, p_supported REAL, reason TEXT, checked_at TEXT NOT NULL,
    PRIMARY KEY (key, rule, statement, actor, software_id, model, prompt_version)
);
CREATE INDEX IF NOT EXISTS links_key ON links(key);
CREATE INDEX IF NOT EXISTS links_pair ON links(actor, software_id);
CREATE INDEX IF NOT EXISTS claims_key ON claims(key);
CREATE INDEX IF NOT EXISTS posts_status ON posts(status);
CREATE INDEX IF NOT EXISTS posts_source ON posts(source);
CREATE INDEX IF NOT EXISTS pages_available ON pages(available_from);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pack(lines: list[dict[str, Any]]) -> bytes:
    return zlib.compress(json.dumps(lines, ensure_ascii=False).encode("utf-8"), 6)


def _unpack(blob: bytes) -> list[dict[str, Any]]:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def _day(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


def _month_end(day: date) -> date:
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return following - timedelta(days=1)


class PageReader:
    """Read-only access to parsed posts for worker processes (no schema change, no writes)."""

    def __init__(self, root: str | Path) -> None:
        path = Path(root) / "store.sqlite"
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row

    def page_lines(self, key: str) -> tuple[str, str, list[dict[str, Any]]]:
        row = self.db.execute(
            "SELECT p.source, g.title, g.lines FROM pages g JOIN posts p ON p.key = g.key "
            "WHERE g.key = ?",
            (key,),
        ).fetchone()
        return row["source"], row["title"] or "", _unpack(row["lines"])

    def close(self) -> None:
        self.db.close()


@dataclass(frozen=True)
class StoreView:
    """What the store held before ``as_of`` (``None``: everything): the only way consumers and
    benchmarks read it, so a case or a test report never sees a post from its own day or later."""

    store: KnowledgeStore
    as_of: date | None = None

    def pages(self, *, sources: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
        yield from self.store.pages(as_of=self.as_of, sources=sources)

    def count(self) -> int:
        return sum(1 for _ in self.store.page_keys(as_of=self.as_of))

    def links(self, *, statuses: Sequence[str] = ("active",)) -> list[dict[str, Any]]:
        return self.store.links_view(self.as_of, statuses=statuses)


class KnowledgeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root / "raw"
        self.path = self.root / "store.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(sites)")}
        if "stopped_at" not in columns:  # stores created before 2026-11-03 18:00
            self.db.execute("ALTER TABLE sites ADD COLUMN stopped_at TEXT")
        stored = self.meta("schema_version")
        if stored is not None and int(stored) != SCHEMA_VERSION:
            raise RuntimeError(
                f"knowledge store {self.path} has schema {stored}, this code expects "
                f"{SCHEMA_VERSION}; rebuild the store"
            )
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ meta
    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def view(self, as_of: date | None = None) -> StoreView:
        return StoreView(self, as_of)

    # ------------------------------------------------------------------ listing
    def add_posts(self, rows: Iterable[dict[str, Any]], *, listed_at: str | None = None) -> int:
        """Insert newly listed posts; for known ones refresh the listing dates. Returns the count
        of new posts."""
        listed_at = listed_at or _now()
        before = self.db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        self.db.executemany(
            "INSERT INTO posts(key, source, url, url_date, feed_date, lastmod, first_listed, "
            "last_listed) VALUES(:key, :source, :url, :url_date, :feed_date, :lastmod, :at, :at) "
            "ON CONFLICT(key) DO UPDATE SET last_listed = excluded.last_listed, "
            "url_date = COALESCE(posts.url_date, excluded.url_date), "
            "feed_date = COALESCE(posts.feed_date, excluded.feed_date), "
            "lastmod = COALESCE(excluded.lastmod, posts.lastmod)",
            [
                {
                    "url_date": None,
                    "feed_date": None,
                    "lastmod": None,
                    **row,
                    "at": listed_at,
                }
                for row in rows
            ],
        )
        self.db.commit()
        return self.db.execute("SELECT COUNT(*) FROM posts").fetchone()[0] - before

    def exclude(self, keys: Iterable[str], reason: str) -> int:
        cursor = self.db.executemany(
            "UPDATE posts SET excluded = ? WHERE key = ? AND excluded IS NULL",
            [(reason, key) for key in keys],
        )
        self.db.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------ downloads
    def pending(
        self,
        *,
        sources: Iterable[str] | None = None,
        max_attempts: int = 3,
        skip_hosts: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Posts to download: never fetched, or failed in a way worth one more try."""
        wanted = set(sources or ())
        skipped = set(skip_hosts)
        marks = ",".join("?" * len(RETRY_STATUSES))
        rows = self.db.execute(
            f"SELECT key, source, url, attempts FROM posts WHERE excluded IS NULL "
            f"AND raw_path IS NULL AND status IN ({marks}) AND attempts < ? ORDER BY source, key",
            (*RETRY_STATUSES, max_attempts),
        ).fetchall()
        out = []
        for row in rows:
            if wanted and row["source"] not in wanted:
                continue
            host = row["url"].split("/")[2].lower() if "//" in row["url"] else ""
            if host in skipped:
                continue
            out.append(dict(row))
        return out

    def _raw_file(self, source: str, key: str) -> Path:
        return self.raw_dir / source / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.html.gz"

    def _write_raw(self, source: str, key: str, content: bytes) -> tuple[str, str, int]:
        path = self._raw_file(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(content, 6))
        return (
            str(path.relative_to(self.root)),
            hashlib.sha256(content).hexdigest(),
            len(content),
        )

    def record_fetch(
        self, key: str, source: str, result: FetchResult, *, at: str | None = None
    ) -> None:
        at = at or _now()
        self.db.execute(
            "INSERT INTO fetch_log(key, at, status, http_status, detail) VALUES(?, ?, ?, ?, ?)",
            (key, at, result.status, result.http_status, result.detail),
        )
        if result.status == "fetched" and result.content is not None:
            raw_path, sha, size = self._write_raw(source, key, result.content)
            self.db.execute(
                "UPDATE posts SET status = 'fetched', attempts = attempts + 1, http_status = ?, "
                "final_url = ?, fetched_at = ?, etag = ?, last_modified = ?, raw_path = ?, "
                "raw_sha256 = ?, raw_bytes = ? WHERE key = ?",
                (
                    result.http_status,
                    result.final_url,
                    at,
                    result.etag,
                    result.last_modified,
                    raw_path,
                    sha,
                    size,
                    key,
                ),
            )
        elif result.status == "not_modified":
            self.db.execute("UPDATE posts SET fetched_at = ? WHERE key = ?", (at, key))
        elif result.status == "site_stopped":
            # not attempted: the site stopped earlier in the run
            self.db.execute("UPDATE posts SET status = 'site_stopped' WHERE key = ?", (key,))
        else:
            self.db.execute(
                "UPDATE posts SET status = ?, attempts = attempts + 1, http_status = ?, "
                "final_url = ? WHERE key = ?",
                (result.status, result.http_status, result.final_url, key),
            )

    def record_site(
        self, host: str, *, robots: str | None, crawl_delay: float | None, stopped: str | None
    ) -> None:
        """Remember a site's robots.txt reading; a stop (refusal, rate limit) is kept, with the
        time it happened, until a later stop replaces it (a run that did not stop the site
        never clears it)."""
        now = _now()
        self.db.execute(
            "INSERT INTO sites(host, robots, crawl_delay, stopped, stopped_at, checked_at) "
            "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(host) DO UPDATE SET robots = excluded.robots, "
            "crawl_delay = excluded.crawl_delay, "
            "stopped = COALESCE(excluded.stopped, sites.stopped), "
            "stopped_at = COALESCE(excluded.stopped_at, sites.stopped_at), "
            "checked_at = excluded.checked_at",
            (host, robots, crawl_delay, stopped, now if stopped else None, now),
        )

    def stopped_sites(self, *, refused_days: int = 30, rate_limited_days: int = 1) -> set[str]:
        """Sites left alone: those that refused the reader in the last ``refused_days`` or
        rate-limited it in the last ``rate_limited_days``."""
        now = datetime.now(UTC)
        hosts = set()
        for row in self.db.execute(
            "SELECT host, stopped, stopped_at FROM sites WHERE stopped_at IS NOT NULL"
        ):
            days = rate_limited_days if row["stopped"] == "http_429" else refused_days
            if row["stopped_at"] >= (now - timedelta(days=days)).isoformat(timespec="seconds"):
                hosts.add(row["host"])
        return hosts

    def rate_limited_sites(self) -> set[str]:
        """Sites whose last stop was a rate limit: read more slowly from then on."""
        rows = self.db.execute("SELECT host FROM sites WHERE stopped = 'http_429'")
        return {row["host"] for row in rows}

    def stored_posts(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT key, source, url, seeded_from FROM posts WHERE raw_path IS NOT NULL "
            "AND excluded IS NULL"
        )
        return [dict(row) for row in rows]

    def commit(self) -> None:
        self.db.commit()

    def seed(self, directory: str | Path, *, label: str | None = None) -> dict[str, int]:
        """Import a report set's downloads (``candidates.json`` + ``raw/<id>.html``, the fresh-set
        builder's layout) so those pages are not requested again."""
        directory = Path(directory)
        label = label or directory.name
        candidates = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
        counts: Counter[str] = Counter()
        for row in candidates["candidates"]:
            raw = directory / "raw" / f"{row['id']}.html"
            if not raw.is_file():
                counts["no_download"] += 1
                continue
            key = normalise_url(row["url"])
            listed = row.get("listed")
            self.add_posts(
                [
                    {
                        "key": key,
                        "source": row["source"],
                        "url": row["url"],
                        "url_date": row.get("url_date"),
                        "feed_date": listed if row.get("method") == "feed" else None,
                        "lastmod": listed if row.get("method") == "sitemap" else None,
                    }
                ],
                listed_at=_now(),
            )
            existing = self.db.execute(
                "SELECT raw_path, source FROM posts WHERE key = ?", (key,)
            ).fetchone()
            if existing["raw_path"]:
                counts["already_stored"] += 1
                continue
            content = raw.read_bytes()
            raw_path, sha, size = self._write_raw(existing["source"], key, content)
            fetched_at = datetime.fromtimestamp(raw.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            )
            self.db.execute(
                "UPDATE posts SET status = 'fetched', http_status = 200, fetched_at = ?, "
                "raw_path = ?, raw_sha256 = ?, raw_bytes = ?, seeded_from = ? WHERE key = ?",
                (fetched_at, raw_path, sha, size, label, key),
            )
            counts["seeded"] += 1
        self.db.commit()
        return dict(counts)

    # ------------------------------------------------------------------ parsing
    def parse(
        self,
        *,
        parse_fn: Callable[[str], dict[str, Any]] = parse_page,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, int]:
        """Parse every download that is new, changed or read by an older parser."""
        rows = self.db.execute(
            "SELECT p.key, p.source, p.raw_path, p.raw_sha256 FROM posts p "
            "LEFT JOIN pages g ON g.key = p.key WHERE p.raw_path IS NOT NULL AND "
            "(g.key IS NULL OR g.raw_sha256 != p.raw_sha256 OR g.parser_version < ?)",
            (PARSER_VERSION,),
        ).fetchall()
        counts: Counter[str] = Counter()
        for index, row in enumerate(rows, 1):
            html = gzip.decompress((self.root / row["raw_path"]).read_bytes()).decode(
                "utf-8", errors="replace"
            )
            try:
                parsed = parse_fn(html)
            except Exception as error:  # a malformed page must not stop the run
                counts[f"parse_error:{type(error).__name__}"] += 1
                continue
            stated = page_date(html, row["source"])
            self.db.execute(
                "INSERT INTO pages(key, raw_sha256, parser_version, title, stated_date, lines, "
                "n_lines) VALUES(?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "raw_sha256 = excluded.raw_sha256, parser_version = excluded.parser_version, "
                "title = excluded.title, stated_date = excluded.stated_date, "
                "lines = excluded.lines, n_lines = excluded.n_lines",
                (
                    row["key"],
                    row["raw_sha256"],
                    PARSER_VERSION,
                    parsed["title"],
                    stated.isoformat() if stated else None,
                    _pack(parsed["lines"]),
                    len(parsed["lines"]),
                ),
            )
            counts["parsed"] += 1
            if index % 200 == 0:
                self.db.commit()
                if progress:
                    progress(index)
        self.db.commit()
        return dict(counts)

    def _iter_lines(self, source: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        cursor = self.db.execute(
            "SELECT g.key, g.lines FROM pages g JOIN posts p ON p.key = g.key WHERE p.source = ?",
            (source,),
        )
        for row in cursor:
            yield row["key"], _unpack(row["lines"])

    def chrome_lines(self, source: str) -> frozenset[str]:
        """Lines on ``CHROME_MIN_PAGES`` or more pages of a source (menus, notices, prompts)."""
        line_pages: Counter[str] = Counter()
        for _, lines in self._iter_lines(source):
            line_pages.update({line["text"] for line in lines})
        return frozenset(text for text, count in line_pages.items() if count >= CHROME_MIN_PAGES)

    def page_lines(self, key: str) -> tuple[str, str, list[dict[str, Any]]]:
        """(source, title, lines) of one parsed post."""
        row = self.db.execute(
            "SELECT p.source, g.title, g.lines FROM pages g JOIN posts p ON p.key = g.key "
            "WHERE g.key = ?",
            (key,),
        ).fetchone()
        return row["source"], row["title"] or "", _unpack(row["lines"])

    def readable_keys(self) -> dict[str, list[str]]:
        """Per source, the parsed posts worth reading: not excluded, not a duplicate."""
        keys: dict[str, list[str]] = defaultdict(list)
        for row in self.db.execute(
            "SELECT g.key, p.source FROM pages g JOIN posts p ON p.key = g.key "
            "WHERE p.excluded IS NULL AND g.duplicate_of IS NULL ORDER BY p.source, g.key"
        ):
            keys[row["source"]].append(row["key"])
        return dict(keys)

    def finalise(self, *, attack_urls: Iterable[str] = ()) -> dict[str, int]:
        """Recompute what depends on all pages of a source: chrome lines, statement counts,
        dates, duplicates and the ATT&CK-cited flag. Streams one source at a time."""
        cited = {normalise_url(url) for url in attack_urls}
        counts: Counter[str] = Counter()
        signatures: dict[str, frozenset[int]] = {}
        posts = {
            row["key"]: dict(row)
            for row in self.db.execute(
                "SELECT p.key, p.source, p.url, p.url_date, p.feed_date, p.lastmod, "
                "p.fetched_at, p.final_url, g.title, g.stated_date FROM posts p "
                "JOIN pages g ON g.key = p.key"
            )
        }
        by_source: dict[str, list[str]] = defaultdict(list)
        for key, post in posts.items():
            by_source[post["source"]].append(key)
        updates: list[tuple[Any, ...]] = []
        for source, keys in sorted(by_source.items()):
            chrome = self.chrome_lines(source)
            stated = {
                key: {"date": _day(posts[key]["stated_date"])}
                for key in keys
                if posts[key]["stated_date"]
            }
            # five posts on one day is a re-dating stamp in a sample, not over years of a busy blog
            batches = batch_dates(
                stated,
                dict.fromkeys(stated, source),
                batch_pages=max(BATCH_PAGES, len(stated) // 200),
            )
            month_urls = SOURCES.get(source, {}).get("dated") == "trend"
            lastmod_counts = Counter(posts[key]["lastmod"] for key in keys if posts[key]["lastmod"])
            lastmod_stamps = {day for day, count in lastmod_counts.items() if count >= BATCH_PAGES}
            statements: dict[str, int] = {}
            for key, lines in self._iter_lines(source):
                texts = statement_texts(lines, chrome)
                statements[key] = len(texts)
                if len(texts) >= DUPLICATE_MIN_STATEMENTS:
                    signatures[key] = frozenset(hash(" ".join(t.casefold().split())) for t in texts)
            for key in keys:
                post = posts[key]
                candidate = {
                    "source": source,
                    "url_date": post["url_date"],
                    "method": "feed" if post["feed_date"] else "sitemap",
                    "listed": post["feed_date"] or post["lastmod"],
                }
                stated_day = _day(post["stated_date"])
                published = publication_date(candidate, stated_day, batches)
                # a month-only URL date ("/research/25/c/") stands for its last day in views
                by_month = (
                    month_urls
                    and published is not None
                    and (stated_day is None or stated_day in batches.get(source, set()))
                    and bool(post["url_date"])
                )
                available, basis = self._available_from(
                    post, _month_end(published) if by_month else published, lastmod_stamps
                )
                living = bool(LIVING_TITLE_RE.search(post["title"] or ""))
                is_cited = normalise_url(post["url"]) in cited or (
                    bool(post["final_url"]) and normalise_url(post["final_url"]) in cited
                )
                updates.append(
                    (
                        statements.get(key, 0),
                        published.isoformat() if published else None,
                        basis,
                        available.isoformat() if available else None,
                        int(living),
                        int(is_cited),
                        key,
                    )
                )
                counts["dated" if published else "undated"] += 1
            counts["sources"] += 1
        self.db.executemany(
            "UPDATE pages SET statements = ?, published = ?, date_basis = ?, available_from = ?, "
            "living = ?, attack_cited = ?, duplicate_of = NULL WHERE key = ?",
            updates,
        )
        duplicates = self._duplicates(signatures)
        self.db.executemany(
            "UPDATE pages SET duplicate_of = ? WHERE key = ?",
            [(canonical, key) for key, canonical in duplicates.items()],
        )
        counts["duplicates"] = len(duplicates)
        self.set_meta("finalised_at", _now())
        self.db.commit()
        return dict(counts)

    @staticmethod
    def _available_from(
        post: dict[str, Any], published: date | None, lastmod_stamps: set[str]
    ) -> tuple[date | None, str | None]:
        if published is None:
            return None, None
        fetched = _day(post["fetched_at"])
        if LIVING_TITLE_RE.search(post["title"] or "") and fetched is not None:
            return max(published, fetched), "living:fetched"
        lastmod = post["lastmod"]
        if lastmod and lastmod not in lastmod_stamps:
            modified = _day(lastmod)
            if modified and modified > published + timedelta(days=UPDATE_GRACE_DAYS):
                return modified, "updated:lastmod"
        return published, "published"

    def _duplicates(self, signatures: dict[str, frozenset[int]]) -> dict[str, str]:
        """key -> the copy available first, for posts sharing at least ``DUPLICATE_SHARE`` of
        their statements (an inverted index over statement hashes, so not every pair is compared)."""
        postings: dict[int, list[str]] = defaultdict(list)
        for key, signature in signatures.items():
            for value in signature:
                postings[value].append(key)
        parent: dict[str, str] = {}

        def find(key: str) -> str:
            while parent.get(key, key) != key:
                key = parent[key]
            return key

        for key, signature in signatures.items():
            shared: Counter[str] = Counter()
            for value in signature:
                posting = postings[value]
                if 1 < len(posting) <= DUPLICATE_MAX_POSTING:
                    shared.update(other for other in posting if other != key)
            for other, count in shared.items():
                share = count / max(1, min(len(signature), len(signatures[other])))
                if share >= DUPLICATE_SHARE:
                    a, b = find(key), find(other)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        groups: dict[str, list[str]] = defaultdict(list)
        for key in signatures:
            groups[find(key)].append(key)
        available = {
            row["key"]: row["available_from"] or "9999-12-31"
            for row in self.db.execute("SELECT key, available_from FROM pages")
        }
        result: dict[str, str] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            canonical = min(members, key=lambda key: (available.get(key, "9999-12-31"), key))
            result.update({key: canonical for key in members if key != canonical})
        return result

    # ------------------------------------------------------------------ facts (step 2)
    def save_reading(
        self, key: str, reading: dict[str, Any], sentences: dict[int, str], version: int
    ) -> None:
        """Replace one post's label, uses links and actor claims (``extract.Reading`` as JSON;
        ``sentences``: the statements the links and claims point at)."""
        self.db.execute("DELETE FROM links WHERE key = ?", (key,))
        self.db.execute("DELETE FROM claims WHERE key = ?", (key,))
        self.db.execute("DELETE FROM name_posts WHERE key = ?", (key,))
        self.db.executemany(
            "INSERT INTO name_posts(software_id, key, introduced) VALUES(?, ?, ?)",
            [(software_id, key, flag) for software_id, flag in reading.get("names", {}).items()],
        )
        self.db.execute(
            "INSERT INTO readings(key, version, label, actor, reason, mentions) "
            "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET version = excluded.version, "
            "label = excluded.label, actor = excluded.actor, reason = excluded.reason, "
            "mentions = excluded.mentions",
            (
                key,
                version,
                reading["label"],
                reading["actor"],
                reading["reason"],
                json.dumps(reading["mentions"], sort_keys=True),
            ),
        )
        self.db.executemany(
            "INSERT INTO links(key, actor, software_id, software_name, software_kind, rule, "
            "hedge, statement, status, reason, sentence) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    key,
                    link["actor"],
                    link["software_id"],
                    link["software_name"],
                    link["software_kind"],
                    link["rule"],
                    link["hedge"],
                    link["statement"],
                    link["status"],
                    link["reason"],
                    sentences.get(link["statement"]),
                )
                for link in reading["links"]
            ],
        )
        self.db.executemany(
            "INSERT INTO claims(key, actor_a, actor_b, relation, hedge, statement, sentence) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    key,
                    claim["actor_a"],
                    claim["actor_b"],
                    claim["relation"],
                    claim["hedge"],
                    claim["statement"],
                    sentences.get(claim["statement"]),
                )
                for claim in reading["claims"]
            ],
        )

    def register_software(self, items: Iterable[tuple[str, str, str]]) -> None:
        self.db.executemany(
            "INSERT INTO software(id, name, kind) VALUES(?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "name = excluded.name, kind = excluded.kind",
            list(items),
        )

    def introduced_software(self, as_of: date | None = None) -> set[str]:
        """Names posts introduced that count as software as of ``as_of``: among the readable posts
        available before it that mention the name, at least ``INTRODUCED_SHARE`` introduce it."""
        sql = (
            "SELECT n.software_id, SUM(n.introduced) AS introduced, COUNT(*) AS mentions "
            "FROM name_posts n JOIN posts p ON p.key = n.key JOIN pages g ON g.key = n.key "
            "WHERE p.excluded IS NULL AND g.duplicate_of IS NULL AND g.available_from IS NOT NULL"
        )
        args: list[Any] = []
        if as_of is not None:
            sql += " AND g.available_from < ?"
            args.append(as_of.isoformat())
        return {
            row["software_id"]
            for row in self.db.execute(sql + " GROUP BY n.software_id", args)
            if row["introduced"] >= 1 and row["introduced"] >= INTRODUCED_SHARE * row["mentions"]
        }

    def links_view(
        self,
        as_of: date | None = None,
        *,
        statuses: Sequence[str] = ("active",),
        model: str | None = MODEL,
        prompt_version: int = PROMPT_VERSION,
        model_cutoff: float = MODEL_CUTOFF,
    ) -> list[dict[str, Any]]:
        """Uses links aggregated per actor and software, from posts available before ``as_of``:
        publishers, posts, first day available, and whether the link counts towards naming
        (``corroborated``: at least two publishers say it without a "possible" hedge). A name
        posts introduced counts only when ``introduced_software`` keeps it as of that date. With
        ``"active"`` among the statuses, an unclear link the local model supports with at least
        ``model_cutoff`` counts as active too (its rule carries ``+model``); ``model=None``
        leaves the model's verdicts out."""
        wanted = ",".join("?" * len(statuses))
        sql = (
            "SELECT l.actor, l.software_id, l.software_name, l.software_kind, l.hedge, l.rule, "
            "l.status, l.key, p.source, g.available_from, c.p_supported FROM links l "
            "JOIN posts p ON p.key = l.key JOIN pages g ON g.key = l.key "
            "LEFT JOIN link_checks c ON c.key = l.key AND c.rule = l.rule "
            "AND c.statement = l.statement AND c.actor = l.actor AND c.software_id = l.software_id "
            "AND c.model = ? AND c.prompt_version = ? "
            "WHERE p.excluded IS NULL AND g.duplicate_of IS NULL AND g.available_from IS NOT NULL "
            f"AND (l.status IN ({wanted}) OR (? AND l.status = 'unclear' AND c.p_supported >= ?))"
        )
        use_model = model is not None and "active" in statuses
        args: list[Any] = [model or "", prompt_version, *statuses, int(use_model), model_cutoff]
        if as_of is not None:
            sql += " AND g.available_from < ?"
            args.append(as_of.isoformat())
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        names = self.introduced_software(as_of)
        for row in self.db.execute(sql, args):
            if row["software_kind"] == "report" and row["software_id"] not in names:
                continue
            pair = pairs.setdefault(
                (row["actor"], row["software_id"]),
                {
                    "actor": row["actor"],
                    "software_id": row["software_id"],
                    "software_name": row["software_name"],
                    "software_kind": row["software_kind"],
                    "posts": set(),
                    "publishers": set(),
                    "naming_publishers": set(),
                    "first": row["available_from"],
                    "rules": set(),
                },
            )
            publisher = PUBLISHERS.get(row["source"], row["source"])
            pair["posts"].add(row["key"])
            pair["publishers"].add(publisher)
            if row["hedge"] != "possible":
                pair["naming_publishers"].add(publisher)
            pair["first"] = min(pair["first"], row["available_from"])
            confirmed = (
                use_model
                and row["status"] == "unclear"
                and row["p_supported"] is not None
                and row["p_supported"] >= model_cutoff
            )
            pair["rules"].add(row["rule"] + ("+model" if confirmed else ""))
        out = []
        for pair in pairs.values():
            out.append(
                {
                    **pair,
                    "posts": len(pair["posts"]),
                    "publishers": sorted(pair["publishers"]),
                    "naming_publishers": sorted(pair["naming_publishers"]),
                    "rules": sorted(pair["rules"]),
                    "corroborated": len(pair["naming_publishers"]) >= 2,
                }
            )
        return sorted(out, key=lambda item: (item["actor"], item["software_id"]))

    def pending_checks(self, model: str, prompt_version: int) -> list[dict[str, Any]]:
        """Unclear links of dated, readable posts that ``model`` has not answered yet."""
        rows = self.db.execute(
            "SELECT l.key, l.rule, l.statement, l.actor, l.software_id, l.software_name, "
            "l.software_kind, l.reason, l.sentence, g.available_from FROM links l "
            "JOIN posts p ON p.key = l.key JOIN pages g ON g.key = l.key "
            "LEFT JOIN link_checks c ON c.key = l.key AND c.rule = l.rule "
            "AND c.statement = l.statement AND c.actor = l.actor AND c.software_id = l.software_id "
            "AND c.model = ? AND c.prompt_version = ? "
            "WHERE l.status = 'unclear' AND p.excluded IS NULL AND g.duplicate_of IS NULL "
            "AND g.available_from IS NOT NULL AND c.key IS NULL "
            "ORDER BY l.key, l.rule, l.statement, l.actor, l.software_id",
            (model, prompt_version),
        )
        return [dict(row) for row in rows]

    def save_check(
        self,
        link: Mapping[str, Any],
        *,
        model: str,
        prompt_version: int,
        verdict: str,
        p_supported: float | None,
        reason: str,
        at: str,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO link_checks(key, rule, statement, actor, software_id, model, "
            "prompt_version, verdict, p_supported, reason, checked_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                link["key"],
                link["rule"],
                link["statement"],
                link["actor"],
                link["software_id"],
                model,
                prompt_version,
                verdict,
                p_supported,
                reason,
                at,
            ),
        )

    # ------------------------------------------------------------------ reading
    def page_keys(self, *, as_of: date | None = None) -> Iterator[str]:
        sql = (
            "SELECT g.key FROM pages g JOIN posts p ON p.key = g.key WHERE p.excluded IS NULL "
            "AND g.duplicate_of IS NULL AND g.available_from IS NOT NULL"
        )
        args: tuple[Any, ...] = ()
        if as_of is not None:
            sql += " AND g.available_from < ?"
            args = (as_of.isoformat(),)
        for row in self.db.execute(sql, args):
            yield row["key"]

    def pages(
        self, *, as_of: date | None = None, sources: Iterable[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Dated, non-duplicate, not excluded pages available before ``as_of``, oldest first."""
        sql = (
            "SELECT p.key, p.source, p.url, g.title, g.published, g.available_from, g.statements, "
            "g.attack_cited, g.lines FROM pages g JOIN posts p ON p.key = g.key "
            "WHERE p.excluded IS NULL AND g.duplicate_of IS NULL AND g.available_from IS NOT NULL"
        )
        args: list[Any] = []
        if as_of is not None:
            sql += " AND g.available_from < ?"
            args.append(as_of.isoformat())
        wanted = list(sources or ())
        if wanted:
            sql += f" AND p.source IN ({','.join('?' * len(wanted))})"
            args.extend(wanted)
        sql += " ORDER BY g.available_from, p.key"
        for row in self.db.execute(sql, args):
            page = dict(row)
            page["lines"] = _unpack(page["lines"])
            yield page

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        """Counts per source (listed, downloaded, refused, parsed, dated, per publication year),
        the sites' robots.txt and stop state, and disk use."""
        per_source: dict[str, dict[str, Any]] = {}
        for row in self.db.execute(
            "SELECT source, COUNT(*) AS listed, SUM(excluded IS NOT NULL) AS excluded, "
            "SUM(raw_path IS NOT NULL) AS downloaded, SUM(seeded_from IS NOT NULL) AS seeded, "
            "SUM(status = 'refused') AS refused, SUM(status = 'robots_disallowed') AS robots, "
            "SUM(status = 'gone') AS gone, SUM(status = 'not_html') AS not_html, "
            "SUM(raw_path IS NULL AND excluded IS NULL AND status IN "
            "('pending', 'transient', 'error', 'rate_limited', 'site_stopped')) AS waiting "
            "FROM posts GROUP BY source ORDER BY source"
        ):
            values = dict(row)
            per_source[values.pop("source")] = {key: value or 0 for key, value in values.items()}
        for row in self.db.execute(
            "SELECT p.source, COUNT(*) AS parsed, SUM(g.published IS NOT NULL) AS dated, "
            "SUM(g.duplicate_of IS NOT NULL) AS duplicates, SUM(g.living) AS living, "
            "SUM(g.attack_cited) AS attack_cited, SUM(g.statements >= 15) AS long_enough, "
            "SUM(g.date_basis = 'updated:lastmod') AS updated_later, "
            "MIN(g.published) AS oldest, MAX(g.published) AS newest "
            "FROM pages g JOIN posts p ON p.key = g.key GROUP BY p.source"
        ):
            values = dict(row)
            per_source.setdefault(values.pop("source"), {}).update(
                {key: 0 if value is None else value for key, value in values.items()}
            )
        years: dict[str, Counter[str]] = defaultdict(Counter)
        for row in self.db.execute(
            "SELECT p.source, substr(g.published, 1, 4) AS year, COUNT(*) AS n FROM pages g "
            "JOIN posts p ON p.key = g.key WHERE g.published IS NOT NULL AND "
            "g.duplicate_of IS NULL AND p.excluded IS NULL GROUP BY p.source, year"
        ):
            years[row["source"]][row["year"]] = row["n"]
        for source, counted in years.items():
            per_source[source]["by_year"] = dict(sorted(counted.items()))
        sites = [dict(row) for row in self.db.execute("SELECT * FROM sites ORDER BY host")]
        raw_bytes = sum(path.stat().st_size for path in self.raw_dir.rglob("*.gz"))
        totals: Counter[str] = Counter()
        for values in per_source.values():
            totals.update({key: value for key, value in values.items() if isinstance(value, int)})
        return {
            "generated_at": _now(),
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "finalised_at": self.meta("finalised_at"),
            "last_listing": self.meta("last_listing"),
            "listing_since": self.meta("listing_since"),
            "totals": dict(totals),
            "per_source": per_source,
            "sites": sites,
            "disk": {
                "raw_gzip_bytes": raw_bytes,
                "database_bytes": self.path.stat().st_size,
            },
        }
