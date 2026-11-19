"""The dated knowledge store's collector (docs/KNOWLEDGE_STORE_DESIGN.md, step 1).

Every command can be re-run. The store lives in ``--root`` (default ``data/knowledge_store``,
gitignored). The text belongs to each publisher, so the store stays local and is never
redistributed.

``list``      post URLs of every approved source from ``--since`` (sitemaps, feeds)
``seed``      import pages a report set already downloaded (``--from data/benchmarks/fresh``)
``fetch``     download every listed post not yet stored: robots.txt honoured, one request per
              second per site, sites in parallel; a refusal stops that site
``parse``     read each download (article prose, stated date), then recompute dates, chrome
              lines, duplicates and the ATT&CK-cited flag over the whole store
``policy``    apply robots.txt and refusals to what is stored (also run after every fetch)
``extract``   read each post's label, uses links and actor claims (step 2, rules only);
              ``--pin`` reads a copy with a pin's older actor lists (``pins.py``)
``check``     put the unclear links to the local model (Ollama, GPU only; resumable)
``members``   write the two ensemble members' artifact from the store or a pin's copy (step 5)
``status``    counts per source and year; ``--write`` saves ``STATUS.json`` in the root
``backfill``  list + seed + fetch + parse + status
``update``    the daily refresh: list the last ``--days`` (default 60), fetch, parse, status
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.knowledge_store.check import (  # noqa: E402
    DEFAULT_URL,
    MODEL_CUTOFF,
    PROMPT_VERSION,
    LocalChecker,
    question,
)
from apt_classifier.knowledge_store.extract import (  # noqa: E402
    EXTRACTOR_VERSION,
    ActorNames,
    NameRules,
    SoftwareMatcher,
    introduced_names,
    protected_words,
    read_post,
)
from apt_classifier.knowledge_store.fetch import FetchResult, PoliteFetcher  # noqa: E402
from apt_classifier.knowledge_store.listing import list_posts, normalise_url  # noqa: E402
from apt_classifier.knowledge_store.parse import statement_texts  # noqa: E402
from apt_classifier.knowledge_store.pins import PINS, pin_root, pinned_names  # noqa: E402
from apt_classifier.knowledge_store.sources import EXCLUDED_SITES, store_sources  # noqa: E402
from apt_classifier.knowledge_store.store import KnowledgeStore, PageReader  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
BACKFILL_SINCE = date(2022, 1, 1)
# seconds between requests to a site that rate-limited an earlier run
RATE_LIMITED_INTERVAL = 10.0
REFUSED_CODES = ("http_401", "http_403", "http_451")
# report sets whose posts the store leaves out while their benchmarks are in use
BENCHMARK_SETS = ("benchmarks/elastic/reports.jsonl", "benchmarks/elastic/reports_open_set.jsonl")
SEED_SETS = ("benchmarks/fresh", "benchmarks/fresh2")
# the MISP galaxy snapshot of 2026-10-29 (CC0 / BSD-2): actor synonyms and malware lists
MISP_DIR = "raw/misp_galaxy_24877de2"
# the model check stops after this many answers in a row slower than this (1.5 s is normal; the
# GPU the laptop throttled to 210 MHz on 2026-11-03 took 25-56 s)
SLOW_SECONDS = 15.0
SLOW_IN_A_ROW = 3


class StoreLock:
    """One collector run per store at a time (a stale lock of a dead process is taken over)."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".collector.lock"

    def __enter__(self) -> StoreLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as error:
                pid = int(self.path.read_text().strip() or 0)
                if pid and Path(f"/proc/{pid}").exists():
                    raise SystemExit(
                        f"another collector run (pid {pid}) holds {self.path}"
                    ) from error
                self.path.unlink(missing_ok=True)
                continue
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            return self
        raise SystemExit(f"could not take {self.path}")

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _excluded_site(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == site or host.endswith(f".{site}") for site in EXCLUDED_SITES)


def _benchmark_keys(data_dir: Path) -> set[str]:
    keys: set[str] = set()
    for relative in BENCHMARK_SETS:
        path = data_dir / relative
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and (url := json.loads(line).get("url")):
                keys.add(normalise_url(url))
    return keys


def _save_sites(store: KnowledgeStore, fetcher: PoliteFetcher) -> None:
    for host, robots in fetcher.robots_status.items():
        store.record_site(
            host,
            robots=robots,
            crawl_delay=fetcher.crawl_delay(host),
            stopped=fetcher.stopped.get(host),
        )
    store.commit()


def run_list(
    store: KnowledgeStore,
    fetcher: PoliteFetcher,
    *,
    since: date,
    sources: list[str] | None,
    data_dir: Path,
    max_feed_pages: int,
    workers: int,
) -> dict[str, Any]:
    print(f"listing posts from {since} ...", flush=True)
    rows, per_source = list_posts(
        store_sources(sources),
        since=since,
        get=fetcher.get_text,
        max_feed_pages=max_feed_pages,
        robots_sitemaps=fetcher.robots_sitemaps,
        workers=workers,
    )
    kept = [row for row in rows if not _excluded_site(row["url"])]
    new = store.add_posts(kept)
    excluded = store.exclude(
        [row["key"] for row in kept if row["key"] in _benchmark_keys(data_dir)], "benchmark_set"
    )
    store.set_meta("last_listing", datetime.now(UTC).isoformat(timespec="seconds"))
    previous = store.meta("listing_since")
    if previous is None or since.isoformat() < previous:
        store.set_meta("listing_since", since.isoformat())
    _save_sites(store, fetcher)
    print(f"listed {len(kept)} posts, {new} new, {excluded} left out (benchmark sets)", flush=True)
    return {"listed": len(kept), "new": new, "benchmark_set": excluded, "per_source": per_source}


def run_seed(store: KnowledgeStore, directories: list[Path]) -> dict[str, Any]:
    out = {}
    for directory in directories:
        if not (directory / "candidates.json").is_file():
            print(f"seed: no candidates.json in {directory}, skipped", flush=True)
            continue
        out[directory.name] = store.seed(directory)
        print(f"seeded from {directory}: {out[directory.name]}", flush=True)
    return out


def run_fetch(
    store: KnowledgeStore,
    fetcher: PoliteFetcher,
    *,
    sources: list[str] | None,
    workers: int,
    max_attempts: int,
    limit: int | None,
) -> dict[str, int]:
    skipped = store.stopped_sites()
    todo = store.pending(sources=sources, max_attempts=max_attempts, skip_hosts=skipped)
    if limit:
        todo = todo[:limit]
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in todo:
        by_host[urlparse(row["url"]).netloc.lower()].append(row)
    print(
        f"downloading {len(todo)} posts from {len(by_host)} sites"
        + (f" (left alone: {', '.join(sorted(skipped))})" if skipped else ""),
        flush=True,
    )
    results: queue.Queue[tuple[dict[str, Any], FetchResult] | None] = queue.Queue()

    def worker(host: str, rows: list[dict[str, Any]]) -> None:
        try:
            for index, row in enumerate(rows):
                result = fetcher.get(row["url"])
                results.put((row, result))
                if host in fetcher.stopped:
                    stop = FetchResult("site_stopped", detail=fetcher.stopped[host])
                    for rest in rows[index + 1 :]:
                        results.put((rest, stop))
                    print(f"  {host}: stopped ({fetcher.stopped[host]})", flush=True)
                    break
        except Exception as error:  # report, never hang the writer
            print(f"  {host}: worker failed: {type(error).__name__}: {error}", flush=True)
        finally:
            results.put(None)

    counts: Counter[str] = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for host, rows in sorted(by_host.items(), key=lambda item: -len(item[1])):
            pool.submit(worker, host, rows)
        finished = 0
        while finished < len(by_host):
            item = results.get()
            if item is None:
                finished += 1
                continue
            row, result = item
            store.record_fetch(row["key"], row["source"], result)
            counts[result.status] += 1
            done += 1
            if done % 100 == 0:
                store.commit()
                print(f"  {done}/{len(todo)} {dict(counts)}", flush=True)
    store.commit()
    _save_sites(store, fetcher)
    print(f"downloads: {dict(counts)}", flush=True)
    return dict(counts)


def run_policy(store: KnowledgeStore, fetcher: PoliteFetcher) -> dict[str, int]:
    """Apply the reading rules to posts imported from a report set (the earlier builders sent a
    browser name and did not read robots.txt): those of a site that refuses the reader, or that
    robots.txt does not allow, are left out of every view (they stay on disk, marked). Posts the
    store fetched itself were allowed when it read them and stay."""
    counts: Counter[str] = Counter()
    refusing = {
        host for host, robots in fetcher.robots_status.items() if robots in ("http_401", "http_403")
    } | {host for host, code in fetcher.stopped.items() if code in REFUSED_CODES}
    refusing |= {
        row["host"]
        for row in store.db.execute(
            "SELECT host FROM sites WHERE robots IN ('http_401', 'http_403') OR stopped IN "
            "('http_401', 'http_403', 'http_451')"
        )
    }
    for post in store.stored_posts():
        if not post["seeded_from"]:
            continue
        if urlparse(post["url"]).netloc.lower() in refusing:
            counts["site_refuses_reader"] += store.exclude([post["key"]], "site_refuses_reader")
        elif not fetcher.allowed(post["url"]):
            counts["robots_disallowed"] += store.exclude([post["key"]], "robots_disallowed")
    _save_sites(store, fetcher)
    print(
        f"reading rules applied to stored posts: {dict(counts) or 'nothing left out'}", flush=True
    )
    return dict(counts)


def _attack_urls(bundle_path: Path) -> set[str]:
    if not bundle_path.is_file():
        print(f"no ATT&CK bundle at {bundle_path}: the ATT&CK-cited flag stays unset", flush=True)
        return set()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    return {
        reference["url"]
        for item in bundle.get("objects", [])
        for reference in item.get("external_references", []) or []
        if reference.get("url")
    }


def run_parse(store: KnowledgeStore, *, attack_bundle: Path) -> dict[str, Any]:
    parsed = store.parse(progress=lambda n: print(f"  parsed {n}", flush=True))
    print(f"parse: {parsed}", flush=True)
    final = store.finalise(attack_urls=_attack_urls(attack_bundle))
    print(f"dates, duplicates, chrome: {final}", flush=True)
    return {"parse": parsed, "finalise": final}


# ------------------------------------------------------------------ step 2: reading facts

# what the worker processes share (set before the fork)
_READER: dict[str, Any] = {}


def _statements(pages: PageReader, key: str) -> tuple[str, list[str]]:
    source, title, lines = pages.page_lines(key)
    return title, statement_texts(lines, _READER["chrome"].get(source, frozenset()))


def _introduced_batch(keys: list[str]) -> list[tuple[str, list[str]]]:
    pages = PageReader(_READER["root"])
    try:
        return [
            (key, sorted(introduced_names(_statements(pages, key)[1], _READER["rules"])))
            for key in keys
        ]
    finally:
        pages.close()


def _read_batch(keys: list[str]) -> list[tuple[str, dict[str, Any], dict[int, str]]]:
    pages = PageReader(_READER["root"])
    out = []
    try:
        for key in keys:
            title, statements = _statements(pages, key)
            reading = read_post(
                title,
                statements,
                actors=_READER["actors"],
                software=_READER["software"],
                label_fn=_READER["label"],
            )
            cited = {link.statement for link in reading.links}
            cited |= {claim.statement for claim in reading.claims}
            out.append((key, reading.to_json(), {index: statements[index] for index in cited}))
    finally:
        pages.close()
    return out


def _map(function: Any, batches: list[list[str]], workers: int) -> Iterator[Any]:
    """``function`` over the batches: in this process for one worker, else in forked workers
    (they inherit ``_READER``; each opens the store read-only)."""
    if workers <= 1:
        yield from map(function, batches)
        return
    import multiprocessing

    with multiprocessing.get_context("fork").Pool(processes=workers) as pool:
        yield from pool.imap_unordered(function, batches)


def run_extract(
    store: KnowledgeStore, *, data_dir: Path, workers: int, pin: str | None = None
) -> dict[str, Any]:
    """Read every stored post with the ATT&CK knowledge base and the MISP snapshot. With a
    ``pin`` the store is first copied to its pin folder and its actors are named only through
    the pin's older lists (``pins.py``); software names stay the current ones."""
    from apt_classifier.attack.kb import AttackKnowledgeBase

    misp_dir = data_dir / MISP_DIR
    kb = AttackKnowledgeBase.load(data_dir / "kb" / "attack_kb.json")
    misp = json.loads((misp_dir / "threat-actor.json").read_text(encoding="utf-8"))
    if pin is None:
        return extract_facts(store, kb=kb, misp=misp, misp_dir=misp_dir, workers=workers)
    target = copy_for_pin(store, pin)
    actors_kb, actors_misp = pinned_names(kb, data_dir, pin)
    with StoreLock(target.root):
        return extract_facts(
            target,
            kb=kb,
            misp=misp,
            misp_dir=misp_dir,
            workers=workers,
            actors=(actors_kb, actors_misp),
        )


def run_members(store: KnowledgeStore, *, data_dir: Path, pin: str | None = None) -> dict[str, Any]:
    from apt_classifier.attack.kb import AttackKnowledgeBase

    kb = AttackKnowledgeBase.load(data_dir / "kb" / "attack_kb.json")
    misp = json.loads((data_dir / MISP_DIR / "threat-actor.json").read_text(encoding="utf-8"))
    if pin is None:
        return build_members(store, kb=kb, misp=misp)
    root = pin_root(store.root, pin)
    if not (root / "store.sqlite").is_file():
        raise SystemExit(f"no copy for pin {pin} in {root}: run extract --pin {pin} first")
    copy = KnowledgeStore(root)
    try:
        return build_members(copy, kb=kb, misp=misp, pin=pin)
    finally:
        copy.close()


def copy_for_pin(store: KnowledgeStore, pin: str) -> KnowledgeStore:
    """A consistent copy of the store's database in its pin folder (the gzipped pages are shared
    through a ``raw`` link); the copy keeps the model's verdicts, which apply to every link the
    pinned reading finds again with the same actor."""
    spec = PINS[pin]
    root = pin_root(store.root, pin)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    if not raw.exists():
        raw.symlink_to((store.root / "raw").resolve(), target_is_directory=True)
    target_path = root / "store.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{target_path}{suffix}").unlink(missing_ok=True)
    target = sqlite3.connect(target_path)
    try:
        store.db.backup(target)
    finally:
        target.close()
    copy = KnowledgeStore(root)
    copy.set_meta("pin", pin)
    copy.set_meta("pin_attack_release", spec.attack_release)
    copy.set_meta("pin_misp_commit", spec.misp_commit)
    copy.set_meta("copied_from", str(store.root))
    copy.set_meta("copied_at", datetime.now(UTC).isoformat(timespec="seconds"))
    copy.commit()
    print(f"copied the store to {root} for pin {pin}", flush=True)
    return copy


def _prepare_reader(
    store: KnowledgeStore,
    *,
    kb: Any,
    misp: dict[str, Any],
    misp_dir: Path,
    workers: int,
    actors: tuple[Any, dict[str, Any]] | None = None,
) -> tuple[list[str], Counter[str]]:
    """Set ``_READER`` up (names, label rule, site chrome, software matcher) for reading the
    store; returns the readable keys and the malware names posts introduce. ``actors``: the
    knowledge base and MISP list that name actors and label posts (a pin's older lists); the
    software names, and the actor names that never count as software, stay ``kb``'s and
    ``misp``'s."""
    from scripts.prepare_fresh_reports import (
        CLUSTER_CODE_RE,
        actor_dictionary,
        label_report,
        name_pattern,
        ordinary_words,
    )
    from scripts.report_anonymizer import _CAPITALS, _COUNTRIES

    dictionary = actor_dictionary(kb, misp)
    naming = actor_dictionary(*actors) if actors is not None else dictionary
    pattern = name_pattern(list(naming["surface"]))
    rules = NameRules(
        ordinary_words(),
        protected_words(kb),
        dictionary["surface"],
        places=[*_CAPITALS, *_COUNTRIES],
    )
    keys_by_source = store.readable_keys()
    print(f"site chrome for {len(keys_by_source)} sources ...", flush=True)
    _READER.clear()
    _READER.update(
        root=store.root,
        rules=rules,
        chrome={source: store.chrome_lines(source) for source in keys_by_source},
        actors=ActorNames(
            naming["surface"], pattern, CLUSTER_CODE_RE, dictionary["software_pattern"]
        ),
        label=lambda title, statements: label_report(title, statements, naming, pattern),
    )
    keys = [key for source in sorted(keys_by_source) for key in keys_by_source[source]]
    batches = [keys[index : index + 100] for index in range(0, len(keys), 100)]
    introduced: Counter[str] = Counter()
    for rows in _map(_introduced_batch, batches, workers):
        for _, names in rows:
            introduced.update(names)
    _READER["software"] = SoftwareMatcher(kb, rules, misp_dir=misp_dir, introduced=introduced)
    print(
        f"{len(introduced)} malware names introduced by posts "
        f"({len(_READER['software'].introduced)} kept as software names)",
        flush=True,
    )
    return keys, introduced


def extract_facts(
    store: KnowledgeStore,
    *,
    kb: Any,
    misp: dict[str, Any],
    misp_dir: Path,
    workers: int,
    actors: tuple[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every readable post's label, uses links and actor claims (rules only): actor names and
    the label rule are the fresh-set builder's, so the store and the test sets name actors
    alike (``actors``: a pin's older lists, see ``_prepare_reader``). Writes ``EXTRACT.json``
    beside the store."""
    keys, introduced = _prepare_reader(
        store, kb=kb, misp=misp, misp_dir=misp_dir, workers=workers, actors=actors
    )
    batches = [keys[index : index + 100] for index in range(0, len(keys), 100)]
    labels: Counter[str] = Counter()
    links: Counter[str] = Counter()
    claims: Counter[str] = Counter()
    software: dict[str, tuple[str, str, str]] = {}
    done = 0
    for rows in _map(_read_batch, batches, workers):
        for key, reading, sentences in rows:
            store.save_reading(key, reading, sentences, EXTRACTOR_VERSION)
            labels[reading["label"]] += 1
            for link in reading["links"]:
                links[f"{link['rule']}:{link['status']}"] += 1
                software[link["software_id"]] = (
                    link["software_id"],
                    link["software_name"],
                    link["software_kind"],
                )
            claims.update(claim["relation"] for claim in reading["claims"])
        done += len(rows)
        store.commit()
        if done % 1000 < len(rows):
            print(f"  read {done}/{len(keys)}", flush=True)
    store.register_software(software.values())
    store.set_meta("extracted_at", datetime.now(UTC).isoformat(timespec="seconds"))
    store.set_meta("extractor_version", str(EXTRACTOR_VERSION))
    store.commit()
    active = store.links_view()
    active_pairs = {(pair["actor"], pair["software_id"]) for pair in active}
    unclear_pairs = {
        (pair["actor"], pair["software_id"]) for pair in store.links_view(statuses=("unclear",))
    }
    summary = {
        "posts": len(keys),
        "labels": dict(labels),
        "links": dict(links),
        "claims": dict(claims),
        "introduced_names": len(introduced),
        "software_names_kept": len(_READER["software"].introduced),
        "software_names_counted": len(store.introduced_software()),
        "pairs_active": len(active),
        "pairs_corroborated": sum(1 for pair in active if pair["corroborated"]),
        "pairs_unclear_only": len(unclear_pairs - active_pairs),
    }
    (store.root / "EXTRACT.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"extraction: {json.dumps(summary)}", flush=True)
    return summary


def build_members(
    store: KnowledgeStore, *, kb: Any, misp: dict[str, Any], pin: str | None = None
) -> dict[str, Any]:
    """The members' artifact (``knowledge_store/members.py``) from ``store`` as extracted:
    the anonymised statements of every readable post labelled to an ATT&CK group, the uses links
    that count (active rule links, unclear links the model confirmed at ``MODEL_CUTOFF``), the
    posts that mention a name posts introduce, each such post's dates and statement keys, the
    copy groups, and every name posts introduce that counts as software at the latest date (the
    behaviour reading masks them). Actor names in the statements are masked with the current
    lists (the builder's ``mask_text``), whichever pin labelled the post."""
    from apt_classifier.knowledge_store.check import MODEL
    from apt_classifier.knowledge_store.extract import INTRODUCED_SHARE
    from apt_classifier.knowledge_store.members import members_path, statement_keys, write_artifact
    from scripts.prepare_fresh_reports import actor_dictionary, mask_text, name_pattern

    dictionary = actor_dictionary(kb, misp)
    pattern = name_pattern(list(dictionary["surface"]))
    software_pattern = dictionary["software_pattern"]
    db = store.db
    readable = {
        row["key"]: row
        for row in db.execute(
            "SELECT p.key, p.source, g.available_from, g.attack_cited FROM pages g "
            "JOIN posts p ON p.key = g.key WHERE p.excluded IS NULL AND g.duplicate_of IS NULL "
            "AND g.available_from IS NOT NULL"
        )
    }
    labelled = {
        row["key"]: row["actor"]
        for row in db.execute("SELECT key, actor FROM readings WHERE label = 'known'")
        if row["key"] in readable and str(row["actor"] or "") in kb.groups
    }
    links = [
        {
            "key": row["key"],
            "actor": row["actor"],
            "software_id": row["software_id"],
            "kind": row["software_kind"],
            "hedge": row["hedge"],
        }
        for row in db.execute(
            "SELECT l.key, l.actor, l.software_id, l.software_kind, l.hedge, l.status, "
            "c.p_supported FROM links l LEFT JOIN link_checks c ON c.key = l.key "
            "AND c.rule = l.rule AND c.statement = l.statement AND c.actor = l.actor "
            "AND c.software_id = l.software_id AND c.model = ? AND c.prompt_version = ? "
            "ORDER BY l.key, l.statement, l.actor, l.software_id",
            (MODEL, PROMPT_VERSION),
        )
        if row["key"] in readable
        and (
            row["status"] == "active"
            or (
                row["status"] == "unclear"
                and row["p_supported"] is not None
                and row["p_supported"] >= MODEL_CUTOFF
            )
        )
    ]
    names = [
        {"software_id": row["software_id"], "key": row["key"], "introduced": row["introduced"]}
        for row in db.execute("SELECT software_id, key, introduced FROM name_posts ORDER BY 1, 2")
        if row["key"] in readable
    ]
    software = {
        row["id"]: [row["name"], row["kind"]]
        for row in db.execute("SELECT id, name, kind FROM software ORDER BY id")
    }
    wanted = set(labelled) | {link["key"] for link in links}
    pages = PageReader(store.root)
    chrome: dict[str, frozenset[str]] = {}
    text_rows: list[dict[str, Any]] = []
    posts: dict[str, list[Any]] = {}
    try:
        for key in sorted(wanted):
            source, _, lines = pages.page_lines(key)
            if source not in chrome:
                chrome[source] = store.chrome_lines(source)
            statements = statement_texts(lines, chrome[source])
            row = readable[key]
            posts[key] = [
                row["source"],
                row["available_from"],
                int(row["attack_cited"] or 0),
                sorted(statement_keys(statements)),
            ]
            if key in labelled:
                text_rows.extend(
                    {
                        "key": key,
                        "group": labelled[key],
                        "text": mask_text(text, pattern, software_pattern),
                    }
                    for text in statements
                )
    finally:
        pages.close()
    duplicates = {
        row["key"]: row["duplicate_of"]
        for row in db.execute("SELECT key, duplicate_of FROM pages WHERE duplicate_of IS NOT NULL")
    }
    masking = sorted(
        software_id.removeprefix("name:") for software_id in store.introduced_software(None)
    )
    metadata = {
        "pin": pin,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "attack_version": str(getattr(kb, "version", "") or ""),
        "store": {key: store.meta(key) for key in ("finalised_at", "extracted_at", "pin")},
        "extractor_version": store.meta("extractor_version"),
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "model_cutoff": MODEL_CUTOFF,
        "introduced_share": INTRODUCED_SHARE,
        "masking_dictionary": "ATT&CK knowledge base + " + MISP_DIR,
        "counts": {
            "labelled_posts": len(labelled),
            "statements": len(text_rows),
            "links": len(links),
            "link_posts": len({link["key"] for link in links}),
            "name_posts": len(names),
            "masking_names": len(masking),
        },
    }
    path = write_artifact(
        members_path(SimpleNamespace(knowledge_store_dir=str(_store_dir(store, pin))), pin),
        {
            "metadata": metadata,
            "posts": posts,
            "duplicates": duplicates,
            "text": text_rows,
            "links": links,
            "names": names,
            "software": software,
            "masking_names": masking,
        },
    )
    print(f"members artifact {path}: {json.dumps(metadata['counts'])}", flush=True)
    return metadata


def _store_dir(store: KnowledgeStore, pin: str | None) -> Path:
    """The main store's folder (a pin's copy lives in ``<store>/pins/<pin>``)."""
    return store.root.parent.parent if pin else store.root


def check_links(
    store: KnowledgeStore,
    *,
    kb: Any,
    misp: dict[str, Any],
    misp_dir: Path,
    workers: int,
    checker: Any,
    limit: int | None = None,
    only: Any = None,
    max_seconds: float = SLOW_SECONDS,
) -> dict[str, Any]:
    """Put every unclear link the model has not answered to the local model (``check.py``);
    resumable, commits as it goes. ``only`` filters the pending links (a trial run). Stops when
    ``SLOW_IN_A_ROW`` answers in a row take over ``max_seconds``: a GPU the laptop has throttled
    answers in tens of seconds instead of 1.5, and, shared with the desktop, makes it lag."""
    _prepare_reader(store, kb=kb, misp=misp, misp_dir=misp_dir, workers=workers)
    pending = store.pending_checks(checker.model, PROMPT_VERSION)
    if only is not None:
        pending = [link for link in pending if only(link)]
    if limit is not None:
        pending = pending[:limit]
    group_names = {group_id: group.name for group_id, group in kb.groups.items()}
    pages = PageReader(store.root)
    verdicts: Counter[str] = Counter()
    confirmed = slow = 0
    stopped = None
    started = time.monotonic()
    print(f"checking {len(pending)} unclear links with {checker.model}", flush=True)
    try:
        for done, link in enumerate(pending, 1):
            title, statements = _statements(pages, link["key"])
            message = question(
                link,
                title,
                statements,
                actors=_READER["actors"],
                software=_READER["software"],
                group_names=group_names,
            )
            asked = time.monotonic()
            result = checker.check(message)
            slow = slow + 1 if time.monotonic() - asked > max_seconds else 0
            store.save_check(
                link,
                model=checker.model,
                prompt_version=PROMPT_VERSION,
                verdict=result.verdict,
                p_supported=result.p_supported,
                reason=result.reason,
                at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            verdicts[result.verdict] += 1
            confirmed += int((result.p_supported or 0.0) >= MODEL_CUTOFF)
            if slow >= SLOW_IN_A_ROW:
                stopped = (
                    f"{slow} answers in a row took over {max_seconds:.0f} s (a throttled GPU); "
                    f"stopped after {done} of {len(pending)}, run again to resume"
                )
                print(stopped, flush=True)
                break
            if done % 25 == 0 or done == len(pending):
                store.commit()
                rate = (time.monotonic() - started) / done
                print(
                    f"  {done}/{len(pending)} ({rate:.1f} s each, "
                    f"{rate * (len(pending) - done) / 60:.0f} min left) {dict(verdicts)}, "
                    f"{confirmed} at P >= {MODEL_CUTOFF}",
                    flush=True,
                )
    finally:
        store.commit()
        pages.close()
    summary = {
        "model": checker.model,
        "prompt_version": PROMPT_VERSION,
        "checked": sum(verdicts.values()),
        "verdicts": dict(verdicts),
        "confirmed": confirmed,
        "stopped": stopped,
        "pairs_active": len(store.links_view()),
        "pairs_corroborated": sum(1 for pair in store.links_view() if pair["corroborated"]),
    }
    (store.root / "CHECK.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"check: {json.dumps(summary)}", flush=True)
    return summary


def run_check(
    store: KnowledgeStore, *, data_dir: Path, workers: int, url: str, limit: int | None
) -> dict[str, Any]:
    import httpx

    from apt_classifier.attack.kb import AttackKnowledgeBase
    from scripts.local_llm_check import gpu_power_note

    misp_dir = data_dir / MISP_DIR
    checker = LocalChecker(url=url)
    local = url == DEFAULT_URL  # the laptop's own GPU; a remote one reports nothing here
    try:
        if local:
            print(gpu_power_note(), flush=True)
        checker.load()  # GPU only: stops here unless all of the model is in GPU memory
        summary = check_links(
            store,
            kb=AttackKnowledgeBase.load(data_dir / "kb" / "attack_kb.json"),
            misp=json.loads((misp_dir / "threat-actor.json").read_text(encoding="utf-8")),
            misp_dir=misp_dir,
            workers=workers,
            checker=checker,
            limit=limit,
        )
        if summary["stopped"] and local:
            print(gpu_power_note(), flush=True)
        return summary
    finally:
        with contextlib.suppress(httpx.HTTPError):  # a server that never answered
            checker.unload()  # frees the GPU for the desktop at once, not after the keep-alive
        checker.close()


def print_status(status: dict[str, Any], years: range) -> None:
    head = (
        f"{'source':15s} {'listed':>6s} {'stored':>6s} {'refused':>7s} {'dated':>6s} {'dupes':>5s}"
    )
    head += "".join(f" {year:>5d}" for year in years) + "  oldest      newest"
    print(head)
    for source, row in sorted(status["per_source"].items()):
        by_year = row.get("by_year", {})
        refused = row.get("refused", 0) + row.get("robots", 0)
        line = (
            f"{source:15s} {row.get('listed', 0):6d} {row.get('downloaded', 0):6d} "
            f"{refused:7d} {row.get('dated', 0):6d} {row.get('duplicates', 0):5d}"
        )
        line += "".join(f" {by_year.get(str(year), 0):5d}" for year in years)
        line += f"  {row.get('oldest') or '-':10s}  {row.get('newest') or '-'}"
        print(line)
    totals = status["totals"]
    disk = status["disk"]
    print(
        f"total: {totals.get('listed', 0)} listed, {totals.get('downloaded', 0)} stored, "
        f"{totals.get('dated', 0)} dated, {totals.get('waiting', 0)} waiting; raw "
        f"{disk['raw_gzip_bytes'] / 1e9:.2f} GB gzip, database {disk['database_bytes'] / 1e6:.0f} MB"
    )
    stopped = [
        site
        for site in status["sites"]
        if site.get("stopped") or site.get("robots") not in ("ok", "missing")
    ]
    for site in stopped:
        print(f"  site {site['host']}: robots {site['robots']}, stopped {site['stopped']}")


def run_status(store: KnowledgeStore, *, write: bool) -> dict[str, Any]:
    status = store.status()
    print_status(status, range(2022, datetime.now(UTC).year + 1))
    if write:
        path = store.root / "STATUS.json"
        path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"wrote {path}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--root", type=Path, default=None, help="default: <data-dir>/knowledge_store"
    )
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--interval", type=float, default=1.0, help="seconds between requests to one site"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "backfill"):
        command = commands.add_parser(name)
        command.add_argument("--since", type=date.fromisoformat, default=BACKFILL_SINCE)
        command.add_argument("--max-feed-pages", type=int, default=400)
        if name == "backfill":
            command.add_argument("--seed", nargs="*", type=Path, default=None)
            command.add_argument("--max-attempts", type=int, default=3)
    seed = commands.add_parser("seed")
    seed.add_argument("--from", dest="directories", nargs="+", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--max-attempts", type=int, default=3)
    fetch.add_argument("--limit", type=int, default=None)
    commands.add_parser("parse")
    commands.add_parser("policy", help="apply robots.txt and refusals to stored posts")
    extract = commands.add_parser(
        "extract", help="read labels, uses links and actor claims (step 2)"
    )
    extract.add_argument(
        "--pin",
        choices=sorted(PINS),
        help="read a copy of the store in its pin folder, naming actors only through the pin's "
        "older ATT&CK and MISP lists (pins.py)",
    )
    members = commands.add_parser("members", help="write the ensemble members' artifact (step 5)")
    members.add_argument(
        "--pin", choices=sorted(PINS), help="from the pin's copy (run extract --pin first)"
    )
    check = commands.add_parser("check", help="put unclear links to the local model (step 2)")
    check.add_argument("--url", default=DEFAULT_URL, help="the Ollama server (GPU only)")
    check.add_argument("--limit", type=int, default=None, help="check at most this many links")
    status = commands.add_parser("status")
    status.add_argument("--write", action="store_true")
    update = commands.add_parser("update")
    update.add_argument("--days", type=int, default=60)
    update.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    root = args.root or data_dir / "knowledge_store"
    attack_bundle = data_dir / "raw" / "enterprise-attack.json"
    if args.command == "status":
        store = KnowledgeStore(root)
        run_status(store, write=args.write)
        return 0
    with StoreLock(root):
        store = KnowledgeStore(root)
        if args.command == "seed":
            run_seed(store, args.directories)
            return 0
        if args.command == "parse":
            run_parse(store, attack_bundle=attack_bundle)
            return 0
        if args.command == "extract":
            run_extract(store, data_dir=data_dir, workers=args.workers, pin=args.pin)
            return 0
        if args.command == "members":
            run_members(store, data_dir=data_dir, pin=args.pin)
            return 0
        if args.command == "check":
            run_check(
                store, data_dir=data_dir, workers=args.workers, url=args.url, limit=args.limit
            )
            return 0
        slow = dict.fromkeys(store.rate_limited_sites(), RATE_LIMITED_INTERVAL)
        with PoliteFetcher(min_interval=args.interval, host_intervals=slow) as fetcher:
            if args.command == "policy":
                run_policy(store, fetcher)
                return 0
            if args.command in ("list", "backfill", "update"):
                since = (
                    date.today() - timedelta(days=args.days)
                    if args.command == "update"
                    else args.since
                )
                run_list(
                    store,
                    fetcher,
                    since=since,
                    sources=args.sources,
                    data_dir=data_dir,
                    max_feed_pages=getattr(args, "max_feed_pages", 400),
                    workers=args.workers,
                )
            if args.command == "backfill":
                seeds = args.seed if args.seed is not None else [data_dir / s for s in SEED_SETS]
                run_seed(store, seeds)
            if args.command in ("fetch", "backfill", "update"):
                run_fetch(
                    store,
                    fetcher,
                    sources=args.sources,
                    workers=args.workers,
                    max_attempts=args.max_attempts,
                    limit=getattr(args, "limit", None),
                )
                run_policy(store, fetcher)
        if args.command in ("backfill", "update"):
            run_parse(store, attack_bundle=attack_bundle)
            run_status(store, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
