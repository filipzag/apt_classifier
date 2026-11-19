"""Public vendor reports as real-report attribution cases (a development set for group ranking).

The ranking's small-group lean shows on CTI-HAL (vendor reports) and cannot be designed there
(test set) nor on the ATT&CK folds or campaigns (they do not show it). This set is vendor
reports on other ATT&CK groups, chosen by a fixed rule, labelled only with the group.

Three steps, each re-runnable:

``--select`` writes the pre-registered labels (``data/vendor_reports/labels.json``, committed
before any download): every ATT&CK group outside CTI-HAL and Elastic that documents at least
``MIN_GROUP_TECHNIQUES`` techniques, its candidate reports in fallback order (the vendor-research
web pages ATT&CK cites for that group and for no other group, newest first; PDFs, news, court
papers and product documentation left out), all groups with at least ``LARGE_GROUP_TECHNIQUES``
techniques and ``SMALL_SAMPLE`` smaller ones drawn with ``SEED``.

``--download`` fetches each group's candidates in order (one request per second) into
``data/benchmarks/vendor/raw/`` until one passes the check: the page's prose names the group
(its ATT&CK name or an alias) more often than any other ATT&CK group, and it has at least
``MIN_STATEMENTS`` sentences. At most ``MAX_TRIES`` candidates per group; a timeout, rate limit
or server error is retried once. A candidate the labels list under ``not_a_report`` (read and
found not to be a report on the group, logged in the labels' ``amendments``) fails the check.

The build (always) reads every page the way ``prepare_elastic_reports.py`` reads a post: the
article body (``<article>``/``<main>``, else the page body; navigation, headers, footers,
scripts, ``<pre>``, tables left out), prose lines split by ``local_observations``, the same
label-blind non-narrative sections dropped (ATT&CK lists, detection rules, indicators,
references, ...) plus generic page furniture (related posts, share, subscribe), blocks that are
mostly link text, and site chrome: lines found on two or more downloaded pages of the same site
(cookie notices, sidebars, sign-up prompts). Each statement
keeps its section path, paragraph id and whether it sits in an attribution section, for the
benchmark's strict anonymization level.

Outputs under ``data/benchmarks/vendor/`` (gitignored; the text belongs to each vendor, local
evaluation only, do not redistribute): ``reports.jsonl`` (as reported to ATT&CK),
``reports_hide_cited.jsonl`` (the same reports, plus ``hidden_texts``: the group and software
procedure examples in ATT&CK that cite the report, hidden from the dense and lexical indexes
by ``attribution_benchmark.py``) and ``MANIFEST.json`` (per-page status, hashes, mention
counts, what was dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.evidence.local_extract import local_observations  # noqa: E402
from apt_classifier.knowledge_store.parse import (  # noqa: E402
    parse_page,
    repeated_lines,
    site_of,
)
from scripts.report_anonymizer import ReportAnonymizer  # noqa: E402

DEFAULT_LABELS = REPO_ROOT / "data" / "vendor_reports" / "labels.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "benchmarks" / "vendor"
LICENCE = "(c) each vendor; local evaluation only, do not redistribute"

# groups of the test sets (never in this set)
CTI_HAL_GROUPS = ("G0008", "G0016", "G0034", "G0037", "G0046", "G0049", "G0102")
ELASTIC_GROUPS = ("G0032", "G0047", "G0050", "G0065", "G0082", "G0092", "G1052")
MIN_GROUP_TECHNIQUES = 5
LARGE_GROUP_TECHNIQUES = 40
SMALL_SAMPLE = 10
SEED = 7
MIN_STATEMENTS = 15
MAX_TRIES = 3
CANDIDATES_KEPT = 5
# vendor research pages (URL substrings); everything else ATT&CK cites is left out
VENDOR_SITES = (
    "anomali.com",
    "blackberry.com",
    "blog.google/threat-analysis-group",
    "blog.qualys.com",
    "clearskysec.com",
    "cloud.google.com/blog",
    "crowdstrike.com",
    "cybereason.com",
    "cyble.com",
    "cybleinc.com",
    "cisa.gov",
    "deepinstinct.com",
    "dragos.com",
    "eset.com",
    "fireeye.com",
    "fortinet.com",
    "group-ib.com",
    "huntress.com",
    "intezer.com",
    "kaspersky.com",
    "lookout.com",
    "malwarebytes.com",
    "mandiant.com",
    "microsoft.com/en-us/security/blog",
    "microsoft.com/security/blog",
    "nccgroup",
    "paloaltonetworks.com",
    "proofpoint.com",
    "recordedfuture.com",
    "redcanary.com",
    "research.checkpoint.com",
    "secureworks.com",
    "securelist.com",
    "securityintelligence.com",
    "security.com/threat-intelligence",
    "sekoia.io",
    "sentinelone.com",
    "symantec-enterprise-blogs.security.com",
    "talosintelligence.com",
    "team-cymru",
    "threatconnect.com",
    "trendmicro.com",
    "volexity.com",
    "welivesecurity.com",
    "zscaler.com",
)
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


def _attack_id(item: dict[str, Any]) -> str | None:
    return next(
        (
            reference["external_id"]
            for reference in item.get("external_references", [])
            if reference.get("source_name") == "mitre-attack"
        ),
        None,
    )


def citation_date(description: str) -> tuple[int, int, int]:
    """(year, month, day) of a citation's publication, from its "(YEAR, Month day)" part."""
    match = re.search(r"\((\d{4})(?:,\s*([A-Za-z]+)(?:\s+(\d{1,2}))?)?", description or "")
    if not match:
        return (0, 0, 0)
    month = _MONTHS.get((match.group(2) or "").lower(), 0)
    return (int(match.group(1)), month, int(match.group(3) or 0))


def is_vendor_page(url: str) -> bool:
    low = url.lower()
    return (
        low.startswith("http") and ".pdf" not in low and any(site in low for site in VENDOR_SITES)
    )


def select(bundle: dict[str, Any], kb: AttackKnowledgeBase) -> dict[str, Any]:
    """The pre-registered group list and each group's candidate reports in fallback order."""
    sets = [
        item
        for item in bundle.get("objects", [])
        if item.get("type") == "intrusion-set"
        and not item.get("revoked")
        and not item.get("x_mitre_deprecated")
    ]
    cited_by: dict[str, set[str]] = {}
    for item in sets:
        for reference in item.get("external_references", []):
            if reference.get("url"):
                cited_by.setdefault(reference["url"], set()).add(_attack_id(item) or "")
    excluded = set(CTI_HAL_GROUPS) | set(ELASTIC_GROUPS)
    pool: list[dict[str, Any]] = []
    for item in sets:
        group_id = _attack_id(item)
        if group_id not in kb.groups or group_id in excluded:
            continue
        techniques = len(kb.groups[group_id].technique_ids)
        if techniques < MIN_GROUP_TECHNIQUES:
            continue
        candidates = []
        for reference in item.get("external_references", []):
            url = reference.get("url") or ""
            if reference.get("source_name") == "mitre-attack" or not is_vendor_page(url):
                continue
            if len(cited_by.get(url, ())) > 1:  # cited for another group too
                continue
            description = reference.get("description") or ""
            candidates.append(
                {
                    "source_name": reference["source_name"],
                    "url": url,
                    "date": "-".join(f"{part:02d}" for part in citation_date(description)),
                    "_sort": citation_date(description),
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda entry: (entry["_sort"], entry["url"]), reverse=True)
        pool.append(
            {
                "group_id": group_id,
                "name": kb.groups[group_id].name,
                "techniques": techniques,
                "candidates": [
                    {key: value for key, value in entry.items() if key != "_sort"}
                    for entry in candidates[:CANDIDATES_KEPT]
                ],
            }
        )
    pool.sort(key=lambda entry: entry["group_id"])
    large = [entry for entry in pool if entry["techniques"] >= LARGE_GROUP_TECHNIQUES]
    small = [entry for entry in pool if entry["techniques"] < LARGE_GROUP_TECHNIQUES]
    drawn = random.Random(SEED).sample(small, min(SMALL_SAMPLE, len(small)))
    chosen = [{**entry, "stratum": "large"} for entry in large] + [
        {**entry, "stratum": "small"} for entry in drawn
    ]
    chosen.sort(key=lambda entry: (-entry["techniques"], entry["group_id"]))
    return {
        "description": "Pre-registered selection and labels for the vendor-report development set "
        "(small-group lean in group ranking, docs 21.3.14). Only ATT&CK ids, URLs and citation "
        "names are stored here; the pages are downloaded by scripts/prepare_vendor_reports.py "
        "into data/benchmarks/vendor/ (gitignored, local evaluation only, (c) each vendor).",
        "attack_version": kb.version,
        "registered": datetime.now(UTC).date().isoformat(),
        "role": "development set for group ranking; never CTI-HAL or Elastic groups",
        "rule": {
            "excluded_groups": {"cti_hal": list(CTI_HAL_GROUPS), "elastic": list(ELASTIC_GROUPS)},
            "min_group_techniques": MIN_GROUP_TECHNIQUES,
            "large_group_techniques": LARGE_GROUP_TECHNIQUES,
            "small_sample": SMALL_SAMPLE,
            "seed": SEED,
            "candidates": "vendor-research web pages ATT&CK cites for the group and for no other "
            "group, newest citation date first (PDFs, news, court papers, product docs left out)",
            "check": f"the page prose names the group (ATT&CK name or alias, case-sensitive) more "
            f"often than any other ATT&CK group (ties fail) and has at least {MIN_STATEMENTS} "
            f"sentences; else the next candidate, at most {MAX_TRIES} per group",
            "vendor_sites": list(VENDOR_SITES),
        },
        "eligible_groups": len(pool),
        "reports": [
            {
                "group_id": entry["group_id"],
                "name": entry["name"],
                "techniques": entry["techniques"],
                "stratum": entry["stratum"],
                "accept": [],
                "candidates": entry["candidates"],
            }
            for entry in chosen
        ],
    }


def statements_of(
    page: dict[str, Any], anonymizer: ReportAnonymizer, chrome: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    lines = [line for line in page["lines"] if line["text"] not in chrome]
    statements = []
    for observation in local_observations("\n".join(line["text"] for line in lines)):
        line = lines[observation.provenance.line_numbers[0] - 1]
        statements.append(
            {
                "text": observation.text,
                "technique_ids": [],
                "section": " > ".join(line["path"]),
                "paragraph": line["paragraph"],
                "section_attribution": any(
                    anonymizer.heading_is_attribution(heading) for heading in line["path"]
                ),
            }
        )
    return statements


def group_name_patterns(kb: AttackKnowledgeBase) -> dict[str, re.Pattern[str]]:
    """Per group, its ATT&CK name and aliases as proper nouns (case-sensitive, whole words)."""
    patterns = {}
    for group_id, group in kb.groups.items():
        names = sorted(
            {name for name in (group.name, *group.aliases) if name and len(name) > 2},
            key=lambda name: (-len(name), name),
        )
        if names:
            patterns[group_id] = re.compile(
                r"(?<![\w-])(?:" + "|".join(re.escape(name) for name in names) + r")(?![\w-])"
            )
    return patterns


def mention_check(
    statements: list[dict[str, Any]], group_id: str, patterns: dict[str, re.Pattern[str]]
) -> tuple[bool, dict[str, int]]:
    text = "\n".join(statement["text"] for statement in statements)
    counts = {
        other: len(pattern.findall(text))
        for other, pattern in patterns.items()
        if pattern.search(text)
    }
    own = counts.get(group_id, 0)
    rivals = [count for other, count in counts.items() if other != group_id]
    passed = own > 0 and all(own > count for count in rivals) and len(statements) >= MIN_STATEMENTS
    top = dict(sorted(counts.items(), key=lambda item: -item[1])[:5])
    return passed, top


def cited_procedures(bundle: dict[str, Any], source_name: str) -> list[str]:
    """Group and software procedure examples (raw ATT&CK text) that cite ``source_name``."""
    kinds = {
        item["id"]: item.get("type")
        for item in bundle.get("objects", [])
        if "id" in item and not item.get("revoked") and not item.get("x_mitre_deprecated")
    }
    marker = f"(Citation: {source_name})"
    found = set()
    for item in bundle.get("objects", []):
        if item.get("type") != "relationship" or item.get("relationship_type") != "uses":
            continue
        if kinds.get(item.get("source_ref")) not in ("intrusion-set", "malware", "tool"):
            continue
        if kinds.get(item.get("target_ref")) != "attack-pattern":
            continue
        description = item.get("description") or ""
        if marker in description:
            found.add(description)
    return sorted(found)


def _raw_path(output: Path, group_id: str, index: int) -> Path:
    return output / "raw" / f"{group_id}-{index}.html"


def load_pages(
    labels: dict[str, Any],
    output: Path,
    pages: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Every downloaded candidate page, parsed (pages already in ``pages`` are kept)."""
    pages = dict(pages or {})
    for entry in labels["reports"]:
        for index, candidate in enumerate(entry["candidates"][:MAX_TRIES]):
            key = (entry["group_id"], index)
            path = _raw_path(output, *key)
            if key in pages or not path.is_file():
                continue
            parsed = parse_page(path.read_text(encoding="utf-8", errors="replace"))
            pages[key] = {**parsed, "site": site_of(candidate["url"]), "path": path}
    return pages


def choose(
    entry: dict[str, Any],
    pages: dict[tuple[str, int], dict[str, Any]],
    chrome: dict[str, frozenset[str]],
    *,
    anonymizer: ReportAnonymizer,
    patterns: dict[str, re.Pattern[str]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The group's first downloaded candidate that passes the check, and every check made."""
    checks = []
    for index, candidate in enumerate(entry["candidates"][:MAX_TRIES]):
        page = pages.get((entry["group_id"], index))
        if index in entry.get("not_a_report", ()):  # read, and not a report (labels amendment)
            checks.append({"index": index, "not_a_report": True})
            continue
        if page is None:
            checks.append({"index": index, "downloaded": False})
            continue
        site_chrome = chrome.get(page["site"], frozenset())
        statements = statements_of(page, anonymizer, site_chrome)
        passed, mentions = mention_check(statements, entry["group_id"], patterns)
        checks.append(
            {
                "index": index,
                "downloaded": True,
                "statements": len(statements),
                "chrome_lines": sum(1 for line in page["lines"] if line["text"] in site_chrome),
                "mentions": mentions,
                "passed": passed,
            }
        )
        if passed:
            return {
                "index": index,
                "candidate": candidate,
                "page": page,
                "statements": statements,
                "mentions": mentions,
                "chrome_lines": checks[-1]["chrome_lines"],
            }, checks
    return None, checks


def fetch(url: str, destination: Path, *, timeout: float = 45.0) -> dict[str, Any]:
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en",
                },
            )
    except httpx.HTTPError as error:
        return {"status": "fetch_error", "error": f"{type(error).__name__}: {error}"[:200]}
    if response.status_code != 200:
        return {"status": f"http_{response.status_code}", "final_url": str(response.url)}
    kind = response.headers.get("content-type", "")
    if "html" not in kind:
        return {"status": "not_html", "content_type": kind[:80]}
    destination.write_bytes(response.content)
    return {"status": "fetched", "final_url": str(response.url), "bytes": len(response.content)}


def _transient(result: dict[str, Any]) -> bool:
    status = str(result.get("status"))
    return (
        status == "http_429"
        or status.startswith("http_5")
        or (status == "fetch_error" and "Timeout" in str(result.get("error")))
    )


def download(
    labels: dict[str, Any],
    output: Path,
    *,
    kb: AttackKnowledgeBase,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Fetch each group's candidates in order until one passes the check (pages on disk are
    reused; a timeout, rate limit or server error is retried once, slower)."""
    anonymizer = ReportAnonymizer(kb, "strict")
    patterns = group_name_patterns(kb)
    log_path = output / "download_log.json"
    log: dict[str, list[dict[str, Any]]] = (
        json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {}
    )
    log = {key: value for key, value in log.items() if re.fullmatch(r"G\d{4}-\d", key)}
    pages = load_pages(labels, output)
    while True:
        chrome = repeated_lines(pages)
        fetched = False
        for entry in labels["reports"]:
            group_id = entry["group_id"]
            choice, _ = choose(entry, pages, chrome, anonymizer=anonymizer, patterns=patterns)
            if choice is not None:
                continue
            for index, candidate in enumerate(entry["candidates"][:MAX_TRIES]):
                key = f"{group_id}-{index}"
                if (group_id, index) in pages or index in entry.get("not_a_report", ()):
                    continue
                attempts = log.get(key, [])
                if attempts and (len(attempts) >= 2 or not _transient(attempts[-1])):
                    continue
                retry = bool(attempts)
                if retry and attempts[-1].get("status") == "http_429":
                    time.sleep(10)
                result = fetch(
                    candidate["url"],
                    _raw_path(output, group_id, index),
                    timeout=90 if retry else 45,
                )
                time.sleep(delay)
                log.setdefault(key, []).append({"url": candidate["url"], **result})
                print(f"{key} {result['status']:<12} {site_of(candidate['url'])}", flush=True)
                fetched = True
                pages = load_pages(labels, output, pages)
                break  # one new page per group per round; the chrome is recomputed first
        if not fetched:
            break
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    labels: dict[str, Any],
    output: Path,
    *,
    kb: AttackKnowledgeBase,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Reports from the first passing page of each group (pages already downloaded)."""
    anonymizer = ReportAnonymizer(kb, "strict")
    patterns = group_name_patterns(kb)
    pages = load_pages(labels, output)
    chrome = repeated_lines(pages)
    rows: list[dict[str, Any]] = []
    per_report: dict[str, Any] = {}
    missing: dict[str, Any] = {}
    for entry in labels["reports"]:
        group_id = entry["group_id"]
        choice, checks = choose(entry, pages, chrome, anonymizer=anonymizer, patterns=patterns)
        if choice is None:
            missing[group_id] = checks
            continue
        candidate, page, statements = choice["candidate"], choice["page"], choice["statements"]
        cited = cited_procedures(bundle, candidate["source_name"])
        row = {
            "report_id": f"vendor-{group_id}",
            "group_id": group_id,
            "accept": list(entry.get("accept") or []),
            "kind": "vendor",
            "stratum": entry["stratum"],
            "title": page["title"],
            "url": candidate["url"],
            "source_name": candidate["source_name"],
            "published": candidate["date"],
            "candidate_index": choice["index"],
            "statements": statements,
        }
        rows.append(row)
        per_report[row["report_id"]] = {
            "url": candidate["url"],
            "page_sha256": _sha256(page["path"]),
            "candidate_index": choice["index"],
            "checks": checks,
            "statements": len(statements),
            "statements_in_attribution_sections": sum(
                1 for statement in statements if statement["section_attribution"]
            ),
            "mentions": choice["mentions"],
            "chrome_lines_dropped": choice["chrome_lines"],
            "dropped": page["dropped"],
            "attack_procedures_citing_report": len(cited),
            "_cited": cited,
        }
    output.mkdir(parents=True, exist_ok=True)
    plain = output / "reports.jsonl"
    hidden = output / "reports_hide_cited.jsonl"
    plain.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    hidden.write_text(
        "".join(
            json.dumps(
                {**row, "hidden_texts": per_report[row["report_id"]]["_cited"]},
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
    )
    for stats in per_report.values():
        stats.pop("_cited")
    manifest = {
        "source": "public vendor reports cited by MITRE ATT&CK (data/vendor_reports/labels.json)",
        "licence": LICENCE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "attack_version": kb.version,
        "labels_sha256": _sha256(DEFAULT_LABELS) if DEFAULT_LABELS.is_file() else None,
        "rules_sha256": _sha256(Path(__file__)),
        "pages_downloaded": len(pages),
        "reports": len(rows),
        "reports_large": sum(1 for row in rows if row["stratum"] == "large"),
        "reports_small": sum(1 for row in rows if row["stratum"] == "small"),
        "statements": sum(len(row["statements"]) for row in rows),
        "no_passing_page": sorted(missing),
        "no_passing_page_checks": missing,
        "outputs_sha256": {path.name: _sha256(path) for path in (plain, hidden)},
        "per_report": per_report,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="write the labels file and stop")
    parser.add_argument("--download", action="store_true", help="fetch missing pages first")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kb", type=Path, default=REPO_ROOT / "data" / "kb" / "attack_kb.json")
    parser.add_argument(
        "--bundle", type=Path, default=REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
    )
    args = parser.parse_args(argv)
    kb = AttackKnowledgeBase.load(args.kb)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if args.select:
        if args.labels.is_file():
            parser.error(f"{args.labels} exists: the selection is pre-registered, never rewritten")
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        labels = select(bundle, kb)
        args.labels.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {"eligible_groups": labels["eligible_groups"], "reports": len(labels["reports"])}
            )
        )
        return 0
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.download:
        download(labels, args.output, kb=kb)
    manifest = build(labels, args.output, kb=kb, bundle=bundle)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "reports",
                    "reports_large",
                    "reports_small",
                    "statements",
                    "no_passing_page",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
