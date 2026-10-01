"""Turn Elastic Security Labs posts into local real-report attribution benchmark files.

The Security Labs RSS feed carries every post with its full HTML. The pre-registered labels in
``data/elastic_security_labs/labels.json`` name the posts (exact slugs), their ATT&CK group (or
none, for the open set) and the curated attribution headings. Each post is read the way the
product's no-LLM mode reads a pasted report: prose lines split by ``local_observations``
(sentence split, at least three words, report-wide de-duplication). What is not prose is
dropped first, label-blind and the same for every post: ``<pre>`` code, text outside any block element
(newer posts carry code that way), markdown pipe tables and code fences inside paragraphs, image and "Figure N" lines, headings themselves, and whole
non-narrative sections (MITRE ATT&CK lists, detection and hunting rules, YARA, observables and
indicators, references, appendix, diamond model). Each statement keeps its section path,
paragraph id and whether it sits in an attribution section, so the benchmark's strict level can
drop Elastic's own attribution argument (``scripts/report_anonymizer.py``).

Report-level technique ids come from every ``attack.mitre.org/techniques/...`` link in the post,
with ATT&CK's revoked-by chains resolved and a parent fallback, for a recall diagnostic.

Outputs under ``data/benchmarks/elastic/`` (gitignored; the text is (c) Elastic, local
evaluation only, do not redistribute): ``reports.jsonl`` (labelled), ``reports_open_set.jsonl``
(``group_id: null``) and ``MANIFEST.json`` (per-post hashes, what was dropped and why, the
leak audit and the sentences the strict level would drop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.kb import AttackKnowledgeBase  # noqa: E402
from apt_classifier.evidence.iocs import match_software_aliases  # noqa: E402
from apt_classifier.evidence.local_extract import local_observations  # noqa: E402
from apt_classifier.vectorstore import (  # noqa: E402
    normalize_procedure_text,
    procedure_sentence_texts,
)
from scripts.report_anonymizer import ReportAnonymizer  # noqa: E402

FEED_URL = "https://www.elastic.co/security-labs/rss/feed.xml"
DEFAULT_LABELS = REPO_ROOT / "data" / "elastic_security_labs" / "labels.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "benchmarks" / "elastic"
CONTENT_TAG = "{http://purl.org/rss/1.0/modules/content/}encoded"
LICENCE = "(c) Elastic N.V.; local evaluation only, do not redistribute"

_BLOCK_TAGS = frozenset(
    [
        "p",
        "li",
        "blockquote",
        "div",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
        "dd",
        "dt",
        "figcaption",
        "section",
        "article",
    ]
)
_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
# whole headings (normalized, lower case) of non-narrative sections, dropped for every post
_DROP_HEADING_RE = re.compile(
    r"^(?:.*mitre att&ck.*"
    r"|tactics|techniques(?: ?(?:/|\() ?sub[- ]?techniques\)?)?"
    r"|observed (?:adversary )?tactics and techniques"
    r"|detections?(?: logic| rules?(?: for .*)?| opportunities| capabilities)?"
    r"|existing detection rules|detecting and preventing .*"
    r"|(?:malware )?(?:prevention|detection) capabilities|preventions?(?: rules?(?: for .*)?)?"
    r"|hunting quer(?:y|ies).*|eql quer(?:y|ies)|es\|?ql quer(?:y|ies)"
    r"|yara(?: rules?(?: for .*)?| signatures?)?"
    r"|how to detect and prevent.*|observations|observables.*|indicators(?: of compromise.*)?|iocs?"
    r"|artifacts|references|appendix.*|(?:the )?diamond model|about elastic security labs"
    r"|defensive recommendations|recommendations)$"
)
_TECHNIQUE_HREF_RE = re.compile(r"attack\.mitre\.org/techniques/(T\d{4})(?:/(\d{3}))?", re.I)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
_FIGURE_LINE_RE = re.compile(r"^Figure \d+\b")


def _normalize_heading(text: str) -> str:
    # some feed headings carry markdown escapes ("MITRE ATT\\&CK")
    return " ".join(text.replace("®", "").replace("\\", "").split())


class _PostParser(HTMLParser):
    """Lines of prose with their heading path and paragraph id; headings are never lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[dict[str, Any]] = []
        self.headings: list[tuple[int, str]] = []
        self.heading_log: list[str] = []
        self.technique_hrefs: list[str] = []
        self.dropped: Counter[str] = Counter()
        self._buffer: list[str] = []
        self._heading_level: int | None = None
        self._heading_buffer: list[str] = []
        self._pre_depth = 0
        self._block_depth = 0
        self._paragraph = 0
        self._in_fence = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if _TECHNIQUE_HREF_RE.search(href):
                self.technique_hrefs.append(href)
        if tag == "pre":
            self._flush()
            if self._pre_depth == 0:
                self.dropped["pre_blocks"] += 1
            self._pre_depth += 1
            return
        if self._pre_depth:
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self._heading_level = _HEADING_TAGS[tag]
            self._heading_buffer = []
        elif tag in _BLOCK_TAGS:
            self._flush()
            self._block_depth += 1
        elif tag == "br":
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            return
        if self._pre_depth:
            return
        if tag in _HEADING_TAGS and self._heading_level is not None:
            text = _normalize_heading("".join(self._heading_buffer))
            level = self._heading_level
            self._heading_level = None
            while self.headings and self.headings[-1][0] >= level:
                self.headings.pop()
            if text:
                self.headings.append((level, text))
                self.heading_log.append(" > ".join(item for _, item in self.headings))
        elif tag in _BLOCK_TAGS:
            self._flush()
            self._block_depth = max(0, self._block_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._pre_depth:
            return
        if self._heading_level is not None:
            self._heading_buffer.append(data)
        elif self._block_depth == 0:
            if data.strip():  # code pasted between paragraphs without a <pre>
                self.dropped["loose_lines"] += len(data.strip().splitlines())
        else:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = "".join(self._buffer)
        self._buffer = []
        if not text.strip():
            return
        self._paragraph += 1
        path = tuple(item for _, item in self.headings)
        for raw in text.split("\n"):
            line = raw.strip()
            if line.startswith("```"):
                self._in_fence = not self._in_fence
                self.dropped["fence_lines"] += 1
                continue
            if self._in_fence:
                self.dropped["fence_lines"] += 1
                continue
            if not line:
                continue
            if line.startswith("|"):
                self.dropped["table_lines"] += 1
                continue
            line = _IMAGE_RE.sub("", line).strip()
            if not line:
                self.dropped["image_lines"] += 1
                continue
            if _FIGURE_LINE_RE.match(line):
                self.dropped["figure_lines"] += 1
                continue
            self.lines.append(
                {"text": " ".join(line.split()), "path": path, "paragraph": self._paragraph}
            )


def parse_post(html: str) -> _PostParser:
    parser = _PostParser()
    parser.feed(html)
    parser.close()
    return parser


def _path_matches(path: tuple[str, ...], entry: str) -> bool:
    """A curated heading: either one heading text or a "Parent > Heading" path suffix."""
    parts = tuple(_normalize_heading(part) for part in entry.split(">"))
    if len(parts) == 1:
        return parts[0] in path
    return any(path[index : index + len(parts)] == parts for index in range(len(path)))


def read_feed(path: Path) -> dict[str, dict[str, Any]]:
    """Feed items keyed by the exact last path segment of their link."""
    root = ET.parse(path).getroot()
    items: dict[str, dict[str, Any]] = {}
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        slug = link.rstrip("/").rsplit("/", 1)[-1]
        content = item.findtext(CONTENT_TAG) or ""
        published = item.findtext("pubDate") or ""
        try:
            date = parsedate_to_datetime(published).date().isoformat() if published else None
        except (TypeError, ValueError):
            date = None
        items[slug] = {
            "slug": slug,
            "url": link,
            "title": item.findtext("title") or "",
            "published": date,
            "content": content,
            "sha256": hashlib.sha256((content + published).encode("utf-8")).hexdigest(),
        }
    return items


def revoked_technique_map(raw_bundle: Path | None) -> dict[str, str]:
    """ATT&CK technique id -> the id that revoked it, from the STIX ``revoked-by`` links."""
    if raw_bundle is None or not raw_bundle.is_file():
        return {}
    payload = json.loads(raw_bundle.read_text(encoding="utf-8"))
    objects = {item["id"]: item for item in payload.get("objects", [])}

    def external_id(item: dict[str, Any]) -> str | None:
        for reference in item.get("external_references") or []:
            if reference.get("source_name") == "mitre-attack":
                return reference.get("external_id")
        return None

    mapping: dict[str, str] = {}
    for item in objects.values():
        if item.get("type") != "relationship" or item.get("relationship_type") != "revoked-by":
            continue
        source, target = objects.get(item["source_ref"]), objects.get(item["target_ref"])
        if not source or not target or source.get("type") != "attack-pattern":
            continue
        old, new = external_id(source), external_id(target)
        if old and new:
            mapping[old] = new
    return mapping


def attack_citations(raw_bundle: Path | None, slugs: list[str]) -> dict[str, list[str]]:
    """For each slug, the ATT&CK objects whose references cite that post (leak audit)."""
    found: dict[str, list[str]] = {slug: [] for slug in slugs}
    if raw_bundle is None or not raw_bundle.is_file():
        return found
    payload = json.loads(raw_bundle.read_text(encoding="utf-8"))
    for item in payload.get("objects", []):
        for reference in item.get("external_references") or []:
            url = str(reference.get("url") or "")
            if "elastic.co" not in url:
                continue
            for slug in slugs:
                if url.rstrip("/").endswith("/" + slug):
                    found[slug].append(f"{item.get('type')}:{item.get('id')}")
    return found


def group_citations_in_year(raw_bundle: Path | None, kb: Any, group_id: str, year: int) -> int:
    """How many of the group's ATT&CK relationship citations were published in the activity year
    (the year in the citation's source name or its "(YEAR, Month day)" date, not the
    "Retrieved" access date)."""
    if raw_bundle is None or not raw_bundle.is_file() or group_id not in kb.groups:
        return 0
    stix_id = kb.groups[group_id].stix_id
    payload = json.loads(raw_bundle.read_text(encoding="utf-8"))
    count = 0
    for item in payload.get("objects", []):
        if item.get("type") != "relationship" or item.get("source_ref") != stix_id:
            continue
        for reference in item.get("external_references") or []:
            published = re.search(r"\((\d{4})\b", str(reference.get("description") or ""))
            if str(year) in str(reference.get("source_name") or "") or (
                published is not None and published.group(1) == str(year)
            ):
                count += 1
                break
    return count


def resolve_techniques(
    hrefs: list[str], kb: Any, revoked: dict[str, str]
) -> tuple[list[str], list[str]]:
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for href in hrefs:
        match = _TECHNIQUE_HREF_RE.search(href)
        if not match:
            continue
        technique_id = match.group(1).upper() + (f".{match.group(2)}" if match.group(2) else "")
        seen: set[str] = set()
        while technique_id in revoked and technique_id not in seen:
            seen.add(technique_id)
            technique_id = revoked[technique_id]
        if technique_id in kb.techniques:
            resolved.add(technique_id)
        elif technique_id.split(".")[0] in kb.techniques:
            resolved.add(technique_id.split(".")[0])
        else:
            unresolved.add(technique_id)
    return sorted(resolved), sorted(unresolved)


def build_report(
    item: dict[str, Any],
    entry: dict[str, Any],
    *,
    kb: Any,
    anonymizer: ReportAnonymizer,
    revoked: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser = parse_post(item["content"])
    curated = list(entry.get("attribution_sections") or [])
    dropped_sections: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for line in parser.lines:
        drop = next(
            (heading for heading in line["path"] if _DROP_HEADING_RE.match(heading.lower())), None
        )
        if drop is not None:
            dropped_sections[drop] += 1
            continue
        kept.append(line)
    for entry_heading in curated:
        if not any(_path_matches(line["path"], entry_heading) for line in parser.lines):
            raise ValueError(f"{item['slug']}: attribution section {entry_heading!r} not found")
    observations = local_observations("\n".join(line["text"] for line in kept))
    statements: list[dict[str, Any]] = []
    for observation in observations:
        line = kept[observation.provenance.line_numbers[0] - 1]
        path = line["path"]
        section_attribution = any(_path_matches(path, heading) for heading in curated) or any(
            anonymizer.heading_is_attribution(heading) for heading in path
        )
        statements.append(
            {
                "text": observation.text,
                "technique_ids": [],
                "section": " > ".join(path),
                "paragraph": line["paragraph"],
                "section_attribution": section_attribution,
            }
        )
    techniques, unresolved = resolve_techniques(parser.technique_hrefs, kb, revoked)
    group_id = entry.get("group_id")
    row = {
        "report_id": f"elastic-{item['slug']}",
        "slug": item["slug"],
        "url": item["url"],
        "published": item["published"],
        "group_id": group_id,
        "accept": list(entry.get("accept") or []),
        "kind": entry.get("kind"),
        "technique_ids": techniques,
        "statements": statements,
    }
    texts = [statement["text"] for statement in statements]
    flags = anonymizer.attribution_flags(
        texts,
        paragraphs=[statement["paragraph"] for statement in statements],
        section_flags=[statement["section_attribution"] for statement in statements],
    )
    software = Counter(
        match.software_id
        for match in match_software_aliases(" ".join(texts), kb)
        if match.software_id
    )
    truth = {group_id, *row["accept"]} - {None}
    exclusive = sorted(
        software_id
        for software_id in software
        if truth
        and kb.software.get(software_id) is not None
        and set(kb.software[software_id].group_ids)
        and set(kb.software[software_id].group_ids) <= truth
    )
    stats = {
        "url": item["url"],
        "published": item["published"],
        "post_sha256": item["sha256"],
        "group_id": group_id,
        "statements": len(statements),
        "statements_in_attribution_sections": sum(
            1 for s in statements if s["section_attribution"]
        ),
        "dropped_markup": dict(sorted(parser.dropped.items())),
        "dropped_sections": dict(sorted(dropped_sections.items())),
        "headings": parser.heading_log,
        "technique_ids": techniques,
        "technique_ids_unresolved": unresolved,
        "software_detected": dict(sorted(software.items())),
        "truth_exclusive_software": exclusive,
        "strict_attribution_drops": [
            {"reason": flag, "text": text} for text, flag in zip(texts, flags, strict=True) if flag
        ],
    }
    return row, stats


def download_feed(url: str, destination: Path, *, timeout: float = 60.0) -> Path:
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "apt-classifier-benchmark/1.0"})
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    feed: Path,
    labels_path: Path,
    output: Path,
    *,
    kb: Any,
    raw_bundle: Path | None = None,
    exemplar_texts: frozenset[str] | None = None,
) -> dict[str, Any]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    items = read_feed(feed)
    anonymizer = ReportAnonymizer(kb, "strict")
    revoked = revoked_technique_map(raw_bundle)
    entries = [(entry, False) for entry in labels.get("reports", [])] + [
        ({**entry, "group_id": None}, True) for entry in labels.get("open_set", [])
    ]
    missing = [entry["slug"] for entry, _ in entries if entry["slug"] not in items]
    if missing:
        raise ValueError(f"slugs not in the feed: {missing}")
    for entry, open_set in entries:
        if not open_set and entry["group_id"] not in kb.groups:
            raise ValueError(f"{entry['slug']}: group {entry['group_id']} is not in the KB")
        for group_id in entry.get("accept") or []:
            if group_id not in kb.groups:
                raise ValueError(f"{entry['slug']}: accept {group_id} is not in the KB")
    procedures = procedure_sentence_texts(kb)
    citations = attack_citations(raw_bundle, [entry["slug"] for entry, _ in entries])
    output.mkdir(parents=True, exist_ok=True)
    labelled: list[dict[str, Any]] = []
    unlabelled: list[dict[str, Any]] = []
    per_report: dict[str, Any] = {}
    for entry, open_set in entries:
        row, stats = build_report(
            items[entry["slug"]], entry, kb=kb, anonymizer=anonymizer, revoked=revoked
        )
        normalized = [normalize_procedure_text(s["text"]) for s in row["statements"]]
        stats["leak_audit"] = {
            "attack_objects_citing_post": citations[entry["slug"]],
            "statements_equal_to_attack_procedures": sum(
                1 for text in normalized if text in procedures
            ),
            "statements_equal_to_exemplars": (
                sum(1 for text in normalized if text in exemplar_texts)
                if exemplar_texts is not None
                else None
            ),
            "truth_group_citations_in_activity_year": (
                group_citations_in_year(
                    raw_bundle, kb, entry["group_id"], int(entry["activity_year"])
                )
                if not open_set and entry.get("activity_year")
                else None
            ),
        }
        per_report[row["report_id"]] = stats
        (unlabelled if open_set else labelled).append(row)
    files = {"reports.jsonl": labelled, "reports_open_set.jsonl": unlabelled}
    hashes: dict[str, str] = {}
    for name, rows in files.items():
        path = output / name
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
        )
        hashes[name] = _sha256(path)
    manifest = {
        "source": FEED_URL,
        "licence": LICENCE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "feed_items": len(items),
        "feed_sha256": _sha256(feed),
        "attack_version": kb.version,
        "labels_sha256": _sha256(labels_path),
        "rules_sha256": {
            name: _sha256(REPO_ROOT / "scripts" / name)
            for name in ("prepare_elastic_reports.py", "report_anonymizer.py")
        },
        "outputs_sha256": hashes,
        "reports": len(labelled),
        "open_set": len(unlabelled),
        "statements": sum(len(row["statements"]) for row in labelled),
        "open_set_statements": sum(len(row["statements"]) for row in unlabelled),
        "per_report": per_report,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _exemplar_texts() -> frozenset[str] | None:
    try:
        from apt_classifier.config import Settings
        from apt_classifier.exemplars import ExemplarIndex

        index = ExemplarIndex.load(Settings().technique_exemplar_index_path)
    except Exception as error:  # the audit is optional; say why it is missing
        print(f"exemplar overlap audit skipped: {error}", file=sys.stderr)
        return None
    if index is None:
        return None
    return frozenset(normalize_procedure_text(text) for text in index.texts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, help="a saved feed.xml (default: the cached download)")
    parser.add_argument("--download", action="store_true", help="fetch the live feed first")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kb", type=Path, default=REPO_ROOT / "data" / "kb" / "attack_kb.json")
    parser.add_argument(
        "--raw-bundle", type=Path, default=REPO_ROOT / "data" / "raw" / "enterprise-attack.json"
    )
    parser.add_argument("--no-exemplar-audit", action="store_true")
    args = parser.parse_args(argv)
    if args.feed is not None and not args.feed.is_file():
        parser.error(
            f"--feed {args.feed} does not exist (the cached feed is never replaced silently)"
        )
    feed = args.feed or args.output / "feed.xml"
    if args.download or not feed.is_file():
        download_feed(FEED_URL, args.output / "feed.xml")
        feed = args.output / "feed.xml"
    kb = AttackKnowledgeBase.load(args.kb)
    manifest = prepare(
        feed,
        args.labels,
        args.output,
        kb=kb,
        raw_bundle=args.raw_bundle,
        exemplar_texts=None if args.no_exemplar_audit else _exemplar_texts(),
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "reports",
                    "statements",
                    "open_set",
                    "open_set_statements",
                    "outputs_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
