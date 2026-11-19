"""Reading a research post: the prose lines of its article body, with heading path and paragraph.

Moved unchanged from the report-set builders (``scripts/prepare_vendor_reports.py`` and
``scripts/prepare_elastic_reports.py`` re-export these names), so the store reads a post exactly as
the vendor and fresh sets did. Non-narrative sections (ATT&CK tables, detections, indicators,
references) and page furniture (related posts, share bars, newsletters) are dropped by heading.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

# a block whose text is mostly link text is navigation, not prose
MAX_LINK_SHARE = 0.6
# page furniture after the article (label-blind, the same for every page)
_FURNITURE_HEADING_RE = re.compile(
    r"^(?:(?:related|recent|more|popular|latest|similar|recommended|further)\b.*"
    r"|about the authors?|share(?: this.*)?|subscribe.*|.*newsletter.*|tags?|topics"
    r"|comments?|leave a (?:reply|comment)|contact us|get in touch|follow us.*"
    r"|table of contents|in this (?:article|post|blog))$"
)
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
_REMOVE_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "button",
    "pre",
    "table",
    "template",
)
_LINE_TAGS = ("p", "li", "blockquote", "figcaption", "dd")
_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
_FIGURE_LINE_RE = re.compile(r"^(?:Figure|Fig\.|Table|Image)\s*\d+\b", re.IGNORECASE)


def _text(element: Any) -> str:
    return " ".join(element.get_text(" ").split())


def _link_share(element: Any, text: str) -> float:
    """Share of a block's text that is link text (menus, tag clouds, related-post cards)."""
    if not text:
        return 0.0
    return sum(len(_text(link)) for link in element.find_all("a")) / len(text)


def parse_page(html: str) -> dict[str, Any]:
    """Prose lines of the article body with their heading path and paragraph id."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = _text(soup.title) if soup.title else ""
    for tag in soup(list(_REMOVE_TAGS)):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"^wm-")):  # the Wayback Machine toolbar
        tag.decompose()
    body = soup.body or soup

    def prose(element: Any) -> int:
        return sum(len(_text(item)) for item in element.find_all("p"))

    containers = [*soup.find_all(["article", "main"]), *soup.find_all(attrs={"role": "main"})]
    container = max(containers, key=prose) if containers else body
    # a small teaser <article> on a page whose story sits elsewhere
    if prose(container) < 0.5 * prose(body):
        container = body
    lines: list[dict[str, Any]] = []
    headings: list[tuple[int, str]] = []
    dropped: Counter[str] = Counter()
    paragraph = 0
    for element in container.find_all([*_HEADING_TAGS, *_LINE_TAGS]):
        if element.name in _HEADING_TAGS:
            text = _text(element).replace("®", "")
            level = _HEADING_TAGS[element.name]
            while headings and headings[-1][0] >= level:
                headings.pop()
            if text:
                headings.append((level, text))
            continue
        # leaf blocks only (a list item holding paragraphs is read through them)
        if element.find(list(_LINE_TAGS)):
            continue
        text = _text(element)
        if not text:
            continue
        if _FIGURE_LINE_RE.match(text):
            dropped["figure_lines"] += 1
            continue
        if _link_share(element, text) > MAX_LINK_SHARE:
            dropped["link_lines"] += 1
            continue
        path = tuple(heading for _, heading in headings)
        section = next(
            (
                heading
                for heading in path
                if _DROP_HEADING_RE.match(heading.lower())
                or _FURNITURE_HEADING_RE.match(heading.lower())
            ),
            None,
        )
        if section is not None:
            dropped[f"section:{section}"] += 1
            continue
        paragraph += 1
        lines.append({"text": text, "path": path, "paragraph": paragraph})
    return {"title": title, "lines": lines, "dropped": dict(sorted(dropped.items()))}


def site_of(url: str) -> str:
    """The site a page belongs to (last two host labels; an archived page counts as its origin)."""
    if "web.archive.org/web/" in url:
        url = re.sub(r"^.*?web\.archive\.org/web/[^/]+/", "", url)
    host = urlparse(url).netloc.lower().split(":")[0]
    return ".".join(host.split(".")[-2:])


def repeated_lines(pages: dict[tuple[str, int], dict[str, Any]]) -> dict[str, frozenset[str]]:
    """Per site, the lines found on two or more of its downloaded pages (site chrome: cookie
    notices, sidebars, "related posts" cards, sign-up prompts)."""
    seen: dict[str, Counter[str]] = {}
    for page in pages.values():
        seen.setdefault(page["site"], Counter()).update({line["text"] for line in page["lines"]})
    return {
        site: frozenset(text for text, count in counts.items() if count >= 2)
        for site, counts in seen.items()
    }


def statement_texts(
    lines: Iterable[dict[str, Any]], chrome: frozenset[str] = frozenset()
) -> list[str]:
    """The page's statements: its non-chrome lines split into sentences of three or more words
    (the local splitter every report set uses, ``evidence.local_extract``)."""
    from apt_classifier.evidence.local_extract import local_observations

    kept = [line["text"] for line in lines if line["text"] not in chrome]
    return [observation.text for observation in local_observations("\n".join(kept))]
