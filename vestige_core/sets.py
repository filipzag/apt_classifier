"""Report sets (``data/benchmarks/<set>/{reports,open_set}.jsonl``) read at one masking level.

The text a report yields equals what ``scripts/attribution_benchmark.py`` gives for
``--report-observations 0 --anonymize <level>`` (``load_report_folds`` with the level's
anonymizer dropping attribution prose, then ``anonymize_report_folds``), without importing that
module (it pulls in torch, sentence-transformers and chromadb, about 1 GB). Checked by
``tests/test_vestige_core.py`` and by the store-text reproduction on fresh2.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vestige_core.util import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apt_classifier.attack.parse import strip_attack_citations  # noqa: E402
from apt_classifier.knowledge_store.listing import normalise_url  # noqa: E402
from scripts.report_anonymizer import (  # noqa: E402
    LEVELS,
    ReportAnonymizer,
    actor_names,
    evidence_name_pattern,
)

_ARTICLE_BEFORE_ADVERSARY = re.compile(r"\b(?:an?|the)\s+(the adversary)", re.IGNORECASE)


@dataclass(frozen=True)
class Report:
    report_id: str
    group_id: str | None  # ATT&CK group; None for an open-set report
    actor: str | None  # the open-set label (misp:<name>, code:<code>), if any
    source: str
    published: str
    as_of: str  # the store is read strictly before this day
    store_keys: tuple[str, ...]
    raw_statements: tuple[str, ...]  # as written (names masked by the set builder), for copy hiding
    statements: tuple[str, ...]  # at the level
    accept: frozenset[str] = frozenset()

    @property
    def known(self) -> bool:
        return self.group_id is not None


def _sanitize(text: str, pattern: re.Pattern[str]) -> str:
    """``vectorstore._sanitize_procedure`` (names level), copied to keep chromadb out."""
    value = strip_attack_citations(text).strip()
    value = pattern.sub(lambda m: "the adversary" + (m.group("possessive") or ""), value)
    value = _ARTICLE_BEFORE_ADVERSARY.sub(r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def load_reports(
    path: str | Path,
    kb: Any,
    level: str,
    *,
    extra_names: Sequence[str] = (),
) -> list[Report]:
    """Every report of one jsonl file at ``level``; reports left with no statement are skipped."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    dropper = ReportAnonymizer(kb, level)
    masker = (
        ReportAnonymizer(kb, level, extra_names=tuple(extra_names)) if level != "names" else None
    )
    pattern = evidence_name_pattern(actor_names(kb))
    out: list[Report] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items = [i for i in row.get("statements", []) if len(str(i.get("text", "")).split()) >= 3]
        if items:
            paragraphs = (
                [i.get("paragraph") for i in items]
                if any("paragraph" in i for i in items)
                else None
            )
            flags = dropper.attribution_flags(
                [str(i["text"]) for i in items],
                paragraphs=paragraphs,
                section_flags=[bool(i.get("section_attribution")) for i in items],
            )
            items = [i for i, flag in zip(items, flags, strict=True) if not flag]
        texts = [str(i["text"]).strip() for i in items]
        if not texts:
            continue
        if masker is None:
            if pattern is not None:
                texts = [_sanitize(t, pattern) if pattern.search(t) else t for t in texts]
        else:
            texts = [masker.replace_names(t) if masker.name_matches(t) else t for t in texts]
            texts, _ = masker.mask(texts)
        own = row.get("store_key") or (normalise_url(str(row["url"])) if row.get("url") else "")
        group = row.get("group_id")
        out.append(
            Report(
                report_id=str(row.get("report_id") or ""),
                group_id=str(group) if group is not None else None,
                actor=str(row["actor"]) if row.get("actor") else None,
                source=str(row.get("source") or ""),
                published=str(row.get("published") or "")[:10],
                as_of=str(row.get("available_from") or row.get("published") or "")[:10],
                store_keys=(str(own),) if own else (),
                raw_statements=tuple(str(i.get("text", "")) for i in row.get("statements", [])),
                statements=tuple(texts),
                accept=frozenset(row.get("accept") or ()),
            )
        )
    return out


def load_set(
    folder: str | Path, kb: Any, level: str, *, extra_names: Sequence[str] = ()
) -> list[Report]:
    """Known and open-set reports of one set folder."""
    folder = Path(folder)
    reports = load_reports(folder / "reports.jsonl", kb, level, extra_names=extra_names)
    open_path = folder / "open_set.jsonl"
    if open_path.is_file():
        reports += load_reports(open_path, kb, level, extra_names=extra_names)
    return reports
