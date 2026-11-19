"""The ranker's corpus at one masking level, cached per store pin.

Store text masks the store's posts for actor names only, while a report at the behaviour level
also loses malware, tool and operation names, so the two sides read different text. Here each
post's statements go through the same anonymizer as the reports (``sets.load_reports``: names
replaced, then ``mask``), once per (pin, level), and are cached as gzipped JSON lines of
``[post key, actor, text]``.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vestige_core.util import DATA_DIR, require_memory

CACHE_DIR = DATA_DIR / "core"


def cache_path(pin: str | None, level: str) -> Path:
    return CACHE_DIR / f"rows_{pin or 'whole'}_{level}.jsonl.gz"


def masked_rows(
    members: Any, kb: Any, level: str, extra_names: Sequence[str] = ()
) -> list[tuple[str, str, str]]:
    """``(post, actor, text)`` for every labelled store statement at ``level``."""
    from scripts.report_anonymizer import ReportAnonymizer

    if level == "store":  # the store's own masking (names only), as store text reads it
        return [(r["key"], r["group"], r["text"]) for r in members.text_rows]
    masker = ReportAnonymizer(kb, level, extra_names=tuple(extra_names))
    by_post: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in members.text_rows:
        by_post[(r["key"], r["group"])].append(r["text"])
    out: list[tuple[str, str, str]] = []
    for n, ((key, actor), texts) in enumerate(sorted(by_post.items()), 1):
        if n % 100 == 0:
            require_memory()
        texts = [masker.replace_names(t) if masker.name_matches(t) else t for t in texts]
        texts, _ = masker.mask(texts)
        out.extend((key, actor, t) for t in texts if len(t.split()) >= 3)
    return out


def load_rows(
    members: Any, kb: Any, pin: str | None, level: str, extra_names: Sequence[str] = ()
) -> list[tuple[str, str, str]]:
    """Cached ``masked_rows``; the cache records the artifact fingerprint it was built from."""
    path = cache_path(pin, level)
    if path.is_file():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            if header.get("fingerprint") == members.fingerprint and header.get("level") == level:
                return [tuple(json.loads(line)) for line in handle]
    rows = masked_rows(members, kb, level, extra_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"fingerprint": members.fingerprint, "level": level, "pin": pin, "rows": len(rows)}
            )
            + "\n"
        )
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows
