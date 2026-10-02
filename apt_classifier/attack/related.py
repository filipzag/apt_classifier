"""Documented overlap between ATT&CK groups, derived from the knowledge base only.

Two groups are *related* when ATT&CK itself links them: one group's description names the
other (or one of its aliases), or they share software that only a handful of groups use.
Widely shared commodity tooling (Mimikatz, PsExec, Cobalt Strike) does not count. The
result is used to present a leading hypothesis together with the documented groups it
cannot be told apart from (Carbanak and FIN7, Sandworm Team and APT28), and to report a
cluster-aware top-1 in the attribution benchmark next to the strict one.
"""

from __future__ import annotations

import re
from typing import Any

from apt_classifier.memo import ObjectMemo

RARE_SOFTWARE_MAX_USERS = 3
MIN_NAME_LENGTH = 4

_RELATED_CACHE: ObjectMemo[dict[str, dict[str, str]]] = ObjectMemo()


def _group_names(group: Any) -> list[str]:
    names = [str(getattr(group, "name", "") or "")]
    names.extend(str(alias) for alias in getattr(group, "aliases", ()) or ())
    return [name.strip() for name in names if name and len(name.strip()) >= MIN_NAME_LENGTH]


def related_groups(
    kb: Any, *, rare_software_max_users: int = RARE_SOFTWARE_MAX_USERS
) -> dict[str, dict[str, str]]:
    """Map each group id to ``{related_group_id: reason}``; symmetric; memoized per KB."""
    groups = getattr(kb, "groups", {}) or {}
    # the parameter is part of the stamp: a call with another user limit must not be served the
    # table cached for the default one
    stamp = (str(getattr(kb, "version", "")), len(groups), int(rare_software_max_users))
    cached = _RELATED_CACHE.get(kb, stamp)
    if cached is not None:
        return cached
    result: dict[str, dict[str, str]] = {group_id: {} for group_id in groups}

    def link(a: str, b: str, reason: str) -> None:
        if a == b:
            return
        result.setdefault(a, {}).setdefault(b, reason)
        result.setdefault(b, {}).setdefault(a, reason)

    users: dict[str, set[str]] = {}
    for group_id, group in groups.items():
        for software_id in getattr(group, "software_ids", ()) or ():
            users.setdefault(str(software_id), set()).add(group_id)
    software = getattr(kb, "software", {}) or {}
    for software_id, group_ids in users.items():
        entry = software.get(software_id)
        if str(getattr(entry, "software_type", "") or "") == "tool":
            continue  # a shared admin utility says nothing about who used it
        if 2 <= len(group_ids) <= rare_software_max_users:
            name = str(getattr(software.get(software_id), "name", "") or software_id)
            ordered = sorted(group_ids)
            for index, first in enumerate(ordered):
                for second in ordered[index + 1 :]:
                    link(first, second, f"shared rarely used software {name} ({software_id})")
    patterns = {
        group_id: [
            re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.IGNORECASE)
            for name in _group_names(group)
        ]
        for group_id, group in groups.items()
    }
    for group_id, group in groups.items():
        text = str(getattr(group, "description", "") or "")
        if not text:
            continue
        for other_id, other_patterns in patterns.items():
            if other_id == group_id:
                continue
            if any(pattern.search(text) for pattern in other_patterns):
                link(
                    group_id,
                    other_id,
                    f"{getattr(group, 'name', group_id)} description references "
                    f"{getattr(groups[other_id], 'name', other_id)}",
                )
    return _RELATED_CACHE.put(kb, stamp, result)


def related_to(kb: Any, group_id: str) -> dict[str, str]:
    return dict(related_groups(kb).get(group_id, {}))
