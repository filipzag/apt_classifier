"""Documented overlap between ATT&CK groups, derived from the knowledge base only.

Two groups are *related* when ATT&CK itself links them: one group's description names the
other (or one of its aliases), or they share software that only a handful of groups use.
Widely shared commodity tooling (Mimikatz, PsExec, Cobalt Strike) does not count. The
result is used to present a leading hypothesis together with the documented groups it
cannot be told apart from (Carbanak and FIN7, Sandworm Team and APT28), and to report a
cluster-aware top-1 in the attribution benchmark next to the strict one.

Names are matched as proper nouns (``name_pattern``): an all-capital vendor name (BITTER, ZINC,
NICKEL) must match exactly, and any other name keeps its capital first letter (a name that starts
in lowercase, like menuPass, may also start a sentence), so an ordinary word ("snake", "zinc",
"bitter") never links a group. Before 2026-10-02 the match ignored case, which
linked Tonto Team to BITTER through "Operation Bitter Biscuit" in Tonto Team's description.

``related_groups`` keeps one reason per pair (shared software first), so a pair that both shares
software and is named in a description reads as the software link there. ``description_links``
keeps only the description references: the documented-cluster gate names a pair that only
shares rare malware (Sandworm Team and APT38 both used KillDisk) only when the profiles nest
(overlap_cluster.py, docs 10.11).
"""

from __future__ import annotations

import re
from typing import Any

from apt_classifier.memo import ObjectMemo

RARE_SOFTWARE_MAX_USERS = 3
MIN_NAME_LENGTH = 4

_RELATED_CACHE: ObjectMemo[dict[str, dict[str, str]]] = ObjectMemo()
_DESCRIPTION_CACHE: ObjectMemo[dict[str, dict[str, str]]] = ObjectMemo()


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
    for group_id, other_id, reason in _description_references(groups):
        link(group_id, other_id, reason)
    return _RELATED_CACHE.put(kb, stamp, result)


def name_pattern(name: str) -> re.Pattern[str]:
    """A whole-word pattern for a group name or alias as a proper noun in prose.

    An all-capital name ("BITTER", "APT41", "BRONZE BUTLER") matches only as written. Any other
    name keeps a capital first letter and may differ in case after it ("Winnti group" for
    "Winnti Group"), so lowercase ordinary words never match ("snake" is not Turla's "Snake");
    a name that starts in lowercase may also start a sentence ("MenuPass" for "menuPass").
    """
    if name.isupper():
        body = re.escape(name)
    else:
        first = name[0]
        head = f"[{first}{first.upper()}]" if first.islower() else re.escape(first)
        body = head + (f"(?i:{re.escape(name[1:])})" if len(name) > 1 else "")
    return re.compile(rf"(?<![\w-]){body}(?![\w-])")


def _description_references(groups: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(group, other, reason) for every group whose description names another group."""
    patterns = {
        group_id: [name_pattern(name) for name in _group_names(group)]
        for group_id, group in groups.items()
    }
    references = []
    for group_id, group in groups.items():
        text = str(getattr(group, "description", "") or "")
        if not text:
            continue
        for other_id, other_patterns in patterns.items():
            if other_id == group_id:
                continue
            if any(pattern.search(text) for pattern in other_patterns):
                references.append(
                    (
                        group_id,
                        other_id,
                        f"{getattr(group, 'name', group_id)} description references "
                        f"{getattr(groups[other_id], 'name', other_id)}",
                    )
                )
    return references


def description_links(kb: Any) -> dict[str, dict[str, str]]:
    """Map each group id to ``{group_id: reason}`` for description references only; symmetric
    (either description naming the other counts); memoized per KB."""
    groups = getattr(kb, "groups", {}) or {}
    stamp = (str(getattr(kb, "version", "")), len(groups))
    cached = _DESCRIPTION_CACHE.get(kb, stamp)
    if cached is not None:
        return cached
    result: dict[str, dict[str, str]] = {group_id: {} for group_id in groups}
    for group_id, other_id, reason in _description_references(groups):
        result.setdefault(group_id, {}).setdefault(other_id, reason)
        result.setdefault(other_id, {}).setdefault(group_id, reason)
    return _DESCRIPTION_CACHE.put(kb, stamp, result)


def related_to(kb: Any, group_id: str) -> dict[str, str]:
    return dict(related_groups(kb).get(group_id, {}))
