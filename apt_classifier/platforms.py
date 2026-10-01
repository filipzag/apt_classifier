"""Target operating systems for technique mapping.

ATT&CK lists the platforms each technique applies to. When the analyst states which operating
systems the affected hosts run (the case's Target OS) or an observation names one, techniques
documented only for other operating systems are not retrieved for that observation: activity on
a Linux server does not map to Windows Scheduled Task, Registry Run Keys or PowerShell.

Only host operating systems constrain anything. A technique that also lists a platform the
operating system says nothing about (IaaS, SaaS, Office Suite, Identity Provider, Network
Devices, Containers, PRE) is always kept. Structured Windows event logs keep their stricter rule:
the technique must list Windows (``PlatformScope.strict``).

Precedence per observation: a platform recorded by the log source, then operating systems named
in the observation's own text (when enabled), then the case's Target OS, else no constraint.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# ATT&CK host platforms, in display order.
HOST_PLATFORMS: tuple[str, ...] = ("Windows", "Linux", "macOS", "ESXi")
_HOST_FOLDED = frozenset(name.casefold() for name in HOST_PLATFORMS)
_ALIASES = {
    **{name.casefold(): name for name in HOST_PLATFORMS},
    "mac os": "macOS",
    "mac os x": "macOS",
    "os x": "macOS",
    "osx": "macOS",
    "esx": "ESXi",
    "vmware esxi": "ESXi",
}

# Where the case-level Target OS is stored: the ledger's free-form metadata, so case files stay
# readable by code that predates the field.
LEDGER_KEY = "target_platforms"

# A browser user-agent string names the client's operating system, not the affected host.
_USER_AGENT = re.compile(r"Mozilla/\d[^()\n]{0,40}\([^)\n]{0,200}\)")
_MENTIONS: dict[str, re.Pattern[str]] = {
    # Case-sensitive: "maintenance windows" is not the operating system.
    "Windows": re.compile(r"\bWindows\b|\bWin(?:32|64)\b"),
    # Kali is the attacker's own distribution, not the target.
    "Linux": re.compile(
        r"(?<!Kali )\bLinux\b|\b(?:Ubuntu|Debian|CentOS|RHEL|Fedora|SUSE)\b", re.IGNORECASE
    ),
    "macOS": re.compile(r"\bmac ?OS\b|\bOS ?X\b", re.IGNORECASE),
    "ESXi": re.compile(r"\bESXi?\b", re.IGNORECASE),
}


def normalize_platforms(values: Any, *, strict: bool = False) -> list[str]:
    """Canonical host OS names in :data:`HOST_PLATFORMS` order.

    Accepts a comma/semicolon/newline separated string or an iterable. Unknown values are
    dropped, or rejected with ``ValueError`` when ``strict`` (API input).
    """
    if values is None:
        return []
    raw = re.split(r"[,;\n]+", values) if isinstance(values, str) else list(values)
    found: set[str] = set()
    unknown: list[str] = []
    for item in raw:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        canonical = _ALIASES.get(text.casefold())
        if canonical is None:
            unknown.append(text)
        else:
            found.add(canonical)
    if unknown and strict:
        raise ValueError(
            f"unknown operating system(s): {', '.join(unknown)}; use {', '.join(HOST_PLATFORMS)}"
        )
    return [name for name in HOST_PLATFORMS if name in found]


def detect_platforms(text: str) -> list[str]:
    """Host operating systems an observation names explicitly (empty when none)."""
    cleaned = _USER_AGENT.sub(" ", text or "")
    return [name for name in HOST_PLATFORMS if _MENTIONS[name].search(cleaned)]


def ledger_platforms(ledger: Any) -> list[str]:
    """The case's Target OS (empty = unknown: no constraint)."""
    metadata = getattr(ledger, "metadata", None) or {}
    return normalize_platforms(metadata.get(LEDGER_KEY) if isinstance(metadata, Mapping) else None)


def set_ledger_platforms(ledger: Any, platforms: Any) -> list[str]:
    """Store the case's Target OS on the ledger; an empty value removes it."""
    values = normalize_platforms(platforms)
    metadata = dict(getattr(ledger, "metadata", None) or {})
    if values:
        metadata[LEDGER_KEY] = values
    else:
        metadata.pop(LEDGER_KEY, None)
    ledger.metadata = metadata
    return values


def _listed(platforms: Any) -> list[str]:
    if not platforms:
        return []
    values = platforms.split(",") if isinstance(platforms, str) else list(platforms)
    return [str(value).strip() for value in values if str(value).strip()]


def _folded(platforms: Any) -> set[str]:
    return {value.casefold() for value in _listed(platforms)}


@dataclass(frozen=True)
class PlatformScope:
    """Operating systems an observation is about, and where that came from.

    ``source`` is ``log`` (recorded by the log source), ``text`` (named in the observation) or
    ``case`` (the case's Target OS).
    """

    platforms: tuple[str, ...]
    source: str
    strict: bool = False

    def allows(self, technique_platforms: Any) -> bool:
        values = _folded(technique_platforms)
        if not values:
            return True
        if values & {platform.casefold() for platform in self.platforms}:
            return True
        if self.strict:
            return False
        # Also documented for a non-host platform: the operating system says nothing about it.
        return not values <= _HOST_FOLDED

    def relevant(self, technique_platforms: Any) -> tuple[str, ...]:
        """A technique's platforms still worth searching under this scope: the scope's operating
        systems and the platforms an operating system says nothing about (cloud, identity, ...)."""
        allowed = {platform.casefold() for platform in self.platforms}
        return tuple(
            value
            for value in _listed(technique_platforms)
            if value.casefold() in allowed or value.casefold() not in _HOST_FOLDED
        )

    def to_dict(self) -> dict[str, Any]:
        return {"platforms": list(self.platforms), "source": self.source}


def observation_scope(
    observation: Any, case_platforms: Iterable[str] = (), *, from_text: bool = False
) -> PlatformScope | None:
    """The platform constraint for one observation (``None`` = unconstrained)."""
    metadata = getattr(observation, "metadata", None) or {}
    explicit = metadata.get("platform") if isinstance(metadata, Mapping) else None
    if explicit:
        values = [explicit] if isinstance(explicit, str) else list(explicit)
        return PlatformScope(tuple(str(value) for value in values), "log", strict=True)
    source = str(getattr(getattr(observation, "provenance", None), "source", ""))
    if getattr(observation, "kind", "") == "winlog" or source == "winlog":
        return PlatformScope(("Windows",), "log", strict=True)
    if from_text:
        named = detect_platforms(str(getattr(observation, "text", "") or ""))
        if named:
            return PlatformScope(tuple(named), "text")
    case = tuple(normalize_platforms(list(case_platforms)))
    return PlatformScope(case, "case") if case else None
