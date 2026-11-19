"""Actor names from before a date (prereg ``knowledge_store_decisive_2026-11-04``,
``store_arm.names_before_the_window`` and ``development.sets``).

A store fact counts for an ATT&CK group only through a name that pointed to that group before the
reports it serves: the group's own name in the current release, its name and aliases in an older
release, or a synonym in an older MISP actor list. A group ATT&CK added later keeps only its
current name. Software names are not pinned: a post that names a malware states it itself.

Two pins, each a copy of the store whose facts were read again with its lists
(``scripts/knowledge_store.py extract --pin``):

- ``before_2025``: ATT&CK 16.1 (2024-11-12) and the MISP actor list at commit 3735fd37
  (2024-12-24), for the decisive set (2025-01-01 to 2026-02-28) and for development reports
  published before 2026-03-01;
- ``before_2026_03``: ATT&CK 18.1 (2025-11-13) and MISP commit 03b461ff (2026-02-16), for
  development reports published from 2026-03-01 on (both fresh sets, three vendor reports).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class Pin:
    name: str
    attack_bundle: str  # relative to the data directory
    attack_release: str
    misp_dir: str  # relative to the data directory; holds threat-actor.json
    misp_commit: str


PINS: dict[str, Pin] = {
    "before_2025": Pin(
        "before_2025",
        "raw/enterprise-attack-16.1.json",
        "16.1",
        "raw/misp_galaxy_3735fd37",
        "3735fd37d5ccc51e0f1430e97b7701755da20ec1",
    ),
    "before_2026_03": Pin(
        "before_2026_03",
        "raw/enterprise-attack-18.1.json",
        "18.1",
        "raw/misp_galaxy_03b461ff",
        "03b461ff837848a475ad26de68c6a4ccf27f3067",
    ),
}
# development reports published from this day on read the later pin
LATER_PIN_FROM = date(2026, 3, 1)


def pin_for(published: date | str) -> str:
    """The pin a report published on ``published`` reads."""
    day = date.fromisoformat(published) if isinstance(published, str) else published
    return "before_2026_03" if day >= LATER_PIN_FROM else "before_2025"


def pin_root(store_root: str | Path, pin: str) -> Path:
    """Where a pin's copy of the store lives (beside the store, local like it)."""
    return Path(store_root) / "pins" / pin


def bundle_group_names(path: str | Path) -> dict[str, tuple[str, ...]]:
    """``G####`` -> the name and aliases of each intrusion set of a STIX bundle that is neither
    revoked nor deprecated."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    names: dict[str, tuple[str, ...]] = {}
    for item in bundle.get("objects", []):
        if item.get("type") != "intrusion-set" or item.get("revoked"):
            continue
        if item.get("x_mitre_deprecated"):
            continue
        group_id = next(
            (
                ref.get("external_id")
                for ref in item.get("external_references", []) or []
                if ref.get("source_name") == "mitre-attack"
            ),
            None,
        )
        if not group_id:
            continue
        found = [item.get("name") or "", *(item.get("aliases") or ())]
        names[group_id] = tuple(dict.fromkeys(name.strip() for name in found if name.strip()))
    return names


class PinnedNames:
    """The knowledge base as the actor dictionary sees it under a pin: every current group keeps
    its id and current name, and its aliases are the names the older release gave it. Everything
    else (software, techniques) is the current knowledge base's."""

    def __init__(self, kb: Any, older: dict[str, tuple[str, ...]]) -> None:
        self._kb = kb
        self.groups = {
            group_id: SimpleNamespace(
                name=group.name,
                aliases=tuple(name for name in older.get(group_id, ()) if name != group.name),
            )
            for group_id, group in kb.groups.items()
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kb, name)


def pinned_names(kb: Any, data_dir: str | Path, pin: str) -> tuple[PinnedNames, dict[str, Any]]:
    """The pinned view of ``kb`` and the pin's MISP actor list."""
    spec = PINS[pin]
    root = Path(data_dir)
    older = bundle_group_names(root / spec.attack_bundle)
    misp = json.loads((root / spec.misp_dir / "threat-actor.json").read_text(encoding="utf-8"))
    return PinnedNames(kb, older), misp
