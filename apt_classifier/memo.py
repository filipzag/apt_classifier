"""Small per-object memo used for knowledge-base derived tables.

The knowledge base is a slotted dataclass (no weak references), and several modules derive
tables from it that are expensive to rebuild (procedure sentences, technique document counts,
documented-related groups). Keying such a cache by ``id(kb)`` alone is unsafe: once a knowledge
base is garbage-collected its address can be reused by a new one with the same cheap stamp, and
the stale table would be served for the wrong object. This memo keeps a strong reference to the
object it cached for, so an address can never be reused while its entry exists, and bounds the
number of entries so per-fold knowledge bases in the benchmarks do not accumulate.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ObjectMemo(Generic[T]):
    def __init__(self, capacity: int = 4) -> None:
        self._entries: OrderedDict[int, tuple[Any, Any, T]] = OrderedDict()
        self._capacity = max(1, int(capacity))

    def get(self, owner: Any, stamp: Any) -> T | None:
        entry = self._entries.get(id(owner))
        if entry is None:
            return None
        cached_owner, cached_stamp, value = entry
        if cached_owner is not owner or cached_stamp != stamp:
            return None
        self._entries.move_to_end(id(owner))
        return value

    def put(self, owner: Any, stamp: Any, value: T) -> T:
        self._entries[id(owner)] = (owner, stamp, value)
        self._entries.move_to_end(id(owner))
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
