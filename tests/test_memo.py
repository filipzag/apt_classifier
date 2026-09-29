"""Per-object memo used for knowledge-base derived tables (apt_classifier/memo.py)."""

from __future__ import annotations

from types import SimpleNamespace

from apt_classifier.memo import ObjectMemo


def test_memo_serves_only_the_same_object_with_the_same_stamp():
    memo: ObjectMemo[dict[str, int]] = ObjectMemo(capacity=2)
    first = SimpleNamespace(version="a")
    value = memo.put(first, ("a", 1), {"x": 1})
    assert memo.get(first, ("a", 1)) is value
    assert memo.get(first, ("a", 2)) is None  # stamp changed
    # a different object that happens to share the stamp is never served the cached table
    other = SimpleNamespace(version="a")
    assert memo.get(other, ("a", 1)) is None


def test_memo_is_bounded_and_keeps_recent_entries():
    memo: ObjectMemo[int] = ObjectMemo(capacity=2)
    owners = [SimpleNamespace(i=i) for i in range(3)]
    for index, owner in enumerate(owners):
        memo.put(owner, index, index)
    assert len(memo) == 2
    assert memo.get(owners[0], 0) is None  # evicted
    assert memo.get(owners[2], 2) == 2
    memo.get(owners[1], 1)  # touch -> most recent
    memo.put(SimpleNamespace(), 9, 9)
    assert memo.get(owners[1], 1) == 1 and memo.get(owners[2], 2) is None
    memo.clear()
    assert len(memo) == 0
