"""User store and sign-in policy (apt_classifier/users.py)."""

from __future__ import annotations

import pytest

from apt_classifier.users import LOCAL_USER, UserStore, email_allowed, parse_list


def test_parse_list_and_allowlist():
    assert parse_list(" Example.com, @Other.org;third.net ") == {
        "example.com",
        "other.org",
        "third.net",
    }
    assert parse_list(None) == set() and parse_list("") == set()
    domains, emails = {"example.com"}, {"guest@partner.org"}
    assert email_allowed("Ann@Example.com", domains=domains, emails=emails)
    assert email_allowed("guest@partner.org", domains=domains, emails=emails)
    assert not email_allowed("ann@partner.org", domains=domains, emails=emails)
    assert not email_allowed("", domains=domains, emails=emails)
    assert not email_allowed("not-an-address", domains=domains, emails=emails)
    # no dot-tricks: the domain is the exact suffix after the last @
    assert not email_allowed("ann@example.com.evil.net", domains=domains, emails=emails)


def test_user_store_first_user_is_admin_and_admin_list_promotes(tmp_path):
    store = UserStore(tmp_path / "users.db")
    assert store.count() == 0
    first = store.upsert_login(email="First@Example.com", name="First", subject="s1")
    assert first.role == "admin" and first.active and first.email == "first@example.com"
    second = store.upsert_login(email="second@example.com", name="Second")
    assert second.role == "analyst"
    promoted = store.upsert_login(
        email="second@example.com", name="Second Renamed", admin_emails={"second@example.com"}
    )
    assert (
        promoted.id == second.id and promoted.role == "admin" and promoted.name == "Second Renamed"
    )
    assert store.get_by_email("SECOND@example.com") is not None
    assert store.get("nope") is None and store.get(None) is None
    assert [u.email for u in store.list()] == ["first@example.com", "second@example.com"]
    assert store.admin_count() == 2


def test_user_store_role_and_active_changes(tmp_path):
    store = UserStore(tmp_path / "users.db")
    user = store.upsert_login(email="a@example.com")
    assert store.set_role(user.id, "analyst").role == "analyst"
    with pytest.raises(ValueError):
        store.set_role(user.id, "root")
    with pytest.raises(KeyError):
        store.set_role("missing", "admin")
    assert store.set_active(user.id, False).active is False
    assert store.admin_count() == 0 and store.admin_count(active_only=False) == 0
    # a deactivated user keeps its row and comes back with the same id on sign-in
    again = store.upsert_login(email="a@example.com")
    assert again.id == user.id and again.active is False
    store.close()


def test_local_user_is_an_admin_without_identity():
    assert LOCAL_USER.is_admin and LOCAL_USER.email == "" and LOCAL_USER.to_dict()["id"] == "local"
