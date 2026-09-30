"""Sign-in behaviour of the API (api/auth.py) with the Google exchange faked."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.app import create_app
from apt_classifier.config import Settings
from apt_classifier.service import CaseNotFound
from apt_classifier.users import UserStore
from tests.test_api import StubService

HEADERS = {"X-Requested-With": "APTAA"}


def google_settings(**overrides: Any) -> Settings:
    base = dict(
        auth_mode="google",
        google_client_id="client-id",
        google_client_secret=SecretStr("client-secret"),
        auth_secret_key=SecretStr("test-secret-key"),
        auth_allowed_domains="example.com",
        auth_allowed_emails="guest@partner.org",
        auth_admin_emails="boss@example.com",
        auth_session_hours=1,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


class RecordingService(StubService):
    """Stub with an owned case so delete permissions can be exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.deleted: list[str] = []
        self.owner = {"id": "owner-1", "name": "Owner", "email": "owner@example.com"}

    def get(self, case_id: str) -> tuple[Any, Any]:
        if case_id != "owned":
            raise CaseNotFound(case_id)
        from apt_classifier.service import CaseMeta

        return object(), CaseMeta(case_id="owned", name="Owned", owner=self.owner)

    def delete(self, case_id: str) -> bool:
        self.deleted.append(case_id)
        return True


def fake_exchange(profile: dict[str, Any]):
    async def _authorize_access_token(request: Any) -> dict[str, Any]:
        return {"userinfo": profile}

    return _authorize_access_token


@pytest.fixture
def google_app(tmp_path):
    users = UserStore(tmp_path / "users.db")
    service = RecordingService()
    app = create_app(service, serve_web=False, settings=google_settings(), users=users)  # type: ignore[arg-type]
    client = TestClient(app, base_url="http://testserver")
    return app, client, users, service


def sign_in(app: Any, client: TestClient, monkeypatch, profile: dict[str, Any]) -> Any:
    google = app.state.auth.oauth.create_client("google")
    monkeypatch.setattr(google, "authorize_access_token", fake_exchange(profile))
    return client.get("/auth/callback?code=x&state=y", follow_redirects=False)


def test_off_mode_runs_as_local_admin_and_needs_no_header(tmp_path):
    app = create_app(
        StubService(), serve_web=False, settings=Settings(_env_file=None, auth_mode="off")
    )  # type: ignore[arg-type,call-arg]
    client = TestClient(app)
    me = client.get("/api/me").json()
    assert me["user"]["id"] == "local" and me["user"]["role"] == "admin"
    assert me["auth"] == {"mode": "off", "provider": None, "allowlist_configured": False}
    assert client.get("/auth/login", follow_redirects=False).status_code == 302
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/api/users").json()[0]["id"] == "local"


def test_google_mode_requires_a_session_and_the_request_header(google_app):
    _, client, _, _ = google_app
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/cases").status_code == 401
    assert client.get("/api/auth").json() == {
        "mode": "google",
        "provider": "google",
        "allowlist_configured": True,
    }
    login = client.get("/auth/login", follow_redirects=False)
    assert login.status_code == 302 and "accounts.google.com" in login.headers["location"]


def test_sign_in_creates_users_with_roles_and_enforces_the_allowlist(google_app, monkeypatch):
    app, client, users, _ = google_app
    first = sign_in(
        app,
        client,
        monkeypatch,
        {"email": "ann@example.com", "email_verified": True, "name": "Ann", "sub": "1"},
    )
    assert first.status_code == 302 and first.headers["location"] == "/"
    me = client.get("/api/me").json()
    assert me["user"]["email"] == "ann@example.com" and me["user"]["role"] == "admin"  # first user
    assert client.post("/auth/logout", headers=HEADERS).status_code == 204
    assert client.get("/api/me").status_code == 401

    rejected = sign_in(
        app, client, monkeypatch, {"email": "mallory@evil.net", "email_verified": True}
    )
    assert rejected.headers["location"] == "/login?error=not_allowed"
    unverified = sign_in(
        app, client, monkeypatch, {"email": "bob@example.com", "email_verified": False}
    )
    assert unverified.headers["location"] == "/login?error=unverified"
    assert users.count() == 1

    guest = sign_in(
        app,
        client,
        monkeypatch,
        {"email": "guest@partner.org", "email_verified": True, "name": "Guest"},
    )
    assert guest.status_code == 302
    assert client.get("/api/me").json()["user"]["role"] == "analyst"
    assert client.get("/api/users").status_code == 403  # analysts cannot manage users
    client.post("/auth/logout", headers=HEADERS)

    boss = sign_in(
        app,
        client,
        monkeypatch,
        {"email": "boss@example.com", "email_verified": True, "name": "Boss"},
    )
    assert boss.status_code == 302
    assert client.get("/api/me").json()["user"]["role"] == "admin"  # AUTH_ADMIN_EMAILS
    assert {u["email"] for u in client.get("/api/users").json()} == {
        "ann@example.com",
        "guest@partner.org",
        "boss@example.com",
    }


def test_mutations_need_the_header_and_admins_manage_users(google_app, monkeypatch):
    app, client, users, service = google_app
    sign_in(
        app,
        client,
        monkeypatch,
        {"email": "ann@example.com", "email_verified": True, "name": "Ann"},
    )
    guest = users.upsert_login(email="guest@partner.org", name="Guest")
    # a state change without the custom header is refused even with a valid session
    assert client.patch(f"/api/users/{guest.id}", json={"role": "admin"}).status_code == 403
    promoted = client.patch(f"/api/users/{guest.id}", json={"role": "admin"}, headers=HEADERS)
    assert promoted.status_code == 200 and promoted.json()["role"] == "admin"
    me_id = client.get("/api/me").json()["user"]["id"]
    assert (
        client.patch(f"/api/users/{me_id}", json={"role": "analyst"}, headers=HEADERS).status_code
        == 400
    )
    assert (
        client.patch(f"/api/users/{me_id}", json={"active": False}, headers=HEADERS).status_code
        == 400
    )
    demoted = client.patch(
        f"/api/users/{guest.id}", json={"role": "analyst", "active": False}, headers=HEADERS
    )
    assert demoted.json() == {**demoted.json(), "role": "analyst", "active": False}
    assert (
        client.patch("/api/users/nope", json={"role": "admin"}, headers=HEADERS).status_code == 404
    )
    assert (
        client.patch(f"/api/users/{guest.id}", json={"role": "root"}, headers=HEADERS).status_code
        == 400
    )
    # a deactivated user loses the session on the next request
    users.set_active(me_id, False) if users.admin_count() > 1 else None


def test_case_delete_is_limited_to_owner_or_admin(google_app, monkeypatch):
    app, client, users, service = google_app
    # first user (admin) may delete anyone's case
    sign_in(
        app,
        client,
        monkeypatch,
        {"email": "ann@example.com", "email_verified": True, "name": "Ann"},
    )
    assert client.delete("/api/cases/owned", headers=HEADERS).status_code == 204
    client.post("/auth/logout", headers=HEADERS)
    # an analyst who is not the owner may not
    sign_in(
        app,
        client,
        monkeypatch,
        {"email": "guest@partner.org", "email_verified": True, "name": "Guest"},
    )
    assert client.delete("/api/cases/owned", headers=HEADERS).status_code == 403
    # the owner may
    owner = users.upsert_login(email="owner@example.com", name="Owner")
    service.owner = {"id": owner.id, "name": owner.name, "email": owner.email}
    client.post("/auth/logout", headers=HEADERS)
    sign_in(
        app,
        client,
        monkeypatch,
        {"email": "owner@example.com", "email_verified": True, "name": "Owner"},
    )
    assert client.delete("/api/cases/owned", headers=HEADERS).status_code == 204
    assert service.deleted == ["owned", "owned"]


def test_google_mode_refuses_to_start_without_an_allowlist(tmp_path):
    with pytest.raises(RuntimeError, match="allowlist"):
        create_app(
            StubService(),  # type: ignore[arg-type]
            serve_web=False,
            settings=google_settings(auth_allowed_domains="", auth_allowed_emails=""),
            users=UserStore(tmp_path / "u.db"),
        )
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        create_app(
            StubService(),  # type: ignore[arg-type]
            serve_web=False,
            settings=google_settings(google_client_id=None),
            users=UserStore(tmp_path / "v.db"),
        )
