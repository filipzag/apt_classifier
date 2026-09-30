"""Sign-in for the web app: Google (OpenID Connect) sessions, or one local analyst.

``install_auth`` reads the settings and either leaves the app open (mode ``off``: every
request runs as ``LOCAL_USER``, an admin) or installs a signed session cookie plus the Google
OAuth client (mode ``google``). ``current_user`` is the dependency every ``/api`` route uses;
it also enforces a custom request header on state-changing calls so a cross-site form post
cannot ride on the cookie. Identity data never reaches a log line.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from apt_classifier.config import Settings
from apt_classifier.users import LOCAL_USER, ROLES, User, UserStore, email_allowed, parse_list

log = logging.getLogger("apt_classifier.api.auth")
GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
SESSION_COOKIE = "vestige_session"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "Vestige"
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


@dataclass
class AuthState:
    mode: str
    users: UserStore | None = None
    oauth: Any = None
    allowed_domains: set[str] = field(default_factory=set)
    allowed_emails: set[str] = field(default_factory=set)
    admin_emails: set[str] = field(default_factory=set)
    public_url: str | None = None

    @property
    def google(self) -> bool:
        return self.mode == "google"

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": "google" if self.google else None,
            "allowlist_configured": bool(self.allowed_domains or self.allowed_emails),
        }


def install_auth(app: Any, settings: Settings, *, users: UserStore | None = None) -> AuthState:
    mode = settings.resolved_auth_mode()
    state = AuthState(
        mode=mode,
        allowed_domains=parse_list(settings.auth_allowed_domains),
        allowed_emails=parse_list(settings.auth_allowed_emails),
        admin_emails=parse_list(settings.auth_admin_emails),
        public_url=(settings.auth_public_url or "").rstrip("/") or None,
    )
    if mode == "google":
        from authlib.integrations.starlette_client import OAuth
        from starlette.middleware.sessions import SessionMiddleware

        if not settings.google_client_id or settings.google_client_secret is None:
            raise RuntimeError("AUTH_MODE=google needs GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET")
        if not (state.allowed_domains or state.allowed_emails):
            raise RuntimeError(
                "Google sign-in needs an allowlist: set AUTH_ALLOWED_DOMAINS and/or AUTH_ALLOWED_EMAILS"
            )
        if settings.auth_secret_key is not None and settings.auth_secret_key.get_secret_value():
            secret = settings.auth_secret_key.get_secret_value()
        else:
            secret = secrets.token_urlsafe(48)
            log.warning("AUTH_SECRET_KEY is not set: sessions will not survive a restart")
        app.add_middleware(
            SessionMiddleware,
            secret_key=secret,
            session_cookie=SESSION_COOKIE,
            max_age=max(1, int(settings.auth_session_hours)) * 3600,
            same_site="lax",
            https_only=bool(settings.auth_cookie_secure),
        )
        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret.get_secret_value(),
            server_metadata_url=GOOGLE_METADATA_URL,
            client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
        )
        state.oauth = oauth
        state.users = users or UserStore(settings.users_db_path)
    app.state.auth = state
    app.include_router(router)
    return state


def _state(request: Request) -> AuthState:
    state = getattr(request.app.state, "auth", None)
    if state is None:
        return AuthState(mode="off")
    return state


def current_user(request: Request) -> User:
    """The signed-in user (or the local analyst when sign-in is off). Mutating requests must
    carry the ``X-Requested-With: Vestige`` header; a browser only adds it from our own script."""
    state = _state(request)
    if not state.google:
        return LOCAL_USER
    if request.method in MUTATING and request.headers.get(CSRF_HEADER, "") != CSRF_VALUE:
        raise HTTPException(403, "missing request header")
    user = state.users.get(request.session.get("user_id")) if state.users else None
    if user is None:
        raise HTTPException(401, "sign in to continue")
    if not user.active:
        request.session.clear()
        raise HTTPException(401, "this account is deactivated")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "administrator role required")
    return user


def complete_login(request: Request, state: AuthState, info: dict[str, Any]) -> User:
    """Turn a verified Google profile into a session. Raises ``PermissionError`` with a short
    code (``unverified`` / ``not_allowed`` / ``inactive``) that the login page can display."""
    email = str(info.get("email") or "").strip().lower()
    if not email or not info.get("email_verified", False):
        raise PermissionError("unverified")
    if not email_allowed(email, domains=state.allowed_domains, emails=state.allowed_emails):
        raise PermissionError("not_allowed")
    assert state.users is not None
    user = state.users.upsert_login(
        email=email,
        name=str(info.get("name") or ""),
        picture=str(info.get("picture") or ""),
        subject=str(info.get("sub") or "") or None,
        admin_emails=state.admin_emails,
    )
    if not user.active:
        raise PermissionError("inactive")
    request.session.clear()
    request.session["user_id"] = user.id
    return user


def _redirect_uri(request: Request, state: AuthState) -> str:
    if state.public_url:
        return f"{state.public_url}/auth/callback"
    return str(request.url_for("auth_callback"))


router = APIRouter()


@router.get("/auth/login", include_in_schema=False)
async def auth_login(request: Request) -> Response:
    state = _state(request)
    if not state.google:
        return RedirectResponse("/", status_code=302)
    client = state.oauth.create_client("google")
    return await client.authorize_redirect(request, _redirect_uri(request, state))


@router.get("/auth/callback", name="auth_callback", include_in_schema=False)
async def auth_callback(request: Request) -> Response:
    state = _state(request)
    if not state.google:
        return RedirectResponse("/", status_code=302)
    client = state.oauth.create_client("google")
    try:
        token = await client.authorize_access_token(request)
    except Exception as exc:  # noqa: BLE001 - provider or state mismatch: back to the login page
        log.warning("Google sign-in failed: %s", type(exc).__name__)
        return RedirectResponse("/login?" + urlencode({"error": "provider"}), status_code=302)
    info = token.get("userinfo") if isinstance(token, dict) else None
    if not info:
        try:
            info = await client.userinfo(token=token)
        except Exception:  # noqa: BLE001
            info = None
    if not isinstance(info, dict):
        return RedirectResponse("/login?" + urlencode({"error": "provider"}), status_code=302)
    try:
        complete_login(request, state, info)
    except PermissionError as exc:
        return RedirectResponse("/login?" + urlencode({"error": str(exc)}), status_code=302)
    return RedirectResponse("/", status_code=302)


@router.post("/auth/logout", include_in_schema=False)
async def auth_logout(request: Request) -> Response:
    state = _state(request)
    if state.google:
        request.session.clear()
    return Response(status_code=204)


@router.get("/api/auth")
def auth_info(request: Request) -> dict[str, Any]:
    """Public: how sign-in works on this deployment (the login page needs it before a session)."""
    return _state(request).describe()


@router.get("/api/me")
def me(request: Request, user: User = Depends(current_user)) -> dict[str, Any]:
    return {"user": user.to_dict(), "auth": _state(request).describe()}


@router.get("/api/users")
def list_users(request: Request, _: User = Depends(require_admin)) -> list[dict[str, Any]]:
    state = _state(request)
    if not state.google or state.users is None:
        return [LOCAL_USER.to_dict()]
    return [user.to_dict() for user in state.users.list()]


class UserPatch(BaseModel):
    role: str | None = None
    active: bool | None = None


@router.patch("/api/users/{user_id}")
def update_user(
    user_id: str, body: UserPatch, request: Request, admin: User = Depends(require_admin)
) -> dict[str, Any]:
    state = _state(request)
    if not state.google or state.users is None:
        raise HTTPException(409, "user management needs Google sign-in")
    target = state.users.get(user_id)
    if target is None:
        raise HTTPException(404, "user not found")
    if body.role is not None:
        if body.role not in ROLES:
            raise HTTPException(400, f"role must be one of {', '.join(ROLES)}")
        if target.id == admin.id and body.role != "admin":
            raise HTTPException(400, "you cannot remove your own administrator role")
        if target.is_admin and body.role != "admin" and state.users.admin_count() <= 1:
            raise HTTPException(400, "at least one active administrator must remain")
        target = state.users.set_role(target.id, body.role)
    if body.active is not None:
        if target.id == admin.id and not body.active:
            raise HTTPException(400, "you cannot deactivate your own account")
        if target.is_admin and not body.active and state.users.admin_count() <= 1:
            raise HTTPException(400, "at least one active administrator must remain")
        target = state.users.set_active(target.id, bool(body.active))
    return target.to_dict()


def unauthorized_json(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=401)
