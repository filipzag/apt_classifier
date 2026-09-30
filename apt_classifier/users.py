"""Users of the web app: a small SQLite store plus the sign-in policy.

Identity comes from Google sign-in (OpenID Connect); this module never sees a password. A
user row is created on first successful sign-in when the address passes the allowlist
(``AUTH_ALLOWED_DOMAINS`` / ``AUTH_ALLOWED_EMAILS``). Roles are ``admin`` (manage users, delete
any case) and ``analyst``; the first user to sign in and every address in ``AUTH_ADMIN_EMAILS``
are admins. With sign-in switched off the app runs as one local analyst with admin rights.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROLES = ("admin", "analyst")


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    picture: str
    role: str
    active: bool
    created_at: str
    last_login_at: str | None = None
    subject: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "picture": self.picture,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


LOCAL_USER = User(
    id="local",
    email="",
    name="Local analyst",
    picture="",
    role="admin",
    active=True,
    created_at="",
)


def parse_list(value: str | None) -> set[str]:
    """Comma/space/semicolon separated, case-insensitive, blanks dropped."""
    if not value:
        return set()
    items = value.replace(";", ",").replace(" ", ",").split(",")
    return {item.strip().lower().lstrip("@") for item in items if item.strip()}


def email_allowed(email: str, *, domains: set[str], emails: set[str]) -> bool:
    address = (email or "").strip().lower()
    if not address or "@" not in address:
        return False
    if address in emails:
        return True
    domain = address.rsplit("@", 1)[1]
    return domain in domains


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class UserStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL DEFAULT '',
                    picture TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT 'analyst',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    subject TEXT
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> User | None:
        if row is None:
            return None
        return User(
            id=str(row["id"]),
            email=str(row["email"]),
            name=str(row["name"] or ""),
            picture=str(row["picture"] or ""),
            role=str(row["role"] or "analyst"),
            active=bool(row["active"]),
            created_at=str(row["created_at"] or ""),
            last_login_at=row["last_login_at"],
            subject=row["subject"],
        )

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def list(self) -> list[User]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM users ORDER BY created_at ASC, email ASC"
            ).fetchall()
        return [user for user in (self._row(row) for row in rows) if user is not None]

    def get(self, user_id: str | None) -> User | None:
        if not user_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._row(row)

    def get_by_email(self, email: str) -> User | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return self._row(row)

    def upsert_login(
        self,
        *,
        email: str,
        name: str = "",
        picture: str = "",
        subject: str | None = None,
        admin_emails: set[str] | None = None,
    ) -> User:
        """Create the user on first sign-in, otherwise refresh profile fields and the login
        time. The first user ever, and any address in ``admin_emails``, becomes an admin;
        an existing analyst listed in ``admin_emails`` is promoted."""
        address = email.strip().lower()
        admins = admin_emails or set()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM users WHERE email = ?", (address,)
            ).fetchone()
            now = _now()
            if row is None:
                first = self._connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
                role = "admin" if first or address in admins else "analyst"
                user_id = uuid4().hex[:16]
                self._connection.execute(
                    "INSERT INTO users (id, email, name, picture, role, active, created_at, "
                    "last_login_at, subject) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)",
                    (user_id, address, name or address, picture or "", role, now, now, subject),
                )
            else:
                user_id = str(row["id"])
                role = "admin" if address in admins else str(row["role"])
                self._connection.execute(
                    "UPDATE users SET name = ?, picture = ?, role = ?, last_login_at = ?, "
                    "subject = COALESCE(?, subject) WHERE id = ?",
                    (
                        name or row["name"] or address,
                        picture or row["picture"] or "",
                        role,
                        now,
                        subject,
                        user_id,
                    ),
                )
            refreshed = self._connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        user = self._row(refreshed)
        assert user is not None
        return user

    def set_role(self, user_id: str, role: str) -> User:
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        with self._lock, self._connection:
            self._connection.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            row = self._connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        user = self._row(row)
        if user is None:
            raise KeyError(user_id)
        return user

    def set_active(self, user_id: str, active: bool) -> User:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id)
            )
            row = self._connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        user = self._row(row)
        if user is None:
            raise KeyError(user_id)
        return user

    def admin_count(self, *, active_only: bool = True) -> int:
        with self._lock:
            query = "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            if active_only:
                query += " AND active = 1"
            return int(self._connection.execute(query).fetchone()[0])
