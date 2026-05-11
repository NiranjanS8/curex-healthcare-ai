"""JWT authentication and local user storage for CureX."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import jwt
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field


DEFAULT_AUTH_DB_PATH = Path("healthcare_auth.sqlite")
JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 120
PASSWORD_ITERATIONS = 260_000


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=8, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, str]


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    username: str


def _db_path(path: str | Path | None = None) -> Path:
    return Path(path or os.getenv("AUTH_DB_PATH") or DEFAULT_AUTH_DB_PATH)


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET") or "dev-only-change-me-use-env-jwt-secret"


def init_auth_db(path: str | Path | None = None) -> Path:
    db_path = _db_path(path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    return db_path


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    password_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        password_salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            "pbkdf2_sha256",
            str(PASSWORD_ITERATIONS),
            base64.b64encode(password_salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_b64, expected_b64 = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(expected_b64)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def create_user(username: str, password: str, *, db_path: str | Path | None = None) -> AuthUser:
    normalized_username = _normalize_username(username)
    user = AuthUser(user_id=str(uuid4()), username=normalized_username)
    try:
        with sqlite3.connect(init_auth_db(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username,
                    _hash_password(password),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists.") from exc
    return user


def authenticate_user(username: str, password: str, *, db_path: str | Path | None = None) -> AuthUser | None:
    with sqlite3.connect(init_auth_db(db_path)) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (_normalize_username(username),),
        ).fetchone()
    if row is None:
        return None
    user_id, stored_username, password_hash = row
    if not _verify_password(password, password_hash):
        return None
    return AuthUser(user_id=str(user_id), username=str(stored_username))


def get_user(user_id: str, *, db_path: str | Path | None = None) -> AuthUser | None:
    with sqlite3.connect(init_auth_db(db_path)) as conn:
        row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return AuthUser(user_id=str(row[0]), username=str(row[1]))


def create_access_token(
    user: AuthUser,
    *,
    expires_delta: timedelta | None = None,
    secret: str | None = None,
) -> str:
    expires_at = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict[str, Any] = {
        "sub": user.user_id,
        "username": user.username,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, secret or _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, *, secret: str | None = None) -> dict[str, Any]:
    try:
        return jwt.decode(token, secret or _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def bearer_token_from_request(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def get_current_user(request: Request) -> AuthUser:
    payload = decode_access_token(bearer_token_from_request(request))
    user_id = str(payload.get("sub") or "")
    user = get_user(user_id, db_path=getattr(request.app.state, "auth_db_path", None))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def token_response(user: AuthUser) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        user={"id": user.user_id, "username": user.username},
    )
