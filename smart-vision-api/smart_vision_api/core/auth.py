"""Simple token authentication (no external deps).

This is intentionally minimal and intended for internal/demo deployments.
If you need production-grade auth, swap this for a proper JWT/OIDC solution.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings


@dataclass(frozen=True)
class AuthSession:
    username: str
    expires_at: float


_bearer = HTTPBearer(auto_error=False)
_sessions: Dict[str, AuthSession] = {}


def issue_token(*, username: str, ttl_seconds: int) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = AuthSession(username=username, expires_at=time.time() + ttl_seconds)
    return token


def _get_session(token: str) -> Optional[AuthSession]:
    session = _sessions.get(token)
    if session is None:
        return None
    if time.time() >= session.expires_at:
        _sessions.pop(token, None)
        return None
    return session


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Optional[str]:
    if not settings.AUTH_ENABLED:
        return None

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    session = _get_session(credentials.credentials)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return session.username


def require_user(username: str | None = Depends(get_current_user)) -> str:
    if not settings.AUTH_ENABLED:
        return "anonymous"
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return username

