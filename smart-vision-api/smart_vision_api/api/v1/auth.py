from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...core.auth import issue_token, require_user
from ...core.config import settings
from ...schemas.auth import AuthStatusResponse, LoginRequest, LoginResponse, MeResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/status", response_model=AuthStatusResponse, summary="Auth status")
async def status() -> AuthStatusResponse:
    return AuthStatusResponse(enabled=bool(settings.AUTH_ENABLED))


@router.post("/login", response_model=LoginResponse, summary="Login")
async def login(payload: LoginRequest) -> LoginResponse:
    if not settings.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Auth is disabled on this server.")

    if payload.username != settings.AUTH_USERNAME or payload.password != settings.AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = issue_token(username=payload.username, ttl_seconds=settings.AUTH_TOKEN_TTL_SECONDS)
    return LoginResponse(access_token=token, username=payload.username)


@router.get("/me", response_model=MeResponse, summary="Current user")
async def me(username: str = Depends(require_user)) -> MeResponse:
    return MeResponse(username=username)
