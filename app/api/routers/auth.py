"""Authentication endpoint — POST /auth/login."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request

from app.api.models import LoginRequest, LoginResponse
from app.config import get_settings
from app.services.rate_limiter import allow_request
from app.services.siakad_session import (
    init_siakad_session,
    issue_student_access_token,
)
from app.utils.exceptions import AppError
from app.utils.security import mask_email, mask_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _check_rate_limit(client_ip: str) -> None:
    """Enforce per-IP rate limiting for login attempts."""
    settings = get_settings()
    limit_key = f"auth_login:{client_ip}"
    if not await allow_request(
        key=limit_key,
        limit=settings.auth_login_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    ):
        raise AppError(
            detail="Too many login attempts.",
            status_code=429,
            title="Too Many Requests",
        )


async def _authenticate(email: str, password: str) -> tuple[str, str]:
    """
    Perform SIAKAD authentication and return (session_id, student_access_token).

    Raises ``AppError(401)`` on failure.
    """
    session_id = str(uuid.uuid4())
    logger.info(
        "SIAKAD login request — session=%s, email=%s",
        mask_session_id(session_id),
        mask_email(email),
    )

    success = await init_siakad_session(
        session_id=session_id, email=email, password=password
    )
    if not success:
        logger.warning(
            "SIAKAD login failed — session=%s, email=%s",
            mask_session_id(session_id),
            mask_email(email),
        )
        raise AppError(
            detail="Login SIAKAD gagal. Periksa kembali email dan password Anda.",
            status_code=401,
            title="Authentication Failed",
        )

    student_access_token = await issue_student_access_token(session_id)
    logger.info(
        "Authenticated session created — session=%s", mask_session_id(session_id)
    )
    return session_id, student_access_token


@router.post("/login", response_model=LoginResponse)
async def login_siakad(payload: LoginRequest, http_request: Request) -> LoginResponse:
    """
    Authenticate against SIAKAD and store cookies in Redis.

    Email and password are processed in-memory and never persisted.
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    await _check_rate_limit(client_ip)

    session_id, token = await _authenticate(payload.email, payload.password)

    return LoginResponse(
        session_id=session_id,
        student_access_token=token,
        status="authenticated",
        message="Login SIAKAD berhasil",
    )
