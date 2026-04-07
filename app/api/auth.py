"""API endpoints for authentication (SIAKAD login)."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from app.config import get_settings
from app.services.rate_limiter import allow_request
from app.services.siakad_session import init_siakad_session, issue_student_access_token
from app.utils.security import mask_email, mask_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    session_id: str
    student_access_token: str
    status: str
    message: str


@router.post("/login", response_model=LoginResponse)
async def login_siakad(request: LoginRequest, http_request: Request):
    """
    Login ke SIAKAD dan simpan cookies ke Redis.
    Email dan password hanya diproses in-memory dan tidak disimpan.
    """
    settings = get_settings()
    client_ip = http_request.client.host if http_request.client else "unknown"
    limit_key = f"auth_login:{client_ip}"
    if not allow_request(
        key=limit_key,
        limit=settings.auth_login_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    ):
        raise HTTPException(status_code=429, detail="Too many login attempts.")

    session_id = str(uuid.uuid4())
    logger.info(
        "Menerima request login SIAKAD untuk session=%s, email=%s",
        mask_session_id(session_id),
        mask_email(str(request.email)),
    )

    success = await init_siakad_session(
        session_id=session_id, email=request.email, password=request.password
    )

    if not success:
        logger.warning(
            "SIAKAD login gagal untuk session=%s, email=%s",
            mask_session_id(session_id),
            mask_email(str(request.email)),
        )
        raise HTTPException(
            status_code=401,
            detail="Login SIAKAD gagal. Periksa kembali email dan password Anda.",
        )

    student_access_token = await issue_student_access_token(session_id)
    logger.info(
        "[login] Authenticated session created: %s", mask_session_id(session_id)
    )
    return LoginResponse(
        session_id=session_id,
        student_access_token=student_access_token,
        status="authenticated",
        message="Login SIAKAD berhasil",
    )
