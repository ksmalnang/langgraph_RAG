"""API endpoints for authentication (SIAKAD login)."""

import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.services.siakad_session import init_siakad_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    session_id: str
    status: str
    message: str


@router.post("/login", response_model=LoginResponse)
async def login_siakad(request: LoginRequest):
    """
    Login ke SIAKAD dan simpan cookies ke Redis.
    Email dan password hanya diproses in-memory dan tidak disimpan.
    """
    session_id = str(uuid.uuid4())
    logger.info(
        "Menerima request login SIAKAD untuk session_id=%s, email=%s",
        session_id,
        request.email,
    )

    success = await init_siakad_session(
        session_id=session_id, email=request.email, password=request.password
    )

    if not success:
        logger.warning(
            "SIAKAD login gagal untuk session_id=%s, email=%s",
            session_id,
            request.email,
        )
        raise HTTPException(
            status_code=401,
            detail="Login SIAKAD gagal. Periksa kembali email dan password Anda.",
        )

    logger.info(
        "[login] Authenticated session created: %s untuk %s", session_id, request.email
    )
    return LoginResponse(
        session_id=session_id, status="authenticated", message="Login SIAKAD berhasil"
    )
