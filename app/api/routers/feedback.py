"""Feedback endpoints — thumbs-up / thumbs-down per assistant response."""

from __future__ import annotations

from fastapi import APIRouter, Header

from app.api.models import (
    FeedbackItem,
    FeedbackRequest,
    FeedbackResponse,
    SessionFeedbackResponse,
)
from app.services.siakad_session import (
    has_student_access_binding,
    verify_student_access_token,
)
from app.utils.exceptions import AppError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["feedback"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _verify_token_if_bound(session_id: str, token: str | None) -> None:
    """Raise 401 when the session requires a valid student access token."""
    if await has_student_access_binding(
        session_id
    ) and not await verify_student_access_token(session_id, token):
        raise AppError(
            detail="Student access token is required for this session.",
            status_code=401,
            title="Token Required",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    session_id: str,
    payload: FeedbackRequest,
    x_student_access_token: str | None = Header(
        default=None, alias="X-Student-Access-Token"
    ),
) -> FeedbackResponse:
    """Submit a thumbs-up or thumbs-down rating for an assistant response.

    - ``session_id`` — the session that produced the response.
    - ``message_id`` — the ``message_id`` field returned by ``POST /chat``.
    - ``rating``     — ``"thumbs_up"`` or ``"thumbs_down"``.
    - ``comment``    — optional free-text comment (max 500 chars).

    Re-submitting feedback for the same ``message_id`` **replaces** the
    previous rating (idempotent).
    """
    from app.services.memory import save_feedback

    await _verify_token_if_bound(session_id, x_student_access_token)

    logger.info(
        "Feedback received session=%s message_id=%s rating=%s",
        session_id[:8] + "…",
        payload.message_id,
        payload.rating,
    )

    try:
        record = await save_feedback(
            session_id=session_id,
            message_id=payload.message_id,
            rating=payload.rating.value,
            comment=payload.comment,
        )
    except Exception as exc:
        logger.exception("Failed to save feedback: %s", exc)
        raise AppError(
            detail="Could not save feedback. Please try again.",
            status_code=500,
            title="Feedback Save Failed",
        ) from exc

    return FeedbackResponse(
        session_id=session_id,
        message_id=record["message_id"],
        rating=record["rating"],
        comment=record.get("comment"),
        created_at=record["created_at"],
    )


@router.get("/chat/{session_id}/feedbacks", response_model=SessionFeedbackResponse)
async def get_feedbacks(
    session_id: str,
    x_student_access_token: str | None = Header(
        default=None, alias="X-Student-Access-Token"
    ),
) -> SessionFeedbackResponse:
    """List all feedback submitted for a session."""
    from app.services.memory import get_session_feedbacks

    await _verify_token_if_bound(session_id, x_student_access_token)

    try:
        raw = await get_session_feedbacks(session_id)
    except Exception as exc:
        logger.exception("Failed to load feedbacks: %s", exc)
        raise AppError(
            detail="Could not retrieve feedbacks.",
            status_code=500,
            title="Feedback Load Failed",
        ) from exc

    items = [
        FeedbackItem(
            message_id=fb["message_id"],
            rating=fb["rating"],
            comment=fb.get("comment"),
            created_at=fb["created_at"],
        )
        for fb in raw
    ]

    return SessionFeedbackResponse(
        session_id=session_id,
        total=len(items),
        feedbacks=items,
    )
