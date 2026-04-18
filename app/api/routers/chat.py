"""Chat-related API endpoints — POST /chat and GET /chat/{session_id}/history."""

from __future__ import annotations

from functools import lru_cache
import uuid

from fastapi import APIRouter, Header, Request

from app.api.models import (
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    SourceChunk,
)
from app.config import get_settings
from app.services.rate_limiter import allow_request
from app.services.siakad_session import (
    has_student_access_binding,
    verify_student_access_token,
)
from app.utils.exceptions import AppError
from app.utils.logger import get_logger
from app.utils.security import mask_session_id

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_session_id(session_id: str | None) -> tuple[str, bool]:
    """
    Return ``(session_id, is_authenticated)``.

    *is_authenticated* is ``True`` when the caller supplied a valid UUID4.
    """
    if session_id is None:
        return str(uuid.uuid4()), False
    try:
        uuid.UUID(session_id, version=4)
        return session_id, True
    except ValueError:
        return str(uuid.uuid4()), False


async def _verify_token_if_bound(session_id: str, token: str | None) -> None:
    """Raise 401 when a bound session requires a valid student token."""
    if await has_student_access_binding(
        session_id
    ) and not await verify_student_access_token(session_id, token):
        logger.warning("Student token verification failed for session=%s", session_id)
        raise AppError(
            detail="Student access token is required for this session.",
            status_code=401,
            title="Token Required",
        )


def _build_initial_state(query: str, session_id: str, message_id: str) -> dict:
    """Construct the LangGraph agent input payload."""
    return {
        "query": query,
        "session_id": session_id,
        "message_id": message_id,
        "chat_history": [],
        "need_retrieval": False,
        "documents": [],
        "reranked_documents": [],
        "relevance_ok": False,
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    }


@lru_cache(maxsize=1)
def _get_graph():
    """Lazily build and cache the compiled LangGraph graph."""
    from app.agent.graph import build_graph

    return build_graph()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    http_request: Request,
    x_student_access_token: str | None = Header(
        default=None, alias="X-Student-Access-Token"
    ),
) -> ChatResponse:
    """Process a chat message through the LangGraph agent."""
    settings = get_settings()
    client_ip = http_request.client.host if http_request.client else "unknown"

    if not await allow_request(
        key=f"chat:{client_ip}",
        limit=settings.chat_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    ):
        raise AppError(
            detail="Too many chat requests.",
            status_code=429,
            title="Too Many Requests",
        )

    session_id, is_authenticated = _resolve_session_id(payload.session_id)
    session_hint = mask_session_id(session_id)

    if is_authenticated:
        await _verify_token_if_bound(session_id, x_student_access_token)

    logger.info(
        "Chat request — session=%s, auth=%s, message_length=%d",
        session_hint,
        is_authenticated,
        len(payload.message),
    )

    # Generate message_id before the graph runs so store_memory node
    # can persist it alongside the assistant turn (avoids a double write).
    message_id = str(uuid.uuid4())

    initial_state = _build_initial_state(payload.message, session_id, message_id)

    try:
        graph = _get_graph()
        result = await graph.ainvoke(initial_state)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        raise AppError(
            detail="Failed to process your message. Please try again.",
            status_code=500,
            title="Agent Invocation Failed",
        ) from exc

    answer = result.get("answer", "Sorry, I couldn't generate an answer.")
    sources = [
        SourceChunk(**s) for s in result.get("sources", []) if isinstance(s, dict)
    ]

    # store_memory (graph node) already persisted the turn — no second write needed.

    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        answer=answer,
        sources=sources,
    )


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    x_student_access_token: str | None = Header(
        default=None, alias="X-Student-Access-Token"
    ),
) -> ChatHistoryResponse:
    """Retrieve chat history for a session from Redis."""
    from app.services.memory import get_history

    await _verify_token_if_bound(session_id, x_student_access_token)

    history = await get_history(session_id)
    items = [
        ChatHistoryItem(
            role=turn["role"],
            content=turn["content"],
            timestamp=turn.get("timestamp"),
            message_id=turn.get("message_id"),
        )
        for turn in history
    ]

    return ChatHistoryResponse(session_id=session_id, history=items)
