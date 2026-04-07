"""Chat API — POST /chat and GET /chat/{session_id}/history."""

from __future__ import annotations

from functools import lru_cache
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    SourceChunk,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


def resolve_session_id(session_id: str | None) -> tuple[str, bool]:
    """
    Return (session_id, is_authenticated).
    is_authenticated = True kalau session_id UUID4 valid dari client.
    """
    if session_id is None:
        return str(uuid.uuid4()), False
    try:
        uuid.UUID(session_id, version=4)
        return session_id, True
    except ValueError:
        return str(uuid.uuid4()), False


@lru_cache
def _get_graph():
    """Lazily build and cache the compiled graph."""
    from app.agent.graph import build_graph

    return build_graph()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the LangGraph agent."""
    session_id, is_authenticated = resolve_session_id(request.session_id)

    logger.info(
        "Chat request: session=%s, auth=%s, message=%s",
        session_id,
        is_authenticated,
        request.message[:80],
    )

    initial_state = {
        "query": request.message,
        "session_id": session_id,
        "chat_history": [],
        "need_retrieval": False,
        "documents": [],
        "reranked_documents": [],
        "relevance_ok": False,
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    }

    try:
        graph = _get_graph()
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to process your message. Please try again.",
        ) from exc

    # Convert raw source dicts to SourceChunk models
    raw_sources = result.get("sources", [])
    source_chunks = [SourceChunk(**s) for s in raw_sources if isinstance(s, dict)]

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", "Sorry, I couldn't generate an answer."),
        sources=source_chunks,
    )


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """Retrieve chat history for a session from Redis."""
    from app.services.memory import get_history

    history = await get_history(session_id)

    items = [
        ChatHistoryItem(
            role=turn["role"],
            content=turn["content"],
            timestamp=turn.get("timestamp"),
        )
        for turn in history
    ]

    return ChatHistoryResponse(session_id=session_id, history=items)
