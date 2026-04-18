"""Redis-backed session / chat-history manager."""

from __future__ import annotations

from datetime import UTC, datetime
import json

import redis.asyncio as aioredis

from app.config import get_settings
from app.utils.exceptions import MemoryStoreError
from app.utils.logger import get_logger
from app.utils.security import mask_session_id

logger = get_logger(__name__)

# Module-level connection pool — initialised lazily.
_pool: aioredis.Redis | None = None


def _key(session_id: str) -> str:
    """Build the Redis key for a session's chat history."""
    return f"session:{session_id}:history"


def _feedback_list_key(session_id: str) -> str:
    """Build the Redis key for a session's feedback list."""
    return f"session:{session_id}:feedbacks"


async def get_redis() -> aioredis.Redis:
    """Return (or create) the singleton Redis connection."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout_seconds,
            socket_connect_timeout=settings.redis_socket_timeout_seconds,
            retry_on_timeout=False,
        )
    return _pool


async def get_history(session_id: str) -> list[dict[str, str]]:
    """Retrieve chat history for a session.

    Returns a list of ``{"role": ..., "content": ...}`` dicts.
    """
    try:
        r = await get_redis()
        raw = await r.get(_key(session_id))
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=get_history session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to load chat history") from exc
    if raw is None:
        return []
    history: list[dict[str, str]] = json.loads(raw)
    logger.debug(
        "Loaded %d turns for session %s",
        len(history),
        mask_session_id(session_id),
    )
    return history


async def save_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    message_id: str | None = None,
) -> None:
    """Append a user/assistant turn and refresh the TTL.

    ``message_id`` is stored alongside the assistant turn so it can be
    referenced when a client submits feedback.
    """
    settings = get_settings()
    try:
        r = await get_redis()
        history = await get_history(session_id)
    except MemoryStoreError:
        raise
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_turn_precheck session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to prepare chat history write") from exc

    now = datetime.now(UTC).isoformat()
    history.append({"role": "user", "content": user_message, "timestamp": now})
    assistant_entry: dict[str, str | None] = {
        "role": "assistant",
        "content": assistant_message,
        "timestamp": now,
    }
    if message_id is not None:
        assistant_entry["message_id"] = message_id
    history.append(assistant_entry)

    try:
        await r.set(
            _key(session_id),
            json.dumps(history),
            ex=settings.session_ttl_seconds,
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_turn session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to persist chat history") from exc
    logger.debug(
        "Saved turn for session %s (total=%d)",
        mask_session_id(session_id),
        len(history),
    )


async def save_feedback(
    session_id: str,
    message_id: str,
    rating: str,
    comment: str | None = None,
) -> dict:
    """Persist a thumbs-up / thumbs-down rating for a specific assistant message.

    Feedback is stored as a JSON list under ``session:{session_id}:feedbacks``.
    If a record for the same ``message_id`` already exists it is **replaced**
    (idempotent re-rating).  Returns the saved feedback record.
    """
    settings = get_settings()
    now = datetime.now(UTC).isoformat()
    record: dict = {
        "message_id": message_id,
        "rating": rating,
        "comment": comment,
        "created_at": now,
    }

    try:
        r = await get_redis()
        raw = await r.get(_feedback_list_key(session_id))
        feedbacks: list[dict] = json.loads(raw) if raw else []
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_feedback_read session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to read existing feedback") from exc

    # Replace existing entry for the same message_id, or append a new one.
    updated = False
    for i, fb in enumerate(feedbacks):
        if fb.get("message_id") == message_id:
            feedbacks[i] = record
            updated = True
            break
    if not updated:
        feedbacks.append(record)

    try:
        await r.set(
            _feedback_list_key(session_id),
            json.dumps(feedbacks),
            ex=settings.session_ttl_seconds,
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_feedback_write session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to persist feedback") from exc

    logger.info(
        "Feedback saved session=%s message_id=%s rating=%s",
        mask_session_id(session_id),
        message_id,
        rating,
    )
    return record


async def get_session_feedbacks(session_id: str) -> list[dict]:
    """Return all feedback records for a session, ordered by creation time."""
    try:
        r = await get_redis()
        raw = await r.get(_feedback_list_key(session_id))
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=get_session_feedbacks session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to load session feedbacks") from exc

    if raw is None:
        return []
    feedbacks: list[dict] = json.loads(raw)
    return feedbacks


async def clear_session(session_id: str) -> None:
    """Delete all history and feedback for a session."""
    try:
        r = await get_redis()
        await r.delete(_key(session_id), _feedback_list_key(session_id))
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=clear_session session_id=%s mode=unexpected",
            mask_session_id(session_id),
            exc_info=True,
        )
        raise MemoryStoreError("Failed to clear chat history") from exc
    logger.info("Cleared session %s", mask_session_id(session_id))


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
