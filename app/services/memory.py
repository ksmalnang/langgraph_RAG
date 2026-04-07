"""Redis-backed session / chat-history manager."""

from __future__ import annotations

from datetime import UTC, datetime
import json

import redis.asyncio as aioredis

from app.config import get_settings
from app.utils.exceptions import MemoryStoreError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level connection pool — initialised lazily.
_pool: aioredis.Redis | None = None


def _key(session_id: str) -> str:
    """Build the Redis key for a session's chat history."""
    return f"session:{session_id}:history"


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
            session_id,
            exc_info=True,
        )
        raise MemoryStoreError("Failed to load chat history") from exc
    if raw is None:
        return []
    history: list[dict[str, str]] = json.loads(raw)
    logger.debug("Loaded %d turns for session %s", len(history), session_id)
    return history


async def save_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a user/assistant turn and refresh the TTL."""
    settings = get_settings()
    try:
        r = await get_redis()
        history = await get_history(session_id)
    except MemoryStoreError:
        raise
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_turn_precheck session_id=%s mode=unexpected",
            session_id,
            exc_info=True,
        )
        raise MemoryStoreError("Failed to prepare chat history write") from exc

    now = datetime.now(UTC).isoformat()
    history.append({"role": "user", "content": user_message, "timestamp": now})
    history.append(
        {"role": "assistant", "content": assistant_message, "timestamp": now}
    )

    try:
        await r.set(
            _key(session_id),
            json.dumps(history),
            ex=settings.session_ttl_seconds,
        )
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=save_turn session_id=%s mode=unexpected",
            session_id,
            exc_info=True,
        )
        raise MemoryStoreError("Failed to persist chat history") from exc
    logger.debug("Saved turn for session %s (total=%d)", session_id, len(history))


async def clear_session(session_id: str) -> None:
    """Delete all history for a session."""
    try:
        r = await get_redis()
        await r.delete(_key(session_id))
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=redis operation=clear_session session_id=%s mode=unexpected",
            session_id,
            exc_info=True,
        )
        raise MemoryStoreError("Failed to clear chat history") from exc
    logger.info("Cleared session %s", session_id)


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
