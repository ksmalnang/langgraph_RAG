"""Simple in-process sliding-window rate limiter."""

from __future__ import annotations

from collections import defaultdict, deque
import time
from typing import DefaultDict

from app.services.memory import get_redis
from app.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """Per-process sliding-window limiter keyed by route/client fingerprint."""

    def __init__(self) -> None:
        self._buckets: DefaultDict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        bucket = self._buckets[key]

        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            return False

        bucket.append(now)
        return True

    def reset(self) -> None:
        self._buckets.clear()


_limiter = InMemoryRateLimiter()


async def allow_request(key: str, limit: int, window_seconds: int) -> bool:
    """Shared helper used by API routes.

    Uses Redis as the primary rate-limit backend (safe across workers),
    and falls back to in-process memory when Redis is unavailable.
    """
    redis_key = f"rate_limit:{key}"
    try:
        r = await get_redis()
        current = await r.incr(redis_key)
        if current == 1:
            await r.expire(redis_key, window_seconds)
        return current <= limit
    except Exception:
        logger.warning(
            "Redis rate limiter unavailable for key=%s; using in-memory fallback",
            key,
        )
        return _limiter.allow(key=key, limit=limit, window_seconds=window_seconds)


def reset_rate_limiter() -> None:
    """Test helper for deterministic rate-limit tests."""
    _limiter.reset()

