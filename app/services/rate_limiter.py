"""Simple in-process sliding-window rate limiter."""

from __future__ import annotations

from collections import defaultdict, deque
import time
from typing import DefaultDict

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


def allow_request(key: str, limit: int, window_seconds: int) -> bool:
    """Shared helper used by API routes."""
    return _limiter.allow(key=key, limit=limit, window_seconds=window_seconds)


def reset_rate_limiter() -> None:
    """Test helper for deterministic rate-limit tests."""
    _limiter.reset()

