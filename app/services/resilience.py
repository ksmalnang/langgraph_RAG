"""Shared retry/latency helpers for remote integrations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import TypeVar

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def now_ms() -> float:
    """Monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    """Elapsed time in milliseconds since ``start_ms``."""
    return time.perf_counter() * 1000 - start_ms


async def retry_async(
    operation: str,
    dependency: str,
    fn: Callable[[], Awaitable[T]],
    retry_on: tuple[type[Exception], ...],
    attempts: int | None = None,
) -> T:
    """Retry an idempotent async operation on transient failures."""
    settings = get_settings()
    max_attempts = attempts or settings.service_retry_attempts
    backoff = settings.service_retry_backoff_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt >= max_attempts:
                raise
            logger.warning(
                "Transient failure; retrying dependency=%s operation=%s attempt=%d/%d error=%s",
                dependency,
                operation,
                attempt,
                max_attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(backoff * attempt)

