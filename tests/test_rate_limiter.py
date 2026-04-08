from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.rate_limiter import allow_request, reset_rate_limiter


@pytest.mark.asyncio
async def test_allow_request_uses_redis_counter_when_available():
    reset_rate_limiter()
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock(side_effect=[1, 2])
    mock_redis.expire = AsyncMock()

    with patch("app.services.rate_limiter.get_redis", new=AsyncMock(return_value=mock_redis)):
        first = await allow_request("chat:1.2.3.4", limit=1, window_seconds=60)
        second = await allow_request("chat:1.2.3.4", limit=1, window_seconds=60)

    assert first is True
    assert second is False
    mock_redis.expire.assert_called_once_with("rate_limit:chat:1.2.3.4", 60)


@pytest.mark.asyncio
async def test_allow_request_falls_back_to_in_memory_when_redis_unavailable():
    reset_rate_limiter()
    with patch(
        "app.services.rate_limiter.get_redis",
        new=AsyncMock(side_effect=ConnectionError("redis down")),
    ):
        first = await allow_request("chat:fallback", limit=1, window_seconds=60)
        second = await allow_request("chat:fallback", limit=1, window_seconds=60)

    assert first is True
    assert second is False

