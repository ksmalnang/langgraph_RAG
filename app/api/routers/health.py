"""Health check endpoint — GET /health."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.api.models import HealthResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


async def _check_qdrant() -> str:
    """Return ``"connected"`` or ``"disconnected"``."""
    try:
        from app.services.vectorstore import get_qdrant_client

        client = await get_qdrant_client()
        await client.get_collections()
        return "connected"
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return "disconnected"


async def _check_redis() -> str:
    """Return ``"connected"`` or ``"disconnected"``."""
    try:
        from app.services.memory import get_redis

        r = await get_redis()
        await r.ping()
        return "connected"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return "disconnected"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check connectivity to Qdrant and Redis."""
    qdrant_status, redis_status = await asyncio.gather(
        _check_qdrant(),
        _check_redis(),
    )

    overall = (
        "healthy"
        if qdrant_status == "connected" and redis_status == "connected"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        redis=redis_status,
    )
