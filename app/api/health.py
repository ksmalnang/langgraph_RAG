"""GET /health — service health check."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check connectivity to Qdrant and Redis."""
    qdrant_status = "unknown"
    redis_status = "unknown"

    # ── Qdrant ──────────────────────────────────────────
    try:
        from app.services.vectorstore import get_qdrant_client

        client = await get_qdrant_client()
        await client.get_collections()
        qdrant_status = "connected"
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        qdrant_status = "disconnected"

    # ── Redis ───────────────────────────────────────────
    try:
        from app.services.memory import get_redis

        r = await get_redis()
        await r.ping()
        redis_status = "connected"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        redis_status = "disconnected"

    overall = (
        "healthy"
        if qdrant_status == "connected" and redis_status == "connected"
        else "degraded"
    )

    return HealthResponse(status=overall, qdrant=qdrant_status, redis=redis_status)
