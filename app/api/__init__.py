"""API package — exposes routers for app assembly."""

from __future__ import annotations

from app.api.routers import (
    auth_router,
    chat_router,
    health_router,
    ingestion_router,
)

__all__ = [
    "auth_router",
    "chat_router",
    "health_router",
    "ingestion_router",
]
