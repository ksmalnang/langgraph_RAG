"""Router sub-package — re-export all routers for app assembly."""

from __future__ import annotations

from app.api.routers.auth import router as auth_router
from app.api.routers.chat import router as chat_router
from app.api.routers.health import router as health_router
from app.api.routers.ingestion import router as ingestion_router
from app.api.routers.telegram import router as telegram_router

__all__ = [
    "auth_router",
    "chat_router",
    "health_router",
    "ingestion_router",
    "telegram_router",
]
