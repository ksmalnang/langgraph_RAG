"""FastAPI application — entrypoint, middleware, and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import auth_router, chat_router, health_router, ingestion_router
from app.config import get_settings
from app.utils.exceptions import register_exception_handlers
from app.utils.logger import get_logger, setup_logging
from app.utils.security import is_local_env, parse_cors_origins


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hooks."""
    setup_logging()
    logger = get_logger(__name__)
    logger.info("\n\033[41;37m[START]\033[0m \nStarting RAG Chatbot API ...")

    yield

    # Cleanup on shutdown
    logger.info("\n\033[41;37m[STOP]\033[0m \nShutting down ...")
    from app.services.memory import close_redis
    from app.services.vectorstore import close_client

    await close_client()
    await close_redis()
    logger.info("Cleanup complete")


app = FastAPI(
    title="Teknik Informatika administration chatbot",
    description="RAG-based chatbot for Teknik Informatika administration Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ───────────────────────────────────────────

settings = get_settings()
cors_origins = parse_cors_origins(settings.cors_allow_origins)

if not cors_origins and is_local_env(settings.app_env):
    cors_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

if "*" in cors_origins and not is_local_env(settings.app_env):
    raise RuntimeError("Wildcard CORS is not allowed outside local development.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(ingestion_router)
app.include_router(chat_router)

# ── Exception handlers ──────────────────────────────────

register_exception_handlers(app)
