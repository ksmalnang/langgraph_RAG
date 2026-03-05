"""Custom exceptions and FastAPI error handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Custom Exceptions ───────────────────────────────────


class AppError(Exception):
    """Base application error."""

    def __init__(
        self, message: str, status_code: int = 500, code: str | None = None
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class IngestionError(AppError):
    """Raised when document ingestion fails."""

    def __init__(self, message: str = "Document ingestion failed") -> None:
        super().__init__(message, status_code=422, code="INGESTION_FAILED")


class RetrievalError(AppError):
    """Raised when vector search fails."""

    def __init__(self, message: str = "Retrieval failed") -> None:
        super().__init__(message, status_code=502, code="RETRIEVAL_FAILED")


class RerankerError(AppError):
    """Raised when reranking fails."""

    def __init__(self, message: str = "Reranking failed") -> None:
        super().__init__(message, status_code=502, code="RERANKER_FAILED")


class LLMError(AppError):
    """Raised when the LLM call fails."""

    def __init__(self, message: str = "LLM generation failed") -> None:
        super().__init__(message, status_code=502, code="LLM_FAILED")


# ── FastAPI handlers ────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Attach custom exception handlers to the FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        logger.error("AppError [%s]: %s", exc.code, exc.message)
        body = ErrorResponse(detail=exc.message, code=exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        body = ErrorResponse(
            detail="An internal server error occurred.", code="INTERNAL_ERROR"
        )
        return JSONResponse(
            status_code=500,
            content=body.model_dump(),
        )
