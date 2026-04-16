"""Custom exceptions and FastAPI error handlers (RFC 7807)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Custom Exceptions ───────────────────────────────────


class AppError(Exception):
    """Base application error with RFC 7807 problem type."""

    status_code: int
    type: str
    title: str
    instance: str = ""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        title: str | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.type = "about:blank"
        self.title = title or f"HTTP {status_code}"
        super().__init__(detail)


class IngestionError(AppError):
    """Raised when document ingestion fails."""

    def __init__(self, detail: str = "Document ingestion failed") -> None:
        super().__init__(detail=detail, status_code=422, title="Ingestion Failed")


class RetrievalError(AppError):
    """Raised when vector search fails."""

    def __init__(self, detail: str = "Retrieval failed") -> None:
        super().__init__(detail=detail, status_code=502, title="Retrieval Failed")


class RerankerError(AppError):
    """Raised when reranking fails."""

    def __init__(self, detail: str = "Reranking failed") -> None:
        super().__init__(detail=detail, status_code=502, title="Reranker Failed")


class LLMError(AppError):
    """Raised when the LLM call fails."""

    def __init__(self, detail: str = "LLM generation failed") -> None:
        super().__init__(detail=detail, status_code=502, title="LLM Failed")


class EmbeddingError(AppError):
    """Raised when embedding generation fails."""

    def __init__(self, detail: str = "Embedding generation failed") -> None:
        super().__init__(detail=detail, status_code=502, title="Embedding Failed")


class VectorStoreError(AppError):
    """Raised when Qdrant operations fail."""

    def __init__(self, detail: str = "Vector store operation failed") -> None:
        super().__init__(detail=detail, status_code=502, title="Vector Store Failed")


class MemoryStoreError(AppError):
    """Raised when Redis memory/session operations fail."""

    def __init__(self, detail: str = "Session store operation failed") -> None:
        super().__init__(detail=detail, status_code=503, title="Memory Store Failed")


class SiakadAuthError(AppError):
    """Raised when SIAKAD authentication fails."""

    def __init__(self, detail: str = "SIAKAD authentication failed") -> None:
        super().__init__(detail=detail, status_code=401, title="Authentication Failed")


class SiakadScrapeError(AppError):
    """Raised when SIAKAD data scraping fails."""

    def __init__(self, detail: str = "SIAKAD scraping failed") -> None:
        super().__init__(detail=detail, status_code=502, title="Scrape Failed")


# ── FastAPI handlers ────────────────────────────────────

# Map status codes to human-readable titles.
_STATUS_TITLES: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Validation Error",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach custom exception handlers to the FastAPI app."""
    from app.api.models import ErrorResponse

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.error("AppError [%s]: %s", exc.type, exc.detail)
        body = ErrorResponse(
            type=exc.type,
            title=exc.title,
            status=exc.status_code,
            detail=exc.detail,
            instance=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error: %s", exc.errors())
        err = exc.errors()[0]
        loc = err.get("loc", [])
        err_type = err.get("type", "")

        field = loc[-1] if loc else ""
        custom_msg = "Format input tidak valid."

        if field == "message":
            if err_type in ("string_too_short", "value_error", "missing"):
                custom_msg = "Pesan tidak boleh kosong."
            elif err_type == "string_too_long":
                custom_msg = "Pesan maksimal 1000 karakter."
        elif field == "email":
            custom_msg = "Format email tidak valid."
        elif field == "password":
            if err_type in ("string_too_short", "missing"):
                custom_msg = "Password minimal 6 karakter."
            elif err_type == "string_too_long":
                custom_msg = "Password maksimal 128 karakter."

        body = ErrorResponse(
            type="about:blank",
            title="Validation Error",
            status=422,
            detail=custom_msg,
            instance=request.url.path,
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        body = ErrorResponse(
            type="about:blank",
            title="Internal Server Error",
            status=500,
            detail="An internal server error occurred.",
            instance=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=body.model_dump(),
        )
