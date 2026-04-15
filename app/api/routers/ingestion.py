"""Data ingestion endpoints — POST /ingest."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, Header, Request, UploadFile

from app.api.models import (
    FileDeleteRequest,
    FileDeleteResponse,
    FileEntry,
    FileListResponse,
    IngestResponse,
)
from app.config import get_settings
from app.ingestion.pipeline import ingest_document
from app.services import vectorstore as vs
from app.services.rate_limiter import allow_request
from app.utils.exceptions import AppError, VectorStoreError
from app.utils.logger import get_logger
from app.utils.security import is_local_env

logger = get_logger(__name__)

router = APIRouter(tags=["ingestion"])

ALLOWED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}
)

_CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_filename(file: UploadFile) -> Path:
    """Return the lowered suffix ``Path`` or raise 422."""
    if file.filename is None:
        raise AppError(
            detail="Filename is required.",
            status_code=422,
            title="Validation Error",
        )
    return Path(file.filename)


def _check_extension(suffix: str) -> None:
    """Raise 422 when the file extension is not allowed."""
    if suffix not in ALLOWED_EXTENSIONS:
        raise AppError(
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            status_code=422,
            title="Validation Error",
        )


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------


async def _save_with_limit(
    file: UploadFile,
    target_path: Path,
    max_bytes: int,
) -> int:
    """
    Stream *file* to *target_path* and reject if it exceeds *max_bytes*.

    Returns the number of bytes written.
    """
    total_bytes = 0
    with open(target_path, "wb") as out:  # noqa: ASYNC230
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise AppError(
                    detail=f"File too large. Maximum allowed size is {max_bytes // (1024 * 1024)} MB.",
                    status_code=413,
                    title="Payload Too Large",
                )
            out.write(chunk)
    return total_bytes


# ---------------------------------------------------------------------------
# Auth / rate-limit helpers
# ---------------------------------------------------------------------------


async def _check_ingest_token(
    x_ingest_token: str | None,
    settings: object,
) -> None:
    """Validate the ingest token or require it in non-local environments."""
    if settings.ingest_api_key:  # type: ignore[attr-defined]
        if x_ingest_token != settings.ingest_api_key:
            raise AppError(
                detail="Invalid ingest token.",
                status_code=403,
                title="Forbidden",
            )
    elif not is_local_env(settings.app_env):
        raise AppError(
            detail="Ingest route is disabled until INGEST_API_KEY is configured.",
            status_code=503,
            title="Service Unavailable",
        )


async def _check_rate_limit(client_ip: str, settings: object) -> None:
    """Enforce per-IP rate limiting for ingest requests."""
    if not await allow_request(
        key=f"ingest:{client_ip}",
        limit=settings.ingest_rate_limit,  # type: ignore[attr-defined]
        window_seconds=settings.rate_limit_window_seconds,  # type: ignore[attr-defined]
    ):
        raise AppError(
            detail="Too many ingest requests.",
            status_code=429,
            title="Too Many Requests",
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/ingest/files", response_model=FileListResponse)
async def list_files() -> FileListResponse:
    """List all ingested files with their chunk counts."""
    raw_files = await vs.list_files()
    entries = [FileEntry(**f) for f in raw_files]
    return FileListResponse(
        total_files=len(entries),
        files=entries,
    )


@router.delete("/ingest/files", response_model=FileDeleteResponse)
async def delete_file(body: FileDeleteRequest) -> FileDeleteResponse:
    """Delete all chunks belonging to an ingested file."""
    try:
        result = await vs.delete_file(body.doc_id)
    except VectorStoreError as exc:
        # doc_id not found → 404
        if "not found" in str(exc).lower():
            raise AppError(
                detail=f"File with doc_id='{body.doc_id}' not found.",
                status_code=404,
                title="Not Found",
            ) from exc
        raise
    return FileDeleteResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        deleted_chunks=result["deleted_chunks"],
        deleted=True,
        message="File and all associated chunks deleted successfully",
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    request: Request,
    x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
) -> IngestResponse:
    """Upload a document and run ingestion in-request."""
    settings = get_settings()

    await _check_ingest_token(x_ingest_token, settings)

    client_ip = request.client.host if request.client else "unknown"
    await _check_rate_limit(client_ip, settings)

    file_path = _validate_filename(file)
    _check_extension(file_path.suffix.lower())

    max_upload_bytes = settings.ingest_max_upload_mb * 1024 * 1024

    # Persist upload to a temp directory; clean up afterwards.
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename  # type: ignore[operator]
    try:
        bytes_written = await _save_with_limit(file, tmp_path, max_upload_bytes)
        logger.info(
            "File saved to %s (%d bytes) — starting ingestion",
            tmp_path,
            bytes_written,
        )
        result = await ingest_document(tmp_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        chunks_count=result["chunks_count"],
        skipped=result.get("skipped", False),
    )
