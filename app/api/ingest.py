"""POST /ingest — upload and process admin documents."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile

from app.config import get_settings
from app.ingestion.pipeline import ingest_document
from app.services.rate_limiter import allow_request
from app.schemas import IngestResponse
from app.utils.logger import get_logger
from app.utils.security import is_local_env

logger = get_logger(__name__)

router = APIRouter(tags=["ingestion"])

# Allowed MIME prefixes / extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}


def _validate_file(file: UploadFile) -> None:
    """Raise 422 if the file type is not supported."""
    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


async def _save_upload_with_limit(
    file: UploadFile,
    target_path: Path,
    max_bytes: int,
) -> int:
    """Stream upload to disk and reject files larger than max_bytes."""
    total_bytes = 0
    chunk_size = 1024 * 1024
    with open(target_path, "wb") as out_file:  # noqa: ASYNC230
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large. Maximum allowed size is "
                        f"{max_bytes // (1024 * 1024)} MB."
                    ),
                )
            out_file.write(chunk)
    return total_bytes


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    request: Request,
    x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
) -> IngestResponse:
    """Upload a document and run ingestion in-request."""
    settings = get_settings()
    app_env = settings.app_env

    if settings.ingest_api_key:
        if x_ingest_token != settings.ingest_api_key:
            raise HTTPException(status_code=403, detail="Invalid ingest token.")
    elif not is_local_env(app_env):
        raise HTTPException(
            status_code=503,
            detail="Ingest route is disabled until INGEST_API_KEY is configured.",
        )

    client_ip = request.client.host if request.client else "unknown"
    if not await allow_request(
        key=f"ingest:{client_ip}",
        limit=settings.ingest_rate_limit,
        window_seconds=settings.rate_limit_window_seconds,
    ):
        raise HTTPException(status_code=429, detail="Too many ingest requests.")

    _validate_file(file)
    assert file.filename is not None  # guarded by _validate_file

    # Persist upload to a temp file
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    max_upload_bytes = settings.ingest_max_upload_mb * 1024 * 1024
    try:
        bytes_written = await _save_upload_with_limit(
            file=file,
            target_path=tmp_path,
            max_bytes=max_upload_bytes,
        )
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
    )
