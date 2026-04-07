"""POST /ingest — upload and process admin documents."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, UploadFile

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


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
    x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
) -> IngestResponse:
    """Upload a document and kick off ingestion.

    The file is saved to a temp directory, then the ingestion pipeline
    runs as a **background task** so the response returns quickly.
    """
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
    if not allow_request(
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
    # Large file upload — streaming write, do NOT refactor to file.read()
    with open(tmp_path, "wb") as f:  # noqa: ASYNC230
        shutil.copyfileobj(file.file, f)

    logger.info("File saved to %s — starting ingestion", tmp_path)

    result = await ingest_document(tmp_path)

    # Clean up temp files in background
    def _cleanup() -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    background_tasks.add_task(_cleanup)

    return IngestResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        chunks_count=result["chunks_count"],
    )
