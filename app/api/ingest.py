"""POST /ingest — upload and process admin documents."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app.ingestion.pipeline import ingest_document
from app.schemas import IngestResponse
from app.utils.logger import get_logger

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
) -> IngestResponse:
    """Upload a document and kick off ingestion.

    The file is saved to a temp directory, then the ingestion pipeline
    runs as a **background task** so the response returns quickly.
    """
    _validate_file(file)
    assert file.filename is not None  # guarded by _validate_file

    # Persist upload to a temp file
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
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
