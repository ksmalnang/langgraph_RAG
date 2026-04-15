"""Data ingestion endpoints — POST /ingest."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, Header, Query, Request, UploadFile

from app.api.models import (
    ChunkCreateRequest,
    ChunkCreateResponse,
    ChunkDeleteRequest,
    ChunkDeleteResponse,
    ChunkEntry,
    ChunkListResponse,
    ChunkUpdateRequest,
    ChunkUpdateResponse,
    FileDeleteRequest,
    FileDeleteResponse,
    FileEntry,
    FileListResponse,
    FileRenameRequest,
    FileRenameResponse,
    IngestResponse,
)
from app.config import get_settings
from app.ingestion.pipeline import ingest_document
from app.services import embeddings as embed_svc
from app.services import vectorstore as vs
from app.services.rate_limiter import allow_request
from app.utils.exceptions import AppError, VectorStoreError
from app.utils.helpers import generate_chunk_point_id
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


@router.get("/ingest/by-file/chunks", response_model=ChunkListResponse)
async def list_chunks(
    doc_id: str = Query(..., description="Document ID to list chunks for"),
) -> ChunkListResponse:
    """List all chunks for an ingested file, sorted by chunk_index."""
    points = await vs.scroll_chunks_by_doc_id(doc_id)

    if not points:
        raise AppError(
            detail=f"File with doc_id='{doc_id}' not found.",
            status_code=404,
            title="Not Found",
        )

    chunks: list[ChunkEntry] = []
    filename = ""
    for point in points:
        p = point.payload or {}
        filename = p.get("filename", "")
        chunks.append(
            ChunkEntry(
                chunk_id=str(point.id),
                chunk_index=p["chunk_index"],
                page=p.get("page"),
                headings=p.get("headings", []),
                content_type=p.get("content_type", "text"),
                doc_category=p.get("doc_category"),
                academic_year=p.get("academic_year"),
                text=p["text"],
                enriched_text=p.get("enriched_text"),
            )
        )

    return ChunkListResponse(
        doc_id=doc_id,
        filename=filename,
        total_chunks=len(chunks),
        chunks=chunks,
    )


def _build_enriched_text(text: str, headings: list[str]) -> str:
    """Build enriched text from headings and text content."""
    if headings:
        return f"[{headings[0]}]\n{text}"
    return text


@router.post("/ingest/by-file/chunks", response_model=ChunkCreateResponse)
async def create_chunk(body: ChunkCreateRequest) -> ChunkCreateResponse:
    """Inject a manually written chunk into an existing file's chunk inventory."""
    existing_chunks = await vs.scroll_chunks_by_doc_id(body.doc_id)
    if not existing_chunks:
        raise AppError(
            detail=f"File with doc_id='{body.doc_id}' not found.",
            status_code=404,
            title="Not Found",
        )

    chunk_id = generate_chunk_point_id(body.doc_id, body.chunk_index)

    for point in existing_chunks:
        payload = point.payload or {}
        if str(point.id) == chunk_id:
            raise AppError(
                detail=f"Chunk at index {body.chunk_index} already exists for doc_id='{body.doc_id}'. Use PATCH to update.",
                status_code=409,
                title="Conflict",
            )

    first_payload = existing_chunks[0].payload or {}
    filename = first_payload.get("filename", "")
    doc_category = body.doc_category if body.doc_category is not None else first_payload.get("doc_category")
    academic_year = body.academic_year if body.academic_year is not None else first_payload.get("academic_year")

    enriched_text = _build_enriched_text(body.text, body.headings)
    dense_vector = await embed_svc.embed_query(enriched_text)
    sparse_vector = vs._encode_sparse([enriched_text])[0]

    payload = {
        "text": body.text,
        "enriched_text": enriched_text,
        "doc_id": body.doc_id,
        "filename": filename,
        "chunk_index": body.chunk_index,
        "headings": body.headings,
        "page": body.page,
        "doc_category": doc_category,
        "academic_year": academic_year,
        "content_type": body.content_type,
    }

    await vs.upsert_single_chunk(chunk_id, dense_vector, sparse_vector, payload)

    return ChunkCreateResponse(
        doc_id=body.doc_id,
        filename=filename,
        chunk_index=body.chunk_index,
        chunk_id=chunk_id,
        created=True,
        message="Chunk injected successfully",
    )


@router.patch("/ingest/by-file/chunks", response_model=ChunkUpdateResponse)
async def update_chunk(body: ChunkUpdateRequest) -> ChunkUpdateResponse:
    """Overwrite the text and optionally metadata of one specific chunk."""
    existing_chunks = await vs.scroll_chunks_by_doc_id(body.doc_id)
    if not existing_chunks:
        raise AppError(
            detail=f"File with doc_id='{body.doc_id}' not found.",
            status_code=404,
            title="Not Found",
        )

    chunk_data = await vs.get_chunk_by_doc_id_and_index(body.doc_id, body.chunk_index)
    if chunk_data is None:
        raise AppError(
            detail=f"Chunk at index {body.chunk_index} not found for doc_id='{body.doc_id}'.",
            status_code=404,
            title="Not Found",
        )

    chunk_id = chunk_data["point_id"]
    existing_payload = chunk_data["payload"]
    filename = existing_payload.get("filename", "")

    merged_payload = dict(existing_payload)
    merged_payload["text"] = body.text

    if body.headings is not None:
        merged_payload["headings"] = body.headings
    if body.page is not None:
        merged_payload["page"] = body.page
    if body.content_type is not None:
        merged_payload["content_type"] = body.content_type

    headings = merged_payload.get("headings", [])
    enriched_text = _build_enriched_text(body.text, headings)
    merged_payload["enriched_text"] = enriched_text

    dense_vector = await embed_svc.embed_query(enriched_text)
    sparse_vector = vs._encode_sparse([enriched_text])[0]

    await vs.upsert_single_chunk(chunk_id, dense_vector, sparse_vector, merged_payload)

    return ChunkUpdateResponse(
        doc_id=body.doc_id,
        filename=filename,
        chunk_index=body.chunk_index,
        chunk_id=chunk_id,
        updated=True,
        message="Chunk updated successfully",
    )


@router.delete("/ingest/by-file/chunks", response_model=ChunkDeleteResponse)
async def delete_chunk(body: ChunkDeleteRequest) -> ChunkDeleteResponse:
    """Remove one chunk from an ingested file's inventory."""
    existing_chunks = await vs.scroll_chunks_by_doc_id(body.doc_id)
    if not existing_chunks:
        raise AppError(
            detail=f"File with doc_id='{body.doc_id}' not found.",
            status_code=404,
            title="Not Found",
        )

    chunk_data = await vs.get_chunk_by_doc_id_and_index(body.doc_id, body.chunk_index)
    if chunk_data is None:
        raise AppError(
            detail=f"Chunk at index {body.chunk_index} not found for doc_id='{body.doc_id}'.",
            status_code=404,
            title="Not Found",
        )

    chunk_id = chunk_data["point_id"]
    existing_payload = chunk_data["payload"]
    filename = existing_payload.get("filename", "")

    await vs.delete_single_chunk(chunk_id)

    return ChunkDeleteResponse(
        doc_id=body.doc_id,
        filename=filename,
        chunk_index=body.chunk_index,
        chunk_id=chunk_id,
        deleted=True,
        message="Chunk deleted successfully",
    )


@router.patch("/ingest/files", response_model=FileRenameResponse)
async def rename_file(body: FileRenameRequest) -> FileRenameResponse:
    """Rename an ingested file by updating its filename payload."""
    # Check for filename collision with another doc_id
    existing_files = await vs.list_files()
    for f in existing_files:
        if f["filename"] == body.filename and f["doc_id"] != body.doc_id:
            raise AppError(
                detail=f"Filename '{body.filename}' is already used by doc_id='{f['doc_id']}'.",
                status_code=409,
                title="Conflict",
            )

    try:
        result = await vs.rename_file(body.doc_id, body.filename)
    except VectorStoreError as exc:
        if "not found" in str(exc).lower():
            raise AppError(
                detail=f"File with doc_id='{body.doc_id}' not found.",
                status_code=404,
                title="Not Found",
            ) from exc
        raise

    return FileRenameResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        updated_chunks=result["updated_chunks"],
        updated=True,
        message="Filename updated successfully",
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
