"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator

# ── Shared / Base ────────────────────────────────────────


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs.

    See https://datatracker.ietf.org/doc/html/rfc7807
    """

    type: str = Field(
        default="about:blank",
        description="A URI reference identifying the problem type.",
    )
    title: str = Field(
        ...,
        description="A short, human-readable summary of the problem type.",
    )
    status: int = Field(
        ...,
        description="The HTTP status code for this occurrence of the problem.",
    )
    detail: str = Field(
        ...,
        description="A human-readable explanation specific to this occurrence.",
    )
    instance: str = Field(
        default="",
        description="A URI reference that identifies the specific occurrence.",
    )


# ── Auth ─────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Request payload for SIAKAD authentication."""

    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    """Response payload after successful SIAKAD authentication."""

    session_id: str
    student_access_token: str
    status: str
    message: str


# ── Chat ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None, description="Unique session identifier (generated if omitted)"
    )
    message: str = Field(
        ..., min_length=1, max_length=1000, description="User message text"
    )

    @field_validator("message")
    @classmethod
    def check_message_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Pesan tidak boleh kosong.")
        return s


class SourceChunk(BaseModel):
    """Represents a single retrieved document chunk used as context."""

    doc_id: str
    filename: str
    page: int | None = None
    score: float | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)


class ChatHistoryItem(BaseModel):
    """Single turn in a conversation."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: str | None = None


class ChatHistoryResponse(BaseModel):
    """Response for GET /chat/{session_id}/history."""

    session_id: str
    history: list[ChatHistoryItem]


# ── Ingest ──────────────────────────────────────────────


class FileEntry(BaseModel):
    """Represents a single ingested file in the listing."""

    doc_id: str
    filename: str
    doc_category: str | None = None
    academic_year: str | None = None
    total_chunks: int


class FileListResponse(BaseModel):
    """Response payload for listing ingested files."""

    total_files: int
    files: list[FileEntry]


class FileDeleteRequest(BaseModel):
    """Request payload for deleting an ingested file."""

    doc_id: str = Field(..., description="Document ID to delete")


class FileDeleteResponse(BaseModel):
    """Response payload after deleting an ingested file."""

    doc_id: str
    filename: str
    deleted_chunks: int
    deleted: bool
    message: str = "File and all associated chunks deleted successfully"


class ChunkEntry(BaseModel):
    """Represents a single chunk from an ingested file."""

    chunk_id: str
    chunk_index: int
    page: int | None = None
    headings: list[str] = Field(default_factory=list)
    content_type: str = "text"
    doc_category: str | None = None
    academic_year: str | None = None
    text: str
    enriched_text: str | None = None


class ChunkListResponse(BaseModel):
    """Response payload for listing chunks of an ingested file."""

    doc_id: str
    filename: str
    total_chunks: int
    chunks: list[ChunkEntry]


class FileRenameRequest(BaseModel):
    """Request payload for renaming an ingested file."""

    doc_id: str = Field(..., description="Document ID to rename")
    filename: str = Field(..., min_length=1, description="New filename")


class FileRenameResponse(BaseModel):
    """Response payload after renaming an ingested file."""

    doc_id: str
    filename: str
    updated_chunks: int
    updated: bool
    message: str = "Filename updated successfully"


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    skipped: bool = Field(
        default=False, description="Whether the file was skipped due to checkpoint"
    )
    message: str = "Document ingested successfully"


# ── Chunk CRUD ──────────────────────────────────────────


class ChunkCreateRequest(BaseModel):
    """Request payload for injecting a new chunk into an existing file."""

    doc_id: str = Field(..., description="Document ID to inject chunk into")
    text: str = Field(..., min_length=1, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk index position")
    page: int | None = Field(default=None, description="Page number")
    headings: list[str] = Field(default_factory=list, description="Section headings")
    content_type: str = Field(default="text", description="Content type")
    doc_category: str | None = Field(default=None, description="Document category")
    academic_year: str | None = Field(default=None, description="Academic year")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkCreateResponse(BaseModel):
    """Response payload after injecting a new chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    created: bool
    message: str = "Chunk injected successfully"


class ChunkUpdateRequest(BaseModel):
    """Request payload for updating an existing chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to update")
    text: str = Field(..., min_length=1, description="Updated chunk text content")
    headings: list[str] | None = Field(default=None, description="Updated headings")
    page: int | None = Field(default=None, description="Updated page number")
    content_type: str | None = Field(default=None, description="Updated content type")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkUpdateResponse(BaseModel):
    """Response payload after updating a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    updated: bool
    message: str = "Chunk updated successfully"


class ChunkDeleteRequest(BaseModel):
    """Request payload for deleting a specific chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to delete")


class ChunkDeleteResponse(BaseModel):
    """Response payload after deleting a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    deleted: bool
    message: str = "Chunk deleted successfully"


# ── Chunk CRUD ──────────────────────────────────────────


class ChunkCreateRequest(BaseModel):
    """Request payload for injecting a new chunk into an existing file."""

    doc_id: str = Field(..., description="Document ID to inject chunk into")
    text: str = Field(..., min_length=1, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk index position")
    page: int | None = Field(default=None, description="Page number")
    headings: list[str] = Field(default_factory=list, description="Section headings")
    content_type: str = Field(default="text", description="Content type")
    doc_category: str | None = Field(default=None, description="Document category")
    academic_year: str | None = Field(default=None, description="Academic year")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkCreateResponse(BaseModel):
    """Response payload after injecting a new chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    created: bool
    message: str = "Chunk injected successfully"


class ChunkUpdateRequest(BaseModel):
    """Request payload for updating an existing chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to update")
    text: str = Field(..., min_length=1, description="Updated chunk text content")
    headings: list[str] | None = Field(default=None, description="Updated headings")
    page: int | None = Field(default=None, description="Updated page number")
    content_type: str | None = Field(default=None, description="Updated content type")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkUpdateResponse(BaseModel):
    """Response payload after updating a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    updated: bool
    message: str = "Chunk updated successfully"


class ChunkDeleteRequest(BaseModel):
    """Request payload for deleting a specific chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to delete")


class ChunkDeleteResponse(BaseModel):
    """Response payload after deleting a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    deleted: bool
    message: str = "Chunk deleted successfully"


# ── Chunk CRUD ──────────────────────────────────────────


class ChunkCreateRequest(BaseModel):
    """Request payload for injecting a new chunk into an existing file."""

    doc_id: str = Field(..., description="Document ID to inject chunk into")
    text: str = Field(..., min_length=1, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk index position")
    page: int | None = Field(default=None, description="Page number")
    headings: list[str] = Field(default_factory=list, description="Section headings")
    content_type: str = Field(default="text", description="Content type")
    doc_category: str | None = Field(default=None, description="Document category")
    academic_year: str | None = Field(default=None, description="Academic year")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkCreateResponse(BaseModel):
    """Response payload after injecting a new chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    created: bool
    message: str = "Chunk injected successfully"


class ChunkUpdateRequest(BaseModel):
    """Request payload for updating an existing chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to update")
    text: str = Field(..., min_length=1, description="Updated chunk text content")
    headings: list[str] | None = Field(default=None, description="Updated headings")
    page: int | None = Field(default=None, description="Updated page number")
    content_type: str | None = Field(default=None, description="Updated content type")

    @field_validator("text")
    @classmethod
    def check_text_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Chunk text cannot be empty or whitespace.")
        return s


class ChunkUpdateResponse(BaseModel):
    """Response payload after updating a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    updated: bool
    message: str = "Chunk updated successfully"


class ChunkDeleteRequest(BaseModel):
    """Request payload for deleting a specific chunk."""

    doc_id: str = Field(..., description="Document ID containing the chunk")
    chunk_index: int = Field(..., ge=0, description="Chunk index to delete")


class ChunkDeleteResponse(BaseModel):
    """Response payload after deleting a chunk."""

    doc_id: str
    filename: str
    chunk_index: int
    chunk_id: str
    deleted: bool
    message: str = "Chunk deleted successfully"


# ── Health ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    qdrant: str = "unknown"
    redis: str = "unknown"


__all__ = [
    "ChatHistoryItem",
    "ChatHistoryResponse",
    "ChatRequest",
    "ChatResponse",
    "ChunkCreateRequest",
    "ChunkCreateResponse",
    "ChunkDeleteRequest",
    "ChunkDeleteResponse",
    "ChunkEntry",
    "ChunkListResponse",
    "ChunkUpdateRequest",
    "ChunkUpdateResponse",
    "ErrorResponse",
    "FileDeleteRequest",
    "FileDeleteResponse",
    "FileEntry",
    "FileListResponse",
    "FileRenameRequest",
    "FileRenameResponse",
    "HealthResponse",
    "IngestResponse",
    "LoginRequest",
    "LoginResponse",
    "SourceChunk",
]
