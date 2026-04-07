"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ── Shared / Base ────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error envelope for all 4xx/5xx responses."""

    detail: str
    code: str | None = None  # e.g. "QDRANT_UNAVAILABLE"


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


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    message: str = "Document ingested successfully"


# ── Health ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    qdrant: str = "unknown"
    redis: str = "unknown"
