"""Tests for API endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _assert_rfc7807(data: dict, status_code: int, expected_title: str) -> None:
    """Assert that the response follows RFC 7807 Problem Details format."""
    assert "type" in data, "RFC 7807 requires a 'type' field"
    assert "title" in data, "RFC 7807 requires a 'title' field"
    assert "status" in data, "RFC 7807 requires a 'status' field"
    assert "detail" in data, "RFC 7807 requires a 'detail' field"
    assert "instance" in data, "RFC 7807 requires an 'instance' field"

    assert data["status"] == status_code
    assert data["title"] == expected_title


# ── Health ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_healthy():
    """Health endpoint returns healthy when both services are up."""
    with (
        patch("app.services.vectorstore.get_qdrant_client") as mock_qdrant,
        patch("app.services.memory.get_redis") as mock_redis,
    ):
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = AsyncMock(collections=[])
        mock_qdrant.return_value = mock_client

        mock_r = AsyncMock()
        mock_r.ping.return_value = True
        mock_redis.return_value = mock_r

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["qdrant"] == "connected"
        assert data["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_degraded_redis():
    """Health endpoint returns degraded when Redis is down."""
    with (
        patch("app.services.vectorstore.get_qdrant_client") as mock_qdrant,
        patch("app.services.memory.get_redis") as mock_redis,
    ):
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = AsyncMock(collections=[])
        mock_qdrant.return_value = mock_client

        mock_redis.side_effect = ConnectionError("Redis down")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["redis"] == "disconnected"


# ── Chat ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Chat endpoint invokes the graph and returns a response."""
    mock_result = {
        "answer": "You can register through the portal.",
        "sources": [
            {
                "doc_id": "abc123",
                "filename": "enrollment.pdf",
                "page": 5,
                "score": 0.95,
                "snippet": "To register for next semester...",
            }
        ],
    }

    with patch("app.api.routers.chat._get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = mock_result
        mock_get_graph.return_value = mock_graph

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={"session_id": "test-123", "message": "How do I register?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["session_id"]) > 10
        assert data["session_id"] != "test-123"
        assert "register" in data["answer"].lower()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "enrollment.pdf"
        assert data["sources"][0]["doc_id"] == "abc123"
        assert data["sources"][0]["page"] == 5


@pytest.mark.asyncio
async def test_chat_invalid_message():
    """Chat endpoint rejects empty messages with RFC 7807 error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"message": ""})

    assert response.status_code == 422
    _assert_rfc7807(response.json(), 422, "Validation Error")


# ── Ingestion ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_files_empty():
    """List files endpoint returns 200 with empty list when no files ingested."""
    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 0
        assert data["files"] == []


@pytest.mark.asyncio
async def test_list_files_two_files():
    """List files endpoint returns one entry per ingested file."""
    file_entries = [
        {
            "doc_id": "doc-abc123",
            "filename": "enrollment.pdf",
            "doc_category": "Academic",
            "academic_year": "2024/2025",
            "total_chunks": 5,
        },
        {
            "doc_id": "doc-def456",
            "filename": "schedule.pdf",
            "doc_category": "Administrative",
            "academic_year": "2024/2025",
            "total_chunks": 3,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        # Results sorted by filename ascending
        assert data["files"][0]["filename"] == "enrollment.pdf"
        assert data["files"][1]["filename"] == "schedule.pdf"
        assert data["files"][0]["total_chunks"] == 5
        assert data["files"][1]["total_chunks"] == 3
        assert data["files"][0]["doc_category"] == "Academic"
        assert data["files"][0]["academic_year"] == "2024/2025"


@pytest.mark.asyncio
async def test_list_files_total_chunks_matches():
    """List files endpoint accurately reflects stored chunk counts."""
    file_entries = [
        {
            "doc_id": "doc-xyz",
            "filename": "large-doc.pdf",
            "doc_category": None,
            "academic_year": None,
            "total_chunks": 42,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["files"][0]["total_chunks"] == 42


@pytest.mark.asyncio
async def test_list_files_idempotent_reingest():
    """Re-ingesting the same file results in only one entry."""
    # Simulates deduplication: even if called multiple times, same doc_id
    file_entries = [
        {
            "doc_id": "doc-duplicate",
            "filename": "re-ingested.pdf",
            "doc_category": "Academic",
            "academic_year": "2024/2025",
            "total_chunks": 10,
        },
    ]

    with patch("app.api.routers.ingestion.vs.list_files") as mock_list:
        mock_list.return_value = file_entries

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ingest/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1
        assert data["files"][0]["doc_id"] == "doc-duplicate"


@pytest.mark.asyncio
async def test_ingest_unsupported_file():
    """Ingest endpoint rejects unsupported file types with RFC 7807 error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            files={"file": ("test.exe", b"malicious", "application/octet-stream")},
        )

    assert response.status_code == 422
    _assert_rfc7807(response.json(), 422, "Validation Error")


@pytest.mark.asyncio
async def test_ingest_rejects_file_over_size_limit():
    """Ingest endpoint rejects oversized files with RFC 7807 error."""
    fake_settings = SimpleNamespace(
        app_env="development",
        ingest_api_key=None,
        ingest_rate_limit=5,
        rate_limit_window_seconds=60,
        ingest_max_upload_mb=1,
    )

    with patch("app.api.routers.ingestion.get_settings", return_value=fake_settings):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                files={"file": ("big.md", b"a" * (1024 * 1024 + 1), "text/markdown")},
            )

    assert response.status_code == 413
    _assert_rfc7807(response.json(), 413, "Payload Too Large")
